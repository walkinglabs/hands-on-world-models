"""第 4 章动手实验：在 CarRacing 上小规模复现 World Models 的 V-M-C 管线。

流程与 Ha & Schmidhuber (2018) 一致：随机策略收集数据 -> 训练 V（ConvVAE）
-> 训练 M（MDN-RNN）-> 用 CMA-ES 进化 C（线性控制器）。规模大幅打折，
原理不打折。默认参数在普通笔记本 CPU 上约 1-3 小时；加 --tiny 可冒烟测试。

用法示例：
    python scripts/run_carracing.py --output runs/carracing-world-model
    python scripts/run_carracing.py --output runs/carracing-no-memory --no-memory
    python scripts/run_carracing.py --output runs/carracing-exploit --tau 0.1
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from hwm.evaluation import RunManifest, sha256_file


def make_env(seed=None):
    try:
        import gymnasium as gym
    except ImportError as error:
        raise SystemExit(
            "需要 gymnasium[box2d]：pip install 'gymnasium[box2d]'"
        ) from error
    env = gym.make(
        "CarRacing-v3",
        domain_randomize=False,
        continuous=True,
        render_mode="rgb_array",
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def resize_frame(observation):
    """CarRacing 原生 96x96，与原文一致缩到 64x64 再喂给 V。"""
    frame = torch.as_tensor(observation, dtype=torch.float32).permute(2, 0, 1)
    frame = F.interpolate(frame[None], size=(64, 64), mode="area")[0]
    return frame.permute(1, 2, 0).contiguous().numpy()


# ---------------------------------------------------------------------------
# V：ConvVAE，把 64x64x3 的观测压成 32 维 latent
# ---------------------------------------------------------------------------


class ConvVAE(nn.Module):
    def __init__(self, z_size=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.mean_head = nn.Linear(256 * 4 * 4, z_size)
        self.logvar_head = nn.Linear(256 * 4 * 4, z_size)
        self.decoder = nn.Sequential(
            nn.Linear(z_size, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    def encode(self, frames):
        frames = frames.float() / 255.0
        if frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2)
        stats = self.encoder(frames)
        return self.mean_head(stats), self.logvar_head(stats)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def loss(self, frames):
        if frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2)
        mean, logvar = self.encode(frames)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z)
        recon_loss = F.mse_loss(reconstruction, frames.float() / 255.0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss / frames.shape[0]
        return recon_loss + 0.001 * kl_loss, recon_loss, kl_loss


def train_vae(frames, z_size=32, epochs=10, batch_size=64, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    frames = torch.from_numpy(np.asarray(frames))
    model = ConvVAE(z_size=z_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        indices = torch.randperm(len(frames))[: len(frames) // batch_size * batch_size]
        for start in range(0, len(indices), batch_size):
            batch = frames[indices[start : start + batch_size]]
            optimizer.zero_grad()
            total, recon, kl = model.loss(batch)
            total.backward()
            optimizer.step()
        history.append(float(recon.detach()))
    return model, history


# ---------------------------------------------------------------------------
# M：MDN-RNN，建模 P(z_{t+1} | a_t, z_t, h_t) 为混合高斯分布
# ---------------------------------------------------------------------------


class MDNRNN(nn.Module):
    def __init__(self, z_size=32, action_size=3, hidden_size=256, components=5):
        super().__init__()
        self.z_size = z_size
        self.hidden_size = hidden_size
        self.components = components
        self.gru = nn.GRUCell(z_size + action_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, components * z_size)
        self.logvar_head = nn.Linear(hidden_size, components * z_size)
        self.logit_head = nn.Linear(hidden_size, components)
        # 与原文一致：M 同时预测下一时刻的奖励（单高斯）
        self.reward_mean_head = nn.Linear(hidden_size, 1)
        self.reward_logvar_head = nn.Linear(hidden_size, 1)

    def forward(self, z, action, hidden):
        hidden = self.gru(torch.cat((z, action), dim=-1), hidden)
        means = self.mean_head(hidden).reshape(-1, self.components, self.z_size)
        logvars = self.logvar_head(hidden).reshape(-1, self.components, self.z_size)
        logits = self.logit_head(hidden)
        reward_mean = self.reward_mean_head(hidden)
        reward_logvar = self.reward_logvar_head(hidden)
        return means, logvars, logits, reward_mean, reward_logvar, hidden

    def sample(self, z, action, hidden, temperature=1.0):
        means, logvars, logits, reward_mean, reward_logvar, hidden = self.forward(
            z, action, hidden
        )
        components = torch.distributions.Categorical(logits=logits / temperature)
        index = components.sample()
        selected_mean = means[torch.arange(means.shape[0]), index]
        selected_std = torch.exp(0.5 * logvars)[torch.arange(logvars.shape[0]), index]
        z_next = selected_mean + temperature * selected_std * torch.randn_like(
            selected_std
        )
        reward = reward_mean + torch.exp(
            0.5 * reward_logvar
        ) * torch.randn_like(reward_mean)
        return z_next, reward, hidden

    def loss(self, z, action, target_z, target_r):
        # teacher forcing：每一步都从零 hidden 出发、以真实 z 为输入，
        # 与原文一致（多步记忆只在梦中由 sample 迭代生成时被使用）
        hidden = torch.zeros(len(z), self.hidden_size)
        means, logvars, logits, reward_mean, reward_logvar, _ = self.forward(
            z, action, hidden
        )
        distribution = torch.distributions.Normal(
            means, torch.exp(0.5 * logvars)
        )
        log_probs = distribution.log_prob(target_z[:, None, :]).sum(dim=-1)
        z_nll = -torch.logsumexp(
            torch.log_softmax(logits, dim=-1) + log_probs, dim=-1
        )
        reward_nll = (
            -torch.distributions.Normal(
                reward_mean.squeeze(-1), torch.exp(0.5 * reward_logvar.squeeze(-1))
            ).log_prob(target_r)
        )
        return z_nll + reward_nll


def train_mdn(model, sequences, epochs=10, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    z_batch = torch.cat([sequence[0] for sequence in sequences])
    action_batch = torch.cat([sequence[1] for sequence in sequences])
    target_z_batch = torch.cat([sequence[2] for sequence in sequences])
    target_r_batch = torch.cat([sequence[3] for sequence in sequences])
    for _ in range(epochs):
        loss = model.loss(
            z_batch, action_batch, target_z_batch, target_r_batch
        ).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


# ---------------------------------------------------------------------------
# C：线性控制器 a_t = W [z_t, h_t] + b，用 CMA-ES 进化
# ---------------------------------------------------------------------------


class LinearController:
    def __init__(self, input_size, action_size=3, seed=0):
        self.input_size = input_size
        self.action_size = action_size
        self.parameters = np.zeros(input_size * action_size + action_size)

    def to_weights(self, parameters):
        weights = parameters[: self.input_size * self.action_size].reshape(
            self.input_size, self.action_size
        )
        bias = parameters[self.input_size * self.action_size :]
        return weights, bias

    def act(self, features, parameters, noise=0.0):
        weights, bias = self.to_weights(parameters)
        action = features @ weights + bias
        if noise > 0:
            action += noise * np.random.randn(*action.shape)
        return np.clip(action, -1.0, 1.0)


class MinimalCMAES:
    """教学版 CMA-ES：仅保留核心逻辑，够进化 867 参数量级的 C。"""

    def __init__(self, dimension, population=32, seed=0, sigma=0.5):
        self.rng = np.random.RandomState(seed)
        self.dimension = dimension
        self.population = population
        self.sigma = sigma
        self.mean = np.zeros(dimension)
        self.covariance = np.eye(dimension)
        self.weights = np.log(population / 2 + 0.5) - np.log(
            np.arange(1, population + 1)
        )
        self.weights = np.maximum(self.weights, 0)
        self.weights = self.weights / self.weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights**2)
        self.covariance_learning = min(
            1.0, 2.0 * self.mu_eff / (dimension + 1.0) ** 2
        )

    def ask(self):
        samples = self.rng.multivariate_normal(
            self.mean, self.sigma**2 * self.covariance, self.population
        )
        return samples

    def tell(self, samples, fitness):
        order = np.argsort(fitness)
        self.mean = np.sum(self.weights[:, None] * samples[order], axis=0)
        deviations = samples[order] - self.mean
        covariance_update = np.einsum(
            "i,ij,ik->jk", self.weights, deviations, deviations
        )
        self.covariance = (
            1 - self.covariance_learning
        ) * self.covariance + self.covariance_learning * covariance_update / self.sigma**2


def evaluate_in_dream(
    vae,
    mdn,
    controller,
    parameters,
    rollouts=4,
    horizon=200,
    temperature=1.0,
    memory=True,
    seed_frame=None,
):
    """梦境评估：C 的 fitness = M 想象中累计预测奖励（与原文一致）。"""
    total_rewards = []
    vae.eval()
    mdn.eval()
    with torch.no_grad():
        for _ in range(rollouts):
            if seed_frame is None:
                frame = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8)
            else:
                frame = torch.from_numpy(np.asarray(seed_frame)).unsqueeze(0)
            mean, logvar = vae.encode(frame)
            z = vae.reparameterize(mean, logvar)
            hidden = torch.zeros(1, mdn.hidden_size)
            reward = 0.0
            for _ in range(horizon):
                features = (
                    torch.cat((z, hidden), dim=-1).numpy()
                    if memory
                    else z.numpy()
                )
                action = controller.act(features, parameters, noise=0.1)
                action_tensor = torch.from_numpy(action).float()
                z, predicted_reward, hidden = mdn.sample(
                    z, action_tensor, hidden, temperature=temperature
                )
                reward += float(predicted_reward.item())
            total_rewards.append(reward)
    return float(np.mean(total_rewards))


def collect_episodes(env, count, seed=0, max_steps=1000):
    episodes = []
    observation, _ = env.reset(seed=seed)
    for _ in range(count):
        episode = {"observations": [], "actions": [], "rewards": []}
        for step in range(max_steps):
            if step == 0 or np.random.rand() < 0.05:
                target_speed = np.random.uniform(0.1, 0.5)
            action = np.array(
                [
                    np.random.uniform(-1, 1),
                    float(target_speed),
                    float(np.random.rand() < 0.1),
                ]
            )
            observation, reward, terminated, truncated, _ = env.step(action)
            observation = resize_frame(observation)
            episode["observations"].append(observation.copy())
            episode["actions"].append(action)
            episode["rewards"].append(float(reward))
            if terminated or truncated:
                observation, _ = env.reset(seed=seed + len(episodes))
                break
        episodes.append(episode)
    return episodes


def extract_frames(episodes):
    frames = []
    for episode in episodes:
        frames.extend(episode["observations"])
    return np.stack(frames)


def build_mdn_sequences(vae, episodes, sequence_length=64):
    sequences = []
    for episode in episodes:
        frames = torch.from_numpy(np.stack(episode["observations"]))
        with torch.no_grad():
            mean, logvar = vae.encode(frames)
            z = vae.reparameterize(mean, logvar)
        actions = torch.from_numpy(np.stack(episode["actions"])).float()
        rewards = torch.from_numpy(np.stack(episode["rewards"])).float()
        for start in range(0, len(z) - 1, sequence_length):
            end = min(start + sequence_length, len(z) - 1)
            sequences.append(
                (z[start:end], actions[start:end], z[start + 1 : end + 1], rewards[start:end])
            )
    return sequences


def evaluate_real(
    vae, mdn, controller, parameters, rollouts=8, seed=0, memory=True, temperature=1.0
):
    """真实环境闭环：V 编码真实帧 -> C 出动作 -> 环境反馈 -> M 更新记忆。"""
    env = make_env()
    rewards = []
    with torch.no_grad():
        for rollout in range(rollouts):
            observation, _ = env.reset(seed=seed + rollout)
            hidden = torch.zeros(1, mdn.hidden_size)
            total = 0.0
            for _ in range(1000):
                frame = torch.from_numpy(resize_frame(observation)).unsqueeze(0)
                mean, logvar = vae.encode(frame)
                z = vae.reparameterize(mean, logvar)
                features = (
                    torch.cat((z, hidden), dim=-1).numpy()
                    if memory
                    else z.numpy()
                )
                action = controller.act(features, parameters, noise=0.0)
                action_tensor = torch.from_numpy(action).float()
                observation, reward, terminated, truncated, _ = env.step(
                    action.reshape(-1)
                )
                total += reward
                z, _, hidden = mdn.sample(
                    z, action_tensor, hidden, temperature=temperature
                )
                if terminated or truncated:
                    break
            rewards.append(total)
    return float(np.mean(rewards))


def evaluate_random_policy(rollouts=8, seed=0):
    env = make_env()
    rewards = []
    for rollout in range(rollouts):
        observation, _ = env.reset(seed=seed + rollout)
        total = 0.0
        for _ in range(1000):
            observation = resize_frame(observation)
            observation, reward, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    return float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/carracing-world-model"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollouts", type=int, default=400)
    parser.add_argument("--vae-epochs", type=int, default=10)
    parser.add_argument("--mdn-epochs", type=int, default=10)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()

    if args.tiny:
        args.rollouts = 4
        args.vae_epochs = 1
        args.mdn_epochs = 1
        args.generations = 2
        args.population = 4

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    env = make_env()
    print("== 1/5 收集数据：随机策略 rollout")
    episodes = collect_episodes(env, args.rollouts, seed=args.seed)
    frames = extract_frames(episodes)

    print("== 2/5 训练 V：ConvVAE")
    vae, vae_losses = train_vae(frames, epochs=args.vae_epochs, seed=args.seed)
    torch.save(vae.state_dict(), args.output / "vae.pt")

    print("== 3/5 训练 M：MDN-RNN")
    mdn = MDNRNN()
    sequences = build_mdn_sequences(vae, episodes)
    mdn_losses = train_mdn(mdn, sequences, epochs=args.mdn_epochs, seed=args.seed)
    torch.save(mdn.state_dict(), args.output / "mdn.pt")

    print("== 4/5 进化 C：CMA-ES")
    memory_size = 32 + mdn.hidden_size if not args.no_memory else 32
    controller = LinearController(memory_size)
    optimizer = MinimalCMAES(controller.parameters.size, population=args.population, seed=args.seed)
    best_fitness = -np.inf
    best_parameters = None
    for generation in range(args.generations):
        samples = optimizer.ask()
        fitness = np.array(
            [
                -evaluate_in_dream(
                    vae,
                    mdn,
                    controller,
                    sample,
                    memory=not args.no_memory,
                    temperature=args.temperature,
                )
                for sample in samples
            ]
        )
        optimizer.tell(samples, fitness)
        if -fitness.min() > best_fitness:
            best_fitness = -fitness.min()
            best_parameters = samples[np.argmin(fitness)]
        if generation % 25 == 0:
            print(f"   generation {generation}: best={best_fitness:.0f}")
    np.save(args.output / "controller.npy", best_parameters)

    print("== 5/5 真实环境评估")
    seed_frame = frames[0]
    dream_score = evaluate_in_dream(
        vae,
        mdn,
        controller,
        best_parameters,
        temperature=args.temperature,
        memory=not args.no_memory,
        seed_frame=seed_frame,
    )
    real_score = evaluate_real(
        vae,
        mdn,
        controller,
        best_parameters,
        memory=not args.no_memory,
        temperature=args.temperature,
    )
    random_score = evaluate_random_policy()

    result = {
        "experiment": "carracing-vmc-reference",
        "memory": not args.no_memory,
        "temperature": args.temperature,
        "dream_score": dream_score,
        "real_score": real_score,
        "random_policy_score": random_score,
        "vae_reconstruction_loss_first": vae_losses[0],
        "vae_reconstruction_loss_last": vae_losses[-1],
        "mdn_nll_first": mdn_losses[0],
        "mdn_nll_last": mdn_losses[-1],
        "rollouts": args.rollouts,
        "generations": args.generations,
        "population": args.population,
        "controller_parameters": int(controller.parameters.size),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint = args.output / "controller.npy"
    manifest = RunManifest(
        experiment="carracing-vmc-reference",
        route="A",
        seed=args.seed,
        dataset="carracing-v3-random-policy",
        split=f"{args.rollouts}-rollouts-seed-{args.seed}",
        command=f"python scripts/run_carracing.py --output {args.output} --seed {args.seed}",
        started_at=started_at,
        wall_time_seconds=time.perf_counter() - start_time,
        device="cpu",
        gpu="not-applicable",
        cuda="not-applicable",
        checkpoint_sha256=sha256_file(checkpoint),
        notes=(
            f"Python {platform.python_version()}, PyTorch {torch.__version__}; "
            "small-scale World Models reproduction, not Dreamer-lite"
        ),
    )
    manifest.save(args.output / "manifest.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
