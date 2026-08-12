"""路线 A 使用的教学版 latent world model。

这不是 DreamerV3 的等价复现。它保留 Encoder、RSSM prior/posterior、
observation/reward/continue heads、imagination 与 Actor-Critic 接口，
让 CPU smoke 可以检查完整数据流。
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal, kl_divergence
import torch.nn.functional as F


@dataclass
class RSSMState:
    deterministic: torch.Tensor
    stochastic: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    @property
    def feature(self):
        return torch.cat((self.deterministic, self.stochastic), dim=-1)


class PixelEncoder(nn.Module):
    def __init__(self, embed_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, embed_size),
            nn.LayerNorm(embed_size),
        )

    def forward(self, observation):
        observation = observation.float()
        if observation.max() > 1.0:
            observation = observation / 255.0
        if observation.shape[-1] == 3:
            observation = observation.permute(0, 3, 1, 2)
        return self.network(observation)


class RSSM(nn.Module):
    def __init__(self, action_size=5, deter_size=64, stoch_size=16, embed_size=64):
        super().__init__()
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.gru = nn.GRUCell(stoch_size + action_size, deter_size)
        self.prior_head = nn.Linear(deter_size, 2 * stoch_size)
        self.posterior_head = nn.Linear(deter_size + embed_size, 2 * stoch_size)

    def initial(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        zeros_deter = torch.zeros(batch_size, self.deter_size, device=device)
        zeros_stoch = torch.zeros(batch_size, self.stoch_size, device=device)
        ones_std = torch.ones_like(zeros_stoch)
        return RSSMState(zeros_deter, zeros_stoch, zeros_stoch, ones_std)

    def distribution(self, stats):
        mean, raw_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(raw_std) + 0.1
        return Independent(Normal(mean, std), 1)

    def imagine_step(self, previous, action, sample=True):
        action = F.one_hot(action.long(), self.action_size).float()
        deterministic = self.gru(
            torch.cat((previous.stochastic, action), dim=-1),
            previous.deterministic,
        )
        distribution = self.distribution(self.prior_head(deterministic))
        stochastic = distribution.rsample() if sample else distribution.mean
        return RSSMState(
            deterministic,
            stochastic,
            distribution.mean,
            distribution.stddev,
        )

    def observe_step(self, previous, action, embed, sample=True):
        prior = self.imagine_step(previous, action, sample=sample)
        posterior_distribution = self.distribution(
            self.posterior_head(torch.cat((prior.deterministic, embed), dim=-1))
        )
        stochastic = (
            posterior_distribution.rsample()
            if sample
            else posterior_distribution.mean
        )
        posterior = RSSMState(
            prior.deterministic,
            stochastic,
            posterior_distribution.mean,
            posterior_distribution.stddev,
        )
        return prior, posterior

    def observe(self, embeds, actions, sample=True):
        """embeds [B,T,E] 与 actions [B,T] 对齐。"""
        state = self.initial(embeds.shape[0], embeds.device)
        priors = []
        posteriors = []
        for time in range(embeds.shape[1]):
            prior, state = self.observe_step(
                state,
                actions[:, time],
                embeds[:, time],
                sample=sample,
            )
            priors.append(prior)
            posteriors.append(state)
        return stack_states(priors), stack_states(posteriors)


def stack_states(states):
    return RSSMState(
        deterministic=torch.stack([state.deterministic for state in states], dim=1),
        stochastic=torch.stack([state.stochastic for state in states], dim=1),
        mean=torch.stack([state.mean for state in states], dim=1),
        std=torch.stack([state.std for state in states], dim=1),
    )


class TinyWorldModel(nn.Module):
    def __init__(self, action_size=5, embed_size=64, deter_size=64, stoch_size=16):
        super().__init__()
        self.encoder = PixelEncoder(embed_size)
        self.rssm = RSSM(action_size, deter_size, stoch_size, embed_size)
        feature_size = deter_size + stoch_size
        self.decoder = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ELU(),
            nn.Linear(256, 16 * 16 * 3),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def decode(self, feature):
        return torch.sigmoid(self.decoder(feature)).reshape(*feature.shape[:-1], 16, 16, 3)

    def forward(self, observations, actions, sample=True):
        batch, time = actions.shape
        # 第 t 个 action 预测 observations[t+1]。
        targets = observations[:, 1 : time + 1]
        embeds = self.encoder(targets.reshape(batch * time, 16, 16, 3))
        embeds = embeds.reshape(batch, time, -1)
        priors, posteriors = self.rssm.observe(embeds, actions, sample=sample)
        feature = posteriors.feature
        return {
            "prior": priors,
            "posterior": posteriors,
            "feature": feature,
            "reconstruction": self.decode(feature),
            "reward": self.reward_head(feature).squeeze(-1),
            "continue_logit": self.continue_head(feature).squeeze(-1),
        }


def world_model_loss(model, observations, actions, rewards, dones, free_nats=1.0):
    outputs = model(observations, actions)
    targets = observations[:, 1 : actions.shape[1] + 1].float() / 255.0
    reconstruction_loss = F.mse_loss(outputs["reconstruction"], targets)
    reward_loss = F.mse_loss(outputs["reward"], rewards.float())
    continue_target = 1.0 - dones.float()
    continue_loss = F.binary_cross_entropy_with_logits(
        outputs["continue_logit"], continue_target
    )
    prior = Independent(Normal(outputs["prior"].mean, outputs["prior"].std), 1)
    posterior = Independent(
        Normal(outputs["posterior"].mean, outputs["posterior"].std), 1
    )
    kl = kl_divergence(posterior, prior).mean()
    kl_loss = torch.maximum(kl, torch.tensor(free_nats, device=kl.device))
    total = reconstruction_loss + reward_loss + continue_loss + 0.1 * kl_loss
    metrics = {
        "total": total,
        "reconstruction": reconstruction_loss.detach(),
        "reward": reward_loss.detach(),
        "continue": continue_loss.detach(),
        "kl": kl.detach(),
    }
    return total, metrics, outputs


class Actor(nn.Module):
    def __init__(self, feature_size=80, action_size=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ELU(),
            nn.Linear(64, action_size),
        )

    def forward(self, feature):
        return torch.distributions.Categorical(logits=self.network(feature))


class Critic(nn.Module):
    def __init__(self, feature_size=80):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, feature):
        return self.network(feature).squeeze(-1)


def imagine(model, actor, start_state, horizon=5):
    """冻结 world model 后，从真实 posterior 开始想象。"""
    state = start_state
    features = []
    actions = []
    rewards = []
    continues = []
    log_probs = []

    for _ in range(horizon):
        distribution = actor(state.feature)
        action = distribution.sample()
        state = model.rssm.imagine_step(state, action)
        feature = state.feature
        features.append(feature)
        actions.append(action)
        log_probs.append(distribution.log_prob(action))
        rewards.append(model.reward_head(feature).squeeze(-1))
        continues.append(torch.sigmoid(model.continue_head(feature).squeeze(-1)))

    return {
        "features": torch.stack(features, dim=1),
        "actions": torch.stack(actions, dim=1),
        "log_probs": torch.stack(log_probs, dim=1),
        "rewards": torch.stack(rewards, dim=1),
        "continues": torch.stack(continues, dim=1),
    }


def lambda_returns(rewards, continues, values, bootstrap, discount=0.99, lambd=0.95):
    """从后往前计算 TD-lambda 目标。"""
    next_values = torch.cat((values[:, 1:], bootstrap[:, None]), dim=1)
    target = bootstrap
    returns = []
    for time in reversed(range(rewards.shape[1])):
        one_step = rewards[:, time] + discount * continues[:, time] * (
            (1 - lambd) * next_values[:, time] + lambd * target
        )
        returns.append(one_step)
        target = one_step
    return torch.stack(list(reversed(returns)), dim=1)


def batch_from_episodes(episodes, sequence_length=8):
    """把项目生成数据整理成一批 torch Tensor。"""
    observations = []
    actions = []
    rewards = []
    dones = []
    for episode in episodes:
        observations.append(episode.observations[: sequence_length + 1])
        actions.append(episode.actions[:sequence_length])
        rewards.append(episode.rewards[:sequence_length])
        dones.append(episode.dones[:sequence_length])
    return (
        torch.from_numpy(np.stack(observations)),
        torch.from_numpy(np.stack(actions)),
        torch.from_numpy(np.stack(rewards)),
        torch.from_numpy(np.stack(dones)),
    )
