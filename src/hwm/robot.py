"""机器人路线使用的项目内桌面数据与最小模型。"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


INSTRUCTIONS = ("移动到红色目标", "移动到绿色目标")


def _draw_disk(image, point, color, radius=2):
    height, width, _ = image.shape
    col = int(np.clip(point[0] * (width - 1), 0, width - 1))
    row = int(np.clip(point[1] * (height - 1), 0, height - 1))
    rr, cc = np.ogrid[:height, :width]
    mask = (rr - row) ** 2 + (cc - col) ** 2 <= radius**2
    image[mask] = color


def render_tabletop(state, size=32):
    """state = gripper, red goal, green goal, obstacle。"""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = 18
    _draw_disk(image, state[2:4], (220, 40, 40))
    _draw_disk(image, state[4:6], (40, 220, 70))
    _draw_disk(image, state[6:8], (70, 100, 230), radius=3)
    _draw_disk(image, state[0:2], (240, 240, 240), radius=2)
    return image


def step_tabletop(state, action, step_size=0.12, obstacle_radius=0.13):
    state = np.asarray(state, dtype=np.float32).copy()
    action = np.asarray(action, dtype=np.float32)
    norm = np.linalg.norm(action)
    if norm > 1:
        action = action / norm
    candidate = np.clip(state[:2] + step_size * action, 0.0, 1.0)
    collision = np.linalg.norm(candidate - state[6:8]) < obstacle_radius
    if not collision:
        state[:2] = candidate
    return state, bool(collision)


def expert_action(state, instruction, obstacle_radius=0.18):
    target = state[2:4] if instruction == 0 else state[4:6]
    direction = target - state[:2]
    direction /= np.linalg.norm(direction) + 1e-6
    candidate = state[:2] + 0.12 * direction
    if np.linalg.norm(candidate - state[6:8]) < obstacle_radius:
        # 直走会碰障碍时，比较左右两个垂直方向。
        left = np.array([-direction[1], direction[0]], dtype=np.float32)
        right = -left
        left_clearance = np.linalg.norm(state[:2] + 0.12 * left - state[6:8])
        right_clearance = np.linalg.norm(state[:2] + 0.12 * right - state[6:8])
        direction = left if left_clearance > right_clearance else right
    return direction.astype(np.float32)


def make_tabletop_dataset(num_samples=256, chunk_size=3, seed=0):
    """生成 image + instruction + proprioception + action chunk。"""
    rng = np.random.default_rng(seed)
    images, states, instructions, chunks = [], [], [], []
    next_states, collisions = [], []
    for _ in range(num_samples):
        state = rng.uniform(0.12, 0.88, size=8).astype(np.float32)
        # 障碍有一部分故意放在抓手与目标之间。
        instruction = int(rng.integers(0, 2))
        target = state[2:4] if instruction == 0 else state[4:6]
        if rng.random() < 0.55:
            state[6:8] = np.clip((state[:2] + target) / 2 + rng.normal(0, 0.03, 2), 0.1, 0.9)
        initial = state.copy()
        chunk = []
        first_next = None
        first_collision = False
        for index in range(chunk_size):
            action = expert_action(state, instruction)
            chunk.append(action)
            state, collision = step_tabletop(state, action)
            if index == 0:
                first_next = state.copy()
                first_collision = collision
        images.append(render_tabletop(initial))
        states.append(initial)
        instructions.append(instruction)
        chunks.append(chunk)
        next_states.append(first_next)
        collisions.append(first_collision)
    return {
        "images": torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0,
        "states": torch.from_numpy(np.stack(states)),
        "instructions": torch.tensor(instructions, dtype=torch.long),
        "action_chunks": torch.tensor(np.asarray(chunks), dtype=torch.float32),
        "next_states": torch.from_numpy(np.stack(next_states)),
        "collisions": torch.tensor(collisions, dtype=torch.float32),
    }


def make_outcome_dataset(num_samples=512, seed=0):
    """用随机候选动作覆盖碰撞与非碰撞后果。"""
    rng = np.random.default_rng(seed)
    states, actions, next_states, collisions = [], [], [], []
    for _ in range(num_samples):
        state = rng.uniform(0.12, 0.88, size=8).astype(np.float32)
        # 一半样本把障碍放在随机动作前方，确保碰撞标签不是常量。
        action = rng.normal(size=2).astype(np.float32)
        action /= np.linalg.norm(action) + 1e-6
        if rng.random() < 0.5:
            state[6:8] = np.clip(state[:2] + 0.1 * action, 0.05, 0.95)
        next_state, collision = step_tabletop(state, action)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        collisions.append(collision)
    return {
        "states": torch.from_numpy(np.stack(states)),
        "actions": torch.from_numpy(np.stack(actions)),
        "next_states": torch.from_numpy(np.stack(next_states)),
        "collisions": torch.tensor(collisions, dtype=torch.float32),
    }


class TinyVLA(nn.Module):
    """教学版 image + language + proprioception → action chunk。"""

    def __init__(self, state_size=8, chunk_size=3):
        super().__init__()
        self.chunk_size = chunk_size
        self.vision = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 20, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.language = nn.Embedding(len(INSTRUCTIONS), 8)
        self.policy = nn.Sequential(
            nn.Linear(80 + 8 + state_size, 64),
            nn.ReLU(),
            nn.Linear(64, chunk_size * 2),
            nn.Tanh(),
        )

    def forward(self, images, instructions, states):
        features = torch.cat(
            (self.vision(images), self.language(instructions.long()), states), dim=-1
        )
        return self.policy(features).reshape(-1, self.chunk_size, 2)


class TabletopOutcomeModel(nn.Module):
    """当前 state + 候选动作 → 下一 state 与碰撞概率。"""

    def __init__(self, state_size=8):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.next_state = nn.Linear(64, state_size)
        self.collision = nn.Linear(64, 1)

    def forward(self, states, actions):
        hidden = self.trunk(torch.cat((states, actions), dim=-1))
        return self.next_state(hidden), self.collision(hidden).squeeze(-1)


def outcome_loss(model, states, actions, next_states, collisions):
    predicted_state, collision_logit = model(states, actions)
    state_loss = F.mse_loss(predicted_state, next_states)
    collision_loss = F.binary_cross_entropy_with_logits(collision_logit, collisions)
    return state_loss + collision_loss, state_loss.detach(), collision_loss.detach()


@torch.no_grad()
def rerank_actions(model, state, instruction, candidates, collision_weight=2.0):
    states = state[None].expand(len(candidates), -1)
    next_states, collision_logits = model(states, candidates)
    target = states[:, 2:4] if int(instruction) == 0 else states[:, 4:6]
    distance = torch.linalg.vector_norm(next_states[:, :2] - target, dim=-1)
    score = -distance - collision_weight * torch.sigmoid(collision_logits)
    return int(score.argmax()), score


@torch.no_grad()
def rollout_vla(model, state, instruction, max_steps=20, success_radius=0.15):
    """每一步重新渲染并执行 action chunk 的第一步。"""
    state = np.asarray(state, dtype=np.float32).copy()
    target = state[2:4] if int(instruction) == 0 else state[4:6]
    initial_distance = float(np.linalg.norm(state[:2] - target))
    collisions = 0
    trajectory = [state[:2].copy()]
    for _ in range(max_steps):
        image = torch.from_numpy(render_tabletop(state)).permute(2, 0, 1)
        image = image.float()[None] / 255.0
        state_tensor = torch.from_numpy(state)[None]
        instruction_tensor = torch.tensor([instruction], dtype=torch.long)
        action = model(image, instruction_tensor, state_tensor)[0, 0].cpu().numpy()
        state, collision = step_tabletop(state, action)
        collisions += int(collision)
        trajectory.append(state[:2].copy())
        if np.linalg.norm(state[:2] - target) < success_radius:
            break
    final_distance = float(np.linalg.norm(state[:2] - target))
    return {
        "success": final_distance < success_radius,
        "collisions": collisions,
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "trajectory": np.stack(trajectory),
    }


def evaluate_vla(model, states, instructions, max_steps=20):
    runs = [
        rollout_vla(model, state, int(instruction), max_steps=max_steps)
        for state, instruction in zip(states, instructions)
    ]
    return {
        "success_rate": float(np.mean([run["success"] for run in runs])),
        "mean_collisions": float(np.mean([run["collisions"] for run in runs])),
        "initial_distance": float(
            np.mean([run["initial_distance"] for run in runs])
        ),
        "final_distance": float(np.mean([run["final_distance"] for run in runs])),
        "runs": runs,
    }


def evaluate_reranker(model, num_cases=128, seed=0, collision_weight=4.0):
    """在直达动作必撞的场景中评价后果模型的安全—进展取舍。"""
    rng = np.random.default_rng(seed)
    direct_collisions = []
    chosen_collisions = []
    chosen_progress = []
    for _ in range(num_cases):
        state = rng.uniform(0.12, 0.88, size=8).astype(np.float32)
        instruction = int(rng.integers(0, 2))
        target = state[2:4] if instruction == 0 else state[4:6]
        direction = target - state[:2]
        direction /= np.linalg.norm(direction) + 1e-6
        state[6:8] = np.clip(state[:2] + 0.1 * direction, 0.05, 0.95)
        perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        candidates = torch.tensor(
            np.stack((direction, perpendicular, -perpendicular, -direction)),
            dtype=torch.float32,
        )
        state_tensor = torch.from_numpy(state)
        chosen, _ = rerank_actions(
            model,
            state_tensor,
            instruction,
            candidates,
            collision_weight=collision_weight,
        )
        outcomes = [step_tabletop(state, action.numpy()) for action in candidates]
        direct_collisions.append(outcomes[0][1])
        chosen_collisions.append(outcomes[chosen][1])
        before = np.linalg.norm(state[:2] - target)
        after = np.linalg.norm(outcomes[chosen][0][:2] - target)
        chosen_progress.append(before - after)
    return {
        "direct_collision_rate": float(np.mean(direct_collisions)),
        "reranked_collision_rate": float(np.mean(chosen_collisions)),
        "reranked_mean_progress": float(np.mean(chosen_progress)),
    }
