"""PixelWorld 上可检查的 learned dynamics 与模型预测控制。

这一层使用图片中可直接量出的方块位置。它不是 Dreamer，也不使用
privileged simulator state；它提供一个容易检查的下游基线，证明 learned
dynamics 的预测确实能被 Planner 用来改善真实行动。
"""

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data import MovingSquareWorld


class PositionDynamics(nn.Module):
    """学习 `position + action → next_position` 的小型残差模型。"""

    def __init__(self, action_size=5, hidden_size=64, coordinate_max=13.0):
        super().__init__()
        self.action_size = action_size
        self.coordinate_max = coordinate_max
        self.network = nn.Sequential(
            nn.Linear(2 + action_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, positions, actions):
        positions = positions.float()
        actions = F.one_hot(actions.long(), self.action_size).float()
        normalized = positions / self.coordinate_max
        delta = torch.tanh(self.network(torch.cat((normalized, actions), dim=-1)))
        # PixelWorld 每步最多移动一个格。残差结构写入这个已知边界。
        return torch.clamp(positions + delta, 0.0, self.coordinate_max)


def fit_position_dynamics(
    model,
    positions,
    actions,
    next_positions,
    updates=120,
    learning_rate=3e-3,
):
    positions = torch.as_tensor(positions, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.long)
    next_positions = torch.as_tensor(next_positions, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(updates):
        prediction = model(positions, actions)
        loss = F.mse_loss(prediction, next_positions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


@dataclass(frozen=True)
class BeamPlan:
    action: int
    actions: tuple
    predicted_distance: float
    expanded_sequences: int


@torch.no_grad()
def beam_plan(model, position, goal=(12.0, 12.0), horizon=4, beam_size=64):
    """在 learned dynamics 中保留最接近目标的若干动作序列。"""
    if horizon < 1:
        raise ValueError("规划 horizon 至少为 1")
    device = next(model.parameters()).device
    start = torch.as_tensor(position, dtype=torch.float32, device=device)
    goal = torch.as_tensor(goal, dtype=torch.float32, device=device)
    beams = [(start, tuple(), 0.0)]
    expanded = 0

    for _ in range(horizon):
        candidates = []
        for state, actions, path_cost in beams:
            repeated = state[None].expand(model.action_size, -1)
            action_ids = torch.arange(model.action_size, device=device)
            next_states = model(repeated, action_ids)
            for action, next_state in enumerate(next_states):
                distance = torch.abs(next_state - goal).sum().item()
                candidates.append(
                    (path_cost + distance, next_state, actions + (action,))
                )
                expanded += 1
        candidates.sort(key=lambda item: (item[0], item[2]))
        beams = [
            (state, actions, path_cost)
            for path_cost, state, actions in candidates[:beam_size]
        ]

    best_state, best_actions, _ = beams[0]
    distance = torch.abs(best_state - goal).sum().item()
    return BeamPlan(best_actions[0], best_actions, distance, expanded)


def run_pixelworld_controller(
    model,
    start,
    max_steps=30,
    horizon=4,
    beam_size=64,
):
    """每次在模型里规划多步，只在真实 PixelWorld 执行第一步。"""
    world = MovingSquareWorld()
    position = tuple(start)
    positions = [position]
    actions = []
    rewards = []
    for _ in range(max_steps):
        plan = beam_plan(
            model,
            position,
            goal=world.goal,
            horizon=horizon,
            beam_size=beam_size,
        )
        position = world.next_position(position, plan.action)
        positions.append(position)
        actions.append(plan.action)
        rewards.append(world.reward(position))
        if position == world.goal:
            break
    return {
        "positions": positions,
        "actions": actions,
        "rewards": rewards,
        "success": position == world.goal,
        "final_distance": float(
            abs(position[0] - world.goal[0]) + abs(position[1] - world.goal[1])
        ),
    }


def run_random_controller(start, max_steps=30, seed=0):
    """与模型规划使用相同真实环境和步数的随机基线。"""
    world = MovingSquareWorld()
    rng = random.Random(seed)
    position = tuple(start)
    positions = [position]
    actions = []
    for _ in range(max_steps):
        action = rng.randrange(5)
        position = world.next_position(position, action)
        positions.append(position)
        actions.append(action)
        if position == world.goal:
            break
    return {
        "positions": positions,
        "actions": actions,
        "success": position == world.goal,
        "final_distance": float(
            abs(position[0] - world.goal[0]) + abs(position[1] - world.goal[1])
        ),
    }


def evaluate_controllers(model, starts, max_steps=30, random_seeds=10):
    """在同一组起点上比较 learned MPC 与多个随机种子。"""
    planned = [run_pixelworld_controller(model, start, max_steps) for start in starts]
    random_runs = [
        run_random_controller(start, max_steps, seed)
        for start in starts
        for seed in range(random_seeds)
    ]
    return {
        "planned_success_rate": float(np.mean([run["success"] for run in planned])),
        "planned_final_distance": float(
            np.mean([run["final_distance"] for run in planned])
        ),
        "random_success_rate": float(
            np.mean([run["success"] for run in random_runs])
        ),
        "random_final_distance": float(
            np.mean([run["final_distance"] for run in random_runs])
        ),
        "planned_runs": planned,
        "random_runs": random_runs,
    }
