"""空间世界路线使用的几何、神经场与占用预测组件。"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .foundations import make_camera_transform, transform_points


def occupancy_iou(logits, targets, threshold=0.5):
    prediction = torch.sigmoid(logits) > threshold
    target = targets.bool()
    intersection = (prediction & target).sum().float()
    union = (prediction | target).sum().float().clamp_min(1)
    return intersection / union


def make_moving_occupancy_dataset(num_samples=64, size=16, past=3, future=3, seed=0):
    """动作控制小方块未来位置，作为驾驶/4D 的共同 toy。"""
    rng = np.random.default_rng(seed)
    histories, actions, futures = [], [], []
    moves = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])
    for _ in range(num_samples):
        position = rng.integers(3, size - 3, size=2)
        velocity = moves[int(rng.integers(0, len(moves)))]
        frames = []
        for _ in range(past):
            grid = np.zeros((size, size), dtype=np.float32)
            grid[position[0] - 1 : position[0] + 2, position[1] - 1 : position[1] + 2] = 1
            frames.append(grid)
            position = np.clip(position + velocity, 1, size - 2)
        action = int(rng.integers(0, len(moves)))
        velocity = moves[action]
        future_frames = []
        for _ in range(future):
            position = np.clip(position + velocity, 1, size - 2)
            grid = np.zeros((size, size), dtype=np.float32)
            grid[position[0] - 1 : position[0] + 2, position[1] - 1 : position[1] + 2] = 1
            future_frames.append(grid)
        histories.append(frames)
        actions.append(action)
        futures.append(future_frames)
    return (
        torch.tensor(np.asarray(histories), dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(np.asarray(futures), dtype=torch.float32),
    )


class TinyOccupancyPredictor(nn.Module):
    def __init__(self, past=3, future=3, action_size=5):
        super().__init__()
        self.future = future
        self.action = nn.Embedding(action_size, 8)
        self.network = nn.Sequential(
            nn.Conv2d(past + 8, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, future, 1),
        )

    def forward(self, history, actions):
        batch, _, height, width = history.shape
        action = self.action(actions.long())[:, :, None, None].expand(batch, -1, height, width)
        return self.network(torch.cat((history, action), dim=1))


class TinyNeuralField(nn.Module):
    """坐标 → occupancy/color 的最小神经场，用于理解 NeRF 接口。"""

    def __init__(self, hidden_size=48):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),
        )

    def forward(self, coordinates):
        output = self.network(coordinates)
        density = F.softplus(output[..., :1])
        color = torch.sigmoid(output[..., 1:])
        return density, color


def make_colored_sphere_samples(num_samples=512, seed=0):
    generator = torch.Generator().manual_seed(seed)
    coordinates = torch.rand(num_samples, 3, generator=generator) * 2 - 1
    radius = torch.linalg.vector_norm(coordinates, dim=-1, keepdim=True)
    density = (radius < 0.65).float()
    color = (coordinates + 1) / 2 * density
    return coordinates, density, color
