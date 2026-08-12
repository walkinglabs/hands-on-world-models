"""共同基础使用的项目内生成数据与轨迹容器。"""

from dataclasses import dataclass
import random

import numpy as np


MOVE = {
    0: (0, 0),
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}

ACTION_NAMES = {
    0: "stay",
    1: "left",
    2: "right",
    3: "up",
    4: "down",
}


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    episode_id: str = "episode"

    def validate(self):
        time = len(self.observations)
        if time < 2:
            raise ValueError("一段经历至少要有两个观察")
        if len(self.actions) != time - 1:
            raise ValueError("T 个观察应对应 T-1 个动作")
        if len(self.rewards) != time - 1 or len(self.dones) != time - 1:
            raise ValueError("reward 和 done 必须与 action 在时间上对齐")
        if np.any(self.dones[:-1]):
            raise ValueError("episode 结束以后不能继续保存 transition")
        return self

    def transitions(self):
        self.validate()
        for index, action in enumerate(self.actions):
            yield {
                "observation": self.observations[index],
                "action": int(action),
                "reward": float(self.rewards[index]),
                "next_observation": self.observations[index + 1],
                "done": bool(self.dones[index]),
                "episode_id": self.episode_id,
                "time": index,
            }


class MovingSquareWorld:
    """用彩色方块构造最小动作条件视频。"""

    def __init__(self, size=16, square_size=3, goal=(12, 12)):
        self.size = size
        self.square_size = square_size
        self.goal = goal

    def _clip(self, value):
        return max(0, min(self.size - self.square_size, value))

    def next_position(self, position, action):
        if action not in MOVE:
            raise ValueError(f"未知动作：{action}")
        row, col = position
        dr, dc = MOVE[action]
        return self._clip(row + dr), self._clip(col + dc)

    def render(self, position):
        image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        goal_row, goal_col = self.goal
        image[
            goal_row : goal_row + self.square_size,
            goal_col : goal_col + self.square_size,
            1,
        ] = 110
        row, col = position
        image[row : row + self.square_size, col : col + self.square_size, 0] = 255
        return image

    def reward(self, position):
        return 1.0 if position == self.goal else -0.01

    def generate(self, actions, start=(2, 2), episode_id="pixelworld-0"):
        positions = [start]
        observations = [self.render(start)]
        rewards = []
        dones = []
        position = start

        for index, action in enumerate(actions):
            position = self.next_position(position, int(action))
            positions.append(position)
            observations.append(self.render(position))
            rewards.append(self.reward(position))
            dones.append(position == self.goal or index == len(actions) - 1)

        episode = Episode(
            observations=np.stack(observations),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=bool),
            episode_id=episode_id,
        ).validate()
        return episode, positions


def make_pixelworld_dataset(num_episodes=12, length=18, seed=0):
    """生成可复现的小型动作视频数据。"""
    rng = random.Random(seed)
    world = MovingSquareWorld()
    episodes = []
    for index in range(num_episodes):
        start = (rng.randint(0, 7), rng.randint(0, 7))
        actions = [rng.choice(tuple(MOVE)) for _ in range(length)]
        episode, _ = world.generate(
            actions,
            start=start,
            episode_id=f"pixelworld-{index:03d}",
        )
        episodes.append(episode)
    return episodes


def split_by_episode(episodes, train_ratio=0.7, val_ratio=0.15):
    """按 episode 切分，避免相邻帧泄漏到不同集合。"""
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("切分比例不合法")
    if train_ratio + val_ratio >= 1:
        raise ValueError("必须为 test 留出 episode")

    total = len(episodes)
    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
    return {
        "train": episodes[:train_end],
        "val": episodes[train_end:val_end],
        "test": episodes[val_end:],
    }


class ReplayBuffer:
    """只在 episode 内采样连续序列的经验回放池。"""

    def __init__(self):
        self.episodes = []

    def add(self, episode):
        self.episodes.append(episode.validate())

    def sample(self, batch_size, sequence_length, seed=0):
        eligible = [
            episode
            for episode in self.episodes
            if len(episode.actions) >= sequence_length
        ]
        if not eligible:
            raise ValueError("没有足够长的 episode")

        rng = random.Random(seed)
        samples = []
        for _ in range(batch_size):
            episode = rng.choice(eligible)
            start = rng.randint(0, len(episode.actions) - sequence_length)
            stop = start + sequence_length
            samples.append(
                {
                    "observations": episode.observations[start : stop + 1],
                    "actions": episode.actions[start:stop],
                    "rewards": episode.rewards[start:stop],
                    "dones": episode.dones[start:stop],
                    "episode_id": episode.episode_id,
                    "start": start,
                }
            )
        return samples
