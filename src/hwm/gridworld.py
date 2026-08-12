"""第 0 章使用的九格世界。

实现只依赖 Python 标准库。它有意保持短小，让世界模型、规划器和
真实环境的边界可以直接看见。
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
import random


ACTIONS = {
    "down": (1, 0),
    "right": (0, 1),
    "up": (-1, 0),
    "left": (0, -1),
}

ACTION_SYMBOLS = {
    "down": "↓",
    "right": "→",
    "up": "↑",
    "left": "←",
}


@dataclass(frozen=True)
class Transition:
    state: tuple
    action: str
    reward: float
    next_state: tuple
    done: bool


@dataclass(frozen=True)
class PlanningResult:
    action: str
    actions: tuple
    predicted_return: float
    evaluated_sequences: int


class GridWorld:
    """一个带墙壁、陷阱和可选打滑的九格环境。"""

    def __init__(
        self,
        rows=3,
        cols=3,
        start=(0, 0),
        goal=(0, 2),
        walls=((1, 1),),
        traps=((0, 1),),
        slip_probability=0.0,
    ):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.walls = frozenset(walls)
        self.traps = frozenset(traps)
        self.slip_probability = slip_probability

        if start in self.walls or goal in self.walls:
            raise ValueError("起点和终点不能位于墙壁中")
        if not 0.0 <= slip_probability <= 1.0:
            raise ValueError("slip_probability 必须在 0 到 1 之间")

    @property
    def terminal_states(self):
        return self.traps | {self.goal}

    def inside(self, state):
        row, col = state
        return 0 <= row < self.rows and 0 <= col < self.cols

    def next_state(self, state, action):
        """人写的确定性世界模型。"""
        if action not in ACTIONS:
            raise ValueError(f"未知动作：{action}")
        if state in self.terminal_states:
            return state

        row, col = state
        dr, dc = ACTIONS[action]
        candidate = (row + dr, col + dc)

        if not self.inside(candidate) or candidate in self.walls:
            return state
        return candidate

    def reward(self, state, next_state):
        if next_state == self.goal:
            return 10.0
        if next_state in self.traps:
            return -10.0
        return -1.0

    def transition(self, state, action):
        next_state = self.next_state(state, action)
        reward = self.reward(state, next_state)
        done = next_state in self.terminal_states
        return Transition(state, action, reward, next_state, done)

    def step(self, state, action, rng=None):
        """在真实环境中执行动作。

        打滑时机器人停在原地。模型可以是确定的，环境仍可带有随机性。
        """
        rng = rng or random
        slipped = rng.random() < self.slip_probability
        actual_next_state = state if slipped else self.next_state(state, action)
        reward = self.reward(state, actual_next_state)
        done = actual_next_state in self.terminal_states
        return Transition(state, action, reward, actual_next_state, done)

    def greedy_action(self, state):
        """只看下一格离终点多远，不考虑陷阱。"""
        choices = []
        for index, action in enumerate(("right", "down", "up", "left")):
            next_state = self.next_state(state, action)
            distance = abs(next_state[0] - self.goal[0]) + abs(
                next_state[1] - self.goal[1]
            )
            choices.append((distance, index, action))
        return min(choices)[2]

    def render(self, state=None, path=()):
        path = set(path)
        cells = []
        for row in range(self.rows):
            line = []
            for col in range(self.cols):
                cell = (row, col)
                symbol = "·"
                if cell in path:
                    symbol = "○"
                if cell in self.walls:
                    symbol = "■"
                elif cell in self.traps:
                    symbol = "×"
                elif cell == self.goal:
                    symbol = "G"
                elif cell == self.start:
                    symbol = "S"
                if cell == state:
                    symbol = "A"
                line.append(symbol)
            cells.append(" ".join(line))
        return "\n".join(cells)


def rollout(model, start, actions):
    """在模型中推演一串尚未执行的动作。"""
    transitions = []
    state = start
    for action in actions:
        transition = model.transition(state, action)
        transitions.append(transition)
        state = transition.next_state
        if transition.done:
            break
    return transitions


def lookahead(model, state, depth, action_order=None):
    """穷举短动作序列，返回预测回报最高的序列。"""
    if depth < 1:
        raise ValueError("规划深度至少为 1")

    action_order = action_order or tuple(ACTIONS)
    best_actions = None
    best_return = float("-inf")
    evaluated = 0

    for actions in product(action_order, repeat=depth):
        imagined = rollout(model, state, actions)
        predicted_return = sum(item.reward for item in imagined)
        evaluated += 1

        if predicted_return > best_return:
            best_return = predicted_return
            best_actions = actions

    return PlanningResult(
        action=best_actions[0],
        actions=best_actions,
        predicted_return=best_return,
        evaluated_sequences=evaluated,
    )


def mpc_episode(environment, model, depth, max_steps=20, seed=0):
    """在模型里规划多步，在环境里只执行一步。"""
    rng = random.Random(seed)
    state = environment.start
    transitions = []
    plans = []

    for _ in range(max_steps):
        plan = lookahead(model, state, depth)
        transition = environment.step(state, plan.action, rng)
        plans.append(plan)
        transitions.append(transition)
        state = transition.next_state
        if transition.done:
            break

    return transitions, plans


class EmpiricalDynamics:
    """从 transition 计数学习的概率世界模型。"""

    def __init__(self):
        self.counts = defaultdict(Counter)
        self.reward_sums = defaultdict(float)
        self.reward_counts = Counter()
        self.terminal_states = set()

    def update(self, transition):
        key = (transition.state, transition.action)
        target = transition.next_state
        self.counts[key][target] += 1
        reward_key = (transition.state, transition.action, target)
        self.reward_sums[reward_key] += transition.reward
        self.reward_counts[reward_key] += 1
        if transition.done:
            self.terminal_states.add(target)

    def fit(self, transitions):
        for transition in transitions:
            self.update(transition)
        return self

    def distribution(self, state, action):
        counts = self.counts.get((state, action))
        if not counts:
            return {}
        total = sum(counts.values())
        return {
            next_state: count / total
            for next_state, count in sorted(counts.items())
        }

    def transition(self, state, action):
        """为规划提供一个最可能的下一状态。

        完整分布仍可通过 distribution 查看。未知转移保持原地，并给出一步代价，
        使课程可以清楚观察“数据没有覆盖”的边界。
        """
        distribution = self.distribution(state, action)
        if not distribution:
            return Transition(state, action, -1.0, state, False)

        next_state = max(
            distribution,
            key=lambda candidate: (distribution[candidate], candidate),
        )
        reward_key = (state, action, next_state)
        reward = self.reward_sums[reward_key] / self.reward_counts[reward_key]
        done = next_state in self.terminal_states
        return Transition(state, action, reward, next_state, done)


def format_trajectory(transitions):
    """把轨迹整理成适合教学展示的一行文字。"""
    if not transitions:
        return "(empty)"
    parts = [str(transitions[0].state)]
    for item in transitions:
        parts.append(ACTION_SYMBOLS[item.action])
        parts.append(str(item.next_state))
    return " ".join(parts)
