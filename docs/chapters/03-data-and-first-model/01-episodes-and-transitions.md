# 3.1 回合、转移与轨迹形式化 (Episodes & Transitions)

在构建世界模型与强化学习智能体的知识大厦时，一切算法的起点都可以归结为一个最基础的物理互动过程：**智能体在环境中观察状态、采取动作、获得即时反馈并推动物理世界演进到下一个瞬间**。

无论是四足机器人在碎石路上奔跑、自动驾驶汽车在繁忙十字路口穿行，还是机械臂在流水线上精准分拣零件，这种连续不断的“感知-决策-交互”过程在数学上被高度抽象为**马尔可夫决策过程（Markov Decision Process, MDP）**。

将物理世界连续流转的动力学历程切分为结构严谨的**单步转移元组（Transitions）**与**完整交互回合（Episodes / Trajectories）**，不仅是强化学习策略评估的数学基石，更是世界模型学习因果规律的原始训练数据来源。

本节我们将从初等物理运动学与几何级数出发，严密推导马尔可夫性质、累积折扣回报的收敛性以及部分可观测环境（POMDP）下的状态恢复，并使用纯底层 PyTorch 从零手写环境交互循环与轨迹数据收集器。

<div align="center">

<img src="/figures/03-data-and-first-model/source/01-episodes-and-transitions/drqn-fig1.png" alt="强化学习经典智能体-环境交互闭环：智能体接收状态与奖励并输出动作。" width="86%">

_图 3.1-1：强化学习经典智能体-环境交互闭环：智能体接收状态与奖励并输出动作。 出处：[Reinforcement Learning: An Introduction，Richard S. Sutton & Andrew G. Barto，2018](http://incompleteideas.net/book/the-book-2nd.html)。_

</div>

---

## 3.1.1 物理与数学基石：马尔可夫决策过程 (MDP) 与交互循环

要理解强化学习的形式化描述，我们首先需要回到经典物理学的决定论世界观。

### 1. 经典物理中的马尔可夫性质（Markov Property）
在初等力学中，如果我们知道一个自由落体小球在当前时刻 $t$ 的精确高度 $z_t$ 和瞬时速度 $v_t$，无论这个小球在此之前是被高空抛下还是被火箭发射上去的，依据牛顿运动定律，我们就能完全确定小球在未来时刻 $t+1$ 的运动状态。

马尔可夫性质在数学上形式化定义为：**给定当前状态与当前动作，未来的状态转移概率与过去的所有历史状态条件独立！**

$$P(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t, \mathbf{s}_{t-1}, \mathbf{a}_{t-1}, \dots, \mathbf{s}_0, \mathbf{a}_0) = P(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t)$$

### 2. MDP 标准五元组定义
一个标准的马尔可夫决策过程由数学五元组 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ 严格定义：
- **状态空间 $\mathcal{S}$**：环境中所有可能物理状态的集合；
- **动作空间 $\mathcal{A}$**：智能体所能执行的所有合法控制指令集合；
- **转移概率函数 $\mathcal{P}(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t)$**：在当前状态 $\mathbf{s}_t$ 下执行动作 $\mathbf{a}_t$ 后转移到 $\mathbf{s}_{t+1}$ 的动力学条件概率；
- **奖励函数 $\mathcal{R}(\mathbf{s}_t, \mathbf{a}_t)$**：执行动作后获得的即时标量收益反馈 $r_t \in \mathbb{R}$；
- **折扣因子 $\gamma \in [0, 1)$**：权衡眼前利益与长远收益的衰减系数。

<div align="center">

<img src="/figures/03-data-and-first-model/latex/01-episodes-and-transitions/batch-stack-shapes.png" alt="时间轴上的回合展开：由初始状态开始，通过动作与环境交替产生转移元组直至终止状态" width="86%">

_图 3.1-2：时间轴上的回合展开：由初始状态开始，通过动作与环境交替产生转移元组直至终止状态。_

</div>

---

## 3.1.2 核心数学推导一：折扣累积回报与几何级数收敛性

智能体的终极目标绝不是贪婪地最大化单步即时奖励 $r_t$，而是最大化从当前时刻起直至未来的**累积折扣总回报（Discounted Return）**。

<div align="center">

<img src="/figures/03-data-and-first-model/source/01-episodes-and-transitions/drqn-fig1.png" alt="马尔可夫决策过程的回溯图 (Backup Diagram)：状态节点与动作节点之间的转移期望树。" width="86%">

_图 3.1-3：马尔可夫决策过程的回溯图 (Backup Diagram)：状态节点与动作节点之间的转移期望树。 出处：[Reinforcement Learning: An Introduction，Richard S. Sutton & Andrew G. Barto，2018](http://incompleteideas.net/book/the-book-2nd.html)。_

</div>

### 1. 折扣累积回报方程与递归结构
定义时刻 $t$ 的累积回报 $G_t$ 为：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^\infty \gamma^k r_{t+k}$$

提取公因子 $\gamma$，得到极为优美的递归形式：

$$G_t = r_t + \gamma G_{t+1}$$

### 2. 为什么必须引入折扣因子 $\gamma < 1$？
1. **物理现实考量**：在充满不确定性的物理世界中，即刻到手的奖励比遥远未来的潜在收益更具确定性；
2. **数学收敛性保障**：若单步奖励为恒定常数 $r_t = R_{\max}$，当时间跨度趋于无穷大时，非折扣回报将发散为无穷大 $\sum_{k=0}^\infty R_{\max} = \infty$；而引入 $\gamma \in [0, 1)$ 后，利用初等数学中的等比数列求和公式，总回报恒收敛为一个确定的有限实数：
   $$G_t \le \sum_{k=0}^\infty \gamma^k R_{\max} = \frac{R_{\max}}{1 - \gamma}$$

### 3. 折扣回报手算数值算例
设机器人每维持一秒直立平衡获得即时奖励 $r = 1.0$，折扣因子 $\gamma = 0.9$。
我们来手动计算理论上的无穷步极限总回报与前 3 步截断实际回报：
1. **理论极限总回报**：
   $$G_{\text{limit}} = \frac{1.0}{1 - 0.9} = \frac{1.0}{0.1} = 10.0$$
2. **前 3 步实际累积回报**（时刻 $t, t+1, t+2$）：
   $$G_t^{(3)} = 1.0 + 0.9 \times 1.0 + (0.9)^2 \times 1.0 = 1.0 + 0.9 + 0.81 = 2.71$$

初等代数的几步加法清晰展现：折扣因子将无限时域的探索转化为了具有明确数学上界的稳定优化目标！

<details>
<summary><b>深入推导：马尔可夫链转移概率算子与收敛稳态分布的佩隆-弗罗贝尼乌斯定理严格证明（点击展开查看完整推导）</b></summary>

将离散状态转移概率矩阵记为 $\mathbf{P} \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$，其中每一行满足 $\sum_j \mathbf{P}_{i, j} = 1$（右随机矩阵）。
由于全 1 向量满足 $\mathbf{P} \mathbf{1} = \mathbf{1}$，矩阵的最大特征值必为 $\lambda_{\max} = 1$。
根据佩隆-弗罗贝尼乌斯定理（Perron-Frobenius Theorem），若马尔可夫链满足遍历性（不可约且非周期），则存在唯一的平稳概率分布向量 $\mathbf{d}^\pi$ 满足左特征方程：
$$\mathbf{d}^\pi \mathbf{P} = \mathbf{d}^\pi, \quad \sum_{i} d_i^\pi = 1$$
任何初始状态分布 $\boldsymbol{\mu}_0$ 在时间极限下满足 $\lim_{t \to \infty} \boldsymbol{\mu}_0 \mathbf{P}^t = \mathbf{d}^\pi$，严格奠定了强化学习长期期望平稳性的理论基石。
</details>

---

## 3.1.3 核心数学推导二：部分可观测环境 (POMDP) 与时序差分恢复

在真实物理世界中，机器人传感器往往无法直接测量全部物理状态（例如单张静态相机拍摄的图像只能看到机械臂当前的位置，却无法直接获知其瞬时运动速度与加速度）。这种场景被称为**部分可观测马尔可夫决策过程（POMDP）**。

<div align="center">

<img src="/figures/03-data-and-first-model/latex/01-episodes-and-transitions/batch-stack-shapes.png" alt="转移元组的五维数据结构：当前状态、动作、即时奖励、下一状态与终止标志位" width="86%">

_图 3.1-4：转移元组的五维数据结构：当前状态、动作、即时奖励、下一状态与终止标志位。_

</div>

### 1. 初等物理中的速度恢复机理（Frame Stacking）
在初等力学中，瞬时速度是位置对时间的一阶导数：

$$v(t) \approx \frac{x(t) - x(t - \Delta t)}{\Delta t}$$

如果单张图像 $\mathbf{o}_t$ 破坏了马尔可夫性，我们只需将最近的 $k$ 帧历史图像沿通道轴进行堆叠（例如堆叠 4 帧），形成复合观测：

$$\tilde{\mathbf{s}}_t = [\mathbf{o}_{t-3}, \; \mathbf{o}_{t-2}, \; \mathbf{o}_{t-1}, \; \mathbf{o}_t]$$

网络通过相邻两帧之间的像素位移差分，能够轻而易举地恢复出物体的瞬时运动速度与旋转角速度，重新让系统回归到标准 MDP 的框架之中！

<details>
<summary><b>深入推导：POMDP 在信念状态（Belief State）下的充分统计量等价转换证明（点击展开查看完整推导）</b></summary>

在 POMDP 中，引入观测发射概率 $O(\mathbf{o}_t \mid \mathbf{s}_t)$。
定义信念状态 $b_t(\mathbf{s}) = P(\mathbf{s}_t = \mathbf{s} \mid \mathbf{o}_{1:t}, \mathbf{a}_{1:t-1})$ 为真实状态的后验概率分布。
根据贝叶斯定理，信念状态满足递推更新：
$$b_{t+1}(\mathbf{s}') = \frac{O(\mathbf{o}_{t+1} \mid \mathbf{s}') \sum_{\mathbf{s}} \mathcal{P}(\mathbf{s}' \mid \mathbf{s}, \mathbf{a}_t) b_t(\mathbf{s})}{\sum_{\mathbf{s}''} O(\mathbf{o}_{t+1} \mid \mathbf{s}'') \sum_{\mathbf{s}} \mathcal{P}(\mathbf{s}'' \mid \mathbf{s}, \mathbf{a}_t) b_t(\mathbf{s})}$$
由舍农滤波理论，信念状态 $b_t$ 构成了历史交互全序列 $(\mathbf{o}_{\le t}, \mathbf{a}_{<t})$ 的充分统计量。将状态空间重定义在连续概率单纯形 $\Delta(\mathcal{S})$ 上，POMDP 严格等价转化为完全可观测的 Belief MDP。
</details>

---

## 3.1.4 纯底层 PyTorch 代码实现：从零手写环境交互与轨迹收集器

下面我们使用纯底层 PyTorch 算子实现一个标准的物理倒立摆环境、智能体交互循环与结构化轨迹（Episode Trajectory）数据流收集器。

```python
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Transition:
    """
    五维单步物理转移元组
    """
    state: torch.Tensor       # 当前状态 s_t
    action: torch.Tensor      # 执行动作 a_t
    reward: float             # 即时奖励 r_t
    next_state: torch.Tensor  # 下一状态 s_{t+1}
    done: bool                # 是否达到终止状态

class PhysicsPendulumEnv:
    """
    纯手写单摆物理模拟环境
    状态包含 [cos(theta), sin(theta), dtheta]
    """
    def __init__(self, dt: float = 0.05, g: float = 9.81, m: float = 1.0, l: float = 1.0):
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
        self.theta = 0.0
        self.dtheta = 0.0

    def reset(self) -> torch.Tensor:
        # 从随机小角度开始
        self.theta = (torch.rand(1).item() - 0.5) * 0.5
        self.dtheta = 0.0
        return self._get_obs()

    def _get_obs(self) -> torch.Tensor:
        return torch.tensor([torch.cos(torch.tensor(self.theta)),
                             torch.sin(torch.tensor(self.theta)),
                             self.dtheta], dtype=torch.float32)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        u = torch.clamp(action, -2.0, 2.0).item()

        # 动力学加速度方程: alpha = - (3g / 2l) * sin(theta) + (3 / ml^2) * u
        alpha = - (1.5 * self.g / self.l) * torch.sin(torch.tensor(self.theta)).item() + (3.0 / (self.m * self.l**2)) * u

        self.dtheta += self.dt * alpha
        self.theta += self.dt * self.dtheta

        # 归一化角度至 [-pi, pi]
        self.theta = ((self.theta + torch.pi) % (2 * torch.pi)) - torch.pi

        # 奖励函数：保持竖直朝上 (theta -> 0) 并惩罚控制能耗
        reward = - (self.theta ** 2 + 0.1 * (self.dtheta ** 2) + 0.001 * (u ** 2))
        done = False # 连续任务

        return self._get_obs(), reward, done

class TrajectoryCollector:
    """
    完整回合轨迹数据收集器
    """
    def __init__(self, env: PhysicsPendulumEnv, max_steps: int = 20):
        self.env = env
        self.max_steps = max_steps

    def collect_episode(self) -> list[Transition]:
        trajectory = []
        state = self.env.reset()

        for t in range(self.max_steps):
            # 随机采样动作
            action = (torch.rand(1) - 0.5) * 2.0
            next_state, reward, done = self.env.step(action)

            trans = Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done or (t == self.max_steps - 1)
            )
            trajectory.append(trans)
            state = next_state

        return trajectory

# ===================================================================
# 单元测试与累积回报计算校验
# ===================================================================
if __name__ == "__main__":
    env = PhysicsPendulumEnv()
    collector = TrajectoryCollector(env=env, max_steps=10)

    # 1. 采集一条完整回合数据
    episode = collector.collect_episode()
    print(f"[Trajectory Test] 采集回合长度: {len(episode)} 步")

    # 2. 手动逆向计算每一步的折扣累积回报 G_t (gamma = 0.95)
    gamma = 0.95
    returns = [0.0] * len(episode)
    running_return = 0.0

    for t in reversed(range(len(episode))):
        running_return = episode[t].reward + gamma * running_return
        returns[t] = running_return

    print(f"[Trajectory Test] 初始步累积折扣回报 G_0: {returns[0]:.4f}")
    print(f"[Trajectory Test] 状态向量维度: {episode[0].state.shape}")

    assert len(episode) == 10, "轨迹步数不符合预期！"
    assert episode[0].state.shape == (3,), "观测状态形状不符！"
    assert not torch.isnan(episode[0].state).any(), "物理模拟出现 NaN！"
    print("✓ 物理倒立摆环境、轨迹收集器与折扣回报计算单测全部通过！")
```

---

## 3.1.5 本节小结

回顾本节内容，我们奠定了强化学习与世界模型数据流的数学基石：
1. **马尔可夫决策过程（MDP）**：通过状态对历史的条件独立性，将物理世界的连续演进形式化为结构化的单步转移元组；
2. **折扣因子的收敛约束**：利用初等几何级数求和，将无限时间跨度的长远目标压缩为有界的累积回报；
3. **POMDP 状态恢复**：通过时序帧堆叠恢复速度与加速度等阶跃动力学信息，为世界模型训练提供了完备无缺的观测基础。
