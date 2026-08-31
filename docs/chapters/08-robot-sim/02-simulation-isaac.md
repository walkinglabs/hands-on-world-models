# 8.2 高性能 GPU 仿真与 Isaac Gym/Isaac Sim

假设一个四足环境每秒推进 1,000 步。只运行一个环境时，收集一亿步需要约 28 小时；若 4,096 个环境能够并行推进，理想下限约为 24 秒。实际速度不会线性增长，但这个数量级差异说明了为什么机器人强化学习重视并行仿真。

<div align="center">
<img src="/figures/08-robot-sim/source/02-simulation-isaac/isaacgym-fig1.png" alt="Isaac Gym 在同一 GPU 仿真平台中运行多类机械臂、四足与灵巧操作任务。" width="86%">

_图 8.2-1：Isaac Gym 在同一 GPU 仿真平台中运行多类机械臂、四足与灵巧操作任务。 出处：Viktor Makoviychuk et al.，[Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning](https://arxiv.org/abs/2108.10470)（2021），Figure 1。_
</div>

本节关注两个问题：多刚体状态怎样组织成批量张量，以及仿真与策略都驻留在 GPU 时，为什么可以减少设备间的数据搬运。Isaac Gym 是这一计算管线的代表；Isaac Sim/Isaac Lab 则提供更完整的机器人、传感器与场景工作流，两者不应被视为同一个接口。

<div align="center">
<img src="/figures/08-robot-sim/source/02-simulation-isaac/orbit-fig1.png" alt="Orbit 将机器人、传感器、场景资产、运动生成与遥操作组合成统一的 Isaac Sim 学习工作流。" width="86%">

_图 8.2-2：Orbit 将机器人、传感器、场景资产、运动生成与遥操作组合成统一的 Isaac Sim 学习工作流。 出处：Mayank Mittal et al.，[Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments](https://arxiv.org/abs/2301.04195)（2023），Figure 1。_
</div>

## 历史溯源与强化学习仿真的通信瓶颈

MuJoCo 面向模型式控制提供快速的刚体动力学、接触求解与导数计算 [[Todorov et al., 2012]](https://doi.org/10.1109/IROS.2012.6386109)。传统机器人学习管线常在 CPU 上推进仿真，再把观测复制到 GPU 训练策略；当并行环境数量增加时，仿真吞吐和 CPU—GPU 数据传输都可能成为瓶颈。这里讨论的是计算管线差异，而不是用 MuJoCo 论文证明所有 CPU 仿真器精度更高或无法扩展。

强化学习循环反复执行“读取观测—计算动作—推进物理—计算奖励”。若仿真在 CPU、策略在 GPU，每一步都要传输观测和动作。环境较少时，这个开销可能并不突出；并行环境增多后，物理计算、同步和数据传输都可能成为瓶颈。

<div align="center">
<img src="/figures/08-robot-sim/source/02-simulation-isaac/isaacgym-fig3.png" alt="Isaac Gym 的端到端 GPU 管线让仿真状态直接进入策略训练，避免逐步往返 CPU。" width="86%">

_图 8.2-3：Isaac Gym 的端到端 GPU 管线让仿真状态直接进入策略训练，避免逐步往返 CPU。 出处：Viktor Makoviychuk et al.，[Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning](https://arxiv.org/abs/2108.10470)（2021），Figure 3。_
</div>

在这一背景下，Makoviychuk 等人提出 Isaac Gym：一个面向机器人学习、把 PhysX 仿真与策略训练数据都保留在 GPU 上的平台 [[Makoviychuk et al., 2021]](https://arxiv.org/abs/2108.10470)。论文强调的是端到端 GPU 管线减少 CPU—GPU 数据传输并支持大量并行环境；“首个”并不是理解其贡献所必需的结论。

## 物理动力学的降维解析与系统张量化

先写出单个环境的动力学，再把环境编号增加为批次维度。

### 从单个质点到欧拉-拉格朗日体系

回顾高中物理，当我们研究单个质点（质量为 $m$）的运动时，最核心的法则是牛顿第二定律。合外力 $F$ 与其加速度 $a$（即位移 $x$ 对时间的二阶导数）的对应关系被定义为：

$$
F = m a = m \frac{d^2 x}{dt^2}
$$

如果是旋转运动，我们有其等价的角动量形式：

$$
\tau = I \alpha = I \frac{d^2 \theta}{dt^2}
$$

其中 $\tau$ 是力矩，$I$ 是转动惯量，$\alpha$ 是角加速度。多关节机器人中，一个关节的运动会改变其他连杆的速度与惯量，因此各自由度相互耦合。直接逐个列写内部约束力会产生较长的常微分—代数方程组，广义坐标可以把它们整理成统一形式。

为了系统且优雅地描述这种复杂动力学，学术界采用了拉格朗日力学（Lagrangian Mechanics）体系。假设机器人的姿态可以由一组广义坐标（Generalized Coordinates） $q \in \mathbb{R}^n$ 唯一确定（例如各个关节的旋转角度），其广义速度为 $\dot{q} \in \mathbb{R}^n$。机器人的动能 $T$ 可以严谨地表达为一个关于速度的二次型形式：

$$
T(q, \dot{q}) = \frac{1}{2} \dot{q}^\top M(q) \dot{q}
$$

在这里，$M(q) \in \mathbb{R}^{n \times n}$ 被称为质量矩阵（Mass Matrix）。它是一个对称且正定的矩阵，物理上反映了机器人在当前姿态 $q$ 下的等效质量或转动惯量分布。同时，系统的势能 $V(q)$ 则仅受当前姿态和重力场影响。系统拉格朗日函数 $\mathcal{L}$ 定义为动能与势能之差：

$$
\mathcal{L}(q, \dot{q}) = T(q, \dot{q}) - V(q)
$$

基于最小作用量原理，系统轨迹必须满足欧拉-拉格朗日方程：

$$
\frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{q}} \right) - \frac{\partial \mathcal{L}}{\partial q} = \tau
$$

将前面的动能表达式与拉格朗日函数代入欧拉-拉格朗日方程中展开推导，我们便得到了机器人控制理论中最著名的多刚体动力学方程（Rigid-body Dynamics Equation）：

$$
M(q) \ddot{q} + C(q, \dot{q}) \dot{q} + G(q) = \tau
$$

我们对该公式中的每一项进行细致拆解：

- $M(q) \ddot{q}$：惯性项。它直接对应了高中物理中的 $m a$，不同之处在于这里的质量 $M(q)$ 不再是常数，而是随着机器人各个关节角度变化而实时更新的矩阵。
- $C(q, \dot{q}) \dot{q}$：科里奥利力与离心力项。这是由连杆之间的非线性运动耦合产生的，本质是动能项中质量矩阵随时间的导数 $\dot{M}(q)$ 引发的附加惯性力。
- $G(q)$：重力项。由势能对坐标的梯度产生，代表为了维持机器人在当前姿态对抗重力所必须克服的静态力。
- $\tau \in \mathbb{R}^n$：广义外力，主要表现为电机输出的控制扭矩。

### 动力学方程的张量化升维

设有 $N$ 个相同机器人环境。为了看清批次维度，可以先把每个环境的无约束加速度示意为：

$$
\text{对于环境 } i = 1, 2, \dots, N: \quad \ddot{q}_i = M(q_i)^{-1} (\tau_i - C(q_i, \dot{q}_i) \dot{q}_i - G(q_i))
$$

实际引擎不会对每个环境显式计算矩阵逆，这个式子只用于标出环境之间可以并行。把广义坐标堆叠后，$\mathbf{Q} \in \mathbb{R}^{N \times n}$ 的第 0 维就是环境编号；观测、动作、奖励和终止标记也采用相同的批次布局。

<div align="center">
<img src="/figures/08-robot-sim/source/02-simulation-isaac/brax-fig2.png" alt="Brax 的吞吐曲线展示批量环境数增加时并行物理仿真的扩展规律与硬件差异。" width="86%">

_图 8.2-4：Brax 的吞吐曲线展示批量环境数增加时并行物理仿真的扩展规律与硬件差异。 出处：C. Daniel Freeman et al.，[Brax — A Differentiable Physics Engine for Large Scale Rigid Body Simulation](https://arxiv.org/abs/2106.13281)（2021），Figure 2。_
</div>

从接口角度，可以把批量计算写成：

$$
\ddot{\mathbf{Q}} = \mathbf{M}(\mathbf{Q})^{-1} \circledast \big(\boldsymbol{\tau} - \mathbf{C}(\mathbf{Q}, \dot{\mathbf{Q}}) \circledast \dot{\mathbf{Q}} - \mathbf{G}(\mathbf{Q})\big)
$$

<div align="center">
<img src="/figures/08-robot-sim/latex/02-simulation-isaac/batched-dynamics-environment-axis.png" alt="批次质量矩阵沿环境轴逐项求解，每个环境只读取同索引的动力学残差" width="86%">

_图 8.2-5：批次维 N 负责并行而不负责混合；第 i 个质量矩阵只求解第 i 个环境的残差，输出继续保留环境轴。本文根据上式绘制。_
</div>

其中，$\circledast$ 表示每个环境独立执行相应的线性代数运算。接触检测、约束求解和内存访问并不等同于一次稠密矩阵乘法，因此吞吐量仍受场景复杂度、接触数量和同步开销影响。

## 非光滑力学：并行接触力求解

机器人足端落地或机械手抓取物体时，会出现接触与摩擦。这类约束使动力学变成非光滑问题。

硬接触的一种理想化写法是线性互补问题（Linear Complementarity Problem, LCP）：地面可以推起足端，却不能用法向力把足端拉向地面。

考虑单个接触点，假设接触面的法向相对加速度为 $a_n$，法向接触力为 $f_n$。物理定律严格要求以下三个条件必须同时成立：

1. **力的单向性**：接触必须只表现为排斥力，即 $f_n \ge 0$。
2. **非穿透约束**：物体的相对加速度在接触时刻不能表现为相互挤压穿透，即 $a_n \ge 0$。
3. **互斥激活条件**：如果两个表面分离（$a_n > 0$），则法向力必须为零（$f_n = 0$）；如果它们承受巨大的受力（$f_n > 0$），那么接触点彼此之间的相对加速度必须为零（$a_n = 0$）。

在数学上，我们将这一逻辑归纳为正交互补性约束：

$$
f_n \cdot a_n = 0
$$

结合以上三点，接触力的求解被规约为如下的标准互补问题格式：

$$
0 \le f_n \perp a_n \ge 0
$$

在多刚体系统中，可将局部法向加速度写成 $a_n = A f_n + b$，其中 $A$ 由质量矩阵与接触雅可比共同决定。PhysX 会用迭代式约束求解器近似处理大批量接触；迭代次数、时间步和接触参数会共同影响速度与误差。这里的互补式用于解释约束逻辑，不等同于 Isaac Gym 内部每一个求解细节。

## 编写高度并行的张量化仿真环境

下面不启动物理引擎，只用 PyTorch 展示 Isaac Gym 张量接口中最重要的批次布局。真实环境还需要从仿真器取得状态张量并把动作张量写回控制接口。

```python
import torch

num_envs = 4096
num_dofs = 12  # 以常见四足机器人的 12 个关节为例
device = "cuda" if torch.cuda.is_available() else "cpu"

# 第 0 维是环境，第 1 维是关节
dof_pos = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)
dof_vel = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)

actions = torch.randn((num_envs, num_dofs), dtype=torch.float32, device=device)

def compute_reward(dof_positions, actions):
    """为每个环境计算一个标量奖励。"""
    target_pos = torch.zeros_like(dof_positions)
    pos_error = torch.sum((dof_positions - target_pos) ** 2, dim=-1)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    return -0.5 * pos_error - 0.01 * action_penalty

batch_rewards = compute_reward(dof_pos, actions)
assert batch_rewards.shape == (num_envs,)
print(batch_rewards.shape, batch_rewards.device)
```

奖励函数没有显式遍历环境，而是沿关节维求和，一次得到形状为 $(N,)$ 的奖励。端到端 GPU 管线还要求状态、动作和策略网络位于同一设备；调试时若频繁调用 `.cpu()` 或 `.numpy()`，仍会重新引入同步与传输开销。

## 小结

- Isaac Gym 的关键是让物理状态和策略张量都驻留在 GPU，并直接交换数据 [[Makoviychuk et al., 2021]](https://arxiv.org/abs/2108.10470)。
- 多环境张量的第 0 维表示环境编号；观测、动作、奖励和终止标记应保持一致的批次顺序。
- 并行并不消除接触求解成本。场景复杂度、接触数量、时间步与同步频率仍会限制吞吐量。
- Isaac Gym、Isaac Sim 与 Isaac Lab 面向不同层次的工作流，迁移代码前需要核对当前接口和运行环境。
