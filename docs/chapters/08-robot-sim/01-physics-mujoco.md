# 8.1 物理仿真与 MuJoCo 基础

> **本章导读**
>
> **讲什么：** 本章讨论机器人策略在哪里学习、怎样从仿真走向现实。我们从刚体与接触的数值模拟开始，扩展到 GPU 并行采样，再用域随机化和特权蒸馏处理虚实差距，并加入想象强化学习、遥操作与人在回路，让失败数据能够回到训练过程。
>
> **为什么不能把仿真器当成现实的复制品：** 在仿真中训练机械臂安全、快速且容易重复，但摩擦系数偏一点、相机晚一帧或物体比模型更软，都可能让现实动作失败。仿真的价值不是保证现实完全相同，而是提供可控制的试验场；要把策略带出去，还必须主动暴露差异、识别差异并在真实失败处修正。
>
> **故事线：** `建立可计算的刚体与接触世界 → 并行生成大量交互 → 随机化未知物理参数 → 用特权信息帮助策略学习 → 在模型想象中继续试错 → 让人类示范与干预修补现实失败`

## 本章总览

<div align="center">

<img src="/figures/08-robot-sim/latex/01-physics-mujoco/chapter-overview.png" alt="第 8 章学习路线：从 MuJoCo 接触物理到具身规划与现实迁移" width="100%">

_第 8 章学习路线：从接触物理与 GPU 并行仿真，逐步解决现实差距、长尾故障与 Sim2Real 规划。_

</div>

先看一个单摆。给定摆杆的长度、质量、当前角度和角速度，仿真器要在一个很短的时间步内算出下一时刻的角度。若摆杆撞到挡板，计算还要同时满足“不能穿透”和“接触力只能把物体推开”等约束。

<div align="center">
<img src="/figures/08-robot-sim/source/01-physics-mujoco/dmcontrol-fig1.png" alt="DeepMind Control Suite 汇集摆、机械臂、行走与游泳任务，展示刚体仿真如何承载连续控制实验。" width="86%">

_图 8.1-1：DeepMind Control Suite 汇集摆、机械臂、行走与游泳任务，展示刚体仿真如何承载连续控制实验。 出处：Yuval Tassa et al.，[DeepMind Control Suite](https://arxiv.org/abs/1801.00690)（2018），Figure 1。_
</div>

机器人控制要求仿真器在速度、稳定性与接触动力学精度之间作出明确取舍；仿真误差是否被闭环放大，取决于系统稳定性，不能一概说成指数增长。Todorov 等人提出 MuJoCo（Multi-Joint dynamics with Contact），采用广义坐标，并把接触动力学写成凸优化问题 [[Todorov et al., 2012]](https://doi.org/10.1109/IROS.2012.6386109)。这正是该论文能够直接支持的技术贡献。

本节从这个最小例子出发，依次建立广义坐标、多刚体动力学、接触约束和时间积分，最后用 Python 读取 MuJoCo 的质量矩阵并推进一步仿真。

## 从牛顿第二定律到广义坐标

我们暂且忘掉复杂的机器人系统，回到高中物理中最经典的质点模型。

### 标量形式的运动学

设想一个质量为 $m$ 的质点在一条直线上运动，其在时刻 $t$ 的位置标量为 $x(t)$。如果质点受到的合外力为 $f(t)$，那么根据牛顿第二定律，力等于质量乘以加速度：

$$f(t) = m \cdot a(t)$$

其中，加速度 $a(t)$ 是速度 $v(t)$ 对时间的导数，而速度 $v(t)$ 又是位置 $x(t)$ 对时间的导数。即：

$$f(t) = m \cdot \frac{d^2 x(t)}{dt^2} = m \cdot \ddot{x}(t)$$

在物理仿真中，我们通常已知系统的当前状态（位置 $x$ 和速度 $v$）以及当前施加的力 $f$，目标是求解出加速度 $\ddot{x}$。这个过程被称为**前向动力学（Forward Dynamics）**。得到加速度后，就可以通过数值积分预测质点在下一个时刻的状态。对于简单质点，前向动力学只需一次标量除法：$\ddot{x}(t) = \frac{f(t)}{m}$。

### 矢量形式与多刚体系统

然而，真实世界中的机器人并非单个在直线上运动的质点，而是由多个具有质量和转动惯量的刚体通过各种关节（如旋转关节、滑动关节）连接而成的铰接系统（Articulated System）。

为了描述这样一个复杂的系统，我们需要引入**广义坐标（Generalized Coordinates）**。我们将系统中所有描述自由度的变量堆叠成一个列向量，记为 $\mathbf{q} \in \mathbb{R}^{n_q}$，其中 $n_q$ 是系统的位置自由度维度。例如，对于一个具有 6 个旋转关节的机械臂，$\mathbf{q}$ 就是一个包含 6 个关节角度的向量。

对应地，我们定义广义速度向量为 $\mathbf{v} \in \mathbb{R}^{n_v}$，以及广义加速度向量为 $\dot{\mathbf{v}} \in \mathbb{R}^{n_v}$。（注意：在许多包含四元数的空间旋转系统中，位置向量 $\mathbf{q}$ 的维度 $n_q$ 可能略大于速度向量 $\mathbf{v}$ 的维度 $n_v$，但在绝大多数全旋铰链系统中，$n_q = n_v$ 且 $\mathbf{v} = \dot{\mathbf{q}}$）。

此时，系统的受力也从单一的标量力升级为**广义力向量（Generalized Forces）** $\boldsymbol{\tau} \in \mathbb{R}^{n_v}$，它代表了施加在每个关节上的驱动力矩（Torque）。

随着维度提升，标量质量 $m$ 被推广为**质量矩阵（Mass Matrix）** $\mathbf{M}(\mathbf{q}) \in \mathbb{R}^{n_v \times n_v}$。它通常是对称正定的，并随姿态 $\mathbf{q}$ 改变。例如，机械臂伸展和收拢时，各关节感受到的等效惯量并不相同。

除了外部施加的控制力矩 $\boldsymbol{\tau}$，由于物体在三维空间中运动，系统内部还会产生离心力（Centrifugal force）、科里奥利力（Coriolis force）以及重力（Gravity）。我们将所有这些非外部控制力统称为**科里奥利与重力偏置项** $\mathbf{c}(\mathbf{q}, \mathbf{v}) \in \mathbb{R}^{n_v}$。

把这些量推广到高维向量空间，就得到机器人学常用的**多刚体动力学方程（Equations of Motion）**：

$$\mathbf{M}(\mathbf{q})\dot{\mathbf{v}} + \mathbf{c}(\mathbf{q}, \mathbf{v}) = \boldsymbol{\tau}$$

忽略接触约束时，前向动力学可以写为：

$$\dot{\mathbf{v}} = \mathbf{M}(\mathbf{q})^{-1} \left( \boldsymbol{\tau} - \mathbf{c}(\mathbf{q}, \mathbf{v}) \right)$$

<div align="center">
<img src="/figures/08-robot-sim/latex/01-physics-mujoco/mass-matrix-coupled-solve.png" alt="质量矩阵的非对角项把两个净力矩分量耦合到两个关节加速度分量" width="86%">

_图 8.1-2：质量矩阵存在非对角项时，前向动力学必须整体求解耦合方程；单个净力矩分量不再只对应一个关节加速度。_
</div>

这个式子用于说明变量关系，不表示实现时真的显式计算 $\mathbf{M}^{-1}$。MuJoCo 使用复合刚体算法构造稀疏质量矩阵，并对其做保持稀疏性的 $L^\top D L$ 分解；求解 $\mathbf{M}^{-1}\mathbf{x}$ 时使用回代 [[MuJoCo Computation]](https://mujoco.readthedocs.io/en/latest/computation.html)。

<div align="center">
<img src="/figures/08-robot-sim/source/01-physics-mujoco/d4rl-fig1.png" alt="D4RL 的 MuJoCo locomotion 数据集把同一物理系统中的不同数据质量组织成可比较的离线控制基准。" width="86%">

_图 8.1-3：D4RL 的 MuJoCo locomotion 数据集把同一物理系统中的不同数据质量组织成可比较的离线控制基准。 出处：Justin Fu et al.，[D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://arxiv.org/abs/2004.07219)（2020），Figure 1。_
</div>

## 接触动力学与线性互补问题

如果在真空中模拟一个悬浮的机械臂，上式已经描述了主要动力学。但在机器人行走、机械臂抓取杯子等任务中，物体还会发生碰撞和接触。接触建模通常也是物理仿真中计算较多、调参较敏感的环节。

<div align="center">
<img src="/figures/08-robot-sim/source/01-physics-mujoco/gym-fig1.png" alt="OpenAI Gym 的机器人环境界面把物理状态、动作和可视化包装成统一交互循环。" width="86%">

_图 8.1-4：OpenAI Gym 的机器人环境界面把物理状态、动作和可视化包装成统一交互循环。 出处：Greg Brockman et al.，[OpenAI Gym](https://arxiv.org/abs/1606.01540)（2016），Figure 1。_
</div>

当两个刚体发生接触时，它们之间会产生接触力 $\mathbf{f}_c \in \mathbb{R}^{n_c}$，其中 $n_c$ 取决于当前接触约束的数量。接触力要先映射到关节空间，才能进入系统动力学方程。为此引入接触雅可比矩阵 $\mathbf{J}_c(\mathbf{q}) \in \mathbb{R}^{n_c \times n_v}$。根据虚功关系，接触坐标中的力 $\mathbf{f}_c$ 对应关节空间中的广义力 $\mathbf{J}_c(\mathbf{q})^\top \mathbf{f}_c$。

因此，包含物理接触后的系统总动力学方程变为：

$$\mathbf{M}(\mathbf{q})\dot{\mathbf{v}} + \mathbf{c}(\mathbf{q}, \mathbf{v}) = \boldsymbol{\tau} + \mathbf{J}_c(\mathbf{q})^\top \mathbf{f}_c$$

现在，加速度 $\dot{\mathbf{v}}$ 与接触力 $\mathbf{f}_c$ 都是未知量。硬接触的理想化形式常写成互补条件：法向力 $f_n$ 不能为负，分离加速度 $a_n$ 不能指向穿透方向，并且两者不能同时为正。

$$
f_n \ge 0, \qquad a_n \ge 0, \qquad f_n a_n = 0
$$

第三个条件的含义很具体：物体正在分离时，接触力应为零；接触力为正时，接触点不能继续沿法线方向互相挤入。

MuJoCo 并不是简单地把所有接触都当作严格的硬约束 LCP。它使用可调软硬程度的约束模型，并把约束力定义为凸优化问题的解：金字塔摩擦锥对应二次规划，椭圆摩擦锥对应锥规划 [[MuJoCo Computation]](https://mujoco.readthedocs.io/en/latest/computation.html)。软约束允许模型表达有限刚度，也让数值求解更稳定；代价是接触行为仍取决于时间步、求解器容差和材料参数。

因此，仿真器给出的不是“真实接触力的唯一答案”，而是在指定模型与数值设置下的一致近似。将策略迁移到真实机器人时，接触参数仍需要校准或随机化。

## 时间离散化与数值积分

前向动力学给出当前时刻的连续时间加速度 $\dot{\mathbf{v}}_t$。计算机还需要选择时间步长 $\Delta t$，再用数值积分把速度和位置推进到下一时刻。时间步越小通常越准确，但每秒需要的求解次数也越多。

<div align="center">
<img src="/figures/08-robot-sim/source/01-physics-mujoco/brax-fig5.png" alt="Brax 对接触系统的能量与动量误差进行比较，显示积分与接触实现会直接改变长期数值行为。" width="86%">

_图 8.1-5：Brax 对接触系统的能量与动量误差进行比较，显示积分与接触实现会直接改变长期数值行为。 出处：C. Daniel Freeman et al.，[Brax — A Differentiable Physics Engine for Large Scale Rigid Body Simulation](https://arxiv.org/abs/2106.13281)（2021），Figure 5。_
</div>

下面先看半隐式欧拉积分（Semi-implicit Euler）。它先更新速度，再用新速度更新位置：

首先，我们在极短的 $\Delta t$ 窗口内对系统的广义加速度进行线性积分：

$$\mathbf{v}_{t+1} = \mathbf{v}_t + \dot{\mathbf{v}}_t \cdot \Delta t$$

随后用 $\mathbf{v}_{t+1}$ 更新广义位置：

$$\mathbf{q}_{t+1} = \mathbf{q}_t + \mathbf{v}_{t+1} \cdot \Delta t$$

相较于用旧速度更新位置的显式欧拉法，这种顺序在许多机械系统中更稳定，但并不保证所有带阻尼、接触和控制输入的系统都守恒能量。MuJoCo 还提供隐式类积分器与四阶 Runge–Kutta；具体选择应由系统刚性、精度要求和计算预算决定 [[MuJoCo Computation]](https://mujoco.readthedocs.io/en/latest/computation.html#numerical-integration)。

## MuJoCo 在深度学习框架中的使用

MuJoCo 使用 MJCF 描述刚体、关节、执行器和传感器。下面的 MJCF 只包含一个胶囊体和一个铰链，因此质量矩阵退化为 $1\times1$。代码先读取这一矩阵，再调用 `mj_step` 推进一步。

```python
import mujoco
import numpy as np
import torch

# 一个带铰链的胶囊体单摆
xml_string = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <light pos="0 1 1" dir="0 -1 -1" diffuse="1 1 1"/>
    <body name="pendulum" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0"/>
      <geom type="capsule" size="0.05 0.5" pos="0 0 -0.5" mass="1.0" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# mjModel 保存拓扑与参数，mjData 保存当前状态与中间结果
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# 初始角度约为 57 度，初始角速度为 0
data.qpos[0] = 1.0  # 约抬起 57 度
data.qvel[0] = 0.0

print(f"初始物理状态 - 位置(q): {data.qpos[0]:.4f}, 速度(v): {data.qvel[0]:.4f}")

# 更新当前状态对应的动力学量，但不推进时间
mujoco.mj_forward(model, data)

# 将稀疏存储的 qM 展开为便于检查的稠密质量矩阵
nv = model.nv
M_dense = np.zeros((nv, nv))
mujoco.mj_fullM(model, M_dense, data.qM)
print(f"当前姿态下的质量矩阵 M(q):\n{M_dense}")

# qfrc_bias 包含科里奥利、离心和重力偏置项
bias_force = data.qfrc_bias.copy()
print(f"偏置力 c(q, v): {bias_force}")

# 求解前向动力学与约束，并按所选积分器推进一个时间步
mujoco.mj_step(model, data)

# 先复制数组，避免后续仿真步原地改写张量所引用的数据
next_q_tensor = torch.tensor(data.qpos.copy(), dtype=torch.float32)
next_v_tensor = torch.tensor(data.qvel.copy(), dtype=torch.float32)

print(f"单步数值积分后的系统状态 (转为 PyTorch Tensor) - 位置: {next_q_tensor.item():.4f}, 速度: {next_v_tensor.item():.4f}")
```

这段代码对应三个层次：`qpos/qvel` 是状态，`qM/qfrc_bias` 是当前状态导出的动力学量，`mj_step` 则完成受力、约束与积分。神经网络通常读取状态或传感器观测，而不是直接读取全部内部量。

## 小结

- 广义坐标下的动力学由质量矩阵、偏置力、控制力和约束力共同决定。
- MuJoCo 把约束力写成凸优化问题，并允许通过参数调节接触的软硬程度。
- 数值积分把连续时间加速度变成离散状态序列；时间步和积分器都会影响稳定性与精度。
- `mjModel` 保存模型，`mjData` 保存状态与中间量。读取 `qM`、`qfrc_bias` 和推进 `mj_step` 可以把公式对应到程序接口。
