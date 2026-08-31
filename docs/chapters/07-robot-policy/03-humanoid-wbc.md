# 人形机器人与全身控制

在本书的前几章中，我们探讨了强化学习和基于视觉的世界模型。然而，当这些高级的智能决策最终要转化为物理世界中机器人的动作时，我们必须面对一个残酷的现实：真实世界的机器人——特别是拥有数十个自由度的人形机器人（Humanoid）——受制于严格的物理定律。在这一章，我们将深入探讨传统机器人学中最为璀璨的明珠之一：全身控制（Whole-Body Control，简称 WBC）。

## 学术溯源与时代背景

在早期的工业机器人时代，机器人大多固定在地面上，且通常通过独立关节控制（Independent Joint Control）来实现运动。这种策略下，每个电机只负责跟踪自己关节的目标角度，完全忽略了机器人各个连杆之间的动力学耦合。

随着机器人技术的发展，研究人员开始让机器人离开固定的基座，走向现实世界。人形机器人的出现带来了两个致命的挑战：第一，高维度与冗余性，一个典型的人形机器人拥有数十个关节；第二，浮动基座与欠驱动，人形机器人的躯干（Base）悬浮在三维空间中，没有任何电机直接控制躯干的运动，躯干的移动完全依赖于脚部与地面的接触力。

Khatib 的操作空间公式（Operational Space Formulation, OSC）系统化了直接在任务空间描述与控制机械臂运动的方法 [[Khatib, 1987]](https://doi.org/10.1177/027836498700600103)。Sentis 等人把任务层级与接触约束纳入全身控制框架 [[Sentis et al., 2007]](https://doi.org/10.1109/ROBOT.2007.363998)。优化式全身控制可以把动力学、摩擦锥和执行器限制写入二次规划；Kuindersma 等人展示了这一路线在 Atlas 机器人上的应用 [[Kuindersma et al., 2016]](https://doi.org/10.1007/s10514-015-9479-3)，但单篇论文不足以证明它是整个业界的“绝对主流”。

在本章中，我们将从最基础的高中物理出发，一步步严谨地推导出人形机器人全身控制的数学本质。

## 浮动基座动力学：从牛顿第二定律到多刚体系统

在高中物理中，我们学过描述质点平移运动的牛顿第二定律：

$$F = m a$$

对于一个绕固定轴旋转的刚体，它的旋转动力学方程为：

$$\tau = I \alpha$$

其中，$\tau$ 是力矩，$I$ 是转动惯量，$\alpha$ 是角加速度。

然而，人形机器人是由多个刚体（连杆）通过关节连接而成的复杂系统。我们需要将这些简单的一维标量方程扩展为高维的矩阵形式。

首先，我们定义机器人的广义坐标（Generalized Coordinates） $\mathbf{q}$。对于固定在地面上的机械臂，$\mathbf{q}$ 仅仅包含了各个关节的角度。但对于人形机器人，它的基座（通常是骨盆区）在空间中是自由浮动的。因此，我们将 $\mathbf{q}$ 拆分为两部分：

$$\mathbf{q} = \begin{bmatrix} \mathbf{q}_{base} \\ \mathbf{q}_{joint} \end{bmatrix} \in \mathbb{R}^{n_b + n_j}$$

其中，$\mathbf{q}_{base}$ 描述了浮动基座在三维空间中的位置和姿态（通常包含 3 个位置自由度和 3 个姿态自由度，即 $n_b=6$），而 $\mathbf{q}_{joint}$ 描述了 $n_j$ 个电机的关节角度。

利用分析力学中的欧拉-拉格朗日方程（Euler-Lagrange Equations），我们可以推导出人形机器人的刚体动力学方程：

$$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) = \mathbf{S}^T \boldsymbol{\tau} + \mathbf{J}_c^T \mathbf{F}_c$$

这是机器人学中最核心的公式之一。结合这两个公式，让我们对该公式中的每一个物理量进行极其严谨的维度和物理意义拆解：

- $\mathbf{M}(\mathbf{q}) \in \mathbb{R}^{(n_b+n_j) \times (n_b+n_j)}$：质量惯性矩阵（Mass-Inertia Matrix）。它是质量 $m$ 与转动惯量 $I$ 的高维矩阵推广。这个矩阵是对称正定的，且会随着机器人的姿态 $\mathbf{q}$ 实时变化。
- $\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) \in \mathbb{R}^{(n_b+n_j) \times (n_b+n_j)}$：科里奥利与离心力矩阵（Coriolis and Centrifugal Matrix）。它反映了多关节高速运动时产生的非线性耦合力。
- $\mathbf{G}(\mathbf{q}) \in \mathbb{R}^{n_b+n_j}$：重力向量（Gravity Vector）。
- $\boldsymbol{\tau} \in \mathbb{R}^{n_j}$：关节电机输出的力矩（控制系统唯一能直接控制的量）。
- $\mathbf{S} = [\mathbf{0}_{n_j \times n_b}, \mathbf{I}_{n_j \times n_j}] \in \mathbb{R}^{n_j \times (n_b+n_j)}$：选择矩阵（Selection Matrix）。因为躯干没有电机，前 6 个自由度对应的控制输入严格为 0。$\mathbf{S}^T \boldsymbol{\tau}$ 将 $n_j$ 维的电机力矩映射到了完整的广义坐标空间。
- $\mathbf{F}_c \in \mathbb{R}^k$：脚底与地面的接触力（Contact Force）。
- $\mathbf{J}_c \in \mathbb{R}^{k \times (n_b+n_j)}$：接触点对应的雅可比矩阵（Jacobian Matrix）。$\mathbf{J}_c^T \mathbf{F}_c$ 即为接触力对各个自由度产生的等效力矩。

如果我们把该公式的前 6 行单独提取出来，就会发现等式右侧没有 $\boldsymbol{\tau}$ 参与。这意味着人形机器人的质心和躯干运动完全不能由自身电机直接驱动，只能通过脚蹬地产生的环境接触力 $\mathbf{F}_c$ 来间接控制。这确立了人形机器人作为一种高阶欠驱动系统的数学本质。

## 雅可比矩阵：连接关节空间与任务空间的桥梁

为了控制机器人的手到达目标位置，我们需要建立关节转动与手部移动之间的严格数学关系。

假设有一个最简单的单连杆系统，长度为 $l$，单关节角度为 $\theta$。连杆末端在垂直方向的高度为：

$$y = l \sin(\theta)$$

对时间求导，我们得到速度的正向映射关系：

$$\dot{y} = l \cos(\theta) \dot{\theta}$$

这里的 $l \cos(\theta)$ 就是连接关节速度 $\dot{\theta}$ 和末端速度 $\dot{y}$ 的比例算子。

对于高维机器人，设末端执行器在三维世界坐标系（我们称之为任务空间，Task Space）的位置和姿态为 $\mathbf{x} \in \mathbb{R}^m$。存在一个复杂的非线性正运动学映射 $\mathbf{x} = f(\mathbf{q})$。对其利用多元微积分的链式法则求时间导数，我们得到：

$$\dot{\mathbf{x}} = \frac{\partial f(\mathbf{q})}{\partial \mathbf{q}} \dot{\mathbf{q}} = \mathbf{J}(\mathbf{q}) \dot{\mathbf{q}}$$

这里，$\mathbf{J}(\mathbf{q}) \in \mathbb{R}^{m \times (n_b+n_j)}$ 即为大名鼎鼎的雅可比矩阵（Jacobian Matrix）。它是一个将高维关节空间的微小变化映射到低维任务空间变化的线性算子。

进一步，对速度该公式求时间微商，我们可以得到加速度层面的映射方程：

$$\ddot{\mathbf{x}} = \mathbf{J}(\mathbf{q}) \ddot{\mathbf{q}} + \dot{\mathbf{J}}(\mathbf{q}) \dot{\mathbf{q}}$$

## 任务空间控制（Operational Space Control）

现在，我们的目标是让末端执行器追踪一个期望的空间轨迹 $\mathbf{x}_{des}$。在任务空间中，我们可以设计一个比例-微分（PD）控制器，计算产生所需运动的虚拟追踪力 $\mathbf{F}_{task}$：

$$\mathbf{F}_{task} = \mathbf{K}_p (\mathbf{x}_{des} - \mathbf{x}) + \mathbf{K}_d (\dot{\mathbf{x}}_{des} - \dot{\mathbf{x}})$$

然而在真实的机器人系统中，不存在物理意义上直接拉动末端执行器的力，我们仅能向各个关节的电机发送扭矩指令 $\boldsymbol{\tau}$。如何将 $\mathbf{F}_{task}$ 严格映射为 $\boldsymbol{\tau}$？

根据分析力学中的虚功原理（Principle of Virtual Work），系统在任务空间做出的虚功必须等于在关节空间做出的虚功：

$$\mathbf{F}_{task}^T \delta \mathbf{x} = \boldsymbol{\tau}_{task}^T \delta \mathbf{q}$$

在无穷小位移下，由于 $\delta \mathbf{x} = \mathbf{J} \delta \mathbf{q}$，代入上式即得：

$$\mathbf{F}_{task}^T \mathbf{J} \delta \mathbf{q} = \boldsymbol{\tau}_{task}^T \delta \mathbf{q}$$

由于该公式对于任意合法位移 $\delta \mathbf{q}$ 均必须成立，我们提取等式两边的系数，可以得到静力学力矩映射的核心公式：

$$\boldsymbol{\tau}_{task} = \mathbf{J}^T \mathbf{F}_{task}$$

通过该公式，我们将抽象的三维空间追踪目标，转换为了具体每个电机需要输出的物理力矩。

## 零空间投影：多任务分层控制 (Null-Space Projection)

人形机器人的自由度往往远远大于完成单个任务所需的最小自由度，例如，控制一只手的位置和姿态最多需要 6 个自由度，而机器人全身可能有超过 30 个自由度。这种物理学上的冗余性（Redundancy）赋予了我们执行多任务的能力。

> 想像你在端着一杯装满水的杯子（主任务，要求保持杯子绝对水平和稳定），同时你需要用同一只手的手肘去关门（次任务）。因为你的手臂有多个关节，只要手肘的运动轨迹不改变手腕的位置和姿态，这两个任务就能完美共存。在数学上，这就是将次任务的控制力投影到主任务的“零空间”中——在这个空间里发生的所有动作，对主任务的影响严格为零。

假设我们有主任务控制器计算出的力矩 $\boldsymbol{\tau}_1$，以及次任务（例如维持整体身体姿态优雅）计算出的力矩 $\boldsymbol{\tau}_2$。如果盲目地将其直接相加 $\boldsymbol{\tau} = \boldsymbol{\tau}_1 + \boldsymbol{\tau}_2$，那么 $\boldsymbol{\tau}_2$ 势必会改变主任务的执行状态，导致水杯倾覆。

在线性代数中，矩阵 $\mathbf{A}$ 的零空间（Null Space）是指所有满足 $\mathbf{A}\mathbf{z} = \mathbf{0}$ 的向量 $\mathbf{z}$ 的集合。对应到全身控制中，我们需要构造一个投影矩阵 $\mathbf{N}$，使得次级任务产生的动力学效应在主任务的雅可比矩阵映射下严格为零。

结合动力学一致性原则 [[Khatib, 1987]](https://doi.org/10.1177/027836498700600103)，投影矩阵 $\mathbf{N}$ 的推导依赖于机器人自身的惯性矩阵。我们首先定义动力学一致的伪逆（Dynamically Consistent Pseudo-inverse）矩阵 $\overline{\mathbf{J}}$：

$$\overline{\mathbf{J}} = \mathbf{M}^{-1} \mathbf{J}^T (\mathbf{J} \mathbf{M}^{-1} \mathbf{J}^T)^{-1}$$

此时，主任务在关节空间维度的干涉算子即为 $\overline{\mathbf{J}} \mathbf{J}$。从整个空间中剔除该干涉算子，我们即可得到次任务的零空间投影矩阵 $\mathbf{N}$：

$$\mathbf{N} = \mathbf{I} - \overline{\mathbf{J}} \mathbf{J}$$

结合这两个公式，最终的多任务全身控制力矩分配法则为：

$$\boldsymbol{\tau} = \mathbf{J}_1^T \mathbf{F}_1 + \mathbf{N}^T \boldsymbol{\tau}_2$$

在这里，$\mathbf{N}^T \boldsymbol{\tau}_2$ 项确保了无论次任务控制器需要产生多大的力矩 $\boldsymbol{\tau}_2$，它在空间中激发的动力学加速度，投射到主任务的工作空间内一定是 0。

## 基于二次规划的全身优化控制 (QP-based WBC)

纯矩阵代数的零空间投影虽然在数学上无比优美，但在真实物理世界中遇到了不可逾越的鸿沟：它无法处理任何不等式约束（Inequality Constraints）。
真实机器人有着苛刻的物理极值限制：
第一，电机扭矩存在刚性的上下限：$\boldsymbol{\tau}_{min} \le \boldsymbol{\tau} \le \boldsymbol{\tau}_{max}$。
第二，机器人的脚底不能打滑，接触力必须严格落在三维库仑摩擦锥（Friction Cone）内部：$\mu F_z \ge \sqrt{F_x^2 + F_y^2}$。

为了显式处理这些不等式，许多现代 WBC 方法把控制写成在线二次规划（Quadratic Programming, QP）[[Kuindersma et al., 2016]](https://doi.org/10.1007/s10514-015-9479-3)。这并不意味着解析投影已被“彻底抛弃”；不同系统仍会组合层级投影、逆动力学与优化约束。

在每一个高频控制周期（例如 1000 Hz），我们需要实时求解以下的大型 QP 优化问题：

$$
\begin{aligned}
\min_{\ddot{\mathbf{q}}, \mathbf{F}_c, \boldsymbol{\tau}} \quad & \sum_{i} \mathbf{w}_i || \ddot{\mathbf{x}}_i - \ddot{\mathbf{x}}_{i, des} ||_2^2 + \mathbf{w}_{\tau} || \boldsymbol{\tau} ||_2^2 \\
\text{subject to} \quad & \mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{G} = \mathbf{S}^T \boldsymbol{\tau} + \mathbf{J}_c^T \mathbf{F}_c \quad & (\text{动力学严格约束}) \\
& \ddot{\mathbf{x}}_i = \mathbf{J}_i \ddot{\mathbf{q}} + \dot{\mathbf{J}}_i \dot{\mathbf{q}} \quad & (\text{运动学映射约束}) \\
& \mathbf{F}_c \in \mathcal{K} \quad & (\text{摩擦力锥约束}) \\
& \boldsymbol{\tau}_{min} \le \boldsymbol{\tau} \le \boldsymbol{\tau}_{max} \quad & (\text{扭矩极值约束})
\end{aligned}
$$

在相关章节节的框架下，通过该公式，所有的多任务竞争不再依赖生硬的代数投影，而是通过目标函数中的惩罚权重 $\mathbf{w}_i$ 进行软性博弈。现代 QP 求解器（如 OSQP）可以在亚毫秒级时间内计算出全局最优解，使得机器人能在极其恶劣的地形约束下维持动态平衡。

## 代码实现：构建任务空间投影引擎

为了将抽象的矩阵方程转化为具体的算法实现，我们将使用 PyTorch 编写一个简化版的多任务零空间投影计算引擎。PyTorch 原生的自动求导机制和高维张量运算非常适合还原这些数学公式的本来面貌。

(**我们定义一个多任务投影模块，给定机器人当前的动力学参数和任务雅可比，计算包含主次任务的控制力矩指令。**)

```python
import torch

class NullSpaceController:
    def __init__(self, num_dof):
        """
        初始化零空间多任务控制器。
        :param num_dof: 机器人的总自由度数 (n_b + n_j)
        """
        self.n = num_dof
        # 为防止张量求逆时的数值奇异性，引入极小的对角正则化项
        self.eps = 1e-5

    def compute_torque(self, M, J1, F1, J2, F2):
        """
        计算考虑了零空间投影的双任务力矩分配。
        :param M: [n, n] 质量惯性张量
        :param J1: [k1, n] 主任务雅可比张量
        :param F1: [k1] 主任务空间虚拟推力
        :param J2: [k2, n] 次任务雅可比张量
        :param F2: [k2] 次任务空间虚拟推力
        :return: [n] 下发到所有关节的最终力矩张量
        """
        # 1. 计算质量矩阵的逆 (M^-1)
        M_inv = torch.linalg.inv(M)

        # 2. 计算主任务的操作空间惯性矩阵的逆: Lambda_1^-1 = J1 * M^-1 * J1^T
        lambda1_inv = J1 @ M_inv @ J1.T

        # 为保证浮点数计算数值稳定，加上微小阻尼后求逆得到 Lambda_1
        lambda1 = torch.linalg.inv(lambda1_inv + self.eps * torch.eye(J1.shape[0]))

        # 3. 计算动力学一致的伪逆 (J_bar_1 = M^-1 * J1^T * Lambda_1)
        # 该张量映射能够考虑系统中各刚体的质量分布，而不仅是几何学投影
        J1_bar = M_inv @ J1.T @ lambda1

        # 4. 计算零空间投影矩阵 (N_1 = I - J_bar_1 * J1)
        I = torch.eye(self.n)
        N1 = I - J1_bar @ J1

        # 5. 主任务力矩静力学映射
        tau_1 = J1.T @ F1

        # 6. 次任务力矩计算并进行零空间投影压制
        # 将次任务产生的作用力首先转化为不受约束的理论力矩
        tau_2_raw = J2.T @ F2
        # 利用 N1 张量进行投影变换，将其严格压制到主任务的零空间内
        tau_2_projected = N1.T @ tau_2_raw

        # 7. 多任务复合输出
        tau_total = tau_1 + tau_2_projected
        return tau_total

# --- 模拟执行与测试 ---
n_dof = 7
controller = NullSpaceController(num_dof=n_dof)

# 伪造当前状态的系统参数 (使用正定矩阵模拟质量矩阵 M)
A = torch.randn(n_dof, n_dof)
M = A @ A.T + torch.eye(n_dof)

# 主任务参数构建（例如：维持末端3D空间坐标）
J1 = torch.randn(3, n_dof)
F1 = torch.tensor([10.0, -5.0, 2.0]) # pd控制器算出的空间推力

# 次任务参数构建（例如：维持肘部的内部2D姿态）
J2 = torch.randn(2, n_dof)
F2 = torch.tensor([1.0, 1.0])

# 计算最终下发给电机的联合力矩
tau = controller.compute_torque(M, J1, F1, J2, F2)
print("下发给各个关节的最终力矩指令: \n", tau)
```

在这个实现中，我们严格遵循了这两个公式。通过张量维度的逐步映射，可以清晰地看到三维世界任务力 `F1` 是如何通过极其严密的代数运算散布到所有的关节力矩指令 `tau` 中的，并确保次任务 `F2` 的引入不会在物理动力学层面对主任务造成毁灭性干扰。

## 小结

- 人形机器人的控制本质上是一个处理**浮动基座系统**（高阶欠驱动）和多自由度（高度冗余）的复杂刚体动力学求解问题。
- **雅可比矩阵**作为最核心的线性算子，将笛卡尔任务空间的速度和受力，严谨地桥接到了高维的关节角度空间。
- 零空间投影提供了一种基于纯代数几何的多任务分配机制，使次优先级任务产生的力矩投影能够严格落在主任务的工作空间之外。
- 现代**全身控制（WBC）**框架彻底倒向了二次规划（QP），对动力学方程、接触摩擦力锥、关节扭矩极限等物理硬边界进行了最底层的约束与求解，这也是当前世界最顶级双足机器人保持稳健行走的数学基石。
