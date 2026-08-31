# 人形机器人与全身控制

设人形机器人需要在双脚不打滑的同时保持躯干稳定，并把右手移动到目标位置。这三个要求共享同一组关节和接触力，不能由三个互不相干的控制器分别完成。**全身控制**（Whole-Body Control, WBC）研究的正是如何在动力学与接触约束下协调这些任务。

<div align="center">

<img src="/figures/07-robot-policy/source/03-humanoid-wbc/kuind-fig1.png" alt="Atlas 在障碍、泥地和坡面上行走，展示全身约束必须同时满足。" width="86%">

_图 7.3-1：Atlas 在障碍、泥地和坡面上行走，展示全身约束必须同时满足。 出处：[Optimization-based Locomotion Planning, Estimation, and Control Design for the Atlas Humanoid Robot，Scott Kuindersma et al.，2014](https://arxiv.org/abs/1311.1839)。_

</div>

## 学术溯源与时代背景

在早期的工业机器人时代，机器人大多固定在地面上，且通常通过独立关节控制（Independent Joint Control）来实现运动。这种策略下，每个电机只负责跟踪自己关节的目标角度，完全忽略了机器人各个连杆之间的动力学耦合。

当机器人离开固定基座后，控制问题出现两个直接变化。第一，系统有数十个关节，同一任务通常存在多组可行动作。第二，浮动基座没有直接对应的驱动器；躯干运动必须通过关节运动和环境接触力间接产生。

<div align="center">

<img src="/figures/07-robot-policy/source/03-humanoid-wbc/sentis-fig9.png" alt="自由浮动人形的支撑接触与反作用力说明基座运动由接触间接产生。" width="86%">

_图 7.3-2：自由浮动人形的支撑接触与反作用力说明基座运动由接触间接产生。 出处：[A Whole-Body Control Framework for Humanoids Operating in Human Environments，Luis Sentis; Oussama Khatib，2006](https://doi.org/10.1109/ROBOT.2006.1642100)。_

</div>

Khatib 的操作空间公式（Operational Space Formulation, OSC）系统化了直接在任务空间描述与控制机械臂运动的方法 [[Khatib, 1987]](https://doi.org/10.1177/027836498700600103)。Sentis 等人把任务层级与接触约束纳入全身控制框架 [[Sentis et al., 2007]](https://doi.org/10.1109/ROBOT.2007.363998)。优化式全身控制可以把动力学、摩擦锥和执行器限制写入二次规划；Kuindersma 等人展示了这一路线在 Atlas 机器人上的应用 [[Kuindersma et al., 2016]](https://doi.org/10.1007/s10514-015-9479-3)，但单篇论文不足以证明它是整个业界的“绝对主流”。

下面从牛顿第二定律出发，把质点方程逐步扩展到多刚体系统。

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

下面逐项检查这个方程中的物理量与维度：

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

考虑一个 7 自由度机械臂：主任务固定杯子的位姿，次任务调整肘部位置。若某个关节速度变化满足 $\mathbf{J}_1\dot{\mathbf{q}}=0$，它在主任务的一阶速度近似下不会移动杯子。这样的方向组成主任务雅可比矩阵的零空间。

<div align="center">

<img src="/figures/07-robot-policy/source/03-humanoid-wbc/sentis-fig3.png" alt="任务、约束和姿态原语按优先级投影，直观呈现全身控制层级。" width="86%">

_图 7.3-3：任务、约束和姿态原语按优先级投影，直观呈现全身控制层级。 出处：[A Whole-Body Control Framework for Humanoids Operating in Human Environments，Luis Sentis; Oussama Khatib，2006](https://doi.org/10.1109/ROBOT.2006.1642100)。_

</div>

假设主任务给出力矩 $\boldsymbol{\tau}_1$，次任务（例如维持躯干姿态）给出力矩 $\boldsymbol{\tau}_2$。若直接相加 $\boldsymbol{\tau}=\boldsymbol{\tau}_1+\boldsymbol{\tau}_2$，次任务可能改变主任务方向，增加水杯倾覆风险。

在线性代数中，矩阵 $\mathbf{A}$ 的零空间（Null Space）是指所有满足 $\mathbf{A}\mathbf{z} = \mathbf{0}$ 的向量 $\mathbf{z}$ 的集合。对应到全身控制中，我们需要构造一个投影矩阵 $\mathbf{N}$，使得次级任务产生的动力学效应在主任务的雅可比矩阵映射下严格为零。

结合动力学一致性原则 [[Khatib, 1987]](https://doi.org/10.1177/027836498700600103)，投影矩阵 $\mathbf{N}$ 的推导依赖于机器人自身的惯性矩阵。我们首先定义动力学一致的伪逆（Dynamically Consistent Pseudo-inverse）矩阵 $\overline{\mathbf{J}}$：

$$\overline{\mathbf{J}} = \mathbf{M}^{-1} \mathbf{J}^T (\mathbf{J} \mathbf{M}^{-1} \mathbf{J}^T)^{-1}$$

此时，主任务在关节空间维度的干涉算子即为 $\overline{\mathbf{J}} \mathbf{J}$。从整个空间中剔除该干涉算子，我们即可得到次任务的零空间投影矩阵 $\mathbf{N}$：

$$\mathbf{N} = \mathbf{I} - \overline{\mathbf{J}} \mathbf{J}$$

结合这两个公式，最终的多任务全身控制力矩分配法则为：

$$ \boldsymbol{\tau} = \mathbf{J}_1^T \mathbf{F}_1 + \mathbf{N}^T \boldsymbol{\tau}_2
$$

<div align="center">

<img src="/figures/07-robot-policy/latex/03-humanoid-wbc/nullspace-secondary-torque.png" alt="主任务力矩与经零空间投影的次任务力矩合流" width="86%">

_图 7.3-4：次任务先经动态一致零空间投影再与主任务合流，因此它在主任务加速度映射中的贡献为零。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

在模型准确、主任务雅可比满足相应秩条件且执行器未饱和时，$\mathbf{N}^T\boldsymbol{\tau}_2$ 在一阶动力学意义下不改变主任务加速度。接近奇异位形时通常要改用伪逆或阻尼逆，实际系统还会受限幅与建模误差影响。

## 基于二次规划的全身优化控制 (QP-based WBC)

零空间投影适合表达任务优先级，但单独使用时不便同时处理多组不等式约束（Inequality Constraints）。真实机器人还必须满足：
第一，电机扭矩存在刚性的上下限：$\boldsymbol{\tau}_{min} \le \boldsymbol{\tau} \le \boldsymbol{\tau}_{max}$。
第二，机器人的脚底不能打滑，接触力必须严格落在三维库仑摩擦锥（Friction Cone）内部：$\mu F_z \ge \sqrt{F_x^2 + F_y^2}$。

为了显式处理这些不等式，许多现代 WBC 方法把控制写成在线二次规划（Quadratic Programming, QP）[[Kuindersma et al., 2016]](https://doi.org/10.1007/s10514-015-9479-3)。这并不意味着解析投影已被“彻底抛弃”；不同系统仍会组合层级投影、逆动力学与优化约束。

<div align="center">

<img src="/figures/07-robot-policy/source/03-humanoid-wbc/kuind-fig3.png" alt="多面体摩擦锥近似把接触可行域转化为 QP 可处理的线性约束。" width="86%">

_图 7.3-5：多面体摩擦锥近似把接触可行域转化为 QP 可处理的线性约束。 出处：[Optimization-based Locomotion Planning, Estimation, and Control Design for the Atlas Humanoid Robot，Scott Kuindersma et al.，2014](https://arxiv.org/abs/1311.1839)。_

</div>

在每个控制周期中，可以求解下面这种 QP。实际频率与规模取决于机器人、求解器和硬件，不能从公式本身推出固定的 1000 Hz：

$$

\begin{aligned}
\min_{\ddot{\mathbf{q}}, \mathbf{F}_c, \boldsymbol{\tau}} \quad & \sum_{i} w_i \| \ddot{\mathbf{x}}_i - \ddot{\mathbf{x}}_{i, des} \|_2^2 + w_{\tau} \| \boldsymbol{\tau} \|_2^2 \\
\text{subject to} \quad & \mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{G} = \mathbf{S}^T \boldsymbol{\tau} + \mathbf{J}_c^T \mathbf{F}_c \quad & (\text{动力学严格约束}) \\
& \ddot{\mathbf{x}}_i = \mathbf{J}_i \ddot{\mathbf{q}} + \dot{\mathbf{J}}_i \dot{\mathbf{q}} \quad & (\text{运动学映射约束}) \\
& \mathbf{F}*c \in \mathcal{K} \quad & (\text{摩擦力锥约束}) \\
& \boldsymbol{\tau}*{min} \le \boldsymbol{\tau} \le \boldsymbol{\tau}_{max} \quad & (\text{扭矩极值约束})
\end{aligned}

$$

目标中的 $w_i$ 表示任务权重，等式与不等式则表示必须满足的物理边界。当目标为凸二次函数、约束为线性形式时，QP 可以求得全局最优解；实际求解时间仍取决于问题规模、稀疏结构和数值条件。

## 代码实现：构建任务空间投影引擎

下面用 PyTorch 实现一个固定基座机械臂的双任务零空间投影。它只演示 $\overline{\mathbf J}$ 与 $\mathbf N$ 的矩阵计算，不包含浮动基座、接触力或 QP，因而不是完整的人形 WBC。

输入质量矩阵和两个任务雅可比后，模块返回固定基座系统的广义关节力矩。

```python
import torch

class NullSpaceController:
    def __init__(self, num_dof):
        """
        初始化零空间多任务控制器。
        :param num_dof: 固定基座机械臂的关节自由度数
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
        # 利用 N1 进行动力学一致的零空间投影
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

这个例子展示了任务力 `F1` 如何通过 $\mathbf J_1^\top$ 映射到关节力矩，以及次任务力矩如何经过 $\mathbf N_1^\top$ 投影。数值阻尼会引入近似误差，因此工程实现还应检查次任务对主任务的残余影响是否足够小。

## 小结

- 人形机器人是带接触的浮动基座系统；关节多并不意味着所有广义自由度都能被直接驱动。
- **雅可比矩阵**把关节速度映射到任务空间速度，其转置把任务空间力映射为广义力。
- 零空间投影表达任务优先级，QP 则便于同时写入动力学、摩擦和执行器边界。工程系统可以组合两类方法。

$$
$$
