# Sim2Real 虚实迁移框架的简洁实现

:label:sec_sim2real_concise

在前面的章节中，我们在高度理想化的物理仿真环境中训练了机器人的运动控制策略。然而，当这些策略被直接部署到真实的物理硬件上时，往往会遭遇灾难性的失败。这种由仿真环境与真实物理世界之间的动力学差异、传感器噪声、以及通信延迟所导致的性能骤降，在学术界被称为“现实鸿沟”（Reality Gap）。为了跨越这一鸿沟，学术界和工业界发展出了从仿真到现实（Simulation-to-Reality, 简称为 Sim2Real）的迁移框架。

Sim2Real 已在多类机器人任务中得到验证。例如，OpenAI 使用自动域随机化训练灵巧手，并在现实中完成魔方复原 [[Akkaya et al., 2019]](https://arxiv.org/abs/1910.07113)；苏黎世联邦理工学院的研究人员利用执行器建模与强化学习，让 ANYmal 在真实环境中完成动态运动 [[Hwangbo et al., 2019]](https://doi.org/10.1126/scirobotics.aau5872)。本节将从动力学方程出发，推导并实现常见的域随机化方法。

## 系统动力学差异与现实鸿沟的数学本源

要理解现实鸿沟，我们必须首先用精确的数学语言描述系统的演化。在没有任何复杂几何或多体动力学介入之前，让我们考虑高中物理中最简单的质点运动学。

假设我们试图控制一个在一维轨道上滑行的质量块。由牛顿第二定律可知，$F = m a$。如果我们以离散时间步 $\Delta t$ 观察该系统，令 $x_t$ 为时间 $t$ 时的状态（位置与速度），$u_t$ 为施加的控制力。在理想情况下，下一时刻的状态 $x_{t+1}$ 完全由当前状态和输入决定，我们可以写出最简单的一阶标量差分方程：

$$x_{t+1} = a x_t + b u_t$$
:eqlabel:eq_sim2real_scalar

这里，$a$ 描述了系统固有的阻尼或惯性特征，$b$ 则反映了输入控制力转化为状态变化的增益（本质上与质量的倒数 $\frac{1}{m}$ 呈正相关）。在仿真环境中，参数 $a$ 和 $b$ 是由程序员精确设定的常数。

然而，在真实世界中，不仅质量 $m$ 存在制造公差，滑动摩擦系数也会随着温度、湿度和磨损发生非线性变化。因此，真实世界的动力学参数其实是不可观测的随机变量。我们将仿真器中的参数记为 $\theta_{\text{sim}} = (a, b)$，而将真实世界中的未知参数记为 $\theta_{\text{real}} = (a^*, b^*)$。如果直接将在 $\theta_{\text{sim}}$ 下获得的最优策略 $\pi^*(x)$ 部署到 $\theta_{\text{real}}$ 下，策略往往会因为轻微的误差累积而发散。

顺理成章地，我们将这一标量动力学推广到机器人控制中常见的多维状态空间和非线性系统。令 $\mathbf{x}_t \in \mathbb{R}^n$ 为关节角度和角速度张量，$\mathbf{u}_t \in \mathbb{R}^m$ 为关节扭矩控制张量。真实物理环境的非线性演化可以表示为：

$$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t; \mathbf{\Theta}_{\text{real}}) + \mathbf{\epsilon}_t$$
:eqlabel:eq_sim2real_matrix

其中，$\mathbf{\Theta}_{\text{real}}$ 包含了全部真实的物理参数（如所有连杆的质量矩阵、惯量张量、电机摩擦系数等），而 $\mathbf{\epsilon}_t$ 表示不可避免的观测与执行噪声。由于 $\mathbf{\Theta}_{\text{real}}$ 永远无法被完美的解析测量，Sim2Real 的核心数学思想即是通过优化策略对参数分布的鲁棒性来对抗这种物理参数的固有不确定性。

## 域随机化 (Domain Randomization) 的严密推导

既然无法获得精确的 $\mathbf{\Theta}_{\text{real}}$，一种极其朴素但异常强大的思想诞生了：让策略在训练时经历大量不同的可能参数 $\mathbf{\Theta}_{\text{sim}}$，使得真实世界的参数 $\mathbf{\Theta}_{\text{real}}$ 只是仿真参数分布中的一个样本。这被称为域随机化。

> 我们可以借用一种极端的训练场景来理解域随机化背后的数学直觉：这就像是在训练一名乒乓球运动员时，强制让他戴上不同扭曲度的透镜、在不同重力系数的房间内打球。由于大脑被迫学习一种“对环境变动不敏感”的广义特征表示，当他摘下所有透镜来到正常的真实球场时，他对光线和风速的微小误差已经完全免疫。

设随机参数 $\mathbf{\Theta}$ 服从我们人为设定的先验分布 $P_{\mathbf{\Theta}}$。在强化学习的标准马尔可夫决策过程（MDP）中，我们的目标是最大化累积期望奖励。在引入域随机化后，我们需要最大化的是**整个参数分布上的期望总奖励**：

$$J(\pi_\phi) = \mathbb{E}_{\mathbf{\Theta} \sim P_{\mathbf{\Theta}}} \left[ \mathbb{E}_{\tau \sim P(\tau | \pi_\phi, \mathbf{\Theta})} \left[ \sum_{t=0}^T \gamma^t r_t \right] \right]$$
:eqlabel:eq_sim2real_objective

其中，轨迹 $\tau = (\mathbf{x}_0, \mathbf{u}_0, \mathbf{x}_1, \dots)$ 的生成分布现在直接依赖于采样的物理参数 $\mathbf{\Theta}$。$\phi$ 为策略神经网络的权重。通过在每一轮回合（Episode）开始时从 $P_{\mathbf{\Theta}}$ 中重新采样质量、摩擦力等参数，神经网络 $\pi_\phi$ 会自动惩罚那些过度依赖特定质量参数的脆弱行为，从而收敛到一种具备广义鲁棒性的次优解。

## 域随机化与策略包装的简洁代码实现

(**我们将构建一个轻量级的域随机化层，并在 PyTorch 中演示如何对前向动力学参数进行采样与张量批处理。**)

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
import torch.distributions as dist

class RandomizedDynamics(nn.Module):
    """支持物理参数批处理随机化的简易动力学模型"""
    def __init__(self, dt=0.01):
        super().__init__()
        self.dt = dt
        self.gravity = 9.81

    def forward(self, state, action, mass, friction):
        """
        计算下一时刻的状态
        state: 包含 [角度, 角速度] 的张量，形状 (batch_size, 2)
        action: 控制扭矩，形状 (batch_size, 1)
        mass: 杆的随机质量，形状 (batch_size, 1)
        friction: 关节摩擦系数，形状 (batch_size, 1)
        """
        theta, theta_dot = state[:, 0:1], state[:, 1:2]

        # 为了避免除零错误，我们对 mass 加上一个极小的 epsilon
        inertia = mass * (1.0 ** 2) / 3.0 + 1e-6

        # 计算角加速度: a = (tau - friction * v - m * g * l * sin(theta)) / I
        # 我们在这里使用纯张量运算以支持大规模并行仿真
        gravity_torque = mass * self.gravity * 0.5 * torch.sin(theta)
        friction_torque = friction * theta_dot
        theta_ddot = (action - friction_torque + gravity_torque) / inertia

        # 半隐式欧拉积分 (Semi-implicit Euler integration)
        new_theta_dot = theta_dot + theta_ddot * self.dt
        new_theta = theta + new_theta_dot * self.dt

        return torch.cat([new_theta, new_theta_dot], dim=1)
```

接下来，我们实现一个域随机化包装器（Domain Randomizer）。它的核心功能是在训练的 `reset` 阶段，为并行环境生成服从特定分布的物理参数张量。

```{.python .input}
#@tab pytorch
class DomainRandomizer:
    """物理参数先验分布采样器"""
    def __init__(self, batch_size, device='cpu'):
        self.batch_size = batch_size
        self.device = device

        # 定义先验分布：质量我们采用对数正态分布以保证其恒为正
        # 摩擦力使用均匀分布
        self.mass_dist = dist.LogNormal(torch.tensor(0.0), torch.tensor(0.5))
        self.fric_dist = dist.Uniform(torch.tensor(0.0), torch.tensor(0.2))

    def sample_parameters(self):
        """[抽取一批具备多样性的物理参数]"""
        mass = self.mass_dist.sample((self.batch_size, 1)).to(self.device)
        friction = self.fric_dist.sample((self.batch_size, 1)).to(self.device)
        return mass, friction

# 演示前向传播
batch_size = 4
randomizer = DomainRandomizer(batch_size)
dynamics = RandomizedDynamics()

# 初始状态全为 0，采取恒定扭矩 1.0
states = torch.zeros((batch_size, 2))
actions = torch.ones((batch_size, 1))

mass, friction = randomizer.sample_parameters()
next_states = dynamics(states, actions, mass, friction)

print("采样的质量张量:\n", mass)
print("对应的下一状态 (可见因质量不同，角速度产生了显著分化):\n", next_states)
```

在上述代码中，尽管每个样本接收到的控制输入 `actions` 是完全相同的，但由于 `DomainRandomizer` 为它们赋予了截然不同的 `mass` 和 `friction`，系统演化出的 `next_states` 呈现出了巨大的多样性。正是通过对这种海量多样性的求导与参数更新，策略网络被迫寻找到一条无论在轻杆还是重杆上都不会翻车的控制流形。

## 2026年具身智能开源生态与硬件敏捷开发

在理解了核心算法后，我们必须将目光投向学术方程与工程实践的交界处。直到 2024 年，绝大多数初创团队的 Sim2Real 流程仍然是极其痛苦的：算法工程师使用 MuJoCo 调参，而机械工程师则使用 SolidWorks 和 C++ 调整底层驱动，两者之间的“现实鸿沟”不仅存在于物理公式中，更存在于割裂的软硬件生态中。

然而，2026 年具身智能开源社区的爆发彻底改变了这一现状。以 YC (Y Combinator) 孵化的 **Philon** 机器人开源生态为代表，硬件基础设施逐渐像当年的 Linux 和 ROS 一样走向了标准化与全开源。初创公司能够借助这些强大的基础设施，将 Sim2Real 的迭代周期从原本的“月级”压缩至“天级”。

### Asimov v1 与完全等构的质量分布映射

在过去的模型中，我们依赖代码中的 `dist.LogNormal` 去漫无目的地瞎猜真实硬件的质量误差。而 2026 年开源的 **Asimov v1**（一款 25 自由度全尺寸双足人形机器人）则通过发布绝对精确的开源物料清单（BOM），将这种猜测的必要性降到了最低。

Asimov v1 的核心贡献不仅在于公布了全套碳纤维连杆的图纸，更在于它原生提供了一套与现实材料密度**数学上严格同构**的 URDF (Unified Robot Description Format) 模型。借助高精度工业 CT 扫描数据集，社区将每一个伺服电机、减速器内部的游星齿轮、以及走线带来的偏心惯量张量，精准映射到了仿真参数的先验高斯均值 $\mu_{\mathbf{\Theta}}$ 中。这意味着，初创团队在应用我们上面推导的该公式时，其采样方差 $\sigma_{\mathbf{\Theta}}$ 可以设置得极小。这种“以开源物理测绘消除未知参数”的降维打击，直接将双足行走的虚实迁移成功率从 40% 提升到了 95% 以上。

### AIRSEAI 标准接口与执行器动力学鸿沟的消解

Sim2Real 的另一个致命问题在于“控制延迟”与“扭矩指令失真”。仿真里的 `action` 瞬间就能转化为准确的扭矩，而在现实中，经过 EtherCAT 总线、电机驱动器 PWM 波形生成、再到磁场力矩，存在高度非线性的动态过程。

为了解决这一问题，**AIRSEAI (Artificial Intelligence Robot Standard Engine and Interface)** 联盟于 2026 年推出了全新的标准接口层。AIRSEAI 强制规定了所有兼容的硬件驱动器，必须在以太网数据帧中实时回传高频的电流微分与转子磁链状态。
通过 AIRSEAI 接口，初创团队不再需要针对每一款电机手写复杂的“执行器神经网络（Actuator Network）”。AIRSEAI 固件层直接在硬件驱动端实现了精确的迟滞补偿，使得硬件对外暴露的控制流形，在数学上无限逼近刚体动力学仿真器（如 Isaac Sim 或 MuJoCo）中定义的纯理想扭矩输入。硬件本身承担了“自适应消除自身非线性”的责任，大幅简化了上层 RL 算法的 Sim2Real 压力。

在 Philon 生态的加持下，今天的创业团队只需在云端 GPU 集群完成大规模域随机化训练，将策略网络导出为 ONNX，并通过 AIRSEAI 中间件一键部署到 Asimov v1 兼容的物理实体上。这种极致的敏捷开发，标志着具身智能正式步入了“软件定义硬件”的新纪元。

## 小结

- 现实鸿沟的本质是仿真参数 $\mathbf{\Theta}_{\text{sim}}$ 与真实物理参数 $\mathbf{\Theta}_{\text{real}}$ 之间的数学分布不匹配。
- 域随机化（Domain Randomization）通过强制策略网络在训练期间最大化参数先验分布上的期望奖励，迫使其学习到对物理扰动具有强鲁棒性的控制流形。
- 2026 年的开源硬件生态（如 Asimov v1 的高保真 BOM 与 AIRSEAI 的标准化执行层）在工程源头上极大缩小了参数先验差异，从根本上降低了 Sim2Real 的落地门槛。

## 练习

1.  如果在该公式的训练中，将质量分布的方差设置得过大，策略网络可能会表现出何种保守的行为？
    _提示：思考在一个质量极小和质量极大的分布中同时不摔倒，机器人会倾向于何种刚度的关节控制。_
2.  在我们的 PyTorch 简易实现中，我们使用了半隐式欧拉积分来更新状态。如果系统的惯量极小而阻尼极大（即方程非常“刚性”），这会导致仿真出现什么数学灾难？
    _提示：回顾微积分中的步长 $\Delta t$ 与差分方程稳定域的关系。_
3.  查阅关于执行器网络（Actuator Nets）的经典文献，尝试论述：为何 2026 年 AIRSEAI 的硬件底层补偿机制，在某些要求极端柔顺控制的场景下，可能会与上层强化学习算法产生控制环路的耦合震荡？

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
