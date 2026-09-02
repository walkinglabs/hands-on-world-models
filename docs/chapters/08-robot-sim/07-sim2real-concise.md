# 8.7 Sim2Real 虚实迁移框架的简洁实现

同一个摆杆控制器，在仿真中可能每次都能稳定直立，装到实机后却持续小幅振荡。原因往往不是策略完全失效，而是电机响应慢了几毫秒、关节摩擦更大，或连杆质量与模型中的数值略有不同。仿真与实机之间这类系统性差异通常称为“现实鸿沟”（Reality Gap）。Sim2Real 的目标，是让策略在面对这些差异时仍能完成任务。

<div align="center">
<img src="/figures/08-robot-sim/source/07-sim2real-concise/rma-fig4.png" alt="RMA 在油膜等突变真实表面上恢复步态，直观呈现 Sim2Real 策略对隐藏物理变化的在线适应。" width="86%">

_图 8.7-1：RMA 在油膜等突变真实表面上恢复步态，直观呈现 Sim2Real 策略对隐藏物理变化的在线适应。 出处：Ashish Kumar et al.，[RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034)（2021），Figure 4。_
</div>

Sim2Real 已在多类机器人任务中得到验证。例如，OpenAI 使用自动域随机化训练灵巧手，并在现实中完成魔方复原 [[Akkaya et al., 2019]](https://arxiv.org/abs/1910.07113)；苏黎世联邦理工学院的研究人员利用执行器建模与强化学习，让 ANYmal 在真实环境中完成动态运动 [[Hwangbo et al., 2019]](https://doi.org/10.1126/scirobotics.aau5872)。本节将从动力学方程出发，推导并实现常见的域随机化方法。

<div align="center">
<img src="/figures/08-robot-sim/source/07-sim2real-concise/tobin-fig6.png" alt="视觉域随机化训练的抓取策略在真实桌面上完成多轮物体抓取试验。" width="86%">

_图 8.7-2：视觉域随机化训练的抓取策略在真实桌面上完成多轮物体抓取试验。 出处：Josh Tobin et al.，[Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907)（2017），Figure 6。_
</div>

## 系统动力学差异与现实鸿沟的数学本源

先从一个一维质量块开始。这个例子足够简单，却能直接展示参数误差如何改变控制结果。

假设我们试图控制一个在一维轨道上滑行的质量块。由牛顿第二定律可知，$F = m a$。如果我们以离散时间步 $\Delta t$ 观察该系统，令 $x_t$ 为时间 $t$ 时的状态（位置与速度），$u_t$ 为施加的控制力。在理想情况下，下一时刻的状态 $x_{t+1}$ 完全由当前状态和输入决定，我们可以写出最简单的一阶标量差分方程：

$$x_{t+1} = a x_t + b u_t$$

这里，$a$ 描述了系统固有的阻尼或惯性特征，$b$ 则反映了输入控制力转化为状态变化的增益（本质上与质量的倒数 $\frac{1}{m}$ 呈正相关）。在仿真环境中，参数 $a$ 和 $b$ 是由程序员精确设定的常数。

然而，在真实世界中，不仅质量 $m$ 存在制造公差，滑动摩擦系数也会随着温度、湿度和磨损发生非线性变化。因此，真实世界的动力学参数其实是不可观测的随机变量。我们将仿真器中的参数记为 $\theta_{\text{sim}} = (a, b)$，而将真实世界中的未知参数记为 $\theta_{\text{real}} = (a^*, b^*)$。如果直接将在 $\theta_{\text{sim}}$ 下获得的最优策略 $\pi^*(x)$ 部署到 $\theta_{\text{real}}$ 下，策略往往会因为轻微的误差累积而发散。

顺理成章地，我们将这一标量动力学推广到机器人控制中常见的多维状态空间和非线性系统。令 $\mathbf{x}_t \in \mathbb{R}^n$ 为关节角度和角速度张量，$\mathbf{u}_t \in \mathbb{R}^m$ 为关节扭矩控制张量。真实物理环境的非线性演化可以表示为：

<div align="center">
<img src="/figures/08-robot-sim/source/07-sim2real-concise/lbc-fig2.png" alt="Learning by Cheating 对比特权教师与相机学生的策略结构，展示仿真可见状态怎样转化为部署输入。" width="86%">

_图 8.7-3：Learning by Cheating 对比特权教师与相机学生的策略结构，展示仿真可见状态怎样转化为部署输入。 出处：Dian Chen et al.，[Learning by Cheating](https://arxiv.org/abs/1912.12294)（2020），Figure 2。_
</div>

$$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t; \mathbf{\Theta}_{\text{real}}) + \mathbf{\epsilon}_t$$

其中，$\mathbf{\Theta}_{\text{real}}$ 包含真实的物理参数（如连杆质量、惯量和电机摩擦系数），$\mathbf{\epsilon}_t$ 表示观测与执行噪声。真实参数通常只能被近似测量，而且还会随温度、磨损和负载变化。Sim2Real 因此不把某一组仿真参数当作唯一真值，而是让策略适应一组合理的参数变化。

## 域随机化（Domain Randomization）

域随机化的做法是：训练时不固定 $\mathbf{\Theta}_{\text{sim}}$，而是从预先设定的分布中反复采样质量、摩擦、时延等参数。这样得到的策略不只适应一个仿真器实例，而是在一组动力学条件下都获得较高回报。

<div align="center">
<img src="/figures/08-robot-sim/source/07-sim2real-concise/peng-fig7.png" alt="动力学随机化对比仿真与真实机器人表现，展示参数分布训练对迁移结果的影响。" width="86%">

_图 8.7-4：动力学随机化对比仿真与真实机器人表现，展示参数分布训练对迁移结果的影响。 出处：Xue Bin Peng et al.，[Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537)（2018），Figure 7。_
</div>

设随机参数 $\mathbf{\Theta}$ 服从我们人为设定的先验分布 $P_{\mathbf{\Theta}}$。在强化学习的标准马尔可夫决策过程（MDP）中，我们的目标是最大化累积期望奖励。在引入域随机化后，我们需要最大化的是**整个参数分布上的期望总奖励**：

$$J(\pi_\phi) = \mathbb{E}_{\mathbf{\Theta} \sim P_{\mathbf{\Theta}}} \left[ \mathbb{E}_{\tau \sim P(\tau | \pi_\phi, \mathbf{\Theta})} \left[ \sum_{t=0}^T \gamma^t r_t \right] \right]$$

<div align="center">
<img src="/figures/08-robot-sim/latex/07-sim2real-concise/domain-randomization-nested-expectation.png" alt="每个回合固定一组物理参数并先累计时间回报，再依次对轨迹和参数样本求期望" width="86%">

_图 8.7-5：每个回合只采样一次物理参数并在整条轨迹中保持固定；先累计该轨迹的折扣回报，再依次平均轨迹随机性与参数随机性。_
</div>

其中，轨迹 $\tau = (\mathbf{x}_0, \mathbf{u}_0, \mathbf{x}_1, \dots)$ 的生成分布依赖于采样的物理参数 $\mathbf{\Theta}$，$\phi$ 为策略网络的权重。每轮回合开始时重新采样参数，可以降低策略对某一组名义参数的依赖。需要注意，上式优化的是参数分布上的平均回报；它并不自动保证最坏参数下也能成功。若任务关心最坏情况，还需要风险敏感目标、对抗式采样或专门的鲁棒控制设计。

## 域随机化与策略包装的简洁代码实现

(**我们将构建一个轻量级的域随机化层，并在 PyTorch 中演示如何对前向动力学参数进行采样与张量批处理。**)

```python
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

        # 均匀细杆绕端点转动时，转动惯量为 mL^2/3；这里令 L=1
        inertia = mass * (1.0 ** 2) / 3.0 + 1e-6

        # 计算角加速度: a = (tau - friction * v - m * g * l * sin(theta)) / I
        gravity_torque = mass * self.gravity * 0.5 * torch.sin(theta)
        friction_torque = friction * theta_dot
        theta_ddot = (action - friction_torque - gravity_torque) / inertia

        # 半隐式欧拉积分 (Semi-implicit Euler integration)
        new_theta_dot = theta_dot + theta_ddot * self.dt
        new_theta = theta + new_theta_dot * self.dt

        return torch.cat([new_theta, new_theta_dot], dim=1)
```

接下来，我们实现一个域随机化包装器（Domain Randomizer）。它的核心功能是在训练的 `reset` 阶段，为并行环境生成服从特定分布的物理参数张量。

```python
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
        """抽取一批物理参数。"""
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
print("对应的下一状态:\n", next_states)
```

在上述代码中，每个样本接收到相同的控制输入，但 `mass` 和 `friction` 不同，因此下一状态也不同。把这种批量动力学放入强化学习环境后，策略梯度会综合多个参数样本的回报进行更新。它学到的是采样分布内较为稳健的折中策略，而不是对所有可能实机参数的保证。

## 从参数猜测到实机校准

域随机化不是把参数范围设得越宽越好。范围过窄，真实系统可能落在训练分布之外；范围过宽，策略又可能为了兼顾彼此矛盾的动力学而变得保守。一个更可靠的工程循环包含四步：

1. 根据 CAD、称重结果和电机规格建立名义模型。
2. 在低风险动作下记录关节位置、速度、电流与控制时延。
3. 用这些日志估计质量、摩擦、执行器响应和延迟的合理区间。
4. 在未参与估计的实机轨迹上验证，再调整随机化分布。

对腿式机器人来说，理想扭矩与实际关节输出之间的差异常常比刚体参数误差更显著。Hwangbo 等人训练执行器网络来近似电机与传动系统的动态响应，再把该模型放回仿真训练环路 [[Hwangbo et al., 2019]](https://doi.org/10.1126/scirobotics.aau5872)。OpenAI 的灵巧手实验则把动力学、观测、动作延迟等因素纳入自动域随机化，并依据训练表现逐步调整范围 [[Akkaya et al., 2019]](https://arxiv.org/abs/1910.07113)。两项工作说明了同一个原则：随机化分布应由实机证据约束，并在部署反馈中持续修订。

实机验证还应与训练数据分开。若每次调参都只看同一条轨迹，很容易把偶然误差当成模型规律。可以保留不同负载、不同电量和不同地面的验证集合，分别检查跟踪误差、振荡幅度、能耗和失败率。这样得到的结论比单一“迁移成功率”更容易定位问题。

## 小结

- 现实鸿沟来自质量、摩擦、执行器、传感器和时延等多方面差异。
- 域随机化优化参数分布上的期望回报，不等同于最坏情况下的性能保证。
- 随机化范围应由实机测量与留出轨迹校准；执行器响应也应进入模型或随机化过程。

## 练习

1. 如果质量分布的方差设置得过大，策略可能出现哪些保守行为？试从关节刚度、动作幅度和任务速度三个方面分析。
2. 当惯量很小、阻尼很大时，固定步长的半隐式欧拉积分仍可能不稳定。改变 $\Delta t$，观察状态何时开始振荡或发散。
3. 设计一组用于估计执行器延迟的低风险输入信号，并说明如何从命令与关节响应的日志中得到延迟范围。
