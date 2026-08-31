# 域随机化

## 现实鸿沟与学术渊源

强化学习与机器人学长期面临一个核心悖论：直接在物理世界中训练机器人成本高昂、危险且极其耗时；而在仿真环境中训练虽然廉价且高效，但仿真世界与真实物理世界之间不可避免地存在差异。这一差异被称为**现实鸿沟**（Reality Gap）。

由于摩擦力、接触动力学（Contact Dynamics）、传感器噪声甚至柔性材质的形变，这些物理现象在数学上属于高度非线性的偏微分方程，现有的物理引擎（如MuJoCo、PyBullet、Isaac Gym）只能对其进行极简的近似。如果在单一的、确定的仿真器中训练一个神经网络控制器，它往往会过拟合于仿真器中不完美的物理法则。一旦部署到真实机器人上，策略便会瞬间失效。

Tobin 等人把**域随机化**系统用于视觉 Sim-to-Real：训练时随机改变纹理、光照、相机与物体属性，让真实图像有机会落入训练分布 [[Tobin et al., 2017]](https://arxiv.org/abs/1703.06907)。这是一种提高鲁棒性的经验策略，并不能保证只要真实参数“落在范围内”就天然免疫所有偏差。Peng 等人进一步随机化质量、摩擦、阻尼等动力学参数，并在推、开门等机器人任务上验证迁移 [[Peng et al., 2018]](https://arxiv.org/abs/1808.00177)；该论文并未报告灵巧手任务，也没有证明“无缝迁移”。

## 从牛顿力学到参数化马尔可夫过程

为了深刻理解域随机化背后的数学本质，我们首先回归到高中阶段最基础的物理模型：一个质量为 $m$ 的滑块放置在水平桌面上，桌面与滑块之间的动摩擦因数为 $\mu$。此时，我们通过外力 $F$ 去推动这个滑块。

根据牛顿第二定律，滑块在任意时刻的加速度 $a$ 可以表示为：

$$
a = \frac{F - \mu m g}{m}
$$

假设我们在离散时间 $\Delta t$ 内对其进行积分（欧拉法），已知当前时刻的速度为 $v_t$，那么下一时刻的速度 $v_{t+1}$ 为：

$$
v_{t+1} = v_t + \left(\frac{F_t}{m} - \mu g\right) \Delta t
$$

在这个微型物理系统中，决定其动力学演化的隐藏物理参数可以写成一个向量 $\boldsymbol{\xi} = [m, \mu]^\top$。

如果在仿真器中，我们设定质量 $m=1.0$，摩擦系数 $\mu=0.1$（即仿真参数 $\boldsymbol{\xi}_{sim} = [1.0, 0.1]^\top$），而真实世界中桌子可能更粗糙，滑块也有磨损（即真实参数 $\boldsymbol{\xi}_{real} = [1.5, 0.2]^\top$）。如果神经网络控制策略仅仅针对 $\boldsymbol{\xi}_{sim}$ 进行了优化，它所输出的 $F_t$ 必然无法使真实系统达到预期的速度 $v_{t+1}$。

为了将其纳入现代强化学习的理论框架，我们将这个带有具体物理参数的系统严格定义为**参数化马尔可夫决策过程**（Parameterized MDP）。记作 $\mathcal{M}_{\boldsymbol{\xi}} = (\mathcal{S}, \mathcal{A}, \mathcal{P}_{\boldsymbol{\xi}}, \mathcal{R}_{\boldsymbol{\xi}}, \gamma, \rho_0)$。请注意，这里的状态转移概率矩阵 $\mathcal{P}_{\boldsymbol{\xi}}(s_{t+1}|s_t, a_t)$ 显式地依赖于系统当前的物理参数 $\boldsymbol{\xi}$。

在标准的强化学习中，我们的目标是寻找一个参数为 $\theta$ 的策略网络 $\pi_\theta(a|s)$，以最大化固定参数下的期望累积回报：

$$
J(\theta; \boldsymbol{\xi}) = \mathbb{E}_{\tau \sim \pi_\theta, \mathcal{P}_{\boldsymbol{\xi}}} \left[ \sum_{t=0}^T \gamma^t \mathcal{R}_{\boldsymbol{\xi}}(s_t, a_t) \right]
$$

而在**域随机化**中，我们将参数 $\boldsymbol{\xi}$ 视为服从某种先验分布 $P_{\Xi}$ 的随机变量。我们的新优化目标 $J_{DR}(\theta)$ 变为了在该分布下的双重期望：

$$
J_{DR}(\theta) = \mathbb{E}_{\boldsymbol{\xi} \sim P_{\Xi}} \left[ \mathbb{E}_{\tau \sim \pi_\theta, \mathcal{P}_{\boldsymbol{\xi}}} \left[ \sum_{t=0}^T \gamma^t \mathcal{R}_{\boldsymbol{\xi}}(s_t, a_t) \right] \right]
$$

通过该公式可以清晰地看出，策略 $\pi_\theta$ 必须在**所有可能被采样出来的物理世界中**都表现良好。这在数学上等价于一种针对模型参数不确定性的鲁棒优化（Robust Optimization）。

## 动力学随机化与方差挑战

在真正的机器人控制中，我们通常需要随机化（Dynamics Randomization）的参数远不止质量和摩擦力。一个典型的机器人关节包括电机、减速器和连杆。因此，分布 $P_{\Xi}$ 包含的维度通常高达数十甚至数百维：

1. **刚体属性**：连杆的质量 $m$、惯性张量 $\mathbf{I}$、质心绝对位置。
2. **接触物理**：滑动摩擦力、扭转摩擦力、接触面恢复系数（代表弹性碰撞程度）。
3. **驱动器属性**：电机的转矩限制、电机内部阻尼、底层PD控制器的刚度系数（Stiffness）。
4. **延迟与噪声**：由于通信带来的动作延迟时长 $d$、传感器观测的高斯噪声协方差 $\boldsymbol{\Sigma}$。

当我们在该公式的基础上对策略网络参数 $\theta$ 进行梯度上升时，根据策略梯度定理（Policy Gradient Theorem），其解析梯度形式为：

$$
\nabla_\theta J_{DR}(\theta) = \mathbb{E}_{\boldsymbol{\xi} \sim P_{\Xi}} \left[ \mathbb{E}_{\tau} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}(s_t, a_t; \boldsymbol{\xi}) \right] \right]
$$

该公式深刻揭示了域随机化在工程训练中的一个核心痛点：**极高的梯度方差**。传统的强化学习梯度方差仅仅来源于策略网络本身的随机动作采样；而在域随机化中，外部物理环境参数 $\boldsymbol{\xi}$ 的随机采样强行引入了一层额外的方差。如果分布 $P_{\Xi}$ 设置得过于宽泛，导致某些环境下的轨迹完全发散，其优势函数 $\hat{A}$ 剧烈波动，整个神经网络的梯度将被噪声淹没从而无法收敛。因此，精心设计 $P_{\Xi}$ 的分布形状与上下界，是成功应用域随机化的命脉所在。

## 视觉域随机化与特征不变性

除了上述的物理动力学参数，对于基于摄像头的端到端控制系统，如何跨越**渲染图像与真实相机图像之间的鸿沟**同样至关重要。

令系统内部真实的3D状态为 $\mathbf{s}_t$（例如机械臂各个关节的三维坐标及目标物块的位姿），计算机图形学引擎的渲染过程可以严格抽象为一个映射函数：

$$
\mathbf{I}_t = f_{render}(\mathbf{s}_t; \boldsymbol{\xi}_{vis})
$$

这里，$\boldsymbol{\xi}_{vis}$ 表示与物体核心几何结构无关的渲染参数，例如光源的位置和光线颜色、摄像机的内参（焦距）与外参（位姿）、背景甚至干扰物的纹理贴图。

视觉域随机化（Visual Domain Randomization）强制要求在每次渲染图像时，从极其宽泛的均匀分布中随机抽取 $\boldsymbol{\xi}_{vis}$。如果我们将卷积神经网络（CNN）的早期特征提取层记为 $f_\phi$，我们希望整个网络在反向传播过程中，自发地被迫学习到一种**特征不变性**（Feature Invariance）。即，对于完全相同的物理状态 $\mathbf{s}_t$，无论渲染参数如何剧烈改变，其提取出的高维特征张量必须几乎相等：

$$
f_\phi\left( f_{render}(\mathbf{s}_t; \boldsymbol{\xi}_{vis}^{(1)}) \right) \approx f_\phi\left( f_{render}(\mathbf{s}_t; \boldsymbol{\xi}_{vis}^{(2)}) \right)
$$

这表明神经网络成功地屏蔽了光影、纹理等外在干扰，真正理解并提纯了目标物体的纯粹几何结构空间。

## 隐式系统辨识：记忆的贝叶斯推断

在前面的小节中，我们探讨的策略均是无记忆的前馈神经网络 $\pi_\theta(a_t | \mathbf{s}_t)$。然而，仅仅依靠当前时刻的一帧观测来输出动作，往往会使策略变得过于保守（Conservative）。因为网络在时间步 $t$ 并不知道当前环境的滑块质量究竟是 $0.5$ 还是 $1.5$，为了不计算出导致系统崩溃的极端控制力，它只能输出一个妥协的、平均化的微小推力。

为了打破这种性能上的妥协，现代强化学习通常向策略网络的主干中引入循环神经网络（如 LSTM 或 GRU 等记忆模块），将策略的定义域扩展为 $\pi_\theta(a_t | \mathbf{s}_t, \mathbf{h}_{t-1})$，其中 $\mathbf{h}_{t-1}$ 是过去所有观测和动作被高度压缩后的隐状态向量。

> 💡 **物理直觉**：如同一个蒙着眼睛的司机驾驶一辆陌生的汽车，虽然他事先不知道车辆的载重和路面摩擦力，但在踩下第一脚油门并感受到加速度反馈的瞬间，他立刻就在大脑中完成了对这些隐藏物理参数的近似推断。具有记忆的神经网络（如RNN）在随机化环境中的行为正是如此——它利用历史动作和观测序列，隐式地充当了一个卡尔曼滤波器（Kalman Filter）或贝叶斯估计器。

在严谨的数学视角下，随着时间序列的步步展开，记忆网络模块其实质是在隐式地估算当前所在环境对应真实物理参数的后验概率分布（Posterior Distribution）：

$$
\hat{P}(\boldsymbol{\xi} | \mathbf{s}_1, a_1, \mathbf{s}_2, a_2, \dots, \mathbf{s}_t)
$$

这就是学术界常说的**隐式系统辨识**（Implicit System Identification）。依靠这一强大的机制，机器人在与物理世界接触的前零点几秒内，便能迅速摸清真实的物理法则，并实时切换到对应的最优控制流形上。

## 实现一个带域随机化的物理环境

[**我们将通过代码具体实现上文推导的滑块物理系统，并在每次重置时对其动力学参数进行域随机化。**] 我们将使用均匀分布对滑块质量和桌面摩擦系数进行采样，并采用一阶欧拉积分来更新连续物理状态。

```python
import torch
from torch.distributions import Uniform

class RandomizedBlockEnv:
    """基于PyTorch的带有动力学域随机化的1D滑块环境"""
    def __init__(self, mass_range=(0.5, 2.0), friction_range=(0.05, 0.3), dt=0.1):
        self.mass_range = mass_range
        self.friction_range = friction_range
        self.dt = dt
        self.mass = None
        self.friction = None
        self.state = None  # 状态向量形式：[位置, 速度]

    def reset(self):
        # 核心机制：环境重置时，从分布中随机采样物理参数
        self.mass = Uniform(self.mass_range[0], self.mass_range[1]).sample().item()
        self.friction = Uniform(self.friction_range[0], self.friction_range[1]).sample().item()

        # 将位置和速度初始化为 0.0
        self.state = torch.zeros(2)
        return self.state

    def step(self, action):
        # action: 在1D方向上施加的外力 F
        pos, vel = self.state[0], self.state[1]

        # 计算滑动摩擦力，方向与速度严格相反
        # 引入微小阈值 1e-3 以保证数值计算稳定性，防止在原点产生无穷震荡
        vel_sign = torch.sign(vel) if torch.abs(vel) > 1e-3 else torch.tensor(0.0)
        f_fric = self.friction * self.mass * 9.8 * vel_sign

        # 根据该公式计算净加速度
        acc = (action - f_fric) / self.mass

        # 根据该公式推进时间步
        new_vel = vel + acc * self.dt
        new_pos = pos + new_vel * self.dt

        # 更新系统的内在状态张量
        self.state = torch.tensor([new_pos, new_vel])

        # 定义一个二次型的简单奖励函数：趋向于让位置到达10.0并维持静止
        reward = -((new_pos - 10.0)**2 + 0.1 * new_vel**2)

        return self.state, reward, False
```

[**下面，我们向环境连续施加相同大小的力，以验证不同的物理参数对相同动作序列究竟会产生多大的状态轨迹偏移。**]

```python
# 实例化随机化物理环境
env = RandomizedBlockEnv()
# 定义一个固定的动作序列：连续施加大小为5.0的恒定推力，持续20个步长
action_sequence = torch.tensor([5.0] * 20)

def simulate_trajectory(env, action_seq):
    env.reset()
    # 提取当前重置时被隐式采样的真实物理参数
    mass = env.mass
    friction = env.friction
    positions = []

    # 将动作逐帧作用于环境，并记录其一维位移轨迹
    for a in action_seq:
        state, _, _ = env.step(a)
        positions.append(state[0].item())
    return mass, friction, positions

# 分别在两个通过随机化生成的独立宇宙（环境实例）中，运行相同的控制序列
m1, f1, pos1 = simulate_trajectory(env, action_sequence)
m2, f2, pos2 = simulate_trajectory(env, action_sequence)

print(f"宇宙A (质量={m1:.2f}, 摩擦={f1:.2f}) 20步后的最终位移: {pos1[-1]:.2f}")
print(f"宇宙B (质量={m2:.2f}, 摩擦={f2:.2f}) 20步后的最终位移: {pos2[-1]:.2f}")
```

由于物理参数 $\boldsymbol{\xi}$ 是在每轮实验（Episode）开始前从我们预设的均匀分布中严格独立采样的，因此，即使神经网络策略输出完全相同的推力张量序列，其反馈的观测空间也会表现出极大的数值分歧。这从代码侧面上印证了前文我们针对高梯度方差的严密理论分析。

## 小结

- **域随机化（Domain Randomization）** 是一种通过向仿真器引入宏观随机性，使得在其中训练的强化学习策略能够零样本（Zero-shot）跨越“现实鸿沟”迁移至真实机器人的核心技术。
- 从数学角度看，域随机化将标准的马尔可夫决策过程拓展为了**参数化马尔可夫决策过程**，迫使策略网络必须在物理参数分布 $\boldsymbol{\xi} \sim P_{\Xi}$ 的双重期望下最大化累积回报。
- **视觉随机化**从分布上抹除了不相关的干扰渲染参数，强制卷积神经网络学习纯粹的几何特征；而**动力学随机化**确保控制器不会过拟合到某一套脆弱的动力学方程中。
- 为了避免前馈网络的性能受制于“保守策略”陷阱，算法实现中往往搭配循环神经网络，通过记忆历史序列来进行**隐式的系统辨识**（Implicit System Identification），赋予了模型在毫秒级自适应真实物理世界的能力。
