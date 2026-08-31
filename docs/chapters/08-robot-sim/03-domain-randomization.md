# 8.3 域随机化

## 现实鸿沟与学术渊源

设仿真中的滑块质量固定为 $1.0$ kg，真实滑块却是 $1.3$ kg。策略若只见过前一种情况，同样的推力会在现实中产生更小的加速度。质量、摩擦、延迟、相机曝光等仿真与现实之间的差异合称**现实鸿沟**（Reality Gap）。

<div align="center">
<img src="/figures/08-robot-sim/source/03-domain-randomization/tobin-fig1.png" alt="域随机化把大量合成外观映射到真实抓取场景，使真实画面成为训练分布中的一种变化。" width="86%">

_图 8.3-1：域随机化把大量合成外观映射到真实抓取场景，使真实画面成为训练分布中的一种变化。 出处：Josh Tobin et al.，[Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907)（2017），Figure 1。_
</div>

域随机化不试图把一个仿真器调成唯一的“正确世界”，而是在训练时主动生成许多稍有不同的世界。策略若能在这些变化中完成任务，便较少依赖某一组精确参数；但迁移是否成功仍取决于随机化范围是否覆盖了关键现实差异。

Tobin 等人把**域随机化**系统用于视觉 Sim-to-Real：训练时随机改变纹理、光照、相机与物体属性，让真实图像有机会落入训练分布 [[Tobin et al., 2017]](https://arxiv.org/abs/1703.06907)。这是一种提高鲁棒性的经验策略，并不能保证只要真实参数“落在范围内”就天然免疫所有偏差。Peng 等人进一步随机化质量、摩擦、阻尼等动力学参数，并在推、开门等机器人任务上验证迁移 [[Peng et al., 2018]](https://arxiv.org/abs/1808.00177)；该论文并未报告灵巧手任务，也没有证明“无缝迁移”。

## 从牛顿力学到参数化马尔可夫过程

先看质量为 $m$ 的滑块。它位于水平桌面上，动摩擦因数为 $\mu$，外力为 $F$。

根据牛顿第二定律，滑块在任意时刻的加速度 $a$ 可以表示为：

$$
a = \frac{F - \mu m g}{m}
$$

假设我们在离散时间 $\Delta t$ 内对其进行积分（欧拉法），已知当前时刻的速度为 $v_t$，那么下一时刻的速度 $v_{t+1}$ 为：

$$
v_{t+1} = v_t + \left(\frac{F_t}{m} - \mu g\right) \Delta t
$$

在这个微型物理系统中，决定其动力学演化的隐藏物理参数可以写成一个向量 $\boldsymbol{\xi} = [m, \mu]^\top$。

设仿真参数为 $\boldsymbol{\xi}_{sim} = [1.0, 0.1]^\top$，真实参数为 $\boldsymbol{\xi}_{real} = [1.5, 0.2]^\top$。同一个 $F_t$ 会得到不同的 $v_{t+1}$；反馈控制可能逐步纠正误差，但基于仿真参数的开环预测已经不再准确。

把带有具体物理参数的系统写成**参数化马尔可夫决策过程**（Parameterized MDP）：$\mathcal{M}_{\boldsymbol{\xi}} = (\mathcal{S}, \mathcal{A}, \mathcal{P}_{\boldsymbol{\xi}}, \mathcal{R}_{\boldsymbol{\xi}}, \gamma, \rho_0)$。这里的状态转移分布 $\mathcal{P}_{\boldsymbol{\xi}}(s_{t+1}|s_t, a_t)$ 显式依赖物理参数 $\boldsymbol{\xi}$。

在标准的强化学习中，我们的目标是寻找一个参数为 $\theta$ 的策略网络 $\pi_\theta(a|s)$，以最大化固定参数下的期望累积回报：

$$
J(\theta; \boldsymbol{\xi}) = \mathbb{E}_{\tau \sim \pi_\theta, \mathcal{P}_{\boldsymbol{\xi}}} \left[ \sum_{t=0}^T \gamma^t \mathcal{R}_{\boldsymbol{\xi}}(s_t, a_t) \right]
$$

而在**域随机化**中，我们将参数 $\boldsymbol{\xi}$ 视为服从某种先验分布 $P_{\Xi}$ 的随机变量。我们的新优化目标 $J_{DR}(\theta)$ 变为了在该分布下的双重期望：

$$
J_{DR}(\theta) = \mathbb{E}_{\boldsymbol{\xi} \sim P_{\Xi}} \left[ \mathbb{E}_{\tau \sim \pi_\theta, \mathcal{P}_{\boldsymbol{\xi}}} \left[ \sum_{t=0}^T \gamma^t \mathcal{R}_{\boldsymbol{\xi}}(s_t, a_t) \right] \right]
$$

这个目标最大化参数分布下的**平均回报**。它鼓励策略覆盖多种物理条件，却不保证每个极端参数都表现良好；若要优化最坏情况，还需要风险敏感或分布鲁棒目标。

## 动力学随机化与方差挑战

<div align="center">
<img src="/figures/08-robot-sim/source/03-domain-randomization/peng-fig1.png" alt="动力学随机化在模拟与真实机器人上覆盖推物和开门任务，直接展示物理参数扰动的迁移目标。" width="86%">

_图 8.3-2：动力学随机化在模拟与真实机器人上覆盖推物和开门任务，直接展示物理参数扰动的迁移目标。 出处：Xue Bin Peng et al.，[Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537)（2018），Figure 1。_
</div>

在真正的机器人控制中，我们通常需要随机化（Dynamics Randomization）的参数远不止质量和摩擦力。一个典型的机器人关节包括电机、减速器和连杆。因此，分布 $P_{\Xi}$ 包含的维度通常高达数十甚至数百维：

1. **刚体属性**：连杆的质量 $m$、惯性张量 $\mathbf{I}$、质心绝对位置。
2. **接触物理**：滑动摩擦力、扭转摩擦力、接触面恢复系数（代表弹性碰撞程度）。
3. **驱动器属性**：电机的转矩限制、电机内部阻尼、底层PD控制器的刚度系数（Stiffness）。
4. **延迟与噪声**：由于通信带来的动作延迟时长 $d$、传感器观测的高斯噪声协方差 $\boldsymbol{\Sigma}$。

当我们在该公式的基础上对策略网络参数 $\theta$ 进行梯度上升时，根据策略梯度定理（Policy Gradient Theorem），其解析梯度形式为：

$$
\nabla_\theta J_{DR}(\theta) = \mathbb{E}_{\boldsymbol{\xi} \sim P_{\Xi}} \left[ \mathbb{E}_{\tau} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}(s_t, a_t; \boldsymbol{\xi}) \right] \right]
$$

与固定环境相比，随机采样 $\boldsymbol{\xi}$ 增加了一层回报差异，因而可能提高梯度估计方差。如果 $P_{\Xi}$ 过宽，批次中会混入大量当前策略无法完成的环境，学习信号可能变得稀疏或不稳定。实践中常逐步扩大范围，或根据真实数据调整参数分布。

## 视觉域随机化与特征不变性

除了上述的物理动力学参数，对于基于摄像头的端到端控制系统，如何跨越**渲染图像与真实相机图像之间的鸿沟**同样至关重要。

令系统内部的三维状态为 $\mathbf{s}_t$（例如机械臂关节状态及目标物块位姿），渲染过程可以抽象为映射函数：

$$
\mathbf{I}_t = f_{render}(\mathbf{s}_t; \boldsymbol{\xi}_{vis})
$$

这里，$\boldsymbol{\xi}_{vis}$ 表示与物体核心几何结构无关的渲染参数，例如光源的位置和光线颜色、摄像机的内参（焦距）与外参（位姿）、背景甚至干扰物的纹理贴图。

视觉域随机化从预先设计的分布中采样 $\boldsymbol{\xi}_{vis}$。若任务标签不随纹理和光照改变，我们希望特征提取器 $f_\phi$ 对这些变化不敏感：

<div align="center">
<img src="/figures/08-robot-sim/source/03-domain-randomization/tobin-fig2.png" alt="同一抓取场景在纹理、光照、相机与干扰物变化下呈现多种渲染外观。" width="86%">

_图 8.3-3：同一抓取场景在纹理、光照、相机与干扰物变化下呈现多种渲染外观。 出处：Josh Tobin et al.，[Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907)（2017），Figure 2。_
</div>

$$
f_\phi\left( f_{render}(\mathbf{s}_t; \boldsymbol{\xi}_{vis}^{(1)}) \right) \approx f_\phi\left( f_{render}(\mathbf{s}_t; \boldsymbol{\xi}_{vis}^{(2)}) \right)
$$

<div align="center">
<img src="/figures/08-robot-sim/latex/03-domain-randomization/visual-nuisance-feature-invariance.png" alt="同一物理状态经过两种渲染扰动产生不同图像，但共享编码器输出应保持接近" width="86%">

_图 8.3-4：两条分支共享同一物理状态，只改变渲染扰动；像素可以不同，但共享编码器提取的任务特征应保持接近。本文根据上式绘制。_
</div>

这个近似等式是期望效果，不是域随机化自动提供的保证。随机化过窄时，网络仍可能利用背景；范围过宽或改变任务相关线索时，也可能损害性能。

## 隐式系统辨识：记忆的贝叶斯推断

在前面的小节中，我们探讨的策略均是无记忆的前馈神经网络 $\pi_\theta(a_t | \mathbf{s}_t)$。然而，仅仅依靠当前时刻的一帧观测来输出动作，往往会使策略变得过于保守（Conservative）。因为网络在时间步 $t$ 并不知道当前环境的滑块质量究竟是 $0.5$ 还是 $1.5$，为了不计算出导致系统崩溃的极端控制力，它只能输出一个妥协的、平均化的微小推力。

为了打破这种性能上的妥协，现代强化学习通常向策略网络的主干中引入循环神经网络（如 LSTM 或 GRU 等记忆模块），将策略的定义域扩展为 $\pi_\theta(a_t | \mathbf{s}_t, \mathbf{h}_{t-1})$，其中 $\mathbf{h}_{t-1}$ 是过去所有观测和动作被高度压缩后的隐状态向量。

> 💡 **物理直觉**：如同一个蒙着眼睛的司机驾驶一辆陌生的汽车，虽然他事先不知道车辆的载重和路面摩擦力，但在踩下第一脚油门并感受到加速度反馈的瞬间，他立刻就在大脑中完成了对这些隐藏物理参数的近似推断。具有记忆的神经网络（如RNN）在随机化环境中的行为正是如此——它利用历史动作和观测序列，隐式地充当了一个卡尔曼滤波器（Kalman Filter）或贝叶斯估计器。

从贝叶斯视角看，历史可以更新对隐藏参数的后验信念：

$$
\hat{P}(\boldsymbol{\xi} | \mathbf{s}_1, a_1, \mathbf{s}_2, a_2, \dots, \mathbf{s}_t)
$$

循环网络不一定显式输出这一概率分布，但其隐状态可以近似携带与隐藏动力学有关的信息，这通常称为**隐式系统辨识**（Implicit System Identification）。适应速度与精度需要由具体任务实验验证。

<div align="center">
<img src="/figures/08-robot-sim/source/03-domain-randomization/rma-fig3.png" alt="RMA 在未见地形、载荷和扰动上测试适应，说明历史响应可支撑在线估计隐藏动力学。" width="86%">

_图 8.3-5：RMA 在未见地形、载荷和扰动上测试适应，说明历史响应可支撑在线估计隐藏动力学。 出处：Ashish Kumar et al.，[RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034)（2021），Figure 3。_
</div>

## 实现一个带域随机化的物理环境

下面实现一个一维滑块。每次 `reset` 都重新采样质量与摩擦系数，然后用半隐式欧拉法更新速度和位置。

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
        self.state = torch.stack([new_pos, new_vel])

        # 定义一个二次型的简单奖励函数：趋向于让位置到达10.0并维持静止
        reward = -((new_pos - 10.0)**2 + 0.1 * new_vel**2)

        return self.state, reward, False
```

向两个随机环境施加同一动作序列，可以直接比较轨迹差异。

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

# 在两个随机环境参数下运行相同控制序列
m1, f1, pos1 = simulate_trajectory(env, action_sequence)
m2, f2, pos2 = simulate_trajectory(env, action_sequence)

print(f"环境 A (质量={m1:.2f}, 摩擦={f1:.2f}) 最终位移: {pos1[-1]:.2f}")
print(f"环境 B (质量={m2:.2f}, 摩擦={f2:.2f}) 最终位移: {pos2[-1]:.2f}")
```

两条轨迹的差异来自采样到的质量与摩擦。训练策略时，这种差异会进入回报和梯度估计；它究竟提高多少方差，需要结合采样数与策略表现测量。

## 小结

- **域随机化（Domain Randomization）** 在训练时采样多组视觉或动力学参数，以降低策略对单一仿真设置的依赖。
- 从数学角度看，域随机化将标准的马尔可夫决策过程拓展为了**参数化马尔可夫决策过程**，迫使策略网络必须在物理参数分布 $\boldsymbol{\xi} \sim P_{\Xi}$ 的双重期望下最大化累积回报。
- **视觉随机化**改变纹理、光照与相机，**动力学随机化**改变质量、摩擦、执行器和延迟；两者都需要保留任务相关信息。
- 历史观测与动作可以帮助循环策略进行**隐式系统辨识**，但适应效果仍取决于数据覆盖和网络容量。
