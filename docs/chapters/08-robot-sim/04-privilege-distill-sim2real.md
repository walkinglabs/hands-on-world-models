# 8.4 特权信息蒸馏与虚实迁移（Sim2Real）

仿真器知道地面摩擦系数、连杆质量和精确物体位姿，真实机器人通常不知道。训练时若直接把这些量交给策略，部署时输入就会缺失；完全不用它们，又放弃了仿真器免费提供的监督。这正是**特权信息**方法要解决的接口差异。

<div align="center">
<img src="/figures/08-robot-sim/source/04-privilege-distill-sim2real/rma-fig1.png" alt="RMA 让四足机器人在岩石、沙地、泥地与负载变化中快速调整运动策略。" width="86%">

_图 8.4-1：RMA 让四足机器人在岩石、沙地、泥地与负载变化中快速调整运动策略。 出处：Ashish Kumar et al.，[RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034)（2021），Figure 1。_
</div>

仿真器不能完整复制真实系统中的执行器延迟、摩擦变化和装配误差。Tobin 等人的域随机化工作在视觉 Sim-to-Real 中随机改变纹理、光照、相机和物体属性，使真实图像有机会落入训练分布 [[Tobin et al., 2017]](https://arxiv.org/abs/1703.06907)。这篇论文主要讨论视觉随机化，不能单独支撑“物理参数随机化必然使策略保守”的普遍结论。

仿真训练可以使用部署时不可得的**特权信息**，但不同论文使用它的方式并不相同。Pinto 等人的不对称 actor–critic 让 critic 读取仿真完整状态，而 actor 只读取可部署的 RGB-D 观测 [[Pinto et al., 2017]](https://arxiv.org/abs/1710.06542)。Lee 等人的四足运动控制器用特权学习构造训练目标，部署策略则根据本体感受历史行动 [[Lee et al., 2020]](https://arxiv.org/abs/2010.11251)。RMA 由基础策略与适应模块组成，训练时利用环境参数，部署时由近期状态—动作历史推断环境表征 [[Kumar et al., 2021]](https://arxiv.org/abs/2107.04034)。因此，“critic 使用特权状态”“教师—学生蒸馏”和“在线适应模块”应当分别表述。

本节先区分三种用法：critic 读取完整状态、教师策略指导学生、适应模块从历史推断环境表征。随后用一个接近 RMA 的教学模型说明第三种用法。

<div align="center">
<img src="/figures/08-robot-sim/source/04-privilege-distill-sim2real/lbc-fig1.png" alt="Learning by Cheating 先用特权俯视状态训练教师，再把知识蒸馏到仅看相机的学生策略。" width="86%">

_图 8.4-2：Learning by Cheating 先用特权俯视状态训练教师，再把知识蒸馏到仅看相机的学生策略。 出处：Dian Chen et al.，[Learning by Cheating](https://arxiv.org/abs/1912.12294)（2020），Figure 1。_
</div>

## 8.4.1 仿真与现实的数学鸿沟

为了严谨地描述 Sim2Real 问题，让我们回顾高中物理中最基础的滑动摩擦力公式。假设一个机器人的足端在地面上滑动，其受到的摩擦力大小 $f$ 为：

$$f = \mu N$$

其中，$\mu$ 是滑动摩擦系数，$N$ 是法向力。仿真中常用少量参数描述摩擦，现实中的有效摩擦却会随材料、表面状态、载荷与速度变化。即使名义系数相同，接触模型和执行器误差也会改变状态转移。

在强化学习的马尔可夫决策过程（MDP）框架下，这意味着仿真环境的转移概率分布 $P_{sim}(s_{t+1} | s_t, a_t)$ 与真实环境的转移概率分布 $P_{real}(s_{t+1} | s_t, a_t)$ 存在不可忽视的差异。我们定义系统的完整物理参数集合为 $\mathbf{e} \in \mathcal{E}$，其中包括了摩擦系数、质量、质心偏移、电机延迟等所有影响系统动力学的根本因素。

部署时，环境参数 $\mathbf{e}$ 往往不可直接观测。传感器只提供观测 $\mathbf{o}_t$，于是策略面对的是部分可观测问题；历史中的“施加了什么动作、系统怎样响应”可以帮助推断隐藏动力学。

## 8.4.2 域随机化与特权马尔可夫决策

如果我们想要让机器人在未知的真实参数 $\mathbf{e}^*$ 下也能正常工作，最直观的思路是在训练期间，从一个广泛的物理参数分布 $p(\mathbf{e})$ 中进行采样。我们的优化目标随之变为最大化参数分布下的期望回报：

$$J(\pi) = \mathbb{E}_{\mathbf{e} \sim p(\mathbf{e}), \tau \sim P_{sim}(\cdot|\mathbf{e}), \pi} \left[ \sum_{t=0}^T \gamma^t r(s_t, a_t) \right]$$

只依赖瞬时观测的策略无法区分“相同姿态、不同摩擦”这类情况，因而可能学到折中动作。是否真的变得保守取决于任务与训练分布，不能由公式单独推出。

训练时可以把隐藏参数作为**特权信息（Privileged Information）**。最直接的教学构造是令 $\mathbf{x}_t=[\mathbf{o}_t,\mathbf{e}_t]$，并让教师或 critic 读取它；部署策略仍只能使用可观测输入。

<div align="center">
<img src="/figures/08-robot-sim/source/04-privilege-distill-sim2real/anymal-fig1.png" alt="ANYmal 的特权学习流程最终在真实四足机器人上执行只依赖可部署传感器的运动技能。" width="86%">

_图 8.4-3：ANYmal 的特权学习流程最终在真实四足机器人上执行只依赖可部署传感器的运动技能。 出处：Joonho Lee et al.，[Learning Agile and Dynamic Motor Skills for Legged Robots](https://arxiv.org/abs/1901.08652)（2020），Figure 1。_
</div>

若 $\mathbf{x}_t$ 包含决策所需的完整状态，教师面对的问题更接近完全可观测 MDP。不过，传入环境参数并不自动保证最优动作，教师仍需要通过强化学习或控制优化获得。

在演员-评论家（Actor-Critic）架构下，其价值函数 $V_T(\mathbf{x}_t)$ 的贝尔曼方程可以被严谨地表示为：

$$V_T(\mathbf{x}_t) = \mathbb{E}_{a_t \sim \pi_T, \mathbf{x}_{t+1}} \left[ r(\mathbf{x}_t, a_t) + \gamma V_T(\mathbf{x}_{t+1}) \right]$$

## 8.4.3 两阶段特权蒸馏架构

现在假设教师已经训练完成，但现实机器人无法获取 $\mathbf{e}_t$。适应模块改为读取近期观测—动作历史
$\mathbf{h}_t=\{(\mathbf{o}_{t-k},\mathbf{a}_{t-k}),\dots,(\mathbf{o}_{t-1},\mathbf{a}_{t-1}),\mathbf{o}_t\}$。

历史提供了输入与响应的对应关系。例如，连续几步施加相同扭矩而角加速度偏小，可能说明负载更大，也可能来自摩擦或执行器偏差。历史通常只能支持近似推断，多个隐藏原因仍可能产生相似观测。

RMA（Rapid Motor Adaptation）把训练分成基础策略学习与适应模块学习两个阶段 [[Kumar et al., 2021]](https://arxiv.org/abs/2107.04034)。下面保留这一核心接口，但网络尺寸是教学设定，不是原论文配置的逐项复现。

<div align="center">
<img src="/figures/08-robot-sim/source/04-privilege-distill-sim2real/rma-fig2.png" alt="RMA 的两阶段图把特权环境编码器、基础策略和部署时历史适应模块分开。" width="86%">

_图 8.4-4：RMA 的两阶段图把特权环境编码器、基础策略和部署时历史适应模块分开。 出处：Ashish Kumar et al.，[RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034)（2021），Figure 2。_
</div>

**第一阶段：教师网络与环境编码（强化学习阶段）**
在这个阶段，我们训练一个环境编码器（Environment Encoder） $E_\phi$，将高维、繁杂的物理特权信息 $\mathbf{e}_t \in \mathbb{R}^{d_e}$ 压缩映射为一个低维的隐变量（Latent Variable） $\mathbf{z}_t \in \mathbb{R}^{d_z}$：

$$\mathbf{z}_t = E_\phi(\mathbf{e}_t)$$

随后，教师策略网络根据当前的本体观测和隐变量输出动作分布：$a_t \sim \pi_{\theta_T}(\cdot | \mathbf{o}_t, \mathbf{z}_t)$。在这个全仿真阶段，我们将 $E_\phi$ 和 $\pi_{\theta_T}$ 联合起来进行端到端的强化学习训练，最大化该公式。

**第二阶段：历史推断与适应（监督蒸馏阶段）**
第二阶段冻结基础策略与环境编码器。适应网络 $A_\psi$ 处理长度为 $K$ 的观测—动作历史，教学代码把每步拼成 $d_o+d_a$ 维，因此输入形状为 $\mathbf{h}_t\in\mathbb{R}^{K\times(d_o+d_a)}$：

$$\hat{\mathbf{z}}_t = A_\psi(\mathbf{h}_t)$$

<div align="center">
<img src="/figures/08-robot-sim/latex/04-privilege-distill-sim2real/history-conv1d-shape-chain.png" alt="历史观测先交换时间与特征轴，两次一维卷积缩短时间长度后展平为物理隐变量" width="86%">

_图 8.4-5：历史张量先把观测维变成 Conv1d 通道；两次有效卷积令时间长度从 K 变为 K−4，展平后再输出 d_z 维估计。本文根据本节张量过程绘制。_
</div>

仿真中可以用冻结的 $E_\phi$ 产生目标 $\mathbf{z}_t$，再最小化预测隐变量与目标之间的均方误差：

$$\mathcal{L}(\psi) = \mathbb{E}_{\tau \sim \mathcal{D}} \left[ \frac{1}{2} \| \mathbf{z}_t - \hat{\mathbf{z}}_t \|_2^2 \right]$$

一旦适应网络的训练收敛，在物理机器人进行实际部署时，我们摒弃需要特权信息的 $E_\phi$。我们在每个控制周期计算推断出的隐变量 $\hat{\mathbf{z}}_t = A_\psi(\mathbf{h}_t)$，将其与当前的瞬时观测 $\mathbf{o}_t$ 进行张量拼接后，直接输入到冻结的教师策略网络中执行动作：$a_t \sim \pi_{\theta_T}(\cdot | \mathbf{o}_t, \hat{\mathbf{z}}_t)$。

这种分解为历史推断提供了明确监督，但不保证 $\mathbf{z}_t$ 与某个单一物理量一一对应。部署速度取决于历史长度、控制频率与网络推理开销。

## 8.4.4 代码实现与张量维度分析

下面用 PyTorch 构建环境编码器、基础策略和适应网络。特权向量 `e_t` 先被压缩为 `z_t`。

```python
import torch
from torch import nn
from torch.nn import functional as F

class EnvironmentEncoder(nn.Module):
    """提取特权信息的环境编码器"""
    def __init__(self, e_dim, z_dim):
        super().__init__()
        # 将高维特权信息（摩擦力分布、精确质量等）压缩至 z_dim 维度的紧凑隐变量
        self.net = nn.Sequential(
            nn.Linear(e_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, z_dim)
        )

    def forward(self, e_t):
        # e_t 张量形状: (batch_size, e_dim)
        # 返回 z_t 张量形状: (batch_size, z_dim)
        return self.net(e_t)

class TeacherPolicy(nn.Module):
    """基于隐特征驱动的教师策略网络"""
    def __init__(self, o_dim, z_dim, a_dim):
        super().__init__()
        # 策略网络的输入是当前观测向量 o_t 和隐变量 z_t 的拼接张量
        self.net = nn.Sequential(
            nn.Linear(o_dim + z_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, a_dim)
        )

    def forward(self, o_t, z_t):
        # o_t 张量形状: (batch_size, o_dim)
        # z_t 张量形状: (batch_size, z_dim)
        # 在特征维度 (dim=1) 上进行张量拼接
        x = torch.cat([o_t, z_t], dim=1)
        # 返回动作预测均值，张量形状: (batch_size, a_dim)
        return self.net(x)
```

假设上述网络已经通过 PPO (Proximal Policy Optimization) 算法在包含大量域随机化的环境中训练至收敛，并被冻结参数。接下来，我们需要实现处理历史时间序列的适应网络。

(**定义用于处理历史序列观测的适应网络**)。由于输入数据不仅包含特征维度，还包含显式的时间序列维度，我们通常采用一维卷积网络（1D CNN）或者时序卷积网络（TCN）来提取局部的时序相关性，从而准确预测当前的隐变量。

```python
class AdaptationNetwork(nn.Module):
    """根据观测—动作历史推断环境隐变量。"""
    def __init__(self, history_dim, hist_len, z_dim):
        super().__init__()
        self.hist_len = hist_len
        # 使用一维卷积提取时序动力学特征
        # 注意：PyTorch 中的 Conv1d 期待输入形状为 (batch_size, channels, sequence_length)
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=history_dim, out_channels=32, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ELU()
        )

        # 严谨计算经过两次核大小为 3，步长为 1 且无填充的卷积后，序列长度的变化
        # 每次卷积导致序列长度减 2，因此最终长度为 hist_len - 4
        flat_size = 32 * (hist_len - 4)

        self.linear_net = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ELU(),
            nn.Linear(128, z_dim)
        )

    def forward(self, h_t):
        # h_t: (batch_size, hist_len, history_dim)
        # Conv1d 需要 (batch_size, history_dim, hist_len)
        x = h_t.transpose(1, 2)
        x = self.conv_net(x)
        # 将局部时序特征展平，形状变为: (batch_size, flat_size)
        x = x.view(x.size(0), -1)
        # 输出预测隐变量 z_hat_t，形状为: (batch_size, z_dim)
        return self.linear_net(x)
```

最后构造一个批次，执行一次适应网络更新。

```python
# 初始化张量维度超参数
batch_size = 64
o_dim = 42       # 本体观测维度（如各关节位置与速度）
e_dim = 16       # 特权物理参数维度
z_dim = 8        # 压缩后的物理隐变量维度
a_dim = 12       # 动作输出维度
hist_len = 20    # 回溯的历史控制周期数

# 实例化整个架构的组件
env_encoder = EnvironmentEncoder(e_dim, z_dim)
teacher_policy = TeacherPolicy(o_dim, z_dim, a_dim)
history_dim = o_dim + a_dim
adaptation_net = AdaptationNetwork(history_dim, hist_len, z_dim)

# 第一阶段结束，强制锁定教师网络和编码器的梯度
env_encoder.eval()
teacher_policy.eval()
for module in (env_encoder, teacher_policy):
    for parameter in module.parameters():
        parameter.requires_grad_(False)

# 从重放缓冲区（Replay Buffer）中随机采样一个批次的轨迹状态
o_t = torch.randn(batch_size, o_dim)
e_t = torch.randn(batch_size, e_dim)
h_t = torch.randn(batch_size, hist_len, history_dim)

# 定义针对适应网络的优化器
optimizer = torch.optim.Adam(adaptation_net.parameters(), lr=1e-3)

# ================== 监督蒸馏优化步 ==================
# 1. 教师视角：利用理想的特权信息 e_t，无梯度计算真实的标签隐变量 z_t
with torch.no_grad():
    z_t = env_encoder(e_t)

# 2. 适应模块利用观测—动作历史预测 z_hat_t
z_hat_t = adaptation_net(h_t)

# 3. 损失计算：计算 z_hat_t 与 z_t 之间的均方误差 (MSE)
loss = F.mse_loss(z_hat_t, z_t)

# 4. 梯度更新：仅通过反向传播更新适应网络 A_psi 的权重
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Distillation Loss: {loss.item():.4f}")
print(f"Target z_t shape: {z_t.shape}")
print(f"Predicted z_hat_t shape: {z_hat_t.shape}")
```

`AdaptationNetwork` 输出的是能够复现教师隐变量的向量，不应直接解释为摩擦系数估计。若低摩擦导致历史响应发生稳定变化，网络可能利用这一信号调整策略输入；是否足以维持稳定，需要真实部署实验验证。

## 8.4.5 小结与讨论

- 特权信息只能在训练时使用；部署路径必须只依赖真实机器人可获得的观测与历史。
- 不对称 actor–critic、教师—学生蒸馏和 RMA 式适应模块使用特权信息的方式不同，引用时需要区分。
- RMA 式实现先学习环境编码与基础策略，再用观测—动作历史拟合环境隐变量。
- 隐变量是任务驱动的表征，不一定等同于可解释的质量或摩擦参数。
