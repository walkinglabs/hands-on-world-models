# 1.2 什么是世界模型？

无模型（Model-Free）算法直接学习策略或价值函数，不要求显式预测环境如何变化。世界模型走另一条路线：它学习“当前状态与动作会怎样影响未来”，再把预测用于规划、数据生成或策略学习。例如，前车亮起刹车灯后，模型可以比较继续加速与立即减速各自可能产生的后果。

## 1.2.1 智能与预测的学术脉络

在正式定义世界模型之前，我们先看它在强化学习中的一条学术脉络。Richard Sutton 于 1990 年提出的 Dyna 架构把直接强化学习、模型学习与规划放在同一个系统中：智能体既从真实交互更新，也从学到的模型生成的模拟经验更新 [[Sutton, 1990]](https://dl.acm.org/doi/10.5555/645530.658292)。

<div align="center">
  <img src="/figures/01-why-world-models/source/02-what-is-a-world-model/dyna-fig1.png" alt="Dyna 总览把评价函数、策略与真实世界或世界模型连接成学习和规划闭环。" width="86%">

_图 1.2-1：Dyna 总览把评价函数、策略与真实世界或世界模型连接成学习和规划闭环。 出处：Richard S. Sutton，[Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](https://dl.acm.org/doi/10.5555/645530.658292)（1990），Figure 1。_

</div>

早期许多模型式方法在离散、低维状态空间中验证。2018 年，David Ha 和 Jürgen Schmidhuber 的 _World Models_ 将高维视觉输入压缩为低维隐变量，并用循环网络预测隐变量与终止信号 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。论文还展示了一个更具体的结果：在 VizDoom 实验中，控制器可以只在学到的模型生成的轨迹中训练，再直接放回真实游戏环境测试；这证明了“在模型中训练”可行，但并不等于对任意环境都能零样本迁移。

<div align="center">
  <img src="/figures/01-why-world-models/source/02-what-is-a-world-model/wm-fig16.png" alt="World Models 的 VizDoom 运行帧展示在梦境模型中训练的控制器回到真实游戏环境后的行为。" width="86%">

_图 1.2-2：World Models 的 VizDoom 运行帧展示在梦境模型中训练的控制器回到真实游戏环境后的行为。 出处：David Ha; Jürgen Schmidhuber，[World Models](https://arxiv.org/abs/1803.10122)（2018），Figure 16。_

</div>

## 1.2.2 状态空间与时间演化：从高中的运动学起步

为了严谨地理解世界模型的本质，我们暂且抛开复杂的神经网络，回到高中物理中最基础的运动学。

假设我们在研究一个在光滑水平面上做直线运动的木块。在任意给定的时刻 $t$，我们如何“完全”描述这个木块的客观存在？在经典力学中，我们只需要知道它的位置 $x_t$ 和速度 $v_t$。如果我们定义当前的状态 $s_t$ 为一个包含位置和速度的集合：

$$s_t = \{x_t, v_t\}$$

如果我们在时刻 $t$ 对木块施加一个恒定的力 $F_t$，根据牛顿第二定律，木块会获得加速度 $a_t = F_t / m$（这里我们将加速度 $a_t$ 视为我们采取的动作或控制输入）。经过一段微小的时间 $\Delta t$ 后，木块在下一个时刻 $t+1$ 的状态将如何演化？根据基础的运动学公式，我们有：

$$x_{t+1} = x_t + v_t \Delta t + \frac{1}{2} a_t \Delta t^2$$
$$v_{t+1} = v_t + a_t \Delta t$$

在质量、受力和时间间隔均已知的理想假设下，$s_{t+1}$ 由当前状态 $s_t$ 和动作 $a_t$ 决定。

我们可以将这种决定性的物理规律抽象为一个通用的数学函数 $f$：

$$s_{t+1} = f(s_t, a_t)$$

在这个简化系统中，函数 $f$ 就是一份已知的动力学模型。给定初始状态和动作序列，可以反复调用 $f$ 进行前向展开。若模型只近似真实环境，滚动得越远，模型误差通常积累得越明显。

当我们把这个问题拓展到多维空间时，状态和动作便不再是简单的标量，而是向量（Vectors）。例如，在三维空间中，位置和速度都是三维向量。此时，$s_t \in \mathbb{R}^n$ 表示一个 $n$ 维的状态向量，$a_t \in \mathbb{R}^m$ 表示一个 $m$ 维的动作向量，而世界模型 $f$ 则成为一个高维向量空间之间的映射函数：$f: \mathbb{R}^n \times \mathbb{R}^m \rightarrow \mathbb{R}^n$。

## 1.2.3 概率视角的引入：处理不确定性

在理想的高中物理题中，世界是确定的（Deterministic）。然而，在现实世界或复杂的强化学习环境中，存在着大量的**不确定性（Uncertainty）**：

1. **偶然不确定性（Aleatoric Uncertainty）**：环境中固有的随机性。例如，你掷出一枚骰子，其结果是无法确切预测的。
2. **认知不确定性（Epistemic Uncertainty）**：来自数据不足、观测不完整或模型知识有限。例如，大雾会使模型难以确定前车状态。这类不确定性反映的是模型所知有限，不是环境本身必然随机。

确定性映射适合单一、可预测的转移；需要表达多种可能未来时，可以把它推广为**概率生成模型（Probabilistic Generative Model）**。我们将时刻 $t+1$ 的状态 $S_{t+1}$ 视为随机变量，并建模其条件概率分布：

$$P(S_{t+1} = s_{t+1} \mid S_t = s_t, A_t = a_t)$$

这个公式读作：在给定当前状态 $s_t$ 和动作 $a_t$ 的条件下，下一个状态为 $s_{t+1}$ 的概率。

如果我们承认环境满足**马尔可夫性质（Markov Property）**，即未来的状态仅依赖于当前的状态和动作，而与过去的历史轨迹无关，那么上述转移概率便构成了马尔可夫决策过程（MDP）的核心动力学基础。

当状态是连续的高维向量时，上述离散的概率分布将变为连续的概率密度函数（Probability Density Function, PDF），记作 $p(s_{t+1} \mid s_t, a_t)$。此时，我们的**世界模型的任务，就是通过某种方式估计或参数化（Parameterize）这个概率密度函数**。例如，我们可以假设下一个状态服从多元高斯分布（Multivariate Gaussian Distribution）：

$$p(s_{t+1} \mid s_t, a_t) = \mathcal{N}(s_{t+1}; \boldsymbol{\mu}_\theta(s_t, a_t), \boldsymbol{\Sigma}_\theta(s_t, a_t))$$

<div align="center"><img src="/figures/01-why-world-models/latex/02-what-is-a-world-model/gaussian-conditional-uncertainty.png" alt="状态动作条件共同产生高斯均值和协方差，定义下一状态分布" width="86%">

_图 1.2-3：模型从同一组状态—动作条件输出均值与协方差；前者给出预测中心，后者给出扩散尺度与相关方向。本文根据上式绘制。_

</div>

在这里，均值向量 $\boldsymbol{\mu}_\theta$ 和协方差矩阵 $\boldsymbol{\Sigma}_\theta$ 均是由参数为 $\theta$ 的模型（如神经网络）计算得出的。通过这种方式，世界模型不仅能够预测“下一步最可能发生什么”（均值），还能输出“对预测结果有多大的把握”（方差/协方差）。

## 1.2.4 世界模型的核心组件：Ha & Schmidhuber 架构

<div align="center">
  <img src="/figures/01-why-world-models/source/02-what-is-a-world-model/dreamer-fig3.png" alt="Dreamer 原论文并列展示从经验学习动力学、在想象中学习行为与回到环境执行的三阶段系统。" width="86%">

_图 1.2-4：Dreamer 原论文并列展示从经验学习动力学、在想象中学习行为与回到环境执行的三阶段系统。 出处：Danijar Hafner et al.，[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)（2020），Figure 3。_

</div>

在现实的强化学习任务（如自动驾驶或视频游戏）中，智能体通常无法直接获得真实的底层物理状态 $s_t$。它所能获取的只有高维的、含有噪声的观测值（Observation）$o_t$（例如 $64 \times 64 \times 3$ 的 RGB 图像）。这种设定被称为部分可观测马尔可夫决策过程（POMDP）。

直接在极高维度的图像空间 $o_t$ 上预测未来 $o_{t+1}$ 是极其困难且低效的。为了解决这个问题，Ha 和 Schmidhuber [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122) 提出了将世界模型解耦为三个核心组件：**V（视觉模型）、M（记忆/动力学模型）和 C（控制策略）**。

### 1. 视觉模型 V (Vision Model)

视觉模型 $V$ 的任务是将高维的观测图像 $o_t$ 压缩为一个低维的隐变量（Latent Variable）向量 $z_t$。在数学上，这可以视为一个推断过程：

$$z_t \sim q_\phi(z \mid o_t)$$

经典 _World Models_ 使用变分自编码器（VAE）实现这一组件。训练目标鼓励 $z_t$ 用较少维度重建画面，但并不自动保证所有维度都有可解释语义，也可能丢失对控制有用的细节。

### 2. 记忆/动力学模型 M (Memory/Dynamics Model)

由于单帧图像（或单次观测 $z_t$）无法捕捉系统的动态信息（例如，从单张照片中你无法判断物体的速度和运动方向），我们需要引入**时间序列记忆**。记忆模型 $M$ 的任务是基于过去所有的历史观测和动作，预测下一个隐状态 $z_{t+1}$。

为了有效地压缩历史信息，我们使用循环神经网络（RNN）。设 $h_t$ 为 RNN 在时刻 $t$ 的隐藏状态（Hidden State），它聚合了直到时刻 $t$ 的所有历史信息：

$$h_t = \text{RNN}(h_{t-1}, z_t, a_t)$$

此时，记忆模型 $M$ 实际上是在对隐空间的概率分布进行建模：

$$p_\theta(z_{t+1} \mid a_t, z_t, h_t)$$

当真实观测暂时不可用时，可以把模型预测的 $z_{t+1}$ 继续作为下一步输入，形成自回归展开。这样得到的是模型内部的一条候选未来轨迹；它能否用于规划，取决于动力学模型在相关状态与动作上的准确性。

### 3. 控制器 C (Controller)

控制器 $C$ 就是智能体的策略（Policy）。它根据当前内部状态产生动作，并通过训练提高累积奖励。在世界模型的框架下，控制器的输入不再是高维原始图像，而是视觉模型的隐向量 $z_t$ 和记忆模型的历史状态 $h_t$：

$$a_t \sim \pi_\psi(\cdot \mid z_t, h_t)$$

## 1.2.5 代码实现：构建一个极简的世界模型组件

为了让上述数学推导落地，我们将使用 PyTorch 实现一个极简版本的 $V$ 模型和 $M$ 模型的核心前向传播逻辑。在此阶段，我们采用最基础的线性层和 GRU 单元，以展现张量维度在各个阶段的演化。

先导入必要的深度学习框架。

```python
import torch
import torch.nn as nn
```

下面定义视觉模型 V 的编码器部分。
在这里，我们将输入的展平图像（例如 $64 \times 64 = 4096$ 维）压缩为一个服从标准正态分布约束的低维隐变量（例如 32 维）。我们输出隐变量分布的均值和对数方差。

```python
class VisionModel(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=32):
        super(VisionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # 输出隐变量的均值
        self.fc_mu = nn.Linear(128, latent_dim)
        # 输出无约束的对数方差；取 exp(logvar) 后方差为正
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, o_t):
        """
        o_t: 观测输入，形状为 (batch_size, input_dim)
        返回均值和对数方差，形状均为 (batch_size, latent_dim)
        """
        hidden = self.encoder(o_t)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
```

接着定义记忆/动力学模型 M。
M 模型接收当前的隐变量 $z_t$ 和动作 $a_t$，并结合自身的隐藏状态 $h_t$（在这里通过 GRU 维护），来预测下一个时刻隐变量分布的均值和对数方差。这正是对前文 RNN 记忆更新公式和高斯动力学公式的直接代码翻译。

```python
class DynamicsModel(nn.Module):
    def __init__(self, latent_dim=32, action_dim=2, hidden_dim=128):
        super(DynamicsModel, self).__init__()
        self.hidden_dim = hidden_dim
        # RNN 核心组件，这里我们使用 GRU (Gated Recurrent Unit)
        # 输入维度是 z_t 和 a_t 的拼接：latent_dim + action_dim
        self.rnn = nn.GRUCell(input_size=latent_dim + action_dim, hidden_size=hidden_dim)

        # 将 RNN 的隐藏状态映射到下一个隐变量状态 z_{t+1} 的分布参数
        self.fc_mu_next = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar_next = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t, a_t, h_t):
        """
        z_t: 当前隐状态，形状 (batch_size, latent_dim)
        a_t: 当前动作，形状 (batch_size, action_dim)
        h_t: RNN 隐藏状态，形状 (batch_size, hidden_dim)
        """
        # [拼接隐变量与动作] 构成 RNN 的输入，形状为 (batch_size, latent_dim + action_dim)
        rnn_input = torch.cat([z_t, a_t], dim=-1)

        # 更新隐藏状态 h_{t+1}，对应公式：h_{t+1} = RNN(h_t, z_t, a_t)
        h_next = self.rnn(rnn_input, h_t)

        # 根据更新后的隐藏状态预测 z_{t+1} 的分布
        mu_next = self.fc_mu_next(h_next)
        logvar_next = self.fc_logvar_next(h_next)

        return mu_next, logvar_next, h_next
```

这两段代码展示了接口和张量形状：`VisionModel` 输出 $z_t$ 的分布参数，采样得到的 $z_t$ 与动作 $a_t$ 进入 `DynamicsModel`，后者更新隐藏状态并输出下一潜变量的分布参数。完整训练还需要重参数化采样、重建损失和动力学似然等步骤。

## 1.2.6 小结

在本节中，我们探讨了世界模型的核心定义及其数学表达：

- 世界模型学习状态、动作与未来之间的关系，可用于预测、规划或策略学习。
- 从确定性的经典运动学方程 $s_{t+1} = f(s_t, a_t)$ 出发，我们逐步将其推广至应对不确定性的概率生成模型 $p(s_{t+1} \mid s_t, a_t)$。
- 为了处理极高维的观测空间和部分可观测问题，经典的世界模型（如 Ha & Schmidhuber 架构）被严谨地拆分为视觉模型（压缩空间）、记忆模型（捕捉时序动态）和控制器（输出策略）三个核心模块。

## 1.2.7 练习

1. 若转移具有多个分离的可能结果，单一高斯分布还能否恰当地建模？为什么？
   _提示：思考高斯分布的单峰性质。对于多模态问题，混合密度网络（MDN）可能是更好的选择。_
2. 假设观测 $o_t$ 只有当前汽车的速度表读数，而任务是在赛道上驾驶。仅凭这一观测能否推断汽车位置？这如何说明记忆模型对 POMDP 的作用？
   _提示：根据牛顿定律，速度只是位置的一阶导数。积分（即累加历史信息）是重构位置（底层真实状态）的必要数学手段。_
3. 在我们的代码实现中，`DynamicsModel` 输出的是对数方差（`logvar_next`）而不是方差或标准差本身。从神经网络优化的数学角度来看，这样做有什么好处？
   _提示：方差必须严格大于 0。如果不使用对数，你需要使用什么激活函数来保证这一点？这又会带来哪些数值稳定性的问题？_
