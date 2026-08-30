# 1.2 什么是世界模型？

在深度学习与强化学习的早期发展中，大多数无模型（Model-Free）算法直接学习从状态到动作的映射（策略）或状态的价值函数。然而，人类和动物的智能显然并非完全依赖于这种反复试错的直接映射。当我们闭上眼睛，我们能够在脑海中预演即将发生的动作及其后果；当我们驾驶汽车时，我们对前车的刹车行为有精确的物理预期。这种在智能体内部构建的、用于模拟和预测外部环境演化规律的数学抽象，便是我们所说的“世界模型”（World Model）。

## 1.2.1 智能与预测的学术脉络

在正式定义世界模型之前，我们必须首先理解其学术渊源。在强化学习领域，使用模型来辅助决策的思想可以追溯到 Richard Sutton 在 1990 年提出的 Dyna 架构 `[Sutton, 1990]`。Dyna 架构首次明确地将强化学习分为两个过程：从真实环境交互中学习，以及从内部模型生成的模拟经验中学习（即规划）。

然而，早期的模型大多只能处理离散的、低维度的状态空间（如简单的网格世界）。随着深度学习的爆发，如何在高维连续空间（如像素级的图像输入）中构建这种模型成为了新的挑战。2018 年，David Ha 和 Jürgen Schmidhuber 发表了具有里程碑意义的论文 *World Models* `[Ha and Schmidhuber, 2018]`。他们提出，可以将高维的视觉输入压缩为低维的隐变量表示，并在隐空间中训练一个循环神经网络（RNN）来预测未来的状态。这一架构不仅极大地提高了样本效率，甚至允许智能体完全在自己“梦境”（即内部生成的世界模型）中训练策略，再将其零样本（Zero-shot）迁移到真实环境中。

## 1.2.2 状态空间与时间演化：从高中的运动学起步

为了严谨地理解世界模型的本质，我们暂且抛开复杂的神经网络，回到高中物理中最基础的运动学。

假设我们在研究一个在光滑水平面上做直线运动的木块。在任意给定的时刻 $t$，我们如何“完全”描述这个木块的客观存在？在经典力学中，我们只需要知道它的位置 $x_t$ 和速度 $v_t$。如果我们定义当前的状态 $s_t$ 为一个包含位置和速度的集合：

$$s_t = \{x_t, v_t\}$$

如果我们在时刻 $t$ 对木块施加一个恒定的力 $F_t$，根据牛顿第二定律，木块会获得加速度 $a_t = F_t / m$（这里我们将加速度 $a_t$ 视为我们采取的动作或控制输入）。经过一段微小的时间 $\Delta t$ 后，木块在下一个时刻 $t+1$ 的状态将如何演化？根据基础的运动学公式，我们有：

$$x_{t+1} = x_t + v_t \Delta t + \frac{1}{2} a_t \Delta t^2$$
$$v_{t+1} = v_t + a_t \Delta t$$

仔细观察 :eqref:eq_kinematics_evolution，你会发现一个深刻的规律：**系统未来的状态 $s_{t+1}$，完全且唯一地由当前状态 $s_t$ 和当前采取的动作 $a_t$ 所决定。** 

我们可以将这种决定性的物理规律抽象为一个通用的数学函数 $f$：

$$s_{t+1} = f(s_t, a_t)$$

在上述语境中，这个函数 $f$（即运动学公式）就是该物理系统的“世界模型”。它完美地模拟了环境的动态（Dynamics）。只要给定初始状态 $s_0$ 和一系列动作序列 $a_0, a_1, \dots$，我们就可以通过反复调用函数 $f$（即前向展开），精准地预测出系统在未来任意时刻的状态 $s_t$。

当我们把这个问题拓展到多维空间时，状态和动作便不再是简单的标量，而是向量（Vectors）。例如，在三维空间中，位置和速度都是三维向量。此时，$s_t \in \mathbb{R}^n$ 表示一个 $n$ 维的状态向量，$a_t \in \mathbb{R}^m$ 表示一个 $m$ 维的动作向量，而世界模型 $f$ 则成为一个高维向量空间之间的映射函数：$f: \mathbb{R}^n \times \mathbb{R}^m \rightarrow \mathbb{R}^n$。

## 1.2.3 概率视角的引入：处理不确定性

在理想的高中物理题中，世界是确定的（Deterministic）。然而，在现实世界或复杂的强化学习环境中，存在着大量的**不确定性（Uncertainty）**：
1. **偶然不确定性（Aleatoric Uncertainty）**：环境中固有的随机性。例如，你掷出一枚骰子，其结果是无法确切预测的。
2. **认知不确定性（Epistemic Uncertainty）**：由于我们对系统的不完全观察或模型自身的缺陷导致的随机性。例如，当你在大雾中驾驶时，由于视野受限，你无法确切知道前方车辆的状态。

为了应对这种不确定性，我们必须摒弃确定性的函数映射 $s_{t+1} = f(s_t, a_t)$，转向**概率生成模型（Probabilistic Generative Models）**。我们将时刻 $t+1$ 的状态 $S_{t+1}$ 视为一个随机变量，并尝试建模其条件概率分布（Conditional Probability Distribution）：

$$P(S_{t+1} = s_{t+1} \mid S_t = s_t, A_t = a_t)$$

这个公式读作：在给定当前状态 $s_t$ 和动作 $a_t$ 的条件下，下一个状态为 $s_{t+1}$ 的概率。

如果我们承认环境满足**马尔可夫性质（Markov Property）**，即未来的状态仅依赖于当前的状态和动作，而与过去的历史轨迹无关，那么上述转移概率便构成了马尔可夫决策过程（MDP）的核心动力学基础。

当状态是连续的高维向量时，上述离散的概率分布将变为连续的概率密度函数（Probability Density Function, PDF），记作 $p(s_{t+1} \mid s_t, a_t)$。此时，我们的**世界模型的任务，就是通过某种方式估计或参数化（Parameterize）这个概率密度函数**。例如，我们可以假设下一个状态服从多元高斯分布（Multivariate Gaussian Distribution）：

$$p(s_{t+1} \mid s_t, a_t) = \mathcal{N}(s_{t+1}; \boldsymbol{\mu}_\theta(s_t, a_t), \boldsymbol{\Sigma}_\theta(s_t, a_t))$$

在这里，均值向量 $\boldsymbol{\mu}_\theta$ 和协方差矩阵 $\boldsymbol{\Sigma}_\theta$ 均是由参数为 $\theta$ 的模型（如神经网络）计算得出的。通过这种方式，世界模型不仅能够预测“下一步最可能发生什么”（均值），还能输出“对预测结果有多大的把握”（方差/协方差）。

## 1.2.4 世界模型的核心组件：Ha & Schmidhuber 架构

在现实的强化学习任务（如自动驾驶或视频游戏）中，智能体通常无法直接获得真实的底层物理状态 $s_t$。它所能获取的只有高维的、含有噪声的观测值（Observation）$o_t$（例如 $64 \times 64 \times 3$ 的 RGB 图像）。这种设定被称为部分可观测马尔可夫决策过程（POMDP）。

直接在极高维度的图像空间 $o_t$ 上预测未来 $o_{t+1}$ 是极其困难且低效的。为了解决这个问题，Ha 和 Schmidhuber `[Ha and Schmidhuber, 2018]` 提出了将世界模型解耦为三个核心组件：**V（视觉模型）、M（记忆/动力学模型）和 C（控制策略）**。

### 1. 视觉模型 V (Vision Model)

视觉模型 $V$ 的任务是将高维的观测图像 $o_t$ 压缩为一个低维的隐变量（Latent Variable）向量 $z_t$。在数学上，这可以视为一个推断过程：

$$z_t \sim q_\phi(z \mid o_t)$$

通常，我们使用变分自编码器（VAE）来实现这一组件。压缩后的隐向量 $z_t$ 滤除了图像中无关紧要的背景噪声，保留了对决策至关重要的核心特征（如物体的位置和姿态）。

### 2. 记忆/动力学模型 M (Memory/Dynamics Model)

由于单帧图像（或单次观测 $z_t$）无法捕捉系统的动态信息（例如，从单张照片中你无法判断物体的速度和运动方向），我们需要引入**时间序列记忆**。记忆模型 $M$ 的任务是基于过去所有的历史观测和动作，预测下一个隐状态 $z_{t+1}$。

为了有效地压缩历史信息，我们使用循环神经网络（RNN）。设 $h_t$ 为 RNN 在时刻 $t$ 的隐藏状态（Hidden State），它聚合了直到时刻 $t$ 的所有历史信息：

$$h_t = \text{RNN}(h_{t-1}, z_t, a_t)$$

此时，记忆模型 $M$ 实际上是在对隐空间的概率分布进行建模：

$$p_\theta(z_{t+1} \mid a_t, z_t, h_t)$$

> **[类比提示]** 
> 我们可以将这一预测过程类比为人类的“做梦”或“心理演练”。在睡眠时，人类的大脑切断了外部视觉输入（即闭上眼睛，没有新的 $o_t$ 产生）。但大脑内部的动力学模型（M 模型）依然在活跃，它利用当前的隐藏状态 $h_t$，结合潜意识产生的动作 $a_t$，自回归地（Autoregressively）生成下一个隐状态 $z_{t+1}$。通过反复将生成的 $z_{t+1}$ 喂给模型自身，大脑可以在毫无真实物理反馈的情况下，于内部“仿真”出连贯的梦境体验。这正是世界模型能够在隐空间内进行零样本策略规划的数学本质。

### 3. 控制器 C (Controller)

控制器 $C$ 就是智能体的策略（Policy）。它的目标是根据当前的世界状态，输出最优的动作 $a_t$ 以最大化累积奖励。在世界模型的框架下，控制器的输入不再是高维的原始图像，而是视觉模型的隐向量 $z_t$ 和记忆模型的历史状态 $h_t$ 的拼接：

$$a_t = \pi_\psi(a_t \mid z_t, h_t)$$

## 1.2.5 代码实现：构建一个极简的世界模型组件

为了让上述数学推导落地，我们将使用 PyTorch 实现一个极简版本的 $V$ 模型和 $M$ 模型的核心前向传播逻辑。在此阶段，我们采用最基础的线性层和 GRU 单元，以展现张量维度在各个阶段的演化。

(**导入必要的深度学习框架。**)

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
```

(**定义视觉模型 V 的编码器部分。**)
在这里，我们将输入的展平图像（例如 $64 \times 64 = 4096$ 维）压缩为一个服从标准正态分布约束的低维隐变量（例如 32 维）。我们输出隐变量分布的均值和对数方差。

```{.python .input}
#@tab pytorch
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
        # 输出隐变量的对数方差（使用对数是为了保证方差计算时的非负性）
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

(**定义记忆/动力学模型 M。**)
M 模型接收当前的隐变量 $z_t$ 和动作 $a_t$，并结合自身的隐藏状态 $h_t$（在这里通过 GRU 维护），来预测下一个时刻隐变量分布的均值和对数方差。这正是对前文 RNN 记忆更新公式和高斯动力学公式的直接代码翻译。

```{.python .input}
#@tab pytorch
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

通过这两段极简的代码，我们可以清晰地看到数据流转的严谨闭环：环境产生的 $o_t$ 被 `VisionModel` 降维并提取为 $z_t$ 的概率分布；采样得到的 $z_t$ 与策略生成的 $a_t$ 一同输入给 `DynamicsModel`，不断推进系统内部的时序隐藏状态 $h_t$，并不断产生对未来 $z_{t+1}$ 的预测概率。

## 1.2.6 小结

在本节中，我们探讨了世界模型的核心定义及其数学表达：
*   **智能的本质之一在于预测**。世界模型是智能体内部构建的环境模拟器。
*   从确定性的经典运动学方程 $s_{t+1} = f(s_t, a_t)$ 出发，我们逐步将其推广至应对不确定性的概率生成模型 $p(s_{t+1} \mid s_t, a_t)$。
*   为了处理极高维的观测空间和部分可观测问题，经典的世界模型（如 Ha & Schmidhuber 架构）被严谨地拆分为视觉模型（压缩空间）、记忆模型（捕捉时序动态）和控制器（输出策略）三个核心模块。

## 1.2.7 练习

1. 在 :eqref:eq_gaussian_dynamics 中，我们假设了转移概率服从多元高斯分布。如果环境中存在高度多模态（Multi-modal）的不确定性（例如：在岔路口，你要么向左转，要么向右转，但绝对不可能直行撞墙），单一的高斯分布还能很好地建模这种动态吗？为什么？
    *提示：思考高斯分布的单峰性质。对于多模态问题，混合密度网络（MDN）可能是更好的选择。*
2. 假设你的观测 $o_t$ 仅仅是当前汽车的速度表读数，而你的真实任务是在赛道上驾驶。仅凭当前时刻的 $o_t$，你能推断出汽车在赛道上的具体位置吗？这解释了为什么记忆模型（维持一个随时间累积的 $h_t$）对于 POMDP 是至关重要的？
    *提示：根据牛顿定律，速度只是位置的一阶导数。积分（即累加历史信息）是重构位置（底层真实状态）的必要数学手段。*
3. 在我们的代码实现中，`DynamicsModel` 输出的是对数方差（`logvar_next`）而不是方差或标准差本身。从神经网络优化的数学角度来看，这样做有什么好处？
    *提示：方差必须严格大于 0。如果不使用对数，你需要使用什么激活函数来保证这一点？这又会带来哪些数值稳定性的问题？*
