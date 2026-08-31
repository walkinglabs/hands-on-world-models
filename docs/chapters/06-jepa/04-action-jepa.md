# 6.4 动作条件的 JEPA（Action-conditional JEPA）

LeCun 在 _A Path Towards Autonomous Machine Intelligence_ 中提出了联合嵌入预测架构（Joint-Embedding Predictive Architecture, JEPA）的总体设想：用上下文表征预测目标表征，而不是要求模型重构原始输入的每个细节 [[LeCun, 2022]](https://openreview.net/forum?id=BZ5a1r-kVsf)。这是架构原则，不保证学到的每个特征都自动具有高阶语义。

本节在这一原则上加入**动作（Action）**：预测器除了读取当前表征，还读取动作，用来预测动作条件下的下一时刻表征。这里的“Action-conditional JEPA”是教学性的组合设计，并不声称复现 LeCun 文章中某个固定模型，也不能仅凭预测损失就推出它已经具备反事实规划能力。

<div align="center">
  <img src="/figures/06-jepa/source/04-action-jepa/muzero-fig1.png" alt="MuZero 把动作送入潜在动力学并递归展开奖励、价值与策略预测，是动作条件隐空间演化的直接前身。" width="86%">

_图 6.4-1：MuZero 把动作送入潜在动力学并递归展开奖励、价值与策略预测，是动作条件隐空间演化的直接前身。 出处：Julian Schrittwieser et al.，[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)（2020），Figure 1。_

</div>

本节从基础运动学出发构造一个最小模型，并分析表征坍塌为何仍可能出现。代码用于说明张量与梯度路径，不代表经过真实机器人基准验证的工业实现。

## 6.4.1 从基础运动学到隐空间的非线性演化

为了透彻理解动作条件下的预测机制，我们首先回到高中物理中最基础的匀速直线运动模型。假设一个物体在时刻 $t$ 的位置为 $x_t \in \mathbb{R}$，在时间间隔 $\Delta t$ 内，它受到了速度为 $v_t$ 的动作输入。那么，它在时刻 $t+1$ 的位置可以通过以下标量方程精确描述：

$$x_{t+1} = x_t + v_t \cdot \Delta t$$

在这个简单的物理系统中，状态 $x_t$ 是完全可观测的（例如物体在坐标轴上的位置），而 $v_t$ 则是我们主动施加的“动作”。如果我们将其推广到多维空间，状态变为向量 $\mathbf{x}_t \in \mathbb{R}^D$，动作变为控制向量 $\mathbf{a}_t \in \mathbb{R}^A$，我们便得到了现代控制理论中的线性离散时间状态方程：

<div align="center">
  <img src="/figures/06-jepa/source/04-action-jepa/world-models-fig4.png" alt="World Models 的 VAE、MDN-RNN 与控制器分工图展示动作如何进入学习到的隐状态动力学。" width="86%">

_图 6.4-2：World Models 的 VAE、MDN-RNN 与控制器分工图展示动作如何进入学习到的隐状态动力学。 出处：David Ha; Jürgen Schmidhuber，[World Models](https://arxiv.org/abs/1803.10122)（2018），Figure 4。_

</div>

$$\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{a}_t$$

其中矩阵 $\mathbf{A}$ 和 $\mathbf{B}$ 描述了系统固有的物理规律。然而，在自动驾驶或机器人控制等真实场景中，我们无法直接获取像“坐标”这样干净的状态向量 $\mathbf{x}_t$。我们能获取的，往往是包含数百万像素的高维图像或高频传感器阵列数据（例如 $\mathbf{o}_t \in \mathbb{R}^{H \times W \times C}$）。更严重的是，这些原始观测数据内部的演化规律是高度非线性的。

为了解决高维观测数据的预测难题，动作条件 JEPA 采用了**降维与抽象**的核心思想。它不直接去预测复杂的 $\mathbf{o}_{t+1}$，而是引入一个**编码器（Encoder）** $E_\theta$，将高维的观测数据映射到一个低维的隐特征空间（Latent Space）中：

$$\mathbf{s}_t = E_\theta(\mathbf{o}_t), \quad \mathbf{s}_t \in \mathbb{R}^d$$

在这个抽象的隐空间中，我们再引入一个由神经网络参数化的非线性**预测器（Predictor）** $P_\phi$，利用当前时刻的隐状态 $\mathbf{s}_t$ 和动作 $\mathbf{a}_t$ 来预测下一时刻的隐状态：

$$\hat{\mathbf{s}}_{t+1} = P_\phi(\mathbf{s}_t, \mathbf{a}_t)$$

动作条件 JEPA 不直接预测下一帧像素，而是预测动作发生后的目标表征。这会改变损失强调的信息，但是否降低任务难度、表征是否足以支持控制，都需要通过下游实验验证。

## 6.4.2 架构解析与严密的数学表达

教学模型由上下文编码器、目标编码器和动作条件预测器组成。先固定每个组件的输入、输出和梯度路径。

1. **上下文编码器（Context Encoder） $E_\theta$**：
   负责处理当前时刻 $t$ 的观测数据 $\mathbf{o}_t$。参数为 $\theta$。其输出被称为上下文表征（Context Representation） $\mathbf{s}_t \in \mathbb{R}^d$。

2. **目标编码器（Target Encoder） $E_{\bar{\theta}}$**：
   负责处理未来时刻 $t+1$ 的真实观测数据 $\mathbf{o}_{t+1}$，产生训练目标表征。参数 $\bar{\theta}$ 不接收该损失的梯度，而是由上下文编码器参数 $\theta$ 的指数移动平均（Exponential Moving Average, EMA）更新。其输出记为 $\mathbf{y}_{t+1} \in \mathbb{R}^d$。EMA 提供缓慢变化的目标，但单独使用它并不构成避免坍塌的数学保证。

3. **动作条件预测器（Action-conditional Predictor） $P_\phi$**：
   接收上下文表征 $\mathbf{s}_t$ 和动作向量 $\mathbf{a}_t \in \mathbb{R}^k$，预测未来状态。参数为 $\phi$。输出为预测表征 $\hat{\mathbf{y}}_{t+1} \in \mathbb{R}^d$。

在给定的批次大小 $B$（Batch Size）下，我们可以将损失函数定义为预测表征 $\hat{\mathbf{Y}}$ 与目标表征 $\mathbf{Y}$ 之间的均方误差（MSE）。设批次中的第 $i$ 个样本的第 $j$ 个特征维度分别为 $\hat{y}_{t+1}^{(i,j)}$ 和 $y_{t+1}^{(i,j)}$，标量形式的损失函数展开如下：

$$\mathcal{L}_{JEPA}(\theta, \phi) = \frac{1}{B \cdot d} \sum_{i=1}^B \sum_{j=1}^d \left( \hat{y}_{t+1}^{(i,j)} - y_{t+1}^{(i,j)} \right)^2$$

<div align="center">
  <img src="/figures/06-jepa/latex/04-action-jepa/jepa-batch-feature-reduction.png" alt="批量特征误差矩阵沿特征维和样本维求和后除以 B 乘 d" width="86%">

_图 6.4-3：每个 e_ij 对应第 i 个样本、第 j 个特征；双重求和消去两维，再除以 Bd 得到逐元素均方误差。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

将其写为紧凑的矩阵形式（即 Frobenius 范数的平方）：

$$\mathcal{L}_{JEPA}(\theta, \phi) = \frac{1}{B \cdot d} \left\| P_\phi(E_\theta(\mathbf{O}_t), \mathbf{A}_t) - E_{\bar{\theta}}(\mathbf{O}_{t+1}) \right\|_F^2$$

请严格注意，在反向传播计算梯度时，梯度**只流向**参数 $\theta$ 和 $\phi$。目标编码器的参数 $\bar{\theta}$ 被视为常数（Stop-Gradient），其更新严格遵循以下 EMA 规则：

$$\bar{\theta} \leftarrow \tau \bar{\theta} + (1 - \tau) \theta$$

其中 $\tau \in [0, 1)$ 是动量系数（Momentum），通常取值非常接近 $1$（如 $0.99$ 或 $0.996$）。

## 6.4.3 EMA 与表征坍塌

先看一个平凡解。若上下文编码器、目标编码器和预测器都输出同一个常数向量，均方误差同样可以为零：

当 $E_\theta(\mathbf{O}) = \mathbf{0}$ 且 $P_\phi(\cdot) = \mathbf{0}$ 时，无论输入什么图像和动作，预测值和目标值永远为 $\mathbf{0}$，损失函数完美等于 $0$。这就是自监督学习中臭名昭著的**表征坍塌（Representation Collapse）**。

EMA 与停止梯度让目标网络不会在同一步直接追随预测器，从而提供较慢变化的学习目标。然而，常数解在代数上仍然存在，不能据此证明模型一定保留环境信息。实际系统还会依赖掩码策略、预测器结构、归一化、数据增强或方差—协方差约束等设计，并通过表征方差和下游任务进行检查。

::: info 说明
可以把目标编码器理解为更新较慢的参照网络。这个类比只解释目标为何较稳定，不应把它理解为“EMA 必然阻止坍塌”的证明。
:::

## 6.4.4 多步预测的自回归展开

真实的规划往往需要预测未来多步的状态。动作条件 JEPA 可以自然地扩展为自回归（Autoregressive）模式。给定初始观测 $\mathbf{o}_t$ 以及一个动作序列 $\mathbf{a}_t, \mathbf{a}_{t+1}, \dots, \mathbf{a}_{t+K-1}$，我们可以递归地展开预测：

<div align="center">
  <img src="/figures/06-jepa/source/04-action-jepa/planet-fig2.png" alt="PlaNet 并列确定性、随机与循环状态空间模型，展示多步潜在动力学中历史与随机状态的不同组织方式。" width="86%">

_图 6.4-4：PlaNet 并列确定性、随机与循环状态空间模型，展示多步潜在动力学中历史与随机状态的不同组织方式。 出处：Danijar Hafner et al.，[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)（2019），Figure 2。_

</div>

1. $\hat{\mathbf{s}}_{t+1} = P_\phi(E_\theta(\mathbf{o}_t), \mathbf{a}_t)$
2. $\hat{\mathbf{s}}_{t+2} = P_\phi(\hat{\mathbf{s}}_{t+1}, \mathbf{a}_{t+1})$
3. ...
4. $\hat{\mathbf{s}}_{t+K} = P_\phi(\hat{\mathbf{s}}_{t+K-1}, \mathbf{a}_{t+K-1})$

相应的，总损失函数将是这 $K$ 步预测误差的累加：

$$\mathcal{L}_{multi} = \sum_{k=1}^K \lambda_k \left\| \hat{\mathbf{s}}_{t+k} - E_{\bar{\theta}}(\mathbf{o}_{t+k}) \right\|_F^2$$

<div align="center">
  <img src="/figures/06-jepa/latex/04-action-jepa/action-conditioned-autoregressive-chain.png" alt="每个动作把上一预测表征推进一步，并在对应未来时刻接受加权目标损失" width="86%">

_图 6.4-5：第 k 步以先前预测和 a_{t+k−1} 为输入；每个预测节点都与同一时刻的目标编码相配，并以 λ_k 加权进入总损失。本文根据上式及递推定义绘制；TikZ/LaTeX 编译。_

</div>

其中 $\lambda_k$ 为不同时间步的权重系数。通过多步展开训练，预测器被迫学习长期的环境动态，而不仅仅是下一步的细微变化。

## 6.4.5 代码实现

下面用 PyTorch 实现一个 MLP 版本的教学骨架，重点检查停止梯度、EMA 更新和张量形状。它不是某篇 Action-JEPA 论文的完整官方实现。

先定义编码器和动作条件预测器。

```python
import torch
import torch.nn as nn
import copy

class Encoder(nn.Module):
    """一个简化的基于 MLP 的编码器，用于提取隐状态。"""
    def __init__(self, obs_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        # x: (Batch_Size, obs_dim)
        return self.net(x)

class ActionPredictor(nn.Module):
    """动作条件预测器：结合当前隐状态和动作，预测下一步隐状态。"""
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        # 在这里，我们将隐状态和动作在特征维度上拼接 (Concatenation)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, state, action):
        # state: (Batch_Size, latent_dim)
        # action: (Batch_Size, action_dim)
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
```

再组装在线分支、目标分支与 EMA 更新。

```python
class ActionConditionalJEPA(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=256, hidden_dim=512, ema_tau=0.99):
        super().__init__()
        self.ema_tau = ema_tau

        # 1. 上下文编码器
        self.context_encoder = Encoder(obs_dim, hidden_dim, latent_dim)

        # 2. 动作条件预测器
        self.predictor = ActionPredictor(latent_dim, action_dim, hidden_dim)

        # 3. 目标编码器 (与上下文编码器结构完全一致，但参数独立)
        self.target_encoder = copy.deepcopy(self.context_encoder)

        # 目标编码器的参数不参与梯度更新，冻结它们
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """执行目标编码器的指数移动平均 (EMA) 更新"""
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            # \bar{\theta} = \tau * \bar{\theta} + (1 - \tau) * \theta
            param_k.mul_(self.ema_tau).add_(param_q, alpha=1.0 - self.ema_tau)

    def forward(self, obs_t, action_t, obs_t_plus_1):
        """
        前向传播计算单步预测损失
        """
        # (1) 提取当前上下文隐状态: (Batch_Size, latent_dim)
        s_t = self.context_encoder(obs_t)

        # (2) 预测下一时刻的隐状态: (Batch_Size, latent_dim)
        s_t_plus_1_pred = self.predictor(s_t, action_t)

        # (3) 使用目标编码器获取真实的下一时刻目标隐状态
        with torch.no_grad():
            s_t_plus_1_target = self.target_encoder(obs_t_plus_1)
            # no_grad 在此处停止目标分支的梯度
            s_t_plus_1_target = s_t_plus_1_target.detach()

        # (4) 计算 MSE 损失
        # 对批量和特征维取平均的均方误差
        loss = nn.functional.mse_loss(s_t_plus_1_pred, s_t_plus_1_target)

        return loss, s_t_plus_1_pred
```

下面的训练循环同时记录损失和隐状态方差。方差趋近于零是完全坍塌的警报；方差非零并不能单独证明表征有用。

```python
# 初始化模型和优化器
obs_dim, action_dim = 128, 4
jepa = ActionConditionalJEPA(obs_dim=obs_dim, action_dim=action_dim)
optimizer = torch.optim.Adam(jepa.parameters(), lr=1e-3)

# 模拟一个批次的随机观测和动作数据
batch_size = 32
obs_t = torch.randn(batch_size, obs_dim)
action_t = torch.randn(batch_size, action_dim)
# 模拟环境的演化：下一时刻的观测，这里加入了一些噪声来模拟不可预测性
obs_t_plus_1 = obs_t + 0.1 * torch.randn(batch_size, obs_dim)

for step in range(50):
    optimizer.zero_grad()

    # 计算损失
    loss, pred_state = jepa(obs_t, action_t, obs_t_plus_1)

    # 反向传播并更新 \theta 和 \phi
    loss.backward()
    optimizer.step()

    # 【关键步骤】手动更新目标编码器 \bar{\theta}
    jepa.update_target_encoder()

    if (step + 1) % 10 == 0:
        # 批内方差接近 0 是完全坍塌警报；非零不等于表征一定有用。
        state_variance = pred_state.var(dim=0).mean().item()
        print(f"Step {step+1}: Loss = {loss.item():.4f}, 隐特征均方差 = {state_variance:.4f}")
```

## 6.4.6 小结与实践指导

动作条件 JEPA 把当前表征和动作映射到未来表征，并用变化较慢的目标分支提供监督。它是否保留了控制所需信息，需要结合预测误差、表征诊断和下游控制结果判断。

实际训练要注意以下两点：

1. **EMA 动量参数（$\tau$）**：较小的 $\tau$ 让目标快速跟随在线网络，较大的 $\tau$ 让目标更平滑但滞后更明显。合适取值依赖优化器、批量和训练长度；一些方法会逐渐调高 $\tau$，而不是把某个阈值当作坍塌定律。
2. **多步预测稳定性**：自回归预测会把前一步误差带入后一步。误差是否放大以及放大速度取决于预测动力学的局部 Jacobian 等因素，并不总是指数增长。可用逐步监督、归一化、较短训练时域或非自回归结构缓和。
