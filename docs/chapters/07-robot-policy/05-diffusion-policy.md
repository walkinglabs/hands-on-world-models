# 扩散策略（Diffusion Policy）

在机器人学习的早期探索中，行为克隆（Behavior Cloning, BC）占据了主导地位。传统的行为克隆模型通常被建模为一个确定性函数或简单的高斯分布，其目标是最小化预测动作与专家动作之间的均方误差（MSE）。然而，由于现实世界中专家数据的内在复杂性，传统方法在处理多模态动作分布（Multimodal Action Distribution）时往往会遭遇严重瓶颈。

为了直观地理解这一点，我们可以想象这样一个极其简单的物理场景：机器人正前方有一根柱子，而目标位于柱子正后方。在专家演示数据中，操作员有时会控制机器人从左侧绕过柱子，有时则从右侧绕过。如果使用传统的基于均方误差的神经网络来拟合这些数据，模型会倾向于输出左转和右转的“平均值”——即径直向前走，从而导致机器人直接撞上柱子。这种现象在学术界被称为“模式平均”（Mode Averaging）。

DDPM 通过学习逆转逐步加噪过程来生成样本 [[Ho et al., 2020]](https://arxiv.org/abs/2006.11239)。Diffusion Policy 把这一思想用于机器人模仿学习，把动作序列建模为以视觉或状态观测为条件的去噪过程 [[Chi et al., 2023]](https://arxiv.org/abs/2303.04137)。它可以表达多模态动作分布，但并不要求所有控制问题都放弃确定性策略。本节据此推导一个简化的条件动作扩散模型。

## 扩散过程的数学基础：从信号到噪声

在探讨复杂的动作轨迹之前，我们先从一个初高中生都能理解的简单物理量起步。假设机器人的当前动作仅仅是一个标量 $a_0 \in \mathbb{R}$（例如方向盘的转动角度）。

扩散模型的核心思想包含两个相对的过程：**前向扩散（Forward Diffusion）**和**逆向去噪（Reverse Denoising）**。

前向扩散过程是一个固定的马尔可夫链（Markov Chain）。在每一步 $k \in \{1, 2, \dots, K\}$ 中，我们向当前的动作变量中加入微小的高斯噪声，直到最后 $a_K$ 几乎变成一个纯粹的标准正态分布噪声。

我们定义每一步的转移概率分布为：

$$q(a_k \mid a_{k-1}) = \mathcal{N}(a_k; \sqrt{\alpha_k} a_{k-1}, (1 - \alpha_k)\mathbf{I})$$

其中，$\alpha_k$ 是一个介于 $0$ 和 $1$ 之间、且随着步数 $k$ 增加而逐渐减小的超参数（我们称由所有 $\alpha_k$ 组成的序列为方差调度计划，Variance Schedule）。

根据正态分布的性质，我们可以将该公式写成一个显式的代数等式。如果我们引入一个服从标准正态分布的随机变量 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$，那么第 $k$ 步的状态可以表示为：

$$a_k = \sqrt{\alpha_k} a_{k-1} + \sqrt{1 - \alpha_k} \boldsymbol{\epsilon}_{k-1}$$

从表面上看，如果要计算任意步 $a_k$，我们需要一步步从 $a_0$ 迭代计算过来。但这在实际训练中是极其低效的。令人惊叹的是，由于高斯分布的加法性质（两个独立的正态分布变量之和仍然是正态分布，其均值为零，方差等于两者方差之和），我们可以直接写出从初始状态 $a_0$ 到任意状态 $a_k$ 的一步转移公式。

让我们展开前向过程的前两步看看：

$$
\begin{aligned}
a_2 &= \sqrt{\alpha_2} a_1 + \sqrt{1 - \alpha_2} \boldsymbol{\epsilon}_1 \\
&= \sqrt{\alpha_2} (\sqrt{\alpha_1} a_0 + \sqrt{1 - \alpha_1} \boldsymbol{\epsilon}_0) + \sqrt{1 - \alpha_2} \boldsymbol{\epsilon}_1 \\
&= \sqrt{\alpha_1 \alpha_2} a_0 + \sqrt{\alpha_2(1 - \alpha_1)} \boldsymbol{\epsilon}_0 + \sqrt{1 - \alpha_2} \boldsymbol{\epsilon}_1
\end{aligned}
$$

注意到 $\boldsymbol{\epsilon}_0$ 和 $\boldsymbol{\epsilon}_1$ 是独立同分布的标准正态随机变量。根据方差的加法法则，后两项可以合并为一个新的高斯噪声项，其联合方差为 $\alpha_2(1 - \alpha_1) + (1 - \alpha_2) = 1 - \alpha_1 \alpha_2$。

如果我们定义 $\bar{\alpha}_k = \prod_{i=1}^k \alpha_i$，通过数学归纳法，我们可以直接写出任意步 $k$ 的边缘分布公式：

$$a_k = \sqrt{\bar{\alpha}_k} a_0 + \sqrt{1 - \bar{\alpha}_k} \boldsymbol{\epsilon}$$

这里 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$。这个公式极其优美且关键：它告诉我们，无论前向扩散了多少步，当前状态 $a_k$ 都可以被看作是初始真实动作 $a_0$ 的衰减（权重为 $\sqrt{\bar{\alpha}_k}$）与累积随机噪声（权重为 $\sqrt{1 - \bar{\alpha}_k}$）的线性组合。由于 $\alpha_k < 1$，随着 $k \to K$，$\bar{\alpha}_K \to 0$，$a_K$ 将完全由噪声主导。

## 逆向去噪：条件概率的参数化重构

前向过程破坏了真实的动作信息，而我们的终极目标是教会神经网络如何逆向执行这一过程——即从纯噪声 $a_K \sim \mathcal{N}(0, \mathbf{I})$ 开始，一步步去除噪声，最终恢复出能够执行任务的合理动作 $a_0$。

在数学上，我们需要求解逆向条件分布 $q(a_{k-1} \mid a_k)$。然而，直接计算它是不可解的，因为它依赖于整个数据集的先验概率。但是，当我们不仅知道当前噪声状态 $a_k$，还知道初始真实状态 $a_0$ 时，这个逆向转移概率 $q(a_{k-1} \mid a_k, a_0)$ 却是可精确求解的。

运用基础的贝叶斯定理，我们可以写出后验概率：

$$
q(a_{k-1} \mid a_k, a_0) = \frac{q(a_k \mid a_{k-1}) q(a_{k-1} \mid a_0)}{q(a_k \mid a_0)}
$$

将等式右侧三个已知的高斯分布概率密度函数代入，并对指数项进行配方展开后，我们可以证明 $q(a_{k-1} \mid a_k, a_0)$ 依然是一个高斯分布 $\mathcal{N}(a_{k-1}; \tilde{\mu}_k, \tilde{\beta}_k \mathbf{I})$，其均值 $\tilde{\mu}_k$ 为：

$$\tilde{\mu}_k(a_k, a_0) = \frac{\sqrt{\bar{\alpha}_{k-1}} (1 - \alpha_k)}{1 - \bar{\alpha}_k} a_0 + \frac{\sqrt{\alpha_k} (1 - \bar{\alpha}_{k-1})}{1 - \bar{\alpha}_k} a_k$$

由于在实际逆向生成时我们是不可能提前知道真实动作 $a_0$ 的，我们需要通过神经网络来预测它。仔细观察该公式，我们可以将 $a_0$ 重新表达为关于 $a_k$ 和噪声 $\boldsymbol{\epsilon}$ 的函数：

$$a_0 = \frac{1}{\sqrt{\bar{\alpha}_k}} \left( a_k - \sqrt{1 - \bar{\alpha}_k} \boldsymbol{\epsilon} \right)$$

将该公式代入该公式并进行代数化简，我们得到了一个极其优雅的均值推导表达式：

$$\tilde{\mu}_k = \frac{1}{\sqrt{\alpha_k}} \left( a_k - \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}} \boldsymbol{\epsilon} \right)$$

这个等式揭示了逆向去噪机制的核心底色：**为了计算前一步更清晰的动作状态 $a_{k-1}$，我们只需要知道当前带噪状态 $a_k$ ，以及在前向过程中具体添加到 $a_k$ 里的累积噪声 $\boldsymbol{\epsilon}$ 即可。**

由于真实的累积噪声 $\boldsymbol{\epsilon}$ 对模型而言是不可见的，我们引入一个神经网络 $\boldsymbol{\epsilon}_\theta$ 来预测它。在扩散策略中，由于机器人的动作必须时刻依赖于其感知到的环境状态（例如当前帧的相机图像特征或关节物理状态），这个神经网络被设计为一个强条件神经网络。它接收三个维度的输入：当前的带噪动作轨迹 $\mathbf{A}_k$、当前的扩散时间步 $k$、以及机器人的多模态观察序列 $\mathbf{O}$。

由此，机器人的逆向采样（决策）迭代公式被最终定义为：

$$\mathbf{A}_{k-1} = \frac{1}{\sqrt{\alpha_k}} \left( \mathbf{A}_k - \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}} \boldsymbol{\epsilon}_\theta(\mathbf{A}_k, k, \mathbf{O}) \right) + \sigma_k \mathbf{z}$$

其中 $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ 引入了不可或缺的退火随机噪声，从热力学的角度来看，这种朗之万动力学（Langevin Dynamics）式的随机扰动确保了采样过程能够在多模态动作分布中进行充分的概率探索；$\sigma_k$ 通常取值为 $\sqrt{1 - \alpha_k}$ 或是其他推导出的近似方差标量。

## 架构创新：动作块与滚动优化

为了将上述纯粹的数学概率理论切实应用到现实世界中对延迟极其敏感的机器人控制上，扩散策略团队引入了两个极具工程智慧的控制逻辑架构设计：

1. **动作块序列（Action Chunking）**：模型不再像传统的强化学习那样仅仅预测单一未来时刻的离散动作，而是直接预测一个完整时间窗口内的连续动作轨迹序列 $\mathbf{A}_{t:t+T_a}$。这种批量的预测极大地提升了动作在时间序列空间上的连续性与物理平滑度，避免了单步自回归生成中极其容易产生的复合误差累积（Compounding Errors）。
2. **滚动优化时间域（Receding Horizon Control）**：受经典模型预测控制（MPC）理念的启发，虽然模型一次大局观地生成了 $T_a$ 步长的动作序列，但机器人只会死板地严格执行其中的前 $T_e$ 步（通常 $T_e \ll T_a$）。执行完毕后，机器人摒弃剩余的计划，立刻获取最新的环境观察结果，并重新扩散生成新的动作轨迹序列。这种设计有效抵御了现实世界中随时可能发生的摩擦、抖动与外部物理扰动。

在深度网络的架构拓扑层面，由于我们需要对时间序列维度的动作轨迹进行去噪，传统的二维卷积图像网络不再适用。扩散策略最常采用带有密集残差连接的一维条件卷积网络（1D Conditional ResNet）或时间序列 Transformer 作为 $\boldsymbol{\epsilon}_\theta$ 的骨干。至关重要的外部观察条件 $\mathbf{O}$ 往往通过 FiLM（Feature-wise Linear Modulation）机制或跨注意力机制（Cross-Attention）深度且均匀地注入到去噪网络的深层架构中。

## 深度解析：代码实现

接下来，我们将使用 PyTorch 从零构建一个高密度的扩散策略架构。为了避免使代码被冗长繁琐的视觉处理管道（如 ResNet 或 ViT 编码器）所淹没，我们将剥离这些前置编码器，假定视觉特征已经被抽取为了连续的向量，从而纯粹地专注于扩散动作生成与条件去噪的本质过程。

[**我们首先定义严格的余弦方差调度计划以及与之匹配的前向加噪模块。**] 余弦调度在实际工程中已被广泛证明在图像与控制动作生成上显著优于最简单的线性调度，因为它能在扩散中后期更加克制地引入噪声，保留动作序列的高频微调信息。

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

class DDPMScheduler:
    def __init__(self, num_train_timesteps=100):
        self.num_train_timesteps = num_train_timesteps

        # 严格计算余弦调度计划 (Cosine Variance Schedule)
        steps = num_train_timesteps + 1
        x = torch.linspace(0, num_train_timesteps, steps)
        # 遵循原始论文引入平滑项 0.008 避免边界奇点
        alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        # 将 beta 裁剪到安全范围内，保证严格的数值稳定性
        self.betas = torch.clip(betas, 0.0001, 0.999)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        """实现方程 7.5.3：前向过程的一步直接加噪闭式解"""
        # 将标量 timesteps 转换为与输入张量对齐的形状
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 线性组合：衰减的信号 + 累积的噪声
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
```

[**接下来，我们定义用于逼近真实噪声分量的一维条件卷积骨干网络。**] 注意这里我们严格遵循了学术界主流的 FiLM（特征级线性调制）特征融合范式，将环境的先验观察条件与时间步特征直接作用于卷积激活后的流形层面上。

```python
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.Mish()

        # FiLM 机制映射层：将多模态条件维度投影为缩放(scale)和偏移(shift)调制因子
        self.cond_encoder = nn.Linear(cond_dim, out_channels * 2)

        # 确保通道维度匹配的残差连接桥梁
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        # x 形状: (Batch, in_channels, sequence_length)
        # cond 形状: (Batch, cond_dim)

        out = self.activation(self.conv1(x))

        # 计算特征维度上的 FiLM 调制因子
        cond_emb = self.cond_encoder(cond)  # (Batch, out_channels * 2)
        scale, shift = cond_emb.chunk(2, dim=-1)

        # 调整张量形状以严格匹配时间序列上的每一帧维度
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)

        # 注入条件：仿射变换调制
        out = out * (scale + 1.0) + shift

        out = self.conv2(self.activation(out))
        return out + self.residual_conv(x)

class SimpleConditionalUnet1D(nn.Module):
    def __init__(self, action_dim, obs_dim, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim

        # 扩散时间步的正弦编码与感知机注入层
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 环境视觉/本体感受状态特征的降维流形层
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        cond_dim = hidden_dim * 2

        # 高度浓缩的一维残差网络拓扑
        self.block1 = ConditionalResidualBlock1D(action_dim, hidden_dim, cond_dim)
        self.block2 = ConditionalResidualBlock1D(hidden_dim, hidden_dim, cond_dim)
        self.block3 = ConditionalResidualBlock1D(hidden_dim, action_dim, cond_dim)

    def forward(self, noisy_action_sequence, timestep, observation):
        # noisy_action_sequence 形状预期为: (Batch, sequence_length, action_dim)
        # PyTorch 的 Conv1d 要求通道维度置前，因此我们先执行转置操作
        x = noisy_action_sequence.transpose(1, 2)

        # 独立编码环境观察先验与时间步特征
        t_emb = self.time_mlp(timestep.unsqueeze(-1).float())
        o_emb = self.obs_mlp(observation)
        cond = torch.cat([t_emb, o_emb], dim=-1)

        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.block3(x, cond)

        # 恢复时间序列维度输出
        return x.transpose(1, 2)
```

[**最后，我们将所有数学模块严丝合缝地组装为一个极简版本的端到端训练与逆向生成采样的闭环流程。**]

```python
# 初始化环境维度界限设定与批次内存开销数据
batch_size = 64
action_sequence_length = 16  # 定义动作块长度 (Action Chunking)
action_dim = 2               # 本文设定为 2 维连续动作，例如：[底盘平移速度, 旋转角速度]
obs_dim = 32                 # 经过视觉骨干网络压缩后的密集感知向量的维度

# 实例化核心网络架构与调度数学内核
model = SimpleConditionalUnet1D(action_dim, obs_dim)
scheduler = DDPMScheduler(num_train_timesteps=100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ----------------- 训练范式 (Training Phase) -----------------
# 我们在此使用张量构造完美的伪造专家动作轨迹与先验观察
expert_actions = torch.randn(batch_size, action_sequence_length, action_dim)
observations = torch.randn(batch_size, obs_dim)

# 1. 为批次中的每条专家轨迹随机且独立地采样扩散时间步
timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,))

# 2. 采样标准的正态分布纯随机噪声
noise = torch.randn_like(expert_actions)

# 3. 根据方程 7.5.3 精确计算叠加了不同程度噪声的退化动作序列
noisy_actions = scheduler.add_noise(expert_actions, noise, timesteps)

# 4. 前向传播：条件神经网络尝试破译并预测出当时被注入的纯噪声分量
predicted_noise = model(noisy_actions, timesteps, observations)

# 5. 计算优化目标函数：简单的欧几里得距离均方误差
loss = F.mse_loss(predicted_noise, noise)
loss.backward()
optimizer.step()

print(f"严密的单步迭代训练已完成，当前微批次全局 MSE Loss: {loss.item():.4f}")

# ----------------- 逆向推理解码 (Inference / Reverse Sampling) -----------------
model.eval()
with torch.no_grad():
    # 彻底抹除人类的先验知识，机器人决策起点仅仅是一段纯随机的高斯噪声序列
    current_action_state = torch.randn(1, action_sequence_length, action_dim)
    # 获取当下的实时真实物理世界观察结果
    single_observation = torch.randn(1, obs_dim)

    # 严格遵循逆向马尔可夫链分布，逐步剥离冗余的噪声分量 (对应方程 7.5.7)
    for t in reversed(range(scheduler.num_train_timesteps)):
        t_tensor = torch.tensor([t])

        # 网络预测剥离量
        pred_noise = model(current_action_state, t_tensor, single_observation)

        # 提取当前时间步在调度计划中对应的衰减权重系数
        alpha = scheduler.alphas[t]
        alpha_cumprod = scheduler.alphas_cumprod[t]

        # 当未抵达链条末端时，必须遵循理论加入适度的退火随机游走噪声
        if t > 0:
            noise = torch.randn_like(current_action_state)
            sigma = scheduler.betas[t] ** 0.5
        else:
            noise = torch.zeros_like(current_action_state)
            sigma = 0.0

        # 根据我们严谨推导出的逆向重参数化转移公式，代数计算出上一步更具物理意义的动作序列
        current_action_state = (1 / alpha**0.5) * (
            current_action_state - ((1 - alpha) / (1 - alpha_cumprod)**0.5) * pred_noise
        ) + sigma * noise

    print("最终通过条件逆向去噪生成的平滑连续动作轨迹矩阵维度:", current_action_state.shape)
```

通过这一系列极其严密的代数推导与工程实现，我们完成了从传统的确定性网络拟合向现代生成式概率模型的宏大跳跃。

## 小结

- 传统的行为克隆在高度复杂的专家多模态动作分布面前，常常陷入灾难性的**模式平均（Mode Averaging）**问题，导致模型采取致命的妥协动作。
- **扩散策略（Diffusion Policy）**巧妙地利用基于严密马尔可夫链理论的去噪过程，将任意复杂的控制策略生成转化为从高斯噪声中提纯物理信号的过程。
- 逆向条件去噪的核心数学引擎在于，通过条件神经网络预测前向过程中引入的纯噪声分量，这一结论是基于严谨的贝叶斯后验概率均值重参数化推导而得出的必然结果。
- **动作序列块（Action Chunking）**和滚动优化时间域（Receding Horizon Control）的组合，赋予了该策略面对真实物理世界时极高的稳健性与连贯性。
