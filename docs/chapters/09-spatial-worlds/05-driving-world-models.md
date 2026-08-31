# 9.5 自动驾驶世界模型

在探讨了通用的空间与视频生成模型之后，我们自然而然地会将目光投向当前具身智能（Embodied AI）最具挑战性、也是商业化落地最迫切的领域——自动驾驶。传统自动驾驶系统通常被拆解为感知（Perception）、预测（Prediction）、规划（Planning）和控制（Control）等多个独立模块。尽管这种模块化设计在工程上便于调试，但由于信息在模块间传递时的不可避免的丢失与误差累积，系统往往难以应对长尾（Long-tail）的复杂边缘场景（Corner Cases）。

GAIA-1 根据视频、文本与车辆动作生成驾驶场景 [[Anthony Hu et al., 2023]](https://arxiv.org/abs/2309.17080)，DriveDreamer 则用结构化交通条件控制驾驶视频生成 [[Xiaofeng Wang et al., 2023a]](https://arxiv.org/abs/2309.09777)。这类模型可以检验对动作条件和场景演化的统计建模能力，但视觉预测逼真并不等同于“理解”全部物理规律；还需要几何、反事实与闭环驾驶评测。

在本节中，我们将从最基础的物理运动学出发，逐步推导自动驾驶世界模型的概率生成框架，并深入解析其背后的核心机制——如何将多模态数据（视频、文本、动作）统一映射至隐空间（Latent Space），以及如何通过自回归（Autoregressive）或扩散（Diffusion）过程在时间维度上进行演化。

## 9.5.1 从高中物理到条件概率预测

在正式引入复杂的深度学习框架之前，让我们先回到最熟悉的高中物理——直线运动学。假设一辆汽车在 $t$ 时刻的位置为 $x_t$，速度为 $v_t$。如果我们对汽车施加一个恒定的加速度 $a_t$（可以视为“动作”或“控制指令”），那么经过一个极小的时间间隔 $\Delta t$ 后，汽车在 $t+1$ 时刻的物理状态可以通过以下经典公式完全确定：

$$x_{t+1} = x_t + v_t \Delta t + \frac{1}{2} a_t \Delta t^2$$

在上述理想化的物理系统中，世界是**确定性的（Deterministic）**。只要我们知道当前状态（$x_t, v_t$）和控制指令（$a_t$），未来的状态就是唯一确定的。

然而，真实的驾驶世界充满了**高度的不确定性（Stochasticity）**：前方的行人可能突然横穿马路，旁边的车辆可能强行并线，天气的变化也会改变路面摩擦力。因此，我们不能再用确定性的等式来描述未来的状态，而必须引入概率论的视角。

令 $s_t$ 表示 $t$ 时刻驾驶世界的完整状态（包括自车状态、环境视觉、其他交通参与者等），$a_t$ 表示 $t$ 时刻自车采取的动作。我们将预测未来视为一个计算条件概率分布的问题：

$$P(s_{t+1} \mid s_t, a_t)$$

更一般地，如果我们要预测未来 $T$ 个时间步的状态序列，并考虑到历史状态的影响以及可能的额外上下文信息 $c$（例如天气提示文本、导航路线），根据概率论的链式法则，整个未来序列的联合概率分布可以分解为：

$$P(s_{t+1:t+T} \mid s_{1:t}, a_{1:t+T-1}, c) = \prod_{k=1}^{T} P(s_{t+k} \mid s_{1:t+k-1}, a_{1:t+k-1}, c)$$

这正是自回归世界模型（如 GAIA-1）的核心数学基础。通过最大化真实驾驶数据在此分布下的似然，模型被迫学习驾驶世界中极其复杂的因果关系与演化规律。

## 9.5.2 多模态隐空间表示：维度的“温柔”压缩

在自动驾驶中，状态 $s_t$ 通常是高分辨率的环视摄像头图像流。直接在高维的像素空间中建模上述条件概率分布 $P(s_{t+1} \mid s_{1:t}, \dots)$ 是极其低效且容易受高频噪声干扰的。为了解决这个问题，GAIA-1 等模型采用了一个经典策略：先将所有模态压缩到一个低维的隐空间（Latent Space）。

> 我们可以将这种降维过程视为物理学中的“质点抽象”：在研究汽车运动时，我们忽略汽车的颜色、车牌号和内饰细节，仅仅提取其质量和质心位置。隐空间的本质，就是利用神经网络自动寻找高维像素中最具物理意义的“质点化”特征表示。

以标量数据为例，假设我们有一个一维连续信号 $x \in \mathbb{R}$，我们希望用有限的离散状态来近似它，最简单的方法就是四舍五入。将其推广到高维向量空间，这就引出了**向量量化（Vector Quantization, VQ）**。

令 $\mathcal{Z} = \{e_1, e_2, \dots, e_K\}$ 为一个包含 $K$ 个可学习向量的编码本（Codebook），其中 $e_i \in \mathbb{R}^D$。给定一张图像 $I_t \in \mathbb{R}^{H \times W \times 3}$，编码器 $E$ 首先将其映射为连续的隐变量特征图 $z_e \in \mathbb{R}^{h \times w \times D}$。

(**对于 $z_e$ 中的每一个空间位置的 $D$ 维向量 $z_e^{(i,j)}$，我们在编码本中寻找与其欧氏距离最近的项：**)

$$z_q^{(i,j)} = e_k, \quad \text{其中 } k = \mathop{\mathrm{argmin}}_{j \in \{1, \dots, K\}} \| z_e^{(i,j)} - e_j \|_2$$

量化后的特征图 $z_q$ 被送入解码器 $D$ 重构原图。由于 $\mathop{\mathrm{argmin}}$ 操作不可导，我们通常使用直通估计器（Straight-Through Estimator）将解码器的梯度直接复制给编码器，并结合承诺损失（Commitment Loss）更新编码本：

$$\mathcal{L}_{\text{VQ}} = \| I_t - D(z_q) \|_2^2 + \beta \| z_e - \text{sg}(z_q) \|_2^2 + \| \text{sg}(z_e) - z_q \|_2^2$$

其中 $\text{sg}(\cdot)$ 表示停止梯度（Stop-Gradient）操作，第一项是重建损失，后两项分别约束编码器输出靠近编码本，以及编码本靠近编码器输出。通过这种方式，我们将一帧复杂的驾驶图像，严谨地映射为了一个离散的整数索引网格 $S_t \in \{1, \dots, K\}^{h \times w}$，即图像的 Token 序列。

## 9.5.3 时序演化：动作条件下的自回归建模

在获得了图像的隐式 Token、动作的隐式表示以及文本指令的 Token 之后，世界模型的任务就转化为一个纯粹的序列预测问题。

设在时刻 $t$，环境状态的离散 Token 序列为 $Z_t$，前置摄像头的当前视野与自车运动状态相互交织。我们将文本条件表示为 $C$，过去的动作序列表示为 $A_{<t}$，过去的视觉状态序列表示为 $Z_{<t}$。

在自回归 Transformer 中，预测下一个 Token $z_i$ 的对数概率可以写为：

$$\log P(Z_t \mid Z_{<t}, A_{<t}, C) = \sum_{i=1}^{|Z_t|} \log P(z_i \mid z_{<i}, Z_{<t}, A_{<t}, C)$$

为了实现这一点，GAIA-1 架构将过去的信息 $Z_{<t}$ 和 $A_{<t}$ 作为 Context，通过因果注意力掩码（Causal Attention Mask），保证在预测 $t$ 时刻的未来状态时，模型只能看到时刻 $t$ 之前以及时刻 $t$ 之前发生的操作。

具体到张量维度的推演：
假设我们在自回归模型中输入了长度为 $L$ 的序列表示 $X \in \mathbb{R}^{L \times d_{\text{model}}}$。在线性投影得到 Query, Key, Value 矩阵后：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

因果注意力机制的输出计算严格遵循：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

其中掩码矩阵 $M \in \mathbb{R}^{L \times L}$ 满足：如果 $i < j$（即试图看向未来的信息），则 $M_{i,j} = -\infty$，否则 $M_{i,j} = 0$。这在数学上绝对保证了时间流动的单向性。

## 9.5.4 扩散模型视角下的世界模型 (DriveDreamer)

与 GAIA-1 纯粹离散自回归的思路不同，DriveDreamer 等工作则探索了基于潜在扩散模型（Latent Diffusion Models, LDM）的世界模型范式。扩散模型并不像自回归模型那样逐个 Token 预测，而是一次性对整个连续的隐空间张量进行迭代去噪。

假设 $z_0$ 为未来驾驶视频的真实隐状态。扩散过程向其中逐步添加高斯噪声，在步数 $n$ 的边缘分布满足：

$$q(z_n \mid z_0) = \mathcal{N}(z_n; \sqrt{\bar{\alpha}_n} z_0, (1 - \bar{\alpha}_n)\mathbf{I})$$

而在逆向生成（去噪）阶段，神经网络 $\epsilon_\theta$ 的目标是预测添加的噪声。为了让世界模型理解**动作指令**和**历史状态**，我们将它们作为额外的条件注入到网络中：

$$\mathcal{L}_{\text{Diffusion}} = \mathbb{E}_{z_0, \epsilon, n} \left[ \| \epsilon - \epsilon_\theta(z_n, n, C_{\text{text}}, C_{\text{action}}, Z_{<t}) \|_2^2 \right]$$

通过交叉注意力（Cross-Attention），特征空间被调制，使得去噪出的新一帧视频严丝合缝地遵循驾驶员（或系统）下达的动作指令（如左转、加速）。

## 9.5.5 代码实现

下面，我们将实现一个简化的、基于 Transformer 的动作条件世界模型（Action-Conditioned World Model）核心架构。为了保持严谨性，我们将使用 PyTorch 演示如何将过去的状态 Token、当前的动作以及环境上下文结合，并应用因果掩码。

```python
import torch
from torch import nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
    """标准的带因果掩码的多头自注意力机制"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence Length, Embedding Dimension

        # 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # 形状变换以支持多头注意力: (B, T, n_heads, C // n_heads) -> (B, n_heads, T, C // n_heads)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # 核心：使用 PyTorch 内置的缩放点积注意力（自动处理因果掩码）
        # 等价于我们在该公式中的数学推导
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 将多头拼接回去
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class DrivingWorldModelBlock(nn.Module):
    """自动驾驶世界模型的一个 Transformer 块"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # 残差连接
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class SimpleActionConditionedWorldModel(nn.Module):
    """
    简化的动作条件世界模型
    接收视觉 Token、文本嵌入和动作向量，自回归预测未来的视觉 Token。
    """
    def __init__(self, vocab_size, d_model, max_seq_len, n_layers=4, n_heads=4):
        super().__init__()
        self.d_model = d_model

        # 视觉 Token 嵌入字典
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # 绝对位置编码
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # 动作投影：将连续动作（如转向角、油门，通常为低维向量）映射到隐空间
        self.action_proj = nn.Linear(2, d_model) # 假设动作维度为 2

        self.blocks = nn.Sequential(*[DrivingWorldModelBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        # 最终输出头，映射回词表概率
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, actions):
        """
        idx: 视觉 Token 的索引序列，形状 (B, T)
        actions: 历史及当前动作序列，形状 (B, T, 2)
        """
        B, T = idx.size()

        # 获取 Token 嵌入与位置嵌入
        tok_embeddings = self.token_emb(idx) # (B, T, d_model)
        pos_embeddings = self.pos_emb[:, :T, :] # (1, T, d_model)

        # 动作特征映射
        act_embeddings = self.action_proj(actions) # (B, T, d_model)

        # 将视觉状态与对应的动作特征在隐空间相加（代表在对应状态下施加了该动作）
        # 这是一种将条件注入模型的标准多模态融合方式
        x = tok_embeddings + act_embeddings + pos_embeddings

        # 通过 Transformer 块进行时序演化
        x = self.blocks(x)
        x = self.ln_f(x)

        # 预测下一时刻的 Token 对数概率 (Logits)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
```

(**让我们验证上述模型的输入输出维度，确保其符合我们对张量计算的严谨预期。**)

```python
# 初始化超参数
batch_size = 4
seq_length = 16
vocab_size = 1024
d_model = 256

model = SimpleActionConditionedWorldModel(vocab_size=vocab_size, d_model=d_model, max_seq_len=64)

# 模拟随机生成的视觉 Token 索引 (如经过 VQ-VAE 量化后的序列)
dummy_vision_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
# 模拟动作序列 (转向角, 速度/油门)
dummy_actions = torch.randn(batch_size, seq_length, 2)

# 前向传播
logits = model(dummy_vision_tokens, dummy_actions)

# 输出张量的形状应该为 (Batch, Sequence Length, Vocab Size)
print("Logits shape:", logits.shape)
```

显然，输出维度的确严格匹配了 `(B, T, vocab_size)`，即每一个时间步 $t$，模型都输出了对其下一个可能视觉状态 $s_{t+1}$ 在有限码本空间上的完整概率分布。这不仅在数学上与该公式实现了完美的闭环，在工程上也构成了自动驾驶生成式仿真的基础构件。

## 9.5.6 小结

自动驾驶世界模型代表了我们对物理世界建模方式的深刻转变。通过摒弃将驾驶拆分为独立子任务的传统方式，我们将整车的传感器输入和动作历史进行统一联合建模。
无论是使用如 GAIA-1 的向量量化结合自回归 Transformer 预测该公式，还是使用 DriveDreamer 基于条件扩散模型不断去噪该公式，其本质都在于：**通过海量数据迫使神经网络隐式地学习微积分与刚体动力学，并在生成未来视频的过程中展现出对物理定律和人类驾驶逻辑的“理解”。**

## 9.5.7 练习

1. 在该公式中，为什么我们需要引入停止梯度操作 `sg()`？如果你将其移除，会对反向传播的过程产生什么数学上的致命影响？
   _提示：思考损失函数第一项，重构误差试图直接拉近原图与解码器输出。如果没有 `sg()`，编码本的更新可能会陷入一种怎样的平庸解（Trivial Solution）？_
2. 我们在代码中使用加法（`tok_embeddings + act_embeddings`）来融合状态和动作。除此之外，还有哪些将额外条件注入 Transformer 的数学方法？
   _提示：可以回顾我们在之前的章节中提到的 Cross-Attention 机制（如在扩散模型中被广泛应用），或者考虑特征的通道拼接（Concatenation）。_
3. 假设我们要用该世界模型评估当前驾驶策略的安全性，你会如何利用它输出的对数概率分布来计算某个危险动作引发碰撞的“风险期望”？
