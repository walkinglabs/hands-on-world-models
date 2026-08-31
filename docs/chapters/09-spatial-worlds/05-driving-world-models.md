# 9.5 自动驾驶世界模型

自动驾驶世界模型要回答一个具体问题：给定过去的视频、车辆状态和准备执行的动作，接下来可能看到什么？传统系统常把感知、预测、规划和控制拆成独立模块；生成式世界模型则尝试联合预测观测与场景变化。两种路线各有接口与评测方式，不能仅凭模块数量判断哪一种更可靠。

<div align="center">
<img src="/figures/09-spatial-worlds/source/05-driving-world-models/drivedreamer-fig1.png" alt="DriveDreamer 根据道路结构与交通参与者条件生成多样、可控的真实驾驶场景序列。" width="86%">

_图 9.5-1：DriveDreamer 根据道路结构与交通参与者条件生成多样、可控的真实驾驶场景序列。 出处：Xiaofeng Wang et al.，[DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving](https://arxiv.org/abs/2309.09777)（2023），Figure 1。_
</div>

GAIA-1 根据视频、文本与车辆动作生成驾驶场景 [[Anthony Hu et al., 2023]](https://arxiv.org/abs/2309.17080)，DriveDreamer 则用结构化交通条件控制驾驶视频生成 [[Xiaofeng Wang et al., 2023a]](https://arxiv.org/abs/2309.09777)。这类模型可以检验对动作条件和场景演化的统计建模能力，但视觉预测逼真并不等同于“理解”全部物理规律；还需要几何、反事实与闭环驾驶评测。

本节先用运动学区分确定性状态更新与多模态未来，再介绍离散视觉词元、自回归预测和条件扩散，最后实现一个动作条件 Transformer 的最小张量接口。

## 9.5.1 从高中物理到条件概率预测

先看直线运动。汽车在时刻 $t$ 的位置为 $x_t$、速度为 $v_t$；若加速度 $a_t$ 在时间间隔 $\Delta t$ 内不变，则：

$$x_{t+1} = x_t + v_t \Delta t + \frac{1}{2} a_t \Delta t^2$$

这个更新式还隐含了模型假设：一维运动、恒加速度，并忽略其他交通参与者。只有在这些假设成立时，给定状态和动作才得到唯一结果。

真实驾驶中，行人的意图、其他车辆的动作和路面条件都不能被当前传感器完整观测。同一段历史因此可能对应多个合理未来，适合用条件分布表示。

<div align="center">
<img src="/figures/09-spatial-worlds/source/05-driving-world-models/drivewm-fig3.png" alt="Drive-WM 同时预测多相机未来视图，并把动作条件和规划候选纳入统一驾驶世界模型。" width="86%">

_图 9.5-2：Drive-WM 同时预测多相机未来视图，并把动作条件和规划候选纳入统一驾驶世界模型。 出处：Yunze Zhou et al.，[Driving into the Future: Multiview Visual Forecasting and Planning with World Model](https://arxiv.org/abs/2311.17918)（2024），Figure 3。_
</div>

令 $s_t$ 表示模型在 $t$ 时刻使用的场景表示，$a_t$ 表示自车动作。单步预测可写为：

$$P(s_{t+1} \mid s_t, a_t)$$

更一般地，如果我们要预测未来 $T$ 个时间步的状态序列，并考虑到历史状态的影响以及可能的额外上下文信息 $c$（例如天气提示文本、导航路线），根据概率论的链式法则，整个未来序列的联合概率分布可以分解为：

$$P(s_{t+1:t+T} \mid s_{1:t}, a_{1:t+T-1}, c) = \prod_{k=1}^{T} P(s_{t+k} \mid s_{1:t+k-1}, a_{1:t+k-1}, c)$$

<div align="center">
<img src="/figures/09-spatial-worlds/latex/05-driving-world-models/autoregressive-conditioning-window.png" alt="未来状态联合分布逐步分解，每个预测因子的状态与动作条件窗口随预测步增长" width="86%">

_图 9.5-3：第 k 个预测因子只读取生成 s\_{t+k} 之前的状态与动作；预测视界向前推进时，条件窗口也随之增长。本文根据上式绘制。_
</div>

自回归模型可用最大似然学习这些条件分布。需要区分统计预测与因果推断：训练数据中的动作—结果相关性并不自动证明模型掌握了真实因果机制，尤其是在训练分布之外的反事实动作上。

## 9.5.2 多模态隐空间表示

高分辨率视频包含大量像素，直接逐像素预测代价很高。视觉分词器可以把图像压缩成较短的离散词元序列，再由时序模型预测这些词元。压缩会保留训练目标偏好的信息，也可能丢失细小目标和精确几何，因此重建质量要单独检查。

以标量数据为例，假设我们有一个一维连续信号 $x \in \mathbb{R}$，我们希望用有限的离散状态来近似它，最简单的方法就是四舍五入。将其推广到高维向量空间，这就引出了**向量量化（Vector Quantization, VQ）**。

令 $\mathcal{Z} = \{e_1, e_2, \dots, e_K\}$ 为一个包含 $K$ 个可学习向量的编码本（Codebook），其中 $e_i \in \mathbb{R}^D$。给定一张图像 $I_t \in \mathbb{R}^{H \times W \times 3}$，编码器 $E$ 首先将其映射为连续的隐变量特征图 $z_e \in \mathbb{R}^{h \times w \times D}$。

(**对于 $z_e$ 中的每一个空间位置的 $D$ 维向量 $z_e^{(i,j)}$，我们在编码本中寻找与其欧氏距离最近的项：**)

$$z_q^{(i,j)} = e_k, \quad \text{其中 } k = \mathop{\mathrm{argmin}}_{m \in \{1, \dots, K\}} \| z_e^{(i,j)} - e_m \|_2$$

量化后的特征图 $z_q$ 被送入解码器 $D$ 重构原图。由于 $\mathop{\mathrm{argmin}}$ 操作不可导，我们通常使用直通估计器（Straight-Through Estimator）将解码器的梯度直接复制给编码器，并结合承诺损失（Commitment Loss）更新编码本：

$$\mathcal{L}_{\text{VQ}} = \| I_t - D(z_q) \|_2^2 + \beta \| z_e - \text{sg}(z_q) \|_2^2 + \| \text{sg}(z_e) - z_q \|_2^2$$

其中 $\text{sg}(\cdot)$ 表示停止梯度，第一项是重建损失；第二项更新编码器，第三项更新编码本。量化后，每个空间位置对应一个编码本索引，形成整数网格 $S_t \in \{1,\dots,K\}^{h\times w}$。

## 9.5.3 时序演化：动作条件下的自回归建模

获得视觉词元、动作表示和文本条件后，可以把预测写成序列建模问题。

设在时刻 $t$，环境状态的离散 Token 序列为 $Z_t$，前置摄像头的当前视野与自车运动状态相互交织。我们将文本条件表示为 $C$，过去的动作序列表示为 $A_{<t}$，过去的视觉状态序列表示为 $Z_{<t}$。

在自回归 Transformer 中，预测下一个 Token $z_i$ 的对数概率可以写为：

$$\log P(Z_t \mid Z_{<t}, A_{<t}, C) = \sum_{i=1}^{|Z_t|} \log P(z_i \mid z_{<i}, Z_{<t}, A_{<t}, C)$$

为了实现这一点，GAIA-1 架构将过去的信息 $Z_{<t}$ 和 $A_{<t}$ 作为 Context，通过因果注意力掩码（Causal Attention Mask），保证在预测 $t$ 时刻的未来状态时，模型只能看到时刻 $t$ 之前以及时刻 $t$ 之前发生的操作。

<div align="center">
<img src="/figures/09-spatial-worlds/source/05-driving-world-models/gaia1-fig2.png" alt="GAIA-1 将视频、动作与文本编码为序列，由世界模型自回归预测未来离散视觉标记。" width="86%">

_图 9.5-4：GAIA-1 将视频、动作与文本编码为序列，由世界模型自回归预测未来离散视觉标记。 出处：Anthony Hu et al.，[GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080)（2023），Figure 2。_
</div>

具体到张量维度的推演：
假设我们在自回归模型中输入了长度为 $L$ 的序列表示 $X \in \mathbb{R}^{L \times d_{\text{model}}}$。在线性投影得到 Query, Key, Value 矩阵后：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

因果注意力写为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

其中，若查询位置 $i$ 试图读取未来位置 $j>i$，则 $M_{i,j}=-\infty$；否则为 0。这个掩码阻止训练时的信息泄漏，但不会保证生成结果符合物理时间演化。

## 9.5.4 扩散模型视角下的世界模型 (DriveDreamer)

与 GAIA-1 纯粹离散自回归的思路不同，DriveDreamer 等工作则探索了基于潜在扩散模型（Latent Diffusion Models, LDM）的世界模型范式。扩散模型并不像自回归模型那样逐个 Token 预测，而是一次性对整个连续的隐空间张量进行迭代去噪。

<div align="center">
<img src="/figures/09-spatial-worlds/source/05-driving-world-models/drivedreamer-fig3.png" alt="DriveDreamer 把结构化交通条件、驾驶动作与扩散生成器组合起来，生成受控未来驾驶画面。" width="86%">

_图 9.5-5：DriveDreamer 把结构化交通条件、驾驶动作与扩散生成器组合起来，生成受控未来驾驶画面。 出处：Xiaofeng Wang et al.，[DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving](https://arxiv.org/abs/2309.09777)（2023），Figure 3。_
</div>

假设 $z_0$ 为未来驾驶视频的真实隐状态。扩散过程向其中逐步添加高斯噪声，在步数 $n$ 的边缘分布满足：

$$q(z_n \mid z_0) = \mathcal{N}(z_n; \sqrt{\bar{\alpha}_n} z_0, (1 - \bar{\alpha}_n)\mathbf{I})$$

而在逆向生成（去噪）阶段，神经网络 $\epsilon_\theta$ 的目标是预测添加的噪声。为了让世界模型理解**动作指令**和**历史状态**，我们将它们作为额外的条件注入到网络中：

$$\mathcal{L}_{\text{Diffusion}} = \mathbb{E}_{z_0, \epsilon, n} \left[ \| \epsilon - \epsilon_\theta(z_n, n, C_{\text{text}}, C_{\text{action}}, Z_{<t}) \|_2^2 \right]$$

动作和历史可通过交叉注意力或特征调制进入去噪网络。模型会学习提高条件一致性，但仍可能忽略动作、生成不可能的运动或遗漏小目标。

## 9.5.5 代码实现

下面实现一个简化的动作条件 Transformer，演示视觉词元、动作嵌入和因果掩码如何组合。它没有图像分词器、文本编码器或视频解码器。

```python
import torch
from torch import nn
from torch.nn import functional as F
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
        B, T, C = x.size()  # 批次、序列长度、嵌入维度

        # 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # 形状变换以支持多头注意力: (B, T, n_heads, C // n_heads) -> (B, n_heads, T, C // n_heads)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # PyTorch 内置缩放点积注意力生成因果掩码
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
        self.action_proj = nn.Linear(2, d_model)  # 假设动作维度为 2

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
        if T > self.pos_emb.shape[1]:
            raise ValueError("序列长度超过 max_seq_len")

        tok_embeddings = self.token_emb(idx)  # (B, T, d_model)
        pos_embeddings = self.pos_emb[:, :T, :]  # (1, T, d_model)

        # 动作特征映射
        act_embeddings = self.action_proj(actions)  # (B, T, d_model)

        # 将视觉状态与对应的动作特征在隐空间相加（代表在对应状态下施加了该动作）
        # 这是一种将条件注入模型的标准多模态融合方式
        x = tok_embeddings + act_embeddings + pos_embeddings

        # 通过 Transformer 块进行时序演化
        x = self.blocks(x)
        x = self.ln_f(x)

        # 预测下一时刻的 Token 对数概率 (Logits)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
```

下面用随机输入检查张量形状。

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

输出形状是 `(B, T, vocab_size)`。只有在训练时把标签向后错开一位，这些 logits 才对应“下一词元”预测；调用 `softmax` 后才得到词表上的概率。随机初始化模型只验证接口，不具备驾驶预测能力。

## 9.5.6 小结

- 驾驶未来具有多模态性，适合用条件分布而不是单一确定值表示。
- 离散词元降低序列建模成本，但会引入量化与重建误差。
- 自回归模型逐词元预测，扩散模型迭代去噪；两者都可以加入动作与文本条件。
- 生成质量、几何一致性、动作响应和闭环安全需要分别评测。

## 9.5.7 练习

1. 在 VQ 损失中，两个 `sg()` 分别阻断哪条梯度路径？去掉它们后，编码器与编码本会收到怎样不同的梯度？
2. 我们在代码中使用加法（`tok_embeddings + act_embeddings`）来融合状态和动作。除此之外，还有哪些将额外条件注入 Transformer 的数学方法？
   _提示：可以回顾我们在之前的章节中提到的 Cross-Attention 机制（如在扩散模型中被广泛应用），或者考虑特征的通道拼接（Concatenation）。_
3. 若用世界模型评估动作风险，需要从条件分布采样多条未来，并定义碰撞判定器。写出碰撞概率的蒙特卡洛估计式。
