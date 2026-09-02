# 2.2 循环神经网络与因果 Transformer (RNN & Transformer)

在世界模型与机器人控制的物理建模中，世界从来不是由一张张孤立割裂的静态快照构成的，而是一条沿着单向时间之箭连续奔涌的物理事件流。

当一个棒球运动员在空中挥棒击球时，他大脑所依据的绝不仅仅是棒球在视网膜上当前的孤立像素位置，而是根据过去半秒钟内棒球的飞行弧线（时序速度与加速度），精准预判出棒球在未来零点几秒后将飞抵的击打甜点位置。

在时序动力学建模的演进历程中，深度学习探索了两大截然不同但又交相辉映的建模范式：
- **循环神经网络（Recurrent Neural Networks, RNN / GRU / LSTM）**：以马尔可夫循环递推为核心，将无限长度的历史压缩在固定维度的隐藏状态中，具备极高的常数级推理吞吐；
- **因果 Transformer（Causal Transformer, GPT 架构）**：以时间因果掩码和全维自注意力为核心，彻底消除了时间维度的串行瓶颈，支持海量时序数据的并行训练。

本节我们将从初等数列递推与因果截断出发，严密推导 RNN 的梯度指数衰减与因果注意力的掩码机制，并使用纯底层 PyTorch 从零手写 GRU 循环单元与因果 Transformer 解码器。

<div align="center">

<img src="/figures/02-foundations/source/02-rnn-and-transformer/transformer-fig1.png" alt="标准 Transformer 模型架构：编码器-解码器结构配合多头自注意力与前馈全连接层。" width="86%">

_图 2.2-1：标准 Transformer 模型架构：编码器-解码器结构配合多头自注意力与前馈全连接层。 出处：[Attention Is All You Need，Ashish Vaswani et al.，2017](https://arxiv.org/abs/1706.03762)。_

</div>

---

## 2.2.1 物理与时序基石：因果单向性与历史信息的压缩

要理解时序建模的数学约束，我们首先必须回到物理世界的根本法则——**时间因果性（Temporal Causality）**。

### 1. 时间的单向因果约束
在经典物理世界中，因果律具有严格的方向性：
- 现在的物理状态 $\mathbf{s}_t$ 严格由过去的状态 $\mathbf{s}_{<t}$ 与过去的动作 $\mathbf{a}_{<t}$ 共同决定；
- 但在预测时刻 $t$ 的状态时，模型**绝不允许偷窥未来时刻 $t+1$ 的任何信息**。
如果算法在训练时不小心看到了未来帧（发生未来信息泄露），在真实物理世界实时部署时，由于未来数据根本不存在，策略将发生灾难性瘫痪。

### 2. 马尔可夫压缩 vs 全历史检索
- **RNN 的隐状态压缩**：试图将 $0$ 到 $t$ 时刻的所有历史经验强制压缩到一个定长的向量 $\mathbf{h}_t \in \mathbb{R}^d$ 中。其优点是内存消耗固定，缺点是时间一长，久远的历史细节会被后续输入逐渐冲刷遗忘；
- **Transformer 的全历史检索**：保留历史的所有词元，在每一步通过自注意力重新审视整个历史窗口。其优点是长程记忆精准无损，缺点是序列越长计算量呈二次方增长。

<div align="center">

<img src="/figures/02-foundations/latex/02-rnn-and-transformer/rnn-jacobian-product.png" alt="沿时间反向传播 (BPTT) 的长程梯度链式传递：多个局部雅可比矩阵相乘引发指数级衰减或爆炸" width="86%">

_图 2.2-2：沿时间反向传播 (BPTT) 的长程梯度链式传递：多个局部雅可比矩阵相乘引发指数级衰减或爆炸。_

</div>

---

## 2.2.2 核心数学推导一：RNN 的递推状态更新与 BPTT 梯度衰减

经典 RNN 在离散时间步 $t$ 上的前向递推方程为：

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

$$\hat{\mathbf{y}}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$

### 1. 沿时间反向传播（Backpropagation Through Time, BPTT）
设在最终时刻 $T$ 产生的损失为 $\mathcal{L}_T$。为了计算该损失对初始时刻隐状态 $\mathbf{h}_1$ 的导数，梯度必须沿着时间链条逆向回传：

$$\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T} \prod_{k=2}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}}$$

其中每一步的局部雅可比矩阵为：

$$\frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \text{diag}(1 - \mathbf{h}_k^2) \cdot \mathbf{W}_{hh}^\top$$

### 2. 梯度指数衰减手算数值算例
设隐状态为标量（维度为 1），状态转移权重 $W_{hh} = 0.5$。激活函数导数取最大值 $1 - h_k^2 \approx 1.0$。
如果时间跨度为 $T = 6$ 步，我们来手动计算局部雅可比矩阵的连乘积：

$$\frac{\partial h_6}{\partial h_1} = \prod_{k=2}^6 (1.0 \times 0.5) = (0.5)^5 = \frac{1}{32} = 0.03125$$

如果时间跨度拉长到 $T = 20$ 步：
$$\frac{\partial h_{20}}{\partial h_1} = (0.5)^{19} \approx 0.0000019$$

初等代数的直观指数运算清晰揭示：仅仅过了 20 个时间步，初始时刻的梯度信号就衰减了 **50 万倍**，导致网络根本无法根据 20 步之前的历史错误来调整初始权重！这一“梯度消失”正是推动 GRU 门控机制与 Transformer 诞生的核心驱动力。

<details>
<summary><b>深入推导：基于矩阵谱半径（Spectral Radius）的 BPTT 梯度指数级爆炸与衰减严格数学证明（点击展开查看完整推导）</b></summary>

设权重矩阵 $\mathbf{W}_{hh}$ 的最大特征值（谱半径）为 $\rho(\mathbf{W}_{hh})$，对角激活导数上界为 $\gamma = \sup_x |\tanh'(x)| = 1.0$。
根据矩阵范数三角不等式与相容性条件，时间连乘梯度范数满足上界：
$$\left\| \prod_{k=2}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} \right\| \le \prod_{k=2}^T \left\| \text{diag}(1 - \mathbf{h}_k^2) \right\| \cdot \|\mathbf{W}_{hh}^\top\| \le (\gamma \cdot \|\mathbf{W}_{hh}\|)^{T-1}$$
- 若 $\|\mathbf{W}_{hh}\| < 1$，当 $T \to \infty$ 时，梯度上界以指数速度收敛于 0（梯度消失）；
- 若 $\|\mathbf{W}_{hh}\| > 1$，当激活函数处于线性区时，梯度将以几何级数无界发散（梯度爆炸）。
</details>

---

## 2.2.3 核心数学推导二：因果掩码自注意力（Causal Masked Self-Attention）

为了既保留 Transformer 的并行计算能力，又严格遵守物理因果单向律，GPT 架构引入了**下三角因果掩码（Causal Masking）**。

<div align="center">

<img src="/figures/02-foundations/source/02-rnn-and-transformer/gru-fig2.png" alt="GPT 采用多层因果掩码自注意力解码器实现无监督预训练与自回归文本/动作序列生成。" width="86%">

_图 2.2-3：GPT 采用多层因果掩码自注意力解码器实现无监督预训练与自回归文本/动作序列生成。 出处：[Improving Language Understanding by Generative Pre-Training，Alec Radford et al.，2018](https://openai.com/research/language-unsupervised)。_

</div>

### 1. 因果掩码矩阵定义
设序列长度为 $L$。我们构造一个 $L \times L$ 的加性因果掩码矩阵 $\mathbf{M}$：

$$\mathbf{M}[i, j] = \begin{cases} 0, & i \ge j \quad (\text{第 } i \text{ 步可以关注过去及当前步 } j) \\ -\infty, & i < j \quad (\text{第 } i \text{ 步绝不允许偷窥未来步 } j) \end{cases}$$

因果自注意力公式写作：

$$\text{CausalAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V}$$

因为在 Softmax 中 $e^{-\infty} = 0$，所有未来位置的注意力权重被数学上绝对截断为严格的 $0$！

### 2. 因果 Softmax 手算数值算例
设一个长度为 $L = 3$ 的时序序列，计算出的原始点积得分矩阵 $\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}$ 为：

$$\mathbf{S} = \begin{bmatrix} 2.0 & 5.0 & 8.0 \\ 1.0 & 3.0 & 6.0 \\ 0.0 & 2.0 & 4.0 \end{bmatrix}$$

加上因果掩码矩阵 $\mathbf{M} = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}$ 后得到：

$$\mathbf{S}_{\text{masked}} = \begin{bmatrix} 2.0 & -\infty & -\infty \\ 1.0 & 3.0 & -\infty \\ 0.0 & 2.0 & 4.0 \end{bmatrix}$$

我们来逐行执行 Softmax 归一化计算：
1. **第 1 行（时刻 1）**：只能看自己
   $$\mathbf{A}[1, :] = [\frac{e^2}{e^2 + 0 + 0}, 0, 0] = [1.0, 0.0, 0.0]$$
2. **第 2 行（时刻 2）**：可以看时刻 1 和 2（设 $e^1 \approx 2.718, e^3 \approx 20.0855$）
   $$\mathbf{A}[2, :] = [\frac{2.718}{2.718 + 20.0855}, \frac{20.0855}{2.718 + 20.0855}, 0] \approx [0.119, 0.881, 0.0]$$
3. **第 3 行（时刻 3）**：可以纵览全部历史 1、2、3
   $$\mathbf{A}[3, :] = [\frac{e^0}{e^0 + e^2 + e^4}, \frac{e^2}{e^0 + e^2 + e^4}, \frac{e^4}{e^0 + e^2 + e^4}] \approx [0.016, 0.117, 0.867]$$

最终形成的注意力权重矩阵为严格的下三角矩阵：
$$\mathbf{A} = \begin{bmatrix} 1.000 & 0.000 & 0.000 \\ 0.119 & 0.881 & 0.000 \\ 0.016 & 0.117 & 0.867 \end{bmatrix}$$

初等代数的几步推导清晰展现了因果掩码的妙处：未来的信息权重被完全阻断为 0，而模型在训练时却可以一次性并行计算出所有时间步的损失！

<details>
<summary><b>深入推导：因果掩码注意力在时间轴拓扑偏序流形上的信息不可逆性证明（点击展开查看完整推导）</b></summary>

将时序序列视为定义在有限偏序集 $(\mathcal{T}, \le)$ 上的离散图结构。
注意力亲和度矩阵 $\mathbf{A} \in \mathbb{R}^{L \times L}$ 为该有向无环图（DAG）的邻接矩阵。
由于因果掩码满足 $\forall i < j, \mathbf{A}_{i, j} = 0$，矩阵 $\mathbf{A}$ 严格为下三角矩阵，其所有特征值全为对角线元素 $\mathbf{A}_{i, i} > 0$。
系统的互信息传递满足 $I(\mathbf{X}_{\ge t}; \mathbf{H}_t \mid \mathbf{X}_{\le t}) = 0$，严格证明了因果注意力在信息论意义上杜绝了一切未来信息倒流。
</details>

---

## 2.2.4 纯底层 PyTorch 代码实现：从零手写 GRU 循环单元与因果 Transformer

下面我们使用纯底层 PyTorch 算子实现一个标准的门控循环单元（GRU）与带因果掩码的 Transformer 自回归解码网络。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGRUCell(nn.Module):
    """
    纯手写门控循环单元 (GRU Cell)
    r_t = sigmoid(W_xr * x + W_hr * h)
    z_t = sigmoid(W_xz * x + W_hz * h)
    n_t = tanh(W_xn * x + r_t * (W_hn * h))
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 重置门 (r) 与更新门 (z) 权重
        self.w_xr = nn.Linear(input_dim, hidden_dim)
        self.w_hr = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_xz = nn.Linear(input_dim, hidden_dim)
        self.w_hz = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 候选隐藏状态 (n) 权重
        self.w_xn = nn.Linear(input_dim, hidden_dim)
        self.w_hn = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, input_dim) 当前输入
        :param h_prev: (B, hidden_dim) 上一时刻隐状态
        :return: (B, hidden_dim) 新隐状态
        """
        r = torch.sigmoid(self.w_xr(x) + self.w_hr(h_prev))
        z = torch.sigmoid(self.w_xz(x) + self.w_hz(h_prev))
        n = torch.tanh(self.w_xn(x) + self.w_hn(r * h_prev))
        h_new = (1.0 - z) * n + z * h_prev
        return h_new

class CausalTransformerDecoder(nn.Module):
    """
    因果自回归 Transformer 解码器
    内置下三角因果掩码，防止未来信息泄露
    """
    def __init__(self, vocab_size: int = 100, d_model: int = 64, n_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, token_seq: torch.Tensor) -> torch.Tensor:
        """
        :param token_seq: (B, seq_len) 离散词元索引序列
        :return: (B, seq_len, vocab_size) 下一步预测的 Logits
        """
        B, L = token_seq.shape
        x = self.tok_embed(token_seq) + self.pos_embed[:, :L, :]

        # 构造下三角因果掩码 (矩阵严格满足未来位置为 -inf)
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=token_seq.device), diagonal=1
        )

        hidden = self.transformer(x, mask=causal_mask)
        logits = self.head(hidden)
        return logits

# ===================================================================
# 单元测试与因果掩码非泄露校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    hidden_dim = 32

    # 1. 测试手写 GRU Cell 循环推进
    gru_cell = SimpleGRUCell(input_dim=16, hidden_dim=hidden_dim)
    h_state = torch.zeros(batch_size, hidden_dim)

    for t in range(seq_len):
        dummy_x = torch.randn(batch_size, 16)
        h_state = gru_cell(dummy_x, h_state)

    print(f"[GRU Test] 推进 {seq_len} 步后隐藏状态形状: {h_state.shape}")
    assert h_state.shape == (batch_size, hidden_dim), "GRU 隐藏状态维度不符！"

    # 2. 测试因果 Transformer 预测
    gpt_model = CausalTransformerDecoder(vocab_size=100, d_model=64)
    dummy_tokens = torch.randint(0, 100, (batch_size, seq_len))
    logits = gpt_model(dummy_tokens)

    print(f"[Transformer Test] 输入词元序列形状: {dummy_tokens.shape}")
    print(f"[Transformer Test] 输出 Logits 形状: {logits.shape}")

    assert logits.shape == (batch_size, seq_len, 100), "因果解码器输出形状不符！"
    assert not torch.isnan(logits).any(), "因果注意力计算出现 NaN！"
    print("✓ 手写 GRU 循环单元与因果 Transformer 解码器单测全部通过！")
```

---

## 2.2.5 本节小结

回顾本节内容，我们建立了时序序列建模的核心图谱：
1. **时间单向性**：物理因果律决定了模型必须严格基于历史预测未来，因果掩码是杜绝未来信息泄露的数学屏障；
2. **RNN 与 Transformer 的权衡**：RNN 具备 $\mathcal{O}(1)$ 常数推理内存但受困于 BPTT 梯度消失，因果 Transformer 支持高并发自回归训练但面临二次方上下文开销；
3. **世界模型动力学基石**：现代世界模型（如 RSSM）巧妙结合了 GRU 的高效紧凑循环状态与 Transformer 的长程全局关联，奠定了预测未来的认知底座。
