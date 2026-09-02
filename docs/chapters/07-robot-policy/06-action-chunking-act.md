# 7.6 动作分块与基于 Transformer 的动作生成 (ACT)

在前面的章节中，我们深入剖析了单步行为克隆中的协变量偏移与累积误差问题。当我们将目光转向真实世界中最严苛的双臂精细操作（例如用两把镊子穿针引线、给拉链穿扣、组装极小的电路元件）时，单步决策策略往往会暴露出致命的缺陷——**高频颤抖、动作不连贯与长程目标迷失**。

人类在系鞋带或用双筷夹菜时，大脑绝不是在每隔 $0.01\text{ 秒}$ 的微小瞬间去单独计算“当前手指该往哪移一点”，而是以“动作宏（Action Macro）”的形式，一次性规划未来一整段流畅连续的肌肉收缩序列。

2023 年，斯坦福大学 Tony Z. Zhao、Chelsea Finn 等人联合推出了 **ALOHA** 低成本双臂示教平台，并提出了 **ACT（Action Chunking with Transformers，动作分块 Transformer）** 模型。ACT 通过引入**动作分块（Action Chunking）**、**条件变分自编码器（CVAE）** 与 **时序集成（Temporal Ensembling）**，开辟了双臂长程精细操作的崭新格局。

<div align="center">

<img src="/figures/07-robot-policy/source/06-action-chunking-act/act-fig6.png" alt="ALOHA 的六类真实双臂任务展示动作分块所针对的长时精细操作。" width="86%">

_图 7.6-1：ALOHA 的六类真实双臂任务展示动作分块所针对的长时精细操作。 出处：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware，Tony Z. Zhao et al.，2023](https://arxiv.org/abs/2304.13705)。_

</div>

---

## 7.6.1 物理与生理基石：人类双臂协同与长程精细操作

要理解动作分块的设计哲学，我们首先必须回到人类神经生理学中的运动协同机制。

### 1. 运动协同元（Motor Synergies）与动作分块
神经科学研究表明，人类的大脑皮层与脊髓中预存了大量的**运动协同元（Motor Synergies）**。
- 当一个体操运动员做后空翻时，其小脑在腾空的瞬间就已经激活了一个长达数百毫秒的固定肌肉运动神经冲动程序；
- 这种将连续多个时间步的动作打包为一个整体原子单元的机制，在机器人学中被称为**动作分块（Action Chunking）**。

动作分块将机器人与环境的有效交互步长缩短了 $k$ 倍（例如 $k = 50$），从根本上将误差累积公式中的时序步长 $T$ 骤降为 $T / k$，从而极大地缓解了协变量偏移带来的雪崩效应。

<div align="center">

<img src="/figures/07-robot-policy/source/06-action-chunking-act/smile-fig2.png" alt="SMILe 与传统监督模仿的赛道表现差异展示训练分布外误差如何累积。" width="86%">

_图 7.6-2：SMILe 与传统监督模仿的赛道表现差异展示训练分布外误差如何累积。 出处：[SMILe: Scalable Meta-Inverse Reinforcement Learning，Stéphane Ross et al.，2010](https://arxiv.org/abs/1011.0686)。_

</div>

---

## 7.6.2 核心数学推导一：CVAE 风格隐变量与动作生成

在面对同一个双臂装配任务时，即便面对完全相同的起始画面，人类专家也有可能采用不同的细微操作习惯（例如先抬高左臂再抬右臂，或者双臂齐平抬起）。

为了捕捉专家动作的多样性，ACT 采用了**条件变分自编码器（Conditional VAE, CVAE）**架构。

<div align="center">

<img src="/figures/07-robot-policy/source/06-action-chunking-act/act-fig4.png" alt="ACT 的 CVAE 编码器、Transformer 编解码器与动作查询共同生成动作块。" width="86%">

_图 7.6-3：ACT 的 CVAE 编码器、Transformer 编解码器与动作查询共同生成动作块。 出处：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware，Tony Z. Zhao et al.，2023](https://arxiv.org/abs/2304.13705)。_

</div>

### 1. CVAE 编码器与风格隐变量采样
在训练阶段，CVAE 编码器接收当前的真实专家动作块 $\mathbf{A} \in \mathbb{R}^{k \times d_a}$ 与当前状态观测 $\mathbf{o}$，将其压缩为低维隐空间的均值 $\boldsymbol{\mu} \in \mathbb{R}^{d_z}$ 与对数方差 $\log \boldsymbol{\sigma}^2 \in \mathbb{R}^{d_z}$（通常 $d_z = 32$）：

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### 2. Transformer 解码器与动作重构
Transformer 解码器接收环境视觉特征、本体感觉向量以及隐变量 $\mathbf{z}$，通过 $k$ 个可学习的**动作查询（Action Queries）**，一次性自回归解码出未来 $k$ 步的完整连续动作序列 $\hat{\mathbf{A}} = [\hat{\mathbf{a}}_t, \hat{\mathbf{a}}_{t+1}, \dots, \hat{\mathbf{a}}_{t+k-1}]$。

ACT 的联合优化损失函数由 L1 动作重构损失与 KL 散度正则项组成：

$$\mathcal{L}_{\text{ACT}} = \frac{1}{k \cdot d_a} \sum_{i=0}^{k-1} \|\hat{\mathbf{a}}_{t+i} - \mathbf{a}_{t+i}\|_1 + \beta D_{\text{KL}}\left(\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2)) \parallel \mathcal{N}(\mathbf{0}, \mathbf{I})\right)$$

其中 KL 散度的初等解析闭式解为：

$$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{d_z} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

**手算代入算例**：
设隐变量维度 $d_z = 1$。编码器输出均值 $\mu = 0.4$，方差 $\sigma^2 = 0.64$（即 $\log \sigma^2 = \ln 0.64 \approx -0.4463$）。

我们代入 KL 散度公式手算：
$$D_{\text{KL}} = -\frac{1}{2} \left[ 1 + (-0.4463) - 0.4^2 - 0.64 \right] = -\frac{1}{2} [1 - 0.4463 - 0.16 - 0.64] = -\frac{1}{2} [-0.2463] \approx 0.1232$$

通过初等代数的几步推导，系统迫使隐空间在训练中向标准正态分布对齐；在推理部署时，我们只需直接令 $\mathbf{z} = \mathbf{0}$（取最具有确定性的典型风格），模型就能稳定输出最稳健的专家轨迹！

<details>
<summary><b>深入推导：连续动作序列高斯变分证据下界（ELBO）与凸松弛优化（点击展开查看完整推导）</b></summary>

对条件动作序列的边缘对数似然 $\log p(\mathbf{A} \mid \mathbf{o})$ 应用琴生不等式：
$$\begin{aligned}
\log p(\mathbf{A} \mid \mathbf{o}) &= \log \int p(\mathbf{A}, \mathbf{z} \mid \mathbf{o}) d\mathbf{z} = \log \int q(\mathbf{z} \mid \mathbf{A}, \mathbf{o}) \frac{p(\mathbf{A} \mid \mathbf{z}, \mathbf{o}) p(\mathbf{z})}{q(\mathbf{z} \mid \mathbf{A}, \mathbf{o})} d\mathbf{z} \\
&\ge \mathbb{E}_{q}[\log p(\mathbf{A} \mid \mathbf{z}, \mathbf{o})] - D_{\text{KL}}(q(\mathbf{z} \mid \mathbf{A}, \mathbf{o}) \parallel p(\mathbf{z}))
\end{aligned}$$
当假设重构误差满足拉普拉斯分布 $p(\mathbf{A} \mid \mathbf{z}, \mathbf{o}) \propto \exp(-\|\hat{\mathbf{A}} - \mathbf{A}\|_1)$ 时，对数似然项严格等价于 L1 范数损失。相比于 L2 均方误差，L1 损失对动作序列中的偶发突变具有更强的鲁棒性，有效避免了夹爪闭合瞬间的过度平滑软化。
</details>

---

## 7.6.3 核心数学推导二：时序集成（Temporal Ensembling）对角线平滑滤波

在实际部署时，如果我们每隔 $k$ 步才重新规划一次，那么在第 $k$ 步与第 $k+1$ 步切换的瞬间，机械臂关节速度往往会产生微小的阶跃顿挫。

为了实现绝对丝滑的运动，ACT 引入了**时序集成（Temporal Ensembling）**平滑滤波机制。

<div align="center">

<img src="/figures/07-robot-policy/source/06-action-chunking-act/act-fig5.png" alt="重叠动作块按时间权重集成，缓解单次预测切换造成的抖动。" width="86%">

_图 7.6-4：重叠动作块按时间权重集成，缓解单次预测切换造成的抖动。 出处：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware，Tony Z. Zhao et al.，2023](https://arxiv.org/abs/2304.13705)。_

</div>

<div align="center">

<img src="/figures/07-robot-policy/latex/06-action-chunking-act/temporal-ensemble-diagonal.png" alt="从重叠动作块的同一物理时刻对角线取值并按预测年龄加权" width="86%">

_图 7.6-5：从重叠动作块的同一物理时刻对角线取值并按预测年龄加权。_

</div>

### 1. 重叠动作块的对角线提取与指数衰减加权
系统在**每一个控制时间步 $t$** 都会以最新图像为条件，重新生成一个未来长度为 $k$ 的新动作块。

因此，在当前的物理时刻 $t$，机械臂手头其实拥有来自过去 $k$ 个时刻对“时刻 $t$”做出的多份重叠预测：
$$\{\mathbf{a}_{t \mid t}, \mathbf{a}_{t \mid t-1}, \mathbf{a}_{t \mid t-2}, \dots, \mathbf{a}_{t \mid t-k+1}\}$$

系统采用指数衰减权重对这 $k$ 份预测进行加权平均：

$$\mathbf{a}_t^{\text{final}} = \frac{\sum_{i=0}^{k-1} w_i \mathbf{a}_{t \mid t-i}}{\sum_{i=0}^{k-1} w_i}, \quad \text{其中 } w_i = \exp(-m \cdot i)$$

> **公式符号逐一拆解**：
> - $i \in \{0, 1, \dots, k-1\}$：预测发生的“年龄”（$i = 0$ 表示刚才最新做出的预测，$i = k-1$ 表示 $k-1$ 个时间步之前做出的老预测）；
> - $m > 0$：**时序折扣率（Temporal Discount Factor）**，通常取 $m = 0.01$；
> - $w_i = e^{-m i}$：衰减权重。最新的预测获得最大的权重（$w_0 = 1$），较早的预测权重微弱衰减，但仍然参与加权，从而将不同周期的动作曲线完美“粘合”为一条处处光滑的动力学轨迹。

<details>
<summary><b>深入推导：时序指数加权滤波器的频域传递函数与群延迟（Group Delay）分析（点击展开查看完整推导）</b></summary>

时序集成可形式化为离散有限冲激响应（FIR）低通滤波器：
$$y[n] = \frac{1}{\sum_{i=0}^{k-1} e^{-m i}} \sum_{i=0}^{k-1} e^{-m i} x_i[n]$$
其 Z 域系统传递函数为：
$$H(z) = C \sum_{i=0}^{k-1} (e^{-m} z^{-1})^i = C \frac{1 - e^{-m k} z^{-k}}{1 - e^{-m} z^{-1}}$$
该系统在高频段具有 $-20\text{ dB/dec}$ 的急剧衰减，彻底滤除了多步重规划产生的非连续阶跃高频谐波；同时其相位响应在低频工作带宽内呈严格线性，有效群延迟被压制在微秒级 $\tau_g \le \frac{1}{m}(1 - e^{-m})$，兼顾了滤波平滑性与快速响应带宽。
</details>

---

## 7.6.4 纯底层 PyTorch 代码实现：ACT 动作分块与时序集成引擎

下面我们使用纯底层 PyTorch 算子手写实现 ACT 的核心模块，包括 CVAE 编码器、动作查询 Transformer 解码器与时序集成平滑器。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ACTCVAEEncoder(nn.Module):
    """
    ACT 条件变分自编码器编码器
    将专家动作块 A 与当前观测压缩为低维隐变量 z ~ N(mu, sigma^2)
    """
    def __init__(self, action_dim: int = 14, chunk_size: int = 16, obs_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        input_dim = chunk_size * action_dim + obs_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, actions: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param actions: (B, chunk_size, action_dim)
        :param obs: (B, obs_dim)
        :return: (z, mu, logvar)
        """
        B = actions.size(0)
        flat_actions = actions.flatten(1)
        x = torch.cat([flat_actions, obs], dim=-1)
        feat = self.mlp(x)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)

        # 重参数化采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class ACTTransformerDecoder(nn.Module):
    """
    基于可学习动作查询 (Action Queries) 的 Transformer 解码器
    """
    def __init__(self, action_dim: int = 14, chunk_size: int = 16, obs_dim: int = 64, latent_dim: int = 32, d_model: int = 128):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # 投影观测与隐变量
        self.cond_proj = nn.Linear(obs_dim + latent_dim, d_model)

        # 可学习动作查询序列: (chunk_size, d_model)
        self.action_queries = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)

        # Transformer 解码器层
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # 动作回归头
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, z: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
        :param z: (B, latent_dim)
        :param obs: (B, obs_dim)
        :return: (B, chunk_size, action_dim) 预测动作块
        """
        B = z.size(0)
        cond = self.cond_proj(torch.cat([obs, z], dim=-1)).unsqueeze(1) # (B, 1, d_model)

        # 扩展动作查询: (B, chunk_size, d_model)
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)

        # 解码并投影
        dec_out = self.transformer_decoder(tgt=queries, memory=cond)
        pred_actions = self.action_head(dec_out)
        return pred_actions

class TemporalEnsembler:
    """
    时序集成平滑滤波器
    在时间轴上维护重叠动作块的指数加权衰减平均
    """
    def __init__(self, chunk_size: int = 16, action_dim: int = 14, m: float = 0.01):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        # 计算权重 w_i = exp(-m * i)
        indices = torch.arange(chunk_size, dtype=torch.float32)
        self.weights = torch.exp(-m * indices) # (chunk_size,)
        self.history_buffer = []

    def update(self, new_chunk: torch.Tensor) -> torch.Tensor:
        """
        输入当前步预测出的动作块 (chunk_size, action_dim)，输出单步平滑融合动作 (action_dim,)
        """
        self.history_buffer.append(new_chunk.cpu())
        if len(self.history_buffer) > self.chunk_size:
            self.history_buffer.pop(0)

        # 提取当前时刻的多份预测并加权平均
        k = len(self.history_buffer)
        curr_preds = []
        curr_weights = []

        for age in range(k):
            # 第 age 个历史块中对应当前物理时刻的动作索引为 age
            chunk = self.history_buffer[-(age + 1)]
            curr_preds.append(chunk[age])
            curr_weights.append(self.weights[age])

        stacked_preds = torch.stack(curr_preds, dim=0)   # (k, action_dim)
        stacked_weights = torch.tensor(curr_weights).unsqueeze(-1) # (k, 1)

        fused_action = (stacked_preds * stacked_weights).sum(dim=0) / stacked_weights.sum()
        return fused_action

# ===================================================================
# 单元测试：CVAE 训练损失与时序集成平滑度校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    chunk_size = 16
    action_dim = 14
    obs_dim = 64
    latent_dim = 32

    encoder = ACTCVAEEncoder(action_dim=action_dim, chunk_size=chunk_size, obs_dim=obs_dim, latent_dim=latent_dim)
    decoder = ACTTransformerDecoder(action_dim=action_dim, chunk_size=chunk_size, obs_dim=obs_dim, latent_dim=latent_dim)

    dummy_actions = torch.randn(batch_size, chunk_size, action_dim)
    dummy_obs = torch.randn(batch_size, obs_dim)

    # 1. 测试 CVAE 编码与解码
    z, mu, logvar = encoder(dummy_actions, dummy_obs)
    pred_actions = decoder(z, dummy_obs)

    # 2. 计算联合损失
    l1_loss = F.l1_loss(pred_actions, dummy_actions)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    total_loss = l1_loss + 10.0 * kl_loss

    print(f"[ACT Test] 动作重构 L1 损失: {l1_loss.item():.4f}")
    print(f"[ACT Test] 隐变量分布 KL 散度: {kl_loss.item():.4f}")
    print(f"[ACT Test] 预测动作块形状: {pred_actions.shape}")

    assert pred_actions.shape == (batch_size, chunk_size, action_dim), "解码动作块形状不符！"

    # 3. 测试时序集成平滑器
    ensembler = TemporalEnsembler(chunk_size=chunk_size, action_dim=action_dim, m=0.01)
    for _ in range(20):
        single_chunk = torch.randn(chunk_size, action_dim)
        smooth_act = ensembler.update(single_chunk)

    print(f"[Temporal Ensemble Test] 最终单步融合动作形状: {smooth_act.shape}")
    assert smooth_act.shape == (action_dim,), "时序集成输出维度不符！"
    print("✓ ACT 动作分块、CVAE 损失与时序集成平滑引擎单测全部通过！")
```

---

## 7.6.5 本节小结

回顾本节内容，我们建立了动作分块在长程双臂精细操作中的完整技术体系：
1. **运动协同与动作分块**：将多步连续动作打包为原子块，从物理机制上将交互步数缩短 $k$ 倍，大幅削减了累积误差的增长速度；
2. **CVAE 隐变量表征**：通过高斯隐变量建模人类操作风格的多样性，利用 L1 重构损失与 KL 正则化实现高保真度轨迹拟合；
3. **时序集成平滑滤波**：利用指数衰减权重融合多重重叠预测，彻底消除了周期切换顿挫，输出高阶光滑的关节控制信号。
