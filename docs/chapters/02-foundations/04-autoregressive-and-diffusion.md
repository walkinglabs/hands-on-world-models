# 2.4 自回归与扩散生成模型 (Autoregressive & Diffusion)

在世界模型推演未来物理现实的核心引擎中，生成式模型（Generative Models）构成了智能体“在脑海中做梦”的数学画笔。

在生成模型的演进长河中，自回归模型（Autoregressive Models）与扩散模型（Diffusion Models）代表了两种截然不同但各具独特魅力的物理哲学：
- **自回归生成**：如同人类说话或书写文字，沿着单向时间轴一个词元接一个词元（Token by Token）地因果推演；
- **扩散生成**：汲取非平衡态热力学扩散理论，将物理画面的生成建模为从完全无序的高熵纯噪声中逐步反向去噪、凝聚出低熵清晰物理结构的过程。

本节我们将从全概率链式法则与高斯热力学扩散出发，严密推导自回归似然损失与 DDPM 任意步加噪闭式解，并使用纯底层 PyTorch 从零手写自回归生成器与 DDPM 扩散去噪采样器。

<div align="center">

<img src="/figures/02-foundations/source/04-autoregressive-and-diffusion/ddpm-fig1.png" alt="去噪扩散概率模型 (DDPM) 的马尔可夫链前向加噪与参数化反向去噪生成过程。" width="86%">

_图 2.4-1：去噪扩散概率模型 (DDPM) 的马尔可夫链前向加噪与参数化反向去噪生成过程。 出处：[Denoising Diffusion Probabilistic Models，Jonathan Ho et al.，2020](https://arxiv.org/abs/2006.11239)。_

</div>

---

## 2.4.1 物理与生成基石：因果逐步演化与热力学逆熵雕刻

要理解生成式世界模型的底层力量，我们首先需要从初等物理与信息论的角度审视两大生成范式。

### 1. 自回归范式的“时序因果链”
在语言模型（如 GPT）或视频预测中，复合事件的发生存在先来后到的因果依赖。自回归模型把联合概率分布严格拆解为单步条件转移的连乘积。它的优点是能够精准捕捉长程逻辑连贯性，缺点是生成高分辨率像素时推理速度较慢。

### 2. 扩散模型的“热力学逆过程”
在自然物理界中，将一滴墨水滴入清水中，墨水分子会自发地做布朗运动扩散至全杯水（熵增，不可逆）。
DDPM 模拟了这一过程：人为地向清晰物理状态中逐步注入微小的高斯扰动，直至其彻底退化为纯高斯白噪声；随后训练神经网络学习局部的逆向速度场（得分函数），在生成时从纯噪声中一步步剔除扰动，最终“逆流而上”雕刻出清晰的世界！

<div align="center">

<img src="/figures/02-foundations/source/04-autoregressive-and-diffusion/wavenet-fig2.png" alt="VideoPoet 将视频与动作统一离散化为自回归词元序列，展示大一统多模态自回归生成能力。" width="86%">

_图 2.4-2：VideoPoet 将视频与动作统一离散化为自回归词元序列，展示大一统多模态自回归生成能力。 出处：[VideoPoet: A Large Language Model for Zero-Shot Video Generation，Dan Kondratyuk et al.，2023](https://arxiv.org/abs/2312.14125)。_

</div>

---

## 2.4.2 核心数学推导一：自回归概率链式分解与负对数似然

设一段由 $T$ 个离散词元组成的序列为 $\mathbf{x}_{1:T} = (x_1, x_2, \dots, x_T)$。

<div align="center">

<img src="/figures/02-foundations/latex/04-autoregressive-and-diffusion/diffusion-direct-sampling.png" alt="DDPM 前向扩散任意步跳跃公式：原始信号保留系数与高斯噪声注入权重的正交几何分解" width="86%">

_图 2.4-3：DDPM 前向扩散任意步跳跃公式：原始信号保留系数与高斯噪声注入权重的正交几何分解。_

</div>

### 1. 全概率公式因果因式分解
根据概率论乘法定理，序列的联合概率分布可完全无损地展开为：

$$p(\mathbf{x}_{1:T}) = p(x_1) \cdot p(x_2 \mid x_1) \cdot p(x_3 \mid x_1, x_2) \dots = \prod_{t=1}^T p(x_t \mid \mathbf{x}_{<t})$$

### 2. 交叉熵训练目标（Negative Log-Likelihood, NLL）
神经网络参数化条件分布 $p_\theta(x_t \mid \mathbf{x}_{<t}) = \text{Softmax}(\mathbf{z}_t)$。训练目标为最小化负对数似然：

$$\mathcal{L}_{\text{AR}}(\theta) = -\sum_{t=1}^T \log p_\theta(x_t \mid \mathbf{x}_{<t})$$

### 3. 自回归似然手算数值算例
设一个由 2 个词元组成的微型动作序列，真实标签为 $\mathbf{x} = (A, B)$：
- 第 1 步预测：网络输出词表概率 $p(A) = 0.80, p(B) = 0.20$；
- 第 2 步预测：在给定历史 $A$ 的条件下，网络输出 $p(B \mid A) = 0.90, p(A \mid A) = 0.10$。

我们来手动计算联合似然与总损失：
1. **计算联合生成概率**：
   $$p(A, B) = p(A) \times p(B \mid A) = 0.80 \times 0.90 = 0.72$$
2. **计算负对数似然损失**（已知 $\ln(0.8) \approx -0.2231, \ln(0.9) \approx -0.1054$）：
   $$\mathcal{L}_{\text{AR}} = -(\ln(0.80) + \ln(0.90)) = -(-0.2231 - 0.1054) = 0.3285$$

初等代数的几步加法清晰证实：最大化联合概率等价于逐步压低每个词元的单步分类交叉熵损失！

<details>
<summary><b>深入推导：自回归时序交叉熵在香农信息熵意义下的渐近最优性证明（点击展开查看完整推导）</b></summary>

设数据真实分布为 $P(\mathbf{X})$。经验交叉熵损失期望为：
$$\mathbb{E}_{P} [\mathcal{L}_{\text{AR}}(\theta)] = -\sum_{\mathbf{x}} P(\mathbf{x}) \log Q_\theta(\mathbf{x}) = H(P) + D_{\text{KL}}(P \parallel Q_\theta)$$
其中 $H(P)$ 为自然物理世界固有的香农信息熵常数。
最小化自回归损失严格等价于极小化真实世界分布与模型分布的 KL 散度 $D_{\text{KL}}(P \parallel Q_\theta)$。当且仅当 $Q_\theta \equiv P$ 时，模型达到了香农信息论极限的无偏表征。
</details>

---

## 2.4.3 核心数学推导二：DDPM 扩散前向跳步闭式解与去噪损失

扩散模型（DDPM）由两个对称过程构成：前向加噪马尔可夫链与逆向去噪生成链。

<div align="center">

<img src="/figures/02-foundations/source/04-autoregressive-and-diffusion/wavenet-fig2.png" alt="VideoPoet 解码结构：自回归语言模型生成长程时序特征并解码为连贯高质量动作视频。" width="86%">

_图 2.4-4：VideoPoet 解码结构：自回归语言模型生成长程时序特征并解码为连贯高质量动作视频。 出处：[VideoPoet: A Large Language Model for Zero-Shot Video Generation，Dan Kondratyuk et al.，2023](https://arxiv.org/abs/2312.14125)。_

</div>

### 1. 前向单步加噪核
设总扩散步数为 $T = 1000$。在第 $t$ 步，向样本注入方差为 $\beta_t \in (0, 1)$ 的微小高斯噪声：

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left( \mathbf{x}_t; \; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \; \beta_t \mathbf{I} \right)$$

令 $\alpha_t = 1 - \beta_t$，以及累积乘积 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

### 2. 任意步直接跳步闭式采样定理（Closed-Form Sampling）
利用独立正态变量相加的高斯性质，我们**无需逐步递推 1000 次**，可以在 $\mathcal{O}(1)$ 常数时间内一步直接从原始清晰样本 $\mathbf{x}_0$ 采样出任意第 $t$ 步的含噪状态：

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \text{其中 } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

> **初等几何分解直觉**：
> 系数平方和恒满足 $(\sqrt{\bar{\alpha}_t})^2 + (\sqrt{1 - \bar{\alpha}_t})^2 = \bar{\alpha}_t + 1 - \bar{\alpha}_t = 1.0$！
> 这意味着样本的能量模长始终守恒：随着扩散步数 $t$ 的增加，原始信号权重 $\sqrt{\bar{\alpha}_t}$ 从 $1.0$ 单调衰减到 $0.0$，而注入的标准噪声权重 $\sqrt{1 - \bar{\alpha}_t}$ 从 $0.0$ 单调爬升到 $1.0$！

### 3. 扩散加噪手算数值算例
设某标量物理状态 $x_0 = 2.0$。在噪声步 $t = 500$ 处，预设的累积系数为 $\bar{\alpha}_t = 0.64$。
计算得：$\sqrt{\bar{\alpha}_t} = \sqrt{0.64} = 0.80$，$\sqrt{1 - \bar{\alpha}_t} = \sqrt{1 - 0.64} = \sqrt{0.36} = 0.60$。
采样到一个标准高斯随机数 $\epsilon = 1.0$。

计算含噪状态 $x_t$：
$$x_t = 0.80 \times x_0 + 0.60 \times \epsilon = 0.80 \times 2.0 + 0.60 \times 1.0 = 1.60 + 0.60 = 2.20$$

### 4. 噪声预测简化训练目标
神经网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 的唯一任务是根据含噪状态 $\mathbf{x}_t$ 与时间步 $t$，精准反求出当初注入的那个随机噪声 $\boldsymbol{\epsilon}$：

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right\|_2^2 \right]$$

<details>
<summary><b>深入推导：DDPM 变分下界（VLB）到噪声预测均方误差的柯尔莫哥洛夫连续性等价证明（点击展开查看完整推导）</b></summary>

真实反向转移条件后验分布 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\tilde{\boldsymbol{\mu}}_t, \tilde{\beta}_t \mathbf{I})$ 根据贝叶斯公式展开：
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t$$
代入 $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon})$，消元化简得：
$$\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)$$
将 KL 散度极小化目标展开，参数化均值误差与噪声预测均方误差严格等价，重加权后即证得 Ho et al. 极简训练目标 $\mathcal{L}_{\text{simple}}$。
</details>

---

## 2.4.4 纯底层 PyTorch 代码实现：从零手写自回归生成器与 DDPM 扩散去噪引擎

下面我们使用纯底层 PyTorch 算子实现一个结构完整的自回归序列预测器与带闭式加噪/去噪循环的 DDPM 扩散模型。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AutoregressiveTokenPredictor(nn.Module):
    """
    自回归词元序列预测模型
    """
    def __init__(self, vocab_size: int = 50, d_model: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        :param x_seq: (B, L)
        :return: (B, L, vocab_size)
        """
        emb = self.embed(x_seq)
        out, _ = self.gru(emb)
        logits = self.lm_head(out)
        return logits

class DDPMSampler(nn.Module):
    """
    纯底层一维 DDPM 扩散生成器
    包含前向闭式加噪与逆向逐步去噪采样
    """
    def __init__(self, data_dim: int = 4, num_timesteps: int = 50):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.data_dim = data_dim

        # 构造线性调度 beta_t 与 alpha_t 累积量
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # 简单的去噪预测网络 epsilon_theta(x_t, t)
        self.denoise_net = nn.Sequential(
            nn.Linear(data_dim + 1, 64),
            nn.Mish(),
            nn.Linear(64, 64),
            nn.Mish(),
            nn.Linear(64, data_dim)
        )

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        前向闭式加噪: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar = self.alphas_cumprod[t].view(-1, 1)
        mean = torch.sqrt(alpha_bar) * x_0
        std = torch.sqrt(1.0 - alpha_bar)
        return mean + std * noise

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_norm = (t.float() / self.num_timesteps).view(-1, 1)
        inputs = torch.cat([x_t, t_norm], dim=-1)
        return self.denoise_net(inputs)

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple) -> torch.Tensor:
        """
        逆向逐步去噪生成循环: 从纯高斯噪声 x_T 迭代还原出 x_0
        """
        device = self.betas.device
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self.predict_noise(x, t_tensor)

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]

            # 逆向均值: mu = (1 / sqrt(alpha_t)) * (x - (beta_t / sqrt(1 - alpha_bar_t)) * pred_noise)
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)

            if t > 0:
                z = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * z
            else:
                x = mean

        return x

# ===================================================================
# 单元测试与生成维度校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    seq_len = 6
    data_dim = 4

    # 1. 测试自回归预测器
    ar_model = AutoregressiveTokenPredictor(vocab_size=50, d_model=32)
    dummy_tokens = torch.randint(0, 50, (batch_size, seq_len))
    ar_logits = ar_model(dummy_tokens)

    ar_loss = F.cross_entropy(ar_logits.view(-1, 50), dummy_tokens.view(-1))
    print(f"[AR Test] 自回归交叉熵损失: {ar_loss.item():.4f}")
    assert ar_logits.shape == (batch_size, seq_len, 50), "自回归输出形状不符！"

    # 2. 测试 DDPM 扩散闭式采样与去噪循环
    ddpm = DDPMSampler(data_dim=data_dim, num_timesteps=50)
    dummy_x0 = torch.randn(batch_size, data_dim)
    t_rand = torch.randint(0, 50, (batch_size,))

    # 前向加噪
    noisy_sample = ddpm.q_sample(dummy_x0, t_rand)
    # 逆向采样生成
    generated_data = ddpm.p_sample_loop((batch_size, data_dim))

    print(f"[DDPM Test] 加噪样本形状: {noisy_sample.shape}")
    print(f"[DDPM Test] 逆向生成数据形状: {generated_data.shape}")

    assert generated_data.shape == (batch_size, data_dim), "扩散采样形状不符！"
    assert not torch.isnan(generated_data).any(), "扩散生成出现 NaN 异常！"
    print("✓ 自回归预测模型与 DDPM 扩散去噪采样引擎单测全部通过！")
```

---

## 2.4.5 本节小结

回顾本节内容，我们建立了现代生成式世界模型的两大数学支柱：
1. **全概率链式分解（自回归）**：沿着时间之箭一个 Token 接一个 Token 地进行因果推演，严格维护了物理因果的确定性传递；
2. **热力学逆过程（扩散模型）**：通过前向任意步闭式跳步加噪与神经网络逆向去噪，从完全无序的高斯纯噪声中高保真地还原出物理现实；
3. **世界模型动力学整合**：自回归擅长离散因果逻辑决策，扩散模型擅长高维连续轨迹拟合，二者在具身智能的宏观策略与微观控制中实现了完美互补。
