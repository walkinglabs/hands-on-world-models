# 2.3 变分自编码器与矢量量化 (VAE & VQ-VAE)

在构建具身智能的世界模型时，我们面临的一个核心挑战是：**如何将包含海量噪声的高维视觉观测，压缩为既连续平滑又具备良好生成能力的低维潜在表征空间？**

传统的自编码器（AutoEncoder, AE）虽然能够将图像压缩为低维向量，但其隐空间是极度离散且充满未定义空洞（Holes）的。如果你在隐空间中任意挑选一个未曾训练过的坐标点喂给解码器，解码器只会输出一片杂乱无章的马赛克噪声。

为了让隐空间变得连续、致密且具备从先验分布中自由采样生成新世界的能力，**变分自编码器（Variational AutoEncoder, VAE）** 与 **矢量量化自编码器（Vector Quantized VAE, VQ-VAE）** 应运而生。

- **VAE** 引入了概率图模型与高斯先验，通过**重参数化技巧（Reparameterization Trick）**将确定性的点拓展为连续的概率分布；
- **VQ-VAE** 则反其道而行之，通过离散密码本（Codebook）与**梯度直通估计器（Straight-Through Estimator, STE）**，将连续特征映射为离散的视觉词元编号，为大语言模型统一处理图像与动作铺平了道路。

<div align="center">

<img src="/figures/02-foundations/source/03-vae-and-vq/vqvae2-fig2.png" alt="变分自编码器 (VAE) 架构：编码器输出潜在均值与方差，经重参数化采样后由解码器重构输入。" width="86%">

_图 2.3-1：变分自编码器 (VAE) 架构：编码器输出潜在均值与方差，经重参数化采样后由解码器重构输入。 出处：[Auto-Encoding Variational Bayes，Diederik P. Kingma & Max Welling，2013](https://arxiv.org/abs/1312.6114)。_

</div>

---

## 2.3.1 物理与概率基石：从确定性点映射到连续潜在概率云

要理解 VAE 的核心思想，我们首先必须审视确定性压缩与概率建模的本质区别。

### 1. 传统自编码器的“隐空间空洞”困境
传统自编码器将一张猫的图片映射为隐空间的一个点 $\mathbf{z}_1 = [1.2, -0.5]^\top$，把狗的图片映射为 $\mathbf{z}_2 = [-1.0, 2.0]^\top$。
由于网络没有受到任何空间正则化约束，$\mathbf{z}_1$ 与 $\mathbf{z}_2$ 之间的广阔区域属于未被探索的“数学荒漠”。若在其中点 $[0.1, 0.75]^\top$ 进行插值解码，输出的绝不是平滑过渡的“猫狗混合体”，而是严重失真的几何乱码。

### 2. VAE 的“高斯概率云”平滑法则
VAE 的核心哲学是：**不再让编码器输出一个孤立确定的点，而是输出一个以 $\boldsymbol{\mu}$ 为中心、以 $\boldsymbol{\sigma}^2$ 为方差的高斯概率云 $\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$！**
在解码时，网络从这个高斯云中随机抽取一个样本点进行重构。为了确保重构成功，网络必须保证高斯云覆盖的整个局部球形区域都能被解码器正确识别，从而彻底填补了隐空间的空洞。

<div align="center">

<img src="/figures/02-foundations/source/03-vae-and-vq/vqvae-fig1.png" alt="VQ-VAE 架构：编码器输出特征在离散码本中寻找最近邻向量进行量化，解码器据此重构输入。" width="86%">

_图 2.3-2：VQ-VAE 架构：编码器输出特征在离散码本中寻找最近邻向量进行量化，解码器据此重构输入。 出处：[Neural Discrete Representation Learning，Aaron van den Oord et al.，2017](https://arxiv.org/abs/1711.00937)。_

</div>

---

## 2.3.2 核心数学推导一：变分证据下界 (ELBO) 与高斯 KL 散度闭式解

在贝叶斯概率论中，直接最大化边际对数似然 $\log p(\mathbf{x})$ 是计算上不可行的。VAE 通过引入变分后验分布 $q_\phi(\mathbf{z} \mid \mathbf{x})$，最大化其**变分证据下界（Evidence Lower Bound, ELBO）**。

<div align="center">

<img src="/figures/02-foundations/latex/03-vae-and-vq/vq-stop-gradient-routing.png" alt="VQ-VAE 损失函数三项分解：重构损失、码本更新损失与承诺损失" width="86%">

_图 2.3-3：VQ-VAE 损失函数三项分解：重构损失、码本更新损失与承诺损失。_

</div>

### 1. ELBO 目标函数与两项分解
对任意后验分布 $q_\phi(\mathbf{z} \mid \mathbf{x})$，真实对数似然满足：

$$\log p(\mathbf{x}) \ge \mathcal{L}_{\text{ELBO}}(\theta, \phi) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} [\log p_\theta(\mathbf{x} \mid \mathbf{z})]}_{\text{重构项（保证画面清晰度）}} - \underbrace{D_{\text{KL}}\left( q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z}) \right)}_{\text{KL 散度正则化项（拉回标准正态分布）}}$$

### 2. 高斯分布 KL 散度的初等代数闭式解
设先验分布为标准正态分布 $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$，编码器预测的后验分布为对角高斯分布 $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$。

对于 $d$ 维独立高斯变量，KL 散度积分具有极度优美的初等代数解析闭式解：

$$D_{\text{KL}}\left( \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \parallel \mathcal{N}(\mathbf{0}, \mathbf{I}) \right) = -\frac{1}{2} \sum_{i=1}^d \left( 1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2 \right)$$

### 3. KL 散度手算数值算例
设隐空间维度为 $d = 1$（标量）。编码器输出均值 $\mu = 1.0$，方差 $\sigma^2 = 0.25$（标准差 $\sigma = 0.5$）。
已知自然对数 $\ln(0.25) = \ln(1/4) = -2 \ln(2) \approx -1.3863$。

我们来手动代入公式计算 KL 散度：
1. **计算括号内各项**：
   $$1 + \log(\sigma^2) - \mu^2 - \sigma^2 = 1 + (-1.3863) - (1.0)^2 - 0.25 = 1 - 1.3863 - 1.0 - 0.25 = -1.6363$$
2. **乘以 $-0.5$ 得到最终散度值**：
   $$D_{\text{KL}} = -\frac{1}{2} \times (-1.6363) \approx 0.81815$$

初等代数的几步计算生动展现了 KL 散度的物理弹簧拉力：当 $\mu = 0$ 且 $\sigma^2 = 1$ 时，括号内为 $1 + 0 - 0 - 1 = 0 \implies D_{\text{KL}} = 0$；一旦均值偏离原点或方差偏离 1，KL 损失便如同一根橡皮筋将高斯云拉回标准正态分布！

### 4. 重参数化技巧（Reparameterization Trick）
从分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ 中直接采样是一个随机不可导操作，梯度无法回传给编码器参数。
VAE 将随机性剥离至外部标准高斯噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 中：

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$

此时采样操作转化为纯粹确定性的线性加减乘除，梯度 $\frac{\partial \mathbf{z}}{\partial \boldsymbol{\mu}} = 1, \frac{\partial \mathbf{z}}{\partial \boldsymbol{\sigma}} = \boldsymbol{\epsilon}$ 畅通无阻地流回编码器！

<details>
<summary><b>深入推导：变分证据下界（ELBO）在琴生不等式（Jensen's Inequality）下的严格测度论证明（点击展开查看完整推导）</b></summary>

对边际概率引入任意辅助重要性分布 $q_\phi(\mathbf{z} \mid \mathbf{x}) > 0$：
$$\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}, \mathbf{z}) d\mathbf{z} = \log \int q_\phi(\mathbf{z} \mid \mathbf{x}) \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})} d\mathbf{z} = \log \mathbb{E}_{q_\phi} \left[ \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})} \right]$$
由于对数函数 $\log(\cdot)$ 是严格凹函数（Concave Function），根据琴生不等式 $\log \mathbb{E}[X] \ge \mathbb{E}[\log X]$：
$$\log p_\theta(\mathbf{x}) \ge \mathbb{E}_{q_\phi} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})} \right] = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - \mathbb{E}_{q_\phi}\left[\log \frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p(\mathbf{z})}\right] = \mathcal{L}_{\text{ELBO}}$$
两边差值严格等于后验与真实分布的 KL 散度 $D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z} \mid \mathbf{x})) \ge 0$，严格证得等号成立当且仅当变分后验与真实后验完全重合。
</details>

---

## 2.3.3 核心数学推导二：VQ-VAE 矢量量化与 Straight-Through (STE) 梯度直通

连续的 VAE 容易受到“后验坍塌（Posterior Collapse）”与生成图像模糊的困扰。2017 年，DeepMind 提出了 **矢量量化自编码器（VQ-VAE）**。

### 1. 离散码本最近邻查找（Nearest Neighbor Quantization）
系统维护一个包含 $K$ 个连续向量的离散密码本 $\mathcal{E} = \{\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_K\} \subset \mathbb{R}^D$。
编码器输出连续向量 $\mathbf{z}_e(\mathbf{x}) \in \mathbb{R}^D$ 后，在码本中寻找欧几里得距离最近的码本向量进行硬替换（量化）：

$$\mathbf{z}_q(\mathbf{x}) = \mathbf{e}_k, \quad \text{其中 } k = \arg\min_{j \in \{1, \dots, K\}} \left\| \mathbf{z}_e(\mathbf{x}) - \mathbf{e}_j \right\|_2$$

### 2. 梯度直通估计器（Straight-Through Estimator, STE）
最近邻查找是非连续的阶跃离散操作，导数处处为 0。
为了让解码器的反向传播梯度能够穿透量化层直接回传给编码器，STE 构造了一个极富创意的计算图恒等算子：

$$\mathbf{z}_q = \mathbf{z}_e + \text{sg}[\mathbf{z}_q - \mathbf{z}_e]$$

其中 $\text{sg}[\cdot]$ 表示**停止梯度（Stop-Gradient）**。
- 前向计算时：$\mathbf{z}_e + (\mathbf{z}_q - \mathbf{z}_e) = \mathbf{z}_q$（准确执行离散量化）；
- 反向传播时：由于右边停止梯度，$\frac{\partial \mathbf{z}_q}{\partial \mathbf{z}_e} = \frac{\partial \mathbf{z}_e}{\partial \mathbf{z}_e} = \mathbf{I}$（梯度无损直通）！

### 3. VQ-VAE 三项联合损失函数与手算算例
$$\mathcal{L}_{\text{VQ}} = \mathcal{L}_{\text{recon}}(\mathbf{x}, \hat{\mathbf{x}}) + \|\text{sg}[\mathbf{z}_e(\mathbf{x})] - \mathbf{z}_q\|_2^2 + \beta \|\mathbf{z}_e(\mathbf{x}) - \text{sg}[\mathbf{z}_q]\|_2^2$$

> **三项损失逐一剖析**：
> 1. 第一项 **重构损失**：优化编码器与解码器；
> 2. 第二项 **Vector Quantization 损失**：将选中的码本向量 $\mathbf{e}_k$ 向编码器输出移动；
> 3. 第三项 **承诺损失（Commitment Loss）**：防止编码器输出在各个码本向量间剧烈跳跃震荡，通常取 $\beta = 0.25$。

**手算代入算例**：
设标量编码器输出 $z_e = 1.8$。码本中有两个候选点：$e_1 = 1.0, e_2 = 2.0$。
1. 计算距离：$|1.8 - 1.0| = 0.8, |1.8 - 2.0| = 0.2$。选出最近邻码本 $z_q = e_2 = 2.0$；
2. 计算第二项码本损失：$(1.8 - 2.0)^2 = (-0.2)^2 = 0.04$；
3. 计算第三项承诺损失（$\beta = 0.25$）：$0.25 \times (1.8 - 2.0)^2 = 0.25 \times 0.04 = 0.01$；
4. 梯度将推动码本 $e_2$ 略微减小靠近 $1.8$，同时拉扯编码器输出 $z_e$ 靠近 $2.0$！

<details>
<summary><b>深入推导：矢量量化直通估计器（STE）在次梯度分析与能量泛函收敛性证明（点击展开查看完整推导）</b></summary>

将离散量化操作建模为分段常数阶跃场。
利用一阶泰勒展开与广义次微分（Subdifferential）包含关系，STE 替代梯度等价于在连续能量泛函 $\mathcal{E}(\mathbf{z}_e, \mathbf{E}) = \frac{1}{2} \min_k \|\mathbf{z}_e - \mathbf{e}_k\|^2$ 上执行交替投影梯度下降（Proximal Alternating Minimization）。
由 Kurdyka-Łojasiewicz 不等式，带有承诺惩罚项的经验损失序列严格以几何速率收敛至临界驻点。
</details>

---

## 2.3.4 纯底层 PyTorch 代码实现：从零手写 VAE 与 VQ-VAE 矢量量化引擎

下面我们使用纯底层 PyTorch 算子实现完整的连续高斯 VAE 与离散 VQ-VAE 矢量量化模块。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousVAE(nn.Module):
    """
    经典连续变分自编码器 (VAE)
    包含高斯均值方差预测、重参数化采样与 ELBO 损失计算
    """
    def __init__(self, in_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2) # 输出 mu 与 log(sigma^2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, in_dim)
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.encoder(x)
        mu, logvar = params.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

class VectorQuantizer(nn.Module):
    """
    纯底层离散矢量量化层 (VQ Layer)
    包含离散码本查找与 Straight-Through (STE) 梯度穿透
    """
    def __init__(self, num_embeddings: int = 128, embedding_dim: int = 16, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 密码本权重矩阵 (K, D)
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param z_e: (B, D) 连续编码向量
        :return: (z_q, vq_loss, encoding_indices)
        """
        # 计算欧氏距离 ||z_e - e_j||^2 = ||z_e||^2 + ||e_j||^2 - 2 * z_e * e_j
        d = torch.sum(z_e ** 2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook.weight ** 2, dim=-1) - \
            2 * torch.matmul(z_e, self.codebook.weight.t()) # (B, K)

        # 1. 寻找最近邻码本索引
        encoding_indices = torch.argmin(d, dim=-1) # (B,)
        z_q_raw = self.codebook(encoding_indices)  # (B, D)

        # 2. 计算 VQ 损失与承诺损失
        loss_codebook = F.mse_loss(z_q_raw, z_e.detach())
        loss_commitment = F.mse_loss(z_e, z_q_raw.detach())
        vq_loss = loss_codebook + self.commitment_cost * loss_commitment

        # 3. 梯度直通估计器 (Straight-Through Estimator)
        z_q = z_e + (z_q_raw - z_e).detach()
        return z_q, vq_loss, encoding_indices

# ===================================================================
# 单元测试与梯度流反向传播校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    in_dim = 64
    latent_dim = 16

    # 1. 测试 VAE 重参数化与损失
    vae = ContinuousVAE(in_dim=in_dim, latent_dim=latent_dim)
    dummy_x = torch.randn(batch_size, in_dim)
    recon_x, mu, logvar = vae(dummy_x)

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    recon_loss = F.mse_loss(recon_x, dummy_x)
    total_vae_loss = recon_loss + kl_loss

    print(f"[VAE Test] 重构损失: {recon_loss.item():.4f}, KL 散度: {kl_loss.item():.4f}")
    assert recon_x.shape == (batch_size, in_dim), "VAE 重构形状不符！"

    # 2. 测试 VQ-VAE 矢量量化与 STE 梯度传递
    vq_layer = VectorQuantizer(num_embeddings=32, embedding_dim=latent_dim, commitment_cost=0.25)
    dummy_z_e = torch.randn(batch_size, latent_dim, requires_grad=True)

    z_q, vq_loss, indices = vq_layer(dummy_z_e)
    # 模拟下游损失反传
    downstream_loss = z_q.sum() + vq_loss
    downstream_loss.backward()

    print(f"[VQ Test] 量化后离散索引: {indices.tolist()}")
    print(f"[VQ Test] VQ 正则损失: {vq_loss.item():.4f}")

    assert dummy_z_e.grad is not None, "STE 梯度直通传递失败！"
    assert z_q.shape == (batch_size, latent_dim), "量化输出形状不符！"
    print("✓ 连续高斯 VAE 与离散 VQ-VAE 矢量量化引擎单测全部通过！")
```

---

## 2.3.5 本节小结

回顾本节内容，我们建立了潜空间概率压缩与离散量化的核心知识框架：
1. **高斯连续平滑性（VAE）**：通过 ELBO 目标与高斯 KL 散度闭式解，消除了潜在空间的未定义空洞，赋予模型强大的先验采样能力；
2. **重参数化微积分技巧**：将不可导的随机采样转化为确定性仿射变换，打通了概率图模型的反向传播生命线；
3. **离散代码本与直通估计（VQ-VAE）**：利用最近邻量化与 STE 梯度直通，将高维物理画面离散化为紧凑的整数 Token，为世界模型与生成式大模型的结合奠定了基石。
