# 空间离散化：VAE与VQ

在深度学习的发展历程中，如何有效地学习数据的高维连续表征一直是一个核心问题。从早期的主成分分析（PCA）到深层自编码器（Autoencoder），研究者们试图将复杂的现实世界数据（如图像、音频）压缩到一个低维的潜在空间（Latent Space）中。然而，传统的自编码器往往缺乏对潜在空间的概率约束，导致生成的特征空间存在大量“空洞”，无法用于生成新样本。

2013 年，Kingma 和 Welling 提出了变分自编码器（Variational Autoencoder, VAE）[[Kingma & Welling, 2013]](https://arxiv.org/abs/1312.6114)，把神经网络、潜变量模型与可微的变分推断结合起来。VAE 使用连续随机潜变量；van den Oord 等人随后提出向量量化变分自编码器（Vector Quantised-Variational AutoEncoder, VQ-VAE），用可学习码本把编码结果离散化 [[van den Oord et al., 2017]](https://arxiv.org/abs/1711.00937)。两者的差异是潜变量参数化方式，不意味着现实数据本身都能简单分为“连续”或“离散”。

在本节中，我们将从最基础的概率法则出发，逐步推导 VAE 的数学基础，并在此基础上深入探讨 VQ-VAE 是如何实现空间离散化的。我们将剥开所有复杂的数学外衣，用高中数学的直觉来理解这些强大的生成模型。

## 隐变量模型与极大似然的困境

在高中概率论中，我们学习过条件概率和全概率公式。假设我们观察到了一系列数据 $x$（例如，一堆人脸图片）。我们假设这些数据的生成是由一些隐含的、我们无法直接观测到的因素 $z$（例如，性别、表情、光照）决定的。这种变量 $z$ 被称为**隐变量**（Latent Variable）。

我们希望找到一个模型参数 $\theta$，使得该模型生成观测数据 $x$ 的概率最大化。这在统计学中被称为极大似然估计（Maximum Likelihood Estimation）。数据 $x$ 的边际概率 $P_\theta(x)$ 可以通过对所有可能的隐变量 $z$ 积分求得：

$$ P_\theta(x) = \int P_\theta(x|z) P(z) dz $$

这里，$P(z)$ 是隐变量的先验分布（通常我们假设它是一个标准正态分布），而 $P_\theta(x|z)$ 是给定隐变量 $z$ 时生成数据 $x$ 的条件概率，在深度学习中，这通常由一个神经网络（即解码器）来建模。

**困境在于积分的计算**：在高维空间中，隐变量 $z$ 可能包含数百个维度。要在整个高维空间上对上述积分进行精确计算，其计算量是指数级增长的，这在现实中是完全不可解（Intractable）的。既然无法直接最大化 $P_\theta(x)$，我们需要寻找一种替代方案。

## 变分下界（ELBO）的严格推导

既然直接计算含有积分的似然函数行不通，数学家们采取了一种极其巧妙的迂回策略：我们不去直接最大化对数似然 $\log P_\theta(x)$，而是去最大化它的一个**下界**（Lower Bound）。只要这个下界被不断抬高，真实的对数似然也会随之增大。

为了引入这个下界，我们需要引入一个新的分布 $Q_\phi(z|x)$。在物理意义上，既然 $P_\theta(x|z)$ 是从隐变量生成数据的**解码过程**，那么 $Q_\phi(z|x)$ 就是给定数据推断隐变量的**编码过程**。

现在，让我们仅使用高中阶段熟悉的对数运算规则和期望的定义，一步步推导这个核心公式。

首先，根据条件概率的定义 $P(x, z) = P(x|z)P(z) = P(z|x)P(x)$，我们可以写出：

$$ \log P_\theta(x) = \log P_\theta(x, z) - \log P_\theta(z|x) $$

这个等式对于任意的 $z$ 都成立。接下来，我们在等式两边同时对分布 $Q_\phi(z|x)$ 求期望（即乘以 $Q_\phi(z|x)$ 并对 $z$ 积分）。因为等式左边的 $\log P_\theta(x)$ 并不包含变量 $z$，所以它对任何关于 $z$ 的分布求期望都等于它本身：

$$ \log P_\theta(x) = \mathbb{E}_{z \sim Q_\phi(z|x)} [ \log P_\theta(x, z) - \log P_\theta(z|x) ] $$

接下来，我们在期望的方括号内部，巧妙地加上并减去同一项 $\log Q_\phi(z|x)$：

$$ \log P_\theta(x) = \mathbb{E}_{z \sim Q_\phi(z|x)} [ \log P_\theta(x, z) - \log Q_\phi(z|x) + \log Q_\phi(z|x) - \log P_\theta(z|x) ] $$

然后，我们将期望拆分成两部分：

$$ \log P_\theta(x) = \underbrace{\mathbb{E}_{z \sim Q_\phi(z|x)} \left[ \log \frac{P_\theta(x, z)}{Q_\phi(z|x)} \right]}_{\text{ELBO}} + \underbrace{\mathbb{E}_{z \sim Q_\phi(z|x)} \left[ \log \frac{Q_\phi(z|x)}{P_\theta(z|x)} \right]}_{\text{KL Divergence}} $$

在这个等式中，右侧的第二项正是分布 $Q_\phi(z|x)$ 和真实后验分布 $P_\theta(z|x)$ 之间的 KL 散度（Kullback-Leibler Divergence），记为 $D_{\text{KL}}(Q_\phi(z|x) \| P_\theta(z|x))$。根据信息论的基本定理，KL 散度衡量了两个概率分布之间的差异，它永远是非负的（$\ge 0$）。

既然 KL 散度非负，那么等式右侧的第一项必然小于或等于左侧的对数似然 $\log P_\theta(x)$。因此，这第一项被称为**证据下界**（Evidence Lower BOund, ELBO）：

$$ \log P_\theta(x) \ge \text{ELBO} = \mathbb{E}_{z \sim Q_\phi(z|x)} [ \log P_\theta(x, z) - \log Q_\phi(z|x) ] $$

至此，我们将一个不可解的积分问题，转化为了一个优化问题：通过调整神经网络的参数 $\phi$（编码器）和 $\theta$（解码器），最大化 ELBO。当 ELBO 最大化时，我们既抬高了数据似然 $\log P_\theta(x)$，又迫使近似后验 $Q_\phi(z|x)$ 逼近真实的后验 $P_\theta(z|x)$。

我们可以将 ELBO 进一步展开，使其具有更明确的物理意义。利用 $P_\theta(x, z) = P_\theta(x|z)P(z)$，我们可以写出：

$$ \text{ELBO} = \mathbb{E}_{z \sim Q_\phi(z|x)} [ \log P_\theta(x|z) ] - \mathbb{E}_{z \sim Q_\phi(z|x)} \left[ \log \frac{Q_\phi(z|x)}{P(z)} \right] $$

$$ \text{ELBO} = \mathbb{E}_{z \sim Q_\phi(z|x)} [ \log P_\theta(x|z) ] - D_{\text{KL}}(Q_\phi(z|x) \| P(z)) $$

这个最终形态极其优雅。它表明，要最大化下界，我们的模型需要做到两点：

1. **最大化重构项** $\mathbb{E}_{z \sim Q_\phi(z|x)} [ \log P_\theta(x|z) ]$：即给定编码器提取的特征 $z$，解码器能够极高概率地还原出原始数据 $x$。
2. **最小化正则项** $D_{\text{KL}}(Q_\phi(z|x) \| P(z))$：即编码器输出的分布 $Q_\phi(z|x)$ 应当尽可能接近我们事先设定的先验分布 $P(z)$（通常是标准正态分布 $\mathcal{N}(0, I)$）。

## 重参数化技巧与 VAE 实现

在具体的实现中，编码器神经网络 $Q_\phi(z|x)$ 会输出一个高斯分布的均值 $\mu$ 和方差 $\sigma^2$。然后，我们需要从这个高斯分布 $\mathcal{N}(\mu, \sigma^2)$ 中采样出一个向量 $z$，再将其输入给解码器。

这里遇到了深度学习中著名的**不可导问题**。采样操作本质上是一个随机过程，在反向传播计算梯度时，我们无法对一个随机过程求导。如果梯度在这里断裂，编码器的参数 $\phi$ 就无法得到更新。

Kingma 和 Welling 提出的**重参数化技巧**（Reparameterization Trick）极其简单却极具革命性。让我们回到高中统计学，考虑最简单的一维情况：如何从一个均值为 $\mu$，标准差为 $\sigma$ 的正态分布 $z \sim \mathcal{N}(\mu, \sigma^2)$ 中采样？

我们可以先从一个固定的**标准正态分布** $\epsilon \sim \mathcal{N}(0, 1)$ 中采样出一个噪声数值，然后对其进行简单的线性平移和缩放：

$$ z = \mu + \sigma \cdot \epsilon $$

在这个公式中，随机性全部被隔离在了 $\epsilon$ 中。而 $\mu$ 和 $\sigma$ 是由神经网络确定性地计算出来的，它们参与了简单的加法和乘法运算。因此，关于 $\mu$ 和 $\sigma$ 的导数可以畅通无阻地向后传播。对于多维向量，这个操作只是对每个维度独立进行。

> 💡 **变分推断的物理类比**
> 如果我们把数据比作散落满地的图纸，$z$ 是图纸在三维空间中的压缩坐标。变分自编码器并不强求每个坐标是一个绝对精准的点，而是认为每个坐标是一个有一定“晃动范围”的弹性小球（均值和方差）。重构项试图让小球准确落在图纸对应的位置上；而 KL 散度正则项则像一根引力弹簧，试图把所有的小球都拉向空间的原点，且要求小球的“晃动范围”不能太大也不能太小，维持在标准大小（标准正态分布）。这种弹性和引力的博弈，使得整个三维空间变得平滑且没有缝隙，从而赋予了模型生成全新图纸的能力。

现在，我们可以用代码来实现一个标准的变分自编码器。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # 编码器部分：将输入映射到隐空间的均值和对数方差
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim) # 预测均值 mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim) # 预测对数方差 log(sigma^2)

        # 解码器部分：将隐空间向量重构回输入空间
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        # (返回均值和对数方差)
        # 为了数值稳定性，神经网络通常预测对数方差而不是直接预测方差
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # 计算标准差 sigma = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        # 从标准正态分布 N(0, 1) 中采样噪声 epsilon
        eps = torch.randn_like(std)
        # (执行重参数化技巧) z = mu + sigma * epsilon
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # 通常在最终输出前使用 Sigmoid 映射到 (0, 1) 区间（假设输入图像已归一化）
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

对应的损失函数，即负的 ELBO（因为我们通常是最小化损失）：

```python
def vae_loss_function(recon_x, x, mu, logvar):
    # 重构损失：通常使用二元交叉熵或均方误差
    # 这里我们假设输入是 (0, 1) 之间的像素值，使用 BCE
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KL 散度损失：推导表明对于两个高斯分布，其解析解为：
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
```

## 向量量化（VQ）：走向离散表征

VAE 虽然在数学上极其优美，但它强迫隐空间服从连续的正态分布。然而，现实世界中的许多概念是天然离散的。例如，在语言中，句子是由离散的单词（词表中的条目）组成的，不存在处于“猫”和“狗”之间某个连续坐标的“半猫半狗”的中间词。

为了在深度网络中学习这种离散结构，van den Oord 等人提出了 VQ-VAE。其核心思想是引入一个可学习的**码本**（Codebook），或者叫字典。隐变量不再是任意连续的值，而必须是码本中某一个离散向量的索引。

假设我们的码本 $E = \{e_1, e_2, \dots, e_K\}$，包含 $K$ 个潜在向量，每个向量的维度为 $D$。

编码器输出一个连续的特征向量 $z_e(x)$。接下来，我们不使用重参数化技巧来采样，而是直接在码本 $E$ 中寻找距离 $z_e(x)$ 最近的那一个向量 $e_k$。这个寻找最近邻的过程被称为**向量量化**（Vector Quantization）：

$$ z_q(x) = e_k, \quad \text{where} \quad k = \arg\min_j \| z_e(x) - e_j \|_2 $$

这一步将连续的 $z_e$ 强行对齐到了离散的码本空间，输出量化后的向量 $z_q(x)$，随后 $z_q(x)$ 被送入解码器以重构图像。

**梯度的直通估计器（Straight-Through Estimator, STE）**

该公式中的 $\arg\min$ 操作是完全不可导的。为了让网络能够端到端训练，VQ-VAE 使用了一个极其简单粗暴但在工程上非常有效的技巧：**直通估计器**（STE）。

在反向传播时，我们直接将解码器关于量化后向量 $z_q(x)$ 的梯度，原封不动地复制给编码器的输出 $z_e(x)$。也就是我们欺骗网络，假装前向传播时 $z_q(x)$ 就是 $z_e(x)$ 本身。

$$ \nabla_{z_e(x)} L \approx \nabla_{z_q(x)} L $$

**VQ-VAE 的损失函数**

既然反向传播绕过了量化步骤，那么码本中的向量 $e_k$ 应该如何更新呢？为了让码本向量向编码器的输出靠近，并且让编码器不要输出距离码本太远的离谱向量，VQ-VAE 定义了以下三个部分的损失函数：

1. **重构损失**：与传统 VAE 相同，确保解码器能从离散向量 $z_q$ 还原数据。
2. **码本损失（Codebook Loss）**：推动码本向量 $e_i$ 靠近编码器的输出 $z_e(x)$。使用停止梯度（Stop-Gradient）操作 `sg()` 确保梯度只流向码本。
3. **承诺损失（Commitment Loss）**：推动编码器的输出 $z_e(x)$ 不要偏离所选的码本向量太远，使得编码器“承诺”使用这个码本。同样使用停止梯度确保梯度只流向编码器。

完整的损失函数如下：

$$ L = \underbrace{\| x - D(z_q) \|_2^2}_{\text{Reconstruction}} + \underbrace{\| \text{sg}[z_e(x)] - e \|_2^2}_{\text{Codebook Loss}} + \beta \underbrace{\| z_e(x) - \text{sg}[e] \|_2^2}_{\text{Commitment Loss}} $$

其中 $\beta$ 是承诺损失的超参数，通常设为一个较小的值，如 $0.25$。

接下来，我们将严谨地实现 VQ-VAE 中的向量量化层。

```python
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # (定义码本 Embedding)
        # 码本包含 num_embeddings 个条目，每个条目维度为 embedding_dim
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # 初始化码本权重，使用均匀分布
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # 假设输入形状为 [batch_size, channels, height, width]
        # 首先将其转换为 [batch_size, height, width, channels] 并展平
        # 使得后续计算距离更方便
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # 展平为二维矩阵 [batch_size * height * width, channels]
        flat_input = inputs.view(-1, self._embedding_dim)

        # (计算输入向量与码本中所有向量的欧氏距离的平方)
        # (a-b)^2 = a^2 + b^2 - 2ab
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # (寻找最近邻)
        # 在第 1 维度（即码本维度）寻找距离最小的索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # 创建一个独热编码（One-hot）矩阵，用于后续计算
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # (对输入进行量化)
        # 根据索引从码本中提取对应的离散向量
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # (计算损失)
        # 使用 detach() 实现停止梯度 (sg) 操作
        e_latent_loss = F.mse_loss(quantized.detach(), inputs) # 承诺损失
        q_latent_loss = F.mse_loss(quantized, inputs.detach()) # 码本损失
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # (直通估计器 (Straight-Through Estimator, STE))
        # 这是 VQ 最核心的一步技巧：
        # 前向传播时使用 quantized
        # 反向传播时，由于 inputs.detach()，梯度将直接穿过 quantized 流向 inputs
        quantized = inputs + (quantized - inputs).detach()

        # 将量化后的特征转回 [batch_size, channels, height, width] 形状
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, encodings
```

这段代码精准地展示了 VQ 机制如何在保持前向离散化查找的同时，利用 `.detach()` 巧妙地解决了梯度断裂问题。通过离散化，模型不仅能够压缩信息，更能够学到诸如物理实体的类别、文本的单词等天然的离散语义单元。

## 小结

在本节中，我们从最基本的概率论出发，详尽地推导了**变分自编码器（VAE）**如何利用变分下界（ELBO）和重参数化技巧，在连续高维空间中寻找概率意义上的最优解。随后，我们探讨了为何离散表征在特定场景下更为重要，并引出了**向量量化变分自编码器（VQ-VAE）**，详细解析了最近邻查找与**直通估计器（STE）**的具体数学和代码实现。从连续到离散，这不仅是数学形式的转换，更是深层表征学习对不同世界运作模式的逼近尝试。
