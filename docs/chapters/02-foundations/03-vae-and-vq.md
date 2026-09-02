# 2.3 空间离散化：VAE 与 VQ

表征学习常把图像、音频等高维观测压缩到较低维的潜在空间（Latent Space）。普通自编码器可以学习重构，却没有直接规定潜变量应服从怎样的概率分布；因此，从任意潜在坐标采样时，解码结果未必合理。VAE 与 VQ-VAE 分别用连续概率分布和离散码本约束潜在表示。

<div align="center">
  <img src="/figures/02-foundations/source/03-vae-and-vq/rezende-fig4.png" alt="随机变分推断模型在 NORB、CIFAR 与 Frey Faces 上的训练样本和生成样本对照，展示连续潜变量模型的生成能力。" width="86%">

_图 2.3-1：随机变分推断模型在 NORB、CIFAR 与 Frey Faces 上的训练样本和生成样本对照，展示连续潜变量模型的生成能力。 出处：Danilo Jimenez Rezende; Shakir Mohamed; Daan Wierstra，[Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082)（2014），Figure 4。_

</div>

2013 年，Kingma 和 Welling 提出了变分自编码器（Variational Autoencoder, VAE）[[Kingma & Welling, 2013]](https://arxiv.org/abs/1312.6114)，把神经网络、潜变量模型与可微的变分推断结合起来。VAE 使用连续随机潜变量；van den Oord 等人随后提出向量量化变分自编码器（Vector Quantised-Variational AutoEncoder, VQ-VAE），用可学习码本把编码结果离散化 [[van den Oord et al., 2017]](https://arxiv.org/abs/1711.00937)。两者的差异是潜变量参数化方式，不意味着现实数据本身都能简单分为“连续”或“离散”。

本节先从边际似然与变分下界推导 VAE，再讨论 VQ-VAE 如何用最近邻码本得到离散表示，并说明重参数化、停止梯度和直通估计器分别解决什么训练问题。

## 隐变量模型与极大似然的困境

在高中概率论中，我们学习过条件概率和全概率公式。假设我们观察到了一系列数据 $x$（例如，一堆人脸图片）。我们假设这些数据的生成是由一些隐含的、我们无法直接观测到的因素 $z$（例如，性别、表情、光照）决定的。这种变量 $z$ 被称为**隐变量**（Latent Variable）。

我们希望找到一个模型参数 $\theta$，使得该模型生成观测数据 $x$ 的概率最大化。这在统计学中被称为极大似然估计（Maximum Likelihood Estimation）。数据 $x$ 的边际概率 $P_\theta(x)$ 可以通过对所有可能的隐变量 $z$ 积分求得：

$$ P_\theta(x) = \int P_\theta(x|z) P(z) dz $$

这里，$P(z)$ 是隐变量的先验分布（通常我们假设它是一个标准正态分布），而 $P_\theta(x|z)$ 是给定隐变量 $z$ 时生成数据 $x$ 的条件概率，在深度学习中，这通常由一个神经网络（即解码器）来建模。

**困难在于积分的计算**：当 $P_\theta(x\mid z)$ 由非线性神经网络给出时，这个高维积分通常没有可直接计算的闭式解。数值积分或朴素采样又可能代价很高，因此难以直接优化 $P_\theta(x)$。变分推断用一个可计算的近似目标绕开这一点。

## 变分下界（ELBO）的严格推导

既然边际似然难以直接计算，可以转而最大化它的一个**下界**（Lower Bound）。需要注意：对固定参数，下界不超过真实对数似然；训练时参数也在变化，因此“下界升高”并不意味着每一步的真实对数似然必然同步升高。

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

至此，我们得到一个可用蒙特卡洛采样估计、并能通过梯度优化的目标。最大化 ELBO 同时鼓励解码器解释数据，并通过变分间隙约束近似后验 $Q_\phi(z\mid x)$ 接近模型后验 $P_\theta(z\mid x)$。

我们可以将 ELBO 进一步展开，使其具有更明确的物理意义。利用 $P_\theta(x, z) = P_\theta(x|z)P(z)$，我们可以写出：

$$ \text{ELBO} = \mathbb{E}_{z \sim Q_\phi(z|x)} [ \log P_\theta(x|z) ] - \mathbb{E}_{z \sim Q_\phi(z|x)} \left[ \log \frac{Q_\phi(z|x)}{P(z)} \right] $$

$$ \text{ELBO} = \mathbb{E}_{z \sim Q_\phi(z|x)} [ \log P_\theta(x|z) ] - D_{\text{KL}}(Q_\phi(z|x) \| P(z)) $$

这个形式把目标拆成两部分：

1. **最大化重构项** $\mathbb{E}_{z \sim Q_\phi(z|x)} [ \log P_\theta(x|z) ]$：即给定编码器提取的特征 $z$，解码器能够极高概率地还原出原始数据 $x$。
2. **最小化正则项** $D_{\text{KL}}(Q_\phi(z|x) \| P(z))$：即编码器输出的分布 $Q_\phi(z|x)$ 应当尽可能接近我们事先设定的先验分布 $P(z)$（通常是标准正态分布 $\mathcal{N}(0, I)$）。

## 重参数化技巧与 VAE 实现

在具体的实现中，编码器神经网络 $Q_\phi(z|x)$ 会输出一个高斯分布的均值 $\mu$ 和方差 $\sigma^2$。然后，我们需要从这个高斯分布 $\mathcal{N}(\mu, \sigma^2)$ 中采样出一个向量 $z$，再将其输入给解码器。

这里的困难不是“随机数本身不可导”，而是直接从参数化分布采样时，普通计算图没有一条从样本 $z$ 回到分布参数 $\mu,\sigma$ 的路径。若不改写采样过程，就不能直接使用低方差的路径导数来更新编码器。

**重参数化技巧**（Reparameterization Trick）把随机性与待学习参数分开。先看一维情形：从 $z \sim \mathcal{N}(\mu, \sigma^2)$ 采样，可以改写为从固定标准正态分布取噪声后再做仿射变换。

我们可以先从一个固定的**标准正态分布** $\epsilon \sim \mathcal{N}(0, 1)$ 中采样出一个噪声数值，然后对其进行简单的线性平移和缩放：

$$ z = \mu + \sigma \cdot \epsilon $$

随机性被放到与模型参数无关的 $\epsilon$ 中，而 $\mu$ 和 $\sigma$ 只参与可微的加法与乘法。这样，给定一次采样的 $\epsilon$ 后，梯度就能沿 $z=\mu+\sigma\epsilon$ 回传。对对角高斯，多维情形按维度执行相同操作。

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

VAE 通常使用连续潜变量，而有些任务更适合让模型从有限个表示中作选择，例如语音单元、图像局部模式或离散生成词元。离散表示并不保证自动对应人类命名的语义，但它便于后续使用自回归模型在有限码本上建模。

为了在深度网络中学习这种离散结构，van den Oord 等人提出了 VQ-VAE。其核心思想是引入一个可学习的**码本**（Codebook），或者叫字典。隐变量不再是任意连续的值，而必须是码本中某一个离散向量的索引。

<div align="center">
  <img src="/figures/02-foundations/source/03-vae-and-vq/vqvae-fig1.png" alt="VQ-VAE 原图把编码器输出映射到最近码本向量，并显示量化选择如何随编码器梯度变化。" width="86%">

_图 2.3-2：VQ-VAE 原图把编码器输出映射到最近码本向量，并显示量化选择如何随编码器梯度变化。 出处：Aaron van den Oord; Oriol Vinyals; Koray Kavukcuoglu，[Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)（2017），Figure 1。_

</div>

假设我们的码本 $E = \{e_1, e_2, \dots, e_K\}$，包含 $K$ 个潜在向量，每个向量的维度为 $D$。

编码器输出一个连续的特征向量 $z_e(x)$。接下来，我们不使用重参数化技巧来采样，而是直接在码本 $E$ 中寻找距离 $z_e(x)$ 最近的那一个向量 $e_k$。这个寻找最近邻的过程被称为**向量量化**（Vector Quantization）：

$$ z_q(x) = e_k, \quad \text{where} \quad k = \arg\min_j \| z_e(x) - e_j \|_2 $$

这一步将连续的 $z_e$ 强行对齐到了离散的码本空间，输出量化后的向量 $z_q(x)$，随后 $z_q(x)$ 被送入解码器以重构图像。

**梯度的直通估计器（Straight-Through Estimator, STE）**

$\arg\min$ 选择会产生离散索引，无法按通常方式对编码器输出求导。VQ-VAE 使用**直通估计器**（STE）近似处理这条梯度路径。

前向传播仍把最近的码本向量送入解码器；反向传播时，则把解码器对 $z_q(x)$ 的梯度近似传给 $z_e(x)$，相当于把量化映射在反向阶段近似为恒等映射。

$$ \nabla_{z_e(x)} L \approx \nabla_{z_q(x)} L $$

**VQ-VAE 的损失函数**

既然反向传播绕过了量化步骤，那么码本中的向量 $e_k$ 应该如何更新呢？为了让码本向量向编码器的输出靠近，并且让编码器不要输出距离码本太远的离谱向量，VQ-VAE 定义了以下三个部分的损失函数：

1. **重构损失**：与传统 VAE 相同，确保解码器能从离散向量 $z_q$ 还原数据。
2. **码本损失（Codebook Loss）**：推动码本向量 $e_i$ 靠近编码器的输出 $z_e(x)$。使用停止梯度（Stop-Gradient）操作 `sg()` 确保梯度只流向码本。
3. **承诺损失（Commitment Loss）**：推动编码器的输出 $z_e(x)$ 不要偏离所选的码本向量太远，使得编码器“承诺”使用这个码本。同样使用停止梯度确保梯度只流向编码器。

完整的损失函数如下：

$$ L = \underbrace{\| x - D(z_q) \|_2^2}_{\text{Reconstruction}} + \underbrace{\| \text{sg}[z_e(x)] - e \|_2^2}_{\text{Codebook Loss}} + \beta \underbrace{\| z_e(x) - \text{sg}[e] \|_2^2}_{\text{Commitment Loss}} $$

<div align="center"><img src="/figures/02-foundations/latex/03-vae-and-vq/vq-stop-gradient-routing.png" alt="码本损失只更新码本向量，承诺损失只更新编码器输出，停止梯度阻断另一侧" width="86%">

_图 2.3-3：码本项把编码器输出当作固定目标，只更新码本；承诺项把码本当作固定目标，只更新编码器。_

</div>

其中 $\beta$ 是承诺损失的超参数，通常设为一个较小的值，如 $0.25$。

下面实现 VQ-VAE 的向量量化层。

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

这段代码展示了前向最近邻量化、码本/承诺损失的梯度分工，以及 `.detach()` 实现的直通估计。离散化能够压缩表示并生成有限索引序列；这些索引是否形成可解释语义，则取决于数据、目标函数和码本使用情况。

<div align="center">
  <img src="/figures/02-foundations/source/03-vae-and-vq/vqvae2-fig2.png" alt="VQ-VAE-2 用上下两级离散潜变量分别承担全局结构与局部细节，展示离散码本的层次化扩展。" width="86%">

_图 2.3-4：VQ-VAE-2 用上下两级离散潜变量分别承担全局结构与局部细节，展示离散码本的层次化扩展。 出处：Ali Razavi; Aaron van den Oord; Oriol Vinyals，[Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446)（2019），Figure 2。_

</div>

## 小结

本节推导了 **VAE** 的变分下界，并说明重参数化如何为连续随机潜变量建立可微路径；随后介绍 **VQ-VAE** 的最近邻码本、直通估计器和两项停止梯度损失。连续潜变量便于平滑采样与概率建模，离散潜变量便于压缩和序列建模；它们是不同的表示选择，而不是孰优孰劣的固定结论。
