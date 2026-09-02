# 2.4 生成模型：自回归与扩散

生成模型（Generative Models）试图用一个可学习分布逼近数据分布，并据此计算似然、补全缺失信息或生成新样本。真实数据分布通常未知，因此模型学到的是受数据、参数化方式与训练目标共同限制的近似。

<div align="center">
  <img src="/figures/02-foundations/source/04-autoregressive-and-diffusion/ddpm-fig1.png" alt="DDPM 的 CelebA-HQ 与 CIFAR-10 无条件样本展示逐步去噪模型最终能够生成的高维图像分布。" width="86%">

_图 2.4-1：DDPM 的 CelebA-HQ 与 CIFAR-10 无条件样本展示逐步去噪模型最终能够生成的高维图像分布。 出处：Jonathan Ho; Ajay Jain; Pieter Abbeel，[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)（2020），Figure 1。_

</div>

在深度学习发展的历程中，生成模型经历了多次演进。生成对抗网络（GAN）[[Goodfellow et al., 2014]](https://arxiv.org/abs/1406.2661) 用生成器与判别器之间的博弈来学习数据分布；变分自编码器（VAE）[[Kingma & Welling, 2013]](https://arxiv.org/abs/1312.6114) 则给出了基于潜变量与变分推断的概率框架。近年来，**自回归模型（Autoregressive Models）**与**扩散模型（Diffusion Models）**成为两条重要路线。GPT 展示了大规模自回归预训练在自然语言处理中的潜力 [[Radford et al., 2018]](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)，DDPM 则展示了扩散模型在图像生成上的高质量结果 [[Ho et al., 2020]](https://arxiv.org/abs/2006.11239)。

本节从概率链式法则推导自回归建模，再从高斯加噪过程推导扩散模型。重点是看清两条路线分别如何定义训练目标，以及它们在生成阶段为什么会产生不同的串行开销。

## 2.4.1 自回归模型

自回归模型的核心思想源于概率的链式法则。香农在信息论奠基论文中用有限阶马尔可夫过程构造了文本近似 [[Shannon, 1948]](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)，这可以看作统计语言模型的重要早期思想。进入深度学习时代后，Bengio 等人提出神经概率语言模型 [[Bengio et al., 2003]](https://www.jmlr.org/papers/v3/bengio03a.html)，Transformer 又提供了更适合并行训练的序列架构 [[Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762)。

<div align="center">
  <img src="/figures/02-foundations/source/04-autoregressive-and-diffusion/pixelrnn-fig2.png" alt="PixelRNN 的因果掩码图明确标出生成当前像素时只能依赖左侧和上方已经生成的像素。" width="86%">

_图 2.4-2：PixelRNN 的因果掩码图明确标出生成当前像素时只能依赖左侧和上方已经生成的像素。 出处：Aaron van den Oord et al.，[Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)（2016），Figure 2。_

</div>

### 联合概率的分解

在高中数学中，我们学习过条件概率的定义。对于任意两个事件 $A$ 和 $B$，它们的联合概率可以表示为边缘概率与条件概率的乘积：
$$P(A, B) = P(A)P(B|A)$$

现在，假设我们观察到的数据是一个序列，例如一段文本、一段语音或一串时间序列数据。我们将序列中的每一个元素视为一个随机变量，记作 $\mathbf{x} = (x_1, x_2, \dots, x_T)$。我们希望对这个长度为 $T$ 的序列的联合概率分布 $p(\mathbf{x})$ 进行建模。

利用概率的链式法则，我们可以将上述二维的条件概率推广到 $T$ 维。具体而言，联合概率可以被严格且无损地分解为一系列条件概率的乘积：
$$p(x_1, x_2, \dots, x_T) = p(x_1) \cdot p(x_2 | x_1) \cdot p(x_3 | x_1, x_2) \dots p(x_T | x_1, \dots, x_{T-1})$$

将其写为连乘的形式：
$$p(\mathbf{x}) = \prod_{t=1}^T p(x_t \mid x_{1:t-1})$$

其中 $x_{1:t-1}$ 表示从序列起点到时刻 $t-1$ 的历史变量。这个恒等分解把联合分布建模转化为一系列条件预测问题：每一步根据已经出现的内容预测下一个元素。

### 模型的参数化

链式法则本身精确成立，但随着时间步 $t$ 增加，条件历史 $x_{1:t-1}$ 会越来越长。如何在有限计算预算下利用这些变长历史来估计 $p(x_t \mid x_{1:t-1})$，是自回归模型的核心设计问题。

假设我们使用一个具有参数 $\theta$ 的神经网络来拟合这个条件概率，记为 $p_\theta(x_t \mid x_{1:t-1})$。训练目标则是最大化真实数据分布在模型上的对数似然（Log-Likelihood）：
$$\mathcal{L}(\theta) = \log p_\theta(\mathbf{x}) = \sum_{t=1}^T \log p_\theta(x_t \mid x_{1:t-1})$$

在工程实现上，我们通常采用固定长度的窗口（即马尔可夫假设）来截断历史信息，或者使用循环神经网络（RNN）将其压缩为隐状态，亦或是使用 Transformer 的因果注意力机制（Causal Attention）直接对历史序列进行并行化建模。为了使得推导过程更具象，我们将从一个最基础的一维时间序列预测任务入手。

<div align="center">
  <img src="/figures/02-foundations/source/04-autoregressive-and-diffusion/wavenet-fig2.png" alt="WaveNet 的因果卷积堆栈把每个音频采样点限制为只读取过去，从网络结构上实现自回归顺序。" width="86%">

_图 2.4-3：WaveNet 的因果卷积堆栈把每个音频采样点限制为只读取过去，从网络结构上实现自回归顺序。 出处：Aäron van den Oord et al.，[WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)（2016），Figure 2。_

</div>

### 自回归模型的实现代码实践

为了验证理论，我们将构建一个最简单的自回归模型来预测连续的时间序列（我们选用带有随机噪声的正弦波）。模型仅利用过去固定长度 $\tau$ 的数据点 $x_{t-\tau}, \dots, x_{t-1}$，来预测 $x_t$。

先生成一段含少量噪声的正弦波作为训练样本。

```python
import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

# 生成总共1000个数据点
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
# 正弦波加上均值为0，标准差为0.2的高斯噪声
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
```

为了将其转化为自回归预测问题，我们需要将一维的序列切分为特征-标签对。我们设定时间窗口大小 $\tau = 4$。这意味着特征是长度为 4 的向量 $\mathbf{x}_t = [x_{t-4}, x_{t-3}, x_{t-2}, x_{t-1}]^\top$，而对应的标签是标量 $x_t$。

接着把原始序列重组为特征矩阵和标签向量。

```python
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size = 16
dataset = data.TensorDataset(features, labels)
data_iter = data.DataLoader(dataset, batch_size, shuffle=True)
```

这里用一个小型多层感知机（MLP）估计条件均值。因为任务是连续值回归，我们采用均方误差（MSE）；若进一步假设观测噪声为固定方差的高斯分布，最小化 MSE 等价于最大化相应条件对数似然。

下面定义 MLP 与训练循环。

```python
# 定义一个包含两个隐藏层的简单MLP
def get_net():
    net = nn.Sequential(nn.Linear(tau, 10), nn.ReLU(),
                        nn.Linear(10, 10), nn.ReLU(),
                        nn.Linear(10, 1))
    # 初始化权重
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    return net

loss = nn.MSELoss()
net = get_net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# 训练模型
epochs = 5
for epoch in range(epochs):
    for X, y in data_iter:
        optimizer.zero_grad()
        l = loss(net(X), y)
        l.backward()
        optimizer.step()
    print(f'epoch {epoch + 1}, loss: {l.item():f}')
```

这个例子体现了自回归模型的核心逻辑：把序列任务拆成逐步条件预测。训练时可以并行计算多个已知目标位置，但生成时后一步依赖前一步的新输出，因此通常需要串行执行。

## 2.4.2 扩散模型

不同于自回归模型沿序列逐项生成数据，**扩散模型（Diffusion Models）**学习逐步逆转加噪过程。Sohl-Dickstein 等人从非平衡热力学得到启发，提出了相应的生成建模方法 [[Sohl-Dickstein et al., 2015]](https://arxiv.org/abs/1503.03585)；Ho 等人的 DDPM 随后用简化的噪声预测目标取得了高质量图像生成结果 [[Ho et al., 2020]](https://arxiv.org/abs/2006.11239)。

要理解扩散模型，我们需要暂时放下对数据结构的特定假设（如序列长度），并将目光聚焦于数据在状态空间中的演化。

> 可以把前向过程理解为不断向清晰图像加入少量高斯噪声，使其逐步接近标准高斯分布。生成模型学习的不是字面意义上的逆物理过程，而是一个参数化的反向转移分布：从噪声样本出发，逐步得到更符合数据分布的样本。

### 前向过程：数学视角的加噪

我们将数据本身（例如一张清晰的图片）定义为随机变量 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$。前向过程（Forward Process，即扩散过程）是一个由固定参数定义的马尔可夫链（Markov Chain）。在链的每一步，我们向当前状态 $\mathbf{x}_{t-1}$ 添加微小的高斯噪声，得到下一状态 $\mathbf{x}_t$。

从高中统计学可知，一维正态分布的概率密度函数由均值和方差完全决定。在多维空间中，我们使用协方差矩阵。前向过程的单步转移概率定义为：
$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

这里，$\beta_t \in (0, 1)$ 是预先设定好的**方差调度参数（Variance Schedule）**。通常，随着时间步 $t$ 从 $1$ 增加到 $T$，$\beta_t$ 会逐渐增大。

我们来仔细拆解该公式中的物理量含义。为什么均值要乘以一个衰减因子 $\sqrt{1 - \beta_t}$？假设原始数据 $\mathbf{x}_0$ 经过归一化，具有近似为 $0$ 的均值和单位方差 $\mathbf{I}$。如果不进行衰减而直接累加方差为 $\beta_t$ 的噪声，多次迭代后 $\mathbf{x}_t$ 的总体方差将会无限发散。引入 $\sqrt{1 - \beta_t}$ 的作用是使得该过程成为一个**方差保持（Variance Preserving）**操作。根据方差的加法性质：
$$\text{Var}(\mathbf{x}_t) = (\sqrt{1 - \beta_t})^2 \text{Var}(\mathbf{x}_{t-1}) + \beta_t = (1 - \beta_t) \cdot 1 + \beta_t = 1$$
若前一时刻近似具有零均值和单位方差，这一步会保持该方差。选择合适的噪声调度并取足够大的 $T$ 后，$\mathbf{x}_T$ 会近似标准高斯分布；有限步下通常不是数学上完全相等。

前向高斯过程还有一个实用的代数性质：利用正态分布的闭包与重参数化，可以直接写出从 $\mathbf{x}_0$ 到任意 $\mathbf{x}_t$ 的边缘分布。
令 $\alpha_t = 1 - \beta_t$，并定义累积乘积 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。我们可以将前向步骤写为等式形式：
$$\mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}, \quad \text{其中} \ \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

将 $\mathbf{x}_{t-1}$ 继续展开：

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}} \boldsymbol{\epsilon}_{t-2} \right) + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \underbrace{\sqrt{\alpha_t(1 - \alpha_{t-1})} \boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}}_{\text{两个独立高斯变量之和}}
\end{aligned}
$$

利用独立正态变量之和仍服从正态分布的性质，其新方差为 $\alpha_t(1 - \alpha_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t \alpha_{t-1}$。继续递推可得：
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$$

<div align="center"><img src="/figures/02-foundations/latex/04-autoregressive-and-diffusion/diffusion-direct-sampling.png" alt="原始样本和标准高斯噪声分别按互补系数缩放，再相加得到任意时刻的扩散样本" width="86%">

_图 2.4-4：累积系数同时决定保留多少原始信号、注入多少标准高斯噪声，因此训练时可跳过中间链直接得到 x_t。_

</div>

其中总噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。相应地，边缘条件概率分布为：
$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$$

训练时不必逐步模拟前向马尔可夫链；给定随机时间步 $t$ 和一次高斯噪声采样，就能直接构造对应的 $\mathbf{x}_t$。

### 逆向过程：学习去噪分布

现在我们来看逆向过程。如果前向过程是向图像中加注不可逆的无序性，那么逆向过程的任务就是从纯高斯噪声 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发，逐步去除噪声，最终还原出清晰的结构。

数学上，真实反向转移概率由贝叶斯公式给出：$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$。当每一步的 $\beta_t$ 足够小（即扩散步长极短）时，理论上可以证明真实的逆向分布也会趋近于一个高斯分布。然而，由于我们不知道总体的 $\mathbf{x}_0$（这正是我们要生成的），因此真实的反向转移分布 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ 是难以精确计算的。

这就轮到深度学习登场了。我们构建一个由参数 $\theta$ 神经网络参数化的高斯分布，来近似这个未知的真实逆向分布：
$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)\right)$$

原始 DDPM 的常见设定是固定或预先指定反向方差，使网络主要学习均值相关参数；后续扩散模型也有学习方差的变体，因此这不是扩散模型的必要条件。

### 从证据下界 (ELBO) 到简化的损失函数

训练逆向网络时，可以从对数似然 $\log p_\theta(\mathbf{x}_0)$ 的证据下界（Evidence Lower Bound, ELBO）出发。整理马尔可夫链中的各项后，目标包含一系列时间步上的 KL 散度：
$$\mathcal{L}_{VLB} = \mathbb{E}_q \left[ \sum_{t>1} D_{KL} \big( q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \big) \right] + C$$

其中，前向后验 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ 有闭式高斯形式。两个高斯分布的 KL 散度一般同时包含均值与协方差项；当协方差固定或相关项与待学习均值无关时，优化才可化为带权的均值平方误差。

DDPM 将反向均值重新参数化为噪声预测器 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)$。在该参数化下，变分目标中的相应项可写成带时间权重的噪声均方误差；论文进一步采用去掉部分权重的简化目标 $\mathcal{L}_{\text{simple}}$。它与完整加权 ELBO 密切相关，但不能笼统地称为完全等价：
$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \, t \right) \right\|^2 \right]$$

该公式深刻地揭示了扩散模型训练的本质：**给定任意时间步 $t$ 下被噪声污染的图像 $\mathbf{x}_t$，神经网络试图估计并剥离出最初添加到这幅图像上的纯噪声 $\boldsymbol{\epsilon}$。**

### 扩散模型前向过程的代码实践

逆向网络通常采用 U-Net 等结构，而前向加噪只需要几项张量运算。下面实现这一过程。

先定义总时间步 $T$ 和线性 $\beta$ 调度，并预计算累积系数 $\bar{\alpha}_t$。

```python
# 扩散步数
num_timesteps = 1000

# 线性方差调度策略（Variance Schedule）
# beta_t 从 1e-4 线性增加到 0.02
betas = torch.linspace(1e-4, 0.02, num_timesteps)

# 计算 alpha 和 alpha_bar
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

# 为了后续根据时间步 t 索引系数时保持张量维度正确
# 这里计算 sqrt(alpha_bar) 和 sqrt(1 - alpha_bar)
sqrt_alphas_bar = torch.sqrt(alphas_bar)
sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
```

有了这些预计算的系数，我们可以立刻实现该公式所描述的前向快速加噪。

函数 `q_sample` 可在一次计算中从 $\mathbf{x}_0$ 得到任意时间步 $t$ 的带噪样本。

```python
def q_sample(x_0, t, noise=None):
    """
    实现了前向扩散过程 x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    # 提取当前批次中每一个样本对应的 sqrt_alphas_bar_t
    sqrt_alphas_bar_t = sqrt_alphas_bar[t].view(-1, 1, 1, 1)

    # 提取对应的 sqrt(1 - alphas_bar_t)
    sqrt_one_minus_alphas_bar_t = sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)

    # 根据重参数化公式叠加噪声
    x_t = sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t * noise
    return x_t
```

由此可见，前向过程在代码层面只需几行张量运算即可实现，这种无需循环迭代的特性，使得扩散模型能够在 GPU 上以极高的批处理效率进行大规模并行训练。

## 2.4.3 比较与联系

自回归模型与扩散模型虽然采用了迥异的数学视角，但它们之间存在深刻的联系。
自回归模型用概率链式法则分解联合分布，适合具有明确顺序的离散或连续序列，但逐词元生成通常是串行的。扩散模型在每个去噪步内可以同时更新所有空间位置，却仍需多次调用网络。两者的速度和质量取决于具体参数化、采样器与数据类型，不能简单归结为“文本用自回归、图像用扩散”。

## 2.4.4 小结

- **自回归模型**基于严格的概率链式法则进行联合分布建模，将高维生成任务转化为条件序列预测任务，在自然语言处理中占据核心地位。
- **扩散模型**从非平衡热力学汲取灵感，其前向过程通过可控方差的马尔可夫链注入噪声破坏数据，逆向过程则由神经网络学习去除噪声以从混沌中恢复结构。
- 借助高斯分布的闭包性质，扩散前向过程可以写出闭式边缘分布，从而直接在任意时间步采样训练对。
- 两类模型都在逼近数据分布，但采用不同的分解方式，并把计算成本放在不同位置。

## 2.4.5 练习

1. 在自回归模型中，如果我们不采用固定窗口，而是想捕捉无限长的历史依赖，我们应该采用什么样的神经网络架构？（提示：思考循环机制或是全局注意力机制。）
2. 假设只有两个扩散步，$\alpha_1=0.8$、$\alpha_2=0.9$。计算 $\bar{\alpha}_2$，再求 $\mathbf{x}_2$ 中信号系数的平方与噪声系数的平方，验证两者之和为 1。
3. 修改 2.4.1 节中的自回归代码，将其改写为一个具有单层 RNN（如 `nn.RNN`）的模型结构。观察其在正弦波预测任务上相比于简单的多层感知机是否有提升。
4. 为什么我们在构建扩散模型的逆向网络时，通常选择预测噪声 $\boldsymbol{\epsilon}$ 而不是直接预测干净的图像 $\mathbf{x}_0$？（提示：结合图像在高度加噪时 $\mathbf{x}_0$ 信息的保留量，思考哪一种预测目标对神经网络的梯度优化更为平滑和稳定。）
