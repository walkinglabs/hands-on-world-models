# 2.4. 生成模型（自回归与扩散）
:label:sec_autoregressive_and_diffusion

生成模型（Generative Models）是无监督学习的核心研究方向之一。如果说判别模型（如图像分类器）旨在寻找决策边界以区分不同类别的数据，那么生成模型的终极目标则是**学习数据本身的真实概率分布**。当我们掌握了数据的真实分布，不仅能对现有数据进行密度估计，更能够从中采样，创造出前所未有的新样本。

在深度学习发展的历程中，生成模型经历了多次演进。早期的受限玻尔兹曼机（RBM）受限于训练效率；随后的生成对抗网络（GAN）[Goodfellow et al., 2014] 通过博弈论思想极大地提升了生成质量，但面临训练不稳定和模式崩塌的挑战；变分自编码器（VAE）[Kingma & Welling, 2013] 则提供了坚实的概率推断框架，但生成的样本往往不够清晰。近年来，**自回归模型（Autoregressive Models）**与**扩散模型（Diffusion Models）**凭借其强大的建模能力和高度的稳定性，逐渐成为生成领域的两大基石。自回归模型在自然语言处理领域（如 GPT 系列模型）占据统治地位 [Radford et al., 2018]，而扩散模型则在图像和音频生成领域（如 DALL-E 2, Stable Diffusion）展现出了惊人的表现 [Ho et al., 2020]。

本节将从基础概率论出发，逐步剥离高维空间带来的复杂性，深入推导这两类生成模型的数学本质与工程实现。我们将秉持学术严谨性，不略过核心推导，同时以最直观的物理与几何视角解析其中看似晦涩的方程。

## 2.4.1. 自回归模型
:label:subsec_autoregressive

自回归模型的核心思想源于极其基础的概率论定理：**概率的链式法则（Chain Rule of Probability）**。早在 1948 年，香农（Claude Shannon）在创立信息论时提出的 $n$-gram 模型，本质上即是一种朴素的自回归生成框架。而在深度学习时代，Bengio 等人 [Bengio et al., 2003] 提出的神经语言模型，以及随后基于 Transformer 架构 [Vaswani et al., 2017] 的巨型模型，皆是这一古老思想在算力与数据加持下的现代重塑。

### 联合概率的分解

在高中数学中，我们学习过条件概率的定义。对于任意两个事件 $A$ 和 $B$，它们的联合概率可以表示为边缘概率与条件概率的乘积：
$$P(A, B) = P(A)P(B|A)$$

现在，假设我们观察到的数据是一个序列，例如一段文本、一段语音或一串时间序列数据。我们将序列中的每一个元素视为一个随机变量，记作 $\mathbf{x} = (x_1, x_2, \dots, x_T)$。我们希望对这个长度为 $T$ 的序列的联合概率分布 $p(\mathbf{x})$ 进行建模。

利用概率的链式法则，我们可以将上述二维的条件概率推广到 $T$ 维。具体而言，联合概率可以被严格且无损地分解为一系列条件概率的乘积：
$$p(x_1, x_2, \dots, x_T) = p(x_1) \cdot p(x_2 | x_1) \cdot p(x_3 | x_1, x_2) \dots p(x_T | x_1, \dots, x_{T-1})$$

将其写为连乘的形式：
$$p(\mathbf{x}) = \prod_{t=1}^T p(x_t \mid x_{1:t-1})$$
:eqlabel:eq_ar_chain_rule

其中 $x_{1:t-1}$ 表示从序列起点到时刻 $t-1$ 的所有历史变量。公式 :eqref:eq_ar_chain_rule 即是自回归模型的**绝对核心**。它告诉我们：**生成一个高维复杂序列的任务，可以等价转化为一系列单步预测任务**——即在已知过去所有历史信息的条件下，预测下一个元素的概率分布。

### 模型的参数化

虽然公式 :eqref:eq_ar_chain_rule 在数学上完美成立，但在实际计算中，随着时间步 $t$ 的增加，条件部分 $x_{1:t-1}$ 的长度是动态增长的，且可能极其庞大。如何有效地利用这些变长历史信息来估算条件概率 $p(x_t \mid x_{1:t-1})$，构成了自回归模型架构设计的核心挑战。

假设我们使用一个具有参数 $\theta$ 的神经网络来拟合这个条件概率，记为 $p_\theta(x_t \mid x_{1:t-1})$。训练目标则是最大化真实数据分布在模型上的对数似然（Log-Likelihood）：
$$\mathcal{L}(\theta) = \log p_\theta(\mathbf{x}) = \sum_{t=1}^T \log p_\theta(x_t \mid x_{1:t-1})$$
:eqlabel:eq_ar_nll

在工程实现上，我们通常采用固定长度的窗口（即马尔可夫假设）来截断历史信息，或者使用循环神经网络（RNN）将其压缩为隐状态，亦或是使用 Transformer 的因果注意力机制（Causal Attention）直接对历史序列进行并行化建模。为了使得推导过程更具象，我们将从一个最基础的一维时间序列预测任务入手。

### 自回归模型的实现代码实践

为了验证理论，我们将构建一个最简单的自回归模型来预测连续的时间序列（我们选用带有随机噪声的正弦波）。模型仅利用过去固定长度 $\tau$ 的数据点 $x_{t-\tau}, \dots, x_{t-1}$，来预测 $x_t$。

(**首先，我们生成一段含有少量噪声的正弦波时间序列数据作为训练样本。**)

```{.python .input}
#@tab pytorch
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

(**接下来，我们将原始时间序列重组为特征矩阵和标签向量，并构建数据迭代器。**)

```{.python .input}
#@tab pytorch
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size = 16
dataset = data.TensorDataset(features, labels)
data_iter = data.DataLoader(dataset, batch_size, shuffle=True)
```

现在，我们定义一个极其简单的多层感知机（MLP）作为我们的自回归模型 $p_\theta(x_t \mid \mathbf{x}_{t-1})$。因为这是一个回归任务，我们输出预测的具体数值，并使用均方误差（MSE）作为损失函数。需要注意的是，在概率视角下，最小化 MSE 损失等价于最大化假设误差服从高斯分布的对数似然。

(**我们定义多层感知机模型并编写标准的训练循环。**)

```{.python .input}
#@tab pytorch
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

在这个简单的例子中，我们见证了自回归模型的核心逻辑：将复杂的序列生成任务拆解为一步步的条件概率预测。这种严格按照时序逐步推进的特性赋予了自回归模型极强的理论完备性，但也导致了其在生成阶段必须串行计算，从而带来推理效率的瓶颈。

## 2.4.2. 扩散模型
:label:subsec_diffusion

不同于自回归模型沿时间轴逐步生成数据，**扩散模型（Diffusion Models）**走的是另一条截然不同的路径。扩散模型从物理学中的非平衡热力学汲取灵感。最早由 Sohl-Dickstein 等人 [Sohl-Dickstein et al., 2015] 提出，随后在 Ho 等人 [Ho et al., 2020] 的去噪扩散概率模型（Denoising Diffusion Probabilistic Models, DDPM）中被大幅改进并成功应用于高分辨率图像生成。

要理解扩散模型，我们需要暂时放下对数据结构的特定假设（如序列长度），并将目光聚焦于数据在状态空间中的演化。

> 我们可以通过一个物理场景来理解这一过程：想象一杯清澈的水，我们在其中滴入一滴高浓度的墨水。随着时间的推移，墨水分子在布朗运动的驱使下随机扩散，逐渐均匀地分布在水中，最终成为一杯浑浊的淡墨水。这是一个**熵增**的过程，系统从高度结构化（墨滴聚集）演变为完全无序的随机状态（均匀分布）。
> 在计算机视觉中，这一过程等价于在一张清晰的照片上不断叠加高斯噪声，直至其变成一幅完全由随机像素构成的纯噪声图。扩散模型的天才之处在于：如果我们能够用神经网络极其精确地学习墨水扩散的“逆物理过程”（即从浑浊的水中预测出墨水分子聚拢的趋势），那么我们就能从纯随机噪声出发，“无中生有”地生成出高度结构化的清晰图像。

### 前向过程：数学视角的加噪

我们将数据本身（例如一张清晰的图片）定义为随机变量 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$。前向过程（Forward Process，即扩散过程）是一个由固定参数定义的马尔可夫链（Markov Chain）。在链的每一步，我们向当前状态 $\mathbf{x}_{t-1}$ 添加微小的高斯噪声，得到下一状态 $\mathbf{x}_t$。

从高中统计学可知，一维正态分布的概率密度函数由均值和方差完全决定。在多维空间中，我们使用协方差矩阵。前向过程的单步转移概率定义为：
$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$$
:eqlabel:eq_diffusion_forward_step

这里，$\beta_t \in (0, 1)$ 是预先设定好的**方差调度参数（Variance Schedule）**。通常，随着时间步 $t$ 从 $1$ 增加到 $T$，$\beta_t$ 会逐渐增大。

我们来仔细拆解公式 :eqref:eq_diffusion_forward_step 中的物理量含义。为什么均值要乘以一个衰减因子 $\sqrt{1 - \beta_t}$？假设原始数据 $\mathbf{x}_0$ 经过归一化，具有近似为 $0$ 的均值和单位方差 $\mathbf{I}$。如果不进行衰减而直接累加方差为 $\beta_t$ 的噪声，多次迭代后 $\mathbf{x}_t$ 的总体方差将会无限发散。引入 $\sqrt{1 - \beta_t}$ 的作用是使得该过程成为一个**方差保持（Variance Preserving）**操作。根据方差的加法性质：
$$\text{Var}(\mathbf{x}_t) = (\sqrt{1 - \beta_t})^2 \text{Var}(\mathbf{x}_{t-1}) + \beta_t = (1 - \beta_t) \cdot 1 + \beta_t = 1$$
这一设计保证了即使经过无数次扩散，最终的变量 $\mathbf{x}_T$ 也能够稳定地服从标准高斯分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$。

前向过程中存在一个极其优雅的代数性质。借助于正态分布的重参数化技巧（Reparameterization Trick），我们可以直接跳过中间步骤，推导出从初始数据 $\mathbf{x}_0$ 到任意时刻 $\mathbf{x}_t$ 的边缘分布。
令 $\alpha_t = 1 - \beta_t$，并定义累积乘积 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。我们可以将前向步骤写为等式形式：
$$\mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}, \quad \text{其中} \ \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

将 $\mathbf{x}_{t-1}$ 继续展开：
$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}} \boldsymbol{\epsilon}_{t-2} \right) + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \underbrace{\sqrt{\alpha_t(1 - \alpha_{t-1})} \boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}}_{\text{两个独立高斯变量之和}}
\end{aligned}
$$

利用独立正态变量之和依旧服从正态分布的性质，其新方差为 $\alpha_t(1 - \alpha_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t \alpha_{t-1}$。通过反复进行数学归纳（这一过程极其严密且美妙），我们最终得到：
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$$
:eqlabel:eq_diffusion_forward_xt

其中总噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。相应地，边缘条件概率分布为：
$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$$
:eqlabel:eq_diffusion_forward_marginal

这一公式极其重要：它表明我们在训练时，**不需要**一步一步地模拟马尔可夫链，而是可以直接通过单次采样获得任意时间步 $t$ 对应的加噪图像。这极大提升了模型训练的并行效率。

### 逆向过程：学习去噪分布

现在我们来看逆向过程。如果前向过程是向图像中加注不可逆的无序性，那么逆向过程的任务就是从纯高斯噪声 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发，逐步去除噪声，最终还原出清晰的结构。

数学上，真实反向转移概率由贝叶斯公式给出：$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$。当每一步的 $\beta_t$ 足够小（即扩散步长极短）时，理论上可以证明真实的逆向分布也会趋近于一个高斯分布。然而，由于我们不知道总体的 $\mathbf{x}_0$（这正是我们要生成的），因此真实的反向转移分布 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ 是难以精确计算的。

这就轮到深度学习登场了。我们构建一个由参数 $\theta$ 神经网络参数化的高斯分布，来近似这个未知的真实逆向分布：
$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)\right)$$
:eqlabel:eq_diffusion_reverse

在实践中，特别是在 DDPM 框架下，方差 $\boldsymbol{\Sigma}_\theta$ 通常被固定为非学习的常数（例如 $\beta_t \mathbf{I}$），使得神经网络的唯一任务是学习均值 $\boldsymbol{\mu}_\theta$。

### 从证据下界 (ELBO) 到简化的损失函数

为了训练这样一个逆向神经网络，我们需要最大化生成数据分布的对数似然 $\log p_\theta(\mathbf{x}_0)$。与变分自编码器（VAE）极其相似，真实的对数似然包含复杂的积分计算，因此我们转向最大化其证据下界（Evidence Lower Bound, ELBO）。经过极其严密但繁复的推导（此处略去冗长项以聚焦核心），ELBO 可以被拆解并转化为最小化一系列时间步上的 KL 散度：
$$\mathcal{L}_{VLB} = \mathbb{E}_q \left[ \sum_{t>1} D_{KL} \big( q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \big) \right] + C$$

其中，前向条件概率 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ 是存在闭式解的真实高斯分布。因为两个高斯分布之间的 KL 散度本质上是它们均值之间差距的 L2 范数，这意味着我们实际上是在训练神经网络 $\boldsymbol{\mu}_\theta$ 去预测真实的后验均值 $\tilde{\boldsymbol{\mu}}_t$。

Ho 等人在 DDPM 中提出了一个极其惊艳的观点。通过仔细代入公式 :eqref:eq_diffusion_forward_xt，发现预测图像 $\mathbf{x}_{t-1}$ 的均值，在数学上完全等价于预测注入图像的**那部分高斯噪声** $\boldsymbol{\epsilon}$。我们将神经网络重新参数化为噪声预测器 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$，并去除目标函数前面繁琐的系数权重。这产生了一个极度简洁的训练损失：
$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \, t \right) \right\|^2 \right]$$
:eqlabel:eq_diffusion_loss

公式 :eqref:eq_diffusion_loss 深刻地揭示了扩散模型训练的本质：**给定任意时间步 $t$ 下被噪声污染的图像 $\mathbf{x}_t$，神经网络试图估计并剥离出最初添加到这幅图像上的纯噪声 $\boldsymbol{\epsilon}$。**

### 扩散模型前向过程的代码实践

尽管逆向神经网络（通常采用 U-Net 架构）的构建较为复杂，但扩散模型的前向加噪过程极其简洁且优雅。我们将通过代码直观展示如何实现这一从秩序到混沌的转变。

(**我们定义预设的总时间步 $T$ 及线性的 $\beta$ 调度，然后利用公式提前计算出所有需要的累积系数 $\bar{\alpha}_t$。**)

```{.python .input}
#@tab pytorch
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

有了这些预计算的系数，我们可以立刻实现方程 :eqref:eq_diffusion_forward_xt 所描述的前向快速加噪。

(**我们编写前向过程的函数 `q_sample`。该函数能够在单次计算中，直接从初始数据 $\mathbf{x}_0$ 获得任意时间步 $t$ 下带噪结果 $\mathbf{x}_t$。**)

```{.python .input}
#@tab pytorch
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

## 2.4.3. 比较与联系
:label:subsec_ar_vs_diffusion

自回归模型与扩散模型虽然采用了迥异的数学视角，但它们之间存在深刻的联系。
自回归模型依靠概率链式法则严格拟合序列数据的联合概率，在自然语言等具有强烈顺序依存关系的数据中表现无懈可击。但其逐个标记（Token）生成的串行特性成为了推理速度的天然瓶颈。
相比之下，扩散模型通过在连续状态空间中平滑过渡，并在生成时可以一次性对整张图像的所有维度进行更新，从而更适合高维度、高分辨率的空间结构数据（如图像生成）。不过，传统的扩散模型需要数以百计的去噪步数才能生成高质量样本，这一反复调用庞大神经网络的过程同样造成了计算资源的巨量消耗。当前前沿的研究正在探索如何桥接这两者的鸿沟，例如结合自回归的离散性与扩散模型的连续性，或是研发更快的非马尔可夫扩散采样算法。

## 2.4.4. 小结

* 自回归模型基于严格的概率链式法则进行联合分布建模，将高维生成任务转化为条件序列预测任务，在自然语言处理中占据核心地位。
* 扩散模型从非平衡热力学汲取灵感，其前向过程通过可控方差的马尔可夫链注入噪声破坏数据，逆向过程则由神经网络学习去除噪声以从混沌中恢复结构。
* 借助于重参数化技巧，扩散模型的前向过程可以写出闭式解 :eqref:eq_diffusion_forward_xt，使我们能够直接在任意时间步上采样以计算重建损失。
* 两类模型各有优势。自回归注重时序推断的绝对严谨，扩散模型则在学习高维复杂连续数据的流形上表现卓越。

## 2.4.5. 练习

1. 在自回归模型中，如果我们不采用固定窗口，而是想捕捉无限长的历史依赖，我们应该采用什么样的神经网络架构？（提示：思考循环机制或是全局注意力机制。）
2. 在公式 :eqref:eq_diffusion_forward_xt 的推导过程中，假设 $\alpha_t$ 和 $\alpha_{t-1}$ 分别为 0.9 和 0.8。此时 $\mathbf{x}_t$ 中来自真实数据 $\mathbf{x}_0$ 的方差占比以及来自噪声的方差占比分别是多少？验证它们的总和是否严格为 1。（提示：分别计算 $\bar{\alpha}_t$ 和 $1 - \bar{\alpha}_t$ 的数值。）
3. 修改 2.4.1 节中的自回归代码，将其改写为一个具有单层 RNN（如 `nn.RNN`）的模型结构。观察其在正弦波预测任务上相比于简单的多层感知机是否有提升。
4. 为什么我们在构建扩散模型的逆向网络时，通常选择预测噪声 $\boldsymbol{\epsilon}$ 而不是直接预测干净的图像 $\mathbf{x}_0$？（提示：结合图像在高度加噪时 $\mathbf{x}_0$ 信息的保留量，思考哪一种预测目标对神经网络的梯度优化更为平滑和稳定。）

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
