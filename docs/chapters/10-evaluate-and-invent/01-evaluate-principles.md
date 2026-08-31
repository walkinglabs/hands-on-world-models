# 10.1 世界模型的评估原则与核心指标

> **本章导读**
>
> **讲什么：** 本章把前面建成的模型放到同一套检验框架中。我们会区分组件指标与系统结果，检查单步预测、多步漂移、动作反事实、陌生场景和下游控制，再从稳定失败中提出解释、设计对照实验，并形成下一台世界模型的最小改动方案。
>
> **为什么漂亮样例和较低损失不能作为终点：** 一个视频模型可以生成清晰画面，却在替换方向盘动作后仍输出相同未来；一个潜在模型也可以有很低的训练误差，却被规划器带到数据之外并自信预测。世界模型的价值取决于它是否保留任务所需的信息、是否响应动作，以及错误是否会在闭环中被放大。
>
> **故事线：** `先检查预测与不确定性 → 再检查动作是否改变未来 → 展开长时程并接入下游任务 → 在分布外场景寻找稳定失败 → 比较多种解释 → 用可证伪实验设计下一台模型`

分类器通常可以用准确率概括一部分能力，世界模型却很难被一个数压缩。它既要预测观测或隐状态，也要响应动作、维持长时程一致性，并最终帮助智能体完成任务。因此，评估的目标不是判定模型是否“真正理解”世界，而是把这些可检验的能力逐项拆开。

<div align="center">
<img src="/figures/10-evaluate-and-invent/source/01-evaluate-principles/worldmodels-fig12.png" alt="World Models 的控制器在真实 CarRacing 环境中稳定通过弯道，说明世界模型最终要由闭环行为而非单一重构分数检验。" width="86%">

_图 10.1-1：World Models 的控制器在真实 CarRacing 环境中稳定通过弯道，说明世界模型最终要由闭环行为而非单一重构分数检验。 出处：David Ha；Jürgen Schmidhuber，[World Models](https://arxiv.org/abs/1803.10122)（2018），Figure 12。_
</div>

本节从点预测误差出发，过渡到概率预测、潜变量诊断和任务效用。每个指标都回答一个具体问题，也都有不能回答的问题。

## 历史背景与学术脉络

在智能体（Agent）研究的早期阶段，研究者们主要依赖无模型（Model-Free）的强化学习算法。这类算法通过试错直接学习策略，但其样本效率极低。为了让智能体能够像人类一样在脑海中“预演”未来，基于模型（Model-Based）的方法逐渐兴起。

Ha 和 Schmidhuber 的 _World Models_ 用 VAE 压缩视觉输入，并用 MDN-RNN 预测潜变量与终止信号 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。VAE 的重构质量是组件诊断之一，但论文对完整系统的关键检验仍是控制器在 CarRacing 与 VizDoom 中取得的回报；不能把当时的评估概括为只看像素重构误差。

像素级重构也有明显局限：现实观测中包含树叶摆动、纹理和传感器噪声等难预测细节。更低的像素误差不一定意味着模型保留了对控制最重要的信息，因此它只能作为组件诊断，不能替代动作响应和下游任务评测。

PlaNet 首先提出 RSSM，Dreamer 随后用 RSSM 的潜在想象轨迹训练策略 [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603)。Dreamer 的训练仍包含观测、奖励和折扣预测，评估则以环境回报和数据效率为主。LeCun 的立场文章进一步主张在抽象表征空间中预测，而不是为不可预测的像素细节分配容量 [[LeCun, 2022]](https://openreview.net/forum?id=BZ5a1r-kVsf)。

这些历史沿革揭示了世界模型评估的两个核心维度：**动力学预测的准确性（Predictive Accuracy）**与**下游任务的效用价值（Behavioral Utility）**。

## 动力学预测的准确性：从标量到矩阵

世界模型的核心任务是预测未来。为了从最基础的概念起步，我们暂且抛开高维图像和复杂的神经网络，回到高中物理中最经典的运动学场景。

### 确定性系统中的预测误差

先看一个理想化例子：忽略空气阻力，并且已知完整初始状态、重力和采样间隔时，平抛小球的下一时刻位置由运动方程确定。这个假设刻意排除了测量误差和未知扰动，方便我们先讨论点预测误差。

假设我们的世界模型（在这里是一个简单的物理公式）给出了预测值 $\hat{x}_{t+1}$。此时，评估模型预测好坏的最自然方式，就是计算预测位置与真实观测位置之间的距离。这在数学上体现为绝对误差或平方误差。为了便于后续求导优化，我们通常选择平方误差：

$$ e_{t+1} = (x_{t+1} - \hat{x}_{t+1})^2 $$

该公式描述的是单一物理量的预测误差。在真实的世界模型中，状态通常不是一个单一的标量，而是一个包含了众多属性（如位置、速度、姿态、环境特征等）的高维向量 $\mathbf{z}_{t+1} \in \mathbb{R}^d$。

此时，我们需要将一维的误差公式严谨地推广到高维向量空间。对于两个向量 $\mathbf{z}_{t+1}$ 和 $\mathbf{\hat{z}}_{t+1}$，它们之间的距离可以通过欧几里得距离（L2范数）的平方来衡量，这便是多维隐空间中的均方误差（Mean Squared Error, MSE）：

$$ \mathcal{L}_{\text{MSE}} = \frac{1}{d} \sum_{i=1}^{d} (z_{t+1}^{(i)} - \hat{z}_{t+1}^{(i)})^2 = \frac{1}{d} \|\mathbf{z}_{t+1} - \mathbf{\hat{z}}_{t+1}\|_2^2 $$

在这里，$z_{t+1}^{(i)}$ 表示真实状态向量的第 $i$ 个分量，$\|\cdot\|_2$ 表示 L2 范数。这个指标容易计算，但只描述一个目标状态与一个点预测之间的平均距离。

### 随机性与概率分布的引入

现实世界并非真空中的平抛运动。假设小球在飞行过程中会受到不可预测的随机阵风影响。此时，即便给定初始状态，小球在下一时刻的确切位置也是不确定的。如果我们的模型仍然只输出一个确定性的点预测 $\mathbf{\hat{z}}_{t+1}$，它将不可避免地产生巨大的误差，因为无论它预测哪个具体的点，阵风都可能让小球偏离。

为了严谨地描述这种随机性，世界模型的输出不再是一个确定的点，而必须是一个**概率分布（Probability Distribution）**。

在一维场景下，假设模型预测小球下一时刻的位置服从一个均值为 $\hat{\mu}_{t+1}$、方差为 $\hat{\sigma}_{t+1}^2$ 的高斯分布。根据高斯分布的概率密度函数，真实观测值 $x_{t+1}$ 出现在该分布中的概率密度（似然）为：

$$ p(x_{t+1} | \hat{\mu}_{t+1}, \hat{\sigma}_{t+1}^2) = \frac{1}{\sqrt{2\pi\hat{\sigma}_{t+1}^2}} \exp\left(-\frac{(x_{t+1} - \hat{\mu}_{t+1})^2}{2\hat{\sigma}_{t+1}^2}\right) $$

一个好的世界模型，应当使其预测的概率分布在真实观测值处具有最大的概率密度。这就是**最大似然估计（Maximum Likelihood Estimation, MLE）**的核心思想。为了将连乘转化为求和并避免数值下溢，我们通常对似然函数取对数，并加上负号，得到负对数似然（Negative Log-Likelihood, NLL）：

$$ \text{NLL} = - \ln p(x_{t+1}) = \frac{1}{2} \ln(2\pi\hat{\sigma}_{t+1}^2) + \frac{(x_{t+1} - \hat{\mu}_{t+1})^2}{2\hat{\sigma}_{t+1}^2} $$

<div align="center">
<img src="/figures/10-evaluate-and-invent/latex/01-evaluate-principles/gaussian-nll-variance-balance.png" alt="固定残差时，高斯负对数似然由随标准差下降的残差项和随标准差上升的对数项共同决定" width="86%">

_图 10.1-2：增大预测标准差会降低残差惩罚，却会抬高对数尺度项；两项平衡使最优标准差等于残差绝对值。本文根据上式绘制。_
</div>

仔细观察第二项：当方差固定且不依赖样本时，最小化高斯 NLL 与最小化平方误差具有相同的最优均值预测。让模型同时预测方差，则可以表达条件分布的尺度；但方差也可能被模型用来掩盖均值误差，所以还要用留出数据检查校准，而不能只看 NLL。

<div align="center">
<img src="/figures/10-evaluate-and-invent/source/01-evaluate-principles/calibration-fig3.png" alt="校准回归把名义置信水平与实际覆盖率对齐，展示为什么概率预测除 NLL 外还必须检查不确定性的校准。" width="86%">

_图 10.1-3：校准回归把名义置信水平与实际覆盖率对齐，展示为什么概率预测除 NLL 外还必须检查不确定性的校准。 出处：Volodymyr Kuleshov；Nathan Fenner；Stefano Ermon，[Accurate Uncertainties for Deep Learning Using Calibrated Regression](https://arxiv.org/abs/1807.00263)（2018），Figure 3。_
</div>

推广到高维向量空间，假设模型预测的状态服从多变量高斯分布 $\mathcal{N}(\boldsymbol{\hat{\mu}}_{t+1}, \boldsymbol{\hat{\Sigma}}_{t+1})$，其中协方差矩阵 $\boldsymbol{\hat{\Sigma}}_{t+1}$ 必须对称正定。其负对数似然为：

$$ \mathcal{L}_{\text{NLL}} = \frac{d}{2}\ln(2\pi) + \frac{1}{2}\ln|\boldsymbol{\hat{\Sigma}}_{t+1}| + \frac{1}{2}(\mathbf{z}_{t+1} - \boldsymbol{\hat{\mu}}_{t+1})^\top \boldsymbol{\hat{\Sigma}}_{t+1}^{-1} (\mathbf{z}_{t+1} - \boldsymbol{\hat{\mu}}_{t+1}) $$

在实际实现中，为了降低计算复杂度，常假设给定条件后各维独立，即 $\boldsymbol{\hat{\Sigma}} = \text{diag}(\hat{\sigma}_1^2, \dots, \hat{\sigma}_d^2)$。此时 NLL 是各维一维 NLL 的和；后面的代码实现正是这种对角高斯，而不是完整协方差模型。

## 潜变量推断与变分下界 (ELBO)

现代世界模型（如 RSSM）在评估时面临一个更加复杂的挑战：隐状态 $\mathbf{z}_t$ 是不可见的（Latent）。我们能观测到的只有图像序列 $\mathbf{x}_{1:T}$ 和动作序列 $\mathbf{a}_{1:T}$。

这就要求模型同时具备两个能力：

1. **后验推断（Posterior Inference）**：根据当前的真实观测图像 $\mathbf{x}_t$ 提取出真实的隐状态分布 $q(\mathbf{z}_t | \mathbf{x}_t, \dots)$。
2. **先验预测（Prior Prediction）**：在不看当前图像的情况下，仅根据历史信息 $\mathbf{z}_{t-1}$ 和动作 $\mathbf{a}_{t-1}$，预测出当前的隐状态分布 $p(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{a}_{t-1})$。

训练这类模型时，常采用变分推断（Variational Inference）框架，最大化观测边缘对数似然的下界，即证据下界（Evidence Lower Bound, ELBO）。省略跨时间求和后，单步形式可以写成：

$$ \mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(\mathbf{z}_t | \mathbf{x}_{\le t},\mathbf{a}_{<t})}\big[\ln p(\mathbf{x}_t | \mathbf{z}_t)\big] - D_{\text{KL}}\Big( q(\mathbf{z}_t | \mathbf{x}_{\le t},\mathbf{a}_{<t}) \,\Big\|\, p(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{a}_{t-1}) \Big) $$

这里写的是单位权重的 ELBO。工程实现常给 KL 项乘上系数 $\beta$ 或使用 free bits；这时它是由 ELBO 改造出的训练目标，不再等同于原始下界。

<div align="center">
<img src="/figures/10-evaluate-and-invent/source/01-evaluate-principles/betavae-fig1.png" alt="β-VAE 图示说明后验分布的重叠、KL 压力与重构区分度彼此牵制，直观对应 ELBO 两项的张力。" width="86%">

_图 10.1-4：β-VAE 图示说明后验分布的重叠、KL 压力与重构区分度彼此牵制，直观对应 ELBO 两项的张力。 出处：Christopher P. Burgess et al.，[Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599)（2018），Figure 1。_
</div>

这个式子包含两项作用不同的训练诊断量：

第一项 $\mathbb{E}_{q}[\ln p(\mathbf{x}_t | \mathbf{z}_t)]$ 是**重构似然（Reconstruction Likelihood）**。它要求基于观测图像提取出的后验隐状态，能够被解码器成功还原回原始图像。这衡量了隐状态是否保留了足够的视觉细节信息。

第二项 $D_{\text{KL}}$ 则是评估世界模型动力学预测能力的核心。KL散度（Kullback-Leibler Divergence）衡量了两个概率分布之间的差异。在该公式中，它迫使**先验预测分布** $p(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{a}_{t-1})$ 必须尽可能地逼近**后验推断分布** $q(\mathbf{z}_t | \mathbf{x}_t)$。

> 💡 **KL散度动力学匹配机制**
> 可以把后验看成“看过当前画面的定位器”，把先验看成“只凭上一状态和动作前推的预测器”。KL 衡量二者分布的差异。差异变小说明先验更接近训练后验，但不等于模型已经恢复了唯一、真实的物理状态；后验本身也可能丢失任务信息。

## 效用导向的评估 (Behavioral Utility)

尽管基于潜变量动力学的 KL 散度评估极具数学优雅性，但世界模型的最终目的往往不是为了“精确预测”，而是为了“辅助决策”。这就引出了另一类重要的评估原则：基于价值等效（Value Equivalence）的评估。

MuZero 不要求隐空间展开重建环境图像，而是训练表示、动力学与预测网络去匹配规划所需的奖励、价值和策略目标 [[Schrittwieser et al., 2020]](https://arxiv.org/abs/1911.08265)。因此，组件诊断应同时检查奖励预测、价值预测与策略分布；完整系统仍需用环境回报和搜索性能评估，不能只看两个标量的拟合误差。

<div align="center">
<img src="/figures/10-evaluate-and-invent/source/01-evaluate-principles/muzero-fig3.png" alt="MuZero 在 Go 与 Atari 上按搜索预算、训练步数和规划消融报告结果，体现任务效用必须通过真实游戏表现来检验。" width="86%">

_图 10.1-5：MuZero 在 Go 与 Atari 上按搜索预算、训练步数和规划消融报告结果，体现任务效用必须通过真实游戏表现来检验。 出处：Julian Schrittwieser et al.，[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)（2020），Figure 3。_
</div>

具体的评估指标退化为联合的损失函数，直接在模型的展开轨迹上评估：

$$ \mathcal{L}_{\text{Utility}} = \sum_{k=0}^{K} \Big[ l^r(r_{t+k}, \hat{r}_t^k) + l^v(v_{t+k}, \hat{v}_t^k) + l^p(\pi_{t+k}, \hat{\mathbf{p}}_t^k) \Big] $$

其中，$\hat{r}_t^k$, $\hat{v}_t^k$ 和 $\hat{\mathbf{p}}_t^k$ 是世界模型从状态 $\mathbf{z}_t$ 出发，向前展开 $k$ 步后预测的奖励、状态价值和策略向量。这里的 $l^r$ 和 $l^v$ 可采用平方误差或离散支持上的交叉熵，$l^p$ 通常采用交叉熵。这类目标把评价重点移向规划所需的量，但仍可能忽略训练目标未覆盖的环境信息，所以要与真实环境中的搜索或控制结果一起报告。

## 代码实现：动力学评估指标

为了将上述理论落实，我们将实现多维高斯分布情况下的核心评估指标，包括负对数似然（NLL）和 KL 散度计算。为了保证数值稳定性，在实践中我们通常让神经网络输出均值 $\boldsymbol{\mu}$ 以及对数方差 $\ln(\boldsymbol{\sigma}^2)$。

```python
import torch
import torch.nn as nn
import torch.distributions as D

class WorldModelEvaluator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prior_mu, prior_logvar, posterior_mu, posterior_logvar, target_z):
        """
        计算世界模型的预测评估指标
        参数：
        prior_mu, prior_logvar: 先验预测分布（模型依靠过去给出的预测）
        posterior_mu, posterior_logvar: 后验推断分布（结合当前观测给出的推断）
        target_z: 目标的隐状态采样值
        """
        # (计算确定性状态预测误差 (MSE))
        # 将对数方差转换为标准差
        prior_std = torch.exp(0.5 * prior_logvar)
        posterior_std = torch.exp(0.5 * posterior_logvar)

        # 为了直观对比，我们先计算退化情况下的均方误差
        mse_loss = nn.functional.mse_loss(prior_mu, target_z)

        # (构建先验与后验的概率分布对象)
        prior_dist = D.Normal(prior_mu, prior_std)
        posterior_dist = D.Normal(posterior_mu, posterior_std)

        # (计算概率分布的独立高斯对数似然 (NLL))
        # sum(-1)表示在特征维度上相加，mean()表示在批次维度上求平均
        nll_loss = -prior_dist.log_prob(target_z).sum(dim=-1).mean()

        # (计算动力学匹配的 KL 散度)
        # D_KL( q(z|x) || p(z|z_{t-1}, a_{t-1}) )
        kl_divergence = D.kl_divergence(posterior_dist, prior_dist).sum(dim=-1).mean()

        return mse_loss, nll_loss, kl_divergence

# 构造模拟数据
batch_size, latent_dim = 32, 64
prior_mu = torch.randn(batch_size, latent_dim)
prior_logvar = torch.zeros(batch_size, latent_dim) # 初始预测方差较大(logvar=0即var=1)

posterior_mu = prior_mu + 0.1 * torch.randn(batch_size, latent_dim) # 后验比先验更精确
posterior_logvar = -2.0 * torch.ones(batch_size, latent_dim) # 后验方差较小

target_z = posterior_mu + 0.05 * torch.randn(batch_size, latent_dim)

evaluator = WorldModelEvaluator()
mse, nll, kl = evaluator(prior_mu, prior_logvar, posterior_mu, posterior_logvar, target_z)
print(f"MSE 误差: {mse.item():.4f}")
print(f"负对数似然 (NLL): {nll.item():.4f}")
print(f"KL散度动力学惩罚: {kl.item():.4f}")
```

代码让网络输出对数方差，再通过指数映射得到正标准差，从参数化上避免了负方差。实际评测还应按预测步长、数据切分和任务类别分别汇总，不能只报告三个全局均值。

## 小结

本节从**均方误差**过渡到**负对数似然**，再讨论 ELBO 中的重构与 KL 项，以及任务导向的奖励、价值和策略诊断。它们不是可以互相替代的一串分数：点误差检查局部预测，NLL 还检查概率尺度，KL 检查先验与训练后验的匹配，而最终控制结果回答模型是否真的对任务有用。
