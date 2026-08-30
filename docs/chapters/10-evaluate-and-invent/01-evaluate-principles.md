# 世界模型的评估原则与核心指标

在深度学习的发展历程中，评估（Evaluation）始终是引导模型演进的灯塔。对于监督学习而言，分类准确率或均方误差是直观且明确的评估指标；对于生成模型，我们有基于人类视觉感知的感知度量（如FID分数）。然而，当我们涉足“世界模型”（World Models）这一领域时，如何评估一个模型是否真正“理解”了世界的运作规律，便成了一个极具挑战性的开放问题。

在本节中，我们将深入探讨世界模型的评估原则。我们将从早期模型架构的历史背景出发，严格推导从确定性预测到概率性推断的核心数学指标，并最终落脚于代码实现。

## 历史背景与学术脉络

在智能体（Agent）研究的早期阶段，研究者们主要依赖无模型（Model-Free）的强化学习算法。这类算法通过试错直接学习策略，但其样本效率极低。为了让智能体能够像人类一样在脑海中“预演”未来，基于模型（Model-Based）的方法逐渐兴起。

现代世界模型的奠基之作之一是由 Ha 和 Schmidhuber 在 2018 年提出的 *World Models* [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。他们提出，智能体可以通过一个变分自编码器（VAE）压缩视觉输入，并利用循环神经网络（RNN）在隐空间中预测未来。此时，评估模型好坏的主要标准是**像素级的重构误差（Pixel-level Reconstruction Error）**。

然而，随着研究的深入，人们发现像素级重构存在致命缺陷。现实世界充满了无关紧要的噪声（例如微风吹动树叶的随机摆动）。如果模型将庞大的计算资源用于完美重构这些随机噪声，它将无法专注于真正决定世界演化规律的核心特征。

为了解决这一问题，[[Hafner et al., 2019]](https://arxiv.org/abs/1912.01603) 提出了 Dreamer 系列模型，引入了循环状态空间模型（RSSM），将评估的重心从“完美的像素重构”转移到了“隐空间中的状态一致性”以及“奖励预测的准确性”。随后，[[LeCun, 2022]](https://openreview.net/forum?id=BZ5a1r-kVsf) 在其关于自主机器智能的立场论文中提出了联合嵌入预测架构（JEPA），进一步强调完全摒弃像素级重构，仅在抽象的隐空间中评估模型的预测能力。

这些历史沿革揭示了世界模型评估的两个核心维度：**动力学预测的准确性（Predictive Accuracy）**与**下游任务的效用价值（Behavioral Utility）**。

## 动力学预测的准确性：从标量到矩阵

世界模型的核心任务是预测未来。为了从最基础的概念起步，我们暂且抛开高维图像和复杂的神经网络，回到高中物理中最经典的运动学场景。

### 确定性系统中的预测误差

假设我们正在观测一个在真空中做平抛运动的小球。如果我们知道小球在 $t$ 时刻的水平位置 $x_t$ 和速度 $v_t$，根据牛顿运动定律，我们可以绝对确定地预测它在 $t+1$ 时刻的位置 $x_{t+1}$。

假设我们的世界模型（在这里是一个简单的物理公式）给出了预测值 $\hat{x}_{t+1}$。此时，评估模型预测好坏的最自然方式，就是计算预测位置与真实观测位置之间的距离。这在数学上体现为绝对误差或平方误差。为了便于后续求导优化，我们通常选择平方误差：

$$ e_{t+1} = (x_{t+1} - \hat{x}_{t+1})^2 $$

该公式描述的是单一物理量的预测误差。在真实的世界模型中，状态通常不是一个单一的标量，而是一个包含了众多属性（如位置、速度、姿态、环境特征等）的高维向量 $\mathbf{z}_{t+1} \in \mathbb{R}^d$。

此时，我们需要将一维的误差公式严谨地推广到高维向量空间。对于两个向量 $\mathbf{z}_{t+1}$ 和 $\mathbf{\hat{z}}_{t+1}$，它们之间的距离可以通过欧几里得距离（L2范数）的平方来衡量，这便是多维隐空间中的均方误差（Mean Squared Error, MSE）：

$$ \mathcal{L}_{\text{MSE}} = \frac{1}{d} \sum_{i=1}^{d} (z_{t+1}^{(i)} - \hat{z}_{t+1}^{(i)})^2 = \frac{1}{d} \|\mathbf{z}_{t+1} - \mathbf{\hat{z}}_{t+1}\|_2^2 $$

在这里，$z_{t+1}^{(i)}$ 表示真实状态向量的第 $i$ 个维度分量，$\|\cdot\|_2$ 表示向量的 L2 范数。这个确定性的评估指标极其直观，但在现实世界中却面临着严峻的挑战。

### 随机性与概率分布的引入

现实世界并非真空中的平抛运动。假设小球在飞行过程中会受到不可预测的随机阵风影响。此时，即便给定初始状态，小球在下一时刻的确切位置也是不确定的。如果我们的模型仍然只输出一个确定性的点预测 $\mathbf{\hat{z}}_{t+1}$，它将不可避免地产生巨大的误差，因为无论它预测哪个具体的点，阵风都可能让小球偏离。

为了严谨地描述这种随机性，世界模型的输出不再是一个确定的点，而必须是一个**概率分布（Probability Distribution）**。

在一维场景下，假设模型预测小球下一时刻的位置服从一个均值为 $\hat{\mu}_{t+1}$、方差为 $\hat{\sigma}_{t+1}^2$ 的高斯分布。根据高斯分布的概率密度函数，真实观测值 $x_{t+1}$ 出现在该分布中的概率密度（似然）为：

$$ p(x_{t+1} | \hat{\mu}_{t+1}, \hat{\sigma}_{t+1}^2) = \frac{1}{\sqrt{2\pi\hat{\sigma}_{t+1}^2}} \exp\left(-\frac{(x_{t+1} - \hat{\mu}_{t+1})^2}{2\hat{\sigma}_{t+1}^2}\right) $$

一个好的世界模型，应当使其预测的概率分布在真实观测值处具有最大的概率密度。这就是**最大似然估计（Maximum Likelihood Estimation, MLE）**的核心思想。为了将连乘转化为求和并避免数值下溢，我们通常对似然函数取对数，并加上负号，得到负对数似然（Negative Log-Likelihood, NLL）：

$$ \text{NLL} = - \ln p(x_{t+1}) = \frac{1}{2} \ln(2\pi\hat{\sigma}_{t+1}^2) + \frac{(x_{t+1} - \hat{\mu}_{t+1})^2}{2\hat{\sigma}_{t+1}^2} $$

仔细观察该公式的第二项。如果模型的方差 $\hat{\sigma}_{t+1}^2$ 被固定为一个常数，那么最小化 NLL 就严格等价于最小化均方误差。这证明了：**均方误差本质上是假设预测分布为等方差高斯分布时的特殊最大似然估计**。通过引入可学习的方差 $\hat{\sigma}_{t+1}^2$，模型学会了表达“不确定性”——当环境随机性大时，模型会输出更大的方差，从而使第一项增大，但避免了因点预测错误导致的第二项剧烈惩罚。

将其严谨地推广到高维向量空间。假设模型预测的高维状态服从多变量高斯分布 $\mathcal{N}(\boldsymbol{\hat{\mu}}_{t+1}, \boldsymbol{\hat{\Sigma}}_{t+1})$，其中 $\boldsymbol{\hat{\Sigma}}_{t+1}$ 为协方差矩阵。高维分布的负对数似然形式为：

$$ \mathcal{L}_{\text{NLL}} = \frac{d}{2}\ln(2\pi) + \frac{1}{2}\ln|\boldsymbol{\hat{\Sigma}}_{t+1}| + \frac{1}{2}(\mathbf{z}_{t+1} - \boldsymbol{\hat{\mu}}_{t+1})^\top \boldsymbol{\hat{\Sigma}}_{t+1}^{-1} (\mathbf{z}_{t+1} - \boldsymbol{\hat{\mu}}_{t+1}) $$

在实际的世界模型实现中，为了降低计算复杂度，通常假设各个维度相互独立，即协方差矩阵为对角阵 $\boldsymbol{\hat{\Sigma}} = \text{diag}(\hat{\sigma}_1^2, \dots, \hat{\sigma}_d^2)$。此时，高维的似然评估退化为各个维度一维似然评估的累加。

## 潜变量推断与变分下界 (ELBO)

现代世界模型（如 RSSM）在评估时面临一个更加复杂的挑战：隐状态 $\mathbf{z}_t$ 是不可见的（Latent）。我们能观测到的只有图像序列 $\mathbf{x}_{1:T}$ 和动作序列 $\mathbf{a}_{1:T}$。

这就要求模型同时具备两个能力：
1. **后验推断（Posterior Inference）**：根据当前的真实观测图像 $\mathbf{x}_t$ 提取出真实的隐状态分布 $q(\mathbf{z}_t | \mathbf{x}_t, \dots)$。
2. **先验预测（Prior Prediction）**：在不看当前图像的情况下，仅根据历史信息 $\mathbf{z}_{t-1}$ 和动作 $\mathbf{a}_{t-1}$，预测出当前的隐状态分布 $p(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{a}_{t-1})$。

评估这样的模型，我们采用了变分推断（Variational Inference）框架。我们要最大化观测数据 $\mathbf{x}_{1:T}$ 的边缘似然的下界，即证据下界（Evidence Lower Bound, ELBO）。对于单一时间步，ELBO 可以拆解为以下严格的数学形式：

$$ \mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(\mathbf{z}_t | \mathbf{x}_t)}\big[\ln p(\mathbf{x}_t | \mathbf{z}_t)\big] - \beta D_{\text{KL}}\Big( q(\mathbf{z}_t | \mathbf{x}_t) \,\Big\|\, p(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{a}_{t-1}) \Big) $$

让我们温柔地拆解这个极其重要的高阶公式。它包含了两项截然不同的评估指标：

第一项 $\mathbb{E}_{q}[\ln p(\mathbf{x}_t | \mathbf{z}_t)]$ 是**重构似然（Reconstruction Likelihood）**。它要求基于观测图像提取出的后验隐状态，能够被解码器成功还原回原始图像。这衡量了隐状态是否保留了足够的视觉细节信息。

第二项 $D_{\text{KL}}$ 则是评估世界模型动力学预测能力的核心。KL散度（Kullback-Leibler Divergence）衡量了两个概率分布之间的差异。在该公式中，它迫使**先验预测分布** $p(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{a}_{t-1})$ 必须尽可能地逼近**后验推断分布** $q(\mathbf{z}_t | \mathbf{x}_t)$。

> 💡 **KL散度动力学匹配机制**
> 在这里，我们可以将后验推断网络比作一位“拥有视觉的引导者”，它能够直接看到当前真实发生的画面 $\mathbf{x}_t$，从而精确判断当前所处的状态。而先验预测网络（即动力学模型）则是一位“被蒙上眼睛的预测者”，它只能依靠对过去 $\mathbf{z}_{t-1}$ 的记忆和执行的动作 $\mathbf{a}_{t-1}$，试图在脑海中描绘出此刻的状态。KL 散度所计算的，正是这两位引导者脑海中状态分布的不一致程度。在训练过程中，我们通过最小化这个 KL 散度，强迫蒙眼的预测者不断修正自己的物理直觉，直到其盲猜的分布与拥有视觉者的判断完全吻合。这便是世界模型“理解”动力学的数学本质。

## 效用导向的评估 (Behavioral Utility)

尽管基于潜变量动力学的 KL 散度评估极具数学优雅性，但世界模型的最终目的往往不是为了“精确预测”，而是为了“辅助决策”。这就引出了另一类重要的评估原则：基于价值等效（Value Equivalence）的评估。

在诸如 MuZero [[Schrittwieser et al., 2020]](https://arxiv.org/abs/1911.08265) 的架构中，并不直接去预测环境的具体特征或图像。模型的好坏完全取决于其对隐空间进行展开后，能否准确预测出对决策有用的标量信号：奖励 $r$ 和价值 $v$。

具体的评估指标退化为联合的损失函数，直接在模型的展开轨迹上评估：

$$ \mathcal{L}_{\text{Utility}} = \sum_{k=0}^{K} \Big[ l^r(r_{t+k}, \hat{r}_t^k) + l^v(v_{t+k}, \hat{v}_t^k) + l^p(\pi_{t+k}, \hat{\mathbf{p}}_t^k) \Big] $$

其中，$\hat{r}_t^k$, $\hat{v}_t^k$ 和 $\hat{\mathbf{p}}_t^k$ 是世界模型从状态 $\mathbf{z}_t$ 出发，在脑海中向前推演 $k$ 步后预测出的奖励、状态价值和策略向量。这里的 $l^r$ 和 $l^v$ 往往采用普通的均方误差（或变体），而 $l^p$ 则多采用交叉熵。这种评估原则彻底摒弃了对环境细节的无谓纠缠，将评估的利刃直指任务的核心。

## 代码实现：动力学评估指标

为了将上述理论落实，我们将实现多维高斯分布情况下的核心评估指标，包括负对数似然（NLL）和 KL 散度计算。为了保证数值稳定性，在实践中我们通常让神经网络输出均值 $\boldsymbol{\mu}$ 以及对数方差 $\ln(\boldsymbol{\sigma}^2)$。

```{.python .input}
#@tab pytorch
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

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class WorldModelEvaluator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    def call(self, prior_mu, prior_logvar, posterior_mu, posterior_logvar, target_z):
        # (计算确定性状态预测误差 (MSE))
        prior_std = tf.exp(0.5 * prior_logvar)
        posterior_std = tf.exp(0.5 * posterior_logvar)
        
        mse_loss = tf.reduce_mean(tf.keras.losses.MSE(target_z, prior_mu))
        
        # (构建先验与后验的概率分布对象)
        prior_dist = tfd.Normal(loc=prior_mu, scale=prior_std)
        posterior_dist = tfd.Normal(loc=posterior_mu, scale=posterior_std)
        
        # (计算概率分布的独立高斯对数似然 (NLL))
        nll_loss = -tf.reduce_mean(tf.reduce_sum(prior_dist.log_prob(target_z), axis=-1))
        
        # (计算动力学匹配的 KL 散度)
        kl_divergence = tf.reduce_mean(tf.reduce_sum(tfd.kl_divergence(posterior_dist, prior_dist), axis=-1))
        
        return mse_loss, nll_loss, kl_divergence

# 构造模拟数据
batch_size, latent_dim = 32, 64
prior_mu = tf.random.normal((batch_size, latent_dim))
prior_logvar = tf.zeros((batch_size, latent_dim))

posterior_mu = prior_mu + 0.1 * tf.random.normal((batch_size, latent_dim))
posterior_logvar = -2.0 * tf.ones((batch_size, latent_dim))

target_z = posterior_mu + 0.05 * tf.random.normal((batch_size, latent_dim))

evaluator = WorldModelEvaluator()
mse, nll, kl = evaluator(prior_mu, prior_logvar, posterior_mu, posterior_logvar, target_z)
print(f"MSE 误差: {mse.numpy():.4f}")
print(f"负对数似然 (NLL): {nll.numpy():.4f}")
print(f"KL散度动力学惩罚: {kl.numpy():.4f}")
```

在这个实现中，我们可以清晰地看到数学公式是如何一步步转化为张量运算的。尤其是对数方差的处理，在深度学习中这是一种极其重要且广泛使用的技巧，它避免了神经网络在直接输出方差时可能产生的负数问题。

## 小结

在本节中，我们严格梳理了世界模型的评估原则。我们从确定性系统的均方误差起步，通过引入不可避免的环境随机性，自然地过渡到了概率性推断的负对数似然。针对隐空间中的动态推演，我们拆解了变分下界中的 KL 散度匹配机制，并讨论了任务导向的价值等效评估原则。这些指标共同构成了评估现代世界模型性能的完整基石。
