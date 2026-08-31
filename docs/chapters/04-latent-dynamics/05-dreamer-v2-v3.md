# 4.5 DreamerV2 与 DreamerV3：离散状态与跨任务稳健性

DreamerV1 使用连续高斯随机状态。DreamerV2 把这部分改成多组分类变量，并针对 Atari 调整了训练方法；DreamerV3 进一步处理不同任务间奖励与价值尺度差异，目标是减少逐任务调参。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/05-dreamer-v2-v3/dreamerv3-fig2.png" alt="DreamerV3 的实验环境拼图同时呈现控制、Atari、DMLab、ProcGen 与 Minecraft，直观看到“一套配置跨领域”的实际任务跨度。" width="86%">

_图 4.5-1：DreamerV3 的实验环境拼图同时呈现控制、Atari、DMLab、ProcGen 与 Minecraft，直观看到“一套配置跨领域”的实际任务跨度。 出处：Danijar Hafner；Jurgis Pasukonis；Jimmy Ba；Timothy Lillicrap，[Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)（2023），Figure 2。_

</div>

本节讨论 DreamerV2 [[Hafner et al., 2021]](https://arxiv.org/abs/2010.02193) 和 DreamerV3 [[Hafner et al., 2023]](https://arxiv.org/abs/2301.04104)。DreamerV2 使用多组分类分布组成离散随机状态，并在 Atari 上取得较强结果。DreamerV3 通过 symlog、two-hot 分布回归、回报尺度归一化等设计，提高同一套超参数跨领域使用的稳定性；原文报告覆盖 8 个领域、150 多项任务。

我们将从最基础的概率论概念起步，逐步推导出离散空间的梯度反向传播技巧，并深入理解旨在解决值域缩放问题的对数变换技巧。

## 连续空间的局限性与离散表征的崛起

<div align="center">
  <img src="/figures/04-latent-dynamics/source/05-dreamer-v2-v3/vqvae-fig2.png" alt="VQ-VAE 的原图把 ImageNet 原图与离散码本重建并排，给出分类潜变量能够保留视觉结构的早期证据。" width="86%">

_图 4.5-2：VQ-VAE 的原图把 ImageNet 原图与离散码本重建并排，给出分类潜变量能够保留视觉结构的早期证据。 出处：Aaron van den Oord；Oriol Vinyals；Koray Kavukcuoglu，[Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)（2017），Figure 2。_

</div>

DreamerV1 用高斯分布表示 RSSM 的随机状态，并通过重参数化获得路径梯度。DreamerV2 的实验发现，分组分类状态在 Atari 训练中更有效；这是经验设计选择，不意味着连续表示原则上不能描述离散概念。

分组分类变量让每组状态从有限类别中选择，并通过多组组合形成较大的表示空间。相比对角高斯，它改变了分布形状、采样方式与 KL 计算；实际收益来自这些建模与优化差异，而不是预先规定每个类别对应“猫”“狗”等人类语义。

具体来说，DreamerV2 用离散的分类分布表示随机状态。网络输出一个形状为 $G \times C$ 的矩阵，代表 $G$ 个独立的分类变量，每个变量有 $C$ 个可能类别。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/05-dreamer-v2-v3/dreamerv2-fig2.png" alt="DreamerV2 世界模型图把分类随机状态的后验、先验、奖励预测和图像重建沿时间展开。" width="86%">

_图 4.5-3：DreamerV2 世界模型图把分类随机状态的后验、先验、奖励预测和图像重建沿时间展开。 出处：Danijar Hafner；Timothy Lillicrap；Mohammad Norouzi；Jimmy Ba，[Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)（2021），Figure 2。_

</div>

### 分类分布的数学表达

假设我们有一个离散随机变量 $z$，它可以取 $C$ 个可能的状态，即 $z \in \{1, 2, \dots, C\}$。该变量服从分类分布（Categorical distribution），其概率质量函数可以由一个概率向量 $\mathbf{p} = [p_1, p_2, \dots, p_C]^\top$ 来参数化，其中：
$$p_i \ge 0, \quad \sum_{i=1}^C p_i = 1.$$

当我们对 $z$ 进行采样时，最标准的形式是生成一个“独热”（One-hot）向量。独热向量 $\mathbf{x} = [x_1, x_2, \dots, x_C]^\top$ 定义为：如果采样结果是类别 $k$，则 $x_k = 1$，其余位置 $x_i = 0\ (i \neq k)$。

直接从分类分布采样得到离散索引，普通计算图没有从该样本回到 logits 的路径导数。因此不能像高斯重参数化那样直接把下游梯度传给采样器参数。

## 解决梯度回传：直通估计器 (Straight-Through Estimator)

在微积分中，如果一个函数的导数处处为零，我们就无法通过它来回传任何有用的误差信号。这正是深度学习在面对离散变量时长期以来的痛点。

直通估计器（Straight-Through Estimator, STE）采用不同的前向与反向规则：前向使用离散样本，反向则用连续概率的梯度近似离散采样的梯度。

::: info 说明
STE 不是离散采样真实导数的无偏估计，而是一种有偏近似；它通常比基于采样回报的似然比估计方差低，但效果取决于任务与参数化。
:::

让我们用数学语言更精确地描述它。假设神经网络输出了一组未归一化的对数概率（Logits），记为 $\mathbf{l} = [l_1, l_2, \dots, l_C]^\top$。我们可以通过 Softmax 函数获得归一化的概率向量 $\mathbf{p}$：
$$\mathbf{p} = \text{Softmax}(\mathbf{l}) \quad \text{其中} \quad p_i = \frac{\exp(l_i)}{\sum_{j=1}^C \exp(l_j)}.$$

根据 $\mathbf{p}$ 采样独热向量 $\mathbf{z}_{\text{one-hot}}$ 后，可构造：
$$\tilde{\mathbf{z}} = \operatorname{sg}(\mathbf{z}_{\text{one-hot}}-\mathbf{p})+\mathbf{p},$$

如果我们在反向传播时切断（Detach） $\mathbf{z}_{\text{one-hot}} - \mathbf{p}$ 的梯度，那么：

- 在前向计算时，由于 $\mathbf{p}$ 和 $-\mathbf{p}$ 相互抵消，计算图的实际值为 $\mathbf{z}_{\text{one-hot}}$，保证了输出是严格离散的独热向量。
- 在反向计算时，梯度流经 $\tilde{\mathbf{z}}$ 时，被截断的部分不会产生梯度，所有的梯度都会流向剩下的 $\mathbf{p}$。而 $\mathbf{p}$ 是通过可导的 Softmax 函数得到的。

我们将通过代码展示这一过程。

```python
import torch
import torch.nn.functional as F
import torch.distributions as D

def straight_through_sample(logits, num_classes):
    """
    (实现直通估计器的独热采样)
    """
    # [1] 计算连续的概率分布
    probs = F.softmax(logits, dim=-1)

    # [2] 根据概率进行真实的分类分布采样
    dist = D.Categorical(probs=probs)
    indices = dist.sample()

    # [3] 将采样结果转换为独热向量
    z_one_hot = F.one_hot(indices, num_classes=num_classes).float()

    # [4] 应用直通估计器技巧 (z_one_hot - probs).detach() + probs
    # 前向传播时等于 z_one_hot，反向传播时等价于 probs
    z_sample = z_one_hot + probs - probs.detach()

    return z_sample
```

若使用 $G=32$ 组、每组 $C=32$ 个类别，组合数的理论上限是 $32^{32}$。这只说明编码空间很大，不代表模型会使用所有组合，也不代表其数量可与环境真实状态数直接比较。

## KL 散度平衡 (KL Balancing)

<div align="center">
  <img src="/figures/04-latent-dynamics/source/05-dreamer-v2-v3/dreamerv2-fig5.png" alt="DreamerV2 消融曲线把离散状态、KL balancing 和 actor 梯度选择对 Atari 表现的影响分开。" width="86%">

_图 4.5-4：DreamerV2 消融曲线把离散状态、KL balancing 和 actor 梯度选择对 Atari 表现的影响分开。 出处：Danijar Hafner；Timothy Lillicrap；Mohammad Norouzi；Jimmy Ba，[Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)（2021），Figure 5。_

</div>

在变分自编码器架构中，我们需要计算后验分布（由编码器结合当前观测得出）与先验分布（由动力学模型基于过去状态预测得出）之间的 KL 散度（Kullback-Leibler divergence）。在优化过程中，我们希望这两个分布相互靠近。

对于一般的损失函数：
$$ \mathcal{L}_{\text{KL}} = \text{KL}(q(z_t \mid \dots) \parallel p(z_t \mid \dots)) $$

在这个极小化过程中，存在两个变量：先验 $p$ 和后验 $q$。由于后验 $q$ 能够直接看到当前时刻真实的观测图像，它通常能够更快地收敛到一个合理的分布。而先验 $p$ 只能依赖历史信息进行猜测，它的训练难度更大。如果直接联合优化，强大的 $q$ 可能会为了迎合弱小的 $p$ 而降低自身提取信息的质量。

DreamerV2 使用 KL 平衡（KL Balancing）分别控制先验与后验从 KL 项接收的梯度强度：让先验更积极地拟合停止梯度后的后验，同时减弱后验仅为迁就先验而改变的趋势。

这一策略通过对 KL 散度进行切断分离（Stop-gradient，在数学中通常表示为 $\text{sg}[\cdot]$）来实现：
$$ \mathcal{L}_{\text{KL}} = \alpha \text{KL}(\text{sg}[q] \parallel p) + (1-\alpha) \text{KL}(q \parallel \text{sg}[p]) $$

当 $\alpha=0.8$ 时，第一项对先验参数的梯度权重更大，第二项对后验参数的梯度权重更小。“80% 对 20%”描述的是该 KL 梯度混合，而不是整个网络训练计算量的比例。

## DreamerV3：走向通用与健壮性

DreamerV3 保留离散世界模型，并重点处理跨任务数值尺度差异与训练稳定性。

在传统的强化学习研究中，对于不同的任务，研究人员往往需要针对性地调节学习率、奖励缩放系数以及神经网络的初始化权重。例如，在某些雅达利游戏中，得分可能是以万为单位的（如 10000 分），而在其他连续控制任务中，奖励可能被严格限制在 $[-1, 1]$ 之间。

奖励尺度增大时，价值目标也会增大，平方误差会让大残差主导梯度。DreamerV3 组合多项尺度处理与正则设计，使同一组超参数能覆盖论文评测中的多种领域。

### 对称对数变换 (Symlog Transformation)

为了压缩具有巨大范围的实数值，一个直观的想法是使用对数函数 $\log(x)$。然而，标准的对数函数存在两个问题：

1. 它未定义于负数区域，而环境的奖励完全可能是负数。
2. 当 $x$ 接近 0 时，$\log(x)$ 会趋向于负无穷，这在数值计算中是不可接受的。

为了应对这两个问题，DreamerV3 引入了对称对数（Symlog）函数。对于任意实数 $x$，它的定义如下：
$$\text{symlog}(x) = \text{sign}(x) \ln(|x| + 1)$$

这个变换包含两步：

- $|x|+1$ 使对数输入至少为 1，因此 $x=0$ 时输出为 0。
- $\text{sign}(x)$ 是符号函数。它保留了原始数值的符号。

相应的逆变换（Symexp）为：
$$\text{symexp}(y) = \text{sign}(y) (\exp(|y|) - 1)$$

symlog 保留符号，并把大幅值压缩到对数尺度。例如 $x=1{,}000{,}000$ 的 symlog 约为 $13.8$；这会减小不同任务目标尺度的差异。

### 双热编码 (Two-Hot Encoding)

DreamerV3 还把标量预测写成有限支撑上的分布回归，并用 two-hot 目标训练。

给定一个目标值 $y$（可能是经过 symlog 压缩后的价值），我们将其投影到一个预定义的等距离散桶（Bins）序列上。假设我们定义了从 $-M$ 到 $M$ 的 $B$ 个离散网格点 $v_1, v_2, \dots, v_B$。

对于任意目标值 $y$，它通常会落在相邻的两个网格点之间，例如 $v_k \le y < v_{k+1}$。我们不使用单一的独热编码，而是使用“双热”（Two-hot）编码。此时，我们将概率质量分配给相邻的这两个桶，分配比例取决于目标值距离桶的远近。

令相邻点之间距离为 $\Delta = v_{k+1} - v_k$。我们赋予右侧桶 $v_{k+1}$ 的概率为：
$$ p_{k+1} = \frac{y - v_k}{\Delta} $$

赋予左侧桶 $v_k$ 的概率为：
$$ p_k = \frac{v_{k+1} - y}{\Delta} $$

<div align="center"><img src="/figures/04-latent-dynamics/latex/05-dreamer-v2-v3/twohot-value-preservation.png" alt="目标值位于两个相邻桶之间，按到对侧桶的距离分配概率后加权桶值仍等于原目标" width="86%">

_图 4.5-5：目标越靠近某个桶，该桶获得的概率越大；两个权重之和为 1，并且桶中心的加权和严格等于原目标 y。本文根据上式绘制。_

</div>

其余桶的概率为 0。若 $y$ 位于支撑范围内，用桶位置的期望可以重构该插值值；范围外的目标会被裁剪，因此并非全局无损。对 logits 而言，Softmax 交叉熵梯度分量为“预测概率减目标概率”，位于 $[-1,1]$；这改善了输出层尺度，但不能保证整个深层网络永不出现梯度爆炸。

## 代码实现：健壮的世界模型组件

下面实现 symlog/symexp 与 two-hot 编码。它们是 DreamerV3 数值处理的一部分，不构成完整算法。

```python
class SymlogTransform:
    """
    (对称对数变换及其逆变换)
    """
    @staticmethod
    def forward(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def inverse(y):
        return torch.sign(y) * torch.expm1(torch.abs(y))

def two_hot_encode(target, min_val=-20.0, max_val=20.0, num_bins=255):
    """
    (将连续目标值转化为双热编码分布)
    """
    # [1] 构建等距的离散网格 (Bins)
    bins = torch.linspace(min_val, max_val, num_bins, device=target.device)

    # [2] 限制目标值的范围以防越界
    target = torch.clamp(target, min_val, max_val)

    # [3] 计算目标值在网格上的相对位置
    # 这里通过减去最小值并除以网格间距得到浮点索引
    step = (max_val - min_val) / (num_bins - 1)
    index_float = (target - min_val) / step

    # [4] 找到相邻的左右两个桶的整数索引
    left_idx = torch.floor(index_float).long()
    right_idx = torch.ceil(index_float).long()

    # [5] 计算分配给右侧桶的权重（距离左侧桶越远，右侧权重越大）
    weight_right = index_float - left_idx.float()
    weight_left = 1.0 - weight_right

    # 处理恰好落在网格点上的情况：全部权重放在该网格点
    mask = (left_idx == right_idx)
    weight_left[mask] = 1.0
    weight_right[mask] = 0.0

    # [6] 将权重散布到一个全零张量中形成 Two-hot 分布
    # 获取 target 的 batch 大小
    batch_shape = target.shape
    two_hot = torch.zeros(*batch_shape, num_bins, device=target.device)

    # 使用 scatter_ 填充概率值
    # 注意：需要增加一个维度以便于 scatter 操作
    two_hot.scatter_(-1, left_idx.unsqueeze(-1), weight_left.unsqueeze(-1))
    two_hot.scatter_add_(-1, right_idx.unsqueeze(-1), weight_right.unsqueeze(-1))

    return two_hot
```

## 小结

- DreamerV2 用**多组分类分布**替换连续高斯随机状态，并用 STE 提供有偏的近似梯度。
- **KL 平衡**分别调节先验与后验从 KL 项接收的梯度，避免两者以同样强度相互迁就。
- DreamerV3 用 symlog、two-hot 分布回归和归一化等组件减小跨任务尺度差异。这些设计提高了论文评测中的稳健性，但不消除模型误差或所有优化风险。
