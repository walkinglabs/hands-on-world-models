# Dreamer V2 与 V3：向离散潜在空间与健壮性进化

在前面的章节中，我们深入探讨了 DreamerV1 及其核心组件——循环状态空间模型（Recurrent State-Space Model, RSSM）。通过将连续的潜在状态与循环神经网络相结合，DreamerV1 在许多视觉控制任务上取得了令人瞩目的成就。然而，随着研究人员试图将世界模型应用于更加复杂、庞大且多样的环境（如雅达利游戏或三维导航任务）时，连续潜在空间的局限性开始逐渐显露。

在这一节中，我们将跟随世界模型的演进脉络，深入探讨由 Hafner 等人提出的 DreamerV2 [Hafner et al., 2020] 和 DreamerV3 [Hafner et al., 2023]。DreamerV2 的核心贡献在于彻底摒弃了此前占统治地位的连续高斯分布，转而采用多个分类分布（Categorical distributions）构成的离散潜在空间，从而大幅提升了模型表征复杂离散概念的能力。而随后的 DreamerV3 则进一步将目光投向了算法的通用性与健壮性，通过引入对称对数变换（Symlog transformation）等技术，首次实现了一个超参数配置能够在所有主流强化学习基准测试中表现优异的成就。

我们将从最基础的概率论概念起步，逐步推导出离散空间的梯度反向传播技巧，并深入理解旨在解决值域缩放问题的对数变换技巧。

## 连续空间的局限性与离散表征的崛起

在 DreamerV1 的 RSSM 中，我们使用多维的高斯分布（Gaussian distribution）来表示随机状态分量。高斯分布的数学性质非常优美：它完全由均值向量和协方差矩阵决定，并且利用重参数化技巧（Reparameterization trick），我们可以非常自然地让梯度穿过随机采样节点。

然而，现实世界中的许多重要概念本质上是离散的。考虑我们如何描述一个物体：它是“一只猫”或是“一条狗”；一个红绿灯是“红”、“黄”还是“绿”。当我们强行使用连续的实数空间去逼近这些界限分明的离散概念时，模型往往需要分配大量的容量来拟合那些过渡区域。此外，连续分布的尾部（Tails）可能会导致状态在长时间预测中发散。

为了克服这一问题，DreamerV2 提出使用离散的分类分布来表示状态。具体而言，它不再输出一个均值和方差，而是输出一个形状为 $G \times C$ 的矩阵，代表 $G$ 个独立的分类变量，每个分类变量有 $C$ 个可能的类别。

### 分类分布的数学表达

假设我们有一个离散随机变量 $z$，它可以取 $C$ 个可能的状态，即 $z \in \{1, 2, \dots, C\}$。该变量服从分类分布（Categorical distribution），其概率质量函数可以由一个概率向量 $\mathbf{p} = [p_1, p_2, \dots, p_C]^\top$ 来参数化，其中：
$$p_i \ge 0, \quad \sum_{i=1}^C p_i = 1.$$

当我们对 $z$ 进行采样时，最标准的形式是生成一个“独热”（One-hot）向量。独热向量 $\mathbf{x} = [x_1, x_2, \dots, x_C]^\top$ 定义为：如果采样结果是类别 $k$，则 $x_k = 1$，其余位置 $x_i = 0\ (i \neq k)$。

这种表征方式非常纯粹。但在神经网络中，直接采样独热向量会带来一个致命的问题：独热采样操作是一个阶跃函数（即在某个特定的实数阈值上发生突变），其导数在几乎所有地方都是 $0$，在突变点处则是无穷大。这意味着，传统的梯度下降算法无法计算梯度并更新采样器之前的神经网络权重。

## 解决梯度回传：直通估计器 (Straight-Through Estimator)

在微积分中，如果一个函数的导数处处为零，我们就无法通过它来回传任何有用的误差信号。这正是深度学习在面对离散变量时长期以来的痛点。

为了解决这个问题，研究人员设计了一种被称为直通估计器（Straight-Through Estimator, STE）的精妙近似方法。STE 的核心思想极为简单：(**在前向传播时，我们严格执行不可导的离散采样；而在反向传播计算梯度时，我们假装这个操作是一个恒等映射，直接将梯度放行**)。

> [!NOTE]
> 这似乎违背了严格的微积分原则，但在实际应用中，这种“欺骗”梯度的方法被证明对于离散表征学习异常有效。它提供了一种带有偏置但方差极低的梯度估计。

让我们用数学语言更精确地描述它。假设神经网络输出了一组未归一化的对数概率（Logits），记为 $\mathbf{l} = [l_1, l_2, \dots, l_C]^\top$。我们可以通过 Softmax 函数获得归一化的概率向量 $\mathbf{p}$：
$$\mathbf{p} = \text{Softmax}(\mathbf{l}) \quad \text{其中} \quad p_i = \frac{\exp(l_i)}{\sum_{j=1}^C \exp(l_j)}.$$

然后，我们根据 $\mathbf{p}$ 采样出一个独热向量 $\mathbf{z}_{\text{one-hot}}$。在 PyTorch 等框架中，STE 的实现通常利用了一个巧妙的代数恒等式。我们构造一个新的变量 $\tilde{\mathbf{z}}$：
$$\tilde{\mathbf{z}} = \mathbf{z}_{\text{one-hot}} - \mathbf{p} + \mathbf{p}.$$

如果我们在反向传播时切断（Detach） $\mathbf{z}_{\text{one-hot}} - \mathbf{p}$ 的梯度，那么：
- 在前向计算时，由于 $\mathbf{p}$ 和 $-\mathbf{p}$ 相互抵消，计算图的实际值为 $\mathbf{z}_{\text{one-hot}}$，保证了输出是严格离散的独热向量。
- 在反向计算时，梯度流经 $\tilde{\mathbf{z}}$ 时，被截断的部分不会产生梯度，所有的梯度都会流向剩下的 $\mathbf{p}$。而 $\mathbf{p}$ 是通过可导的 Softmax 函数得到的。

我们将通过代码展示这一过程。

```{.python .input}
#@tab pytorch
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

通过这一巧妙的技术，DreamerV2 得以在保持端到端可导的同时，在内部维护一个离散的隐状态表示。由多个分类分布构成的离散状态具有强大的组合泛化能力。例如，如果使用 $G=32$ 组，每组 $C=32$ 个类别的分类分布，模型可以表示多达 $32^{32}$ 种不同的状态，这远远超过了复杂游戏环境的实际状态数。

## KL 散度平衡 (KL Balancing)

在变分自编码器架构中，我们需要计算后验分布（由编码器结合当前观测得出）与先验分布（由动力学模型基于过去状态预测得出）之间的 KL 散度（Kullback-Leibler divergence）。在优化过程中，我们希望这两个分布相互靠近。

对于一般的损失函数：
$$ \mathcal{L}_{\text{KL}} = \text{KL}(q(z_t \mid \dots) \parallel p(z_t \mid \dots)) $$

在这个极小化过程中，存在两个变量：先验 $p$ 和后验 $q$。由于后验 $q$ 能够直接看到当前时刻真实的观测图像，它通常能够更快地收敛到一个合理的分布。而先验 $p$ 只能依赖历史信息进行猜测，它的训练难度更大。如果直接联合优化，强大的 $q$ 可能会为了迎合弱小的 $p$ 而降低自身提取信息的质量。

为了防止“先验拖后腿”，DreamerV2 提出了 KL 平衡（KL Balancing）技巧。其核心思路是：(**为 $q$ 向 $p$ 移动和 $p$ 向 $q$ 移动赋予不同的学习率**)。具体而言，我们希望先验 $p$ 更快地去拟合后验 $q$，而限制后验 $q$ 为了迎合先验而妥协。

这一策略通过对 KL 散度进行切断分离（Stop-gradient，在数学中通常表示为 $\text{sg}[\cdot]$）来实现：
$$ \mathcal{L}_{\text{KL}} = \alpha \text{KL}(\text{sg}[q] \parallel p) + (1-\alpha) \text{KL}(q \parallel \text{sg}[p]) $$

其中 $\alpha$ 是一个大于 0.5 的常数（通常取 0.8）。这使得网络花费 80% 的“力气”来更新动力学先验模型，而只用 20% 的“力气”去修改基于视觉特征的后验表征。

## DreamerV3：走向通用与健壮性

如果说 DreamerV2 确立了离散表征在世界模型中的主导地位，那么 DreamerV3 解决的则是另一个极其棘手的工程与数学难题：多个环境之间数值尺度的巨大差异。

在传统的强化学习研究中，对于不同的任务，研究人员往往需要针对性地调节学习率、奖励缩放系数以及神经网络的初始化权重。例如，在某些雅达利游戏中，得分可能是以万为单位的（如 10000 分），而在其他连续控制任务中，奖励可能被严格限制在 $[-1, 1]$ 之间。

当奖励的尺度变大时，由贝尔曼方程（Bellman Equation）推导出的价值估计（Value function）也会随之变得极大。此时，计算均方误差（MSE）时产生的梯度可能会导致网络权重爆炸。DreamerV3 的愿景是设计一套通用的组件，使得同一套代码和超参数可以原封不动地在跨越多种尺度的数据集上稳定运行。

### 对称对数变换 (Symlog Transformation)

为了压缩具有巨大范围的实数值，一个直观的想法是使用对数函数 $\log(x)$。然而，标准的对数函数存在两个问题：
1. 它未定义于负数区域，而环境的奖励完全可能是负数。
2. 当 $x$ 接近 0 时，$\log(x)$ 会趋向于负无穷，这在数值计算中是不可接受的。

为了应对这两个问题，DreamerV3 引入了对称对数（Symlog）函数。对于任意实数 $x$，它的定义如下：
$$\text{symlog}(x) = \text{sign}(x) \ln(|x| + 1)$$

让我们仔细拆解这个巧妙的公式：
- 绝对值加上 1（即 $|x|+1$）保证了对数函数的输入永远大于等于 1。因此，当 $x=0$ 时，$\ln(1) = 0$。这完美解决了零点附近的发散问题。
- $\text{sign}(x)$ 是符号函数。它保留了原始数值的符号。

相应的逆变换（Symexp）为：
$$\text{symexp}(y) = \text{sign}(y) (\exp(|y|) - 1)$$

通过这种变换，无论是正负数，无论其绝对值有多大，都会被压缩到一个增长极其缓慢的尺度中。例如，$x = 1,000,000$ 经过 symlog 后仅为约 $13.8$。这极大降低了价值网络和奖励预测网络的拟合难度。

### 双热编码 (Two-Hot Encoding)

仅仅压缩数值还不够。如果我们仍然使用均方误差（MSE）作为回归任务的损失函数，大尺度的异常值依然会产生占主导地位的梯度。DreamerV3 的做法是：(**将回归任务转化为分类任务**)。

给定一个目标值 $y$（可能是经过 symlog 压缩后的价值），我们将其投影到一个预定义的等距离散桶（Bins）序列上。假设我们定义了从 $-M$ 到 $M$ 的 $B$ 个离散网格点 $v_1, v_2, \dots, v_B$。

对于任意目标值 $y$，它通常会落在相邻的两个网格点之间，例如 $v_k \le y < v_{k+1}$。我们不使用单一的独热编码，而是使用“双热”（Two-hot）编码。此时，我们将概率质量分配给相邻的这两个桶，分配比例取决于目标值距离桶的远近。

令相邻点之间距离为 $\Delta = v_{k+1} - v_k$。我们赋予右侧桶 $v_{k+1}$ 的概率为：
$$ p_{k+1} = \frac{y - v_k}{\Delta} $$

赋予左侧桶 $v_k$ 的概率为：
$$ p_k = \frac{v_{k+1} - y}{\Delta} $$

其余所有桶的概率均为 0。这样，连续的实数值被无损地转化为一个离散的概率分布。随后，网络只需输出 $B$ 维的分类对数（Logits），利用交叉熵损失（Cross-Entropy Loss）进行优化。交叉熵损失的梯度最大被限制在 $\pm 1$ 之间，从而从根本上避免了任何形式的梯度爆炸。

## 代码实现：健壮的世界模型组件

(**我们将把上述理论转化为实际的代码实现**)。这段代码展示了 Symlog 变换和 Two-hot 编码的核心逻辑，它们是构建具备高度泛化能力的 DreamerV3 模型的基石。

```{.python .input}
#@tab pytorch
class SymlogTransform:
    """
    (对称对数变换及其逆变换)
    """
    @staticmethod
    def forward(x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
    
    @staticmethod
    def inverse(y):
        return torch.sign(y) * (torch.exp(torch.abs(y)) - 1.0)

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
    
    # 处理恰好落在网格点上的情况
    # 此时 left_idx == right_idx，我们平均分配权重
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

* 连续高斯潜在空间在表征强离散特征（如游戏类别、迷宫位置）时存在瓶颈。DreamerV2 创造性地通过多组分类分布（Categorical distribution）重构了 RSSM 的潜在空间。
* 直通估计器（Straight-Through Estimator, STE）巧妙地化解了离散采样不可导的难题，使得计算图既能在前向传播中保持纯粹的离散状态，又能在反向传播中提供高质量的梯度信号。
* KL 平衡（KL Balancing）解决了先验分布难以优化的问题，使得序列模型的动力学推演更加稳定。
* DreamerV3 彻底革新了数值处理机制。对称对数变换（Symlog）将各种规模的奖励和价值压缩到紧凑区间内；双热编码（Two-hot Encoding）则将回归问题优雅地转化为分类问题，彻底根除了大尺度误差带来的梯度爆炸现象，为世界模型的通用化奠定了坚实的工程与数学基础。
