# 探索下一个世界模型：前沿与未来

前面几章讨论了使用循环网络、潜变量模型、自回归模型和扩散模型表示环境动力学的方法。LeCun 的立场文章指出，像素空间包含大量难以预测、却未必与任务相关的细节，因此主张在抽象表征空间预测 [[LeCun, 2022]](https://openreview.net/forum?id=BZ5a1r-kVsf)。这篇文章为“避免预测所有像素细节”提供了理论动机，但没有对所有自回归或扩散世界模型给出统一的效率、泛化或长序列误差定理。

在本节中，我们将把目光投向地平线之外，探索有望定义“下一个时代”世界模型的前沿架构与理论基石。我们将从联合嵌入预测架构（JEPA）的能量模型视角出发，推演放弃像素级重建的数学必然性；随后，我们将引入理论神经科学中的变分自由能原则（Free Energy Principle, FEP），探讨主动推断（Active Inference）如何统一预测与决策；最后，我们将利用连续时间的状态空间模型（State Space Models, SSMs）突破离散时间步长的桎梏。这些前沿探索不仅是工程架构的革新，更是对“智能系统如何认知物理世界”这一哲学命题的严密数学重构。

## 联合嵌入预测架构（JEPA）的数学必然性

现有的多数世界模型（如基于 Transformer 或 VAE 的架构）通常致力于在观察空间（如图像像素）中预测下一个状态。假设当前状态为 $\mathbf{x}_t$，动作为 $\mathbf{a}_t$，预测目标往往是 $\mathbf{x}_{t+1}$。然而，物理世界充满了不可预测的细微噪音（例如风中摇曳的树叶像素变化）。强迫模型耗费巨大的模型容量去拟合这些对高级决策毫无意义的高频细节，是不理智的。

Yann LeCun 在其标志性的论文 [[LeCun, 2022]](https://openreview.net/forum?id=BZ5a1r-kVsf) 中正式提出了联合嵌入预测架构（Joint-Embedding Predictive Architecture, JEPA）。其核心思想是：抛弃在原始观测空间中的生成过程，转而在一个高度抽象、去除了不可预测噪声的隐变量空间（Latent Space）中进行预测。

### 从自编码器到能量模型

要理解 JEPA，我们首先回顾基础的自编码器重建逻辑。设观测值 $\mathbf{x} \in \mathbb{R}^N$，编码器 $E_\theta$ 将其映射为隐变量 $\mathbf{z} \in \mathbb{R}^d$（其中 $d \ll N$），解码器 $D_\phi$ 重建观测。这种范式的损失函数通常是均方误差（MSE）：

$$L_{\text{recon}} = \|\mathbf{x} - D_\phi(E_\theta(\mathbf{x}))\|_2^2$$

在预测任务中，引入动作变量 $\mathbf{a}$，我们得到传统的隐变量预测模型：

$$L_{\text{pred}} = \|\mathbf{x}_{t+1} - D_\phi(P_\psi(E_\theta(\mathbf{x}_t), \mathbf{a}_t))\|_2^2$$

这种做法依然依赖于 $D_\phi$ 将低维向量强行映射回高维的 $N$ 维像素空间。JEPA 则完全移除了生成组件 $D_\phi$。给定初始状态 $\mathbf{x}$ 和目标状态 $\mathbf{y}$，分别计算它们的表征 $\mathbf{s}_x = E_\theta(\mathbf{x})$ 和 $\mathbf{s}_y = E_\theta(\mathbf{y})$。预测器 $P_\psi$ 仅在表征空间内工作：

$$\hat{\mathbf{s}}_y = P_\psi(\mathbf{s}_x, \mathbf{a})$$

此时的损失函数直接定义在隐空间之上。我们可以将其视为一种能量函数（Energy Function） $F(\mathbf{x}, \mathbf{y}, \mathbf{a})$，衡量输入变量配置的不兼容程度：

$$F(\mathbf{x}, \mathbf{y}, \mathbf{a}) = \|\hat{\mathbf{s}}_y - \mathbf{s}_y\|_2^2 = \|P_\psi(E_\theta(\mathbf{x}), \mathbf{a}) - E_\theta(\mathbf{y})\|_2^2$$

### 表征坍塌与 VICReg 正则化

直接最小化该公式会遇到一个灾难性的平凡解（Trivial Solution）：**表征坍塌（Representation Collapse）**。如果 $E_\theta$ 将所有输入无论如何都映射为零向量（$\mathbf{s}_x = \mathbf{s}_y = \mathbf{0}$），且 $P_\psi$ 也总是输出零向量，那么能量函数 $F=0$。这种情况下，网络根本没有学习到物理世界的任何动态规律。

为了防止坍塌，我们必须对能量模型施加正则化。这里我们引入方差-协方差正则化（VICReg）方法 [[Bardes et al., 2021]](https://arxiv.org/abs/2105.04906)，其核心思想利用了基础的统计学原理，我们可以用一维方差与多维向量协方差的概念来进行严密的约束。

对于隐空间中的一个批次（Batch）样本，设特征矩阵为 $\mathbf{S} \in \mathbb{R}^{B \times d}$，其中 $B$ 为批次大小，$d$ 为表征维度。对于第 $j$ 个特征维度（列向量 $\mathbf{s}_{*, j}$），我们希望它在不同样本间具有区分度，即方差不能太小。方差损失项定义为：

$$v(\mathbf{S}) = \frac{1}{d} \sum_{j=1}^d \max\left(0, \gamma - \sqrt{\text{Var}(\mathbf{s}_{*, j}) + \epsilon}\right)$$

其中 $\gamma$ 是目标标准差，我们强迫每个特征维度的标准差至少为 $\gamma$。

同时，我们希望这 $d$ 个特征维度彼此独立，不要编码冗余的信息。这在几何上等价于要求特征的协方差矩阵除了对角线之外的元素尽可能为零。去均值化后的特征矩阵 $\mathbf{S}'$ 的协方差矩阵为 $\mathbf{C} = \frac{1}{B-1} (\mathbf{S}')^\top \mathbf{S}'$。协方差损失定义为非对角线元素的平方和：

$$c(\mathbf{S}) = \frac{1}{d} \sum_{i \neq j} C_{i, j}^2$$

通过组合不变性（预测与目标的距离）、方差（保持信息量）和协方差（去相关性），JEPA 实现了一个无需在像素层面生成、且具有严格数学保障的抽象世界模型。这种架构为处理高维复杂物理系统铺平了道路。

## 主动推断与变分自由能

仅仅能够预测未来还不够，世界模型最终需要服务决策。Friston 的自由能原则从感知与行动共同最小化变分自由能的角度给出了一种理论视角 [[Friston, 2010]](https://doi.org/10.1038/nrn2787)。它与标准强化学习的奖励最大化并不天然等价；若要把两者统一，还需要明确生成模型、偏好分布和行动推断等额外假设。

### 惊讶与信息熵

从热力学和统计力学的视角来看，生物体或智能系统的首要目标是维持自身结构的稳定，避免陷入高熵的热力学平衡态。在信息论中，这等价于最小化智能体所观测到的环境状态的“惊讶度”（Surprisal）。

设智能体的生成模型（世界模型）为联合概率分布 $P(\mathbf{o}, \mathbf{s})$，其中 $\mathbf{o}$ 是观测变量，$\mathbf{s}$ 是隐状态。由于物理环境 $\mathbf{s}$ 是不可直接观测的，智能体只能推断其后验分布 $P(\mathbf{s} \mid \mathbf{o})$。惊讶度定义为边缘似然的负对数：

$$\mathcal{I}(\mathbf{o}) = -\log P(\mathbf{o}) = -\log \int P(\mathbf{o}, \mathbf{s}) d\mathbf{s}$$

在一般情况下，上述积分对于高维连续状态空间是无法解析求解的。因此，我们引入一个由神经网络参数化的近似后验分布 $Q(\mathbf{s} \mid \mathbf{o})$。

> **[唯一的直觉类比]**
> 在这里，我们可以将环境视为一个极其复杂且封闭的黑箱房间。智能体在房间内只能通过小孔观察光影（观测 $\mathbf{o}$）。试图直接猜透房间内所有物体的绝对真理（计算 $P(\mathbf{o})$）如同徒手解开无尽的结；但智能体可以在自己脑中建立一个粗糙但够用的内部模型（变分分布 $Q$）。智能体的目标是在内部模型与现实观测之间建立共振，使得两者间的能量差（变分自由能）降至最低，从而在这个充满混沌的房间内存活下来。

### 变分自由能的严格推导

我们对惊讶度进行如下恒等变形，结合对数函数的凹性，利用期望的线性性质和 Jensen 不等式：

$$
\begin{aligned}
-\log P(\mathbf{o}) &= -\log \int P(\mathbf{o}, \mathbf{s}) \frac{Q(\mathbf{s} \mid \mathbf{o})}{Q(\mathbf{s} \mid \mathbf{o})} d\mathbf{s} \\
&= -\log \mathbb{E}_{Q} \left[ \frac{P(\mathbf{o}, \mathbf{s})}{Q(\mathbf{s} \mid \mathbf{o})} \right] \\
&\le \mathbb{E}_{Q} \left[ -\log \frac{P(\mathbf{o}, \mathbf{s})}{Q(\mathbf{s} \mid \mathbf{o})} \right] \quad \text{(根据 Jensen 不等式)}
\end{aligned}
$$

不等式右侧的量即为**变分自由能（Variational Free Energy, VFE）** $\mathcal{F}$。由于 $\mathcal{F}$ 是惊讶度的上界（Upper Bound），最小化自由能即可隐式地最小化惊讶度（避免系统陷入意外的高熵态）。

我们可以将自由能 $\mathcal{F}$ 重新整理为两种极具物理意义的形式。第一种形式（复杂性与准确性）：

$$
\begin{aligned}
\mathcal{F} &= \mathbb{E}_{Q} [-\log P(\mathbf{o} \mid \mathbf{s})] + \mathbb{E}_{Q} \left[ \log \frac{Q(\mathbf{s} \mid \mathbf{o})}{P(\mathbf{s})} \right] \\
&= \underbrace{-\mathbb{E}_{Q}[\log P(\mathbf{o} \mid \mathbf{s})]}_{\text{预期惊讶 (不准确性)}} + \underbrace{D_{KL}(Q(\mathbf{s} \mid \mathbf{o}) \| P(\mathbf{s}))}_{\text{复杂性惩罚}}
\end{aligned}
$$

该公式表明，一个优秀的世界模型既需要能够准确解释观测结果（最小化第一项，即重构误差），又必须保持自身的简约性（最小化第二项，使推断的后验分布不要偏离先验分布太多）。

### 统一认知与行动

主动推断最迷人的地方在于，它认为智能系统可以通过两种方式最小化自由能：

1. **感知（Perception）**：改变内部信念 $Q(\mathbf{s} \mid \mathbf{o})$ 以更好地拟合环境（即常规的模型训练）。
2. **行动（Action）**：通过执行动作 $a$ 改变外部物理世界，产生新的观测 $\mathbf{o}$，使得新观测符合模型的先验预期（即目标导向的决策）。

在主动推断框架中，策略的选择完全转化为对**期望自由能（Expected Free Energy, EFE）**的最小化。这种统一，消除了强化学习中外在奖励信号的绝对必要性，使得探索（降低模型不确定性）和利用（实现预期目标）被同等地编码在同一个公式中。这是迈向“通用世界模型”的重要理论基石。

## 连续时间与状态空间模型（SSMs）

我们目前探讨的世界模型大多建立在离散的时间步 $t, t+1, t+2$ 之上。然而，真实的物理世界是连续流动的。对连续动态的离散化往往会导致信息丢失，并且在跨越多时间尺度的长序列预测时，循环网络或 Transformer 面临着灾难性的遗忘或平方级的计算复杂度。

近年来，S4 等结构化状态空间序列模型从连续时间状态空间系统出发，设计了可高效计算的长序列层 [[Gu et al., 2021]](https://arxiv.org/abs/2111.00396)。这里引用的是 S4 论文，不是后来提出的 Mamba；这类模型为长序列建模提供了工具，但其对具体世界模型是否有效仍需实验验证。

### 线性状态空间动态系统

考虑一个连续时间的一维输入信号 $u(t)$ 和输出信号 $y(t)$，其内部隐藏状态为高维向量 $\mathbf{x}(t) \in \mathbb{R}^N$。连续状态空间模型可以由经典的线性微分方程表示：

$$
\begin{aligned}
\dot{\mathbf{x}}(t) &= \mathbf{A}\mathbf{x}(t) + \mathbf{B}u(t) \\
y(t) &= \mathbf{C}\mathbf{x}(t) + \mathbf{D}u(t)
\end{aligned}
$$

其中，矩阵 $\mathbf{A} \in \mathbb{R}^{N \times N}$ 编码了系统的演化动态，$\mathbf{B} \in \mathbb{R}^{N \times 1}$ 定义了输入如何驱动状态演变，$\mathbf{C} \in \mathbb{R}^{1 \times N}$ 定义了状态如何映射为输出。由于我们关注时间动态，通常省略直通项令 $\mathbf{D} = 0$。

对于具备基础微积分知识的读者而言，一阶线性常微分方程 $\frac{dx}{dt} = ax$ 的解析解是 $x(t) = e^{at}x(0)$。同理，上述矩阵微分方程的连续解深刻依赖于矩阵指数 $e^{\mathbf{A}t}$。

### 精确离散化（Zero-Order Hold）

尽管物理世界是连续的，但在数字计算机上处理输入信号序列 $(u_0, u_1, \ldots)$，我们必须以采样间隔 $\Delta$ 进行离散化。SSMs 采用零阶保持器（Zero-Order Hold, ZOH）假设，即在区间 $[k\Delta, (k+1)\Delta)$ 内，输入信号保持恒定 $u(t) = u_k$。

通过对连续系统方程在时间区间内进行精确积分，我们可以推导出严密的离散化递推方程：

$$
\mathbf{x}_k = \mathbf{x}(k\Delta) = e^{\mathbf{A}\Delta}\mathbf{x}_{k-1} + \left( \int_{0}^{\Delta} e^{\mathbf{A}\tau} d\tau \right) \mathbf{B}u_k
$$

令离散化的参数矩阵为 $\bar{\mathbf{A}} = e^{\mathbf{A}\Delta}$，以及 $\bar{\mathbf{B}} = (\mathbf{A}^{-1}(e^{\mathbf{A}\Delta} - \mathbf{I}))\mathbf{B}$，我们得到了类似于 RNN 的严格离散递归表示：

$$
\begin{aligned}
\mathbf{x}_k &= \bar{\mathbf{A}}\mathbf{x}_{k-1} + \bar{\mathbf{B}}u_k \\
y_k &= \mathbf{C}\mathbf{x}_k
\end{aligned}
$$

与普通 RNN 使用启发式的非线性门控机制（如 LSTM）不同，该公式具有极其优美的解析性质。结合特定的矩阵 $\mathbf{A}$ 构造（如 HiPPO 矩阵，旨在正交多项式基上记忆完整的历史记录），SSMs 在进行物理动态的时序预测时，展现出极其优秀的数学收敛性和无需注意力机制的线性时间复杂度。这使得 SSMs 成为模拟长周期复杂物理动态现象的绝佳候选架构。

## 代码实现：构建下一代世界模型的核心模块

(**为了将理论落地，我们展示一个简化但严密的联合嵌入预测架构（JEPA）核心模块实现。**) 我们将通过 PyTorch 实现包含了特征提取、前向预测以及 VICReg 正则化损失计算的模型雏形。

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    """联合嵌入预测架构的预测器网络"""
    def __init__(self, embed_dim, action_dim, hidden_dim):
        super().__init__()
        # 接收当前状态表征和动作序列，映射为下一状态的表征
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, state_repr, action):
        # 拼接状态表征与动作向量
        x = torch.cat([state_repr, action], dim=-1)
        return self.net(x)

def vicreg_loss(x, y, sim_weight=25.0, var_weight=25.0, cov_weight=1.0, epsilon=1e-4):
    """
    计算基于 VICReg 的正则化损失。
    x: 预测器的输出 (Batch_size, embed_dim)
    y: 目标状态的真实表征 (Batch_size, embed_dim)
    """
    batch_size, embed_dim = x.shape

    # 1. 不变性损失 (Invariance Loss)：等价于我们在公式中提到的 MSE 距离
    sim_loss = F.mse_loss(x, y)

    # 2. 方差损失 (Variance Loss)：防止表征坍塌到单个点
    std_x = torch.sqrt(x.var(dim=0) + epsilon)
    std_y = torch.sqrt(y.var(dim=0) + epsilon)
    # 使用 ReLU 确保方差至少达到阈值 (通常设为 1.0)
    std_loss = torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2

    # 3. 协方差损失 (Covariance Loss)：对特征进行去相关，强制不同维度编码独立信息
    x_mean = x - x.mean(dim=0)
    y_mean = y - y.mean(dim=0)
    cov_x = (x_mean.T @ x_mean) / (batch_size - 1)
    cov_y = (y_mean.T @ y_mean) / (batch_size - 1)

    # 提取非对角线元素的平方和
    off_diag_mask = ~torch.eye(embed_dim, dtype=torch.bool, device=x.device)
    cov_loss = (cov_x[off_diag_mask].pow(2).sum() / embed_dim +
                cov_y[off_diag_mask].pow(2).sum() / embed_dim)

    # 综合损失
    loss = sim_weight * sim_loss + var_weight * std_loss + cov_weight * cov_loss
    return loss, sim_loss, std_loss, cov_loss

# 模拟环境交互
batch_size = 64
embed_dim = 128
action_dim = 16

# 假设编码器 E_theta 已经从观测输出了初始表征和目标表征
s_x = torch.randn(batch_size, embed_dim) # 初始状态表征
s_y = torch.randn(batch_size, embed_dim) # 目标状态表征 (无梯度，作为目标)
a = torch.randn(batch_size, action_dim)  # 当前采取的动作

predictor = Predictor(embed_dim, action_dim, hidden_dim=256)
s_y_hat = predictor(s_x, a)

total_loss, sim, var, cov = vicreg_loss(s_y_hat, s_y)
print(f"Total Loss: {total_loss.item():.4f}")
print(f"Similarity Loss: {sim.item():.4f}, Variance Loss: {var.item():.4f}, Covariance Loss: {cov.item():.4f}")
```

在这个实现中，通过 `vicreg_loss` 函数，我们无需繁重的像素解码器网络，在数学层面优雅地保障了网络在隐变量空间学习到了无偏且多元的动态规律。

## 小结

下一代世界模型的探索已经越过了单纯增加网络参数或数据的阶段，转入对智能体认知物理世界底层逻辑的反思：

- **JEPA** 证明了对于高熵现实世界，我们必须在抽象的隐变量空间进行预测，并通过方差与协方差正则化等手段对抗信息坍塌。
- **主动推断与自由能原则** 将控制论与生成模型统一，使模型不再满足于单纯的未来猜测，而是主动通过行动降低环境的不可预测性。
- **连续状态空间模型** 利用微分方程重新审视序列建模，为物理规律中的长程、多尺度依赖提供了理论上完美的框架。
