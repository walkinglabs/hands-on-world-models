# 掩码预测与表征坍塌（Representation Collapse）问题

自监督学习（Self-Supervised Learning, SSL）利用数据自身构造训练目标。例如，BERT 根据未遮挡的上下文预测被遮挡的词元 [[Devlin et al., 2018]](https://arxiv.org/abs/1810.04805)，掩码自编码器（Masked Autoencoders, MAE）则重构被遮挡图像块的像素 [[He et al., 2022]](https://arxiv.org/abs/2111.06377)。这两篇论文分别支撑词元空间与像素空间中的掩码预测。

像素重构要求模型解释颜色、纹理与局部噪声；其中一些细节对特定下游任务可能并不重要。表征预测试图把训练容量更多用于可由上下文推断的结构，但“哪些细节无关”取决于任务，不能预先一概删去。

为了减少对像素解码的依赖，data2vec 使用教师网络产生的潜在表征作为预测目标 [[Baevski et al., 2022]](https://arxiv.org/abs/2202.03555)，I-JEPA 则在图像块的表征空间中进行掩码预测 [[Assran et al., 2023]](https://arxiv.org/abs/2301.08243)。data2vec 与 I-JEPA 属于相关的潜在目标预测方法，但前者并未以 JEPA 命名。只预测动态生成的表征时，需要专门设计训练机制来避免**表征坍塌（Representation Collapse）**。

<div align="center">
  <img src="/figures/06-jepa/source/02-mask-collapse/ijepa-fig4.png" alt="I-JEPA 的上下文块与多目标块实例显示，掩码预测实际作用于哪些图像区域。" width="86%">

_图 6.2-1：I-JEPA 的上下文块与多目标块实例显示，掩码预测实际作用于哪些图像区域。 出处：Mahmoud Assran et al.，[Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)（2023），Figure 4。_

</div>

本节从代数与几何视角分析常数表示为何是一个平凡解，并讨论 BYOL 等方法使用的非对称架构、停止梯度与动量编码器 [[Grill et al., 2020]](https://arxiv.org/abs/2006.07733)。这些设计在实验中避免了坍塌，但不能概括为所有基于梯度的优化器都“不可避免”坍塌，或某个单一组件能彻底消除风险。

<div align="center">
  <img src="/figures/06-jepa/source/02-mask-collapse/simsiam-fig2.png" alt="SimSiam 的有无停止梯度对比显示，去掉不对称更新后损失会迅速退化并伴随无效表征。" width="86%">

_图 6.2-2：SimSiam 的有无停止梯度对比显示，去掉不对称更新后损失会迅速退化并伴随无效表征。 出处：Xinlei Chen; Kaiming He，[Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)（2021），Figure 2。_

</div>

## 潜在空间预测与优化的“惰性”

先看一维函数。设同一对象的两个增强视图为 $x_1$ 和 $x_2$，特征映射为 $f_\theta$，训练目标要求两者的表示接近。

先看一个一维场景。输入变量 $x \in \mathbb{R}$ 代表某种原始数据，我们希望学习一个特征映射函数 $f_\theta(x)$，其中 $\theta$ 是可学习的参数。给定输入的两个不同视角的观测或增强（记作 $x_1$ 和 $x_2$），模型的目标是让它们在潜在空间中的表征尽可能接近。

最直观的度量方式便是均方误差（Mean Squared Error）：

$$L(\theta) = \frac{1}{2} (f_\theta(x_1) - f_\theta(x_2))^2$$

在传统的监督学习中，$f_\theta(x_2)$ 通常被替换为一个固定的常数标签 $y$。标签 $y$ 是外界赋予的，不受参数 $\theta$ 的控制，因此模型必须努力调整 $\theta$ 以逼近 $y$。
然而，在自监督学习的潜在空间预测中，损失函数的两端 $f_\theta(x_1)$ 和 $f_\theta(x_2)$ 都直接受控于参数 $\theta$。

损失函数只编码“两个输出相等”，并没有要求输出保留多少输入信息。优化器只按这个数值目标更新，因此常数表示不会因缺少语义而自动受到惩罚。

若对任意 $x$ 都有 $f_\theta(x)=c$，则 $f_\theta(x_1)=f_\theta(x_2)$，损失为 0。常数 $c$ 可以非零；关键是不同输入不再可区分。这就是完全表征坍塌的最简单形式。

## 从标量到高维张量的坍塌证明

为了看清高维情形，把编码器简化为没有偏置和非线性的线性映射。这个模型足以证明零映射是一个全局最小点，但结论不能直接替代对深层网络优化轨迹的分析。

假设输入是高维向量 $\mathbf{x} \in \mathbb{R}^d$，特征映射为 $\mathbf{h} = f_{\mathbf{W}}(\mathbf{x}) = \mathbf{W}\mathbf{x} \in \mathbb{R}^k$，其中 $\mathbf{W} \in \mathbb{R}^{k \times d}$ 是权重矩阵。模型在大规模数据集上的预期风险（Expected Risk）或总体优化目标可以表示为所有样本对的均方误差之期望：

$$L(\mathbf{W}) = \mathbb{E}_{\mathbf{x}_1, \mathbf{x}_2} \left[ \|\mathbf{h}_1 - \mathbf{h}_2\|_2^2 \right]$$

将线性映射代入上式，并利用矩阵乘法的分配律，我们可以得到：

$$L(\mathbf{W}) = \mathbb{E}_{\mathbf{x}_1, \mathbf{x}_2} \left[ \|\mathbf{W}\mathbf{x}_1 - \mathbf{W}\mathbf{x}_2\|_2^2 \right] = \mathbb{E}_{\mathbf{x}_1, \mathbf{x}_2} \left[ \|\mathbf{W}(\mathbf{x}_1 - \mathbf{x}_2)\|_2^2 \right]$$

我们定义输入视角的差异向量 $\mathbf{\Delta x} = \mathbf{x}_1 - \mathbf{x}_2$。依据向量二范数平方的定义 $\|\mathbf{v}\|_2^2 = \mathbf{v}^\top \mathbf{v}$，损失函数可以进一步化简为关于 $\mathbf{\Delta x}$ 的二次型形式：

$$L(\mathbf{W}) = \mathbb{E}_{\mathbf{\Delta x}} \left[ (\mathbf{W}\mathbf{\Delta x})^\top (\mathbf{W}\mathbf{\Delta x}) \right] = \mathbb{E}_{\mathbf{\Delta x}} \left[ \mathbf{\Delta x}^\top \mathbf{W}^\top \mathbf{W} \mathbf{\Delta x} \right]$$

此时，我们需要利用线性代数中矩阵迹（Trace）的一个重要循环性质：对于任意向量 $\mathbf{a}$ 和矩阵 $\mathbf{A}$，存在 $\mathbf{a}^\top \mathbf{A} \mathbf{a} = \mathrm{Tr}(\mathbf{A} \mathbf{a} \mathbf{a}^\top)$。由此，我们可以将随机变量 $\mathbf{\Delta x}$ 的期望计算移入迹函数内部：

$$L(\mathbf{W}) = \mathbb{E}_{\mathbf{\Delta x}} \left[ \mathrm{Tr}\left( \mathbf{W}^\top \mathbf{W} \mathbf{\Delta x} \mathbf{\Delta x}^\top \right) \right] = \mathrm{Tr}\left( \mathbf{W}^\top \mathbf{W} \mathbb{E}[\mathbf{\Delta x} \mathbf{\Delta x}^\top] \right)$$

令 $\mathbf{\Sigma} = \mathbb{E}[\mathbf{\Delta x} \mathbf{\Delta x}^\top]$。这实际上是输入差异向量的未中心化协方差矩阵（Covariance Matrix）。根据定义，$\mathbf{\Sigma}$ 必定是一个实对称且半正定（Positive Semi-Definite）的矩阵。同时，对于任意实矩阵 $\mathbf{W}$，其格拉姆矩阵（Gram Matrix）$\mathbf{W}^\top \mathbf{W}$ 也必然是半正定矩阵。

数学上已知，两个半正定矩阵乘积的迹必定大于或等于 $0$：

$$L(\mathbf{W}) = \mathrm{Tr}\left( \mathbf{W}^\top \mathbf{W} \mathbf{\Sigma} \right) \geq 0$$

<div align="center">
  <img src="/figures/06-jepa/latex/02-mask-collapse/collapse-trace-eigenspace.png" alt="输入差异协方差覆盖多个方向，而零权重把所有方向映射到同一原点" width="86%">

_图 6.2-3：Σ 描述数据差异方向；当 W=0 时，所有 Δx 都映射为零，因此非负迹损失恰好达到下界 0。_

</div>

这个非负损失的下界是 0，而 $\mathbf{W}=\mathbf{0}$ 确实达到下界，因为此时 $\mathbf{W}^\top\mathbf{W}=\mathbf{0}$。这证明零映射是一个全局最小解；它并不证明任意初始化和优化器都必然收敛到该点。

当 $\mathbf{W} = \mathbf{0}$ 时，无论输入的图像或文本包含什么信息，所有输入 $\mathbf{x}$ 都会被映射到特征空间的原点 $\mathbf{0} \in \mathbb{R}^k$。这称为**点坍塌（Point Collapse）**。除了点坍塌，如果网络具有非线性结构或归一化层，还可能发生**维度坍塌（Dimensional Collapse）**：特征虽然不为零，却集中在一个低维子空间内，大部分表征维度不再携带有效变化。

## 破局之路：对比学习的排斥力与局限

要打破这种坍塌，最直接的思路是从物理学中汲取灵感：既然吸引力（使得相似样本在空间中靠近）会导致坍塌聚拢，那么我们只要引入一种排斥力即可。这正是对比学习（Contrastive Learning，如 SimCLR [[Chen et al., 2020]](https://arxiv.org/abs/2002.05709)）的核心思想。

对比学习不仅要求正样本对（同一个 $x$ 的不同视角 $x_1, x_2$）的表征 $\mathbf{h}_1$ 和 $\mathbf{h}_2$ 相互吸引，还强制要求负样本对（来自不同实体的数据 $x^-$）的表征 $\mathbf{h}^-$ 相互排斥。通过经典的 InfoNCE 损失函数，分母中的负样本项形成了一种维持特征空间均匀分布的排斥力：

<div align="center">
  <img src="/figures/06-jepa/source/02-mask-collapse/vicreg-fig1.png" alt="VICReg 把不变性、方差与协方差三项并列，展示无需负样本时约束表示分布的另一条路径。" width="86%">

_图 6.2-4：VICReg 把不变性、方差与协方差三项并列，展示无需负样本时约束表示分布的另一条路径。 出处：Adrien Bardes et al.，[VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906)（2022），Figure 1。_

</div>

$$L_{InfoNCE} = - \log \frac{\exp(\mathbf{h}_1^\top \mathbf{h}_2 / \tau)}{\exp(\mathbf{h}_1^\top \mathbf{h}_2 / \tau) + \sum_{j=1}^{N} \exp(\mathbf{h}_1^\top \mathbf{h}_j^- / \tau)}$$

<div align="center">
  <img src="/figures/06-jepa/latex/02-mask-collapse/infonce-softmax-competition.png" alt="一个正样本相似度与 N 个负样本相似度共同进入指数归一化分母" width="86%">

_图 6.2-5：正样本指数项既在分子中，也与全部负样本指数项共享分母；降低损失要求提高相对而非绝对正样本得分。_

</div>

对比学习的效果会受负样本数量与质量影响。SimCLR 使用大批量获得更多批内负样本，MoCo 则用队列把负样本数量与当前批大小部分解耦。另一条路线是不使用显式负样本，而通过在线—目标分支、预测器和停止梯度等不对称设计训练。

## 非对称优化与动量编码器（EMA）

世界模型（如 JEPA）与 BYOL 吸取了前人的教训，采用了一种极具工程美感且数学上极为精妙的设计：**引入不对称的前向预测通路，并通过动量指数移动平均（EMA）构建停止梯度（Stop-Gradient）的目标网络。**

具体而言，网络被物理隔离为两个并行的分支：

1. **在线网络（Online Network）**：负责实际的特征提取与预测，参数记为 $\theta$。优化器根据损失函数的梯度专门更新该网络的参数。
2. **目标网络（Target Network）**：负责为在线网络提供优化的“目标靶点”，参数记为 $\xi$。我们**禁止**目标网络通过反向传播获取梯度，即施加停止梯度（Stop-Gradient）操作。

在这样的架构下，掩码预测的损失函数变更为不对称形式：

$$L(\theta) = \mathbb{E}_{\mathbf{x}_1, \mathbf{x}_2} \left[ \| f_\theta(\mathbf{x}_1) - f_\xi(\mathbf{x}_2) \|_2^2 \right]$$

对当前损失求导时，参数 $\xi$ 被视作常数。如果目标网络永久冻结，在线网络是在拟合一套固定目标特征；这些特征是否有用不能从随机初始化本身保证。实际方法让目标分支通过 EMA 缓慢跟随在线分支。
因此，我们需要一种方法让 $\xi$ 能够缓慢演进，吸收在线网络 $\theta$ 学习到的知识，但又不能快到与 $\theta$ 陷入相同的数学陷阱。这种方法就是**指数移动平均（EMA）**：在每次训练迭代 $t$ 后，手动使用在线网络的参数来平滑更新目标网络的参数：

$$\xi_t \leftarrow \tau \xi_{t-1} + (1 - \tau) \theta_t$$

其中 $\tau\in[0,1)$ 是动量系数。它越接近 1，目标参数越多地保留上一步状态，单次变化越小。

::: info 非数学类比
这也是我们在全篇推导中唯一允许的一个非数学类比：

可以把当前一步看成学生核对一份暂时固定的答案：停止梯度使在线分支不能在同一步直接改写目标，EMA 则让这份答案跨步缓慢变化。这个比喻只解释计算图的不对称性；它不证明答案一定有语义，也不证明 EMA 单独足以避免坍塌。
:::

正是由于 EMA 提供的“惯性锚点”，使得预测的靶点在短时期内是半静态的。基于梯度的优化器无法通过同时将 $\theta$ 和 $\xi$ 压缩至零来走捷径，从而从根本上消除了表征坍塌的数学温床。

## 代码实战：观察坍塌与引入动量编码器

下面用一个小实验同时记录预测损失与特征方差。它用于展示诊断方法，不是对 BYOL 或 I-JEPA 的复现实验，也不能证明 EMA 在所有配置下都阻止坍塌。

先定义一个小型多层感知机编码器。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt

# 确保实验结果可复现
torch.manual_seed(42)

class SimpleEncoder(nn.Module):
    """一个简单的非线性特征提取器。"""
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
```

再用独立高斯扰动构造同一基础向量的两个视图。
在这里，我们通过向原始高维向量中注入高斯噪声，来生成同一实体的不同视角。

```python
def generate_views(batch_size, input_dim, noise_std=0.1):
    """生成同一批次数据的两个不同视角。"""
    # 原始基础特征 (代表潜在的物理实体)
    base_data = torch.randn(batch_size, input_dim)
    # 视角1和视角2，加入了独立的噪声
    view1 = base_data + torch.randn_like(base_data) * noise_std
    view2 = base_data + torch.randn_like(base_data) * noise_std
    return view1, view2
```

实验一让同一个编码器的两个输出直接接近，并观察特征方差。
我们直接通过最小化两个视角的特征均方误差来训练编码器。为了监测坍塌，我们计算每个批次特征向量的**方差（Variance）**。如果所有样本输出相同的值（坍塌），方差将迅速跌至零。

```python
def train_naive_predictor(steps=500):
    encoder = SimpleEncoder()
    optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

    variances = []
    losses = []

    for step in range(steps):
        v1, v2 = generate_views(batch_size=256, input_dim=128)

        # 提取特征
        h1 = encoder(v1)
        h2 = encoder(v2)

        # 计算均方误差损失
        loss = F.mse_loss(h1, h2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录特征在批次维度上的方差平均值
        var = h1.var(dim=0).mean().item()
        variances.append(var)
        losses.append(loss.item())

    return losses, variances

naive_losses, naive_vars = train_naive_predictor()
print(f"原生模型最终特征方差: {naive_vars[-1]:.6f} (坍塌为0)")
```

实验二加入停止梯度和 EMA 目标编码器。它只复现了非对称分支的一部分，并未包含 BYOL 的预测器等组件。

```python
def update_ema_variables(online_net, target_net, tau):
    """
    使用指数移动平均(EMA)缓慢更新目标网络参数。
    公式: \xi \leftarrow \tau \xi + (1 - \tau) \theta
    """
    with torch.no_grad(): # 确保更新过程不被计算图追踪
        for online_param, target_param in zip(online_net.parameters(), target_net.parameters()):
            target_param.mul_(tau).add_(online_param, alpha=1 - tau)

def train_ema_predictor(steps=500, tau=0.99):
    online_encoder = SimpleEncoder()
    # 目标网络是独立存在的，初始权重与在线网络相同
    target_encoder = copy.deepcopy(online_encoder)

    # 停止目标网络的梯度计算
    for param in target_encoder.parameters():
        param.requires_grad = False

    # 优化器只负责更新在线网络
    optimizer = optim.Adam(online_encoder.parameters(), lr=1e-3)

    variances = []
    losses = []

    for step in range(steps):
        v1, v2 = generate_views(batch_size=256, input_dim=128)

        # 在线网络前向传播
        h_online = online_encoder(v1)

        # 目标网络前向传播（不计算梯度）
        with torch.no_grad():
            h_target = target_encoder(v2)

        # 计算非对称的均方误差损失
        loss = F.mse_loss(h_online, h_target)

        # 反向传播，仅更新在线网络参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 手动 EMA 更新目标网络参数
        update_ema_variables(online_encoder, target_encoder, tau)

        var = h_online.var(dim=0).mean().item()
        variances.append(var)
        losses.append(loss.item())

    return losses, variances

ema_losses, ema_vars = train_ema_predictor()
print(f"EMA模型最终特征方差: {ema_vars[-1]:.6f}")
```

运行时应同时比较两条损失曲线和方差曲线。损失下降而方差接近 0 是坍塌信号；方差非零只说明不同样本仍有差异，不等价于表征具有语义。由于随机种子、网络和优化配置都会影响结果，这个小实验的输出应当作诊断样例，而不是普遍结论。

## 小结

- 在未标注的高维潜在空间进行直接回归预测时，基于梯度下降的优化过程天然倾向于让网络输出常数以走捷径，此现象即为**表征坍塌（Representation Collapse）**。
- 在线性简化模型中，半正定迹损失使全零权重成为一个全局最小点；深层网络还可能出现常数或低维坍塌等其他形式。
- 早期的自监督方法（如 SimCLR）依赖于庞大的负样本构建 **InfoNCE 空间排斥力**来抵抗坍塌，但受限于计算资源瓶颈。
- BYOL、I-JEPA 等方法使用**非对称计算图**：在线分支接受梯度，目标分支停止梯度并通过 EMA 缓慢更新。是否避免坍塌要结合预测器、归一化、掩码或增强策略以及实验结果判断。
