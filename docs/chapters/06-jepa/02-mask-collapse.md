# 掩码预测与表征坍塌（Representation Collapse）问题

自监督学习（Self-Supervised Learning, SSL）利用数据自身构造训练目标。例如，BERT 根据未遮挡的上下文预测被遮挡的词元 [[Devlin et al., 2018]](https://arxiv.org/abs/1810.04805)，掩码自编码器（Masked Autoencoders, MAE）则重构被遮挡图像块的像素 [[He et al., 2022]](https://arxiv.org/abs/2111.06377)。这两篇论文分别支撑词元空间与像素空间中的掩码预测。

然而，随着我们对人工智能的期望逐渐从“模式识别”升级为“理解复杂物理规律”的世界模型（World Models），原始像素空间的局限性开始暴露无遗。在原始高维像素空间中进行精确重建，不仅计算代价极其高昂，而且会迫使模型将庞大的网络容量浪费在预测高频但缺乏语义价值的纹理细节（如微风吹过的草地纹理、随机的背景噪声）上。

为了减少对像素解码的依赖，data2vec 使用教师网络产生的潜在表征作为预测目标 [[Baevski et al., 2022]](https://arxiv.org/abs/2202.03555)，I-JEPA 则在图像块的表征空间中进行掩码预测 [[Assran et al., 2023]](https://arxiv.org/abs/2301.08243)。data2vec 与 I-JEPA 属于相关的潜在目标预测方法，但前者并未以 JEPA 命名。只预测动态生成的表征时，需要专门设计训练机制来避免**表征坍塌（Representation Collapse）**。

本节从代数与几何视角分析常数表示为何是一个平凡解，并讨论 BYOL 等方法使用的非对称架构、停止梯度与动量编码器 [[Grill et al., 2020]](https://arxiv.org/abs/2006.07733)。这些设计在实验中避免了坍塌，但不能概括为所有基于梯度的优化器都“不可避免”坍塌，或某个单一组件能彻底消除风险。

## 潜在空间预测与优化的“惰性”

在我们深入高维矩阵之前，让我们先将视角拉回高中阶段的基础代数。本质上，深度神经网络是一个极其复杂的参数化复合函数。而在无监督环境下的掩码预测任务，本质上是希望网络针对同一个物理实体的不同视图（或掩码版本），能够输出一致的特征。

假设我们有一个非常简单的一维场景。输入变量 $x \in \mathbb{R}$ 代表某种原始数据，我们希望学习一个特征映射函数 $f_\theta(x)$，其中 $\theta$ 是可学习的参数。给定输入的两个不同视角的观测或增强（记作 $x_1$ 和 $x_2$），模型的目标是让它们在潜在空间中的表征尽可能接近。

最直观的度量方式便是均方误差（Mean Squared Error）：

$$L(\theta) = \frac{1}{2} (f_\theta(x_1) - f_\theta(x_2))^2$$

在传统的监督学习中，$f_\theta(x_2)$ 通常被替换为一个固定的常数标签 $y$。标签 $y$ 是外界赋予的，不受参数 $\theta$ 的控制，因此模型必须努力调整 $\theta$ 以逼近 $y$。
然而，在自监督学习的潜在空间预测中，损失函数的两端 $f_\theta(x_1)$ 和 $f_\theta(x_2)$ 都直接受控于参数 $\theta$。

这就引出了基于梯度下降优化的一个根本属性：**优化的“惰性”**。优化器（如随机梯度下降，SGD）是“盲目”的，它仅仅沿着损失平面的最陡下降方向前进，寻找能使得 $L(\theta)$ 达到最小值的参数解，而完全不关心这个解是否具有我们期望的物理或语义意义。

从代数的角度看，如果要最小化该公式使得损失为 $0$，最简单的数学解是什么？显然，只需要让函数 $f_\theta$ 退化为一个不受输入 $x$ 影响的常数函数，即对于任意的 $x$，都有 $f_\theta(x) = c$。
一旦 $f_\theta(x_1) = c$ 且 $f_\theta(x_2) = c$，那么误差 $L = 0$。模型以最快速、最取巧的数学途径达到了全局最优解。然而，这个恒定的常数 $c$ 彻底丢失了关于输入 $x$ 的所有信息。这种为了迎合损失函数而导致的特征空间信息量归零的现象，正是最基础的表征坍塌。

## 从标量到高维张量的坍塌证明

现实中的深度学习模型处理的是极高维度的数据。表征坍塌在高维空间中又是如何表现的呢？为了保持数学推导的严谨性与直观性，我们将神经网络极度简化为一个没有非线性激活函数的单层线性映射。

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

我们要寻找使得该非负损失函数取得最小值的参数 $\mathbf{W}$。在没有任何结构性约束（例如正交性约束或信息论瓶颈）的情况下，基于梯度的优化器会轻易地找到绝对下界：直接令权重矩阵全为零，即 $\mathbf{W} = \mathbf{0}$。此时 $\mathbf{W}^\top \mathbf{W} = \mathbf{0}$，从而使得 $L(\mathbf{W})$ 严格等于 $0$。

当 $\mathbf{W} = \mathbf{0}$ 时，无论输入的图像或文本蕴含着何种丰富的语义，所有输入 $\mathbf{x}$ 都会被映射到特征空间的绝对原点 $\mathbf{0} \in \mathbb{R}^k$。此时，整个高维潜在空间的表征发生了彻底的**点坍塌（Point Collapse）**。除了点坍塌，如果网络具有非线性结构或归一化层，还可能发生**维度坍塌（Dimensional Collapse）**，即特征虽不为零，但全部挤压在一个低维子空间内，大部分表征维度成为无用的冗余。

## 破局之路：对比学习的排斥力与局限

要打破这种坍塌，最直接的思路是从物理学中汲取灵感：既然吸引力（使得相似样本在空间中靠近）会导致坍塌聚拢，那么我们只要引入一种排斥力即可。这正是对比学习（Contrastive Learning，如 SimCLR [[Chen et al., 2020]](https://arxiv.org/abs/2002.05709)）的核心思想。

对比学习不仅要求正样本对（同一个 $x$ 的不同视角 $x_1, x_2$）的表征 $\mathbf{h}_1$ 和 $\mathbf{h}_2$ 相互吸引，还强制要求负样本对（来自不同实体的数据 $x^-$）的表征 $\mathbf{h}^-$ 相互排斥。通过经典的 InfoNCE 损失函数，分母中的负样本项形成了一种维持特征空间均匀分布的排斥力：

$$L_{InfoNCE} = - \log \frac{\exp(\mathbf{h}_1^\top \mathbf{h}_2 / \tau)}{\exp(\mathbf{h}_1^\top \mathbf{h}_2 / \tau) + \sum_{j=1}^{N} \exp(\mathbf{h}_1^\top \mathbf{h}_j^- / \tau)}$$

然而，对比学习对硬件资源极度贪婪。为了确保排斥力能够均匀覆盖整个高维潜在空间，模型在每一次梯度更新时都需要极其庞大的负样本数量（Batch Size 往往高达数千乃至上万），这在普通计算设备上几乎是不可承受的。那么，有没有办法在不需要负样本的情况下，仅仅通过改变架构的数学对称性来避免坍塌呢？

## 非对称优化与动量编码器（EMA）

世界模型（如 JEPA）与 BYOL 吸取了前人的教训，采用了一种极具工程美感且数学上极为精妙的设计：**引入不对称的前向预测通路，并通过动量指数移动平均（EMA）构建停止梯度（Stop-Gradient）的目标网络。**

具体而言，网络被物理隔离为两个并行的分支：

1. **在线网络（Online Network）**：负责实际的特征提取与预测，参数记为 $\theta$。优化器根据损失函数的梯度专门更新该网络的参数。
2. **目标网络（Target Network）**：负责为在线网络提供优化的“目标靶点”，参数记为 $\xi$。我们**禁止**目标网络通过反向传播获取梯度，即施加停止梯度（Stop-Gradient）操作。

在这样的架构下，掩码预测的损失函数变更为不对称形式：

$$L(\theta) = \mathbb{E}_{\mathbf{x}_1, \mathbf{x}_2} \left[ \| f_\theta(\mathbf{x}_1) - f_\xi(\mathbf{x}_2) \|_2^2 \right]$$

注意该公式中，变量 $\xi$ 被视作常数，不参与关于 $\theta$ 的偏导数计算。如果 $\xi$ 完全固定不动，这就等价于传统的监督学习（用随机初始化的网络生成随机但不坍塌的伪标签）。但固定的特征空间过于贫瘠，无法学到高质量的语义。
因此，我们需要一种方法让 $\xi$ 能够缓慢演进，吸收在线网络 $\theta$ 学习到的知识，但又不能快到与 $\theta$ 陷入相同的数学陷阱。这种方法就是**指数移动平均（EMA）**：在每次训练迭代 $t$ 后，手动使用在线网络的参数来平滑更新目标网络的参数：

$$\xi_t \leftarrow \tau \xi_{t-1} + (1 - \tau) \theta_t$$

其中 $\tau \in [0, 1)$ 是动量衰减系数（Momentum Coefficient）。在实践中，$\tau$ 通常设置得非常高（例如 $0.99$ 或 $0.999$），这意味着目标网络 $\xi$ 的变化极其缓慢，绝大多数信息继承自上一步的历史状态。

::: info 非数学类比
这也是我们在全篇推导中唯一允许的一个非数学类比：

假设两个学生（在线网络 $\theta$ 和目标网络 $\xi$）通过互相核对答案来完成一份难度极高的开卷试卷。如果允许他们随意商量并随时修改对方的答案，为了最快完成任务，他们极有可能达成共识：“我们在所有题目上都填 0”。这就发生了坍塌。

但是，如果规则改变：学生 $\xi$ 绝不允许涂改自己当前的答案（停止梯度），但他可以极其缓慢地、以极小的权重参考学生 $\theta$ 刚刚写出的解法（EMA 更新）。此时，学生 $\theta$ 面对的是一份不断进步但不妥协、无法被轻易拉低的标准答案。为了使答案接近，学生 $\theta$ 别无选择，只能真正去学习题目的内在规律。这种不对称的机制彻底打破了共同奔向零解的数学对称性。
:::

正是由于 EMA 提供的“惯性锚点”，使得预测的靶点在短时期内是半静态的。基于梯度的优化器无法通过同时将 $\theta$ 和 $\xi$ 压缩至零来走捷径，从而从根本上消除了表征坍塌的数学温床。

## 代码实战：观察坍塌与引入动量编码器

空谈理论不如直接观测。接下来，我们将构建一个极简的高维潜在空间预测实验。我们将首先展示如果不使用动量编码器，模型是如何在几百步内迅速发生彻底的表征坍塌的；随后，我们将引入 EMA，证明其对抗坍塌的强大效力。

(**首先，我们导入必要的库，并定义一个简单的多层感知机作为编码器。**)

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

(**为了模拟数据增强或不同的掩码视角，我们构造一个简单的数据生成函数。**)
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

(**实验一：原生的潜在空间预测（观察表征坍塌）。**)
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

(**实验二：引入动量编码器（EMA）与停止梯度。**)
现在，我们遵循 JEPA 和 BYOL 的核心机制，将网络分为在线网络（更新梯度）和目标网络（EMA更新）。

```python
def update_ema_variables(online_net, target_net, tau):
    """
    使用指数移动平均(EMA)缓慢更新目标网络参数。
    公式: \xi \leftarrow \tau \xi + (1 - \tau) \theta
    """
    with torch.no_grad(): # 确保更新过程不被计算图追踪
        for online_param, target_param in zip(online_net.parameters(), target_net.parameters()):
            target_param.data.mul_(tau).add_(online_param.data, alpha=1 - tau)

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
print(f"EMA模型最终特征方差: {ema_vars[-1]:.6f} (保持健康分布)")
```

通过实验可以清晰地观测到：由于优化梯度的对称性，原生模型的预测损失固然极其迅速地降至零，但其代价是输出方差同样跌至绝对的零值，特征表征已经毫无意义；而采用了 EMA 动量目标的网络，不仅损失平稳收敛，且在整个训练生命周期内维持了极其健康的非零方差，成功抵御了优化的“惰性”，有效避免了表征坍塌。

## 小结

- 在未标注的高维潜在空间进行直接回归预测时，基于梯度下降的优化过程天然倾向于让网络输出常数以走捷径，此现象即为表征坍塌（Representation Collapse）。
- 从高维矩阵的视角，如果没有显式约束，损失函数矩阵迹项的半正定性必定使得全零矩阵权重成为优化过程中的全局最优点。
- 早期的自监督方法（如 SimCLR）依赖于庞大的负样本构建 InfoNCE 空间排斥力来抵抗坍塌，但受限于计算资源瓶颈。
- 现代预测架构（如 JEPA、BYOL）利用非对称的计算图流：在线网络通过梯度下降进行剧烈学习，而目标网络施加停止梯度（Stop-Gradient）并使用指数移动平均（EMA）提供缓慢漂移的标量锚点，从根本上打破了坍塌的数学对称性条件。
