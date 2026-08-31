# 6.3 目标网络（Target Network）与指数移动平均（EMA）

在之前的章节中，我们探讨了如何让模型通过预测自身或部分遮蔽的输入来学习有用的表征。然而，当我们试图让一个神经网络去预测它自己的输出时，往往会遭遇一个极其致命的数学问题——**表征坍塌（Representation Collapse）**。为了解决这一问题，深度学习研究者们引入了“目标网络”（Target Network）以及“指数移动平均”（Exponential Moving Average, EMA）技术。本节将从基础代数出发，极其详尽地推导 EMA 的数学本质，并探究它如何在联合嵌入预测架构（JEPA）及各种自监督学习模型中起到定海神针般的作用。

## 6.3.1 学术追溯与历史背景

DQN 使用一个延迟更新的**目标网络（Target Network）**计算时序差分目标 [[Mnih et al., 2015]](https://doi.org/10.1038/nature14236)。在线网络持续更新，而目标网络的参数会冻结一段时间，再从在线网络复制。这样可以在若干次参数更新期间保持回归目标相对稳定；论文同时还使用了经验回放，因此不能把训练稳定性完全归因于目标网络一个组件。

随后，动量编码器进入自监督学习。MoCo 用动量编码器维护较一致的键表示，但仍依赖队列中的负样本 [[He et al., 2020]](https://arxiv.org/abs/1911.05722)；BYOL 才展示了不使用显式负样本的在线网络—目标网络方案 [[Grill et al., 2020]](https://arxiv.org/abs/2006.07733)。data2vec [[Baevski et al., 2022]](https://arxiv.org/abs/2202.03555) 与 I-JEPA [[Assran et al., 2023]](https://arxiv.org/abs/2301.08243) 也使用停止梯度与 EMA 目标编码器来提供稳定的学习目标。

## 6.3.2 自举预测与表征坍塌的代数本质

在深入探讨目标网络之前，我们必须用最基础的数学语言，搞清楚为什么简单的“自己预测自己”行不通。

假设我们有一个非常简单的目标：给定同一个物体的两个不同视角（例如一张猫的图片的两次不同裁剪）$x_1$ 和 $x_2$，我们希望神经网络 $f_\\theta$ 对它们的特征表示尽可能一致。
最直观的损失函数（Loss Function）可以定义为它们特征向量之间的均方误差（Mean Squared Error, MSE）：

$$L(\\theta) = \\| f_\\theta(x_1) - f_\\theta(x_2) \\|^2$$

其中，$\\theta$ 是神经网络的参数。在高中代数中，如果要让等式 $(a - b)^2 = 0$ 成立，最平庸的解是什么？是 $a = b = 0$。或者 $a = b = C$，其中 $C$ 是任意常数。

映射到神经网络的优化过程中，如果我们在上式中对参数 $\\theta$ 求导并进行梯度下降，优化器（Optimizer）会发现一条阻力最小的捷径（Shortcut）：不需要费尽心机去提取 $x_1$ 和 $x_2$ 中关于猫的复杂纹理特征，只需要将参数 $\\theta$ 全部变成 0，或者使得对于任意输入 $x$，网络 $f_\\theta(x)$ 总是输出同一个固定的常数向量 $\\mathbf{c}$。一旦发生这种情况，损失函数瞬间降为 0，但这时的网络已经变成了一个毫无意义的常数函数。这种现象，在学术界被称为**表征坍塌（Representation Collapse）**。

## 6.3.3 引入不对称性：从方程求解到目标网络

为了打破这种坍塌，我们必须打破方程的对称性。在数学上，如果方程左右两边的变量都受你控制，系统很容易滑向平凡解。但如果等式右边的目标是固定的（或者不受当前梯度更新的影响），系统就必须真正去拟合这个目标。

我们引入一组完全独立的参数 $\\theta'$，用来专门生成“目标（Target）”。此时模型被拆分为两个部分：

1. **在线网络（Online Network, 亦称学生网络）**：参数为 $\\theta$，负责根据输入 $x_1$ 进行预测，并在训练过程中接收梯度进行更新。
2. **目标网络（Target Network, 亦称教师网络）**：参数为 $\\theta'$，负责接收输入 $x_2$ 生成目标。关键在于，**（目标网络的计算图必须截断梯度（Stop-Gradient））**。

修改后的损失函数变为：

$$L(\\theta) = \\| f_\\theta(x_1) - \\text{sg}(f_{\\theta'}(x_2)) \\|^2$$

其中 $\\text{sg}(\\cdot)$ 表示停止梯度（Stop-Gradient）操作。此时，对于在线网络 $\\theta$ 来说，$f_{\\theta'}(x_2)$ 在当前一步被当作常数目标。不过，停止梯度本身并不能从理论上排除目标网络随时间一起坍塌；实际方法还依赖 EMA、预测器、归一化或其他正则与架构设计。

但这就引出了一个新的问题：目标网络 $\\theta'$ 的参数从何而来？如果 $\\theta'$ 是随机初始化的且永远不更新，那么在线网络最终只能学会预测随机噪声。如果 $\\theta'$ 也是通过同样的损失函数训练的，那就回到了原本坍塌的死胡同。

在强化学习的 DQN 中，研究者的做法是**（硬更新（Hard Update））**：每隔 $K$ 个训练步骤，直接把在线网络的参数强行复制给目标网络：

$$\\theta' \\leftarrow \\theta, \\quad \\text{every } K \\text{ steps}$$

硬更新虽然有效，但会导致目标网络的行为呈现出剧烈的阶跃式变化（Step-function behavior），使得训练过程产生不必要的震荡。我们需要一种更加平滑、连续的参数同步机制。

## 6.3.4 指数移动平均（EMA）的严密推导

为了让目标网络既能缓慢地跟上在线网络的学习进度，又不会因为更新过快而导致系统陷入坍塌震荡，研究者借用了统计学与信号处理中的**指数移动平均（Exponential Moving Average, EMA）**技术。我们将目标网络参数的更新规则定义为一种**软更新（Soft Update）**。

在训练的第 $t$ 步，假设在线网络的参数为 $\\theta_t$，目标网络的参数为 $\\theta'_t$，EMA 的递推公式严格定义为：

$$\\theta'_t = \\tau \\theta'_{t-1} + (1 - \\tau) \\theta_t$$

这里的 $\\tau \\in [0, 1)$ 是一个极其关键的超参数，被称为动量系数（Momentum Coefficient）或衰减率（Decay Rate）。在实际的自监督学习模型（如 BYOL 或 JEPA）中，$\\tau$ 通常取一个非常接近 1 的值，例如 $\\tau = 0.99$ 或 $\\tau = 0.996$。

### 展开递推公式：几何级数的物理意义

为了深刻理解 EMA 为什么有效，我们不能仅停留在上述的一阶递推公式上，必须将其在时间维度上完全展开。假设我们在 $t=0$ 时刻初始化参数 $\\theta'_0 = \\theta_0$。

计算第 1 步：
$$\\theta'_1 = \\tau \\theta'_0 + (1 - \\tau) \\theta_1$$

计算第 2 步：

$$ \begin{aligned}
\\theta'_2 &= \\tau \\theta'_1 + (1 - \\tau) \\theta_2 \\\\
&= \\tau (\\tau \\theta'_0 + (1 - \\tau) \\theta_1) + (1 - \\tau) \\theta_2 \\\\
&= \\tau^2 \\theta'_0 + \\tau (1 - \\tau) \\theta_1 + (1 - \\tau) \\theta_2
\\end{aligned}$$

计算第 3 步：
$$\\begin{aligned}
\\theta'_3 &= \\tau \\theta'_2 + (1 - \\tau) \\theta_3 \\\\
&= \\tau^3 \\theta'_0 + \\tau^2 (1 - \\tau) \\theta_1 + \\tau (1 - \\tau) \\theta_2 + (1 - \\tau) \\theta_3
\\end{aligned}$$

根据数学归纳法，我们可以推导出第 $t$ 步时的目标网络参数通项公式：

$$\\theta'_t = \\tau^t \\theta'_0 + (1 - \\tau) \\sum_{i=1}^t \\tau^{t-i} \\theta_i$$

让我们运用高中数学中等比数列（几何级数）的知识来剖析这个公式。

等号右边第一项 $\\tau^t \\theta'_0$ 表示初始参数的影响。因为 $0 < \\tau < 1$，随着时间 $t \\to \\infty$，$\\tau^t$ 将呈指数级衰减趋近于 0。这意味着，系统最终会“遗忘”其初始状态。

等号右边第二项是一个加权求和，我们可以将其改写为对历史所有在线网络参数 $\\theta_i$ 的积分形式。对于时刻 $i$，它的权重为 $(1 - \\tau) \\tau^{t-i}$。
注意这里的指数是 $t-i$。由于 $i \\le t$，$t-i$ 表示历史时刻 $i$ 距离当前时刻 $t$ 的**时间差**。时间差越大（即参数越古老），指数 $t-i$ 越大，权重 $\\tau^{t-i}$ 就呈指数级变小。

因此，目标网络 $\\theta'_t$ 本质上是**在线网络历史参数的一个指数加权移动平均**。离当前时刻越近的在线参数，获得的权重越大；离当前时刻越远的参数，权重越小。

::: info 说明
在这一复杂的动态系统中，我们可以将“在线网络”视为一个锐意进取但时常犯错的学生，而“目标网络”则是一个稳重的老师。老师不会盲目听信学生当下的每一步疯狂尝试（因为在线网络的单步梯度可能含有大量噪声），而是将学生过去成千上万步的经验进行“平滑加权融合”。这种基于时间尺度的平滑，使得老师能够提供一个极其稳定、高质量的目标，从而反过来指导学生的下一步学习，彻底打破了自己预测自己导致的坍塌闭环。
:::

### 权重归一化证明

作为严谨的数学检验，我们需要确认这些历史权重的总和是否为 1。假设我们考察一个无穷序列，过去所有权重的总和为：
$$S = (1 - \\tau) \\sum_{k=0}^{\\infty} \\tau^k$$
根据无穷等比数列求和公式 $\\sum_{k=0}^{\\infty} q^k = \\frac{1}{1-q}$（当 $|q| < 1$ 时），我们有：
$$S = (1 - \\tau) \\cdot \\frac{1}{1 - \\tau} = 1$$
这一严格的证明保证了目标网络参数在任何时刻，都在参数空间中保持合法的缩放比例，不会发生数值爆炸。

## 6.3.5 结合张量维度的代码实现

现在，我们将这些严谨的数学公式转化为现代深度学习框架下的代码。我们需要实现两个关键组件：第一是包含停止梯度机制的前向传播流程；第二是能够无缝处理多维张量更新的 EMA 函数。

首先，我们定义一个极其简单的双层感知机（MLP）作为我们的编码器网络。我们将展示如何手动进行 EMA 更新，而不是依赖于优化器。

```python
import torch
import torch.nn as nn
import copy

# 定义一个简单的多层感知机作为网络骨干
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 实例化在线网络
online_network = SimpleEncoder()

# 目标网络是完全不参与梯度计算的，我们首先深拷贝在线网络
target_network = copy.deepcopy(online_network)

# [强制冻结目标网络的所有参数，彻底截断梯度回传]
for param in target_network.parameters():
    param.requires_grad = False
```

接下来，我们实现 EMA 更新的核心函数。在 PyTorch 中，网络的所有参数可以通过 `.parameters()` 获取，这是一个包含多个不同维度张量（如权重矩阵为二维张量，偏置向量为一维张量）的生成器。我们的更新逻辑需要逐元素（element-wise）地对这些张量应用该公式。

```python
def update_target_network_ema(online_net, target_net, tau):
    """
    使用指数移动平均（EMA）更新目标网络参数。

    参数：
        online_net (nn.Module): 正在接收梯度更新的在线网络
        target_net (nn.Module): 需要被平滑更新的目标网络
        tau (float): 动量系数，通常接近 1，例如 0.99
    """
    # 确保网络结构完全一致，zip 将两个网络的参数张量一一配对
    with torch.no_grad(): # 更新过程绝对不能被记录在计算图中
        for online_param, target_param in zip(online_net.parameters(), target_net.parameters()):
            # 执行核心数学公式： theta'_t = tau * theta'_{t-1} + (1 - tau) * theta_t
            # 在张量操作中，我们使用 inplace 乘法和加法来节省显存
            target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)
```

为了让读者直观感受到参数的演变，我们可以模拟几个训练步骤。在这个模拟中，我们假设输入数据的批量大小（Batch Size）为 `32`，特征维度为 `128`，即输入张量的形状为 `(32, 128)`。

```python
# 模拟训练数据
x1 = torch.randn(32, 128) # 视图 1
x2 = torch.randn(32, 128) # 视图 2

# 定义优化器，仅更新在线网络参数
optimizer = torch.optim.Adam(online_network.parameters(), lr=1e-3)
tau = 0.99 # 设置极高的动量系数

# 模拟 3 个训练 Step
for step in range(3):
    # 1. 在线网络进行前向传播
    # 输出形状: (32, 128)
    online_pred = online_network(x1)

    # 2. 目标网络生成目标
    # [关键点：通过 torch.no_grad() 确保完全没有梯度流向 target_network]
    with torch.no_grad():
        target_proj = target_network(x2)

    # 3. 计算均方误差损失
    # 此处的 loss 张量仅包含关于 online_network 的偏导数信息
    loss = nn.functional.mse_loss(online_pred, target_proj)

    # 4. 反向传播与在线网络更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 5. [应用 EMA 更新目标网络参数]
    update_target_network_ema(online_network, target_network, tau)

    print(f"Step {step + 1}: Loss = {loss.item():.4f}")
```

在这个过程中，如果去掉 EMA 更新步骤（即让 `tau=1.0` 使得目标网络永远不变），模型最终会因为目标过于静态而无法学到充分的新知识；相反，如果 `tau=0.0`（完全等同于没有任何目标网络），损失会在几个 epoch 内急剧下降至零，因为两组参数瞬间同步，模型立刻找到了输出全为零的坍塌捷径。通过在张量层面小心翼翼地平衡这两种力量，自监督学习系统才能维持在表征丰富且不坍塌的临界边缘。

## 6.3.6 小结

在本节中，我们深入探究了自监督预测架构中最致命的缺陷——表征坍塌。通过回归基础的代数方程求解逻辑，我们论证了破坏系统完全对称性的必要性。引入截断梯度的目标网络，并在时间维度上施加指数移动平均（EMA），是目前维持表征空间稳定性的黄金标准。EMA 使得目标网络成为了在线网络所有历史状态的一个低通滤波器（Low-pass Filter），滤除了每一步梯度更新中的高频噪声，提供了一个缓慢但坚定进化的锚点（Anchor）。这一机制构成了后续理解 JEPA 和数据空间掩码建模不可或缺的理论基石。
$$
