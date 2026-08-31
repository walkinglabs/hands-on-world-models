# 6.3 目标网络（Target Network）与指数移动平均（EMA）

自监督学习常让一个网络预测由另一条分支生成的目标。如果两条分支同时随同一个损失自由变化，常数表示会成为一种没有信息却能降低损失的解，这就是**表征坍塌**（Representation Collapse）。目标网络（Target Network）与指数移动平均（Exponential Moving Average, EMA）为预测分支提供变化较慢的目标。本节先展开 EMA 的权重，再说明它在 DQN、BYOL、data2vec 与 I-JEPA 中分别承担什么作用。

## 6.3.1 学术追溯与历史背景

DQN 使用一个延迟更新的**目标网络（Target Network）**计算时序差分目标 [[Mnih et al., 2015]](https://doi.org/10.1038/nature14236)。在线网络持续更新，而目标网络的参数会冻结一段时间，再从在线网络复制。这样可以在若干次参数更新期间保持回归目标相对稳定；论文同时还使用了经验回放，因此不能把训练稳定性完全归因于目标网络一个组件。

随后，动量编码器进入自监督学习。MoCo 用动量编码器维护较一致的键表示，但仍依赖队列中的负样本 [[He et al., 2020]](https://arxiv.org/abs/1911.05722)；BYOL 才展示了不使用显式负样本的在线网络—目标网络方案 [[Grill et al., 2020]](https://arxiv.org/abs/2006.07733)。data2vec [[Baevski et al., 2022]](https://arxiv.org/abs/2202.03555) 与 I-JEPA [[Assran et al., 2023]](https://arxiv.org/abs/2301.08243) 也使用停止梯度与 EMA 目标编码器来提供稳定的学习目标。

<div align="center">
  <img src="/figures/06-jepa/source/03-target-network-ema/moco-fig1.png" alt="MoCo 的查询编码器、动量键编码器与队列图显示 EMA 目标分支最初如何服务于一致的对比字典。" width="86%">

_图 6.3-1：MoCo 的查询编码器、动量键编码器与队列图显示 EMA 目标分支最初如何服务于一致的对比字典。 出处：Kaiming He et al.，[Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)（2020），Figure 1。_

</div>

<div align="center">
  <img src="/figures/06-jepa/source/03-target-network-ema/byol-fig2.png" alt="BYOL 的在线分支与目标分支图明确标出预测器、停止梯度和动量更新形成的不对称结构。" width="86%">

_图 6.3-2：BYOL 的在线分支与目标分支图明确标出预测器、停止梯度和动量更新形成的不对称结构。 出处：Jean-Bastien Grill et al.，[Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733)（2020），Figure 2。_

</div>

## 6.3.2 自举预测与表征坍塌的代数本质

在深入探讨目标网络之前，我们必须用最基础的数学语言，搞清楚为什么简单的“自己预测自己”行不通。

假设同一个物体经过两次不同裁剪后得到 $x_1$ 和 $x_2$，我们希望神经网络 $f_\theta$ 对它们的特征表示尽可能一致。
最直观的损失函数（Loss Function）可以定义为它们特征向量之间的均方误差（Mean Squared Error, MSE）：

$$L(\theta) = \| f_\theta(x_1) - f_\theta(x_2) \|^2$$

其中，$\theta$ 是神经网络的参数。让 $(a-b)^2$ 等于 0，只要求 $a=b$；因此 $a=b=0$ 与 $a=b=C$ 都是解。

对应到神经网络，如果对于任意输入 $x$ 都有 $f_\theta(x)=\mathbf{c}$，两端会恒等，损失也会降为 0。此时输出不再保留输入差异：目标函数得到满足，表征却失去用途。要注意，全零参数只是某些简化网络中的一个例子；含偏置、归一化和非线性的模型也可能以其他常数形式坍塌。

## 6.3.3 引入不对称性：从方程求解到目标网络

为了打破这种坍塌，我们必须打破方程的对称性。在数学上，如果方程左右两边的变量都受你控制，系统很容易滑向平凡解。但如果等式右边的目标是固定的（或者不受当前梯度更新的影响），系统就必须真正去拟合这个目标。

我们引入另一组参数 $\theta'$，专门生成目标。模型由两部分组成：

1. **在线网络（Online Network）**：参数为 $\theta$，根据 $x_1$ 产生预测，并接受梯度更新。
2. **目标网络（Target Network）**：参数为 $\theta'$，根据 $x_2$ 生成目标；这条分支在当前损失中停止梯度（Stop-Gradient）。

修改后的损失函数变为：

$$L(\theta) = \| f_\theta(x_1) - \operatorname{sg}(f_{\theta'}(x_2)) \|^2$$

其中 $\operatorname{sg}(\cdot)$ 表示停止梯度。对在线参数 $\theta$ 求导时，$f_{\theta'}(x_2)$ 在当前一步被当作常数目标。不过，停止梯度本身不能从理论上排除目标网络随时间一起坍塌；实际方法还依赖 EMA、预测器、归一化、掩码策略或其他正则与架构设计。

目标网络 $\theta'$ 还需要随训练演化。若它随机初始化后永久冻结，在线网络只是在拟合一套固定的随机特征；若它通过同一个损失同步接受梯度，两条分支又恢复了对称性。

在强化学习的 DQN 中，研究者的做法是**（硬更新（Hard Update））**：每隔 $K$ 个训练步骤，直接把在线网络的参数强行复制给目标网络：

$$\theta' \leftarrow \theta, \qquad \text{每隔 } K \text{ 步}$$

<div align="center">
  <img src="/figures/06-jepa/latex/03-target-network-ema/hard-vs-ema-response.png" alt="硬更新目标呈阶梯变化，而 EMA 目标逐步平滑跟随在线参数" width="86%">

_图 6.3-3：每 K 步复制会让目标参数保持后突然跳变；每步 EMA 则只吸收 1−τ 比例的在线参数变化，形成平滑响应。本文根据硬更新与 EMA 两式绘制；TikZ/LaTeX 编译。_

</div>

硬更新虽然有效，但会导致目标网络的行为呈现出剧烈的阶跃式变化（Step-function behavior），使得训练过程产生不必要的震荡。我们需要一种更加平滑、连续的参数同步机制。

## 6.3.4 指数移动平均（EMA）的严密推导

为了让目标网络既能缓慢地跟上在线网络的学习进度，又不会因为更新过快而导致系统陷入坍塌震荡，研究者借用了统计学与信号处理中的**指数移动平均（Exponential Moving Average, EMA）**技术。我们将目标网络参数的更新规则定义为一种**软更新（Soft Update）**。

在训练的第 $t$ 步，设在线网络参数为 $\theta_t$，目标网络参数为 $\theta'_t$。软更新写为：

$$\theta'_t = \tau \theta'_{t-1} + (1 - \tau) \theta_t$$

这里的 $\tau \in [0,1)$ 是动量系数或衰减率。BYOL、data2vec 和 I-JEPA 等方法通常让它接近 1，并可能在训练过程中调度；具体取值是方法与训练配置的一部分，不是普遍常数。

### 展开递推公式：几何级数的物理意义

把递推式沿时间展开，可以直接看到每个历史参数的权重。设初始化时 $\theta'_0=\theta_0$。

计算第 1 步：
$$\theta'_1 = \tau \theta'_0 + (1 - \tau) \theta_1$$

计算第 2 步：

$$
\begin{aligned}
\theta'_2 &= \tau \theta'_1 + (1 - \tau) \theta_2 \\
&= \tau (\tau \theta'_0 + (1 - \tau) \theta_1) + (1 - \tau) \theta_2 \\
&= \tau^2 \theta'_0 + \tau (1 - \tau) \theta_1 + (1 - \tau) \theta_2
\end{aligned}
$$

计算第 3 步：

$$
\begin{aligned}
\theta'_3 &= \tau \theta'_2 + (1 - \tau) \theta_3 \\
&= \tau^3 \theta'_0 + \tau^2 (1 - \tau) \theta_1 + \tau (1 - \tau) \theta_2 + (1 - \tau) \theta_3
\end{aligned}
$$

根据数学归纳法，我们可以推导出第 $t$ 步时的目标网络参数通项公式：

$$\theta'_t = \tau^t \theta'_0 + (1 - \tau) \sum_{i=1}^t \tau^{t-i} \theta_i$$

让我们运用高中数学中等比数列（几何级数）的知识来剖析这个公式。

第一项 $\tau^t\theta'_0$ 是初始参数的剩余贡献。只要 $0<\tau<1$，它会随 $t$ 指数衰减。

第二项是历史在线参数的离散加权和。时刻 $i$ 的权重是 $(1-\tau)\tau^{t-i}$；离当前越远，权重越小。

因此，目标网络 $\theta'_t$ 本质上是**在线网络历史参数的指数加权移动平均**。离当前时刻越近的在线参数，权重越大；离当前时刻越远，权重越小。

::: info 说明
可以把在线网络看作快速变化的信号，把目标网络看作它的低通版本。一次在线更新只以 $1-\tau$ 的比例进入目标参数，因此高频的逐步波动会被削弱。这个解释说明了“平滑”，但不等价于对不坍塌或目标质量的证明。
:::

### 权重归一化证明

若忽略仍保留的初始化项，并考察无限历史，EMA 权重之和为：

$$S = (1 - \tau) \sum_{k=0}^{\infty} \tau^k$$
根据无穷等比数列求和公式 $\sum_{k=0}^{\infty} q^k = \frac{1}{1-q}$（当 $|q| < 1$ 时），可得：
$$S = (1 - \tau) \cdot \frac{1}{1 - \tau} = 1$$

有限步时，历史在线参数的权重和为 $1-\tau^t$，再加上初始化项的权重 $\tau^t$，总和恰为 1。因此，目标参数是这些参数快照的凸组合；这解释了权重归一化，但网络训练的整体数值稳定性仍取决于在线更新等其他因素。

## 6.3.5 结合张量维度的代码实现

现在，我们将这些严谨的数学公式转化为现代深度学习框架下的代码。我们需要实现两个关键组件：第一是包含停止梯度机制的前向传播流程；第二是能够无缝处理多维张量更新的 EMA 函数。

首先定义一个双层感知机（MLP）作为编码器，并手动更新 EMA，而不是把目标参数交给优化器。

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
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 实例化在线网络
online_network = SimpleEncoder()

# 目标网络是完全不参与梯度计算的，我们首先深拷贝在线网络
target_network = copy.deepcopy(online_network)

# 冻结目标网络参数，使当前损失不向目标分支回传梯度
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
            target_param.mul_(tau).add_(online_param, alpha=1.0 - tau)
```

为了让读者直观感受到参数的演变，我们可以模拟几个训练步骤。在这个模拟中，我们假设输入数据的批量大小（Batch Size）为 `32`，特征维度为 `128`，即输入张量的形状为 `(32, 128)`。

```python
# 模拟训练数据
x1 = torch.randn(32, 128) # 视图 1
x2 = torch.randn(32, 128) # 视图 2

# 定义优化器，仅更新在线网络参数
optimizer = torch.optim.Adam(online_network.parameters(), lr=1e-3)
tau = 0.99 # 示例动量系数

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

当 `tau=1.0` 时，目标网络保持初始化状态；当 `tau=0.0` 时，每次更新后目标参数直接复制在线参数。两种极端都会改变目标随时间演化的方式，但仅凭 `tau` 不能断言何时必然坍塌。实际结果还取决于预测器、归一化、数据增强、掩码和优化配置，训练时应同时监测损失与表征方差，而不是只看损失下降。

## 6.3.6 小结

本节得到三个结论：目标网络让当前损失只更新在线分支；EMA 把目标参数写成在线参数历史的指数加权平均；较大的 $\tau$ 会让目标变化得更慢。它们解释了 I-JEPA 等方法中目标分支为何比在线分支平滑。是否避免表征坍塌则必须结合完整架构和实验诊断判断，不能由 EMA 单独推出。
