# 2.6 基础模块的简洁实现

早期神经网络系统常要求研究者自行实现许多算子的前向与反向计算。PyTorch 等现代框架 [[Paszke et al., 2019]](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) 用张量算子、计算图和自动微分封装了这些重复工作，使模型代码可以更直接地表达数学结构。

<div align="center">
  <img src="/figures/02-foundations/source/06-basic-components-concise/pytorch-fig1.png" alt="PyTorch 的原始执行轨迹显示 Python 主机控制流如何提前排队 GPU 算子，直观体现 eager 执行的运行时行为。" width="86%">

_图 2.6-1：PyTorch 的原始执行轨迹显示 Python 主机控制流如何提前排队 GPU 算子，直观体现 eager 执行的运行时行为。 出处：Adam Paszke et al.，[PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)（2019），Figure 1。_

</div>

上一节用基础张量操作显式实现了数据批处理、参数初始化、前向传播、损失和 SGD。本节改用框架提供的标准模块完成同一任务，同时保留对形状、归约方式和梯度状态的检查。

我们依次使用数据迭代器、`nn.Linear`、损失函数和优化器，并说明每个封装隐藏了哪些数学步骤。

## 2.6.1 深度学习框架的演进与抽象哲学

Theano [[Bergstra et al., 2010]](https://doi.org/10.25080/Majora-92bf1922-003) 以符号表达式构建并编译计算图，Caffe [[Jia et al., 2014]](https://arxiv.org/abs/1408.5093) 则以网络层和配置驱动模型定义。这类设计便于统一优化执行计划，但对数据依赖控制流和交互式调试不够直接。

<div align="center">
  <img src="/figures/02-foundations/source/06-basic-components-concise/theano-fig2.png" alt="Theano 原论文用符号变量、损失与梯度更新代码展示先定义计算图再编译执行的静态工作流。" width="86%">

_图 2.6-2：Theano 原论文用符号变量、损失与梯度更新代码展示先定义计算图再编译执行的静态工作流。 出处：James Bergstra et al.，[Theano: A CPU and GPU Math Compiler in Python](https://www.iro.umontreal.ca/~lisa/pointeurs/theano_scipy2010.pdf)（2010），Figure 2。_

</div>

PyTorch 的 eager 执行允许普通 Python 控制流参与前向计算，并为实际执行的张量操作记录自动微分图。现代框架也常结合图捕获或编译优化，因此“静态”与“动态”更像一组实现权衡，而不是互斥阵营。

## 2.6.2 数据处理的管道化：从数组到迭代器

任何机器学习任务的起点都是数据。假设我们的数据集中包含了 $N$ 个样本，每个样本由特征 $\mathbf{x}_i \in \mathbb{R}^d$ 和标签 $y_i$ 组成。在理想情况下，我们希望在每一步迭代中，使用整个数据集来计算损失函数的精确梯度。这被称为批量梯度下降（Batch Gradient Descent）。假设我们优化的目标是最小化整个数据集上的平均损失：

$$
L(\mathbf{\theta}) = \frac{1}{N} \sum_{i=1}^N l(f_{\mathbf{\theta}}(\mathbf{x}_i), y_i)
$$

然而，在面对包含数百万样本的现代数据集时，直接计算全量数据集的梯度将导致显存溢出，且计算时间长得令人难以忍受。为了解决这一问题，我们将求和的范围缩小到一个大小为 $B$（通常为 $32$ 到 $256$ 之间）的随机子集，即小批量（Minibatch）：

$$
L_B(\mathbf{\theta}) = \frac{1}{B} \sum_{i \in \mathcal{B}} l(f_{\mathbf{\theta}}(\mathbf{x}_i), y_i)
$$

若 $\mathcal{B}$ 从数据集中均匀抽样，并对批内损失取正确平均，则小批量梯度是全量平均梯度的无偏估计。单个批次仍含采样噪声，并不保证每一步都沿全量损失的下降方向。

在框架中，数据的加载和批处理被抽象为了迭代器。我们只需要提供张量格式的数据，框架就会自动为我们处理打乱（Shuffle）和切片（Slicing）的逻辑。

<div align="center">
  <img src="/figures/02-foundations/source/06-basic-components-concise/caffe-fig1.png" alt="Caffe 的 MNIST 网络图把数据层、卷积与池化层、参数 blob 和损失层连接为显式训练管线。" width="86%">

_图 2.6-3：Caffe 的 MNIST 网络图把数据层、卷积与池化层、参数 blob 和损失层连接为显式训练管线。 出处：Yangqing Jia et al.，[Caffe: Convolutional Architecture for Fast Feature Embedding](https://arxiv.org/abs/1408.5093)（2014），Figure 1。_

</div>

先生成合成数据，并用 `TensorDataset` 与 `DataLoader` 组成数据管道。

```python
import torch
from torch import nn
from torch.utils import data

# 生成合成数据
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features = torch.randn(1000, 2)
labels = torch.matmul(features, true_w) + true_b
# 添加高斯噪声
labels += torch.randn(labels.shape) * 0.01

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

通过内置的迭代器，我们可以直接在 `for` 循环中使用 `data_iter`，它在每次产出 $10$ 个样本后自动推进，且在每次遍历完整个数据集（一个 Epoch）后，如果指定了 `is_train=True`，则会重新打乱索引。

## 2.6.3 线性层的封装：全连接神经网络的积木

接下来，我们需要定义模型本身。在最基础的标量场景中，一个线性映射仅仅是一个简单的初等代数公式：对于输入标量 $x$，权重为 $w$，偏置为 $b$，则输出 $y = wx + b$。

随着问题复杂度的提升，假设输入不再是单一的数值，而是描述某个物体属性的 $d$ 个特征向量（例如长度、宽度、重量等）。我们将输入升级为向量 $\mathbf{x} \in \mathbb{R}^d$。此时，我们需要 $d$ 个权重分量来分别评估每一个特征的重要性，即权重变为了向量 $\mathbf{w} \in \mathbb{R}^d$。此时线性关系变为两个向量的点积：

$$
y = \mathbf{w}^\top \mathbf{x} + b = \sum_{j=1}^d w_j x_j + b
$$

在现代神经网络中，我们不仅要同时处理多个特征，还要同时计算多个并行的输出神经元。如果我们的网络层接收 $d$ 个维度的输入，并投射到 $c$ 个维度的输出，那么我们将有 $c$ 个独立的偏置 $b_1, \dots, b_c$。权重则从向量演化为矩阵 $\mathbf{W} \in \mathbb{R}^{d \times c}$。在此情况下，对于单样本 $\mathbf{x} \in \mathbb{R}^{1 \times d}$，其线性映射变为：

$$
\mathbf{y} = \mathbf{x} \mathbf{W} + \mathbf{b}
$$

为了批量计算，把 $n$ 个样本堆叠成输入矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$。矩阵乘法后，输出 $\mathbf{Y}$ 的形状为 $n \times c$：

$$
\mathbf{Y} = \mathbf{X} \mathbf{W} + \mathbf{b}
$$

上式中，$\mathbf{X}\mathbf{W}$ 的形状为 $n \times c$，偏置 $\mathbf{b}$ 的形状为 $1 \times c$。框架通过**广播（Broadcasting）**在样本维复用同一个偏置；概念上可视作复制 $n$ 次，实际实现通常不需要真的分配一份完整副本。

在 PyTorch 中，上述整个维度的推导、权重的显存分配以及初始化逻辑，都被无缝封装在一个名为“全连接层”（Fully-Connected Layer）或“线性层”（Linear Layer）的抽象类中。

这里用 `nn.Sequential` 包装一个输入维度为 2、输出维度为 1 的 `nn.Linear`。

```python
# 构建模型：这里相当于 y = XW + b
# PyTorch 的 nn.Linear 第一个参数是输入特征数 d，第二个参数是输出特征数 c
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数，我们通常从特定方差的正态分布中抽取权重
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

## 2.6.4 损失函数的计算图抽象与数值稳定性

定义了模型输出后，我们需要通过损失函数来量化模型预测值与真实值之间的偏差。对于回归任务，最直接的度量方式是均方误差（Mean Squared Error, MSE）。框架提供了 `nn.MSELoss` 类，它内部会自动将计算过程注册到动态图中，为反向传播铺平道路。

然而，更具学术探讨价值的是分类任务中使用的损失函数。让我们稍微偏离回归任务，深入探讨一下深度学习中最容易踩坑的数值稳定性问题。在分类问题中，我们通常将线性层的原始输出向量 $\mathbf{o}$ 称为“逻辑值”（Logits）。为了将其转化为合法的概率分布 $\mathbf{\hat{y}}$，我们会使用 Softmax 操作：

$$
\hat{y}_j = \frac{\exp(o_j)}{\sum_{k=1}^c \exp(o_k)}
$$

然后，我们将预测概率代入交叉熵损失函数中：

$$
l(\mathbf{y}, \mathbf{\hat{y}}) = - \sum_{j=1}^c y_j \log \hat{y}_j
$$

把 Softmax 概率代入对数项可得：

$$
\log \hat{y}_j = o_j - \log \left( \sum_{k=1}^c \exp(o_k) \right)
$$

浮点数的表示范围有限。若某个 $o_k$ 很大，例如 $1000$，直接计算 $\exp(1000)$ 通常会溢出为 `Inf`，后续的除法或对数还可能产生 `NaN`。若所有 logit 都是很大的负数，指数项又可能下溢为 $0$，使 $\log(0)$ 变成负无穷。

常用实现会合并 Softmax 与交叉熵，并采用 **LogSumExp 技巧**。先取最大 logit $M = \max(o_k)$，再统一减去 $M$：

$$
\log \hat{y}_j = (o_j - M) - \log \left( \sum_{k=1}^c \exp(o_k - M) \right)
$$

<div align="center"><img src="/figures/02-foundations/latex/06-basic-components-concise/logsumexp-stabilization.png" alt="所有逻辑值减去最大值后，指数不超过一且指数和至少为一" width="86%">

_图 2.6-4：统一减去最大 logit 不改变 Softmax 比例，却把最大指数固定为 1，并保证求和项不会变成 0。_

</div>

变换后最大的指数项为 $\exp(0)=1$，其他指数项都不超过 $1$，从而避免这一处的指数上溢；求和项至少为 $1$，也不会在此处出现 $\log 0$。极小项下溢为零通常只表示其概率贡献可忽略。`nn.CrossEntropyLoss()` 直接接收未经过 Softmax 的 logits，并在内部使用数值稳定的组合计算。

而在本节的线性回归演示中，我们直接使用 MSE 损失。

在线性回归示例中，直接使用均方误差损失。

```python
loss = nn.MSELoss()
```

## 2.6.5 优化器的模块化：随机梯度下降的封装

最后，当我们通过反向传播计算得到损失函数关于所有模型参数的梯度 $\nabla_{\mathbf{\theta}} L(\mathbf{\theta})$ 时，我们需要按照一定的策略对参数进行迭代更新。在高中物理中，我们知道物体在势能曲面上会沿着最陡峭的方向（梯度的反方向）加速滑落。优化算法的核心思想正是在高维损失曲面上模拟这一滑落过程。

最经典的更新规则是小批量随机梯度下降（Stochastic Gradient Descent, SGD）：

$$
\mathbf{\theta}_{t} = \mathbf{\theta}_{t-1} - \eta \nabla_{\mathbf{\theta}} L_B(\mathbf{\theta}_{t-1})
$$

其中 $\eta$ 表示学习率（Learning Rate），控制参数更新尺度。过大时优化可能震荡或发散，过小时收敛会变慢。

在框架中，所有的参数矩阵（如 `net[0].weight` 和 `net[0].bias`）在内部都维护着两个连续的内存块：一个存储当前的参数值数据 `.data`，另一个存储刚刚计算出来的梯度信息 `.grad`。我们不再需要手动书写遍历和相减的代码。框架将整个更新法则和超参数管理器统筹在 `torch.optim` 模块中。我们只需将待优化的参数列表交由优化器对象托管即可。

下面实例化 `SGD` 优化器，并指定参数与学习率。

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

## 2.6.6 端到端的模型训练生命周期

现在把数据迭代器、网络、损失函数和 SGD 优化器组合成完整训练脚本。

在深层框架的视角下，每一次训练迭代（Step）都遵循着一段严格的生命周期逻辑，不可随意颠倒：

1. 从迭代器获取下一批特征 $\mathbf{X}$ 与标签 $\mathbf{y}$。
2. 调用 `net(X)` 构建前向计算图并得到预测值 $\mathbf{\hat{y}}$。
3. 计算标量损失值 `l = loss(y_hat, y)`。
4. **清理遗留状态**：在反向传播前，必须调用 `trainer.zero_grad()`。这是因为框架的底层设计为了支持复杂的循环神经网络等架构，默认会将计算出的梯度累加到原有梯度上，而不是替换。
5. **触发反向传播**：调用 `l.backward()`。此时框架的自动微分引擎会自顶向下遍历整个动态计算图，利用链式法则计算出所有叶子节点（即模型参数）的梯度，并将其填充到参数的 `.grad` 属性中。
6. **执行更新步**：调用 `trainer.step()`。优化器按设定的更新规则修改其托管参数。

完整训练代码如下。

```python
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    # 每个 epoch 结束后，在全量数据上评估当前损失
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

高级 API 虽然简洁，但没有改变底层数学：线性层仍是矩阵乘法与广播，交叉熵仍需稳定地计算 LogSumExp，自动微分仍按计算图应用链式法则。掌握这些对应关系，才能在出现形状、数值或梯度问题时判断封装内部发生了什么。

## 2.6.7 练习

1. 如果我们将本节中用于生成合成数据的真实标签权重从一个二维向量变为形状为 $2 \times 3$ 的矩阵，我们需要对数据迭代器、模型维度和损失函数的形状进行哪些改动？
   - **提示**：回顾小批量矩阵乘法的维度。当输出维度 $c = 3$ 时，偏置向量 $\mathbf{b}$ 在广播时的维度是什么？
2. 试着在训练循环内部，将 `trainer.zero_grad()` 这一行注释掉。观察几轮训练后的损失值变化，并用微积分链式法则和梯度累加的机制解释这一现象。
3. 在 `nn.MSELoss` 的文档中存在一个名为 `reduction` 的参数。探索如果将其从默认的 `'mean'` 改为 `'sum'`，为了保证收敛，我们需要对优化器的学习率做何种数值量级上的等比例缩放？
