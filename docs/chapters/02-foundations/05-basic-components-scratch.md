# 2.5 基础模块的从零开始实现

PyTorch 的高层模块能省去大量样板代码，但也容易让张量形状、损失归约和参数更新变得不易察觉。本节暂时不用 `torch.nn`，只依赖基础张量运算与自动微分，把训练过程中最关键的几步逐一写出来。

麦卡洛克和皮茨在 1943 年提出逻辑神经元模型 [[McCulloch & Pitts, 1943]](https://doi.org/10.1007/BF02478259)，罗森布拉特随后发展了感知机 [[Rosenblatt, 1958]](https://doi.org/10.1037/h0042519)。沿着“输入加权、非线性变换、按误差更新”的主线，我们从线性层开始组装一个最小训练系统。

<div align="center">
  <img src="/figures/02-foundations/source/05-basic-components-scratch/rosenblatt-fig1.png" alt="Rosenblatt 的感知机组织图把感知单元、联结区与响应单元串成早期可学习神经系统。" width="86%">

_图 2.5-1：Rosenblatt 的感知机组织图把感知单元、联结区与响应单元串成早期可学习神经系统。 出处：Frank Rosenblatt，[The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://doi.org/10.1037/h0042519)（1958），Figure 1。_

</div>

我们会实现线性层、激活函数、损失函数和小批量 SGD。目标不是替代框架，而是把公式中的每个量与代码中的张量对应起来，为后续排查复杂模型的形状和梯度问题打基础。

## 线性层：仿射变换的几何与代数

神经网络中最基础的积木是线性层（Linear Layer），也称为全连接层（Fully Connected Layer）或稠密层（Dense Layer）。在多层感知机（MLP）提出之前，简单的线性分类器就是由单层线性层构成的。

### 标量场景下的线性变换

在高中物理中，我们经常遇到简单的线性关系。例如，弹簧的形变量 $x$ 与弹力 $F$ 之间的关系可以用胡克定律表示为 $F = kx$。如果我们考虑一个初始存在的静摩擦力或预设的弹力偏差 $b$，该关系可以扩展为一元一次方程：

$$
y = w x + b
$$

其中，$x$ 是输入，$w$ 是权重（斜率），$b$ 是偏置（截距），$y$ 是输出。这个方程描述了一个一维空间中的仿射变换（Affine Transformation）：先进行缩放（乘以 $w$），再进行平移（加上 $b$）。

### 向量与矩阵：高维空间的仿射变换

在机器学习中，我们的输入往往不是单一的标量，而是一个包含多个特征的向量。假设我们正在预测房屋的价格，输入特征可能包括房屋面积、卧室数量、房龄等 $d$ 个特征。我们可以将这些特征表示为一个列向量 $\mathbf{x} \in \mathbb{R}^{d \times 1}$。

此时，我们需要为每一个特征分配一个权重。设权重向量为 $\mathbf{w} \in \mathbb{R}^{d \times 1}$，偏置为一个标量 $b$。那么，输出的标量 $y$ 可以通过向量的点积（内积）来计算：

$$
y = \mathbf{w}^\top \mathbf{x} + b = \sum_{i=1}^d w_i x_i + b
$$

这仅仅是单个输出节点的计算。在神经网络的隐藏层中，我们通常需要计算多个输出节点。假设我们希望将 $d$ 维的输入向量映射到 $q$ 维的输出向量 $\mathbf{y} \in \mathbb{R}^{q \times 1}$。对于每一个输出节点 $j \in \{1, 2, \ldots, q\}$，我们需要一个独立的权重向量 $\mathbf{w}_j$ 和一个独立的偏置 $b_j$：

$$
y_j = \mathbf{w}_j^\top \mathbf{x} + b_j
$$

为了同时计算多个输出，把所有权重向量 $\mathbf{w}_j$ 按行排成矩阵 $\mathbf{W} \in \mathbb{R}^{q \times d}$，并把偏置组成向量 $\mathbf{b} \in \mathbb{R}^{q \times 1}$。

此时，整个层的计算可以严谨地表示为矩阵向量乘法：

$$
\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b}
$$

### 小批量计算的张量扩展

在实际的深度学习训练中，为了利用现代GPU的并行计算能力并降低梯度估计的方差，我们几乎不会一次只处理一个样本，而是同时处理一个小批量（Minibatch）的数据。

设批量大小为 $n$。我们将 $n$ 个输入样本作为行向量，堆叠成一个特征矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$。为了匹配维度，此时我们通常将权重矩阵的形状转置为 $\mathbf{W} \in \mathbb{R}^{d \times q}$，偏置向量为 $\mathbf{b} \in \mathbb{R}^{1 \times q}$。

此时，前向传播的最终张量公式变为：

$$
\mathbf{Y} = \mathbf{X} \mathbf{W} + \mathbf{b}
$$

<div align="center"><img src="/figures/02-foundations/latex/05-basic-components-scratch/bias-broadcast-shapes.png" alt="矩阵乘积得到 n 乘 q 输出，单行偏置沿批次维复制 n 次后逐元素相加" width="86%">

_图 2.5-2：XW 产生 n×q 的小批量输出；偏置的单行沿样本维复用 n 次，使每个样本加上同一组 q 维偏置。本文根据上式绘制。_

</div>

这里，$\mathbf{X} \mathbf{W}$ 的结果维度是 $n \times q$。根据线性代数规则，直接将 $n \times q$ 的矩阵与 $1 \times q$ 的行向量相加在数学上是未定义的。但在计算机张量运算中，这里触发了**广播机制**（Broadcasting）：偏置向量 $\mathbf{b}$ 会被隐式地复制 $n$ 次，以匹配矩阵的形状，从而实现对每一个样本的输出都加上相同的偏置。

下面实现线性层的前向计算。权重可用小幅随机数初始化，偏置初始化为零。

```python
import torch

def linreg(X, w, b):
    """
    实现线性回归模型的正向传播。
    X: 形状为 (批量大小, 输入维度) 的张量
    w: 形状为 (输入维度, 输出维度) 的张量
    b: 形状为 (1, 输出维度) 的张量
    """
    return torch.matmul(X, w) + b
```

## 激活函数：引入非线性

仅仅依靠线性层是远远不够的。根据线性代数的性质，无论我们堆叠多少个线性层，多层仿射变换的组合最终仍然等价于一个单一的仿射变换。即：

$$
\mathbf{Y} = (\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2 = \mathbf{X} (\mathbf{W}_1 \mathbf{W}_2) + (\mathbf{b}_1 \mathbf{W}_2 + \mathbf{b}_2) = \mathbf{X} \mathbf{W}' + \mathbf{b}'
$$

这意味着，如果网络只有仿射层，多层组合仍是一个仿射映射，无法表示一般的非线性关系。通常需要在隐藏层之间加入**激活函数**（Activation Function）；输出层是否需要激活，则取决于任务和损失函数。

经典的激活函数包括 Sigmoid 函数，它在早期神经网络中被广泛使用，其数学表达式为：

$$
\sigma(x) = \frac{1}{1 + \exp(-x)}
$$

Rumelhart 等人的工作系统展示了如何用反向传播训练多层网络 [[Rumelhart et al., 1986]](https://doi.org/10.1038/323533a0)。对于 Sigmoid 函数，当输入 $x$ 的绝对值较大时，导数会迅速趋近于零；在深层网络中反复相乘后，这会造成“梯度消失”（Vanishing Gradient）。

<div align="center">
  <img src="/figures/02-foundations/source/05-basic-components-scratch/backprop-fig1.png" alt="反向传播原论文的网络与内部表示图展示误差信号如何训练隐藏单元形成任务相关表征。" width="86%">

_图 2.5-3：反向传播原论文的网络与内部表示图展示误差信号如何训练隐藏单元形成任务相关表征。 出处：David E. Rumelhart; Geoffrey E. Hinton; Ronald J. Williams，[Learning Representations by Back-Propagating Errors](https://doi.org/10.1038/323533a0)（1986），Figure 1。_

</div>

Nair 和 Hinton 在受限玻尔兹曼机中展示了修正线性单元（Rectified Linear Unit, ReLU）的效果 [[Nair & Hinton, 2010]](https://icml.cc/Conferences/2010/papers/432.pdf)。此后，ReLU 成为深度网络中常用的激活函数之一。它的数学定义很简单：保留正数，将负数截断为零。

<div align="center">
  <img src="/figures/02-foundations/source/05-basic-components-scratch/relu-fig1.png" alt="ReLU 论文比较二元单元期望、softplus 近似与噪声整流响应，说明整流非线性的建模来源。" width="86%">

_图 2.5-4：ReLU 论文比较二元单元期望、softplus 近似与噪声整流响应，说明整流非线性的建模来源。 出处：Vinod Nair; Geoffrey E. Hinton，[Rectified Linear Units Improve Restricted Boltzmann Machines](https://icml.cc/Conferences/2010/papers/432.pdf)（2010），Figure 1。_

</div>

$$
\text{ReLU}(x) = \max(x, 0)
$$

ReLU 不仅计算极快（仅需比较操作），而且在正半轴上的梯度恒为 1，极大缓解了梯度消失问题。

下面利用逐元素最大值实现 ReLU。

```python
def relu(X):
    """
    实现ReLU激活函数。
    通过创建一个形状相同且全为0的张量，然后取两者的元素级最大值。
    """
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

## 损失函数：量化模型的误差

模型能够输出预测值后，我们需要一种机制来评估预测值 $\hat{\mathbf{y}}$ 与真实标签 $\mathbf{y}$ 之间的差距。这个衡量差距的函数被称为**损失函数** (Loss Function) 或目标函数 (Objective Function)。

### 均方误差 (Mean Squared Error, MSE)

在回归任务（如预测连续的房价）中，最常用的损失函数是均方误差。它的思想可以追溯到高斯 (Gauss) 和勒让德 (Legendre) 在18世纪末发展的最小二乘法 (Method of Least Squares)。

对于第 $i$ 个样本，预测值为 $\hat{y}^{(i)}$，真实值为 $y^{(i)}$。我们将这两者的差值称为残差 (Residual)。平方损失定义为残差平方的一半：

$$
l^{(i)} = \frac{1}{2} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

公式中常数 $\frac{1}{2}$ 的作用是，在对损失函数求导时，二次项的系数 2 会与 $\frac{1}{2}$ 抵消，使得梯度的数学表达式更加简洁。

对于包含 $n$ 个样本的整个小批量数据，均方误差是所有单个样本损失的平均值：

$$
L(\mathbf{W}, \mathbf{b}) = \frac{1}{n} \sum_{i=1}^n \frac{1}{2} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

下面实现逐样本平方损失。预测张量 $\hat{\mathbf{y}}$ 与标签 $\mathbf{y}$ 的形状必须一致，以免广播产生意外的矩阵扩展。

```python
def squared_loss(y_hat, y):
    """
    实现均方损失函数。
    返回的形状与 y_hat 和 y 相同，即逐样本的损失，尚未进行均值操作。
    """
    # 将真实标签的形状转换为预测值的形状以确保匹配
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

### 交叉熵损失 (Cross-Entropy Loss)

尽管我们在上面的代码中实现了均方误差，但分类任务通常需要输出各类别的概率分布，此时常用**交叉熵损失**。它建立在 Shannon 信息论中的熵与编码思想之上 [[Shannon, 1948]](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)，但把交叉熵用作分类目标是后续统计学习实践的发展。

> 💡 **精炼类比**：在信息论中，交叉熵衡量的是：当我们用一个错误的分布 $Q$（模型的预测概率）来编码一个真实分布 $P$（真实标签的一热编码）时，所需要的额外比特数。模型预测越准，额外开销越小，损失越接近于零。

设真实的类别分布为 $\mathbf{y} \in \{0,1\}^q$（通常为一热编码格式），模型预测的概率分布为 $\hat{\mathbf{y}} \in (0,1)^q$，且满足 $\sum_j \hat{y}_j = 1$。单个样本的交叉熵损失严谨地定义为：

$$
l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j
$$

在世界模型的基础架构讲解中，我们将在后续关于 Softmax 层的章节深入探讨交叉熵的数值稳定实现，目前我们暂且聚焦于基础张量运算体系的跑通。

## 优化算法：小批量随机梯度下降 (Minibatch SGD)

有了模型（前向传播）和损失函数（评估误差），接下来的核心任务是：如何调整模型参数（权重 $\mathbf{W}$ 和偏置 $\mathbf{b}$）以最小化损失函数？

由于深度神经网络的损失函数通常是高度非凸的高维曲面，我们无法像高中求二次函数顶点那样直接通过令导数为零求得解析解 (Analytical Solution)。相反，我们必须采用数值优化算法 (Numerical Optimization)。

随机梯度方法可追溯到 Robbins 和 Monro 的随机逼近框架 [[Robbins & Monro, 1951]](https://doi.org/10.1214/aoms/1177729586)。现代深度学习通常把若干样本组成小批量，用其平均梯度更新参数，这就是**小批量随机梯度下降**（Minibatch Stochastic Gradient Descent, SGD）。

### 梯度的几何意义与下降方向

微积分告诉我们，对于一个多元函数 $L(\mathbf{W})$，它在某一点的梯度 $\nabla_{\mathbf{W}} L$ 指向该函数值增加最快的方向。因此，为了最小化损失函数，我们应该沿着梯度的**反方向**迈出一步。

每一次参数更新的严谨数学表达式如下：

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{1}{n} \sum_{i=1}^n \nabla_{\mathbf{W}} l^{(i)}(\mathbf{W}, \mathbf{b})
$$

$$
\mathbf{b} \leftarrow \mathbf{b} - \eta \frac{1}{n} \sum_{i=1}^n \nabla_{\mathbf{b}} l^{(i)}(\mathbf{W}, \mathbf{b})
$$

各符号的含义如下：

- $\leftarrow$ 表示赋值操作，即用更新后的值替换当前值。
- $\frac{1}{n} \sum_{i=1}^n \nabla l^{(i)}$ 是当前小批量数据上计算出的**平均梯度**。使用平均梯度而不是总梯度，可以使得学习率的选择与批量大小解耦，避免在改变批量大小时需要大幅调整学习率。
- $\eta$（读作 eta）是**学习率** (Learning Rate)，控制每次参数更新的尺度。过大时优化可能震荡或发散，过小时收敛会很慢。

下面实现小批量 SGD。由于 `squared_loss` 返回逐样本损失，训练时若对损失调用 `.sum().backward()`，梯度就是小批量总和，因此更新时除以 `batch_size` 得到平均梯度。参数更新不应写入计算图，更新后还要清空累计梯度。

```python
def sgd(params, lr, batch_size):
    """
    实现小批量随机梯度下降算法。
    params: 包含参数张量的列表，如 [W, b]
    lr: 学习率
    batch_size: 批量大小
    """
    with torch.no_grad():
        for param in params:
            # 执行梯度下降更新
            param -= lr * param.grad / batch_size
            # 清除梯度，为下一次反向传播做准备
            param.grad.zero_()
```

## 组装一切：训练循环的解剖

有了 `linreg`、`squared_loss` 和 `sgd`，就可以写出一个最小**训练循环** (Training Loop)。

一次参数更新通常包含以下步骤：

1. **数据抽取**：从小批量数据生成器中取出一批特征 $\mathbf{X}$ 和对应的标签 $\mathbf{y}$。
2. **前向传播**：将 $\mathbf{X}$ 输入模型，计算出预测值 $\hat{\mathbf{y}}$。
3. **计算损失**：使用损失函数计算 $\hat{\mathbf{y}}$ 与 $\mathbf{y}$ 之间的误差。
4. **反向传播**：对损失张量调用自动微分系统（如PyTorch的 `.backward()`），计算所有参数的梯度。
5. **参数更新**：调用优化器算法，根据梯度更新参数。

大多数监督学习训练循环都能归纳为这几步，但具体顺序会因梯度累积、混合精度、多个优化器或强化学习的数据收集过程而变化。

把公式与张量形状逐项对齐后，广播错误、重复平均和忘记清空梯度等常见问题会更容易定位。

## 小结

- 我们追溯了早期神经网络的历史，并使用严格的矩阵乘法构建了多维输入下的**线性层**的前向计算公式 $\mathbf{Y} = \mathbf{X} \mathbf{W} + \mathbf{b}$。
- 为了赋予模型非线性表达能力，我们引入了 **ReLU 激活函数**，打破了纯线性变换的局限。
- 我们使用均方误差量化了回归问题的预测偏差，并讨论了交叉熵在分类任务中的物理意义。
- 我们通过微积分推导了**小批量随机梯度下降 (SGD)** 的更新公式，并手工编写了参数的梯度更新与清零逻辑。
- 这些基础组件会以更高层的封装反复出现在生成模型与世界模型的训练代码中。
