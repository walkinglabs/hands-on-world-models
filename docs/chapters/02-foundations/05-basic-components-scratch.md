# 基础模块的从零开始实现
:label:`sec_basic_components_scratch`

在前面的章节中，我们已经探讨了深度学习的数学基础与自动微分的原理。然而，现代深度学习框架（如PyTorch或TensorFlow）的高度封装，往往会掩盖模型内部的数学本质与工程细节。为了真正掌握深度学习的核心机制，我们必须剥开这些高级API的外衣。

在本节中，我们将回到深度学习的黎明时期。受神经科学先驱麦卡洛克和皮茨 (McCulloch & Pitts) 在1943年提出的逻辑演算神经网络模型 `[McCulloch et al., 1943]` 以及罗森布拉特 (Rosenblatt) 于1958年发明的感知机 (Perceptron) `[Rosenblatt, 1958]` 的启发，我们将从最底层的张量运算开始，一步步构建出神经网络的核心基础模块。

我们将不依赖任何框架的高层API（如 `torch.nn`），仅使用基础的张量运算与自动微分功能，从零开始实现线性层（全连接层）、非线性激活函数、损失函数以及优化算法。这种“造轮子”的硬核过程，是深刻理解世界模型（World Models）中复杂架构（如Transformer和状态空间模型）不可或缺的基石。

## 线性层：仿射变换的几何与代数
:label:`subsec_linear_layer_scratch`

神经网络中最基础的积木是线性层（Linear Layer），也称为全连接层（Fully Connected Layer）或稠密层（Dense Layer）。在多层感知机（MLP）提出之前，简单的线性分类器就是由单层线性层构成的。

### 标量场景下的线性变换

在高中物理中，我们经常遇到简单的线性关系。例如，弹簧的形变量 $x$ 与弹力 $F$ 之间的关系可以用胡克定律表示为 $F = kx$。如果我们考虑一个初始存在的静摩擦力或预设的弹力偏差 $b$，该关系可以扩展为一元一次方程：

$$
y = w x + b
$$
:eqlabel:`eq_scalar_linear`

其中，$x$ 是输入，$w$ 是权重（斜率），$b$ 是偏置（截距），$y$ 是输出。这个方程描述了一个一维空间中的仿射变换（Affine Transformation）：先进行缩放（乘以 $w$），再进行平移（加上 $b$）。

### 向量与矩阵：高维空间的仿射变换

在机器学习中，我们的输入往往不是单一的标量，而是一个包含多个特征的向量。假设我们正在预测房屋的价格，输入特征可能包括房屋面积、卧室数量、房龄等 $d$ 个特征。我们可以将这些特征表示为一个列向量 $\mathbf{x} \in \mathbb{R}^{d \times 1}$。

此时，我们需要为每一个特征分配一个权重。设权重向量为 $\mathbf{w} \in \mathbb{R}^{d \times 1}$，偏置为一个标量 $b$。那么，输出的标量 $y$ 可以通过向量的点积（内积）来计算：

$$
y = \mathbf{w}^\top \mathbf{x} + b = \sum_{i=1}^d w_i x_i + b
$$
:eqlabel:`eq_vector_linear`

这仅仅是单个输出节点的计算。在神经网络的隐藏层中，我们通常需要计算多个输出节点。假设我们希望将 $d$ 维的输入向量映射到 $q$ 维的输出向量 $\mathbf{y} \in \mathbb{R}^{q \times 1}$。对于每一个输出节点 $j \in \{1, 2, \ldots, q\}$，我们需要一个独立的权重向量 $\mathbf{w}_j$ 和一个独立的偏置 $b_j$：

$$
y_j = \mathbf{w}_j^\top \mathbf{x} + b_j
$$

为了极大地提升计算效率并保持数学表达的简洁性，我们利用线性代数中的矩阵乘法。我们将所有的权重向量 $\mathbf{w}_j$ 按行排列，组合成一个权重矩阵 $\mathbf{W} \in \mathbb{R}^{q \times d}$。将所有的偏置 $b_j$ 组合成一个偏置向量 $\mathbf{b} \in \mathbb{R}^{q \times 1}$。

此时，整个层的计算可以严谨地表示为矩阵向量乘法：

$$
\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b}
$$
:eqlabel:`eq_matrix_vector_linear`

### 小批量计算的张量扩展

在实际的深度学习训练中，为了利用现代GPU的并行计算能力并降低梯度估计的方差，我们几乎不会一次只处理一个样本，而是同时处理一个小批量（Minibatch）的数据。

设批量大小为 $n$。我们将 $n$ 个输入样本作为行向量，堆叠成一个特征矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$。为了匹配维度，此时我们通常将权重矩阵的形状转置为 $\mathbf{W} \in \mathbb{R}^{d \times q}$，偏置向量为 $\mathbf{b} \in \mathbb{R}^{1 \times q}$。

此时，前向传播的最终张量公式变为：

$$
\mathbf{Y} = \mathbf{X} \mathbf{W} + \mathbf{b}
$$
:eqlabel:`eq_minibatch_linear`

这里，$\mathbf{X} \mathbf{W}$ 的结果维度是 $n \times q$。根据线性代数规则，直接将 $n \times q$ 的矩阵与 $1 \times q$ 的行向量相加在数学上是未定义的。但在计算机张量运算中，这里触发了**广播机制**（Broadcasting）：偏置向量 $\mathbf{b}$ 会被隐式地复制 $n$ 次，以匹配矩阵的形状，从而实现对每一个样本的输出都加上相同的偏置。

(**下面，我们用代码实现这个核心的线性层计算过程。**) 我们需要生成标准正态分布的随机数来初始化权重，并将偏置初始化为零。

```{.python .input}
#@tab pytorch
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
:label:`subsec_activation_scratch`

仅仅依靠线性层是远远不够的。根据线性代数的性质，无论我们堆叠多少个线性层，多层仿射变换的组合最终仍然等价于一个单一的仿射变换。即：

$$
\mathbf{Y} = (\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2 = \mathbf{X} (\mathbf{W}_1 \mathbf{W}_2) + (\mathbf{b}_1 \mathbf{W}_2 + \mathbf{b}_2) = \mathbf{X} \mathbf{W}' + \mathbf{b}'
$$

这意味着，如果没有非线性介入，深度神经网络将退化为最简单的线性模型，无法拟合现实世界中复杂的非线性规律。为了打破这种线性退化，我们必须在每个线性层的输出之后，施加一个非线性函数，即**激活函数**（Activation Function）。

经典的激活函数包括 Sigmoid 函数，它在早期神经网络中被广泛使用，其数学表达式为：

$$
\sigma(x) = \frac{1}{1 + \exp(-x)}
$$
:eqlabel:`eq_sigmoid`

然而，在反向传播 `[Rumelhart et al., 1986]` 过程中，当输入 $x$ 的绝对值较大时，Sigmoid 函数的梯度会迅速趋近于零，导致“梯度消失”（Vanishing Gradient）问题。

为了解决这一问题，Nair和Hinton在2010年将其引入深度学习的修正线性单元 (Rectified Linear Unit, ReLU) `[Nair & Hinton, 2010]` 成为目前的标准选择。ReLU 的数学定义极其简单，即保留正数，将负数截断为零：

$$
\text{ReLU}(x) = \max(x, 0)
$$
:eqlabel:`eq_relu`

ReLU 不仅计算极快（仅需比较操作），而且在正半轴上的梯度恒为 1，极大缓解了梯度消失问题。

(**让我们从零开始实现 ReLU 激活函数。**) 我们可以利用张量的比较与掩码操作来完成。

```{.python .input}
#@tab pytorch
def relu(X):
    """
    实现ReLU激活函数。
    通过创建一个形状相同且全为0的张量，然后取两者的元素级最大值。
    """
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

## 损失函数：量化模型的误差
:label:`subsec_loss_scratch`

模型能够输出预测值后，我们需要一种机制来评估预测值 $\hat{\mathbf{y}}$ 与真实标签 $\mathbf{y}$ 之间的差距。这个衡量差距的函数被称为**损失函数** (Loss Function) 或目标函数 (Objective Function)。

### 均方误差 (Mean Squared Error, MSE)

在回归任务（如预测连续的房价）中，最常用的损失函数是均方误差。它的思想可以追溯到高斯 (Gauss) 和勒让德 (Legendre) 在18世纪末发展的最小二乘法 (Method of Least Squares)。

对于第 $i$ 个样本，预测值为 $\hat{y}^{(i)}$，真实值为 $y^{(i)}$。我们将这两者的差值称为残差 (Residual)。平方损失定义为残差平方的一半：

$$
l^{(i)} = \frac{1}{2} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$
:eqlabel:`eq_squared_loss`

公式中常数 $\frac{1}{2}$ 的作用是，在对损失函数求导时，二次项的系数 2 会与 $\frac{1}{2}$ 抵消，使得梯度的数学表达式更加简洁。

对于包含 $n$ 个样本的整个小批量数据，均方误差是所有单个样本损失的平均值：

$$
L(\mathbf{W}, \mathbf{b}) = \frac{1}{n} \sum_{i=1}^n \frac{1}{2} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$
:eqlabel:`eq_mse`

(**接下来，我们实现平方损失函数。**) 在实现中，我们需要确保预测张量 $\hat{\mathbf{y}}$ 和真实标签张量 $\mathbf{y}$ 的形状完全一致，避免由于广播机制导致意外的矩阵扩展。

```{.python .input}
#@tab pytorch
def squared_loss(y_hat, y):
    """
    实现均方损失函数。
    返回的形状与 y_hat 和 y 相同，即逐样本的损失，尚未进行均值操作。
    """
    # 将真实标签的形状转换为预测值的形状以确保匹配
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

### 交叉熵损失 (Cross-Entropy Loss)

尽管我们在上面的代码中实现了均方误差，但在世界模型（如自然语言处理中的语言模型）通常涉及的分类任务中，均方误差并不合适。分类问题需要输出每个类别的概率分布，此时更具有信息论背景的**交叉熵损失** `[Shannon, 1948]` 是首选。

> 💡 **精炼类比**：在信息论中，交叉熵衡量的是：当我们用一个错误的分布 $Q$（模型的预测概率）来编码一个真实分布 $P$（真实标签的一热编码）时，所需要的额外比特数。模型预测越准，额外开销越小，损失越接近于零。

设真实的类别分布为 $\mathbf{y} \in \{0,1\}^q$（通常为一热编码格式），模型预测的概率分布为 $\hat{\mathbf{y}} \in (0,1)^q$，且满足 $\sum_j \hat{y}_j = 1$。单个样本的交叉熵损失严谨地定义为：

$$
l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j
$$
:eqlabel:`eq_cross_entropy`

在世界模型的基础架构讲解中，我们将在后续关于 Softmax 层的章节深入探讨交叉熵的数值稳定实现，目前我们暂且聚焦于基础张量运算体系的跑通。

## 优化算法：小批量随机梯度下降 (Minibatch SGD)
:label:`subsec_optimization_scratch`

有了模型（前向传播）和损失函数（评估误差），接下来的核心任务是：如何调整模型参数（权重 $\mathbf{W}$ 和偏置 $\mathbf{b}$）以最小化损失函数？

由于深度神经网络的损失函数通常是高度非凸的高维曲面，我们无法像高中求二次函数顶点那样直接通过令导数为零求得解析解 (Analytical Solution)。相反，我们必须采用数值优化算法 (Numerical Optimization)。

其中最基础也是最经典的算法是**小批量随机梯度下降** (Minibatch Stochastic Gradient Descent, SGD) `[Robbins & Monro, 1951]`。

### 梯度的几何意义与下降方向

微积分告诉我们，对于一个多元函数 $L(\mathbf{W})$，它在某一点的梯度 $\nabla_{\mathbf{W}} L$ 指向该函数值增加最快的方向。因此，为了最小化损失函数，我们应该沿着梯度的**反方向**迈出一步。

每一次参数更新的严谨数学表达式如下：

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{1}{n} \sum_{i=1}^n \nabla_{\mathbf{W}} l^{(i)}(\mathbf{W}, \mathbf{b})
$$
:eqlabel:`eq_sgd_w`

$$
\mathbf{b} \leftarrow \mathbf{b} - \eta \frac{1}{n} \sum_{i=1}^n \nabla_{\mathbf{b}} l^{(i)}(\mathbf{W}, \mathbf{b})
$$
:eqlabel:`eq_sgd_b`

这里的物理量解释极其重要：
- $\leftarrow$ 表示赋值操作，即用更新后的值替换当前值。
- $\frac{1}{n} \sum_{i=1}^n \nabla l^{(i)}$ 是当前小批量数据上计算出的**平均梯度**。使用平均梯度而不是总梯度，可以使得学习率的选择与批量大小解耦，避免在改变批量大小时需要大幅调整学习率。
- $\eta$（读作 eta）是**学习率** (Learning Rate)，一个极其关键的超参数。它控制了我们沿着梯度反方向迈出的一步有多大。若 $\eta$ 过大，模型可能在最优解附近剧烈震荡甚至发散；若 $\eta$ 过小，模型的收敛速度将慢如蜗牛。

(**接下来，我们实现小批量随机梯度下降算法。**) 在这个函数中，我们传入包含模型所有参数的列表、学习率以及批量大小。需要特别注意的是，在PyTorch中，参数更新过程不应被计算图记录，因此必须在 `torch.no_grad()` 上下文中进行。同时，更新完成后必须将参数的梯度清零，否则梯度会在后续的迭代中累加。

```{.python .input}
#@tab pytorch
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
:label:`subsec_training_loop_scratch`

现在，所有的基础零部件都已经准备就绪：我们拥有了线性模型 `linreg`、损失函数 `squared_loss` 以及优化器 `sgd`。接下来，我们需要编写一个**训练循环** (Training Loop)，将这些模块精密地咬合在一起，驱动模型开始学习。

训练循环的每一次迭代 (Epoch) 通常包含以下严格的步骤：
1. **数据抽取**：从小批量数据生成器中取出一批特征 $\mathbf{X}$ 和对应的标签 $\mathbf{y}$。
2. **前向传播**：将 $\mathbf{X}$ 输入模型，计算出预测值 $\hat{\mathbf{y}}$。
3. **计算损失**：使用损失函数计算 $\hat{\mathbf{y}}$ 与 $\mathbf{y}$ 之间的误差。
4. **反向传播**：对损失张量调用自动微分系统（如PyTorch的 `.backward()`），计算所有参数的梯度。
5. **参数更新**：调用优化器算法，根据梯度更新参数。

这就是深度学习模型学习的全过程。无论是本节中简单的线性模型，还是包含千亿参数的大型世界模型，其底层的训练逻辑都严格遵循这五个步骤。

由于我们强调“从零开始”，读者现在应该对每一个操作背后的数学张量流转有了极度清晰的认知。理解这些底层的数学方程与代码实现之间的映射关系，将极大提升未来排查复杂模型bug的能力。

## 小结

- 我们追溯了早期神经网络的历史，并使用严格的矩阵乘法构建了多维输入下的线性层的前向计算公式 $\mathbf{Y} = \mathbf{X} \mathbf{W} + \mathbf{b}$。
- 为了赋予模型非线性表达能力，我们引入了ReLU激活函数，打破了纯线性变换的局限。
- 我们使用均方误差量化了回归问题的预测偏差，并讨论了交叉熵在分类任务中的物理意义。
- 我们通过微积分推导了小批量随机梯度下降 (SGD) 的更新公式，并手工编写了参数的梯度更新与清零逻辑。
- 这些从零开始构建的基础组件，构成了任何复杂深度学习架构（包括现代生成式世界模型）最核心的基因。

## 练习

1. 在 `sgd` 函数中，为什么我们需要使用 `param -= lr * param.grad / batch_size` 而不是直接乘以学习率而不除以 `batch_size`？
   - *提示*：回想我们在公式 :eqref:`eq_sgd_w` 中计算的是“平均梯度”还是“总和梯度”，并思考这与损失函数 `squared_loss` 中未做均值处理的联系。
2. 如果我们将激活函数 ReLU 替换为线性函数 $f(x) = c \cdot x$（$c$ 为常数），整个包含两层隐藏层的多层感知机能否拟合非线性数据？
   - *提示*：尝试在草稿纸上写出 $\mathbf{Y} = c (\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$，看看最终的代数形式是否依然是一个单一的仿射变换。
3. 在代码实现中，如果我们忘记在 `sgd` 函数的末尾调用 `param.grad.zero_()`，会发生什么？
   - *提示*：查阅自动微分章节中关于梯度累加机制的设计初衷。

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
