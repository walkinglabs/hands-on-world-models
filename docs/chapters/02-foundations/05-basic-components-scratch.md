# 2.5 从零实现基础深度学习组件 (Basic Components from Scratch)

在现代深度学习框架（如 PyTorch、TensorFlow）高度封装的今天，我们只需调用 `nn.Linear`、`F.cross_entropy` 或 `loss.backward()` 就能在几行代码内完成一个神经网络的构建与训练。

然而，对于致力于探索具身智能与世界模型的研究者而言，如果仅仅将这些核心组件视为黑盒调用，当遇到模型训练不收敛、梯度爆炸、数值下溢（Underflow）或需要为特定硬件定制加速算子时，就会陷入束手无策的困境。

深度学习的底层并没有不可捉摸的黑魔法——它的本质就是基础矩阵代数、多元复合函数微积分求导与张量广播机制的严密协同。

本节我们将彻底抛弃 PyTorch 高级 API，从纯数学定义出发，从零推导全连接层（Dense）、激活函数与交叉熵损失的矩阵求导公式，并使用纯底层张量操作手动实现前向传播与反向传播（Backward）引擎。

<div align="center">

<img src="/figures/02-foundations/source/05-basic-components-scratch/relu-fig1.png" alt="多层感知机 (MLP) 的网络连接拓扑：输入层、隐藏层与输出层之间的全连接权重矩阵与偏置。" width="86%">

_图 2.5-1：多层感知机 (MLP) 的网络连接拓扑：输入层、隐藏层与输出层之间的全连接权重矩阵与偏置。 出处：[Dive into Deep Learning，Aston Zhang et al.，2023](https://d2l.ai/)。_

</div>

---

## 2.5.1 物理与微积分基石：计算图与多元链式求导法则

要理解自动微分的底层运作，我们首先必须审视基础代数函数构成的计算图（Computation Graph）。

### 1. 经典多元微积分链式法则（Chain Rule）
设复合函数为 $z = f(y)$，其中 $y = g(x)$。微积分一阶导数满足：

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

在由数千个矩阵相乘串联而成的深度网络中，损失标量 $\mathcal{L}$ 对任意中间参数矩阵 $\mathbf{W}$ 的梯度，就是上游回传的损失误差梯度与当前层局部雅可比矩阵的连续矩阵乘积。

### 2. 张量维度的空间对齐准则
在手动推导矩阵求导时，初学者最容易在转置与维度对齐上迷失。
请牢记一条不可动摇的**初等代数守恒法则**：
- 如果权重矩阵 $\mathbf{W}$ 的形状是 $(D_{\text{in}}, D_{\text{out}})$，那么损失对它的梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ 的形状**必须严格是 $(D_{\text{in}}, D_{\text{out}})$**！
- 如果输入矩阵 $\mathbf{X}$ 的形状是 $(B, D_{\text{in}})$，那么回传给输入的梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$ 的形状**必须严格是 $(B, D_{\text{in}})$**！

<div align="center">

<img src="/figures/02-foundations/latex/05-basic-components-scratch/bias-broadcast-shapes.png" alt="全连接层前向矩阵乘法与反向梯度计算图数据流" width="86%">

_图 2.5-2：全连接层前向矩阵乘法与反向梯度计算图数据流。_

</div>

---

## 2.5.2 核心数学推导一：全连接层的矩阵微分与反向传播

设一个小批量输入张量为 $\mathbf{X} \in \mathbb{R}^{B \times D_{\text{in}}}$（$B$ 为批量大小，$D_{\text{in}}$ 为输入特征维度），权重矩阵为 $\mathbf{W} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$，偏置向量为 $\mathbf{b} \in \mathbb{R}^{1 \times D_{\text{out}}}$。

### 1. 前向传播矩阵方程
$$\mathbf{Y} = \mathbf{X} \mathbf{W} + \mathbf{b} \in \mathbb{R}^{B \times D_{\text{out}}}$$

其中偏置 $\mathbf{b}$ 沿批次维度 $B$ 进行广播加法。

### 2. 反向传播三大梯度推导
假设上游已经回传了损失标量 $\mathcal{L}$ 对输出矩阵 $\mathbf{Y}$ 的梯度：

$$\mathbf{G}_Y = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \in \mathbb{R}^{B \times D_{\text{out}}}$$

利用初等矩阵微积分，我们推导回传给三个变量的梯度：

1. **对权重矩阵 $\mathbf{W}$ 的梯度**：
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^\top \mathbf{G}_Y \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$$
2. **对偏置向量 $\mathbf{b}$ 的梯度**：由于偏置在 $B$ 个样本上被复用，梯度为沿批次维度的求和：
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_{i=1}^B \mathbf{G}_Y[i, :] \in \mathbb{R}^{1 \times D_{\text{out}}}$$
3. **回传给上一层输入 $\mathbf{X}$ 的梯度**：
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \mathbf{G}_Y \mathbf{W}^\top \in \mathbb{R}^{B \times D_{\text{in}}}$$

### 3. 全连接层手动反向传播数值算例
设批次 $B = 1, D_{\text{in}} = 2, D_{\text{out}} = 2$：
- 输入向量 $\mathbf{X} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}$；
- 权重矩阵 $\mathbf{W} = \begin{bmatrix} 0.5 & -0.5 \\ 1.0 & 0.0 \end{bmatrix}$；
- 偏置向量 $\mathbf{b} = \begin{bmatrix} 0.1 & 0.1 \end{bmatrix}$；
- 上游回传的输出梯度 $\mathbf{G}_Y = \begin{bmatrix} 1.0 & 3.0 \end{bmatrix}$。

我们来手动求解反向梯度：
1. **计算权重梯度 $\nabla_{\mathbf{W}} \mathcal{L} = \mathbf{X}^\top \mathbf{G}_Y$**：
   $$\nabla_{\mathbf{W}} \mathcal{L} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} \begin{bmatrix} 1.0 & 3.0 \end{bmatrix} = \begin{bmatrix} 1.0 \times 1.0 & 1.0 \times 3.0 \\ 2.0 \times 1.0 & 2.0 \times 3.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 3.0 \\ 2.0 & 6.0 \end{bmatrix}$$
2. **计算偏置梯度 $\nabla_{\mathbf{b}} \mathcal{L} = \mathbf{G}_Y$**：
   $$\nabla_{\mathbf{b}} \mathcal{L} = \begin{bmatrix} 1.0 & 3.0 \end{bmatrix}$$
3. **计算回传给输入的梯度 $\nabla_{\mathbf{X}} \mathcal{L} = \mathbf{G}_Y \mathbf{W}^\top$**：
   $$\nabla_{\mathbf{X}} \mathcal{L} = \begin{bmatrix} 1.0 & 3.0 \end{bmatrix} \begin{bmatrix} 0.5 & 1.0 \\ -0.5 & 0.0 \end{bmatrix} = \begin{bmatrix} 1.0 \times 0.5 + 3.0 \times (-0.5) & 1.0 \times 1.0 + 3.0 \times 0.0 \end{bmatrix} = \begin{bmatrix} -1.0 & 1.0 \end{bmatrix}$$

初等代数的几步矩阵乘法极为清晰，展现了自动微分引擎在幕后执行的全部真实算术步骤！

<details>
<summary><b>深入推导：基于弗罗贝尼乌斯内积（Frobenius Inner Product）的矩阵微分迹数理证明（点击展开查看完整推导）</b></summary>

根据全微分与矩阵迹（Trace）的关系，标量损失的全微分为：
$$d\mathcal{L} = \text{Tr}\left( \left(\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}\right)^\top d\mathbf{Y} \right) = \text{Tr}\left( \mathbf{G}_Y^\top (d\mathbf{X} \mathbf{W} + \mathbf{X} d\mathbf{W} + d\mathbf{b}) \right)$$
利用矩阵迹的循环置换性质 $\text{Tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \text{Tr}(\mathbf{C}\mathbf{A}\mathbf{B}) = \text{Tr}(\mathbf{B}\mathbf{C}\mathbf{A})$：
1. 提取 $d\mathbf{W}$ 项：$\text{Tr}(\mathbf{G}_Y^\top \mathbf{X} d\mathbf{W}) = \text{Tr}((\mathbf{X}^\top \mathbf{G}_Y)^\top d\mathbf{W}) \implies \frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^\top \mathbf{G}_Y$；
2. 提取 $d\mathbf{X}$ 项：$\text{Tr}(\mathbf{G}_Y^\top d\mathbf{X} \mathbf{W}) = \text{Tr}(\mathbf{W} \mathbf{G}_Y^\top d\mathbf{X}) = \text{Tr}((\mathbf{G}_Y \mathbf{W}^\top)^\top d\mathbf{X}) \implies \frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \mathbf{G}_Y \mathbf{W}^\top$。
严格证得矩阵求导公式与转置法则。
</details>

---

## 2.5.3 核心数学推导二：Softmax 与交叉熵损失的惊人极简复合导数

在多分类任务与离散动作预测中，模型最后一步通常是 Softmax 归一化与交叉熵损失。

<div align="center">

<img src="/figures/02-foundations/source/05-basic-components-scratch/relu-fig1.png" alt="交叉熵损失函数对预测概率分布与真实独热标签分布的对数距离度量。" width="86%">

_图 2.5-3：交叉熵损失函数对预测概率分布与真实独热标签分布的对数距离度量。 出处：[Dive into Deep Learning，Aston Zhang et al.，2023](https://d2l.ai/)。_

</div>

### 1. 复合前向方程
设网络输出的未归一化对数几率（Logits）为 $\mathbf{z} = [z_1, z_2, \dots, z_K]^\top$。
- **Softmax 概率分布**：$p_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}$；
- **交叉熵损失**：$\mathcal{L} = -\sum_{i=1}^K y_i \log(p_i)$（其中 $\mathbf{y}$ 为真实的 One-Hot 标签）。

### 2. 惊人极简的解析复合梯度
如果分开对 Softmax 和交叉熵分别求导，中间过程充斥着复杂的雅可比张量消元；然而当二者复合后，损失 $\mathcal{L}$ 对原始 Logits $z_i$ 的导数退化为一个不可思议的初等代数减法公式：

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$

### 3. 极简导数手算数值算例
设一个 3 分类任务（真实类别为第 1 类，即标签 $\mathbf{y} = [1.0, 0.0, 0.0]^\top$）：
网络输出的 Logits 经 Softmax 计算后的预测概率分布为：
$$\mathbf{p} = [0.70, 0.20, 0.10]^\top$$

我们直接一步口算出反向传播给 Logits 的梯度向量：
$$\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y} = \begin{bmatrix} 0.70 - 1.00 \\ 0.20 - 0.00 \\ 0.10 - 0.00 \end{bmatrix} = \begin{bmatrix} -0.30 \\ +0.20 \\ +0.10 \end{bmatrix}$$

> **代数直觉**：
> - 对于正确类别（第 1 类），梯度为 $-0.30$（负梯度推动 Logit $z_1$ 增大，提升正确概率）；
> - 对于错误类别（第 2、3 类），梯度为正数 $+0.20, +0.10$（正梯度打压错误 Logits 减小）；
> - 三个梯度分量之和恒等于 $-0.30 + 0.20 + 0.10 = 0.00$！

这一极度纯粹对称的数学特性，保障了深度分类网络在海量训练中的数值超强稳定性！

<details>
<summary><b>深入推导：Softmax 与交叉熵复合求导极简梯度的雅可比矩阵消元证明（点击展开查看完整推导）</b></summary>

首先对 Softmax 求偏导：
- 当 $i = j$ 时：$\frac{\partial p_i}{\partial z_i} = p_i(1 - p_i)$；
- 当 $i \ne j$ 时：$\frac{\partial p_i}{\partial z_j} = -p_i p_j$。
将交叉熵求导链式展开：
$$\frac{\partial \mathcal{L}}{\partial z_k} = \sum_{i=1}^K \frac{\partial \mathcal{L}}{\partial p_i} \frac{\partial p_i}{\partial z_k} = -\sum_{i=1}^K \frac{y_i}{p_i} \frac{\partial p_i}{\partial z_k} = -\frac{y_k}{p_k} p_k(1 - p_k) - \sum_{i \ne k} \frac{y_i}{p_i} (-p_i p_k) = -y_k(1 - p_k) + \sum_{i \ne k} y_i p_k$$
利用 One-Hot 标签和为 1，即 $\sum_{i=1}^K y_i = 1 \implies \sum_{i \ne k} y_i = 1 - y_k$：
$$\frac{\partial \mathcal{L}}{\partial z_k} = -y_k + y_k p_k + (1 - y_k) p_k = p_k - y_k$$
严格证得极简复合导数。
</details>

---

## 2.5.4 纯底层 PyTorch 代码实现：从零手动实现全连接前向与反向传播引擎

下面我们使用纯底层 PyTorch 基础张量，完全不借助 `torch.autograd` 或 `nn.Linear`，从零手写全连接层、ReLU 激活函数、交叉熵损失及手动 SGD 梯度更新。

```python
import torch

class ScratchLinear:
    """
    纯手动实现全连接层 (Dense Layer)
    显式维护前向计算、缓存与解析反向传播梯度
    """
    def __init__(self, in_features: int, out_features: int):
        # 高斯初始化权重与零初始化偏置
        self.W = torch.randn(in_features, out_features) * 0.05
        self.b = torch.zeros(1, out_features)

        # 梯度容器
        self.grad_W = torch.zeros_like(self.W)
        self.grad_b = torch.zeros_like(self.b)

        # 缓存前向输入供反向求导使用
        self.cached_x = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.cached_x = x.clone()
        out = torch.matmul(x, self.W) + self.b
        return out

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        :param grad_output: 上游梯度 (B, out_features)
        :return: 回传给输入的梯度 (B, in_features)
        """
        # 1. 权重梯度 grad_W = X^T * G_out
        self.grad_W = torch.matmul(self.cached_x.t(), grad_output)
        # 2. 偏置梯度 grad_b = sum(G_out, dim=0)
        self.grad_b = grad_output.sum(dim=0, keepdim=True)
        # 3. 输入梯度 grad_x = G_out * W^T
        grad_input = torch.matmul(grad_output, self.W.t())
        return grad_input

    def step(self, lr: float):
        """
        手动 SGD 梯度更新
        """
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b

def scratch_relu(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    前向 ReLU 与反向掩码
    """
    out = torch.clamp_min(x, 0.0)
    mask = (x > 0.0).float()
    return out, mask

def scratch_softmax_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
    """
    手写 Softmax + 交叉熵损失及其极简梯度 (p - y)
    :param logits: (B, num_classes)
    :param targets: (B,) 整数类别标签
    :return: (scalar_loss, grad_logits)
    """
    B, num_classes = logits.shape

    # 数值稳定技巧：减去最大值防止 exp 溢出
    max_logits = logits.max(dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True) # (B, num_classes)

    # 构造 One-Hot 真实标签
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

    # 交叉熵损失
    eps = 1e-12
    loss = - torch.sum(one_hot * torch.log(probs + eps)) / B

    # 极简复合梯度: grad = (p - y) / B
    grad_logits = (probs - one_hot) / B
    return loss.item(), grad_logits

# ===================================================================
# 单元测试与手动梯度下降收敛校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    in_dim = 8
    num_classes = 3

    layer = ScratchLinear(in_features=in_dim, out_features=num_classes)

    dummy_x = torch.randn(batch_size, in_dim)
    dummy_labels = torch.tensor([0, 2, 1, 0], dtype=torch.long)

    # 1. 验证初始损失
    initial_logits = layer.forward(dummy_x)
    initial_loss, grad_logits = scratch_softmax_cross_entropy(initial_logits, dummy_labels)
    print(f"[Scratch Test] 初始分类损失: {initial_loss:.4f}")

    # 2. 手动执行 50 步纯张量反向传播与 SGD 优化
    for epoch in range(50):
        logits = layer.forward(dummy_x)
        loss, grad_out = scratch_softmax_cross_entropy(logits, dummy_labels)
        layer.backward(grad_out)
        layer.step(lr=0.5)

    final_logits = layer.forward(dummy_x)
    final_loss, _ = scratch_softmax_cross_entropy(final_logits, dummy_labels)
    print(f"[Scratch Test] 50步手动反向优化后损失: {final_loss:.4f}")

    assert final_loss < initial_loss, "手动反向传播未促使损失下降！"
    assert layer.grad_W.shape == (in_dim, num_classes), "手动计算权重梯度维度不符！"
    print("✓ 纯张量底层全连接层、Softmax 极简梯度与手动反向传播单测全部通过！")
```

---

## 2.5.5 本节小结

回顾本节内容，我们剥离了深度学习框架的重重包装，直面其底层的数学本质：
1. **矩阵全连接微积分**：通过前向矩阵相乘与反向转置对齐，严格确保了多元链式梯度的维度守恒；
2. **Softmax 复合导数的数学对称美**：联合交叉熵导数直接化简为预测概率与真实标签的误差差分 $\mathbf{p} - \mathbf{y}$，构成了分类优化的核心引擎；
3. **计算图闭环**：通过纯底层张量操作手动实现前向缓存与反向求导，使我们具备了从物理第一性原理定制和优化任意复杂世界模型算子的硬核底气。
