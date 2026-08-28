# 6.2　掩码机制与表示坍缩

JEPA 没有像素重建，也没有对比学习的负样本。这听起来很自由，但自由得危险。

设想一种偷懒的解法：Context Encoder 和 Target Encoder 都输出全零向量，Predictor 也输出零。代入 6.1 的损失：

$$
\mathcal{L}_{\text{JEPA}}
= \big\|g_\phi(\mathbf{0}) - f_{\bar\theta}(y)\big\|_2^{2}
= \|\mathbf{0} - \mathbf{0}\|_2^{2} = 0.
$$

损失直接归零，模型"赢"了，可表示里没留下任何信息。这就是**表示坍缩**（representation collapse）：所有输入被映射到同一个点，预测变得平庸地正确。

本节先处理出题方式（mask）和梯度截断（stop-gradient）。靶子怎样缓慢移动，交给下一节的目标网络。

## Mask 在问什么

图像被切成一块块 patch。我们遮住一部分区域，只把可见 patch 交给 Context Encoder $f_\theta$，让 Predictor $g_\phi$ 去猜被遮区域的 target feature。

设可见 patch 集合为 $c$，被遮 patch 集合为 $y$，那么 6.1 的损失改写得更精确一点：

$$
\mathcal{L}
= \Big\|g_\phi\big(f_\theta(c)\big) - f_{\bar\theta}(y)\Big\|_2^{2}.
$$

mask 的形状，实际上定义了这道题问的是什么。若 $y$ 就紧挨着 $c$，模型靠局部纹理就能蒙混；若 $y$ 更大、更远，模型就得真的理解对象与场景结构。所以 mask 不是数据增强，而是出题方式。

## 两个 Encoder 与 stop-gradient

设想 Context Encoder 和 Target Encoder 是同一棵树、共用同一组梯度。它们可以一起朝"全零"的方向滑，谁也不拦谁。

JEPA 的做法是把目标分支从预测损失的梯度里截断。这叫 **stop-gradient**：

$$
\tilde{y} = \mathrm{sg}\big(f_{\bar\theta}(y)\big),
\qquad
\frac{\partial \mathcal{L}}{\partial \bar\theta} = 0.
$$

$\mathrm{sg}(\cdot)$ 表示前向照常取值，反向时把梯度当作 $0$。预测损失只会拉动 $f_\theta$ 和 $g_\phi$，不会直接拉动 $f_{\bar\theta}$。目标特征成了一个"固定的靶子"，Predictor 只能去逼近它，不能反过来把靶子挪到自己更舒服的位置。

$f_{\bar\theta}$ 不接收梯度，参数就得从别处来——那是下一节 EMA 的工作。

## 训练时必须主动检查坍缩

1. 每个特征维度的标准差 $\mathrm{std}(z_d)$，接近 $0$ 即危险；
2. 不同样本间的余弦相似度 $\cos(z_i, z_j)$，普遍接近 $1$ 即危险；
3. 特征协方差矩阵的有效秩；
4. linear probe 是否明显优于"猜均值"的常数基线。

一个明确的警报信号是：$\mathcal{L}_{\text{JEPA}}$ 在下降，而特征方差却在逼近 $0$。这时模型不是在学世界，是在学闭嘴。

## 小结

- mask 决定模型从哪些上下文推断哪些目标，本质是出题方式。
- stop-gradient 把目标分支从预测损失的梯度里截断，避免两个编码器共谋坍缩。
- 防坍缩不能靠直觉，要查特征统计与下游 probe。

下一篇 [6.3 目标网络](./03-video-jepa.md) 说明靶子怎样缓慢跟随在线参数，并把同一套骨架搬到视频上。
