# 5.3　视频 JEPA

前两节讲的是 JEPA 的骨架。它一开始是为单张图设计的，可世界是连续的。视频多了一个时间维度，把 JEPA 从图片搬到视频上，是本节要处理的事情。

## 从 patch 到 tubelet

图片的 patch 只覆盖空间：一张 $H\times W$ 图切成一块块 $p\times p$ 的小格。视频的形状是 $[T,H,W,C]$，时间上还压着一摞帧。

所谓 **tubelet**（时空管块），就是把 patch 沿时间方向也拉长：一个 $t\times p_h\times p_w$ 的小柱体，覆盖 $t$ 帧、$p_h$ 像素高、$p_w$ 像素宽。设输入视频张量 $x\in\mathbb{R}^{T\times H\times W\times C}$，使用 tubelet $(t, p_h, p_w)$ 切分后，token 数为：

$$
N_T = \frac{T}{t},\quad
N_H = \frac{H}{p_h},\quad
N_W = \frac{W}{p_w},
\qquad
\text{token 总数} = N_T N_H N_W.
$$

例如 PixelWorld 的 `16×16` 单帧、`patch=4` 时每帧得 $4\times4=16$ 个 token；若视频有 $3$ 帧，就是 $3\times16=48$ 个 token。这正是 C1 里 `patchify_video` 看到的形状。

时间跨度的选择会改变问题本身。$t$ 太短，token 只看到瞬时外观；$t$ 太长，快速运动的物体被混进同一个 token，运动信息就被抹平了。

## 时空 masking

图片的 mask 是空间区域，视频的 mask 是时空区域。V-JEPA 常用的两类 mask：

- **空间遮挡**：在多帧里把同一物体的区域整段遮掉。它要求模型理解对象一致性——这块东西虽然看不见，但应该还在那儿。
- **时间遮挡**：遮住一段较长的未来片段。它要求模型预测运动——根据过去几帧，推接下来会怎样。

课程同时使用短程与长程 mask，比较模型能否读出当前位置、速度，以及物体被遮挡后再次出现的位置。

## 被动观看能学到什么

无动作的视频能学到物体、运动、外观、场景变化。在 C1 里，我们用 PixelWorld 的合成片段训练一个 Tiny Video-JEPA，先回答最基本的问题：特征预测能不能稳定训练、会不会坍缩。

但被动视频没有控制信号。它无法回答"如果机器人换一个动作，画面会怎么变"。这是被动 JEPA 的天然边界——它是一个表示模型，并不自动成为可控的规划模型。

## Linear Probe：怎么知道特征里有什么

特征本身不能直接看。一个常用工具是**线性探针**（linear probe）：冻结 Encoder $f_{\bar\theta}$，只在它上面训一个线性层 $W$ 去预测某个我们关心的量 $q$（比如方块位置、速度、类别）：

$$
\hat{q} = W\,f_{\bar\theta}(x) + b.
$$

$W$ 和 $b$ 用普通最小二乘或梯度下降拟合，不更新 Encoder。如果这样简单的一层线性映射就能读出位置，说明表示以"好用"的形式保留了它。

要写成公式，设冻结特征矩阵 $Z\in\mathbb{R}^{M\times d}$、标签矩阵 $Q\in\mathbb{R}^{M\times k}$，闭式解是：

$$
W^{*} = \big(Z^{\top}Z + \lambda I\big)^{-1} Z^{\top} Q.
$$

其中 $\lambda$ 是一点正则项，保证数值稳定。

probe 成绩仍有上限。它证明某种信息**可读**，不证明所有下游任务都会受益。在 C1/C2 中，probe 的训练与测试 episode 按 seed 分开，避免"记住样本"被误读成"理解了位置"。

## 小结

- tubelet 把空间 patch 扩展到时间维度，token 数随 $T/t$、$H/p_h$、$W/p_w$ 缩减。
- 时空 mask 的范围决定模型学短程外观还是长程运动。
- 被动视频能验证表示质量，不能验证动作可控性。
- linear probe 是一个最小但有用的"特征里有什么"探针。

[上一篇 4.2 掩码、EMA 与表示坍缩](./05-02-mask-ema-collapse.md) · [下一篇 → 4.4 动作条件 JEPA（Action-JEPA）](./05-04-action-jepa.md) · [回到第 4 章](./index.md) · [动手：C1 视频特征预测](/chapters/05-jepa/05-05-jepa)
