# 6.3　目标网络（EMA）

上一节把目标分支从梯度里截断了。$f_{\bar\theta}$ 不接收梯度，参数就得从别处来。JEPA 的做法是**指数移动平均**（Exponential Moving Average，EMA）：让 $\bar\theta$ 跟着在线参数 $\theta$ 缓慢更新。

$$
\bar\theta \leftarrow m\,\bar\theta + (1 - m)\,\theta,
\qquad m \in [0.99,\, 0.999].
$$

$m$ 叫动量系数。$m$ 越接近 $1$，靶子动得越慢。动量之所以要这样大，是为了避免 Predictor 追逐自己的影子。如果靶子每步都跟着 $\theta$ 同步跳，今天刚学会的预测，明天靶子就变了。EMA 让 $f_{\bar\theta}$ 像一个动作迟缓但方向稳定的对手，Predictor 追它要费真功夫，而非共谋一个平凡解。

把三件事合起来看，JEPA 训练时的梯度流向是这样的：

```text
              (梯度不流回)
   sg ─────────────────────►  f_{barθ}  ──EMA(m)──  f_θ
     ▲                                          ▲
     │  target feature                          │ context
     │                                          │
  Predictor g_φ ◄──── 损失 ◄────────────────────┘
```

要诚实：stop-gradient 和 EMA 只规定优化的路径，不保证最后的表示适合任何任务。坍缩仍可能发生，只是变难了。

## 从 patch 到 tubelet

这套骨架一开始是为单张图设计的，可世界是连续的。图片的 patch 只覆盖空间；视频的形状是 $[T,H,W,C]$，时间上还压着一摞帧。

所谓 **tubelet**（时空管块），就是把 patch 沿时间方向也拉长：一个 $t\times p_h\times p_w$ 的小柱体。设输入视频张量 $x\in\mathbb{R}^{T\times H\times W\times C}$，使用 tubelet $(t, p_h, p_w)$ 切分后，token 数为：

$$
N_T = \frac{T}{t},\quad
N_H = \frac{H}{p_h},\quad
N_W = \frac{W}{p_w},
\qquad
\text{token 总数} = N_T N_H N_W.
$$

例如 PixelWorld 的 `16×16` 单帧、`patch=4` 时每帧得 $4\times4=16$ 个 token；若视频有 $3$ 帧，就是 $3\times16=48$ 个 token。这正是 6.5 第一份 Notebook 里 `patchify_video` 看到的形状。

$t$ 太短，token 只看到瞬时外观；$t$ 太长，快速运动的物体被混进同一个 token，运动信息就被抹平了。

## 时空 masking

- **空间遮挡**：在多帧里把同一物体的区域整段遮掉。它要求模型理解对象一致性。
- **时间遮挡**：遮住一段较长的未来片段。它要求模型预测运动。

课程同时使用短程与长程 mask，比较模型能否读出当前位置、速度，以及物体被遮挡后再次出现的位置。

## 被动观看能学到什么

无动作的视频能学到物体、运动、外观、场景变化。但被动视频没有控制信号。它无法回答"如果机器人换一个动作，画面会怎么变"。这是被动 JEPA 的天然边界——它是一个表示模型，并不自动成为可控的规划模型。

## Linear Probe

特征本身不能直接看。一个常用工具是**线性探针**（linear probe）：冻结 Encoder $f_{\bar\theta}$，只在它上面训一个线性层去预测某个我们关心的量 $q$（比如方块位置、速度）：

$$
\hat{q} = W\,f_{\bar\theta}(x) + b.
$$

设冻结特征矩阵 $Z\in\mathbb{R}^{M\times d}$、标签矩阵 $Q\in\mathbb{R}^{M\times k}$，闭式解是 $W^{*} = \big(Z^{\top}Z + \lambda I\big)^{-1} Z^{\top} Q$。probe 成绩证明某种信息**可读**，不证明所有下游任务都会受益。

## 小结

- EMA 让 Target Encoder 缓慢跟随在线参数，提供稳定但非静止的靶子。
- tubelet 把空间 patch 扩展到时间维度；时空 mask 的范围决定模型学短程外观还是长程运动。
- 被动视频能验证表示质量，不能验证动作可控性。
- linear probe 是一个最小但有用的"特征里有什么"探针。

[上一篇 6.2 掩码机制与表示坍缩](./02-mask-collapse.md) · [下一篇 → 6.4 动作条件特征预测](./04-action-jepa.md) · [动手：联合嵌入预测](/chapters/06-jepa/05-jepa-scratch)
