# 1.8 动手：基础实验

> **本节目标**：跑通两份 Notebook，把表示和空间接起来。一份用卷积和 patch 压缩一张小图，一份把深度像素算进三维格子并用 CEM 搜连续动作。从经历里学习动态留给 2.4。

> **本节代码**：[看见与压缩](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/01_foundations/F1-see-remember-compress.ipynb) · [空间与规划](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/01_foundations/F2-space-plan-train.ipynb) · [foundations.py](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/foundations.py)

> **前置知识**：你已经读过第 1 章和第 2 章，并最好刚跑完 [0.6 动手：从零重新发明世界模型](/chapters/00-why-world-models/00-06-invent-a-world-model)。这一节把那些零件真跑一遍。

---

0.6 里你用整数和字典发明了一台世界模型。九格网格，转移表，贪心会走进陷阱，有了模型就能绕开。

现在世界变大了一点。观察不再是一个格子编号，而是一张小图；空间不再是 3×3，而是深度相机拍下来的三维点；动态不再是你手写的字典，而是从轨迹里数出来的概率。

你当时大概会问：这些零件——卷积、patch、点云、Occupancy、CEM——各自都眼熟，可它们是怎么接成一台世界模型的？

这一节把它们接起来。两份 Notebook，只依赖 NumPy，CPU 上几秒内跑完。目标不是训练大网络，而是看清接口，以及接口在哪里先失败。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/f1-foundations.png" alt="基础概念" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">这一节要接起来的三件事：看见（一张图怎么变成特征）、走进空间（深度怎么变成格子）、从经历里学习（轨迹怎么变成转移概率）。模型从未见过物理定律，它要从这些计算里自己发现规律。</div>
</div>

## 本次会得到什么

运行结束后，你会得到：

- 一张 16×16 的 PixelWorld，以及三种 3×3 卷积核在它上面的响应
- 16 个 48 维 patch token，以及拼回去之后 MSE 为 0 的往返
- 块平均压缩后的重建误差
- 6×6 深度图反投影出的 36 个点，平移 1 米后落入 12×12 Occupancy 的 8 个格子
- 外参写成 1.3 米时，点云平均偏 0.300 米
- CEM 在一维线上搜出的 5 步动作
- 一条带 20% 打滑的 LineWorld，以及从 140 段轨迹里数出来的转移表
- 一步预测在按 episode 切开的测试集上大约 0.83 的准确率
- 一次 MPC：第一步打滑停在原地，后面三步仍然走到终点

## 怎样运行

```bash
python -m pip install -r requirements.txt
jupyter lab
```

两份 Notebook 在：

```text
notebooks/01_foundations/F1-see-remember-compress.ipynb
notebooks/01_foundations/F2-space-plan-train.ipynb
```

即使暂时不打开 Notebook，也可以先跑测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

下面每一段数字，都是用 `PYTHONPATH=src` 对着源码跑出来的。

## 看见、记住与压缩

路径：

```text
notebooks/01_foundations/F1-see-remember-compress.ipynb
```

0.6 的观察是一个格子坐标。现在观察是一张图。世界模型要先回答：这张图里，哪些数值得留下来？

### 第一步：从零写一个卷积

卷积是局部加权和。教学版 `conv2d_valid` 不做 padding，一张 \(H\times W\) 的图被 \(k\times k\) 的核扫过，输出是 \((H-k+1)\times(W-k+1)\)：

$$
(I * K)[i, j]
= \sum_{u=0}^{k-1}\sum_{v=0}^{k-1}
I[i+u,\, j+v]\, K[u, v]
$$

```python
from hwm.data import make_pixelworld_dataset
from hwm.foundations import conv2d_valid, rgb_to_gray
import numpy as np

episodes = make_pixelworld_dataset(num_episodes=2, length=6, seed=0)
image = episodes[0].observations[0]
gray = rgb_to_gray(image)
kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)
response = conv2d_valid(gray, kernel)
print(image.shape, response.shape, float(response.min()), float(response.max()))
```

**运行这一步，你会看到什么？**

```
(16, 16, 3) (14, 14) -228.73 228.73
```

16×16 的图被 3×3 核扫成 14×14。核没有训练过，水平差分已经能抓住红块的上下沿。这就是卷积的归纳偏置：它先假设「重要的东西是局部的」。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/f1-convolution.png" alt="卷积响应图" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">三种 3×3 核扫过同一张小图。没有训练，水平核、垂直核和角点核已经能把红块的边勾出来。</div>
</div>

**这就是「看见」的最小实现**：还没有网络，只有一个滑动窗口。后面所有 Encoder，第一步都在做这件事。

### 第二步：切成 ViT 风格的 patch

Vision Transformer 不扫卷积核，它把图切成块，每一块当成一个 token。16×16、patch 边长 4，得到 16 个向量，每个是 \(4\times 4\times 3 = 48\) 维：

```python
from hwm.foundations import patchify, unpatchify, reconstruction_mse

tokens = patchify(image, 4)
back = unpatchify(tokens, image.shape, 4)
print(tokens.shape, reconstruction_mse(image, back))
```

**运行这一步，你会看到什么？**

```
(16, 48) 0.0
```

拼回去 MSE 是 0——`patchify` / `unpatchify` 是可逆的，没有丢信息。压缩比也不是 5.3：\(16\times 16\times 3 = 768\) 个数，还是 768 个数，只是排成了 16 个 token。真正的压缩发生在下一步，当你用线性层把 48 维压到 8 维的时候。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/f1-patchify.png" alt="ViT patch 切分" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">16×16 切成 4×4 的块，得到 16 个 token。切分本身不压缩，它只改数据的形状，好让后面的 Transformer 按块读图。</div>
</div>

**一个值得做的实验**：把 patch 边长改成 2 或 8。边长 2 得到 64 个 12 维 token，边长 8 得到 4 个 192 维 token。token 越多，表示越细，后面自注意力的平方代价也越大。

### 第三步：同一张末帧，两段不同的历史

两段轨迹可以停在同一格，却从相反方向走来。只看最后一帧，分不出来；把相邻中心差拼进状态，速度方向就写在里面了。

```python
from hwm.foundations import remember_velocity

memory = remember_velocity(episodes[0].observations)
print(memory[0], memory[-1])
```

**运行这一步，你会看到什么？** seed=0 第一段：

```
第一帧: [7. 7. 0. 0.]     # 中心在 (7,7)，还没有速度
最后帧: [6. 9. 0. 1.]     # 中心到了 (6,9)，这一步向下走了 1
```

`remember_velocity` 把中心和相邻差分拼成 4 个数。这不是 LSTM，只是最小的时序记忆：当前在哪，刚刚往哪走。

**这就是「当前观察不够用」**：0.6 的格子坐标是完整状态；像素世界里，一张静帧丢掉了速度。后面 RSSM 的确定性隐状态 \(h_t\)，干的是同一件事，只是不再用手写差分。

### 第四步：块平均压缩

真压缩发生在这里。每 \(4\times 4\) 个像素取均值，16×16 变成 4×4×3：

```python
from hwm.foundations import block_average_encode, block_average_decode

latent = block_average_encode(image, 4)
recon = block_average_decode(latent, 4)
print(latent.shape, reconstruction_mse(image, recon))
```

**运行这一步，你会看到什么？**

```
(4, 4, 3) 691.75
```

像素值在 0–255 上，MSE 691 大约对应每个通道差 26。红块的边被抹成色块。这就是「只保留决策需要的信息」在没有神经网络时的样子：位置大概还在，纹理没了。

## 从相机到空间，再交给规划器

路径：

```text
notebooks/01_foundations/F2-space-plan-train.ipynb
```

格子世界的坐标是送进模型的。相机拍到的是深度图。要规划「会不会撞」，必须先把像素变回三维点，再落到俯视格子上。

### 第一步：深度像素变成三维点

针孔反投影：

$$
X = \frac{(u-c_x)\,Z}{f_x},\qquad
Y = \frac{(v-c_y)\,Z}{f_y},\qquad
Z = \mathrm{depth}(u,v)
$$

```python
from hwm.foundations import depth_to_points

depth = np.full((6, 6), 4.0, np.float32)
depth[2:4, 2:4] = 2.0
points = depth_to_points(depth, fx=6, fy=6, cx=2.5, cy=2.5)
print(points.shape, points[:, 2].min(), points[:, 2].max())
```

**运行这一步，你会看到什么？**

```
(36, 3) 2.0 4.0
```

36 个像素变成 36 个点。中间 2×2 是近处 \(Z=2\)，四周是 \(Z=4\)。\(X, Y\) 大约落在 \([-1.67, 1.67]\)。

### 第二步：外参把相机放进同一个世界

```python
from hwm.foundations import make_camera_transform, transform_points

world = transform_points(points, make_camera_transform(tx=1.0))
print(world.mean(0) - points.mean(0))
```

**运行这一步，你会看到什么？**

```
[1.  0.  0.]
```

相机右移 1 米，点云整体平移 1 米。齐次矩阵左乘点。对不上，先查是 `T @ p` 还是 `p @ T`，再查轴。

### 第三步：点落到俯视 Occupancy

Occupancy 不看颜色，只记哪里有东西。\(x\in[-2,4]\)，\(z\in[0,6]\)，分辨率 0.5 m：

```python
from hwm.foundations import points_to_occupancy

occupancy = points_to_occupancy(world, (-2, 4), (0, 6), 0.5)
print(occupancy.shape, int(occupancy.sum()))
```

**运行这一步，你会看到什么？**

```
(12, 12) 8
```

36 个点挤进 8 个格子。近处的 4 个像素叠在一起，远处摊得更开——这就是透视。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/f2-depth-occupancy.png" alt="深度反投影到 Occupancy" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">深度图 → 相机系点云 → 俯视 Occupancy。颜色没有了，只剩「这里有没有东西」。碰撞检查和驾驶预测，用的就是这张格子。</div>
</div>

把平移写成 1.3 米：

```python
wrong = transform_points(points, make_camera_transform(tx=1.3))
print(np.linalg.norm(wrong - world, axis=1).mean())
```

```
0.300
```

点云平均偏了 0.3 米，占用会错开几格。神经网络可以在训练集上适应固定偏差，却不能把错误几何变正确。

**这就是「几何必须先算对」**：后面所有 BEV、Occupancy、NeRF，都站在这三行公式上。写反一个轴，后面的模型只是在拟合你的笔误。

### 第四步：CEM 在一条线上搜动作

世界暂时缩成一维：\(x_{t+1}=x_t+a_t\)，\(a\in[-1,1]\)。交叉熵方法采一群动作序列，留下精英，收缩均值和方差，再采一轮：

```python
from hwm.foundations import cem_plan_1d

mean, history = cem_plan_1d(start=0.0, target=3.0, horizon=5, seed=0)
print(mean)
print(history)
```

**运行这一步，你会看到什么？**

```
mean:    [0.50  0.56  0.64  0.57  0.73]
history: [-0.026, -0.020, -0.020, -0.018, -0.018]
```

五步加起来约 3.00，正好到目标。分数（负的末端平方误差）一轮轮变好。种群 400、精英 40、5 轮——这是 PlaNet 在连续动作里做的事，只是世界还是一条线。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/f2-cem-search.png" alt="CEM 搜索过程" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">CEM：从随机采样到精英集中。绿线是目标，红线是均值，蓝线是样本。几轮之后，分布堆到能走到 3.0 的那一组动作上。</div>
</div>

### 第五步：Symlog 与梯度裁剪

奖励可以从 \(-100\) 到 \(+100\)。DreamerV3 用对称对数压范围：

$$
\mathrm{symlog}(x) = \mathrm{sign}(x)\,\log(1+|x|)
$$

```python
from hwm.foundations import symlog, clip_by_norm

print([float(symlog(x)) for x in (-100, -1, 0, 1, 100)])
print(clip_by_norm(np.array([3.0, 4.0]), 2.5))
```

**运行这一步，你会看到什么？**

```
symlog: [-4.615, -0.693, 0.0, 0.693, 4.615]
clip:   [1.5 2. ]
```

\(\pm 100\) 被压到 \(\pm 4.62\)，0 还是 0。长度为 5 的梯度被裁到 2.5，方向不变。这两行不是模型，是训练工程：后面 Actor-Critic 的回报跨几个数量级时，先过 symlog，再裁梯度。

**一个值得做的实验**：把 CEM 的种群从 50 提到 500。一维世界里 50 往往已经够用；维度升高之后，同样的精英比例会开始不够——这就是为什么 PlaNet 要迭代收缩，而不是一次采完。

## 从经历中学习概率动态

从经历里学习动态已单独成页，挂在第 2 章侧栏，不要再挤在 1.8 里跟做。

> 👉 [2.4 动手：第一台可学习世界模型](/chapters/02-data-and-first-model/02-04-learn-a-table-world)

那里从装 `Episode`、段内取样，数到 \(\hat P(s'\mid s,a)\)，再用 MPC 走到终点。1.8 只负责看见与空间。

## 自动检查

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

当前检查包括 shape、动作时间对齐、卷积、patch 往返、记忆方向、块压缩、反投影、Occupancy、CEM、Symlog、计数动态、MPC，以及 Notebook 全格执行。

## 已知简化与坑

- **PixelWorld 是 16×16 的红块。** 卷积和 patch 在这里很容易看懂，换到真实照片，同样的 3×3 核什么也勾不出来。
- **深度是合成的。** 没有噪声、没有缺失。写错 0.3 米就能把占用挪开，真实深度相机两者都有。
- **CEM 的世界是一条线。** \(x_{t+1}=x_t+a_t\) 没有动力学误差。真正的 CEM 在学到的模型里打分，模型错了，精英也会错。
- **2.4 的一步准确率 0.83 含打滑。** 不要把它写成「模型没学好」。先看 `slip_probability`。
- **`lookahead` 默认四个动作。** 在 LineWorld 里会搜到 `up` / `down`。写自己的 Planner 时，把 `action_order=world.actions` 传进去。
- **块平均的 MSE 691 和复制帧的 0.006 不在同一个尺度。** 前者像素在 0–255，后者通常先除以 255。比较之前先统一。

## 扩展练习

按从便宜到昂贵：

1. **patch 扫描**：边长 2 / 4 / 8，画 token 数和块平均重建误差。有没有一个「既不太碎、边又没被抹掉」的点？
2. **标定扫描**：平移误差从 0.05 m 扫到 1.0 m，看占用错几格。几何多准才「够用」？
下一步是 [2.4 动手：第一台可学习世界模型](/chapters/02-data-and-first-model/02-04-learn-a-table-world)，然后是 [PA0](/assignments/pa0)。那里不再给你完整转移表，你要自己留一个缺口，并让一种失败稳定出现。

## 本节小结

- **卷积和 patch 读图**：3×3 核没有训练也能勾边；切 patch 本身不压缩，块平均才压缩，重建 MSE 在 0–255 尺度上是 691。
- **同一张末帧可以对应两种历史**：速度必须写进状态，否则当前观察不够用。
- **从深度走进三维**：36 个点、平移 \([1,0,0]\)、占用 8 格；写成 1.3 米就偏 0.300 米。几何错了，后面的网络救不回来。
- **CEM 在一条线上就能看懂精英迭代**：五步加起来到 3.0，分数一轮轮变好。
- **2.4 从轨迹里数动态**：140 段、1494 条转移，一步准确率 0.83；MPC 第一步打滑，后面仍然走到终点。
- **Smoke 不是完整训练**：NumPy、CPU、几秒。目标是看清接口和失败，不是复现大型世界模型。

从 0.6 的九格网格到 2.4 的打滑直线，从手写字典到计数分布——核心思想没有变：**在行动之前，先在内部预见行动的后果。** 这一节你亲手把看见、空间和动态接了起来。

## 后续工作

1.8 与 2.4 只把零件放到正确的位置。后面三条线都从这里分叉。

**卷积变成 Encoder。** 3.6 的 ConvVAE、3.7 的 `PixelEncoder`，第一步仍是滑动窗口，只是核变成了可学习的。ViT 则从本节的 `patchify` 出发，不再扫核。

**点云变成 BEV 和神经场。** 第 7 章的 LSS、Occupancy、tiny NeRF，都站在这三行反投影上。标定写错一米，后面所有 IoU 都不可信。

**计数变成 RSSM。** 2.4 的 \(\hat P(s'|s,a)\) 在离散格子里够用。状态变成 80 维特征之后，同一句话变成 prior / posterior。MPC 的「只执行第一步」则一直活到 PlaNet 的 CEM 和 Dreamer 的想象更新。

这些方法的骨架，就是你刚刚跑通的两份 Notebook。

## 参考文献

1. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) —— ViT：把图片切成 patch 的原始论文。
2. Rubinstein, R. Y. (1999). The Cross-Entropy Method for Combinatorial and Continuous Optimization. *Methodology and Computing in Applied Probability*, 1, 127–190. [链接](https://doi.org/10.1023/A:1010091220143) —— CEM 的原始论文。
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [链接](http://incompleteideas.net/book/the-book.html) —— 第 8 章讲 Dyna 与规划，2.4 的计数动态从这里来。
4. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. *arXiv:2301.04104*. [链接](https://arxiv.org/abs/2301.04104) —— DreamerV3：symlog 与梯度裁剪写进了标准训练配方。
5. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. —— 针孔相机与反投影的标准参考。
