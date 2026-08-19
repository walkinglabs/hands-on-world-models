# PA1-E · 动手：空间世界二选一

> **本节目标**：先完成 E1 的共同基础（深度反投影、坐标变换、BEV Occupancy），再选择 E2a（3D/4D 动态场）或 E2b（驾驶 Occupancy 预测）中的一个方向，完成一次完整的空间世界模型实验。不是重建最漂亮的场景，而是用证据回答「模型真的理解了三维空间吗？」
>
> **本节代码**：[E1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_spatial/E1-from-camera-to-space.ipynb) · [E2a Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_spatial/E2a-build-a-small-4d-world.ipynb) · [E2b Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_spatial/E2b-predict-driving-space.ipynb) · [spatial.py](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/spatial.py) · [foundations.py](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/foundations.py)
>
> **前置知识**：你已经跑过路线 E 的 E1（相机几何 + BEV Occupancy）、E2a（神经场 smoke）或 E2b（占用预测 smoke）。PA1-E 把它们扩成可评价的训练，并写清静态 / 动态 / 可控、开环 / 闭环的结论边界。

---

E1 不训练大模型。它只做一件事：把针孔几何算对。6×6 深度图反投影得到 36 个点，相机右移 1 米后面均值平移正好是 \([1, 0, 0]\)，落到 0.5 m 网格上占用 8 格。把平移写成 1.3 m，点云平均误差 0.3 m，占用 IoU 从 1.0 掉到 0.5。

E2a / E2b 的 smoke 只证明接口连通：静态场 loss 能降，动态场换时间后密度会变，占用预测的 IoU 能超过复制上一帧。这些数字离「理解三维空间」还差很远。

PA1-E 的任务是：**用完整训练回答「模型真的理解了三维空间吗？」** 你会看见 novel-view 在训练视角内还行、外推崩溃；占用预测短期有 IoU、长期漂移；标定写错一截，后面的网络救不回来。

## 为什么 PA1-E 是二选一

路线 E 有两条分叉，一次只走一条：

| 项目 | E2a（3D/4D 动态场） | E2b（驾驶 Occupancy） |
| ---- | ------------------- | --------------------- |
| 观测 | 坐标查询；PA 可升级到多视角 RGB | 历史 BEV；PA 可升级到多相机 |
| 表示 | 神经场 \((x,y,z)\) 或 \((x,y,z,t,a)\) | \(16\times 16\) 占用网格 |
| 预测 | 密度 / 颜色 | 未来占用 |
| 评价 | 场拟合误差 + 反事实密度差 | IoU + 复制/匀速基线 + 动作敏感性 |
| 应用 | 场景表示、新视角 | 开环驾驶预测 |

**不要把两个项目各做一半。** 选一种，完整提交数据—模型—预测—评价。E1 对两条路都是必做。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/pa1e-unproject.png" alt="标定误差导致占用错位" style="max-width:min(900px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">同一张 6×6 深度图。左：平移 1.0 m，占用 8 格；右：写成 1.3 m，占用 7 格，IoU 掉到 0.5。几何错了，后面的网络适应不了错误的相机。</div>
</div>

## 本次会得到什么

这是作业。E1 人人都交；E2a / E2b 只交你选的那一支。

**E1 必交**

- 反投影点云 shape、近/远 \(z\)
- 外参平移的均值差，必须能复现 \([1, 0, 0]\)
- BEV 网格 shape 与占用格数
- 把平移写成 1.3 m 后的点云误差和占用 IoU

**E2a 必交（若选这条）**

- 静态场：样本数、占用比例、loss 曲线、密度/颜色 MSE、参数量
- 动态场：\((x,y,z,t,a)\) 查询、loss 曲线、固定坐标只换时间/动作的密度差
- 结论边界表：静态 / 动态 / 可控，哪一条你有资格声称
- 若声称新视角或几何精度，另交 PSNR / 深度误差；没有体渲染就不要报 PSNR

**E2b 必交（若选这条）**

- 历史 / 动作 / 未来的 shape，以及占用比例
- 学习预测 IoU，以及复制上一帧、匀速外推两个基线
- 同一历史换动作后的占用差
- horizon 曲线（至少 1 / 2 / 3 步）
- 结论边界：只能称开环预测，除非你另有模拟器

## 怎样运行

```bash
python -m pip install -r requirements-neural.txt   # E2a / E2b 需要
jupyter lab
# notebooks/07_spatial/E1-from-camera-to-space.ipynb
# 然后只打开 E2a 或 E2b 其中一个
```

```bash
PYTHONPATH=src python -m unittest tests.test_routes_de tests.test_foundations -v
```

E1 只用 NumPy。E2a / E2b 用 PyTorch，CPU 就能跑 smoke。

## 第一步：E1 共同基础

### 1.1 深度像素怎样变成三维点

针孔反投影是从 2D 走到 3D 的第一公式：

$$
X = \frac{(u-c_x)\,Z}{f_x},\quad
Y = \frac{(v-c_y)\,Z}{f_y},\quad
Z = \mathrm{depth}(u,v)
$$

```python
from hwm.foundations import depth_to_points

depth = np.full((6, 6), 4.0, dtype=np.float32)
depth[2:4, 2:4] = 2.0
points = depth_to_points(depth, fx=6, fy=6, cx=2.5, cy=2.5)
```

**真实判据**：`points.shape == (36, 3)`，近/远 \(z\) 是 **2.0 / 4.0**。\(X,Y\) 范围大约 \([-1.67, 1.67]\)。如果所有点落在一个平面上，多半是 \(Z\) 写成了常数，或 \((u,v)\) 和 \((x,y)\) 对调了。

### 1.2 外参把不同相机放进同一世界

```python
from hwm.foundations import make_camera_transform, transform_points

world = transform_points(points, make_camera_transform(tx=1.0, yaw=0.0))
print(world.mean(0) - points.mean(0))
# [1.  0. -0.]
```

齐次矩阵左乘点。平移写在最后一列，偏航绕 \(y\)。**真实判据**：均值平移必须是 \([1, 0, 0]\)，容差 \(10^{-5}\)。对不上，先查矩阵是 `T @ p` 还是 `p @ T`，再查相机系和世界系的轴。

### 1.3 点落到俯视 Occupancy

Occupancy 不看颜色，只记哪里有东西。E1 用 \(x\in[-2,4]\)、\(z\in[0,6]\)、分辨率 0.5 m，得到 **\((12, 12)\) 网格、8 个占用格**。近处 \(z=2\) 的 4 个像素挤进两格，远处 \(z=4\) 的像素摊得更开——这就是透视。

### 1.4 标定误差会怎样

```python
wrong = transform_points(points, make_camera_transform(tx=1.3))
err = np.linalg.norm(wrong - world, axis=1).mean()
# 0.3 m
```

**真实判据**：点云平均误差 **0.3 m**；错误占用 7 格；与正确占用的 IoU **0.5**。旧讲义里的 `occupancy_misalignment: 0.25` 对不上当前网格，不要再抄。神经网络可以在训练集上适应固定偏差，却不能把错误几何变正确。

## 第二步：选择方向

E1 通过后再选。选 E2a 就不要再交半成品 E2b，反过来也一样。

---

## E2a：构建一个小型 4D 世界

教学实现不是完整 NeRF。`TinyNeuralField` 是坐标 \(\rightarrow\)（密度, 颜色）的 3 层 MLP，约 **2,740** 个参数，没有射线、没有体渲染。`TinyDynamicField` 把时间和 5 类动作嵌进去，约 **5,292** 个参数。

体积渲染的积分才是 NeRF 原文的核心 [2]：

$$
C(r) = \int_{t_n}^{t_f} T(t)\,\sigma(r(t))\,c(r(t), d)\,dt,\quad
T(t)=\exp\Bigl(-\int_{t_n}^{t}\sigma(r(s))\,ds\Bigr)
$$

PA 如果只拟合坐标查询，就报场拟合误差，不要报 PSNR。只有你自己加了射线采样和体渲染，才有资格写 PSNR / SSIM。

### 2a.1 静态场

```python
from hwm.spatial import TinyNeuralField, make_colored_sphere_samples

coords, density, color = make_colored_sphere_samples(640, seed=0)
# occupied fraction = 0.134
```

样本是半径 0.65 的彩色球。**真实判据（80 步、Adam \(5\times 10^{-3}\)）**：场 loss 从 **0.609 降到 0.089**；拆开是密度 MSE 0.071、颜色 MSE 0.016。

### 2a.2 加入时间与动作

`make_moving_sphere_samples(1024, seed=1)` 让球心随 \((t, a)\) 移动。占用比例只有 **0.029**，比静态球稀疏得多。动态 loss（BCE + 颜色 MSE）从 **1.005 降到 0.087**。

固定坐标 \((0.35, 0, 0)\)，五个动作在 \(t=0\) 和 \(t=1\) 的密度是：

```text
t=0 : 0.052, 0.057, 0.079, 0.002, 0.102
t=1 : 0.118, 0.131, 0.098, 0.006, 0.165
mean |Δ| = 0.045
```

差大于 0，只说明时间和动作进了查询。它**不**证明运动方向对，更不证明可控。

### 2a.3 结论边界

| 声称 | 你必须交出的证据 |
| ---- | ---------------- |
| 静态重建 | 场拟合误差；若有体渲染，再加 PSNR / 深度误差 |
| 动态重建 | 时间变化后误差仍可控；固定坐标换 \(t\) 密度会变 |
| 可控 | 固定历史，只换动作，未来必须按动作方向分开 |

换动作后未来不变，就只能称动态重建，不能称可控。3D Gaussian Splatting [3] 是另一种显式表示，渲染快，但同样不会自动带上动作动态。

---

## E2b：预测驾驶空间

驾驶路线不以「下一帧好看」为目标。`TinyOccupancyPredictor` 把过去 3 帧 \(16\times 16\) 占用和 5 类 ego action 变成未来 3 帧占用，约 **12,587** 个参数。动作 embedding 铺到整个 BEV，再和历史帧一起进卷积——这是 LSS [1]「先抬到 3D，再落到 BEV」在教学尺度上的亲戚，不是完整 LSS。

```python
from hwm.spatial import (
    TinyOccupancyPredictor, make_moving_occupancy_dataset, occupancy_iou,
)

history, actions, future = make_moving_occupancy_dataset(96, seed=1)
# (96, 3, 16, 16), (96,), (96, 3, 16, 16)
# occupancy fraction = 0.035
```

正样本极少，所以训练用 `pos_weight=18` 的 BCE。IoU 定义为

$$
\mathrm{IoU} = \frac{|\{p>0.5\}\cap\{y=1\}|}{|\{p>0.5\}\cup\{y=1\}|}
$$

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/pa1e-occupancy.png" alt="占用预测相对复制基线" style="max-width:min(860px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">96 个样本上，学到的未来占用 IoU 0.436，复制上一帧只有 0.277。有增益，但离「能规划」还远。</div>
</div>

**真实判据（80 步、seed=1）**：

| 指标 | 数值 |
| ---- | ---: |
| loss | 1.127 → 0.253 |
| 学习预测 IoU | 0.436 |
| 复制上一帧 IoU | 0.277 |
| 换动作后的占用差 | 0.085–0.110 |

同一历史换 5 个动作，相对动作 0 的平均绝对差都大于 0。没有 ego action 的数据只能训练开环未来预测，不能称动作条件世界模型。

PA1-E 还要画 horizon 曲线：把未来 1 / 2 / 3 步的 IoU 拆开，不要只报一个 3 步平均。再加一个匀速外推基线——把历史最后一帧的速度沿用下去。学习模型若赢不了这两个傻子，就不要写「学到了动态」。

**没有模拟器，不能报闭环碰撞率。** 开环 IoU 再高，也不等于车不会撞。

## 共同 24GB 目标

降低场景、相机数、分辨率、射线/高斯或 BEV 网格，单卡 reserved 设计目标不超过 22GB。每个分支都要独立实测。当前记录为 0，不得标成已验证。

## 评分

| 项目 | 分数 | 检查重点 |
| ---- | ---: | -------- |
| E1 几何 | 25 | 36 点、平移 \([1,0,0]\)、占用 8 格、0.3 m / IoU 0.5 的标定实验 |
| 所选分支的训练 | 25 | 曲线、参数量、shape 与代码一致 |
| 基线与反事实 | 20 | E2a 换时间/动作；E2b 复制帧 + 匀速 + 换动作 |
| 结论边界 | 20 | 静态/动态/可控，或开环/闭环，不越界声称 |
| 表达与复现 | 10 | Notebook 可运行；seed 与输出完整 |

选错分支、两个各做一半，按未完成计。把 E2a 的坐标查询叫做「新视角合成」，或把 E2b 的离线 IoU 叫做「闭环驾驶」，该项零分。

## 已知简化与坑

- **E1 的深度是合成的**，没有噪声、没有缺失。真实深度相机两者都有。
- **占用格数对标定极其敏感**。0.3 m 平移就能让 IoU 腰斩。
- **E2a 的 moving-sphere 是 4D 查询 toy**，不是多视角场景重建，也不是 3DGS。
- **E2b 的占用比例只有 3.5%**。不用 `pos_weight`，模型会预测全空，IoU 接近 0。
- **复制上一帧是很强的基线**。慢速场景里，学得不好的网络经常输给它。
- **24GB 是设计目标**，当前未完整实测。

## 本节小结

- **PA1-E 先把几何算对，再谈网络。** 36 个点、1 m 平移、8 个占用格、0.3 m 标定误差，是后面一切的尺子。
- **E2a 和 E2b 走不同的路**：坐标场重建场景，占用网格预测驾驶空间。一次只走一条。
- **静态、动态、可控是三种结论。** smoke 里时间密度差 0.045，只跨过「动态」的门槛。
- **开环预测不是闭环规划。** E2b 的 IoU 0.436 超过复制帧 0.277，仍然不能报碰撞率。
- **标定错误救不回来。** 写错 0.3 m，占用 IoU 直接掉到 0.5。

从 E1 的合成深度到 PA1-E 的完整训练，空间世界走的是一条和其他路线不同的路：先把二维图像变成三维可规划的东西，再决定你要重建外观，还是预测占用。

## 参考文献

1. Philion, J., & Fidler, S. (2020). Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D. *ECCV 2020*. [arXiv:2008.05711](https://arxiv.org/abs/2008.05711) —— 多相机「抬起—泼溅—射击」到 BEV 的原始方法。
2. Mildenhall, B., et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV 2020*. [arXiv:2003.08934](https://arxiv.org/abs/2003.08934) —— 神经辐射场与体渲染积分。
3. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *SIGGRAPH 2023*. [项目页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) —— 显式高斯的实时辐射场。
4. Huang, Y., et al. (2023). Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction. *CVPR 2023*. [arXiv:2302.07817](https://arxiv.org/abs/2302.07817) —— TPVFormer：用三平面做语义占用预测。
5. Wang, X., et al. (2023). DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. *arXiv:2309.09777*. [链接](https://arxiv.org/abs/2309.09777) —— 用世界模型做驾驶视频生成与数据增强。
