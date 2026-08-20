# 7.6 动手：空间世界实验

> **本节目标**：先把深度像素算进同一个三维世界，再选 4D 动态场或驾驶 Occupancy，把「空间」变成可以查询、可以预测的东西。

> **本节代码**：[E1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_spatial/E1-from-camera-to-space.ipynb) · [E2a Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_spatial/E2a-build-a-small-4d-world.ipynb) · [E2b Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_spatial/E2b-predict-driving-space.ipynb) · [spatial.py](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/spatial.py)

> **前置知识**：你已经读过 7.1–7.5，知道针孔相机、深度反投影、BEV Occupancy。最好刚跑完 [6.5 动手：机器人与 VLA 实验](/chapters/06-robot-vla/05-robot-vla)。这一节把几何真算一遍。

---

6.5 的手还在桌子上走。现在世界变成一张深度图、一块会动的俯视占用。

自动驾驶需要知道「向右打方向之后，前方哪个格子会被占住」。开放道路上你不可能把所有场景都录一遍，必须让模型自己想象没见过的未来。

2018 年的 World Models 是在梦里学会开车。同一年之后，DriveDreamer 把这句话搬到了路上：历史画面加上候选方向盘，模型交出未来几秒的道路。规模差了几个数量级，骨架没有变。

这一节规模打折，原理不打折。先把深度图片算进同一个三维世界，再选 4D 小场景或驾驶 Occupancy，二选一往下走。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/de-spatial-world.png" alt="空间世界" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">这就是我们要预测的「世界」：16×16 俯视占用里有一块会动的方块。过去三帧告诉你它怎么走，一个离散动作告诉你接下来往哪偏，未来三帧是监督。颜色没有了，只剩「这里有没有东西」。</div>
</div>

## 本次会得到什么

运行结束后，你会得到：

- 6×6 深度图反投影出 36 个点，平移 1 米后落入 12×12 Occupancy 的 8 个格子
- 外参写成 1.3 米，点云平均偏 0.300 米，占用错开 5 格
- 静态场损失 \(0.609 \rightarrow 0.089\)，动态场 \(1.005 \rightarrow 0.087\)
- 未来 Occupancy：loss \(1.127 \rightarrow 0.253\)，总 IoU 0.436，三步分别是 0.535 / 0.490 / 0.327

## 怎样运行

```text
notebooks/07_spatial/E1-from-camera-to-space.ipynb
notebooks/07_spatial/E2a-build-a-small-4d-world.ipynb
notebooks/07_spatial/E2b-predict-driving-space.ipynb
```

E2a / E2b 需要 PyTorch；E1 只需要 NumPy。

```bash
python -m pip install -r requirements-neural.txt
```

即使暂时不跑 Notebook，也可以先跑测试：

```bash
PYTHONPATH=src python -m unittest tests.test_routes_de -v
```

## 第一步：先把几何算对

上一章还在模仿专家手臂。这一节换一个问题：一张深度图里的像素，怎样变成可以规划的空间？

E1 不训练大模型。内参 \(f_x, f_y, c_x, c_y\) 描述针孔怎样成像。已知像素 \((u, v)\) 的深度 \(z\)，反投影到相机坐标是：

$$
x = \frac{(u - c_x)\, z}{f_x},\qquad
y = \frac{(v - c_y)\, z}{f_y}
$$

这就是 `depth_to_points` 在做的事。一张照片本身给不出 \(z\)，一个像素只确定一条射线；有了深度，射线上才钉得住一个点。

```python
import numpy as np
from hwm.foundations import depth_to_points

depth = np.full((6, 6), 4.0, dtype=np.float32)
depth[2:4, 2:4] = 2.0
points_camera = depth_to_points(depth, fx=6, fy=6, cx=2.5, cy=2.5)
print('points:', points_camera.shape,
      'near/far z:', points_camera[:, 2].min(), points_camera[:, 2].max())
```

**运行这一步，你会看到什么？**

```
points: (36, 3) near/far z: 2.0 4.0
```

6×6 每个像素一个点。中间 2×2 深度是 2，其余是 4。中心四个点落在

```
(-0.167, -0.167, 2)
( 0.167, -0.167, 2)
(-0.167,  0.167, 2)
( 0.167,  0.167, 2)
```

角落 \((0,0)\) 和 \((5,5)\) 是 \((\pm 1.667, \pm 1.667, 4)\)。深度减半，同样的像素偏移对应的平面距离也减半——这就是透视。

## 第二步：外参把相机放进同一个世界

相机向右平移 1 米。齐次变换是

$$
\begin{bmatrix} x_w \\ y_w \\ z_w \\ 1 \end{bmatrix}
=
\begin{bmatrix}
\cos\psi & 0 & \sin\psi & t_x \\
0 & 1 & 0 & t_y \\
-\sin\psi & 0 & \cos\psi & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x_c \\ y_c \\ z_c \\ 1 \end{bmatrix}
$$

\(\psi=0\) 时，它就是给每个点的 \(x\) 加 \(t_x\)。矩阵乘法写反、或者把「相机到世界」和「世界到相机」弄混，多视角会对不齐。

```python
from hwm.foundations import make_camera_transform, transform_points

camera_to_world = make_camera_transform(tx=1.0, yaw=0.0)
points_world = transform_points(points_camera, camera_to_world)
shift = points_world.mean(0) - points_camera.mean(0)
print('mean shift:', np.round(shift, 3))
```

**运行这一步，你会看到什么？**

```
mean shift: [1.  0. -0.]
```

整朵点云整体右移 1 米。\(y\) 和 \(z\) 几乎不动。如果这里出现 \([-1, 0, 0]\) 或某个奇怪的旋转，先回头看变换矩阵，不要先怪 Occupancy。

## 第三步：点落到俯视 Occupancy

Occupancy 不关心表面颜色，只记录空间哪里已经有东西。把世界坐标的 \(x\) 和 \(z\) 按分辨率切成格子：

$$
i_x = \Bigl\lfloor \frac{x - x_{\min}}{\Delta} \Bigr\rfloor,\qquad
i_z = \Bigl\lfloor \frac{z - z_{\min}}{\Delta} \Bigr\rfloor
$$

落在范围内的格子记 1。教学版分辨率 \(\Delta=0.5\) 米，\(x\in[-2,4]\)，\(z\in[0,6]\)，所以网格是 \(12\times 12\)。

```python
from hwm.foundations import points_to_occupancy

occupancy = points_to_occupancy(
    points_world, x_range=(-2, 4), z_range=(0, 6), resolution=0.5
)
print('occupancy shape/occupied:', occupancy.shape, int(occupancy.sum()))
```

**运行这一步，你会看到什么？**

```
occupancy shape/occupied: (12, 12) 8
```

36 个点挤进 8 个格子。近处那 4 个点（\(z=2\)）和远处那圈（\(z=4\)）不在同一行——深度不同，俯视上的前后位置就不同。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/de-depth-occupancy.png" alt="深度到占用" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">左：6×6 深度，中间亮块是 2 米，四周是 4 米。中：反投影后的 36 个相机坐标，中间四个大点是近处。右：平移 1 米之后落入俯视网格，8 个红格被点云点亮。</div>
</div>

这 8 个格子就是后面碰撞检查和驾驶预测要吃的东西。Philion 与 Fidler 的 Lift-Splat-Shoot 做的是同一件事的可学习版本：从多相机图像「抬」到 3D，再「摊」到 BEV。E1 先用手算把几何钉死。

## 第四步：外参写错 0.3 米

把平移写成 1.3 米，整朵点云再偏 0.3 米。神经网络可以在训练集上记住这个固定偏差，却不能让错误几何变正确——换一个场景，偏的还是 0.3 米。

```python
wrong_world = transform_points(
    points_camera, make_camera_transform(tx=1.3)
)
calibration_error = np.linalg.norm(wrong_world - points_world, axis=1).mean()
print('mean calibration error:', round(float(calibration_error), 3), 'm')
```

**运行这一步，你会看到什么？**

```
mean calibration error: 0.3 m
```

错 Occupancy 占用 7 格，和正确的那张重叠 5 格，对不齐 5 格。分辨率是 0.5 米，0.3 米的平移刚好够把一部分点推进隔壁格子。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/de-calibration.png" alt="标定误差" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">左：平移 1.0 米，8 格。中：平移 1.3 米，7 格，整块右移。右：两张网格的差异，5 格对不齐。后面无论用 OccNet 还是 3D 高斯，标定错了，预测都会跟着错。</div>
</div>

**一个值得做的实验**：把 \(t_x\) 的误差从 0 扫到 1.0 米，数对不齐的格子。分辨率是 0.5 米，误差过了半格，Occupancy 就会跳。神经网络救不了这一格。

E1 到此结束。相机内参负责从像素到射线，外参负责从相机到世界，Occupancy 把三维点变成可规划网格。完成之后，在 E2a 神经场和 E2b 未来占用里选一份。

## 第五步：先拟合一颗不会动的球

NeRF 的第一步是学一个函数：给定空间坐标，交出密度和颜色。教学版不沿相机射线做体渲染，只在坐标上监督——接口是真的，渲染是省掉的。

样本在 \([-1,1]^3\) 里均匀撒 640 个点。半径小于 0.65 的点密度为 1，颜色按坐标从黑拉到白；球外密度和颜色都是 0。占用大约 13%。

```python
from hwm.spatial import TinyNeuralField, make_colored_sphere_samples

coordinates, density, color = make_colored_sphere_samples(640, seed=0)
print('coordinates/density/color:',
      tuple(coordinates.shape), tuple(density.shape), tuple(color.shape))
print('occupied fraction:', round(float(density.mean()), 3))
```

**运行这一步，你会看到什么？**

```
coordinates/density/color: (640, 3) (640, 1) (640, 3)
occupied fraction: 0.134
```

静态场是三层 MLP，约 2700 个参数。密度走 `softplus`，颜色走 `sigmoid`。损失是密度 MSE 加颜色 MSE：

$$
\mathcal{L}_{\text{static}}
= \bigl\| \hat\sigma(x) - \sigma^*(x) \bigr\|_2^2
+ \bigl\| \hat c(x) - c^*(x) \bigr\|_2^2
$$

```python
field = TinyNeuralField()
opt = torch.optim.Adam(field.parameters(), lr=5e-3)
losses = []
for _ in range(80):
    opt.zero_grad()
    predicted_density, predicted_color = field(coordinates)
    loss = (
        F.mse_loss(predicted_density, density)
        + F.mse_loss(predicted_color, color)
    )
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
print('field loss:', round(losses[0], 3), '→', round(losses[-1], 3))
```

**运行这一步，你会看到什么？**

```
field loss: 0.609 → 0.089
```

拆开最后一步：密度 MSE 0.071，颜色 MSE 0.016。只在球内点上算颜色，MSE 是 0.102，对应 PSNR 大约 9.9 dB——糊成一团。真正的 NeRF 在真实多视角图像上能到 30 dB 以上，那是沿射线积分、分层采样、位置编码之后的事。这里只证明：坐标进网络，密度和颜色能往监督靠。

Mildenhall 等人的 NeRF 把这个函数写成 \(\mathrm{MLP}(x, d)\rightarrow(\sigma, c)\)，再沿视线做体渲染。Kerbl 等人的 3D 高斯溅射换成一堆显式椭球，渲染快几个数量级。E2a 两者都不实现，只留下「给一个点，问这里有没有东西」这句话。

## 第六步：把时间和动作写进查询

静态函数只有 \((x,y,z)\)。现在生成一颗会走的小球：五个离散动作对应五个位移方向，球心是 \(\mathrm{move}(a)\cdot t\)。半径 0.38，比静态球更小，1024 个点里大约只有 3% 落在球内。

动态场把坐标、时间和动作 embedding 拼成 \(3+1+8=12\) 维，再映射到密度和颜色。密度这里改成 `sigmoid`，损失也改成 BCE 加颜色 MSE：

$$
\mathcal{L}_{\text{4D}}
= \mathrm{BCE}\bigl(\hat\sigma(x,t,a),\, \sigma^*\bigr)
+ \bigl\| \hat c(x,t,a) - c^* \bigr\|_2^2
$$

注意：静态场和动态场的密度激活不一样。对照损失曲线时，不要把 0.089 和 0.087 当成同一个指标。

```python
from hwm.spatial import TinyDynamicField, make_moving_sphere_samples

coordinates_4d, times, actions, density_4d, color_4d = (
    make_moving_sphere_samples(1024, seed=1)
)
dynamic_field = TinyDynamicField()
opt = torch.optim.Adam(dynamic_field.parameters(), lr=5e-3)
dynamic_losses = []
for _ in range(100):
    predicted_density, predicted_color = dynamic_field(
        coordinates_4d, times, actions
    )
    loss = (
        F.binary_cross_entropy(predicted_density, density_4d)
        + F.mse_loss(predicted_color, color_4d)
    )
    opt.zero_grad()
    loss.backward()
    opt.step()
    dynamic_losses.append(float(loss.detach()))
print('dynamic loss:',
      round(dynamic_losses[0], 3), '→', round(dynamic_losses[-1], 3))
```

**运行这一步，你会看到什么？**

```
dynamic loss: 1.005 → 0.087
```

100 步之后损失掉了一个数量级。球很小、正样本很少，BCE 一开始接近「全部预测成空」的值，后来才慢慢把那 3% 的球心摸出来。这仍然是坐标查询，不是多视角重建，更不是 4D 高斯。

## 第七步：固定坐标，只换时间和动作

动作条件模型至少应在条件变化时给出不同的查询结果。差异本身还不证明运动正确；结果完全相同一定有问题。

把查询钉在 \((0.35, 0, 0)\)，五个动作各问一次 \(t=0\) 和 \(t=1\)：

```python
query = torch.tensor([[0.35, 0.0, 0.0]]).expand(5, -1)
with torch.no_grad():
    early_density, _ = dynamic_field(query, torch.zeros(5), torch.arange(5))
    late_density, _ = dynamic_field(query, torch.ones(5), torch.arange(5))
print('t=0 density:', [round(float(x), 3) for x in early_density])
print('t=1 density:', [round(float(x), 3) for x in late_density])
```

**运行这一步，你会看到什么？**

```
t=0 density: [0.052, 0.057, 0.079, 0.002, 0.102]
t=1 density: [0.118, 0.131, 0.098, 0.006, 0.165]
```

平均绝对差 0.045。五个动作里，「左」几乎一直是空的，「右」从 0.10 升到 0.17。球心按 \(t\cdot\mathrm{move}(a)\) 走，\(t=0\) 时所有动作的球心都在原点，按理密度应该接近；模型没有把这一点学干净，但至少时间和动作已经进了前向。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/de-4d-query.png" alt="4D 查询" style="max-width:min(760px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">同一坐标 (0.35, 0, 0)。浅柱是 t=0，深柱是 t=1。五个动作的密度都变了，左边几乎空，右边最高。这只说明查询接口吃进了时间和动作，不说明球沿着正确的轨迹在走。</div>
</div>

NeRF、3DGS、Mesh 是三种空间表示：连续场平滑但慢，高斯快，三角形方便碰撞。它们都不自动包含动作动态。E2a 交出的是两个最小接口——静态场和动作条件动态场。PA1-E 再补相机射线、体渲染和多视角。

## 第八步：预测未来哪一格会被占住

驾驶路线不以「下一帧好看」为主要目标。规划器想知道的是：接下来几步，哪些格子里会有东西。E2b 把过去三帧俯视占用和候选 ego action 变成未来三帧占用。

数据是 16×16 的小方块。过去三帧里方块按一个常速度挪；然后换一个离散动作（停、上、下、左、右），再走三帧作为未来。占用大约 3.5%——绝大多数格子是空的。

```python
from hwm.spatial import make_moving_occupancy_dataset

history, actions, future = make_moving_occupancy_dataset(96, seed=1)
print('history/action/future:',
      tuple(history.shape), tuple(actions.shape), tuple(future.shape))
```

**运行这一步，你会看到什么？**

```
history/action/future: (96, 3, 16, 16) (96,) (96, 3, 16, 16)
```

96 条、三帧历史、三帧未来。动作五个取值大致均匀（15 / 21 / 21 / 19 / 20）。历史和未来必须分开：混在一起，模型会靠「复制最后一帧」混分数。

## 第九步：动作铺到整张 BEV 上

预测器大约 1.3 万个参数。动作先变成 8 维 embedding，再复制到每一个格子上，和 3 帧历史拼成 11 个通道，两层 3×3 卷积之后用 1×1 卷积吐出 3 张未来 logit。

空格子远多于占用，不加权的 BCE 会让模型全部答「空」。Notebook 给正类乘上 18：

$$
\mathcal{L}
= \mathrm{BCEWithLogits}\bigl(
    M(O_{t-2:t}, a),\;
    O_{t+1:t+3}
\bigr),\qquad
w_{+}=18
$$

评价用 IoU，阈值 0.5：

$$
\mathrm{IoU}
= \frac{\lvert \hat O \cap O \rvert}{\lvert \hat O \cup O \rvert}
$$

```python
from hwm.spatial import TinyOccupancyPredictor, occupancy_iou

model = TinyOccupancyPredictor()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
positive_weight = torch.tensor(18.0)
losses = []
for _ in range(80):
    opt.zero_grad()
    logits = model(history, actions)
    loss = F.binary_cross_entropy_with_logits(
        logits, future, pos_weight=positive_weight
    )
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
iou = occupancy_iou(logits.detach(), future)
print('loss:', round(losses[0], 3), '→', round(losses[-1], 3),
      'IoU:', round(float(iou), 3))
```

**运行这一步，你会看到什么？**

```
loss: 1.127 → 0.253  IoU: 0.436
horizon 1 IoU: 0.535
horizon 2 IoU: 0.490
horizon 3 IoU: 0.327
```

第一步还能对上大约一半的占用格，第三步掉到 0.327。复合误差在 Occupancy 里一样会发生：每一步的方块位置偏一点，三步之后交并比就塌了。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/de-occupancy-future.png" alt="未来占用预测" style="max-width:min(900px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">同一条样本，动作是 right。上排是过去，中排是真未来，下排是预测。前两步方块还在，第三步摊成一条红带——模型开始不敢确定它会停在哪一格。总 IoU 0.44，不是 0.67，更没有 0.89 的「碰撞检测准确率」。</div>
</div>

## 第十步：同一段历史，换一个动作

如果数据真的包含动作条件未来，固定历史只换动作，预测的占用应该变。没有 ego action 的数据只能训练开环外推。

```python
same_history = history[:1].expand(5, -1, -1, -1)
all_actions = torch.arange(5)
with torch.no_grad():
    counterfactual = torch.sigmoid(model(same_history, all_actions))
differences = [
    (counterfactual[0] - counterfactual[i]).abs().mean().item()
    for i in range(1, 5)
]
print('换动作后的 occupancy 差异:', [round(x, 4) for x in differences])
```

**运行这一步，你会看到什么？**

```
换动作后的 occupancy 差异: [0.1103, 0.1003, 0.0854, 0.0974]
```

四个对照动作相对「停」，平均每个格子差大约 0.1。差异不是零。它不证明预测的方块走对了方向，只证明动作进了卷积。DriveDreamer 和 GAIA-1 用的是同一条最低门槛：换方向盘，生成的未来必须跟着变。

IoU 只检查离线占用。碰撞率、驶出道路率、安全接管，需要 Planner、车辆动力学和可交互模拟器。本 Notebook 不报告这些指标，也不把 0.436 的 IoU 写成闭环驾驶。

**一个值得做的实验**：把未来从 3 步提到 8 步，画出逐步 IoU。你会再次看见 0.6 和 3.6 里那张图：单步还行，多步很快糊掉。占用网格里的复合误差，和梦境画面糊掉，是同一件事。

## 运行与产物

```bash
python -m pip install -r requirements-neural.txt
PYTHONPATH=src python -m unittest tests.test_routes_de -v
```

跑完三份 Notebook 后，你应该有：

- **E1**：36 个点、平移 \([1,0,0]\)、占用 8 格、标定误差 0.300 m
- **E2a**：静态场 \(0.609\rightarrow 0.089\)，动态场 \(1.005\rightarrow 0.087\)，同一坐标换 \(t\) 密度会变
- **E2b**：loss \(1.127\rightarrow 0.253\)，IoU 0.436，三步 0.535 / 0.490 / 0.327

## 已知简化与坑

- **E1 的深度是合成的。** 没有噪声、没有空洞、没有畸变。真实深度相机每一项都会让 Occupancy 多出一圈毛边。
- **E2a 不是多视角重建。** moving-sphere 是坐标查询。没有射线、没有 PSNR 25.6，球内颜色 PSNR 只有大约 10 dB。
- **E2b 是开环占用。** 没有车辆动力学，没有 nuScenes，不能报闭环碰撞率。
- **Smoke 不是完整训练。** CPU 运行——目标是检查数据流，不是复现 DriveDreamer。

## 扩展练习

1. **E1 的标定扫描**：把 \(t_x\) 的误差从 0 扫到 1.0 米，数对不齐的格子。分辨率 0.5 米时，误差过了半格，Occupancy 就会跳。
2. **E2a 的反事实**：固定 \((x,y,z,t)\)，只扫五个动作，看密度最大的方向是不是球心真正走的方向。
3. **E2b 的 horizon**：把未来从 3 步提到 8 步，画 IoU 曲线。你会再次看见 0.6 和 3.6 里那张图：单步还行，多步很快糊掉。

完成后进入 [PA1-E · 动手：空间世界二选一](/assignments/pa1-e)。

## 本节小结

- **先把几何算对。** 深度反投影、外参平移、俯视 Occupancy——8 个占用格、0.300 米的标定误差，都是手算出来的，不靠网络。
- **E2a 把时间和动作写进坐标查询。** 静态场和动态场的损失都能降；同一点换 \(t\) 和 \(a\)，密度会变。这不是多视角 4D 重建。
- **E2b 预测的是未来占用，不是好看的视频。** 总 IoU 0.436，三步 0.535 / 0.490 / 0.327；换动作占用会变，但开环 IoU 不是闭环驾驶。

从 3.6 的赛车到这一节的俯视网格，世界模型的身体在变，那句话没有变：**在行动之前，先在内部预见行动的后果。**

## 后续工作

占用预测还不是驾驶。E2b 的方块按五个离散动作平移。真车上要处理多相机、标定、动态障碍和自车运动学。**Lift-Splat-Shoot** 把图像抬到 BEV；**TPVFormer** 用三张正交平面表示占用；**DriveDreamer** 和 **GAIA-1** 进一步让未来视频或未来占用接受方向盘条件。换动作必须改变预测——第十步已经是这条线的最小检查。

## 参考文献

1. Philion, J., & Fidler, S. (2020). Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D. *ECCV 2020*. [arXiv:2008.05711](https://arxiv.org/abs/2008.05711) —— 从多相机图像到 BEV 的原始方法。
2. Huang, Y., et al. (2023). Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction. *CVPR 2023*. [arXiv:2302.07817](https://arxiv.org/abs/2302.07817) —— TPVFormer：用三张正交平面表示占用。
3. Tian, X., et al. (2023). Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving. *NeurIPS 2023*. [arXiv:2304.14365](https://arxiv.org/abs/2304.14365) —— 占用预测的评测基准。
4. Mildenhall, B., et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV 2020*. [arXiv:2003.08934](https://arxiv.org/abs/2003.08934) —— 坐标到密度和颜色的连续场。
5. Kerbl, B., et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM TOG (SIGGRAPH 2023)*. [arXiv:2308.04079](https://arxiv.org/abs/2308.04079) —— 显式高斯表示。
6. Pumarola, A., et al. (2021). D-NeRF: Neural Radiance Fields for Dynamic Scenes. *CVPR 2021*. [arXiv:2011.13961](https://arxiv.org/abs/2011.13961) —— 把时间写进神经场。
7. Wang, X., et al. (2023). DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. [arXiv:2309.09777](https://arxiv.org/abs/2309.09777) —— 用世界模型做驾驶数据增强。
8. Hu, A., et al. (2023). GAIA-1: A Generative World Model for Autonomous Driving. [arXiv:2309.17080](https://arxiv.org/abs/2309.17080) —— 动作条件的驾驶视频世界模型。
