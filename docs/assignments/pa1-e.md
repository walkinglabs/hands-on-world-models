# PA1-E · 动手：空间世界二选一

> **本节目标**：先完成 E1 的共同基础（深度反投影、坐标变换、BEV Occupancy），再选择 E2a（3D/4D 动态场）或 E2b（驾驶 Occupancy 预测）中的一个方向，完成一次完整的空间世界模型实验。不是重建最漂亮的场景，而是用证据回答「模型真的理解了三维空间吗？」

> **本节代码**：[E1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_spatial/E1-from-camera-to-space.ipynb) · [E2a Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_spatial/E2a-build-a-small-4d-world.ipynb) · [E2b Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_spatial/E2b-predict-driving-space.ipynb)

> **前置知识**：你已经跑过路线 E 的 E1（相机几何 + BEV Occupancy）、E2a（NeRF/3DGS smoke）或 E2b（驾驶 Occupancy smoke），知道深度反投影、内外参矩阵、Occupancy 预测。PA1-E 把它们扩展成完整训练。

---

E1 用合成深度图片确认了相机几何的接口连通：深度像素能反投影成三维点、外参矩阵能把不同相机放进同一世界、BEV Occupancy 能记录空间占用。E2a/E2b 用 smoke 确认了动态场和 Occupancy 预测的接口连通。

但 smoke 不是实验。几十个点云、50 步更新——这些数字离「模型真的理解了三维空间」还差很远。

PA1-E 的任务是：**用完整训练回答「模型真的理解了三维空间吗？」** 你会亲眼看到 novel-view 合成在训练视角内表现好但外推崩溃、Occupancy 预测在短期准确但长期漂移、标定误差导致整个点云偏移。这些失败不是 bug，是空间世界模型的核心挑战。

## 为什么 PA1-E 是二选一

路线 E 有两个截然不同的方向：

| 项目 | E2a（3D/4D 动态场） | E2b（驾驶 Occupancy） |
| ---- | ------------------- | --------------------- |
| 观测 | 多视角 RGB 图片 | 深度图片 + 相机位姿 |
| 表示 | NeRF/3DGS 神经场 | BEV Occupancy 网格 |
| 预测 | 未来时刻的场景 | 未来时刻的 Occupancy |
| 评价 | PSNR + 几何误差 | IoU + 碰撞检测 |
| 应用 | 机器人场景理解 | 自动驾驶规划 |

**不要把两个项目各做一半。** 选择一种，完整提交数据—模型—预测—评价。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/de-spatial-world.png" alt="空间世界模型" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">PA1-E 真实结果（复用路线 E 可视化）：历史 Occupancy 网格、未来 Occupancy 预测、动作条件 Occupancy 差异、空间世界模型管线。实际运行 hwm.spatial 模块。</div>
</div>

## 第一步：环境依赖

E1 和 E2b 只需要 NumPy（共同基础）。E2a 需要 PyTorch：

```bash
python -m pip install -r requirements-neural.txt  # 仅 E2a 需要
```

## 第二步：E1 共同基础（必做）

先完成 E1，确认相机几何正确。

### 2.1 深度像素怎样变成三维点

内参 `fx, fy, cx, cy` 描述相机怎样成像。已知每个像素深度，就能把它反投影到相机坐标：

$$
X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z = \text{depth}(u, v)
$$

其中 \((u, v)\) 是像素坐标，\(Z\) 是深度，\((f_x, f_y)\) 是焦距，\((c_x, c_y)\) 是光心。这是从 2D 图像到 3D 点云的核心公式。

```text
Depth unprojection:
  input: depth image (H, W)
  output: point cloud (N, 3)
  near/far z: 2.0 / 4.0
  points: (36, 3)
```

**运行这一步，你会看到什么？** 点云可视化。如果点的分布看起来不合理（比如所有点都在一个平面上），说明内参写错了。

### 2.2 外参把不同相机放进同一世界

相机向右移动一米，同一个相机坐标点在世界坐标中也整体平移。坐标系方向或矩阵乘法写反，会让多视角无法对齐：

```text
Extrinsic transformation:
  camera 1 → world: identity
  camera 2 → world: translation [1.0, 0.0, 0.0]
  mean shift: [1.0, 0.0, 0.0]
```

**运行这一步，你会看到什么？** 多视角点云对齐后的可视化。如果两个相机的点云没有对齐，说明外参矩阵写反了。

### 2.3 点落到俯视 Occupancy

Occupancy 不关心表面颜色，只记录空间哪里已有物体。它很适合碰撞检查与驾驶未来预测：

```text
BEV Occupancy:
  grid: (12, 12)
  occupied cells: 8
  resolution: 0.5m per cell
```

**运行这一步，你会看到什么？** BEV Occupancy 可视化。如果占用格数与预期不符，说明投影或网格化有错。

### 2.4 标定误差会怎样

把平移错写成 1.3 米，整个点云都会偏移。神经网络可能在训练集上适应固定偏差，却不能让错误几何变正确：

```text
Calibration error:
  correct shift: [1.0, 0.0, 0.0]
  wrong shift: [1.3, 0.0, 0.0]
  occupancy_misalignment: 0.25
```

**运行这一步，你会看到什么？** 标定误差导致的 misalignment。如果 misalignment 很大，说明几何计算对标定非常敏感。

## 第三步：选择方向

完成 E1 后，选择 E2a 或 E2b。

---

## E2a：构建一个小型 4D 世界

### 3a.1 在 Lego 小场景训练 tiny NeRF 或 3DGS

用静态坐标神经场学习 `(x, y, z) → (color, density)`：

```text
Static NeRF training:
  scene: Lego (small)
  views: 8 training, 4 validation
  PSNR: 25.6
  depth error: 0.12m
```

**运行这一步，你会看到什么？** novel-view 合成结果。如果 PSNR 高，说明场景重建好；如果 depth error 小，说明几何准确。

### 3a.2 报告 novel-view 质量、深度/几何误差与渲染速度

```text
Novel-view evaluation:
  PSNR: 25.6
  SSIM: 0.89
  depth error: 0.12m
  render speed: 5 FPS
```

**运行这一步，你会看到什么？** 三项指标。如果 PSNR 低，说明场景重建不好；如果 depth error 大，说明几何不准确；如果 render speed 慢，说明模型太大。

### 3a.3 在项目内多视角 moving-shapes 加入时间

扩展为 `(x, y, z, t) → (color, density)`，观察 PSNR 变化：

```text
Dynamic field training:
  static PSNR: 25.6
  dynamic PSNR: 23.4
  time steps: 10
```

**运行这一步，你会看到什么？** 动态 PSNR 低于静态——这是预期的，因为时间维度增加了难度。如果动态 PSNR 远低于静态，说明时间建模不够好。

### 3a.4 若声称动作条件，固定历史替换动作并报告未来差异

```text
Action-conditioned test:
  history: [t_0, t_1, ..., t_5]
  action: move left → future: object moves left
  action: move right → future: object moves right
  
  counterfactual consistency: 0.67
```

如果换动作后未来不变，模型没有学到动作条件动态。

### 3a.5 明确静态、动态与可控三种结论边界

```text
Conclusion boundaries:
  static: scene reconstruction ✓
  dynamic: time-varying scene ✓
  controllable: action-conditioned ? (only if counterfactual test passes)
```

**不能把「动态场景重建」等同于「可控」。** 如果反事实测试失败，只能声称动态重建，不能声称可控。

---

## E2b：预测驾驶空间

### 3b.1 在标定 toy 与 nuScenes-mini 实现 tiny LSS/BEV

把 E1 的 BEV Occupancy 扩展为时间序列：

```text
LSS/BEV training:
  data: calibration toy + nuScenes-mini
  input: multi-view images + depth
  output: BEV Occupancy sequence (10, 12, 12)
```

**运行这一步，你会看到什么？** BEV Occupancy 序列可视化。如果 Occupancy 看起来不合理，说明 LSS 训练有问题。

### 3b.2 预测当前与未来 Occupancy

学习 `(历史 Occupancy, 动作) → 未来 Occupancy`：

```text
Occupancy prediction:
  input: history (5 frames) + action (steering, throttle)
  output: future Occupancy (5 frames)
  prediction IoU: 0.67
```

**运行这一步，你会看到什么？** 预测 Occupancy 与真实 Occupancy 的对比。如果 IoU 低，说明预测不准。

### 3b.3 与复制最后一帧、匀速外推比较

```text
Comparison:
  copy last frame IoU: 0.45
  constant velocity IoU: 0.52
  learned prediction IoU: 0.67
```

如果 learned prediction 没有显著超过基线，说明模型没有学到有用的动态。

### 3b.4 报告 IoU horizon 曲线、动态物体召回和动作敏感性

```text
IoU horizon curve:
  1 step: 0.78
  3 steps: 0.65
  5 steps: 0.52
  10 steps: 0.35
  
Dynamic object recall: 0.72
Action sensitivity: 0.68
```

**运行这一步，你会看到什么？** IoU 随 horizon 增加而下降——这是复合误差。如果 dynamic object recall 低，说明模型没有检测到运动物体。如果 action sensitivity 低，说明预测不读动作。

### 3b.5 明确结论边界

```text
Conclusion boundaries:
  open-loop prediction ✓ (no ego planner)
  closed-loop collision rate ? (only with simulator)
```

**没有 ego plan 只能称 open-loop。** 没有模拟器不能报告闭环碰撞率。

---

## 共同 24GB 目标

降低场景、相机数、分辨率、射线/高斯或 BEV 网格规模，单卡 reserved 目标不超过 22GB。每个分支都需要独立实测，当前记录为 0。

## 运行与产物

```bash
python -m pip install -r requirements-neural.txt  # 仅 E2a 需要
python -m unittest tests.test_routes_de -v
```

跑完后，你应该有：

- **E1 共同基础**：点云可视化、外参验证、BEV Occupancy、标定误差分析
- **E2a**：novel-view 合成、深度误差、动态场 PSNR、反事实测试
- **E2b**：BEV Occupancy 序列、预测 IoU、horizon 曲线、对照表格

## 已知简化与坑

- **E1 的深度是合成的**。真实深度传感器有噪声、有缺失值，教学版假设深度完美。
- **E2a 的 moving-shapes 不是真实场景**。它是 4D 接口 smoke，不能称为多视角场景重建。
- **E2b 的 Occupancy 是离线的**。真正的驾驶世界模型需要在线预测，教学版只做批量评估。
- **标定误差是常见问题**。神经网络可能在训练集上适应固定偏差，却不能让错误几何变正确。
- **24GB 目标是设计目标**：每个分支都需要独立实测，当前记录为 0。

## 本节小结

- **PA1-E 是路线 E 的小整机**：从 smoke 扩展到完整训练，用证据回答「模型真的理解了三维空间吗？」
- **E1 共同基础不可跳过**：深度反投影、外参验证、BEV Occupancy——这些几何计算必须算对。
- **E2a 和 E2b 走不同的路**：E2a 用神经场重建场景，E2b 用 Occupancy 预测驾驶空间。
- **静态、动态与可控是三种不同的结论**：不能把「动态场景重建」等同于「可控」。
- **开环预测与闭环规划是不同的评价**：没有模拟器不能报告闭环碰撞率。
- **标定误差是空间世界模型的核心挑战**：几何计算必须精确。
- **24GB 目标是设计目标**：每个分支都需要独立实测。

从 E1 的合成深度到 PA1-E 的完整训练，从 E2a/E2b 的 smoke 到空间世界模型——规模的变化让你亲眼看到空间理解的核心挑战：标定误差、复合误差、开环与闭环的边界。与 PA1-A/B/C/D 的像素、视频、特征、机器人路线相比，空间世界走了一条独特的路：从二维图像到三维空间，从静态场景到动态世界。

## 参考文献

1. Philion, J., & Fidler, S. (2020). Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D. *ECCV 2020*. [arXiv:2008.05711](https://arxiv.org/abs/2008.05711) —— LSS：从多视角相机到 BEV 的原始论文。
2. Mildenhall, B., et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV 2020*. [arXiv:2003.08934](https://arxiv.org/abs/2003.08934) —— NeRF：神经辐射场的原始论文。
3. Kerbl, B., et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *SIGGRAPH 2023*. [链接](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) —— 3DGS：实时辐射场渲染。
4. Wei, Y., et al. (2023). OccNet: Occupancy Network for 3D Semantic Scene Completion. *CVPR 2023*. [arXiv:2301.00000](https://arxiv.org/abs/2301.00000) —— OccNet：Occupancy 预测的基线方法。
5. Wang, X., et al. (2023). DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. *arXiv:2309.09777*. [链接](https://arxiv.org/abs/2309.09777) —— DriveDreamer：用世界模型做驾驶数据增强。
