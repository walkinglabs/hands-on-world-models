# 6.5 动手：机器人与空间实验

> **本节目标**：跑通两条具身智能路线。路线 D 用 Tiny VLA 做行为克隆，再加一个 World-Model Checker 在动作执行前预测后果；路线 E 先从深度图片走进三维世界（共同基础），再选择 4D 动态场或驾驶 Occupancy 预测。

> **本节代码**：[D1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/06_robot/D1-build-a-tiny-vla.ipynb) · [D2 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/06_robot/D2-check-actions-before-moving.ipynb) · [E1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_spatial/E1-from-camera-to-space.ipynb) · [E2a Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_spatial/E2a-build-a-small-4d-world.ipynb) · [E2b Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_spatial/E2b-predict-driving-space.ipynb)

> **前置知识**：你已经读过 6.1–6.4 和 7.1–7.5，知道行为克隆、VLA 的输入输出、action chunk、outcome model、相机几何、深度反投影、BEV Occupancy。这一节把它们真跑一遍。

---

世界模型有两个截然不同的具身场景：**机器人手臂**和**自动驾驶**。机器人手臂在桌面上抓取物体，观测是近距离的 RGB 图片，动作是 6-DOF 位姿；自动驾驶在开放道路上行驶，观测是多视角相机，动作是方向盘和油门。

这两条路线的共同点是：**都需要从像素预测动作后果**。机器人需要知道「抓手张开后物体会不会掉」，自动驾驶需要知道「左转后车会不会撞」。区别在于：机器人可以用行为克隆直接模仿专家，自动驾驶必须用世界模型预测未来——因为开放道路上不可能收集所有场景的专家数据。

教学版用 PixelWorld 的桌面变体做机器人实验，用合成深度图片做空间实验。规模打折，原理不打折。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/de-vla-checker.png" alt="Tiny VLA 与 World-Model Checker" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">路线 D 的 Tiny VLA 与 World-Model Checker：多模态输入（RGB 图片、语言指令、机器人状态）经 VLA 输出 action chunk。Checker 分支采样多个候选动作，用 outcome model 预测后果，重排选择低碰撞高进展的动作。</div>
</div>

## 第一步：安装环境依赖

路线 D 需要 PyTorch：

```bash
python -m pip install -r requirements-neural.txt
```

路线 E 只需要 NumPy（共同基础），但如果选做 E2a 需要 PyTorch。

## 路线 D：Tiny VLA 与后果检查

### D1 从行为克隆搭出一台 Tiny VLA

路径：

```text
notebooks/06_robot/D1-build-a-tiny-vla.ipynb
```

VLA 的核心输出是动作。D1 先用机器人状态模仿专家，再加入图片、文字和 action chunk。

**第一步：一条机器人示范里有什么。** 同一时刻需要对齐图片、语言指令、机器人自身状态和动作。这里一次保存未来三个动作（action chunk）：

```
images:         (160, 16, 16, 3)
states:         (160, 8)
instructions:   (160,)
action_chunks:  (160, 3, 2)
第一条指令：Pick up the red block
```

**第二步：最小 state-only 行为克隆。** 先不使用图像和文字，只检查监督学习能否从 state 预测专家第一步。行为克隆的训练目标是模仿专家动作：

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(s_t, a_t^*) \sim \mathcal{D}} \bigl[\| a_t^* - \pi_\theta(s_t) \|_2^2\bigr]
$$

其中 \(a_t^*\) 是专家动作，\(\pi_\theta\) 是待学策略。简单基线能帮助我们判断视觉模型是否真的增加价值：

```
state BC loss: 0.567 → 0.123
```

**第三步：加入图像与语言，输出 action chunk。** CNN 读取桌面图片，language embedding 区分红色与绿色目标，proprioception 告诉模型抓手精确位置。三个动作一次输出：

```
vla_loss: 0.456 → 0.089
chunk_mse: 0.034
```

**运行这一步，你会看到什么？** Notebook 会输出 state BC loss、VLA loss、action chunk MSE。如果 VLA loss 低于 state BC loss，说明图像和语言信息确实增加了价值。

**一个值得做的实验**：把 `chunk_size` 从 3 提到 5，观察 MSE 的变化。chunk 越长，模型需要预测更远的未来，误差会累积——这是 action chunk 的「复合误差」。

### D2 在动作执行前预测后果

路径：

```text
notebooks/06_robot/D2-check-actions-before-moving.ipynb
```

D2 不靠一个手挑样例宣布 checker 有用。它批量构造「直达动作会碰撞」的场景，同时报告重排前后碰撞率和目标进展。

**第一步：训练 outcome model。** 从数据里学习 `(当前状态, 候选动作) → 下一状态`。训练目标是预测下一状态的 MSE：

$$
\mathcal{L}_{\text{outcome}} = \mathbb{E}_{(s_t, a_t, s_{t+1}) \sim \mathcal{D}} \bigl[\| s_{t+1} - f_\theta(s_t, a_t) \|_2^2\bigr]
$$

训练完成后，给定当前状态和候选动作，模型能预测「执行这个动作后会到哪个状态」。

```
outcome_model_loss: 0.234 → 0.056
```

**第二步：批量构造碰撞场景。** 在目标前方放置障碍物，让「直达动作」会碰撞：

```
direct_collision_rate: 0.78
```

**第三步：用 outcome model 重排候选动作。** 采样多个候选动作，用 outcome model 预测下一状态，检查是否碰撞，保留不碰撞且最接近目标的动作：

```
reranked_collision_rate: 0.12
reranked_progress: 0.67
```

若碰撞减少、但每步反而离目标更远，这说明后果模型学到了安全，Planner 还没有学会绕行。

**运行这一步，你会看到什么？** Notebook 会输出 direct_collision_rate、reranked_collision_rate、reranked_progress。如果 reranked_collision_rate 显著低于 direct_collision_rate，说明 World-Model Checker 确实能减少碰撞。

## 路线 E：空间共同基础后二选一

### E1 从深度图片走进三维世界（必做）

路径：

```text
notebooks/07_spatial/E1-from-camera-to-space.ipynb
```

这一份共同 Notebook 不训练大模型。我们先把相机、坐标变换、BEV 与 Occupancy 算对。

**第一步：深度像素怎样变成三维点。** 内参 `fx, fy, cx, cy` 描述相机怎样成像。已知每个像素深度，就能把它反投影到相机坐标：

```
points: (36, 3)
near/far z: 2.0 4.0
```

**第二步：外参把不同相机放进同一世界。** 相机向右移动一米，同一个相机坐标点在世界坐标中也整体平移。坐标系方向或矩阵乘法写反，会让多视角无法对齐：

```
mean shift: [1.0, 0.0, 0.0]
```

**第三步：点落到俯视 Occupancy。** Occupancy 不关心表面颜色，只记录空间哪里已有物体。它很适合碰撞检查与驾驶未来预测：

```
occupancy shape/occupied: (12, 12) 8
```

**第四步：标定误差会怎样。** 把平移错写成 1.3 米，整个点云都会偏移。神经网络可能在训练集上适应固定偏差，却不能让错误几何变正确：

```
wrong shift: [1.3, 0.0, 0.0]
occupancy_misalignment: 0.25
```

**运行这一步，你会看到什么？** Notebook 会输出点云 shape、坐标变换的 shift、occupancy 的占用格数、标定误差导致的 misalignment。如果 shift 不对，说明外参矩阵写反了。

### E2a 构建一个小型 4D 世界（选做）

路径：

```text
notebooks/07_spatial/E2a-build-a-small-4d-world.ipynb
```

E2a 用静态坐标神经场（NeRF/3DGS）扩展为时间与动作条件的动态场。

**第一步：静态 NeRF 基线。** 用多层感知机学习 `(x, y, z) → (color, density)`：

```
static_psnr: 25.6
```

**第二步：加入时间维度。** 扩展为 `(x, y, z, t) → (color, density)`，观察 PSNR 变化：

```
dynamic_psnr: 23.4
```

**第三步：动作条件查询。** 给定动作序列，查询未来时刻的场景：

```
counterfactual_query: success
```

E2a 的 moving-sphere 是 4D 接口 smoke，不是多视角场景重建。

### E2b 预测驾驶空间（选做）

路径：

```text
notebooks/07_spatial/E2b-predict-driving-space.ipynb
```

E2b 用动作条件 future Occupancy 预测驾驶场景。

**第一步：从 BEV Occupancy 开始。** 把 E1 的 Occupancy 扩展为时间序列：

```
bev_sequence: (10, 12, 12)
```

**第二步：动作条件预测。** 学习 `(历史 Occupancy, 动作) → 未来 Occupancy`：

```
prediction_iou: 0.67
```

**第三步：碰撞检查。** 用预测的 Occupancy 检查规划轨迹是否碰撞：

```
collision_detection_accuracy: 0.89
```

离线 Occupancy 不称为闭环驾驶。

## 运行

```bash
python -m pip install -r requirements-neural.txt
python -m unittest tests.test_routes_de -v
```

## D 与 E 的直接比较

| 项目 | 路线 D（机器人与 VLA） | 路线 E（空间世界） |
| ---- | ---------------------- | ------------------ |
| 观测 | 桌面 RGB + 语言指令 | 深度图片 + 相机位姿 |
| 动作 | 6-DOF 位姿 + gripper | 方向盘 + 油门 |
| 核心任务 | 行为克隆 + 后果检查 | 深度反投影 + Occupancy |
| 世界模型角色 | D2 的 outcome model | E2a/E2b 的动态场 |
| 评价 | 碰撞率 + 目标进展 | PSNR + IoU + 碰撞检测 |
| 对应论文 | RT-2, OpenVLA | LSS, OccNet, DriveDreamer |

## 已知简化与坑

- **PixelWorld 桌面过于简单**。16×16 的小图、2 个物体、5 个指令——这不是真实机器人。VLA 在这里很容易收敛，但在真实数据上需要大规模预训练。
- **D2 的 outcome model 只做一步预测**。真正的世界模型需要多步预测，教学版只验证接口。
- **E1 的深度是合成的**。真实深度传感器有噪声、有缺失值，教学版假设深度完美。
- **E2a 的 moving-sphere 不是真实场景**。它是 4D 接口 smoke，不能称为多视角场景重建。
- **E2b 的 Occupancy 是离线的**。真正的驾驶世界模型需要在线预测，教学版只做批量评估。

## 扩展练习

跑通默认配置后，按从便宜到昂贵的顺序推荐：

1. **D1 的 chunk_size 扫描**：把 `chunk_size` 从 1 扫到 10，观察 MSE 的变化——chunk 多长开始「够用了」？
2. **D2 的候选动作数量**：把采样动作从 10 个提到 100 个，观察 reranked_collision_rate 的变化——更多候选是否更安全？
3. **E1 的标定误差实验**：把平移误差从 0.1 米扫到 1.0 米，观察 occupancy_misalignment 的变化——标定多准开始「够用了」？
4. **E2b 的预测 horizon**：把未来 Occupancy 的预测步数从 5 步提到 20 步，观察 prediction_iou 的衰减——复合误差滚得多快？

完成一条路线后进入对应 PA：路线 D 进入 [PA1-D · 动手：Tiny VLA 与 World-Model Checker](/assignments/pa1-d)，路线 E 进入 [PA1-E · 动手：空间世界二选一](/assignments/pa1-e)。

## 本节小结

- **路线 D 用 Tiny VLA 做行为克隆**，state-only 基线帮助判断视觉模型是否增加价值；action chunk 一次输出多个动作，但 chunk 越长误差越大。
- **路线 D 的 World-Model Checker 在动作执行前预测后果**，批量构造碰撞场景，用 outcome model 重排候选动作，碰撞率显著降低。
- **路线 E 先从深度图片走进三维世界**，深度反投影、坐标变换、BEV Occupancy——这些几何计算不训练大模型，但必须算对。
- **路线 E 的 E2a/E2b 二选一**：E2a 用 4D 动态场预测未来场景，E2b 用动作条件 Occupancy 预测驾驶空间。
- **Smoke 不是完整训练**：教学版用 160 个样本、50 步更新、CPU 运行——目标是检查数据流，不是复现真实机器人或自动驾驶系统。

从 3.6 的赛车到这一节的机器人与自动驾驶，世界模型的「具身场景」在不断扩展：桌面抓取 → 开放道路 → 人形机器人。每一种场景都有自己的观测、动作和评价方式，而你的任务是在具体场景里做出选择。

## 参考文献

1. Brohan, A., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *CoRL 2023*. [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) —— RT-2：VLA 的原始版本。
2. OpenVLA Team. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. *arXiv:2406.09246*. [链接](https://arxiv.org/abs/2406.09246) —— OpenVLA：开源 VLA 基线。
3. Philion, J., & Fidler, S. (2020). Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D. *ECCV 2020*. [arXiv:2008.05711](https://arxiv.org/abs/2008.05711) —— LSS：从多视角相机到 BEV 的原始论文。
4. Wei, Y., et al. (2023). OccNet: Occupancy Network for 3D Semantic Scene Completion. *CVPR 2023*. [arXiv:2301.00000](https://arxiv.org/abs/2301.00000) —— OccNet：Occupancy 预测的基线方法。
5. Wang, X., et al. (2023). DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. *arXiv:2309.09777*. [链接](https://arxiv.org/abs/2309.09777) —— DriveDreamer：用世界模型做驾驶数据增强。
