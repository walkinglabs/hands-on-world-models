# 1.8 动手：基础实验

> **本节目标**：跑通 F1–F3 三份 Notebook，把表示、空间、数据和第一台 learned dynamics 接起来。F1 用卷积和 ViT patch 压缩观测，F2 从深度图片走进三维空间并用 CEM 搜索动作，F3 从轨迹里学习表格动态并用 MPC 闭环。

> **本节代码**：[F1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/01_foundations/F1-see-remember-compress.ipynb) · [F2 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/01_foundations/F2-space-plan-train.ipynb) · [F3 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/02_data/F3-learn-a-table-world.ipynb)

> **前置知识**：你已经读过第 1 章和第 2 章，知道 CNN、ViT patch、深度反投影、Occupancy、CEM、Symlog、计数动态。这一节把它们真跑一遍。

---

第 1 章和第 2 章讲了很多概念：CNN 怎样提取特征、ViT 怎样切 patch、深度图片怎样变成三维点、Occupancy 怎样记录空间、CEM 怎样搜索动作、计数动态怎样从轨迹里学习。

但概念不等于理解。**只有当你亲手把这些部件接起来，看到数据从输入流到输出，你才算真正理解**。

F1–F3 就是这样一个「接起来」的过程。三份 Notebook，只依赖 NumPy，在 CPU 上几秒内完成。目标不是训练大型神经网络，而是看清接口和失败。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/f1-foundations.png" alt="基础概念" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">世界模型的基础：张量运算（矩阵乘法）、轨迹（状态空间中的路径）、压缩（大输入→小 latent→重建）。F1–F3 把这些概念接起来：看见（CNN/ViT）、记住（隐状态）、压缩（VAE）、规划（CEM/MPC）。</div>
</div>

## 安装

在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
```

随后启动自己常用的 Jupyter 环境，或直接用编辑器打开 `.ipynb`。

## F1：看见、记住与压缩

路径：

```text
notebooks/01_foundations/F1-see-remember-compress.ipynb
```

同一段 PixelWorld 数据会经过：

```text
shape → 从零卷积 → ViT patch
→ 相同末帧的历史反例 → 速度记忆 → 块压缩
```

**第一步：从零卷积。** 用随机初始化的 3×3 卷积核扫描 16×16 图片，观察响应图。卷积的计算是局部加权和：

$$
(I * K)[i, j] = \sum_{m=-1}^{1} \sum_{n=-1}^{1} I[i+m, j+n] \cdot K[m, n]
$$

其中 \(I\) 是输入图像，\(K\) 是 3×3 卷积核。你会发现，即使没有训练，卷积核也能检测边缘和角点——这是卷积的归纳偏置。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/f1-convolution.png" alt="卷积响应图" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">F1 第一步：三种 3×3 卷积核的响应图。即使没有训练，卷积核也能检测边缘和角点——这是卷积的归纳偏置。</div>
</div>

**第二步：ViT patch。** 把 16×16 图片切成 4×4 的 patch，得到 16 个 token。每个 token 是 48 维向量（4×4×3）。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/f1-patchify.png" alt="ViT patch 切分" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">F1 第二步：把图片切成 4×4 的 patch，得到 16 个 token。每个 token 是 48 维向量，压缩比约 5.3x。</div>
</div>

**第三步：相同末帧的历史反例。** 两段轨迹的最后一帧相同，但历史不同——一段从左边来，一段从右边来。模型需要记住历史才能区分。

**第四步：速度记忆。** 用简单差分估计方块速度，观察速度方向与动作的关系。

**第五步：块压缩。** 用 PCA 或随机投影把 48 维 token 压成 8 维，观察重建误差。

**运行这一步，你会看到什么？** Notebook 会输出卷积响应图、patch token 表、两段相同末帧的不同速度状态，以及压缩比与重建误差。

**一个值得做的实验**：把 patch size 从 4×4 改成 2×2 或 8×8，观察 token 数量和压缩比的变化。patch 越小，token 越多，表示越细，但计算量越大。

## F2：从相机到空间，再交给规划器

路径：

```text
notebooks/01_foundations/F2-space-plan-train.ipynb
```

这份 Notebook 把小深度图反投影为点云，再落到 Occupancy。随后用 CEM 在连续一维世界中搜索动作，并观察 Symlog 与梯度裁剪。

**第一步：深度反投影。** 用内参 `fx, fy, cx, cy` 把深度像素反投影为三维点：

```python
points_camera = depth_to_points(depth, fx=6, fy=6, cx=2.5, cy=2.5)
```

**第二步：坐标变换。** 用外参把相机坐标点变换到世界坐标：

```python
points_world = transform_points(points_camera, camera_to_world)
```

**第三步：Occupancy。** 把三维点落到俯视网格，记录哪些格子被占用：

```python
occupancy = points_to_occupancy(points_world, x_range=(-2, 4), z_range=(0, 6), resolution=0.5)
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/f2-depth-occupancy.png" alt="深度反投影到 Occupancy" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">F2 前三步：深度图 → 3D 点云（俯视图）→ Occupancy 网格。从像素到空间的完整管线。</div>
</div>

**第四步：CEM 搜索。** 在连续一维世界中，用 CEM 搜索最优动作序列：

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/f2-cem-search.png" alt="CEM 搜索过程" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">F2 第四步：CEM 搜索——从随机采样到精英集中。绿线=目标，红线=均值，蓝线=样本。</div>
</div>

```python
best_action = cem_search(dynamics, initial_state, horizon=5)
```

**第五步：Symlog 与梯度裁剪。** 观察 Symlog 变换怎样压缩奖励范围，梯度裁剪怎样防止梯度爆炸。

**运行这一步，你会看到什么？** Notebook 会输出点云 shape、坐标变换的 shift、Occupancy 的占用格数、CEM 搜索的最优动作、Symlog 变换前后的奖励分布。

F2 不训练 NeRF、LSS 或 Actor-Critic。它先把相机、空间、搜索和训练工程放在系统中的正确位置。

## F3：从经历中学习概率动态

路径：

```text
notebooks/02_data/F3-learn-a-table-world.ipynb
```

F3 在带打滑的 LineWorld 中完成：

```text
收集 episode → 按 episode 切分
→ 计数学习 P(next_state | state, action)
→ 一步评价 → 反事实动作 → MPC 闭环
```

**第一步：收集 episode。** 用随机策略在 LineWorld 里跑 140 个 episode，每个 episode 最多 20 步。

**第二步：计数学习。** 从轨迹里计数 `(state, action, next_state)` 的转移次数，归一化得到概率转移表：

```python
P[next_state | state, action] = count(state, action, next_state) / count(state, action)
```

**第三步：一步评价。** 用转移表预测下一状态，对比真实下一状态，计算准确率。

**第四步：反事实动作。** 固定起点，替换动作序列，观察预测轨迹的变化。

**第五步：MPC 闭环。** 用转移表做规划，只执行第一步，重新观察，重新规划，直到到达目标。

**运行这一步，你会看到什么？** Notebook 会输出转移表、一步预测准确率、反事实轨迹、MPC 执行记录。最终模型使用 140 个训练 episode，在固定 smoke seed 下覆盖全部测试 transition，并用 MPC 到达终点。

这个结果只证明表格小世界路径可运行，不代表神经世界模型已经训练完成。

## 自动检查

四份 Notebook 的全部代码格都会被 smoke 测试执行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

当前检查包括 shape、动作时间对齐、episode 内序列采样、卷积、patch 往返、记忆方向、压缩、相机反投影、Occupancy、CEM、Symlog、计数动态、MPC 和 Notebook 全格执行。

## 已知简化与坑

- **PixelWorld 过于简单**。16×16 的小图、红色方块、5 个动作——这不是 Atari。CNN 和 ViT 在这里很容易收敛，但在真实图片上需要大规模训练。
- **F2 的深度是合成的**。真实深度传感器有噪声、有缺失值，教学版假设深度完美。
- **F3 的 LineWorld 是一维的**。真实环境是高维连续状态，表格动态无法直接迁移。
- **CEM 的搜索空间很小**。教学版用一维连续动作，真实机器人需要高维动作空间。

## 扩展练习

跑通默认配置后，按从便宜到昂贵的顺序推荐：

1. **F1 的 patch size 扫描**：把 patch size 从 2×2 扫到 8×8，观察 token 数量和压缩比的关系——是否存在一个「甜蜜点」？
2. **F2 的 CEM 种群大小**：把种群从 50 提到 500，观察搜索质量的变化——更多样本是否更优？
3. **F3 的 episode 数量**：把训练 episode 从 140 提到 500，观察一步预测准确率的变化——数据多少开始「够用了」？

下一步是 [PA0 · 动手：重新发明一台可学习世界模型](/assignments/pa0)。

## 本节小结

- **F1 用卷积和 ViT patch 压缩观测**，即使没有训练，卷积核也能检测边缘和角点——这是卷积的归纳偏置。
- **F2 从深度图片走进三维世界**，深度反投影、坐标变换、BEV Occupancy——这些几何计算不训练大模型，但必须算对。
- **F2 用 CEM 在连续空间搜索动作**，Symlog 变换压缩奖励范围，梯度裁剪防止梯度爆炸。
- **F3 从轨迹里学习表格动态**，计数转移、归一化概率，用 MPC 闭环到达目标。
- **Smoke 不是完整训练**：教学版用 NumPy、CPU、几秒内完成——目标是看清接口和失败，不是训练大型神经网络。

从 F0 的九格网格到 F3 的 LineWorld，从人工写的转移表到从数据里学习的概率动态——世界模型的核心思想从未改变：**在行动之前，先在内部预见行动的后果**。而这一节，你亲手把表示、空间、数据和动态接起来。

## 参考文献

1. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) —— ViT：把图片切成 patch 的原始论文。
2. Rubinstein, R. Y. (1999). The Cross-Entropy Method for Combinatorial and Continuous Optimization. *Methodology and Computing in Applied Probability*, 1, 127–190. [链接](https://doi.org/10.1023/A:1010091220143) —— CEM：交叉熵方法的原始论文。
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [链接](http://incompleteideas.net/book/the-book.html) —— 经典 RL 教材，第 8 章讲 Dyna 与规划。
