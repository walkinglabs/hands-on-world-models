# F1–F3 · 共同基础实验

共同基础使用四份 Notebook：F0 重新发明世界模型，F1–F3 再把表示、空间、数据和第一台 learned dynamics 接起来。

F1–F3 只依赖 NumPy，在 CPU 上几秒内完成。这里的目标是看清接口和失败，不提前训练大型神经网络。

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

最终产物包括卷积响应、patch token 表、两段相同末帧的不同速度状态，以及压缩比与重建误差。

## F2：从相机到空间，再交给规划器

路径：

```text
notebooks/01_foundations/F2-space-plan-train.ipynb
```

这份 Notebook 把小深度图反投影为点云，再落到 Occupancy。随后用 CEM 在连续一维世界中搜索动作，并观察 Symlog 与梯度裁剪。

F2 不训练 NeRF、LSS 或 Actor-Critic。它先把相机、空间、搜索和训练工程放在系统中的正确位置。

## F3：从经历中学习概率动态

路径：

```text
notebooks/01_foundations/F3-learn-a-table-world.ipynb
```

F3 在带打滑的 LineWorld 中完成：

```text
收集 episode → 按 episode 切分
→ 计数学习 P(next_state | state, action)
→ 一步评价 → 反事实动作 → MPC 闭环
```

最终模型使用 140 个训练 episode，在固定 smoke seed 下覆盖全部测试 transition，并用 MPC 到达终点。这个结果只证明表格小世界路径可运行，不代表神经世界模型已经训练完成。

## 自动检查

四份 Notebook 的全部代码格都会被 smoke 测试执行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

当前检查包括 shape、动作时间对齐、episode 内序列采样、卷积、patch 往返、记忆方向、压缩、相机反投影、Occupancy、CEM、Symlog、计数动态、MPC 和 Notebook 全格执行。

下一步是 [PA0：重新发明一台可学习世界模型](/assignments/pa0)。
