# 6.5　动手：视频 JEPA 的从零实现

> **本节目标**：在同一份 PixelWorld 上跑通 Video-JEPA。只预测未来特征，不画回像素，再用线性探针和换动作实验审问这些特征里还剩什么。

> **本节代码**：[学出视频特征](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/06_jepa/learn-video-features.ipynb) · [检验并控制特征](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/06_jepa/test-and-control-features.ipynb)

> **前置知识**：你已经读过 6.1–6.4，知道 EMA 与表示坍缩。最好刚跑完 [3.5 动手：表格世界模型的从零开始实现](/chapters/03-data-and-first-model/05-learn-a-table-world)。上一节还能看见画面；这一节把画面收起来。

---

5.6 还在「画出下一帧」。token 猜对了，你至少能把编号画回来，看红块走没走。

可有人会问：决策真的需要把草地的纹理画回来吗？Yann LeCun 走得更远——JEPA 根本不预测像素，只预测未来特征。

你当时大概和我一样，第一反应是：「不画出来，我怎么知道它学会了？」

这一节只预测特征，用线性探针审问这些特征里还剩什么。跑完你会发现一件别扭的事：loss 更低、训练更稳，但你必须另找证据，才能知道模型有没有学到东西。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-pixelworld.png" alt="PixelWorld 小世界" style="max-width:min(900px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">这就是我们要让模型学会的「世界」：16×16 的黑底小图，红色 3×3 是自己，绿色 3×3 是目标，固定在 (12, 12)。模型从未被告知「红色是智能体」，它要从像素流里自己发现「向右走，红块会右移」——只是这一次，它不必把红块画回来。</div>
</div>

## 本次会得到什么

运行结束后，你会得到：

- 第一份 Notebook「学出视频特征」的 feature loss 与 feature spread 曲线
- 第二份 Notebook「检验并控制特征」的 held-out 探针 MSE，以及同一段历史上换动作后的特征差

## 怎样运行

```text
notebooks/06_jepa/learn-video-features.ipynb
notebooks/06_jepa/test-and-control-features.ipynb
```

需要 PyTorch：

```bash
python -m pip install -r requirements-neural.txt
```

教学版在 CPU 上运行。即使暂时不打开 Notebook，也可以先跑测试：

```bash
PYTHONPATH=src python -m unittest tests.test_routes_bc -v
```

## 第一步：不画回像素，只预测未来特征

这一节改变的是预测目标。我们仍看连续视频，但不要求模型猜中下一帧每个像素。图片被切成 patch；视频再多一个时间维度。三帧 `16×16`，每帧 16 个 `4×4` patch，每个 patch 是 \(3 \times 4 \times 4 = 48\) 维。

```python
from hwm.data import make_pixelworld_dataset
from hwm.jepa import TinyVideoJEPA, patchify_video, jepa_batch_from_episodes

episodes = make_pixelworld_dataset(6, 6, seed=0)
video, actions, positions = jepa_batch_from_episodes(episodes, history_length=3)
patches = patchify_video(video[:2], patch_size=4)
print("video:", tuple(video.shape), "patches:", tuple(patches.shape))
```

**运行这一步，你会看到什么？**

```
video: (30, 3, 3, 16, 16) patches: (2, 3, 16, 48)
```

6 段、每段 6 帧，按长度 3 滑窗，得到 30 个短视频块。`[B, T, C, H, W] = (30, 3, 3, 16, 16)`。最后一帧是要预测的目标，前两帧是历史。

Online encoder 从历史提取信息，Predictor 预测下一帧每个 patch 的特征。Target encoder 只交出训练目标，不接收梯度。源码用 Smooth L1，不是像素 L2：

$$
\mathcal{L}_{\text{JEPA}}
= \mathrm{SmoothL1}\bigl(
    g_\phi(f_\theta(x_{\text{ctx}}), a),\;
    \mathrm{sg}[f_{\bar\theta}(x_{\text{future}})]
\bigr)
$$

Target 的参数靠 EMA 缓慢跟随 online，动量默认 \(m = 0.99\) [7]：

$$
\bar\theta \leftarrow m\,\bar\theta + (1-m)\,\theta
$$

若两边一起被同一个 loss 快速推动，模型更容易找到「所有输入都输出同一个常量」的捷径。那就是表示坍缩：loss 可以很低，特征里什么都没有。

```python
from hwm.jepa import feature_spread

model = TinyVideoJEPA(feature_size=16)
loss, prediction, target, features = model.loss(video, actions=None)
print("prediction/target:", tuple(prediction.shape), tuple(target.shape))
print("target requires_grad:", target.requires_grad)

parameters = list(model.online_encoder.parameters()) + list(model.predictor.parameters())
optimizer = torch.optim.Adam(parameters, lr=3e-3)
losses, spreads = [], []
for _ in range(35):
    optimizer.zero_grad()
    loss, prediction, target, features = model.loss(video, actions=None)
    loss.backward()
    optimizer.step()
    model.update_target(momentum=0.99)
    losses.append(float(loss.detach()))
    spreads.append(float(feature_spread(features).detach()))
print("feature loss:", round(losses[0], 3), "→", round(losses[-1], 3))
print("feature spread:", round(spreads[0], 3), "→", round(spreads[-1], 3))
```

`feature_spread` 把特征展平，对每个通道求标准差再取平均。它接近零，就是坍缩的警报。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-jepa-pipeline.png" alt="Video-JEPA 管线" style="max-width:min(900px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">JEPA 路线（第 6 章）的数据流。历史 patch 进 online encoder，predictor 加上位置编码；target encoder 对未来帧给出无梯度目标。第一份 Notebook 默认不读动作。</div>
</div>

**运行这一步，你会看到什么？**

```
prediction/target: (30, 16, 16) (30, 16, 16)
target requires_grad: False
feature loss: 0.238 → 0.030
feature spread: 0.240 → 0.326
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-jepa-curves.png" alt="JEPA 损失与 spread" style="max-width:min(720px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">35 次更新。loss 从 0.238 掉到 0.030，spread 从 0.240 升到 0.326。loss 下降且 spread 不降，说明还没有塌成常量。</div>
</div>

shape 是 `[B, 16 patches, 16 dim]`。target 不接收梯度。loss 下降、spread 上升——这是你希望看见的方向。反过来，loss 下降且 spread 掉到接近零，才是坍缩。

第一份 Notebook 最后用棋盘 mask 只计算一半 patch 的误差，确认接口：

```
masked positions: 240  masked loss: 0.030
```

30 个 clip × 8 个被选中的 patch = 240。真正的 V-JEPA 会遮住整块时空区域 [8]；这里只看清「mask 是出题方式，不是数据增强」。

**一个值得做的实验**：把 `momentum` 从 0.9 扫到 0.999，同时画 loss 和 spread。动量太小，target 跟着 online 一起跳，更容易共谋一个平凡解；动量太大，靶子几乎不动，早期预测会长时间对不准。

## 第二步：特征里还有没有位置

低 feature loss 可能只是找到了容易预测的常量。第二份 Notebook 冻结表示，只用一个带偏置的岭回归读方块中心。位置在构造数据时按 `15.0` 归一化到 \([0, 1]\)。如果连这个简单属性都读不出，特征很难支持后续空间任务。

训练和测试按 episode seed 切开：10 段训练、4 段测试，避免探针在同一条轨迹上自测。

```python
from hwm.jepa import fit_linear_probe_weights, apply_linear_probe

train_episodes = make_pixelworld_dataset(10, 7, seed=2)
test_episodes = make_pixelworld_dataset(4, 7, seed=31)
video, actions, positions = jepa_batch_from_episodes(train_episodes, 3)
test_video, test_actions, test_positions = jepa_batch_from_episodes(test_episodes, 3)

model = TinyVideoJEPA(feature_size=16)
parameters = (
    list(model.online_encoder.parameters())
    + list(model.predictor.parameters())
    + list(model.action_embedding.parameters())
)
optimizer = torch.optim.Adam(parameters, lr=3e-3)
for _ in range(40):
    optimizer.zero_grad()
    loss, prediction, target, features = model.loss(video, actions)
    loss.backward()
    optimizer.step()
    model.update_target(0.99)

with torch.no_grad():
    _, train_target, _ = model(video, actions)
    _, test_target, _ = model(test_video, test_actions)
weights = fit_linear_probe_weights(train_target.flatten(1), positions)
pred = apply_linear_probe(test_target.flatten(1), weights)
probe_mse = F.mse_loss(pred, test_positions)
constant = positions.mean(0).expand_as(test_positions)
print("held-out probe/base MSE:",
      round(float(probe_mse), 4),
      round(float(F.mse_loss(constant, test_positions)), 4))
```

平均掉 patch 会抹掉位置；`flatten` 保留每个 patch 在网格中的槽位。这不是分类准确率，是二维坐标的 MSE。

**运行这一步，你会看到什么？**

```
loss: 0.184 → 0.026   spread: 0.227 → 0.281
held-out probe/base MSE: 0.0052  0.0223
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-probe.png" alt="线性探针位置" style="max-width:min(520px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">测试集上的方块位置。绿圈是真值，红点是探针。探针赢过「永远猜训练集均值」，但单点仍然糊在一团——特征里有位置，远没有像素坐标那么干净。</div>
</div>

0.0052 低于常数基线 0.0223，assert 能过。把前三个测试点摊开：真值大约在 `(0.07, 0.53)`，探针给出 `(0.29, 0.40)`。赢了基线，不等于读准了格子。

**这就是线性探针能回答的问题**：特征里有没有位置。它不能回答「已经够规划了」。

## 第三步：固定历史，只替换动作

被动视频表示可以保存运动，却不能证明模型懂得控制。第二份 Notebook 把同一段历史复制五份，只换动作，看预测特征会不会变。

```python
same_history = video[:1].expand(5, -1, -1, -1, -1)
all_actions = torch.arange(5)
with torch.no_grad():
    predictions, _, _ = model(same_history, all_actions)
differences = [
    (predictions[0] - predictions[i]).square().mean().item()
    for i in range(1, 5)
]
print("换动作后的 feature MSE:", [round(x, 5) for x in differences])
```

同一个 held-out 探针再把五个候选特征变回位置，选离绿色目标 `(12/15, 12/15)` 最近的动作。若探针在新 episode 上不可靠，这个动作选择也没有可信依据。

**运行这一步，你会看到什么？**

```
换动作后的 feature MSE: [1.00163, 0.00196, 0.00281, 0.00256]
候选位置: stay/left/right/up/down 都挤在 0.28–0.31 附近
到目标距离: [1.7288, 0.7211, 0.7151, 0.6956, 0.7242]
选择动作: 3 (up)
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-action-swap.png" alt="换动作后的特征差" style="max-width:min(720px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">相对 stay，另外四个动作都把预测特征挪开了一点。差是 10⁻³ 量级，不是零。探针在这五个候选里选了 up。</div>
</div>

动作进了 predictor。差很小，五个解码位置几乎重叠，选 up 只是「五个糊点里离目标稍近的那个」。把它写成「已经能规划」，过了。

再补一个容易写错的对照：在测试集上，被动 JEPA 的特征 MSE 是 0.0687，动作条件是 0.0673，把动作故意加一之后是 0.0670。三个数几乎一样。可控性的证据在「换动作特征会变」，不在「动作条件 MSE 远低于被动」。

## 运行与产物

```bash
python -m pip install -r requirements-neural.txt
PYTHONPATH=src python -m unittest tests.test_routes_bc -v
```

跑完两份 Notebook 后，你应该有：

- **第一份 Notebook**：feature loss 与 feature spread，以及 target 不接收梯度
- **第二份 Notebook**：held-out 探针对常数基线、换动作后的特征差、一次动作选择

| 项目 | 本节 smoke           | 6.6                             |
| ---- | -------------------- | ------------------------------- |
| 数据 | 6–12 段 PixelWorld   | 更大的 PixelWorld，选做真实视频 |
| 训练 | 20–40 步，CPU        | 直到曲线稳定                    |
| 目的 | 检查 EMA、探针接口   | 检查特征是否真能支撑控制        |
| 结论 | 接口通了；探针还很弱 | JEPA 是否形成稳定闭环           |

## 已知简化与坑

- **第一份 Notebook 的损失是 Smooth L1，不是像素 L2。** 公式必须和 `F.smooth_l1_loss` 对齐。
- **第二份 Notebook 的探针是坐标回归，不是分类准确率。** 报告 MSE 对常数基线，不要编一个 0.78 的 accuracy。
- **被动视频不能证明 controllability。** 必须分开报告「换动作特征差」和「被动预测误差」。后者在这份数据上几乎不动。
- **平均 pool 会抹掉位置。** 第二份 Notebook 之所以 flatten，就是因为 `mean(dim=1)` 会把方块在哪个 patch 丢掉。
- **PixelWorld 过于简单。** 16×16、红方块——特征在这里很容易「看起来没塌」，换真实视频会难得多。

## 扩展练习

1. **EMA 动量**：把第一份 Notebook 的 `momentum` 从 0.9 扫到 0.999，同时看 loss 和 spread。哪一侧先坍缩？
2. **探针怎么读**：把第二份 Notebook 的 `flatten` 换成 `mean(dim=1)`，看 held-out MSE 会不会重新输给常数基线。空间槽位被平均掉以后，位置信息还在不在？

完成后进入 [6.6 动手：审问一个视频 JEPA](/chapters/06-jepa/06-video-jepa)。

## 本节小结

- **Video-JEPA 只预测未来特征**，loss 稳、spread 没塌，但必须靠探针和换动作实验，才能知道特征里还剩什么。
- **EMA 是防止表示坍缩的关键**：target 不接收梯度，只按 \(m=0.99\) 跟着 online 走。
- **Smoke 不是完整训练**：8–12 段 episode、20–40 步、CPU 运行。目标是检查数据流，不是复现 V-JEPA。

从 5.6 的「要不要画像素」，到这一节的「不画出来以后怎么证明」，被替换的是监督目标，不是那句老话。

## 后续工作

这两份 Notebook 只把特征预测接到了 PixelWorld。不重建像素以后，你必须另找证据。

第二份 Notebook 已经演示了最小的审问方式：线性探针和反事实动作。I-JEPA [2] 把这件事做到图像，V-JEPA [3] 做到视频，V-JEPA 2 [4] 再加上动作，用几十小时机器人数据做零样本规划。它们仍然不画像素。评价从 PSNR 换成了探针、检索、以及下游控制——这也是这条路线比 5.6 更难「一眼看懂」的原因。

下一台模型要回答的，是你刚刚亲眼看见的那些失败：特征没塌但探针仍然读不准格子。

## 参考文献

1. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. _OpenReview_. [链接](https://openreview.net/forum?id=BZ5a1r-kVsf) —— JEPA 的立场论文：预测特征，而不是像素。
2. Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. _CVPR 2023_. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) —— I-JEPA：图像上的掩码特征预测。
3. Bardes, A., et al. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video. _arXiv:2404.08471_. [链接](https://arxiv.org/abs/2404.08471) —— V-JEPA：视频版 JEPA，时空掩码与 EMA 目标编码。
4. Bardes, A., et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. _arXiv:2506.09985_. [链接](https://arxiv.org/abs/2506.09985) —— 动作条件版 V-JEPA 2-AC 与零样本规划。
