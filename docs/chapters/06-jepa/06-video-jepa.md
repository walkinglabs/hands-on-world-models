# 6.6　动手：审问一个视频 JEPA

> **本节目标**：不是把 feature loss 降到最低，而是弄清表示保留了什么、是否受动作控制、能否帮助一个小任务。完成一次完整的 JEPA 实验：训练被动 Video-JEPA → 排查坍缩 → linear probe → 加入动作 → 反事实测试 → 一步选择 → 对照实验。

> **本节代码**：[学出视频特征](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/06_jepa/learn-video-features.ipynb) · [检验并控制特征](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/06_jepa/test-and-control-features.ipynb) · [`src/hwm/jepa.py`](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/jepa.py)

> **前置知识**：你已经跑过 JEPA 路线（第 6 章）的两份 Notebook——第一份是被动 Video-JEPA 的 smoke，第二份做 linear probe 与动作条件——知道 online / target encoder 分工、EMA、表示坍缩。本节把它们扩展成完整训练。

---

第一份 Notebook 用 6 段 episode 确认了接口连通：loss 能下降、target 不接收梯度、EMA 在动。第二份用 ridge 探针读出方块位置，再固定历史只换动作。

但 smoke 不是实验。6–10 段 episode、35–40 步更新——这些数字离「特征真的有用」还差很远。更麻烦的是：feature loss 可以降一个数量级，所有样本的特征却几乎指向同一个方向。

本节的任务是：**用完整训练回答「JEPA 特征保留了什么？丢掉了什么？这是否有益？」** 你会亲眼看到 loss 下降但样本余弦相似度贴着 1、target 特征能读出位置而 predictor 读不出来、换五个动作后面位置几乎不动。这些失败不是 bug，是 JEPA 不重建像素之后必须面对的事。

## 为什么本节与 5.7 走不同的路

5.7 用 VQ-VAE + Transformer 重建像素，能画出可观看的画面。本节用 Video-JEPA 只预测特征，不画回像素。

**两种目标，两种代价：**

| 项目                   | 5.7（重建像素）                     | 本节（预测特征）                  |
| ---------------------- | ----------------------------------- | --------------------------------- |
| 预测目标               | 离散 token → 重建画面               | 连续特征 → 不重建                 |
| 可观看性               | 能画出画面，一眼判断好坏            | 无法直接观看，需要 probe          |
| 训练稳定性             | 码本可能坍缩                        | EMA 稳定，但可能表示坍缩          |
| 可控性证据             | 换动作后画面改变                    | 换动作后预测特征与 probe 位置改变 |
| smoke 里已经看见的失败 | token accuracy 等于复制上一组 token | loss 降了，样本几乎共线           |

本节的优势是训练更稳、不需要解码器；劣势是屏幕上什么都不画，必须用 spread、探针和下游任务判断特征质量。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/jepa-quality.png" alt="JEPA 的损失、spread 与特征统计" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">左边 loss 在降，中间特征 spread 在告诉你有没有塌成一个点，右边是这些特征自己的统计。屏幕上什么都不画，所以这三张图就是你仅有的眼睛。</div>
</div>

## 本次会得到什么

运行结束后，你会得到：

- 按 episode seed 切开的训练 / 测试 clip，以及历史最后一帧对应的方块位置
- 被动 JEPA 的 feature loss 曲线和 `feature_spread`
- 至少两项坍缩诊断：维标准差、样本间余弦相似度
- 冻结 target 特征上的 ridge 探针，对照「预测训练集均值」这条常数基线
- 动作条件 predictor 相对被动预测的 smooth L1 / MSE
- 同一历史替换五种动作的特征差，以及探针读出的五个位置
- 一步动作选择：五个候选里谁离绿色目标更近
- 与无动作 predictor 的公平对照，以及资源清单

## 怎样运行

仓库中的 Notebook 位于：

```text
notebooks/06_jepa/learn-video-features.ipynb
notebooks/06_jepa/test-and-control-features.ipynb
```

可复用实现位于 `src/hwm/jepa.py`。先装神经网络依赖，再跑 JEPA 路线的单元测试：

```bash
python -m pip install -r requirements-neural.txt
PYTHONPATH=src python -m unittest tests.test_routes_bc -v
```

验证 PyTorch 可用：

```python
import torch
print('PyTorch:', torch.__version__, 'device:', 'cuda' if torch.cuda.is_available() else 'cpu')
```

教学版 smoke 在 CPU 上就能跑。本节的完整训练建议用单张 GPU；没有 GPU 就缩小数据量和更新次数，并明确标注「CPU 缩减版」。

## 第一步：从 patch 到短视频块

图片被切成 patch；视频再多一个时间维度。`patchify_video` 把 `[B,T,C,H,W]` 切成 `[B,T,N,patch_dim]`。三帧 16×16 图片、`patch_size=4`，每帧得到 16 个 4×4 patch，每个 patch 是 \(3\times4\times4=48\) 维。

```python
from hwm.data import make_pixelworld_dataset
from hwm.jepa import jepa_batch_from_episodes, patchify_video

episodes = make_pixelworld_dataset(6, 6, seed=0)
video, actions, positions = jepa_batch_from_episodes(episodes, history_length=3)
patches = patchify_video(video[:2], patch_size=4)
print(tuple(video.shape), tuple(patches.shape))
```

**运行这一步，你会看到什么？** 第一份 Notebook 的默认 seed 下：

```text
video:   (30, 3, 3, 16, 16)
patches: (2, 3, 16, 48)
actions: (30,)   positions: (30, 2)
```

30 条 clip 来自 6 段、每段 6 步：每段能切出 \(6-3+1=4\) 条长度为 3 的窗口。`jepa_batch_from_episodes` 规定：历史含最后一帧 target，动作取窗口里倒数第二个——也就是「倒数第二帧怎样变成最后一帧」的那个按键。位置是最后一帧红通道最大值的平均行列，再除以 15，落在 \([0,1]\)。

判断句：如果动作取错一格，后面所有「特征受动作控制」的结论都不可信。

## 第二步：训练被动 Tiny Video-JEPA

被动版不把真实动作喂给 predictor。`actions=None` 时，动作通道是全零向量。online encoder 读历史 patch，对时间和 patch 做平均，得到一条上下文；predictor 再把它、动作通道和可学习位置拼起来，预测每一块 target patch 的特征。

target encoder 是 online encoder 的一份拷贝，`requires_grad=False`。损失在特征空间算，不重建像素。代码用的是 Smooth L1，不是草稿里常写的纯 L2：

$$
\mathcal{L}_{\text{JEPA}}
= \mathrm{SmoothL1}\Bigl(
    g_\phi\bigl(f_\theta(x_{\lt T}),\, 0,\, p\bigr),\;
    \mathrm{sg}\bigl[f_{\bar\theta}(x_T)\bigr]
\Bigr)
$$

其中 \(p\) 是 patch 位置编码，\(\mathrm{sg}\) 截断梯度。target 参数只走 EMA，不走这条损失：

$$
\bar\theta \leftarrow \tau\,\bar\theta + (1-\tau)\,\theta,
\qquad \tau = 0.99
$$

`update_target` 逐参数执行 `target ← τ·target + (1-τ)·online`。τ 越接近 1，靶子动得越慢。两边若被同一个 loss 一起推，模型很容易找到「所有输入都输出同一个常量」的捷径。

```python
from hwm.jepa import TinyVideoJEPA, feature_spread

model = TinyVideoJEPA(feature_size=16)
loss, prediction, target, features = model.loss(video, actions=None)
print(tuple(prediction.shape), tuple(target.shape), target.requires_grad)

params = list(model.online_encoder.parameters()) + list(model.predictor.parameters())
opt = torch.optim.Adam(params, lr=3e-3)
losses, spreads = [], []
for _ in range(35):
    opt.zero_grad()
    loss, _, _, features = model.loss(video, actions=None)
    loss.backward()
    opt.step()
    model.update_target(momentum=0.99)
    losses.append(float(loss.detach()))
    spreads.append(float(feature_spread(features).detach()))
print(losses[0], '→', losses[-1], 'spread', spreads[-1])
```

**运行这一步，你会看到什么？** 第一份 Notebook、`seed=0`、CPU、PyTorch 2.11.0：

```text
prediction/target: (30, 16, 16) (30, 16, 16)
target requires_grad: False
feature loss: 0.238 → 0.030
feature spread: 0.240 → 0.326
online+predictor 参数: 7456
target encoder 参数: 4304（冻结）
```

判断句：loss 下降且 spread 没有掉到 0，只说明「还没有塌成一个点」。它不说明特征里有位置，更不说明特征受动作控制。第一份 Notebook 到这里就停了；本节必须继续审问。

棋盘 mask 只是接口检查：第一份 Notebook 用 `[1,0]` 重复遮住一半 patch，masked loss 仍是 0.030，240 个位置参与平均。真正的 V-JEPA 会遮时空管块，教学版先看清「只在被选中的位置算误差」。

## 第三步：排查表示坍缩

表示坍缩是所有输入映射到同一个特征。`feature_spread` 是把特征展平后、每个维度标准差再取平均。接近零就危险。但它会被整体缩放骗过：向量可以变长，却仍然共线。

所以至少再报一项样本间相似度。把每个 clip 的 patch 特征平均后做余弦：

```python
import torch.nn.functional as F

with torch.no_grad():
    _, _, feat = model(video, actions=None)
flat = feat.reshape(-1, feat.shape[-1])
pooled = F.normalize(feat.mean(dim=(1, 2)), dim=-1)
sim = pooled @ pooled.T
off = sim[~torch.eye(len(sim), dtype=torch.bool)]
print('dim std mean/min:', float(flat.std(0).mean()), float(flat.std(0).min()))
print('off-diag cosine mean/max:', float(off.mean()), float(off.max()))
```

第一份 Notebook 训练结束后的实测：

```text
feature dim std mean: 0.327    min dim std: 0.105
pooled 样本间余弦: mean 0.9997, max 1.000
```

判断句：spread 从 0.24 升到 0.33，看起来「没有坍缩」；余弦 0.9997，说明 30 条 clip 的平均特征几乎在一条射线上。尺度还在，方向没了。只报 spread、不报相似度，会把这次失败写成成功。

本节至少保留四条诊断，并解释它们为什么可能打架：

```text
feature loss
feature_spread（维标准差）
样本间余弦 / 最近邻相似度
特征协方差是否接近「一个方向」
```

如果 loss 下降、spread 还在、余弦却贴着 1，结论应当写成：模型找到了一条有尺度、无方向的捷径。不要写成「表示健康」。

## 第四步：Linear probe——特征里到底有没有位置

特征不能直接看。线性探针冻结 encoder，只用一个仿射层去读方块中心。第二份 Notebook 里的位置是连续坐标，不是四分类，所以指标是 MSE，不是「准确率 0.78」。

闭式 ridge，实现见 `fit_linear_probe_weights`：

$$
W^{*} = (Z^{\top}Z + \lambda I)^{-1} Z^{\top} Q,
\qquad \lambda = 10^{-3}
$$

\(Z\) 是冻结特征加一列 1，\(Q\) 是归一化的 \((r,c)\)。平均所有 patch 会抹掉位置，所以第二份 Notebook 把 16 个 patch 的 16 维特征 flatten 成 256 维，保留每个格子在网格里的槽位。

探针必须按 episode seed 分开拟合和测试。第二份 Notebook 用 `seed=2` 的 10 段训练、`seed=31` 的 4 段测试。常数基线是「永远预测训练集位置的均值」。

```python
from hwm.jepa import fit_linear_probe_weights, apply_linear_probe

with torch.no_grad():
    _, train_target, _ = model(video, actions)
    _, test_target, _ = model(test_video, test_actions)
weights = fit_linear_probe_weights(train_target.flatten(1), positions)
pred = apply_linear_probe(test_target.flatten(1), weights)
probe_mse = torch.nn.functional.mse_loss(pred, test_positions)
constant = positions.mean(0).expand_as(test_positions)
print(float(probe_mse), float(torch.nn.functional.mse_loss(constant, test_positions)))
```

第二份 Notebook 在 Action-JEPA 的 target 特征上实测：

```text
held-out probe MSE: 0.00525
常数基线 MSE:     0.02226
```

同一套权重，如果改去读 **predictor 吐出的特征**，held-out MSE 变成 0.0224，和常数基线一样差。

判断句：target encoder 里还有位置，线性层读得出来。predictor 的未来特征里，位置几乎读不出来。probe 成绩证明某种信息可读，不证明预测出来的未来也能用。非线性探针会得到更好的数，但更差的解释——线性探针才说明特征本身好不好用。

## 第五步：加入真实动作，训练 Action-JEPA

被动 JEPA 的动作通道是零。Action-JEPA 把最后一个动作送进 `Embedding(5, 8)`，再和上下文、位置编码拼接。这不是 5.7 的 FiLM，就是拼接：

```text
history feature ──┐
action embedding ─┼─ concat ─► MLP predictor ─► 下一帧 patch 特征
patch position  ──┘
```

损失仍是 predictor 与冻结 target 之间的 Smooth L1。同一套权重上，再算一遍「把动作通道清零」的被动预测，才能知道动作有没有进动态。

```python
model = TinyVideoJEPA(feature_size=16)
params = (
    list(model.online_encoder.parameters())
    + list(model.predictor.parameters())
    + list(model.action_embedding.parameters())
)
opt = torch.optim.Adam(params, lr=3e-3)
for _ in range(40):
    opt.zero_grad()
    loss, _, _, features = model.loss(video, actions)
    loss.backward()
    opt.step()
    model.update_target(0.99)

with torch.no_grad():
    pred_a, tgt, _ = model(video, actions)
    pred_p, _, _ = model(video, None)
print(float(torch.nn.functional.mse_loss(pred_a, tgt)),
      float(torch.nn.functional.mse_loss(pred_p, tgt)))
```

第二份 Notebook、40 步、CPU 实测：

```text
action-JEPA loss: 0.184 → 0.026
feature spread:   0.227 → 0.281
参数量: 7496

              Smooth L1    MSE
train 有动作    0.0264    0.0540
train 无动作    0.0350    0.0707
test  有动作    0.0328    0.0673
test  无动作    0.0411    0.0834
```

判断句：有动作比无动作低一截，说明 predictor 确实在用这个 8 维 embedding。差距不大，而且这是特征空间的误差，不是「方块走对了」。下一步必须换动作看位置。

## 第六步：反事实——同一历史，只换动作

固定第一条训练 clip，复制成 5 份，分别送 `stay / left / right / up / down`。特征差是：

$$
\Delta_k = \bigl\| g_\phi(h, a^{(k)}) - g_\phi(h, a^{(0)}) \bigr\|_2^2
$$

五个 \(\Delta_k\) 都接近 0，动作就没进动态。只看 \(\Delta_k\) 也不够：模型可能对动作敏感，却敏感在无关维度上。所以再用同一个 held-out probe，把五个预测映射回位置。

```python
same = video[:1].expand(5, -1, -1, -1, -1)
with torch.no_grad():
    predictions, _, _ = model(same, torch.arange(5))
deltas = [(predictions[0] - predictions[i]).square().mean().item() for i in range(1, 5)]
pos = apply_linear_probe(predictions.flatten(1), weights)
print(deltas)
print(pos)
```

第二份 Notebook 实测：

```text
换动作后的 feature MSE vs stay: 0.00163, 0.00196, 0.00281, 0.00256
五个探针位置（归一化行, 列）:
  stay  (0.307, 0.263)
  left  (0.295, 0.286)
  right (0.299, 0.290)
  up    (0.303, 0.314)
  down  (0.281, 0.295)
```

真实方块每次移动约 \(1/15 \approx 0.067\)。五个预测挤在 0.03 的小团里，方向也对不上「上减行、右加列」。

判断句：特征动了，位置没按键走。第二份 Notebook 里的 `assert max(differences) > 0` 只保证不是完全死的接口。本节必须写出：动了多少、方向对不对、和真实步长比是不是小了一个数量级。

## 第七步：一步选择，不要冒充 MPC 成功

有了能区分动作的预测，就可以做最朴素的规划：对五个动作各预测一步，用探针把特征变回位置，选离绿色目标更近的那个。

$$
a^{*} = \arg\min_{a\in\{0,\dots,4\}}
\bigl\| W\,g_\phi(h,a) - g_{\text{goal}} \bigr\|_2
$$

目标在像素 \((12,12)\)，归一化后是 \((0.8, 0.8)\)。第二份 Notebook 在第一条训练历史上选出动作 3（`up`），五个距离分别是 0.729、0.721、0.715、0.696、0.724。因为五个点挤在一起，这个 `argmin` 对噪声极其敏感。

判断句：这是一步 lookahead 接口，不是 horizon=3、candidates=10 的 MPC，更不是「成功率 0.67」。若 probe 在 held-out episode 上已经退化到常数基线，这个选择没有依据。本节若要做短 horizon 搜索，必须回到真实 PixelWorld 执行第一步，报成功次数，并对照随机动作。smoke 没有这个数字，就写「未跑」，不要编。

## 第八步：公平对照

固定 split、seed、更新数。至少比较：

```text
                  probe MSE（target）  probe MSE（预测）  有动作 MSE  无动作 MSE
被动 JEPA         待你补全              待你补全           —           待你补全
Action-JEPA       0.00525               0.0224            0.067       0.083
复制上一帧位置    取决于相邻帧有多像
常数位置          0.0223                0.0223            —           —
```

如果 Action-JEPA 的预测探针没有低于常数基线，动作条件只赢在特征损失，没有赢在「能读出的未来位置」。如果像素预测加探针更好，说明 JEPA 丢掉的信息里包含了这道题真正需要的东西。

## 第九步：记录资源

提交显存、时间、曲线与 checkpoint 哈希。下面是格式，不是实测账单。

```text
Resource log:
  device: ...
  passive JEPA time: ...
  action-JEPA time: ...
  peak allocated / reserved: ...
  total time: ...
  checkpoint hash: sha256:...
```

## 必交证据

缺一不可：

1. **数据卡**：clip 长度、分辨率、动作取自窗口哪一格、按 episode seed 的切分。
2. **坍缩诊断**：loss、spread、至少一项相似度或协方差。只报 loss 下降不得分。
3. **Linear probe**：held-out MSE，对照常数基线；写明探针读的是 target 还是 predictor。
4. **被动 vs 动作条件**：同一模型、同一批数据上的两套特征误差。
5. **反事实**：同一历史、五种动作的特征差和探针位置。特征差大于 0 却位置不动，要写出来。
6. **一步选择或短 horizon 控制**：要么在真实环境执行并报对照，要么明确标为接口演示、未评成功率。
7. **对照表**：被动 / 动作条件 / 常数位置，必要时加像素预测。
8. **资源清单**：环境、显存、时间、checkpoint 哈希。

## 评分

| 项目         | 分数 | 检查重点                                         |
| ------------ | ---: | ------------------------------------------------ |
| 数据与切分   |   10 | 按 episode seed 切开，动作与最后一帧对齐         |
| 坍缩诊断     |   20 | loss、spread、相似度齐全；能解释它们为何可能打架 |
| Probe        |   15 | held-out MSE + 常数基线；不把回归写成分类准确率  |
| 动作条件     |   15 | 有动作 / 无动作分开报，不用被动视频声称可控      |
| 反事实与选择 |   20 | 换动作后看位置，不看「feature MSE > 0」了事      |
| 对照与资源   |   10 | 一次只换一个轴；哈希与显存齐全                   |
| 表达与复现   |   10 | Notebook 可运行，失败写得诚实                    |

## 24GB 目标

`8–16` 帧、`112×112` 或更小分辨率、Tiny/Small ViT、混合精度可选，峰值 reserved 目标不超过 22GB。当前未完整实测。

两份 Notebook 的全部数字都来自 16×16、6–10 段 episode、`PatchEncoder` 而不是 ViT、CPU。把 0.030 的 feature loss 写成「Tiny ViT 已经训好」，算作编造。

## 选做迁移

在 UCF101-mini 上做被动视频表示迁移。它可以报告 action recognition 或 probe，不能用于声称机器人控制。没有动作标签的数据，不得写入「Action-JEPA 成功」。

## 最后回答

完成所有实验后，回答三个问题。用你自己的数字，不要抄下面的例句。

**1. 模型丢掉了什么？**

JEPA 不重建像素，所以它丢掉了像素级信息。第二份 Notebook 里更具体：target 特征还留着位置（probe MSE 0.005），predictor 的未来把位置丢掉了（0.022，等于猜均值）。丢掉的是「下一步方块在哪」，不是草地纹理。

**2. 在当前任务里这是否有益？**

如果下游只需要「这一帧方块在哪」，冻结 target 已经够用，预测未来反而有害。如果下游需要按键后的位置，当前 predictor 没有交出可用坐标。有益与否取决于你问的是表示还是动态。

**3. 换一个任务以后，原来被丢掉的信息会不会重新变得重要？**

会。换到需要外观、遮挡后重现、或像素级可看性的任务，这条特征就不够。V-JEPA 2 后来补上动作条件和机器人数据，正是因为被动特征回答不了「如果手往这边伸会怎样」。

## 已知简化与坑

- **PixelWorld 仍然简单**。16×16、红色方块、5 个动作——这不是真实视频。JEPA 在这里很容易把 loss 打到 0.03。
- **数据量仍然有限**。第一份 Notebook 只有 30 条 clip。余弦 0.9997 也可能是样本太少、特征维 16 太大。
- **损失是 Smooth L1，不是 L2**。对照别人的公式时先对上 `TinyVideoJEPA.loss`。
- **动作进入方式是拼接，不是 FiLM**。不要把 5.7 的注入表直接抄过来。
- **EMA momentum 是关键超参**。太小，靶子跳得快，容易共谋；太大，靶子几乎不动，online 在追过期目标。
- **Linear probe 是线性**。非线性探针分数更高，解释更差。
- **被动视频不能证明 controllability**。必须分开报告 passive 和 action-conditioned。
- **一步 `argmin` 不是 MPC 成功率**。没有真实环境回合，就不要写成功率。

## 扩展练习

跑通默认配置后，按从便宜到昂贵的顺序推荐：

1. **EMA 扫描**：momentum 从 0.9 扫到 0.999，同时画 spread 和样本余弦。spread 上升、余弦贴 1 的区间，就是第一份 Notebook 已经踩到的坑。
2. **探针读谁**：同一套权重分别读 online 均值、target flatten、predictor flatten。三者差距就是「表示里有什么」和「预测出了什么」的差价。
3. **非线性探针**：两层 MLP 读位置。如果线性很差、MLP 很好，特征里有信息，只是没摆成一条直线。
4. **真实一步控制**：把选出的动作送回 `MovingSquareWorld`，看方块是否更靠近目标。对照随机动作。这才是下游数字。

## 本节小结

- **本节是 JEPA 路线的小整机**：从 smoke 扩展到完整训练，弄清表示保留了什么、是否受动作控制、能否帮助小任务。
- **loss 下降不是健康证明**。第一份 Notebook 里 loss 0.238→0.030，spread 还升了，样本余弦却是 0.9997。
- **Probe 要报 MSE，并说清读的是谁**。target 上 0.005 赢了常数基线 0.022；predictor 上打平。
- **动作进了网络，不等于位置听按键**。有动作 MSE 更低，五个探针位置仍挤在 0.03 的小团里。
- **一步选择只是接口**。没有真实执行，就没有成功率。
- **JEPA 丢掉了像素，也丢掉了未预测出来的位置**。这是否有益，取决于下游问的是哪一句。
- **24GB 目标是设计目标**：当前未完整实测。

从第一份 Notebook 的 6 段 episode 到本节的完整训练，规模的变化让你亲眼看到 JEPA 的核心挑战：不画像素之后，眼睛只剩下统计量；统计量之间会打架；动作可以降低损失，却不保证未来位置可读。与 5.7 的像素重建相比，这是另一条路，不是更轻松的路。

## 后续工作

本节用最小的特征预测模型问了三件事：**表示里有什么、动作能不能把它分开、分开之后能不能选一步。** 后面的工作把同一问题做到了图像、视频和机器人。

**I-JEPA**（Assran 等人，2023）是图像版：遮住大块区域，预测被遮区域的抽象表示，而不是像素。第一份 Notebook 的 mask 接口对应这篇的出题方式，只是视频多了一个时间维。

**V-JEPA**（Bardes 等人，2024）把同一骨架搬到视频，用时空 mask 和 EMA 目标编码学运动与对象一致性。课程的 Tiny Video-JEPA 是它的接口缩影，不是它的规模复现。仓库第 6 章收录的正式条目是 [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)。

**V-JEPA 2 / V-JEPA 2-AC**（Bardes 等人，2025）在被动视频表示之上补了动作条件，并用大约 62 小时机器人数据做零样本规划。第二份 Notebook 的「有动作才谈控制」在这里变成一条完整的研究路线。

**VICReg**（Bardes 等人，2022）从另一侧进攻坍缩：直接约束方差、不变性和协方差。第一份 Notebook 里余弦贴 1、spread 却还在，正是「只看一项统计会误判」的例子；VICReg 把几项统计写进损失，而不是只写进日志。

**Dreamer** 与 Action-JEPA 在「特征空间里预测未来」上接壤。差别是 Dreamer 继续训 reward、continue、Actor 和 Critic，目标是真实回报；JEPA 停在表示。一步规划若赢不过「保持原动作」，先怀疑表示里没有可控信息，再去加搜索。

## 参考文献

1. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. _OpenReview_. [链接](https://openreview.net/pdf?id=BZ5a1r-kVsf) —— JEPA 的立场论文：预测特征，而不是像素。
2. Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. _CVPR 2023_. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) —— I-JEPA：图像上的掩码特征预测。
3. Bardes, A., et al. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) —— V-JEPA：视频版 JEPA，时空掩码与 EMA 目标编码。
4. Bardes, A., et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) —— 动作条件 V-JEPA 2-AC 与机器人规划。
5. Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. _ICLR 2022_. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906) —— 防坍缩正则：方差、不变性、协方差要一起看。
6. Grill, J.-B., et al. (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. _NeurIPS 2020_. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733) —— BYOL：EMA target encoder 的自监督先例。
7. Hafner, D., et al. (2019). Dream to Control: Learning Behaviors by Latent Imagination. _ICLR 2020_. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) —— 同是隐空间预测，但继续走到 Actor-Critic 与真实回报。
