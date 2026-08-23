# 5.5 动手：交互视频实验

> **本节目标**：在 PixelWorld 上跑通一台离散视频世界模型。用 VQ-VAE 把每一帧压成离散 token，再用动作条件 Transformer 猜下一组编号，解码后你能看见画面。

> **本节代码**：[B1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/05_interactive_video/B1-compress-and-predict-video.ipynb) · [B2 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/05_interactive_video/B2-make-video-controllable.ipynb)

> **前置知识**：你已经读过 4.1–4.4，知道 VQ 码本、STE 与动作注入。最好刚跑完 [4.6 动手：复现 World Models](/chapters/04-decision-and-planning/06-reproduce-world-models) 和 [4.7 动手：决策与规划实验](/chapters/04-decision-and-planning/07-decision-and-planning)。这一节把「按键之后画面会不会跟着变」真跑一遍。

---

4.6 和 4.7 都还在「画出下一帧」。VAE 重建赛车，RSSM 重建红方块。你看着糊掉的画面，至少知道模型有没有在胡编。

可有人会问：决策真的需要把草地的纹理画回来吗？视频世界模型还要多回答一件事——从同一段历史出发，按下不同的键，接下来的画面会不会发生相应变化？

你当时大概和我一样，第一反应是：「token 猜对了，红块就一定会走吗？」

这一节把每一帧压成离散 token，再按动作生成下一段你能看见的视频。跑完你会发现一件别扭的事：token accuracy 可以很高，画面却比你预想的更难看，连续按键两步就会停住。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-pixelworld.png" alt="PixelWorld 小世界" style="max-width:min(900px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">这就是我们要让模型学会的「世界」：16×16 的黑底小图，红色 3×3 是自己，绿色 3×3 是目标，固定在 (12, 12)。模型从未被告知「红色是智能体」，它要从像素流里自己发现「向右走，红块会右移」。</div>
</div>

## 本次会得到什么

运行结束后，你会得到：

- 64 个对齐好的转移：`current / action / following` 的 shape
- 复制上一帧的像素 MSE，以及它在方向准确率上的零分
- VQ 预热损失、打开码本后的损失，以及 16 个码字的真实占用
- 一步 token accuracy，和把 token 画回画面之后的方向准确率
- `none / additive / film` 三种动作注入的 sensitivity
- teacher-forced 一步分数，以及连续向右 8 步的红块中心
- Diffusion Forcing 三种目标的 shape 与标准差

## 怎样运行

两份 Notebook 在：

```text
notebooks/05_interactive_video/B1-compress-and-predict-video.ipynb
notebooks/05_interactive_video/B2-make-video-controllable.ipynb
```

需要 PyTorch：

```bash
python -m pip install -r requirements-neural.txt
```

教学版在 CPU 上运行，不需要 GPU。即使暂时不打开 Notebook，也可以先跑测试：

```bash
PYTHONPATH=src python -m unittest tests.test_routes_bc -v
```

做完这一节再进入 [PA1-B](/assignments/pa1-b)。特征空间预测留给 [6.5](/chapters/06-jepa/05-jepa)。

## 第一步：先把时间接对

第 `t` 个按键造成 `frame[t] → frame[t+1]`。把动作错开一格，模型会学到一套看似收敛却无法控制的规律。B1 先检查这件事。

```python
from hwm.data import make_pixelworld_dataset
from hwm.video import video_batch_from_episodes

episodes = make_pixelworld_dataset(num_episodes=8, length=8, seed=0)
current, actions, following = video_batch_from_episodes(episodes)
print("current/action/following:",
      tuple(current.shape), tuple(actions.shape), tuple(following.shape))
```

`video_batch_from_episodes` 把每段 episode 切成转移：观察丢掉最后一帧当 `current`，丢掉第一帧当 `following`，动作原样留下。8 段、每段 8 步，一共 64 个转移。

**运行这一步，你会看到什么？**

```
current/action/following: (64, 3, 16, 16) (64,) (64, 3, 16, 16)
```

64 张当前帧，64 个动作，64 张下一帧。三者长度必须相等。动作是 5 个整数：`0 stay / 1 left / 2 right / 3 up / 4 down`。seed=0 这一批里，stay 11、left 10、right 17、up 9、down 17——随机策略，没有专家示范。

如果时间维对不上，后面所有「按键改变画面」的实验都会看错帧。世界模型里，时间对齐比模型结构更先出错。

## 第二步：复制上一帧能有多好

相邻帧本来就很像。红块每步最多走一格，16×16 的黑底几乎不动。把当前帧原样拿去当下一帧，像素 MSE 会很好看。它完全不读动作，也不会让物体移动。B1 先把它留下来，当作后面所有模型的下限。

```python
import torch.nn.functional as F
from hwm.video import motion_direction_accuracy

copy_mse = F.mse_loss(current, following)
copy_direction = motion_direction_accuracy(current, current, following)
print("复制帧 MSE:", round(float(copy_mse), 5))
print("复制帧方向准确率:", copy_direction)
```

`motion_direction_accuracy` 只看红块中心的主位移方向，并且丢掉真实位移小于 0.25 的 stay 样本。复制当前帧时，预测位移是零，所以只要红块真的动了，这一项就是 0。

**运行这一步，你会看到什么？**

```
复制帧 MSE: 0.00647
复制帧方向准确率: 0.0
```

像素分数很漂亮，方向分数是零。4.7 的 A1 已经让你见过这件事：15 步 RSSM 重建仍然赢不过复制上一帧。这里再次出现，是因为离散视频模型也很容易被「背景没变」骗过去。

**这就是复制帧基线的用途**：后面无论 token accuracy 多高，只要画回来的红块不会走，你就还没有一台视频世界模型。

## 第三步：先预热 Autoencoder，再打开码本

像素最直接，连续 latent 适合去噪，语义特征适合只保留任务信息。B1 选离散 token，是为了让下一步用交叉熵做离散预测。一张 `16×16` 图经过两层 stride-2 卷积，变成 `4×4` 个 8 维向量；每个向量在 16 个码字里找最近邻，得到 16 个编号。

最近邻没有普通梯度。直通估计器（STE）在前向使用选中的码字 \(e_k\)，在反向把解码器的梯度近似送回编码器：

$$
k = \arg\min_j \|z_e - e_j\|_2^2, \qquad
z_q = e_k, \qquad
\tilde{z} = z_e + \mathrm{sg}[z_q - z_e]
$$

量化损失按 van den Oord 等人的 VQ-VAE [1] 拆成两项。源码里 \(\beta = 0.25\)：

$$
\mathcal{L}_{\text{quant}}
= \underbrace{\|z_q - \mathrm{sg}[z_e]\|_2^2}_{\text{码本}}
+ \beta \underbrace{\|z_e - \mathrm{sg}[z_q]\|_2^2}_{\text{commitment}}
$$

重建项不是普通 MSE。红块只占少量像素，一张几乎全黑的图也能拿到很低的平均误差。源码用前景加权：目标图里通道最大值大于 0.2 的位置，权重大约是背景的 13 倍，再除以均值，避免被大面积黑底淹没。

小数据上还有一个更常见的失败：16 个码字全部挤进同一个。B1 先做 30 步普通 Autoencoder 预热，再用 encoder 特征做远点采样初始化码本，然后才打开量化。

```python
from hwm.video import TinyVQVAE

images = torch.cat((current, following))
tokenizer = TinyVQVAE(codebook_size=16, embedding_size=8)
optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
for _ in range(30):
    optimizer.zero_grad()
    loss, _ = tokenizer.continuous_loss(images)
    loss.backward()
    optimizer.step()
tokenizer.initialize_codebook(images)
print("AE warm-up loss:", round(float(loss.detach()), 4))
```

**运行这一步，你会看到什么？**

```
AE warm-up loss: 0.1695
```

预热结束，码本还没参与训练。接下来把学习率降到 `1e-4`，打开 VQ：

```python
optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-4)
losses = []
for _ in range(20):
    optimizer.zero_grad()
    output = tokenizer(images)
    output["loss"].backward()
    optimizer.step()
    losses.append(float(output["loss"].detach()))
used_codes = torch.unique(output["tokens"]).numel()
print("VQ loss:", round(losses[0], 4), "→", round(losses[-1], 4),
      "used codes:", used_codes)
print("token grid:", tuple(output["tokens"].shape))
```

**运行这一步，你会看到什么？**

```
VQ loss: 0.1692 → 0.1594 used codes: 16
token grid: (128, 4, 4)
```

128 张图，每张 `4×4` 个 token。16 个码字都用上了，没有坍成一个。但占用极不均匀：0、9、12、15 四个码字扛了大部分像素，其余码字只有几十次。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-codebook.png" alt="VQ 码本占用" style="max-width:min(720px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">20 步 VQ 之后的码字计数。16 个都出现了，但 12 号出现 567 次，最小的只有 21 次。码本「用上了」不等于「用均匀了」。</div>
</div>

把解码结果和原图、复制上一帧并排放在一起，你会立刻看到教学步数的上限：

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-vq-recon.png" alt="VQ 重建对照" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">同一批帧。上排是原图，中排是复制上一帧，下排是 20 步 VQ 之后的解码。红块糊成棋盘格。重建项是前景加权损失 0.1533，不能拿去和复制帧的普通 MSE 0.00647 直接比大小。</div>
</div>

**这就是 VQ 在这份小数据上的真实样子**：接口通了，码本没死，画面还远远不能当「看得见的世界」。PA1-B 才会把 tokenizer 训到能辨认方块。

**一个值得做的实验**：把 `codebook_size` 从 16 提到 64，观察 `used codes` 和占用直方图。码本越大，每个码字代表的模式越细，但 128 张 16×16 的小图喂不饱它——使用率会掉下去。这是表示容量和数据量的张力，不是调参玄学。

## 第四步：用当前 token 与动作猜下一组编号

一张图现在只剩 16 个编号。B1 用一层很小的 Transformer，读取当前帧的全部 token，加上动作，预测下一帧每个位置的码字。训练目标是交叉熵：

$$
\mathcal{L}_{\text{AR}}
= -\sum_{i=1}^{16} \log p_\theta\bigl(z_{t+1}^{(i)} \mid z_t, a_t\bigr)
$$

源码默认 `action_injection="additive"`：动作 embedding 直接加到每个 token 上。帧内 16 个位置同时可见，没有因果掩码——时间因果性由「当前帧预测下一帧」的数据对齐保证，不是靠注意力三角阵。B2 再拿同一套宽度，和 `none`、`film` 比。

```python
from hwm.video import ActionTokenTransformer, token_accuracy, red_centers

with torch.no_grad():
    current_tokens = tokenizer.encode_tokens(current).flatten(1)
    next_tokens = tokenizer.encode_tokens(following).flatten(1)
dynamics = ActionTokenTransformer(codebook_size=16, model_size=32)
optimizer = torch.optim.Adam(dynamics.parameters(), lr=3e-3)
losses = []
for _ in range(35):
    optimizer.zero_grad()
    loss = dynamics.loss(current_tokens, actions, next_tokens)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
with torch.no_grad():
    logits = dynamics(current_tokens, actions)
    predicted = tokenizer.decode_tokens(logits.argmax(-1).reshape(-1, 4, 4))
accuracy = token_accuracy(logits, next_tokens)
direction = motion_direction_accuracy(current, predicted, following)
print("token loss:", round(losses[0], 3), "→", round(losses[-1], 3))
print("token accuracy:", round(float(accuracy), 3),
      "decoded motion direction:", round(float(direction), 3))
print("predicted/true center:",
      red_centers(predicted[:1]), red_centers(following[:1]))
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-vq-pipeline.png" alt="VQ 与 Transformer 管线" style="max-width:min(900px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">路线 B 的数据流。两层卷积把 16×16 压成 4×4×8，码本换成编号，Transformer 按动作猜下一组编号，解码器再画回画面。B1 默认 additive，还不到 FiLM。</div>
</div>

**运行这一步，你会看到什么？**

```
token loss: 2.873 → 0.850
token accuracy: 0.811
decoded motion direction: 0.283
predicted/true center: [[7.97, 6.73]] [[7.00, 7.00]]
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-token-pred.png" alt="token 解码对照" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">上排是真实下一帧，中排是复制当前帧，下排是 token 预测再解码。token accuracy 0.811，方向准确率只有 0.283。背景编号猜对了，红块还没学会走。</div>
</div>

交叉熵从 2.87 掉到 0.85，随机猜 16 类大约是 \(1/16 = 0.0625\)，0.811 看起来很强。把它画回来，方向准确率只比乱猜好一点。第一张样本的预测中心在 (7.97, 6.73)，真值在 (7, 7)——差在格子内部，看不出「向哪个键走了一步」。

**这就是 B1 真正要你看见的事**：离散 AR 可以在小数据上把 token 拟合得很准，而不等于学会了动作条件动态。某一个数字变好，不代表整台模型已经可控。

## 第五步：动作怎样进入模型

B2 只换一个条件。token、模型宽度和数据都固定，把 `action_injection` 在 `none / additive / film` 之间切换。`none` 是必要基线：换动作，输出必须完全一样。additive 最简单。FiLM 用动作产生一组缩放和平移 [4]：

$$
\tilde{h} = (1 + \gamma(a)) \odot h + \beta(a)
$$

复杂方法不一定更好。B2 用同一起点、五个动作，看 logits 平均绝对差（sensitivity），并解码出红块中心。

```python
same_start = current_tokens[:1].expand(5, -1)
all_actions = torch.arange(5)
for injection in ("none", "additive", "film"):
    torch.manual_seed(7)
    model = ActionTokenTransformer(
        codebook_size=16, model_size=32, action_injection=injection
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(25):
        optimizer.zero_grad()
        train_loss = model.loss(current_tokens, actions, next_tokens)
        train_loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(same_start, all_actions)
        frames = tokenizer.decode_tokens(logits.argmax(-1).reshape(-1, 4, 4))
    sensitivity = (logits - logits[:1]).abs().mean().item()
    print(injection,
          "loss=", round(float(train_loss.detach()), 3),
          "sensitivity=", round(sensitivity, 4),
          "centers=", red_centers(frames).tolist())
```

**运行这一步，你会看到什么？**

```
none     loss= 0.757  sensitivity= 0.0000  centers 全是 [6.51, 7.41]
additive loss= 0.868  sensitivity= 0.1886  centers 在 6.51–6.76 之间微动
film     loss= 0.797  sensitivity= 0.1992  centers 几乎不动
```

三种注入的 token accuracy 都在 0.80–0.83，方向准确率都在 0.31–0.32。FiLM 没有赢。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-action-ablation.png" alt="动作注入消融" style="max-width:min(720px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">同一起点换五个动作之后，logits 相对 stay 的平均绝对差。none 必须是 0。additive 和 FiLM 说明动作进了网络，不说明方向已经对。</div>
</div>

**这就是动作消融在教学规模上的诚实结论**：`none` 用来抓接口错误；additive 和 FiLM 都能让输出随动作变。谁更好，要看解码位置和长时生成，不能看 loss，更不能先写「FiLM 最高」再填数字。

## 第六步：Teacher forcing 与 free rollout

一步预测时，输入来自真实 token；自由生成时，第二步开始输入来自模型自己。两者的差距，就是 4.6 里那张 free-running 图在离散 token 上的对应物。

```python
from hwm.video import rollout_token_model

with torch.no_grad():
    teacher_tokens = dynamics(current_tokens, actions).argmax(-1)
    teacher_frames = tokenizer.decode_tokens(teacher_tokens.reshape(-1, 4, 4))
teacher_direction = motion_direction_accuracy(current, teacher_frames, following)
right_actions = [torch.tensor([2]) for _ in range(8)]
rollout = rollout_token_model(
    dynamics, tokenizer, current_tokens[:1], right_actions, (4, 4)
)
print("teacher-forced token acc:",
      round(float((teacher_tokens == next_tokens).float().mean()), 3))
print("teacher-forced 一步方向准确率:",
      round(float(teacher_direction), 3))
print("free rollout 连续向右时的解码中心:", red_centers(rollout))
```

**运行这一步，你会看到什么？**

```
teacher-forced token acc: 0.802
teacher-forced 一步方向准确率: 0.311
free rollout 连续向右时的解码中心:
  t=0  (6.51, 7.41)
  t=1  (6.76, 7.60)
  t=2  (6.76, 7.60)
  ...
  t=8  (6.76, 7.60)
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-free-rollout.png" alt="自由 rollout" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">连续按 right 八步。第 0 步到第 1 步中心挪了一点点，之后钉在 (6.76, 7.60)。中心停止、反向或消失，都是自由 rollout 暴露的失败。</div>
</div>

teacher-forced 的 token 分数仍然好看。让模型吃自己的输出，红块走一步就停。复合误差在 16 个 token 上一样会发生——而且因为解码器本身还是棋盘格，你甚至很难从画面里读出「它以为自己走到了哪」。

**一个值得做的实验**：把 free rollout 从 8 步提到 20 步，把中心坐标画成曲线。如果从某一步起变成 `NaN`，说明红块在解码里消失了。那比「停住」更糟：模型连物体还在不在都没保住。

## 第七步：Diffusion Forcing 的最小接口

离散 AR 不是唯一生成方式。B2 最后只验证一件事：给 `[B, T, C, H, W]` 里每一帧指定自己的噪声等级，同一份 noisy video 可以配 `x / epsilon / v` 三种监督目标 [5]。历史可以接近干净，较远未来可以更嘈杂。它不冒充大型扩散视频模型。

```python
from hwm.video import (
    TinyConditionalDenoiser,
    add_independent_frame_noise,
    diffusion_prediction_target,
)

clip = following[:12].reshape(4, 3, *following.shape[1:])
levels = torch.tensor([[0.0, 0.5, 1.0]]).expand(4, -1)
noisy_clip, sampled_noise = add_independent_frame_noise(clip, levels)
for name in ("x", "epsilon", "v"):
    target = diffusion_prediction_target(clip, sampled_noise, levels, name)
    print(name, "target shape:", tuple(target.shape),
          "std:", round(float(target.std()), 3))
```

**运行这一步，你会看到什么？**

```
x       target shape: (4, 3, 3, 16, 16)  std: 0.117
epsilon target shape: (4, 3, 3, 16, 16)  std: 0.994
v       target shape: (4, 3, 3, 16, 16)  std: 0.647
tiny epsilon loss: 1.013 → 0.864
```

三种目标的 shape 都和 clip 一致。`x` 是干净画面，标准差最小；`epsilon` 接近标准正态；`v` 介于两者之间。后面那个 tiny denoiser 只做了 20 步、单一噪声等级 0.2 的 epsilon 回归，loss 从 1.013 掉到 0.864。接口通了，不是 GameNGen。

## 运行与产物

```bash
python -m pip install -r requirements-neural.txt
PYTHONPATH=src python -m unittest tests.test_routes_bc -v
```

跑完两份 Notebook 后，你应该有：

- **B1**：复制帧 MSE、码本使用数、token accuracy、解码方向准确率
- **B2**：三种注入的 sensitivity、teacher-forced 与 free rollout 中心

| 项目 | 本节 smoke                     | PA1-B                           |
| ---- | ------------------------------ | ------------------------------- |
| 数据 | 6–12 段 PixelWorld             | 更大的 PixelWorld，选做真实视频 |
| 训练 | 20–40 步，CPU                  | 直到曲线稳定                    |
| 目的 | 检查数据流、码本、动作注入接口 | 检查可观看性是否真能支撑控制    |
| 结论 | 接口通了；画面还很弱           | tokenizer 是否形成稳定闭环      |

## 已知简化与坑

教学版有几处刻意的简化，数字和论文对不上时先从这里找原因：

- **PixelWorld 过于简单。** 16×16、红方块、5 个动作——这不是 YouTube，也不是 Atari。VQ 在这里很容易「用满码本」，换真实视频使用率会低得多。
- **数据量极小。** B1 只有 64 个转移，Transformer 35 步就能把 token 拟合到 0.811。那是过拟合，不是泛化。
- **VQ 只训了 20 步。** 解码是棋盘格。不要把 token accuracy 写成「已经能还原可观看画面」。
- **重建损失是前景加权 MSE。** 权重约 13 倍，数字不能和复制帧的普通 MSE 直接比。
- **B1 的 Transformer 没有因果掩码。** 帧内 token 同时可见。它预测的是「下一整帧的 16 个编号」，不是帧内从左到右的语言建模。
- **B1 默认 additive，不是 FiLM。** B2 才比较三种注入。教学步数里 FiLM 并不更准。
- **B2 的 Diffusion Forcing 只验证接口。** 真正的扩散视频模型需要多步去噪和长 clip。教学版是一步、一个 tiny 卷积。

## 扩展练习

跑通默认配置后，按从便宜到昂贵的顺序推荐：

1. **码本大小扫描**：把 B1 的 `codebook_size` 从 8 扫到 64，画使用率和 token accuracy。有没有一个「再加大就开始空转」的点？
2. **Free rollout 长度**：把 B2 的连续按键从 8 步提到 20 步，观察中心坐标和 `NaN` 出现的步数。

完成后进入 [PA1-B · 动手：做出一个听从按键的视频小世界](/assignments/pa1-b)。下一章换一种预测目标：不画回像素，只预测未来特征，见 [6.5 动手：JEPA 实验](/chapters/06-jepa/05-jepa)。

## 本节小结

- **VQ-VAE 加动作条件 Transformer 做离散预测**，能还原画面，但教学步数里画面是棋盘格，free rollout 两步就会停住。
- **复制上一帧的像素 MSE 是 0.00647，方向准确率是 0。** 像素分数会奖励偷懒，方向分数不会。
- **token accuracy 0.811 和方向准确率 0.283 可以同时成立。** 猜对背景编号，不等于红块会走。
- **动作消融的第一课是 none 必须为零**，第二课才是 additive 和 FiLM 谁更好。这份 smoke 里两者差不多。
- **Smoke 不是完整训练**：8–12 段 episode、20–40 步、CPU 运行。目标是检查数据流，不是复现 IRIS。

从 4.6 的「画出下一帧」，到这一节的离散 token，世界模型的预测目标开始分叉。下一章把像素换成特征。

## 后续工作

B1 / B2 只把离散视频接到了 PixelWorld。它们留下的短板，就是真论文要攻的东西。

### 短板一：16 个码字、一层 Transformer，撑不住可观看的世界

教学版的 tokenizer 20 步之后仍是棋盘格。IRIS [3] 把同一件事做到 Atari：先用 VQ 把画面变成 token，再用 Transformer 按动作自回归生成下一帧，并在这个世界里训练策略。Genie [6] 走得更远，从无动作标注的视频里学出潜在动作，让用户在生成的世界里真的按键。码本利用率、长时一致性和可玩性，是这条线一直在还的债。

### 短板二：teacher forcing 与自由生成不是同一件事

B2 里，一步 token 分数还在 0.80，连续向右第二步就停。Diffusion Forcing [5] 用逐帧不同噪声，把「猜下一帧」和「生成整段」放进同一个训练接口；GameNGen 一类工作则靠噪声增强压自回归漂移。教学版只构造了 `x / epsilon / v`，没有训出一个能玩的引擎。

从 4.6 的「在梦里开车」，到这一节的「按键之后画面会不会变」，被替换的是表示，不是那句老话。下一台模型要回答的，是你刚刚亲眼看见的那些失败：token 很准但方块不走、FiLM 并不自动更强。

## 参考文献

1. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. _NeurIPS 2017_. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937) —— VQ-VAE：码本、STE 与 commitment loss 的原文。
2. Vaswani, A., et al. (2017). Attention Is All You Need. _NeurIPS 2017_. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) —— Transformer：自回归生成的基础架构。
3. Micheli, V., Alonso, E., & Fleuret, F. (2023). Transformers are Sample-Efficient World Models. _ICLR 2023_. [arXiv:2209.00588](https://arxiv.org/abs/2209.00588) —— IRIS：离散 token 加 Transformer 的 Atari 世界模型，是本节最接近的论文形态。
4. Perez, E., et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. _AAAI 2018_. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871) —— FiLM：用条件信号缩放和平移特征。
5. Chen, B., et al. (2024). Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion. _NeurIPS 2024_. [arXiv:2407.01392](https://arxiv.org/abs/2407.01392) —— 逐帧不同噪声等级，统一下一帧预测与整段生成。
6. Bruce, J., et al. (2024). Genie: Generative Interactive Environments. _ICML 2024_. [arXiv:2402.15391](https://arxiv.org/abs/2402.15391) —— 从无动作标注视频学出可交互世界。
