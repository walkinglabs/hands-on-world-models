# 5.7　动手：听从按键的视频小世界

> **本节目标**：完成一次完整、可复现的视频世界模型实验——从同一段历史出发，更换动作，未来随之改变；模型连续读取自己的输出后，仍能说明从哪里开始失效。不是生成最好看的片段，而是用证据回答「模型真的听从按键了吗？」

> **本节代码**：[压缩并预测视频](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/05_interactive_video/compress-and-predict-video.ipynb) · [让视频听动作](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/05_interactive_video/make-video-controllable.ipynb) · [`src/hwm/video.py`](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/video.py)

> **前置知识**：你已经跑过互动视频路线（第 5 章）的两份 Notebook——第一份「压缩并预测视频」搭起 VQ tokenizer 与 token Transformer，第二份「让视频听动作」做动作注入对照和 free rollout——知道码本、直通估计器、动作 embedding。本节把它们扩展成完整训练。

---

第一份 Notebook 用 8 段 16×16 的 PixelWorld 确认了数据能对齐、码本能用起来、交叉熵能下降。第二份 Notebook 用同一套小数据比较了 `none / additive / film`，再让模型连续吃自己的输出。

但 smoke 不是实验。8 段 episode、35 步更新——这些数字离「模型真的听从按键」还差很远。更麻烦的是：第一份 Notebook 的 token accuracy 可以看起来很高，却只是在抄上一帧。

本节的任务是：**用完整训练回答「模型真的听从按键了吗？」** 你会亲眼看到 token accuracy 等于复制上一组 token、换五个按键解码中心几乎不动、free rollout 走一步就停住。这些失败不是 bug，是视频世界模型的核心挑战。

## 为什么本节是互动视频路线的小整机

互动视频路线的叙事是：用离散 token 表示视频帧，用 Transformer 预测下一组 token，用动作条件让模型「听从按键」。两份跟做 Notebook 确认了这套管线在接口层面能跑。本节要确认它在训练层面能否被证伪。

**完整训练意味着什么？**

```text
数据收集 → VQ-VAE 训练 → token 序列化 → 动作条件 Transformer
        → 动作反事实测试 → free rollout 漂移测试 → 对照实验
```

每一步的输出是下一步的输入。如果 VQ-VAE 的码本坍缩，token 序列失去多样性，后面的 Transformer 只能预测常量；如果 Transformer 不读动作，换动作后画面不变；如果 free rollout 误差累积，长视频变成噪声。

本节的目标不是打破这些问题——教学版的数据量和计算量不够。目标是**让你亲眼看到这些问题的存在**，并用证据回答「模型在哪里开始失效」。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/controllable-video.png" alt="同一起点换五个按键" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">上面是原图和重建，下面是损失、码本用了几个字、换一个按键未来会不会变。token accuracy 很高但换按键画面不动——这种失败，才是这一题要你抓住的。</div>
</div>

## 本次会得到什么

运行结束后，你会得到：

- 一张按 episode 切分的数据卡，以及 `frame[t] + action[t] → frame[t+1]` 的对齐检查
- 复制上一帧、复制上一组 token、不读动作，三条基线
- VQ-VAE 的重建误差、码本使用数、红色方块位置误差
- 动作条件 Transformer 的 token 交叉熵、token accuracy、运动方向准确率
- 同一历史替换五种动作的反事实网格
- teacher-forced 与 free rollout 的对照曲线
- 一次只换一个轴的对照表：`none / additive / film`，或 tiny 去噪目标
- 延迟、峰值显存、总时间、checkpoint 哈希

## 怎样运行

仓库中的 Notebook 位于：

```text
notebooks/05_interactive_video/compress-and-predict-video.ipynb
notebooks/05_interactive_video/make-video-controllable.ipynb
```

可复用实现位于 `src/hwm/video.py`。先装神经网络依赖，再跑互动视频路线的单元测试：

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

## 第一步：先把时间接对

第 `t` 个按键造成 `frame[t] → frame[t+1]`。把动作错开一格，模型会学到一套看似收敛却无法控制的规律。

PixelWorld 的真实接口比草稿里常写的「64×64、50 段」更小。`MovingSquareWorld` 默认是 16×16 RGB、边长 3 的红方块、5 个离散动作：`stay / left / right / up / down`。第一份 Notebook 用 8 段、每段 8 步，正好得到 64 个转移。

```python
from hwm.data import make_pixelworld_dataset, ACTION_NAMES
from hwm.video import video_batch_from_episodes

episodes = make_pixelworld_dataset(num_episodes=8, length=8, seed=0)
current, actions, following = video_batch_from_episodes(episodes)
print(tuple(current.shape), tuple(actions.shape), tuple(following.shape))
assert len(current) == len(actions) == len(following)
```

**运行这一步，你会看到什么？** 在第一份 Notebook 的默认 seed 下，形状是：

```text
current/action/following: (64, 3, 16, 16)  (64,)  (64, 3, 16, 16)
action counts: stay 11, left 10, right 17, up 9, down 17
```

`video_batch_from_episodes` 把每段 episode 的 `observations[:-1]`、`actions`、`observations[1:]` 拼成 batch，像素除以 255。这就是后面所有公式里的 \(x_t, a_t, x_{t+1}\)。

判断句：如果这三个长度对不上，后面所有「听从按键」的结论都不可信。

## 第二步：两个基线，先把下限钉死

在训练任何模型之前，先问两个笨问题：复制上一帧能有多好？复制上一组 token 能有多好？

相邻帧本来就很像。红方块每次只走 1 个像素，16×16 画面里大部分是黑底。复制上一帧会得到不错的像素 MSE，但它完全不读动作，也不会让物体移动。

```python
import torch
from hwm.video import motion_direction_accuracy, psnr

copy_mse = torch.nn.functional.mse_loss(current, following)
copy_direction = motion_direction_accuracy(current, current, following)
print('复制帧 MSE:', float(copy_mse))
print('复制帧 PSNR:', float(psnr(current, following)))
print('复制帧方向准确率:', float(copy_direction))
```

`motion_direction_accuracy` 先用 `red_centers` 找红色质心，再比较预测位移和真实位移的主轴与符号；真实位移小于 0.25 像素的 stay 样本会被丢掉。复制上一帧的预测位移是 0，所以凡是真的在动的样本，它全部判错。

第一份 Notebook、`seed=0`、CPU、PyTorch 2.11.0 的实测：

```text
复制帧 MSE: 0.00647
复制帧 PSNR: 21.89
复制帧方向准确率: 0.0
```

打开 tokenizer 之后，还要加一条 **复制上一组 token** 的基线。相邻帧的码字本来就大部分相同：

```text
copy-token accuracy: 0.811
```

判断句：后面任何「token accuracy = 0.81」如果不先和 0.811 比，都只是在重复相邻帧很像这件事。方向准确率才是「有没有让方块走对」的下限。

## 第三步：VQ-VAE，先决定预测什么

一帧 16×16×3 有 768 个数。直接预测所有像素并非不可能，但红方块只占 9 个像素，平均 MSE 会被大面积黑底骗过去。所以 `TinyVQVAE` 不报普通 MSE，而用前景加权：

$$
\mathcal{L}_{\text{recon}}
= \frac{1}{N}\sum_i w_i \,\|x_i - \hat{x}_i\|_2^2,
\qquad
w_i \propto 1 + 12\cdot\mathbf{1}[\max_c x_{i,c} > 0.2]
$$

实现见 `foreground_weighted_mse`。权重按 batch 再归一化，避免只靠放大损失数值假装收敛。

向量量化把连续特征换成码本里最近的字：

$$
k = \arg\min_j \|z_e - e_j\|_2,
\qquad
z_q = e_k
$$

最近邻没有普通梯度。直通估计器在前向使用 \(z_q\)，反向把 Decoder 的梯度近似送回 Encoder：

$$
z_{\text{st}} = z_e + \mathrm{sg}[z_q - z_e]
$$

量化损失与 van den Oord 等人的 VQ-VAE 一致，commitment 系数在代码里是 \(0.25\)：

$$
\mathcal{L}_{\text{VQ}}
= \underbrace{\|x - \hat{x}\|_{w}^{2}}_{\text{重建}}
+ \underbrace{\|\mathrm{sg}[z_e] - e\|_2^2}_{\text{码本}}
+ 0.25\,\underbrace{\|z_e - \mathrm{sg}[e]\|_2^2}_{\text{commitment}}
$$

第一份 Notebook 先预热普通 Autoencoder 30 步，再用 encoder 特征做远点采样初始化码本，最后才打开量化。这是为了避免 64 张小图全部挤进同一个码字。

```python
from hwm.video import TinyVQVAE, red_centers

images = torch.cat((current, following))
tokenizer = TinyVQVAE(codebook_size=16, embedding_size=8)
opt = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
for _ in range(30):
    opt.zero_grad()
    loss, recon = tokenizer.continuous_loss(images)
    loss.backward()
    opt.step()
tokenizer.initialize_codebook(images)

opt = torch.optim.Adam(tokenizer.parameters(), lr=1e-4)
for _ in range(20):
    opt.zero_grad()
    out = tokenizer(images)
    out['loss'].backward()
    opt.step()

used = torch.unique(out['tokens']).numel()
pos_err = torch.nanmean(torch.linalg.vector_norm(
    red_centers(out['reconstruction']) - red_centers(images), dim=-1
))
print('VQ loss:', float(out['loss']), 'used codes:', used)
print('position error px:', float(pos_err))
```

**运行这一步，你会看到什么？** 第一份 Notebook 的默认配方、同一 seed 的实测：

```text
AE warm-up loss: 0.170
VQ loss: 0.169 → 0.159
used codes: 16/16
VQ reconstruction loss: 0.153
VQ pixel MSE: 0.151
红色方块位置误差: 4.30 px
tokenizer 参数量: 5803
```

判断句：16 个码字全用上了，所以这一次不是码本坍缩。但位置误差 4.3 像素，发生在 16×16 的画布上——方块本身只有 3×3。tokenizer 先把物体位置弄丢了，后面的 Transformer 再强也找不回来。

一张 16×16 图经过两层 stride-2 卷积，变成 `4×4=16` 个 token。这就是下一步要预测的全部对象。

## 第四步：用当前 token 和动作预测下一组 token

`ActionTokenTransformer` 不是逐步吐下一个 token 的语言模型。它一次读完整张当前帧的 16 个编号，加上一个动作，一次写出下一帧的 16 个编号。帧内 token 已经同时可见；时间因果性由数据对齐保证，不靠因果掩码。

$$
\mathcal{L}_{\text{AR}}
= \mathrm{CE}\bigl(f_\theta(z_t, a_t),\, z_{t+1}\bigr)
= -\frac{1}{16}\sum_{i=1}^{16}\log p_\theta\bigl(z_{t+1,i}\mid z_t, a_t\bigr)
$$

动作有三种进入方式，对应 `action_injection`：

```text
none      完全不读动作
additive  token embedding + 动作 embedding
film      h' = (1 + γ(a)) ⊙ h + β(a)
```

注意分子：代码里的 FiLM 是 \(1+\gamma(a)\)，不是 \(\gamma(a)\)。\(\gamma,\beta\) 由 `Linear(action_emb) → 2d` 再 `chunk` 得到。第一份 Notebook 默认用 additive；第二份再和 `none`、`film` 比。

```python
from hwm.video import ActionTokenTransformer, token_accuracy

with torch.no_grad():
    current_tokens = tokenizer.encode_tokens(current).flatten(1)
    next_tokens = tokenizer.encode_tokens(following).flatten(1)

dynamics = ActionTokenTransformer(codebook_size=16, model_size=32)  # additive
opt = torch.optim.Adam(dynamics.parameters(), lr=3e-3)
for _ in range(35):
    opt.zero_grad()
    loss = dynamics.loss(current_tokens, actions, next_tokens)
    loss.backward()
    opt.step()

with torch.no_grad():
    logits = dynamics(current_tokens, actions)
    pred = tokenizer.decode_tokens(logits.argmax(-1).reshape(-1, 4, 4))
print('token acc:', float(token_accuracy(logits, next_tokens)))
print('direction:', float(motion_direction_accuracy(current, pred, following)))
```

**运行这一步，你会看到什么？** 第一份 Notebook 默认跑 35 步：

```text
token loss: 2.873 → 0.850
token accuracy: 0.811
copy-token accuracy: 0.811
decoded motion direction: 0.283
Transformer 参数量: 12368
```

判断句：交叉熵确实在降，但 token accuracy 刚好等于复制上一组 token。模型学会了「下一帧很像这一帧」，还没有学会「按右键方块往右走」。0.283 的方向准确率只比乱猜四个方向的 0.25 高一点。

**一个值得做的实验**：把 `codebook_size` 从 16 提到 64，同时盯住使用率和位置误差。码本越大，每个字代表的模式越细，但小数据下使用率会掉——这是表示容量和数据量的张力，不是调参游戏。

## 第五步：动作反事实——同一历史，只换按键

这是本节的核心测试。固定同一组当前 token，分别送进五种动作，看解码中心动不动。logits 变了不算数，方块位置变了才算。

```python
same = current_tokens[:1].expand(5, -1)
all_actions = torch.arange(5)
with torch.no_grad():
    logits = dynamics(same, all_actions)
    frames = tokenizer.decode_tokens(logits.argmax(-1).reshape(-1, 4, 4))
print(red_centers(frames))
print('sensitivity:', float((logits - logits[:1]).abs().mean()))
```

第二份 Notebook 在相同数据、相同宽度、相同 25 步、相同 seed=7 下，把注入方式当作唯一变量。CPU 实测：

```text
none      loss 0.757  token acc 0.827  direction 0.324  sensitivity 0.000
          五个动作的解码中心完全相同: (6.51, 7.41)

additive  loss 0.868  token acc 0.802  direction 0.311  sensitivity 0.189
          中心略有分开，最大位移不到 0.3 像素

film      loss 0.797  token acc 0.809  direction 0.324  sensitivity 0.199
          五个中心几乎叠在一起
```

三种注入的参数量都是 12368——`film` 多一个 `Linear(d, 2d)`，在这个宽度下被四舍五入吃掉了。

判断句：`none` 的 sensitivity 精确为 0，说明接口没写错。additive 和 film 让 logits 动了，但解码位置几乎不动。第二份 Notebook 的跟做 **没有**证明 FiLM 比 additive 更好；方向准确率三者都在 0.31–0.32。更复杂的注入没有提升，也可以得到满分，只要实验公平、解释诚实。

本节必须把这张网格画出来：同一起点，五个按键，五张未来。如果五张图认不出来差别，就不要写「模型听从按键」。

## 第六步：Teacher forcing 与 free rollout

一步预测时，输入来自真实 token；自由生成时，第二步开始输入来自模型自己。两者的差距就是复合误差。

`rollout_token_model` 每一步取 `argmax`，再把预测送回自己：

```python
from hwm.video import rollout_token_model

with torch.no_grad():
    teacher = dynamics(current_tokens, actions).argmax(-1)
    teacher_frames = tokenizer.decode_tokens(teacher.reshape(-1, 4, 4))
print('teacher token acc:', float((teacher == next_tokens).float().mean()))
print('teacher direction:', float(motion_direction_accuracy(current, teacher_frames, following)))

right = [torch.tensor([2]) for _ in range(8)]   # 连续按右
frames = rollout_token_model(dynamics, tokenizer, current_tokens[:1], right, (4, 4))
print(red_centers(frames))
```

第二份 Notebook 里 additive 模型的实测：

```text
teacher-forced token accuracy: 0.802
teacher-forced 方向准确率: 0.311

连续按右的解码中心:
  t=0  (6.51, 7.41)
  t=1  (6.76, 7.60)
  t≥2  (6.76, 7.60)   ← 停住了

第一段 episode 按真实动作自由跑，逐步 token accuracy:
  1.000, 0.938, 0.938, 0.938, 0.812, 0.812, 0.812, 0.750
```

真实下一帧的红点中心在 `(1,2) → (3,3)` 一带走动；模型给出的中心从一开始就漂在画面中部，走一步就冻住。

判断句：teacher-forced 的 0.80 只说明「看见真历史时会抄相邻帧」。free rollout 的中心停住，才是连续生成已经失败。本节至少要在 1、5、15、30 步——若算力够再到 100 步——画出两条曲线，并标出中心停止、反向或消失的第一帧。

## 第七步：二选一对照，一次只换一个轴

在以上基础上，选一个方向做深入，不要两个各做一半。

### 方向 1：动作怎样进入模型

已经有 `none / additive / film`。保持数据、宽度、更新数和 seed 不变，补上动作一致性、参数量和一步延迟。AdaLN 可以自己加，但不是必做。

第二份 Notebook 已经给出一个反例：film 的 sensitivity 略高于 additive，方向准确率没有更高。把这句话写进报告，比抄「FiLM 更好」有用。

### 方向 2：连续 latent 与逐帧噪声

`add_independent_frame_noise` 允许每一帧拥有自己的噪声等级：历史可以接近干净，较远未来可以更嘈杂。同一份 noisy video 可以配三种监督目标：

$$
\begin{aligned}
x &\colon && x \\
\varepsilon &\colon && \varepsilon \\
v &\colon && (1-\ell)\,\varepsilon - \ell\, x
\end{aligned}
$$

其中 \(\ell\in[0,1]\) 是该帧的噪声等级。这是 Diffusion Forcing 的最小接口，不是完整视频扩散。

第二份 Notebook 把 12 帧收成 `[4, 3, 3, 16, 16]`，噪声等级设成 `[0, 0.5, 1.0]`，再拿 `TinyConditionalDenoiser` 做 20 步 epsilon 回归：

```text
x target std: 0.117
epsilon target std: 0.994
v target std: 0.647
tiny epsilon loss: 1.013 → 0.864
denoiser 参数量: 9163
```

判断句：三种目标的 shape 对得上、epsilon 的标准差接近 1，只说明接口通了。20 步损失从 1.01 降到 0.86，不能拿去对比 Genie 或 DIAMOND。

## 第八步：记录资源与产物

记录 tokenizer、动态模型、Decoder 和端到端的延迟、峰值显存、总时间与 checkpoint 哈希。下面是格式，不是实测账单——当前没有本节的 24GB 完整运行记录。

```text
Resource log:
  device: ...
  tokenizer training time: ...
  Transformer training time: ...
  peak allocated / reserved: ...
  total time: ...
  checkpoint hash: sha256:...
```

## 必交证据

缺一不可：

1. **数据卡**：episode 数、每段长度、分辨率、按 episode 的 train/val/test 切分、动作时间对齐检查。
2. **三条基线**：复制上一帧的 MSE / 方向准确率；复制上一组 token 的 accuracy；`none` 注入的反事实。
3. **tokenizer 验收**：重建损失、码本使用数、红色方块位置误差。不得只用普通像素 MSE。
4. **动作反事实网格**：同一历史、五种动作、解码后的中心或画面。logits 变化不能代替位置变化。
5. **漂移曲线**：teacher-forced 与 free rollout，至少覆盖 1/5/15/30 步，并标出首个失败时间。
6. **一次只换一个轴的对照表**：注入方式，或去噪目标。同时更换模型、数据和预算，把差异算进废纸篓。
7. **成功与失败片段**：固定 seed，至少各一张。
8. **资源清单**：环境、显存、时间、checkpoint 哈希。

## 结果页至少包含

- 动作反事实网格：同一起点，五种动作，五种未来
- 对照表格：复制帧、复制 token、`none`、additive、film
- 漂移曲线：teacher-forced vs free rollout
- 成功与失败片段
- 指标汇总：画面指标、动作指标、首个失败时间、端到端延迟
- 资源清单

## 评分

| 项目           | 分数 | 检查重点                                                    |
| -------------- | ---: | ----------------------------------------------------------- |
| 数据与时间对齐 |   10 | 按 episode 切分，`frame[t]+action[t]→frame[t+1]` 可核对     |
| 基线           |   15 | 复制帧、复制 token、`none` 三者齐全，且被后面的数字真正比较 |
| Tokenizer      |   15 | 码本使用、位置误差与重建分开报；能解释小物体有没有丢        |
| 动作反事实     |   20 | 换按键后面面或中心必须变；只报 logits 不得分                |
| Free rollout   |   15 | 有逐步曲线和首个失败时间，不拿 teacher-forced 冒充长时生成  |
| 对照与资源     |   15 | 一次只换一个轴；延迟、显存、哈希齐全                        |
| 表达与复现     |   10 | Notebook 可运行，seed 与输出完整，失败写得诚实              |

## 24GB 目标

建议从 `64×64 RGB`、每帧最多 64 个 token、短上下文和小型 Transformer 开始，单卡 peak reserved 设计目标不超过 22GB。先用两份跟做 Notebook 的配方验证数据与 loss，再扩大 batch 或 horizon。

这是课程设计预算，不是实测结果。当前没有本节的 24GB 完整运行记录；在日志、曲线和 checkpoint 齐全前，不得标为「24GB 已验证」。

两份跟做 Notebook 的全部数字都来自 16×16、8–12 段 episode、CPU。把这些数字写进「完整训练已经听从按键」，算作编造。

## 真实数据迁移（拔高）

可以迁移到 DINO-WM PushT 或 CarRacing 小数据，但必须提交 loader、许可、动作时间、控制频率和固定 split。当前仓库只发布了这些来源的数据合约，尚未发布可复现 artifact，因此不能把下载链接当作已完成实验。

## 已知简化与坑

- **PixelWorld 仍然简单**。16×16、5 个动作、红色方块——这不是 YouTube 视频。VQ tokenizer 在这里很容易把 16 个码字用满，但位置照样可以偏 4 个像素。
- **相邻帧太像**。复制 token 就能到 0.81 accuracy。不设这条基线，交叉熵下降会被误读成「动态学会了」。
- **跟做用的 Transformer 不是逐步语言模型**。它一次预测整帧 16 个 token。不要把本节公式写成「对视频时间逐步 next-token」却调用现在的 `ActionTokenTransformer.loss`。
- **FiLM 在这个宽度上没有赢**。第二份 Notebook 的 direction 三者持平。本节不得把「更复杂的注入一定更好」写成先验结论。
- **Free rollout 崩溃是预期行为**。复合误差在视频生成里不可避免。本节的目标不是消除它，而是标出从哪一步开始中心停住、反向或消失。
- **动作反事实验证的是可控性，不是真实性**。中心动了，画面仍可能是糊的。

## 扩展练习

跑通默认配置后，按从便宜到昂贵的顺序推荐：

1. **把复制 token 画进同一张图**：横轴是更新步数，纵轴同时画 token accuracy 和 copy-token accuracy。两条线重合的区间，就是模型还在抄相邻帧。
2. **Free rollout 长度**：从 8 步提到 30 步，记下中心第一次停止的步数。
3. **码本大小扫描**：8、16、32、64，同时报使用率和位置误差。
4. **换分辨率**：把 PixelWorld 改到 32×32 或 64×64，看位置误差是缩小了还是 tokenizer 先崩。

## 不接受的结论

- 只交最好看的一段视频
- logits 随动作改变，就声称解码画面可控
- 只报 PSNR、FVD 或 one-step loss，不做动作反事实
- token accuracy 高于 0.8，却不报复制上一组 token
- 使用无动作视频，却声称学到了真实控制
- 在 teacher forcing 下表现好，就声称可以长时间生成
- 同时更换模型、数据和训练预算，再把差异归因于某一个组件
- 把两份跟做 Notebook 的数字写成 24GB 完整训练结果

## 本节小结

- **本节是互动视频路线的小整机**：从跟做实验扩展到完整训练，用证据回答「模型真的听从按键了吗？」
- **复制帧和复制 token 是真正的下限**。第一份 Notebook 里 token accuracy 0.811 等于 copy-token，方向准确率只有 0.283。
- **Tokenizer 先丢位置，动态救不回来**。16/16 码字用满，位置误差仍有 4.3 像素。
- **动作反事实看的是解码中心，不是 logits**。第二份 Notebook 里 `none` 的中心完全不动；additive / film 让 logits 动了，方块几乎没走。
- **Free rollout 走一步就停**。teacher-forced 的 0.80 不能代替连续生成。
- **24GB 目标是设计目标**：只有完整训练并提交实测数据后，才能标为「已验证」。

从第一份 Notebook 的 8 段 episode 到本节的完整训练，规模的变化让你亲眼看到视频世界模型的核心挑战：相邻帧太像、小物体被平均损失淹没、复合误差、动作进了网络却没进画面。这些挑战没有银弹，但你现在知道怎样用证据量化它们。

## 后续工作

本节用最小的离散 token 模型问了一件事：**按键能不能改变下一帧？** 后面的工作把同一问题做到了可玩的尺度，但判据没有变。

**IRIS**（Micheli, Alonso, Fleuret, 2023）把 VQ tokenizer 和自回归 Transformer 接到 Atari 上，用离散世界模型提高样本效率。结构与互动视频路线最近：先压成 token，再按动作预测下一组 token，最后在想象里行动。

**Genie**（Bruce 等人，2024）从没有动作标注的视频里学潜在动作，再按这个动作生成下一帧。用户按一个键，模型就真的吐出下一张可玩画面。它不再把 tokenizer、动态和解码器当成三个独立步骤，整台系统就是「按动作条件生成视频」。

**DIAMOND**（Alonso 等人，2024）把动态从离散 AR 换成扩散，直接在去噪世界里训练策略。第二份 Notebook 的 tiny denoiser 只验证了逐帧噪声接口；DIAMOND 证明这条路可以拿来做控制，而不是只做生成展示。

**Diffusion Forcing**（Chen 等人，2024）允许同一片段里每一帧拥有自己的噪声等级，从而用一个网络同时做下一帧预测和整段生成。第二份 Notebook 的 `add_independent_frame_noise` 就是这个接口的最小教学版。

这些方法的画面比 PixelWorld 复杂几个数量级，但验收清单仍是本节这四句：时间有没有接对、基线有没有钉死、换按键画面会不会变、自己的输出喂回去以后从哪一步开始坏。

## 参考文献

1. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. _NeurIPS 2017_. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937) —— VQ-VAE：离散 token、码本损失与直通估计器。
2. Vaswani, A., et al. (2017). Attention Is All You Need. _NeurIPS 2017_. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) —— Transformer：自注意力与位置编码。
3. Perez, E., et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. _AAAI 2018_. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871) —— 用条件信号缩放和平移特征；第二份 Notebook 的 `film` 是它的最小实现。
4. Micheli, V., Alonso, E., & Fleuret, F. (2023). Transformers are Sample-Efficient World Models. _ICLR 2023_. [arXiv:2209.00588](https://arxiv.org/abs/2209.00588) —— IRIS：离散 token + Transformer 的世界模型，与互动视频路线结构最接近。
5. Bruce, J., et al. (2024). Genie: Generative Interactive Environments. _ICML 2024_. [arXiv:2402.15391](https://arxiv.org/abs/2402.15391) —— 从视频学习可交互环境。
6. Chen, B., et al. (2024). Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion. _NeurIPS 2024_. [arXiv:2407.01392](https://arxiv.org/abs/2407.01392) —— 逐帧不同噪声等级；第二份 Notebook 只实现接口，不复现该系统。
7. Alonso, E., et al. (2024). Diffusion for World Modeling: Visual Details Matter in Atari. _NeurIPS 2024_. [arXiv:2405.12399](https://arxiv.org/abs/2405.12399) —— DIAMOND：用扩散模型当世界模型并在其中训练策略。
