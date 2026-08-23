# PA1-A · 动手：做出一台 Dreamer-lite

> **本节目标**：把路线 A 的 smoke 接成一条有证据的真实循环——从环境收集数据，在 RSSM 中想象，用 Actor-Critic 学习，再回到环境检查。不是复现 DreamerV3 的排行榜，而是亲眼看到「接口连通」与「策略收敛」之间的鸿沟，并诚实报告它从哪里开始裂开。

> **本节代码**：[PA1 模板](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/assignments/PA1-route-template.ipynb) · [A1](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/04_decision/A1-learn-a-latent-world.ipynb) · [A2](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/04_decision/A2-act-in-imagination.ipynb) · [`neural.py`](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/neural.py) · [`control.py`](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/control.py)

> **前置知识**：你已经跑过 [4.7 动手：决策与规划](/chapters/04-decision-and-planning/07-decision-and-planning)。A1 确认 RSSM 接口连通，A2 用位置模型证明 learned dynamics 能改善真实行动，再用一次 Actor 更新接通想象训练。PA1-A **不再重复那两份 smoke**，而是把它们扩展成一次完整训练。

---

A1 用 4 段 PixelWorld、15 次更新证明了：shape 对、梯度通、loss 能降。它也证明了另一件事——复制上一帧的像素 MSE 是 `0.0061`，15 步之后的重建仍是 `0.0144`。loss 下降不等于看见了世界。

A2 用 180 条单步位置转移证明了另一件事：`PositionDynamics` 的 MSE 从 `0.315` 降到 `0.0001` 之后，四个不在训练网格上的起点，规划成功率 `1.0`，随机是 `0.0`，平均剩余距离 `0` 对 `15.54`。从 `(5, 5)` 走到 `(12, 12)` 只要 14 步。然后它把位置换成 RSSM latent，做了一次 5 步想象和一次 Actor 更新。参数变了，策略没有被证明变好。

这些数字属于 4.7，不必在作业里再跑一遍当主结果。PA1-A 要回答的是：**把数据、更新次数和闭环都加大之后，latent 策略有没有在真实环境里变好？如果没有，失败稳定地出在哪一环？**

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/pa1a-dreamer-lite.png" alt="Dreamer-lite 完整训练循环" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">完整循环就这四步：收集像素、在 RSSM 里做梦、用梦里的回报更新 Actor-Critic、再回到环境。A1 / A2 只把接口接通；这里要让这个圈真的转起来。</div>
</div>

## 本次会得到什么

提交时，你应该能拿出：

- 一份按 episode 切分的数据卡，以及 `ReplayBuffer` 采出的连续序列 shape
- RSSM 的四条 loss：reconstruction、reward、continue、KL，横轴是更新步
- one-step 与 5 / 15 / 30 步预测，并且明确和「复制上一帧」比过
- 至少一个基线：随机策略，或 A2 那条位置模型 MPC（`planned 1.0` / `random 0.0`）
- **真实环境 return 对真实交互步数**，不是对训练步数
- 同一起点只换动作的反事实
- 一组 Actor 或 Planner 利用模型漏洞的失败，固定 seed 可重复
- GPU / CPU、peak 显存、墙钟时间和 checkpoint sha256

下面先用仓库里能在 CPU 上跑完的一档（20 段 × 16 步，40 次世界模型更新）把数字钉死。你的作业必须比 smoke 大，可以比这一档更大，但不能用 A1 的 4 段、15 步冒充完整训练。

## 为什么 PA1-A 是路线 A 的小整机

路线 A 的叙事是：VAE 的 latent 不够稳，RSSM 用 prior / posterior 双路替代；CMA-ES 没有梯度，Actor-Critic 用反向传播替代。A1 / A2 确认这些替代在接口上可行。PA1-A 要确认它们在训练上是否可行。

完整训练是一个闭环：

```text
收集 episode → 写入 ReplayBuffer
→ 训练 RSSM → 从 posterior 出发用 prior 想象
→ 更新 Actor-Critic → 用新 Actor 再收集
→ 重复
```

每一步的输出是下一步的输入。RSSM 不准，想象里的 Actor 会学错；Actor 学错，新数据会偏；数据一偏，RSSM 更不准。

教学版的数据量和计算量不够打破这个循环。目标是**让你看见这个循环存在**，并报告它从哪一环开始裂开。

## 第一步：写清缺口，不要重跑 smoke

先用一段话回答：A1 / A2 已经证明了什么，还缺什么。可以引用 4.7 的数字，不要把那两份 Notebook 再抄一遍。

```text
已证明
  RSSM 前向 shape = [B, T, 80]
  15 次更新 total 1.328 → 0.545，但重建仍差过复制上一帧
  位置模型 MPC：planned 1.0，random 0.0
  一次 Actor 更新后参数改变 = True

未证明
  latent 策略的真实 return 随交互步数上升
  多步 prior rollout 在 H=15 / 30 仍然可用
  换动作后想象轨迹会分叉
  闭环收集不会把 buffer 带进更窄的分布
```

**运行这一步，你会看到什么？** 一段不超过半页的对照表。如果这一段写成「我复现了 A1」，作业从这里就偏了。

**这就是衔接而不是重复**：4.7 的结论是作业的前提，不是作业的正文。

**一个值得做的实验**：把 A2 位置模型的训练起点收成左上角 3×3（45 条转移）。在我们跑过的这一次里，测试集四个起点仍然全部成功——残差 MLP 把「每步最多一格」写进了结构，短距离平移很容易外推。所以「缩小覆盖就一定掉点」不是自动成立的。你要找的失败，往往在 RSSM 和闭环里，不在这条可解释基线上。

## 第二步：环境依赖与数据卡

路线 A 需要 PyTorch。教学版 CPU 可跑；完整规模建议单张 24GB，并标明「24GB 目标」还是「CPU 缩减版」。

```bash
python -m pip install -r requirements-neural.txt
PYTHONPATH=src python -m unittest tests.test_control tests.test_neural -v
```

```python
import torch
print('PyTorch:', torch.__version__,
      'device:', 'cuda' if torch.cuda.is_available() else 'cpu')
```

仓库里的 PixelWorld 是 **16×16 RGB**，不是 64×64。`TinyWorldModel` 的解码器按 16×16 写死。先在这个分辨率上把闭环跑通；若要升到 64×64，必须改 encoder / decoder，并单独报显存。

数据必须按 episode 进 `ReplayBuffer`。它只在同一段经历内部切连续序列，观察比动作多一帧：

```python
from hwm.data import make_pixelworld_dataset, ReplayBuffer, split_by_episode

episodes = make_pixelworld_dataset(num_episodes=20, length=16, seed=0)
split = split_by_episode(episodes, train_ratio=0.70, val_ratio=0.15)
buffer = ReplayBuffer()
for episode in split['train']:
    buffer.add(episode)

batch = buffer.sample(batch_size=16, sequence_length=8, seed=0)
print(split['train'] and len(split['train']), len(split['val']), len(split['test']))
print(batch[0]['observations'].shape, batch[0]['actions'].shape)
```

**运行这一步，你会看到什么？** `seed=0`、20 段 × 16 步：

```
split: train 14 / val 3 / test 3
sample observations: (9, 16, 16, 3)
sample actions:      (8,)
同一 batch 里出现 11 个不同 episode_id
第 t 个动作对应 observations[t] → observations[t+1]
20 段随机策略的奖励全程 -0.01，goal hits = 0
```

随机策略从左上 8×8 出发、只走 16 步，几乎碰不到 `(12, 12)`。所以 reward head 一开始看见的几乎全是 `-0.01`。这不是实现 bug，是数据分布。

数据卡至少写：

```text
- 生成器、seed、episode 数、每段长度
- observation / action / reward / done 的 shape 与时间对齐
- train / val / test 按 episode 切，还是按 seed 切
- buffer 采样是连续序列，不是打散的单步
- 随机策略有没有碰到目标
```

**这就是 ReplayBuffer 要保住的东西**：时间连续性。把 transition 打散再训练，RSSM 的 GRU 就在背不存在的历史。

**一个值得做的实验**：用错误切法——把每段的前 8 步放 train、后 8 步放 test。多步指标会虚高，因为模型已经在同一条轨迹的前半段里见过方块怎么走。作业里必须写你没有这样做。

## 第三步：把 RSSM 训练到能画曲线

教学版的损失与 `world_model_loss` 一致：

$$
\mathcal{L}
= \|o_{t+1}-\hat o_{t+1}\|_2^2
+ \|r_t-\hat r_t\|_2^2
+ \mathrm{BCE}(c_t, \hat c_t)
+ 0.1 \cdot \max\bigl(\mathrm{KL}(q\|p),\, 1\bigr)
$$

训练时走 posterior，部署时走 prior：

$$
h_t = f(h_{t-1}, z_{t-1}, a_{t-1}),\quad
z_t \sim q(z_t \mid h_t, o_t),\quad
\hat z_t \sim p(\hat z_t \mid h_t)
$$

KL 有 1 nat 的 free bits。差得不够大，先不罚。

```python
from hwm.neural import TinyWorldModel, world_model_loss, batch_from_episodes
import torch.nn.functional as F

observations, actions, rewards, dones = batch_from_episodes(
    episodes, sequence_length=16
)
model = TinyWorldModel()          # 约 2.93×10^5 参数
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
history = []
for _ in range(40):
    optimizer.zero_grad()
    loss, metrics, outputs = world_model_loss(
        model, observations, actions, rewards, dones
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    optimizer.step()
    history.append({k: float(v) for k, v in metrics.items()})

targets = observations[:, 1:].float() / 255.0
copy_last = observations[:, :-1].float() / 255.0
print(history[0]['total'], history[-1]['total'])
print('copy-last', float(F.mse_loss(copy_last, targets)))
print('recon', history[-1]['reconstruction'])
```

CPU 上 20 段 × 16 步、`seed=0`、40 次更新：

```
obs        (20, 17, 16, 16, 3)
actions    (20, 16)

step  0: total 1.379  recon 0.249  reward 0.039  continue 0.717  kl 3.741
step 15: total 0.402  recon 0.012  reward 0.003  continue 0.286  kl 0.660
step 39: total 0.338  recon 0.011  reward 0.000  continue 0.227  kl 0.362

copy-last MSE: 0.0057
40 步重建 MSE: 0.0107
```

比 A1 的 4 段 × 8 步更大，曲线仍然是同一句话：total 在降，重建还是赢不过复制上一帧。reward 头很快降到接近 0，因为它几乎只见过 `-0.01`。continue 从抛硬币的 `0.72` 降到 `0.23`，是因为每段最后一步 `done=True`，模型在学「16 步该结束了」，不是在学「碰到绿块该结束」。

24GB 目标配方（你要自己改网络才能用，不是当前 `TinyWorldModel` 的默认值）：

| 项目                       | 教学默认  |       24GB 目标 |
| -------------------------- | --------- | --------------: |
| 观察                       | 16×16 RGB |       64×64 RGB |
| batch                      | 整集或 16 |              16 |
| sequence length            | 16        |              32 |
| deterministic / stochastic | 64 / 16   |        256 / 32 |
| imagination horizon        | 15        |              15 |
| mixed precision            | 关        |            可选 |
| peak reserved              | CPU       | 目标不超过 22GB |

**运行这一步，你会看到什么？** 四条曲线，以及「重建对复制上一帧」的同一张图。如果只交 total，不交 copy-last，这一步不算过。

**这就是 A1 在更大一档上的重演**：数据加了五倍，更新加了一倍多，像素损失照样奖励偷懒。

**一个值得做的实验**：把 `num_episodes` 从 20 扫到 80，看重建第一次低于 `0.0057` 发生在哪一档。4.7 把这件事列为扩展；PA1-A 允许你把它做成主证据之一，但不能只交这一条。

## 第四步：想象训练，而不是一次更新

从真实 posterior 的最后一步出发，之后只走 prior。Actor 约 5509 参数，Critic 约 5249。教学版 `imagine` 冻结世界模型，用 REINFORCE，不把梯度传回 RSSM。

$$
G_t^{\lambda}
= \hat r_t
+ \gamma\, \hat c_t \bigl[(1-\lambda)\hat v_{t+1} + \lambda G_{t+1}^{\lambda}\bigr]
$$

代码里 \(\gamma=0.99\)，\(\lambda=0.95\)，从后往前折，最后用 bootstrap value 收尾。

```python
from hwm.neural import Actor, Critic, RSSMState, imagine, lambda_returns

with torch.no_grad():
    outputs = model(observations, actions, sample=False)
start = RSSMState(
    outputs['posterior'].deterministic[:, -1],
    outputs['posterior'].stochastic[:, -1],
    outputs['posterior'].mean[:, -1],
    outputs['posterior'].std[:, -1],
)
actor, critic = Actor(), Critic()
actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-3)

for _ in range(50):          # smoke 是 1 次；这里至少几十次
    imagined = imagine(model, actor, start, horizon=15)
    values = critic(imagined['features'])
    returns = lambda_returns(
        imagined['rewards'].detach(),
        imagined['continues'].detach(),
        values.detach(),
        values[:, -1].detach(),
    )
    actor_loss = -(imagined['log_probs'] * returns.detach()).mean()
    critic_loss = F.mse_loss(values, returns.detach())
    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
    critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
```

40 次世界模型更新之后、Actor 仍是随机初始化时，一次 15 步想象是：

```
imagined features: (20, 15, 80)
actions[0, :8]:    [2, 4, 4, 1, 4, 4, 4, 0]
reward mean / std: -0.006 / 0.020
continue mean:     0.889
TD-λ[0, :5]:       [-0.003, 0.021, 0.050, 0.032, 0.038]
```

动作没有朝向目标。预测奖励在 0 附近晃。continue 接近 0.9——梦还没学会结束。这是「梦境能展开」的证据，不是「梦里已经会走路」的证据。

同一起点、只换动作、走 prior 8 步（`sample=False`）：

```
feature RMSE  一直向右 vs 一直向左:  0.062
feature RMSE  一直向右 vs 一直停留:  0.084
```

分叉存在，但很小。像素空间里向右和向左是相反的位移；80 维特征里，它们只隔了 0.06。反事实「有响应」不等于「响应够用」。

**运行这一步，你会看到什么？** Actor / Critic 的曲线，以及真实环境 return。若 Actor loss 下降而真实 return 不动，先怀疑模型漏洞，不要先写「还没训够」。

**这就是想象训练的风险**：目标是 \(\hat r\) 和 \(\hat v\)，不是环境里的 \(r\)。头错了，策略会朝着错误的山爬。

**一个值得做的实验**：把 horizon 从 5 提到 15 再提到 30，画出预测奖励的均值和方差。复合误差在隐空间里一样会发生——4.6 那张 free-running 图的对应物。

## 第五步：回到环境，把圈接上

用更新后的 Actor 采新 episode，写入同一个 buffer，再训 RSSM。这一步才是闭环。横轴必须是**真实交互步数**。

对照至少保留两条：

```text
随机策略          真实 return 的下限
位置模型 MPC      可解释动态的上限（A2：planned 1.0 / random 0.0）
RSSM + Actor      你要交的曲线
```

`scripts/run_a2_reference.py --seed 0 --updates 100` 给出位置模型那一档：

```
training_transitions:     180
planned_success_rate:     1.0
random_success_rate:      0.0
planned_final_distance:   0.0
random_final_distance:    15.54
(5,5) → (12,12) 14 步
```

RSSM Actor 若在同样的 `(5, 5)`、24 步预算里到不了，不能写成「PixelWorld 太难」——位置模型已经到了。只能写成 latent 策略还没学会用动态。

**运行这一步，你会看到什么？** 三条 return 曲线画在同一张图上。新 episode 可能比随机好，也可能更差：Actor 在梦里找到的捷径，真实环境里不存在。

**这就是闭环**：数据分布跟着策略走。只在固定的 20 段随机数据上过拟合 Actor，不算完成第五步。

**一个值得做的实验**：冻结 RSSM，只更新 Actor 200 次，再解冻 RSSM 用新数据更新。看真实 return 是在哪一段开始掉。掉在想象阶段，是头或策略的问题；掉在重新训练 RSSM 之后，是数据偏移。

## 第六步：八项必交证据

缺一不可。下面每项都写了「怎样才算交了」。

**1. 数据卡与环境 wrapper**

字段、shape、seed、切分、时间对齐。wrapper 与 A1 / A2 一致：`[B, T+1, H, W, 3]` 对 `[B, T]` 动作。

**2. 世界模型四条 loss**

reconstruction / reward / continue / KL。标注更新次数。上面 40 步那组可以当 CPU 缩减版的对照，不能当「已经训完」。

**3. One-step 与多步**

one-step 重建必须和复制上一帧写在一起。再报 5 / 15 / 30 步 prior 的 MSE 或特征距离。多步应随 horizon 变差；若不变，先检查是不是一直在喂真实下一帧。

**4. 基线**

随机策略，或位置模型 MPC，或两者都有。位置模型不是「旁门左道」，它是这条路线上唯一已经证明能改善真实行动的东西。

**5. 真实环境 return 对真实交互步数**

最关键的一条。横轴不是 optimizer step。曲线不上升，就写不上升，并分析是数据、模型还是 Planner。

**6. 反事实**

固定起点，替换动作序列。上面 right vs left 的特征 RMSE `0.062` 是下限例子：有差异，但很小。你需要说明差异是否大到能改变规划。

**7. 一组利用模型漏洞的失败**

固定 seed。候选解释包括：RSSM 预测不准、reward head 只见过 `-0.01`、Actor 过度信任 prior、imagination 走进 OOD、buffer 被坏策略污染。至少两种解释，并排除一种。

**8. 资源清单与 checkpoint 哈希**

设备、CUDA 或 `cpu`、peak allocated / reserved、墙钟时间、权重 sha256。24GB 目标只有在你提交这些数字之后，才能在状态页改成「已验证」。

## 第七步：阶段二——DMC Cartpole（选做）

DeepMind Control Suite 的 Cartpole，固定小配置。本地装不上 MuJoCo，就标「未运行」。**不能用别人的曲线代替。**

## 替代选题：Mini-MuZero

若更喜欢棋类搜索，可用四子棋做 Mini-MuZero，替代 Dreamer-lite。必须包含 Representation、Dynamics、Prediction、MCTS，以及搜索改进前后的胜率。不能两个项目各做一半。

## 运行与产物

```bash
python -m pip install -r requirements-neural.txt
PYTHONPATH=src python -m unittest tests.test_control tests.test_neural -v
python scripts/run_a2_reference.py --output runs/a2-reference
```

第二条命令是基线，不是作业本体。本体在你的 PA1 Notebook 里：收集、训练、想象、再收集。

跑完后，你应该有：

- RSSM / Actor / Critic 曲线
- 真实环境 return 对交互步数
- 5 / 15 / 30 步预测
- 反事实
- 失败案例
- 资源清单和 checkpoint 哈希

## Smoke 与 PA1-A 的区别

| 项目 | A1 / A2 smoke（4.7 已完成）           | PA1-A                                             |
| ---- | ------------------------------------- | ------------------------------------------------- |
| 数据 | 4 段 × 8 步；位置模型 180 条          | ≥20 段 PixelWorld，按 episode 进 buffer；选做 DMC |
| 训练 | 15–100 步；Actor 更新 1 次            | 直到形成可讨论的曲线，并再收集至少一轮            |
| 目的 | 控制增益、接口、梯度                  | latent policy 的真实 return 与样本效率            |
| 资源 | CPU                                   | 单张 24GB，或明确标注的 CPU 缩减版                |
| 结论 | 可解释动态能帮助行动；RSSM 接口可运行 | 闭环有没有转起来，失败在哪一环                    |

## 评分

| 项目          | 分数 | 检查重点                                |
| ------------- | ---: | --------------------------------------- |
| 问题与接口    |   10 | 写清 A1 / A2 已证明什么，本作业补哪一环 |
| 数据与 buffer |   15 | episode 边界、时间对齐、连续序列采样    |
| 世界模型      |   15 | 四条 loss，重建对复制上一帧             |
| 想象与策略    |   15 | horizon、TD-λ、真实 return 对交互步数   |
| 基线与反事实  |   15 | 至少一条强基线；换动作预测会变          |
| 失败诊断      |   15 | 稳定、可复现、至少两种解释              |
| 资源与复现    |   15 | seed、哈希、设备、墙钟时间              |

## 不接受的结论

- 「loss 下降了，所以 Dreamer-lite 训成了。」——40 步后重建 `0.0107` 仍高于复制上一帧 `0.0057`。
- 「Actor 参数变了，所以策略变好了。」——那是 A2 smoke 的结论，不是 PA1-A 的结论。
- 「真实 return 没动，是因为 PixelWorld 不可控。」——位置模型已经 `1.0` 对 `0.0`。
- 「用 A1 的 4 段曲线代替完整训练。」
- 「DMC 没跑，贴一张论文图。」
- 「下一步上 Diffusion + Transformer + MCTS。」——只允许一项与失败对应的最小改动。

## 已知简化与坑

- **PixelWorld 仍然简单**。16×16、5 个动作、红方块。RSSM 在这里学到的世界非常有限。
- **随机数据几乎没有 +1**。reward head 很容易变成「永远报 `-0.01`」的常数器。
- **复制上一帧仍是更强的像素基线**。不要把 reconstruction 下降写成「学会了动态」。
- **教学版没有 world-model gradient**。`imagine` 里世界模型冻结，Actor 是 REINFORCE。完整 Dreamer 会让梯度穿过 dynamics。
- **prior 不是 posterior**。A1 里 15 步后两者的 mean 距离仍有 `0.72`。梦从第一步就会偏。
- **24GB 是设计目标，不是实测结果**。只有提交设备、显存、时间和哈希以后，才能改状态页。
- **位置模型太强，不能当「失败环境」**。它证明的是 Planner + 可解释动态，不是 RSSM 已经够用。

## 扩展练习

1. **数据量**：episode 从 20 提到 80，看真实 return 何时离开随机基线。
2. **imagination horizon**：5 / 15 / 30，观察 Actor loss 和真实 return 是否同向变化。
3. **world-model gradient**：让 Actor 的梯度穿过 `imagine_step`，比较样本效率。
4. **把梦解码出来**：对 prior 特征调用 `model.decode`，看第 2、5、15 步还能不能认出红块。

## 研究问题

完成训练后，回答：**模型最稳定的失败来自哪一项？**

```text
状态没有保存重要信息
动态在长 horizon 漂移
reward / value 误导 Actor
Planner 进入 OOD 区域
真实数据没有覆盖策略访问的状态
```

至少提出两种解释，再选一项最小改动带到第 9 章或 PA2。

## 本节小结

- **PA1-A 是路线 A 的小整机**：从 smoke 扩到闭环，看「接口连通」和「策略收敛」之间的距离。
- **八项证据缺一不可**：数据卡、四条 loss、多步、基线、真实 return、反事实、失败、资源清单。
- **位置模型已经能到终点**：RSSM Actor 若到不了，问题在 latent 闭环，不在环境。
- **Actor 会利用模型漏洞**：梦里的捷径在真实环境里可能不存在。
- **24GB 是设计目标**：只有提交实测数据后才能标「已验证」。

从 A1 的 4 段到至少 20 段，从一次 Actor 更新到一条对交互步数的曲线——规模变化不是量变，是质变。你会看见 Dreamer 的承诺和代价：在想象中训练是可能的，但想象不是现实。

## 后续工作

PA1-A 用连续高斯 \(z_t\) 和 REINFORCE，已经够你看见闭环怎么裂开。后面三条常见的下一步，各自对应一种你可能刚刚写下来的失败：

离散 latent（DreamerV2 / V3）对付「向左和向右被平均成停在中间」。CEM / PlaNet 对付「动作空间不再能枚举」。MuZero 干脆去掉重建头，只保留决策需要的预测——若你的 copy-last 一直压着 reconstruction，这条路比再加一层解码器更值得试。

不要一次全做。把第七步里那一项最小改动带走就够了。

## 参考文献

1. Hafner, D., et al. (2019). Learning Latent Dynamics for Planning from Pixels. _ICML 2019_. [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) —— PlaNet：RSSM + CEM，A2 beam search 的连续动作亲戚。
2. Hafner, D., et al. (2019). Dream to Control: Learning Behaviors by Latent Imagination. _ICLR 2020_. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) —— DreamerV1：本节 RSSM、imagination 与 Actor-Critic 的原文。
3. Hafner, D., et al. (2021). Mastering Atari with Discrete World Models. _ICLR 2021_. [arXiv:2010.02193](https://arxiv.org/abs/2010.02193) —— DreamerV2：离散 latent；若连续高斯把左右平均掉，从这里改。
4. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. _arXiv:2301.04104_. [链接](https://arxiv.org/abs/2301.04104) —— DreamerV3：一套超参打通 150+ 任务，是 24GB 目标对照的工程标杆，不是本作业的及格线。
5. Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. _Nature_, 588, 604–609. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) —— MuZero：不重建像素；也是 Mini-MuZero 选题的原文。
6. Sutton, R. S. (1988). Learning to Predict by the Methods of Temporal Differences. _Machine Learning_, 3, 9–44. [链接](https://doi.org/10.1007/BF00115009) —— TD-λ；`lambda_returns` 的 \(\gamma=0.99,\lambda=0.95\) 从这里来。
7. Tassa, Y., et al. (2018). DeepMind Control Suite. _arXiv:1801.00330_. [链接](https://arxiv.org/abs/1801.00330) —— 选做 DMC Cartpole 的环境说明。
