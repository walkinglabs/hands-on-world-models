# 4.7　动手：Dreamer 的简化实现

> **本节目标**：跑通一条从「像素观测」到「在想象中规划行动」的完整链路。第一份 Notebook 把 CNN Encoder、RSSM 和预测 head 接起来，确认 loss 能下降；第二份先在一个可解释的位置模型上证明 learned dynamics 真的能帮助行动，再把位置换成 RSSM latent，让 Actor-Critic 在隐空间里走完一次更新。

> **本节代码**：[学出一个潜在世界](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/04_decision/learn-a-latent-world.ipynb) · [在想象中行动](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/04_decision/act-in-imagination.ipynb)

> **前置知识**：你已经读过 4.1–4.5，知道 RSSM 的 prior/posterior、Dreamer 的 imagination 循环、Actor-Critic 的 TD-λ。最好刚跑完 [4.6 动手：World Models 的复现](/chapters/04-decision-and-planning/06-reproduce-world-models)。这一节把它们真跑一遍。

---

如果你刚做完 4.6，你大概还记得那种感觉：VAE 把赛车压成 32 个数，MDN-RNN 在梦里开车，867 个参数的线性控制器完全在想象中学会了转弯。画面模糊，但车确实在动。

然后三个问题会一起冒出来。

第一，VAE 的 latent 空间不够稳——β 太小会塌，太大会丢信息，调参像走钢丝。第二，MDN-RNN 的复合误差滚得太快——100 步后画面变成噪声，控制器只能在短期梦境里进化。第三，CMA-ES 没有梯度——867 个参数靠种群统计量慢慢摸索，贵而且慢。

你当时大概和我一样，会问：能不能看得更清楚一点？能不能让策略直接沿着梦境反传，而不是蒙着眼进化？

这一节的两份 Notebook，就是 Dreamer 对这三个问题的回答。规模打折，原理不打折。跑完之后，你会对「在想象中训练」这句话有另一层理解。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/pixelworld.png" alt="PixelWorld 小世界" style="max-width:min(900px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">这就是我们要让模型学会的「世界」：16×16 的黑底小图，红色方块是自己，绿色方块是目标。模型从未被告知「红色是智能体、绿色是终点」，它要从像素流里自己发现「向右走，红块会右移」。</div>
</div>

Dreamer 的完整训练需要 GPU、需要大量数据、需要跑几个小时。教学版不复现排行榜。它只在 CPU 上用 4 段 8 步的小世界，把完整数据流跑通——shape 对、梯度通、loss 降、接口连。

## 本次会得到什么

运行结束后，你会得到：

- 一张 16×16 的 PixelWorld，以及红方块朝绿目标走的几帧
- 第一份 Notebook 一次前向的 shape：观测 `[4, 9, 16, 16, 3]`，特征 `[4, 8, 80]`
- 15 次更新的 total / reconstruction / KL 曲线
- 原图、复制上一帧、RSSM 重建的并排对比
- 位置模型的损失曲线，以及同一起点上规划 vs 随机的轨迹
- 一次 5 步 imagination：动作、预测奖励、TD-λ
- 一次 Actor / Critic 更新后，参数确实改变的检查

## 第一步：安装环境依赖

决策与规划路线（第 4 章）第一次使用 PyTorch。共同基础只需要 NumPy；选了这条路线，才装神经依赖：

```bash
python -m pip install -r requirements-neural.txt
```

安装完成后，确认 PyTorch 可用：

```python
import torch
print('PyTorch:', torch.__version__, 'device:',
      'cuda' if torch.cuda.is_available() else 'cpu')
```

教学版在 CPU 上运行，不需要 GPU。4.8 的完整训练建议用单张 24GB。

两份 Notebook 在：

```text
notebooks/04_decision/learn-a-latent-world.ipynb
notebooks/04_decision/act-in-imagination.ipynb
```

也可以先跑测试：

```bash
PYTHONPATH=src python -m unittest tests.test_control tests.test_neural -v
```

## 第二步：看清这个小世界

PixelWorld 是 16×16 的黑底。红色 3×3 是智能体，绿色 3×3 是目标，固定在 `(12, 12)`。动作只有五个：

```python
MOVE = {
    0: (0,  0),   # stay
    1: (0, -1),   # left
    2: (0,  1),   # right
    3: (-1, 0),   # up
    4: (1,  0),   # down
}
```

每走一步奖励 `-0.01`，踩到目标给 `+1.0` 并结束。出界会被夹回来。这不是 Atari，也不是 CarRacing。它小到你可以在像素里数出方块挪了几格——后面第二份 Notebook 正是靠这一点，先做一台你看得懂的世界模型。

第一份 Notebook 用随机策略采 4 段、每段 8 步。第 `t` 个动作对应 `observations[t] → observations[t+1]`，所以观察比动作多一帧：

```python
from hwm.data import make_pixelworld_dataset
from hwm.neural import batch_from_episodes

episodes = make_pixelworld_dataset(num_episodes=4, length=8, seed=0)
observations, actions, rewards, dones = batch_from_episodes(
    episodes, sequence_length=8
)
print('observations:', tuple(observations.shape))
print('actions:     ', tuple(actions.shape))
print('rewards:     ', tuple(rewards.shape))
```

**运行这一步，你会看到什么？** Notebook 会打印这一批数据的形状：

```
observations: (4, 9, 16, 16, 3) torch.uint8
actions:      (4, 8) torch.int64
rewards:      (4, 8)
第一段动作:   [0, 2, 4, 3, 3, 2, 3, 2]
第一段奖励:   全程 -0.01
```

4 段 episode，每段 8 步，观察却是 9 帧——最后一帧是第 8 个动作走完之后的世界。第一段 8 步里没碰到目标，所以奖励一直是 `-0.01`。

如果时间维对不上，后面所有 head 都会看错帧。世界模型里，时间对齐比模型结构更先出错。

## 第三步：一次前向同时长出什么

第一份 Notebook 的核心是 **RSSM**（Recurrent State-Space Model）。和 4.6 的 MDN-RNN 不同，它同时维护两段状态：确定性隐状态 \(h_t\) 带着长期记忆，随机 latent \(z_t\) 带着眼前的不确定性。

$$
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
$$

$$
\text{posterior: } z_t \sim q(z_t \mid h_t, o_t), \qquad
\text{prior: } \hat{z}_t \sim p(\hat{z}_t \mid h_t)
$$

训练时能看见真实下一帧，用 posterior；部署时没有未来图片，只能走 prior。训练目标是让 prior 尽量靠近 posterior：

$$
\mathcal{L}
= \underbrace{\|o_{t+1}-\hat o_{t+1}\|_2^2}_{\text{重建}}
+ \underbrace{\|r_t-\hat r_t\|_2^2}_{\text{奖励}}
+ \underbrace{\mathrm{BCE}(c_t, \hat c_t)}_{\text{是否结束}}
+ 0.1 \cdot \max\bigl(\mathrm{KL}(q\|p),\, 1\bigr)
$$

重建项让模型看见世界，奖励项让模型知道好坏，continue 项让模型知道故事有没有讲完，KL 项让 prior 在没有真实观测时也能给出靠谱的采样。KL 有 1 nat 的 free bits——差得不够大，先不罚。

```python
from hwm.neural import TinyWorldModel, world_model_loss

model = TinyWorldModel()
loss, metrics, outputs = world_model_loss(
    model, observations, actions, rewards, dones
)
print('feature:', tuple(outputs['feature'].shape), '[B,T,deter+stoch]')
print('reconstruction:', tuple(outputs['reconstruction'].shape))
print('losses:', {k: round(float(v), 4) for k, v in metrics.items()})
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/rssm-dataflow.png" alt="RSSM 数据流" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">一次前向的真实 shape。像素经 CNN 压成 64 维，RSSM 拆成 64 维记忆和 16 维随机 latent，拼起来是 80 维特征。M 和 C 之后读的，就是这 80 个数。</div>
</div>

**运行这一步，你会看到什么？** Notebook 会打印一次前向的 shape 和四项损失：

```
feature: (4, 8, 80) [B,T,deter+stoch]
reconstruction: (4, 8, 16, 16, 3)
reward: (4, 8)
losses: {'total': 1.319, 'reconstruction': 0.249, 'reward': 0.037, 'continue': 0.701, 'kl': 3.323}
```

feature 的最后一维是 80 = 64 + 16。这就是 RSSM 的核心设计：记忆和不确定性分开存放，用的时候再拼回去。

`continue` 接近 \(0.69\)，差不多是还没训练的抛硬币。`kl` 大于 1，说明 prior 和 posterior 此刻差得很远——模型还不会在没有图片的时候猜下一帧。

这不是一次完整的 Dreamer 训练。4 段 episode 太少，模型会迅速过拟合。第一份 Notebook 的目标是确认接口连通：shape 对、各个 head 的数值合理、梯度能流。

## 第四步：15 次更新

教学版故意拿同一小批数据反复更新。若 loss 连下降都做不到，先修代码和 shape，不急着收集更多数据。

```python
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
losses = []
for _ in range(15):
    optimizer.zero_grad()
    loss, metrics, outputs = world_model_loss(
        model, observations, actions, rewards, dones
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    optimizer.step()
    losses.append(float(loss.detach()))
print('初始/最终 loss:', round(losses[0], 4), round(losses[-1], 4))
```

**运行这一步，你会看到什么？**

```
total:          1.328 → 0.545
reconstruction: 0.249 → 0.014
KL:             3.312 → 0.561
最后一次梯度 norm: 0.32
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/world-model-loss-curve.png" alt="世界模型损失曲线" style="max-width:min(720px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">15 次更新。total 和 reconstruction 明显下降，KL 从 3.3 掉到 0.56。第二条更新 total 还会先抬一下——优化器在找路，不是曲线画错了。</div>
</div>

数字在往下走。但像素损失往下走，不一定等于模型看见了世界。相邻帧本来就很像，有一个不读动作的基线经常更强：直接复制上一帧。

```python
targets = observations[:, 1:].float() / 255.0
copy_last = observations[:, :-1].float() / 255.0
print('复制上一帧 MSE:', round(float(F.mse_loss(copy_last, targets)), 4))
print('15 步重建 MSE: ', round(float(metrics['reconstruction']), 4))
```

**运行这一步，你会看到什么？**

```
复制上一帧 MSE: 0.0061
15 步重建 MSE:  0.0144
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/rssm-reconstruction.png" alt="原图、复制上一帧与 RSSM 重建" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">同一段 episode 的前 4 个下一帧。上排是真值，中排是复制上一帧，下排是 15 次更新后的解码。红方块在重建里糊成一团；复制上一帧几乎看不出差别——相邻帧本就很像。</div>
</div>

你会发现，训练后的解码器仍然差过「把上一帧原样拿来」。loss 下降了，动态没学会。

**这就是第一份 Notebook 真正要你看见的事**：loss 下降只说明这 32 个转移上的优化路径连通。它没有证明多步世界是对的，也没有证明模型能帮助行动。

**一个值得做的实验**：把 `num_episodes` 从 4 提到 20，观察 reconstruction 能不能低于 0.006，以及 KL 会怎么变。数据变多之后，KL 通常会先升——posterior 看见了更多样的下一帧，prior 一时跟不上。这正是 RSSM 要最小化的东西：让 prior 尽量靠近 posterior，这样部署时（没有真实观测）采样才靠谱。

## 第五步：prior 还不是 posterior

训练时 posterior 能看见真实下一帧；到了梦里，没有未来图片，只能走 prior。两者现在不该重合。

```python
prior_mean = outputs['prior'].mean[0, -1]
posterior_mean = outputs['posterior'].mean[0, -1]
distance = torch.linalg.vector_norm(prior_mean - posterior_mean)
print('最后一步 prior/posterior mean 距离:', round(float(distance), 4))
print('KL:', round(float(metrics['kl']), 4))
```

**运行这一步，你会看到什么？**

```
最后一步 prior/posterior mean 距离: 0.721
KL: 0.561
```

第二份 Notebook 会从这段 posterior 的最后一步出发，之后只调用 prior。如果 prior 学得不像 posterior，梦境从第一步就会偏。

Posterior 用真实图片修正状态，prior 学习在没有未来图片时靠动作预测；想象阶段能用的只有后者。

## 第六步：为什么先做位置模型，再做 latent 模型

第二份 Notebook 有一个刻意的迂回：它不直接用 RSSM 做规划，而是先从图片里量出方块的 \((x, y)\)，学一个 `位置 + 动作 → 下一位置` 的小模型，在这个你看得懂的模型里做 beam search，再回到真实环境验收。

为什么要绕这个弯？因为你需要先确认 **learned dynamics 真的能帮助行动**，而不是把「loss 下降」当成成功的证据。位置模型的预测是二维坐标，你能一眼看出「预测的下一位置离真实下一位置差了多少」。如果这个可解释模型都不能改善行动，那 RSSM 更不可能——问题出在 Planner 或数据，而不是 latent 表示。

现在把 RSSM 先放下。从图片里用颜色阈值量出红方块左上角，训练一个两层残差 MLP：

$$
\hat p_{t+1}
= \mathrm{clip}\Bigl(
    p_t + \tanh\bigl(\mathrm{MLP}([p_t/13;\; \mathrm{one\_hot}(a_t)])\bigr),
    0,\; 13
\Bigr)
$$

残差结构写进了「每步最多走一格」。训练数据是 6×6 个起点乘 5 个动作，一共 180 条单步转移。

```python
from hwm.control import PositionDynamics, fit_position_dynamics
from hwm.data import MovingSquareWorld, pixelworld_transition_arrays

world = MovingSquareWorld()
episodes = []
for row in (0, 3, 6, 9, 12, 13):
    for col in (0, 3, 6, 9, 12, 13):
        for action in range(5):
            episode, _ = world.generate([action], start=(row, col))
            episodes.append(episode)

positions, actions, next_positions = pixelworld_transition_arrays(episodes)
model = PositionDynamics(hidden_size=48)
losses = fit_position_dynamics(
    model, positions, actions, next_positions, updates=100
)
print('position loss:', round(losses[0], 4), '→', round(losses[-1], 4))
```

**运行这一步，你会看到什么？**

```
position loss: 0.3341 → 0.0001
```

100 步之后，MSE 掉了三个数量级。位置是二维坐标，预测点和真实点叠不叠，你自己能看出来——这是 RSSM 的 80 维特征给不了的。

## 第七步：在模型里试动作，再回到真实环境

Planner 是 beam search：往前看 4 步，保留离目标最近的若干条，只在真实 PixelWorld 执行第一步，再重新观察。这就是 MPC——1.1 里那句「只执行第一步，然后重新观察真实世界」。

```python
from hwm.control import evaluate_controllers, run_pixelworld_controller

metrics = evaluate_controllers(
    model,
    starts=[(1, 1), (2, 8), (8, 2), (5, 5)],
    max_steps=24,
    random_seeds=5,
)
example = run_pixelworld_controller(model, (5, 5), max_steps=24)
print('planned_success_rate:', round(metrics['planned_success_rate'], 3))
print('random_success_rate: ', round(metrics['random_success_rate'], 3))
print('一条真实路线:', example['positions'])
```

**运行这一步，你会看到什么？**

```
planned_success_rate:   1.000
random_success_rate:    0.000
planned_final_distance: 0.000
random_final_distance:  14.15

一条真实路线 (5,5) → (12,12):
(5,5) → (5,6) → (5,7) → (6,7) → (6,8) → (6,9)
      → (7,9) → (7,10) → (7,11) → (8,11) → (8,12)
      → (9,12) → (10,12) → (11,12) → (12,12)
```

四个测试起点都不在那 6×6 训练网格上。规划器用学到的动态走了 14 步到达目标；同样 24 步预算里，随机策略一次都没碰到。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/planned-vs-random.png" alt="规划轨迹对比随机" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">同一起点 (5,5)。左：学到的动态带着规划器走进绿色目标；右：随机策略在左上角打转。这不是 loss 曲线能告诉你的。</div>
</div>

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/position-planning.png" alt="位置模型训练、预测与规划" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">位置模型的三张证据：损失降下来、单步预测贴着真实点、beam search 在模型里走出一条朝向目标的路径。</div>
</div>

如果 planned 明显高于 random，说明 learned dynamics 的预测确实能被 Planner 用来改善真实行动——**这不是 loss 下降能告诉你的**。

**一个值得做的实验**：把训练起点改成只覆盖左上角 3×3，再在右下的 `(8, 2)`、`(5, 5)` 上评估。训练集上的 MSE 仍然可以很低，成功率通常会掉。1.1 里你见过「没见过的 (状态, 动作)」；换成神经网络，泛化边界一样在。

## 第八步：把位置换成 RSSM latent

上面的 Planner 用的是从图片量出来的二维坐标。现在换成 CNN 和 RSSM 从像素里学 latent。这里仍是 smoke：10 次更新只证明训练接口连通，不能覆盖前面那条真实控制证据。

```python
from hwm.neural import Actor, Critic, RSSMState, imagine, lambda_returns

with torch.no_grad():
    outputs = model(observations, actions, sample=False)
posterior = outputs['posterior']
start = RSSMState(
    posterior.deterministic[:, -1].detach(),
    posterior.stochastic[:, -1].detach(),
    posterior.mean[:, -1].detach(),
    posterior.std[:, -1].detach(),
)

actor = Actor()
critic = Critic()
imagined = imagine(model, actor, start, horizon=5)
values = critic(imagined['features'].detach())
returns = lambda_returns(
    imagined['rewards'].detach(),
    imagined['continues'].detach(),
    values.detach(),
    values[:, -1].detach(),
)
print('imagined features:', tuple(imagined['features'].shape))
print('actions:', imagined['actions'][0].tolist())
print('TD-lambda:', [round(x, 3) for x in returns[0].tolist()])
```

从真实 posterior 出发以后，想象阶段不再读取未来图片。Actor 给出动作，RSSM prior 更新 latent，reward 与 continue heads 给出训练信号：

```python
for _ in range(horizon):
    action = actor(state.feature).sample()          # 不再读图片
    state = model.rssm.imagine_step(state, action)  # 只走 prior
    reward = model.reward_head(state.feature)
```

TD-λ 从后往前折，\(\gamma=0.99\)，\(\lambda=0.95\)：

$$
G_t^{\lambda}
= \hat r_t
+ \gamma\, \hat c_t \bigl[(1-\lambda)\hat v_{t+1} + \lambda G_{t+1}^{\lambda}\bigr]
$$

**运行这一步，你会看到什么？**

```
RSSM smoke loss: 0.672
posterior feature: (4, 80)
imagined features: (4, 5, 80)
actions: [3, 2, 3, 1, 2]
TD-lambda: [-0.107, 0.022, 0.067, 0.211, 0.084]
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/imagination-training.png" alt="五步 imagination" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">从真实 posterior 出发的 5 步梦境。Actor 还是随机初始化，动作没有朝向目标；reward head 几乎没被训练，预测奖励会在正负之间晃。这一步检查的是「梦境能不能展开」，不是「梦境里已经会走路」。</div>
</div>

`[4, 5, 80]` 是 4 段起点、5 步想象、80 维特征。动作是 5 个离散值。return 没有变成 NaN。梦境展开了——但 Actor 还不会走路，这很正常。

## 第九步：各更新一次 Actor 与 Critic

教学版 Actor 使用 REINFORCE：高 return 的动作提高 log probability。完整 Dreamer 还会让梯度穿过动态模型本身。这里只走最容易检查的那条路。

```python
actor_loss = -(imagined['log_probs'] * returns.detach()).mean()
critic_loss = F.mse_loss(values, returns.detach())

actor_before = next(actor.parameters()).detach().clone()
actor_optimizer.zero_grad()
actor_loss.backward()
actor_optimizer.step()
critic_optimizer.zero_grad()
critic_loss.backward()
critic_optimizer.step()

print('actor loss:', round(float(actor_loss.detach()), 4))
print('critic loss:', round(float(critic_loss.detach()), 4))
print('Actor 参数改变:', bool(torch.any(
    actor_before != next(actor.parameters()).detach()
)))
```

**运行这一步，你会看到什么？** Notebook 会完成一次 Actor 与 Critic 更新，并检查参数确实改变：

```
actor loss: 0.2353
critic loss: 0.0295
Actor 参数改变: True
```

参数变了，说明梯度从「想象里的回报」流回了 Actor。它不说明策略变好了。一次更新、4 段起点、几乎没训练过的 reward head——任何「成功率上升」都是噪声。

位置模型那一段有真实环境对照；RSSM 这一段仍然只证明训练接口连通，不能写成 Dreamer-lite 已完成。

## 运行与产物

```bash
PYTHONPATH=src python -m unittest tests.test_control tests.test_neural -v
python scripts/run_position_dynamics_reference.py --output runs/reference/position-dynamics
```

跑完两份 Notebook 后，你应该有：

- **第一份 Notebook**：loss 下降曲线、prior / posterior 距离、原图 vs 复制上一帧 vs 重建
- **第二份 Notebook 的位置模型**：planned vs random 的成功率，一条 `(5,5) → (12,12)` 的真实路线
- **第二份 Notebook 的 RSSM 部分**：一次 5 步 imagination，以及「Actor 参数改变 = True」

`run_position_dynamics_reference.py` 还会写出 `position-dynamics.pt`、`metrics.json` 和带 sha256 的 `manifest.json`。那是第 9 章运行证据规范在这条路线上的最小实例。

## 本节与 4.8 的区别

| 项目 | 本节                                  | 4.8                               |
| ---- | ------------------------------------- | --------------------------------- |
| 数据 | 4 段 PixelWorld；位置模型 180 条单步  | 更大的 PixelWorld；选做 DMC       |
| 训练 | 15–100 步；Actor 更新 1 次            | 直到形成稳定曲线                  |
| 目的 | 检查控制增益、接口与梯度              | 检查完整 latent policy 的回报     |
| 资源 | CPU                                   | 单张 24GB GPU                     |
| 结论 | 可解释动态能帮助行动；RSSM 接口可运行 | latent 模型与策略是否形成稳定闭环 |

## 已知简化与坑

教学版有几处刻意的简化，跑不通时先从这里找原因：

- **复制上一帧是更强的像素基线。** 15 步重建 MSE 0.014 仍高于 0.006。不要把 reconstruction 下降写成「学会了动态」。
- **数据量极小。** 4 段 episode 只有 32 个转移，第一份 Notebook 里的模型会迅速过拟合。loss 下降不代表泛化，只代表接口连通。
- **PixelWorld 过于简单。** 16×16、5 个动作、红方块——这不是 Atari。RSSM 在这里学到的「世界」非常有限。
- **位置来自颜色阈值，不是模拟器开挂。** `locate_red_square` 读的是像素。换一张没有大红块的图，这条基线立刻失效。
- **第二份 Notebook 的 RSSM 部分只做了一次更新。** 真正的 Dreamer 需要数千次想象。它只证明梯度能流、参数能变，不证明策略能学。
- **教学版 Actor 是 REINFORCE。** 完整 Dreamer 还会让梯度穿过动态模型。动作采样把那条路截断了。

## 扩展练习

跑通默认配置后，按从便宜到昂贵的顺序推荐：

1. **重建对基线**：把第一份 Notebook 的重建 MSE 和复制上一帧画在同一张图上，扫 `num_episodes` 从 4 到 40。哪一个规模上，模型第一次赢过复制？
2. **改变 horizon**：把第二份 Notebook 里位置模型的规划深度从 1 提到 8，观察步数和成功率。horizon 越长，并不总是越好。
3. **想象长度**：把 `imagine` 的 horizon 从 5 提到 20，观察预测奖励会不会爆、continue 会不会塌到 0。复合误差在隐空间里一样会发生。
4. **把重建画进梦境**：对想象出来的特征调用 `model.decode`，看 5 步 prior rollout 的画面。前两步也许还能辨认方块，后面通常糊掉——这就是 4.6 里那张 free-running 图，在 RSSM 里的对应物。

完成两份 Notebook 后进入 [4.8 动手：Dreamer 的完整闭环](/chapters/04-decision-and-planning/08-dreamer-loop)：那台模型不再用 4 段 episode 做 smoke，而是收集、想象、更新、再收集。你会亲身体会「接口连通」与「策略收敛」之间的巨大鸿沟。

## 本节小结

- **第一份 Notebook 确认 RSSM 接口连通**：CNN 压像素，RSSM 的 prior / posterior 双路结构，三个预测 head，15 次更新后 loss 能下降。
- **第一份 Notebook 同时证明 loss 不够**：复制上一帧的像素误差仍低于训练后的解码器。相邻帧太像，像素损失会奖励偷懒。
- **第二份 Notebook 先做可解释基线**：位置模型的 MSE 下降，beam search 的成功率显著高于随机——证明 learned dynamics 真的能帮助行动。
- **第二份 Notebook 再接通 Actor-Critic**：把位置换成 RSSM latent，Actor 采样动作、prior 推演、Critic 给 value、TD-λ 算 target，一次更新后参数确实改变。
- **Smoke 不是完整训练**：4 段 episode、15–100 步、CPU 运行——目标是检查数据流，不是复现 DreamerV3 的排行榜。

从 4.6 的 World Models 到这一节的 Dreamer 接口，核心思想从未改变：**在行动之前，先在内部预见行动的后果**。4.6 用 CMA-ES 在梦境里进化控制器，这一节用梯度下降在梦境里训练 Actor-Critic——前者像蒙眼摸索，后者像看着地图走。而 4.8 会让你亲手把这张地图画完整。

## 后续工作

这两份 Notebook 只把 Dreamer 的接口接到了 PixelWorld。它留下了三个明显的缺口，后来的工作逐一走得更远。

### 短板一：还在用连续高斯猜下一帧

教学版的 \(z_t\) 是对角高斯。PixelWorld 这种干净小图还能撑；换到 Atari，连续 latent 很容易把「向左」和「向右」平均成「停在中间」。**DreamerV2 / V3**（文献 2、3）把随机状态换成离散 latents，一套超参打通 150 多个任务。离散化之后，KL 和 straight-through 会变成新的坑。

### 短板二：还在用枚举搜索动作

第二份 Notebook 的 beam search 只适用于 5 个离散动作。连续控制里，动作树立刻爆炸。**PlaNet** [1] 还不用 Actor-Critic，它在 RSSM 里跑 CEM：采样一群动作序列，留下精英，收缩分布，再采一轮。和 beam search 是亲戚——都在模型里试动作，都只执行第一步。

### 短板三：像素重建可能是负担

第一份 Notebook 里，复制上一帧赢过了训练后的解码器。这已经在暗示一件事：像素不一定是对的监督。**MuZero** [4] 连重建头都去掉，只预测奖励、策略和价值，在隐空间里做蒙特卡洛树搜索。世界模型不需要重建世界，只需要重建决策需要的信息。

从 4.6 的 867 个参数，到这一节的一次可微想象，被替换的是搜索方式，不是那句老话。下一台模型要回答的，是你刚刚亲眼看见的那些失败：重建赢不过复制上一帧、梦境里的 Actor 还不会走路、一次梯度改变了参数却改变不了行为。

## 参考文献

1. Hafner, D., et al. (2019). Learning Latent Dynamics for Planning from Pixels. _ICML 2019_. [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) —— PlaNet：RSSM + CEM。
2. Hafner, D., et al. (2019). Dream to Control: Learning Behaviors by Latent Imagination. _ICLR 2020_. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) —— DreamerV1：本节 RSSM + imagination + Actor-Critic 的原文。
3. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. _arXiv:2301.04104_. [链接](https://arxiv.org/abs/2301.04104) —— DreamerV3：一套超参打通 150+ 任务。
4. Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. _Nature_. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) —— MuZero：不重建像素，只在隐空间搜索。
5. Sutton, R. S. (1988). Learning to Predict by the Methods of Temporal Differences. _Machine Learning_, 3, 9–44. [链接](https://doi.org/10.1007/BF00115009) —— TD-λ 的原始论文。
