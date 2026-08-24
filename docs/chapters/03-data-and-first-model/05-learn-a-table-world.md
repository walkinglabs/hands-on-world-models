# 3.5　动手：表格世界模型的从零开始实现

> **本节目标**：把一段经历装进 Episode，在段内取样、按段切开，再从打滑的 LineWorld 里数出转移概率，用 MPC 走到终点。第一台世界模型甚至可以没有梯度。

> **本节代码**：[本节 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/03_data/learn-a-table-world.ipynb) · [data.py](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/data.py) · [gridworld.py](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/gridworld.py)

> **前置知识**：你已经读过第 3 章前四篇，知道观察比动作多一帧、不能跨 episode 拼接，以及怎样检查一台模型。这一节把它们真跑一遍。

---

0.6 的转移表是你写的。现在拿走那张表。一段经历怎样保存、怎样取样、怎样切分，再怎样变成 \(\hat P(s'|s,a)\)，必须亲手接起来。跑完你会对「第一台世界模型」四个字有完全不同的理解——它甚至可以没有梯度。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/f3-mpc.png" alt="LineWorld 上的 MPC" style="max-width:min(900px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">这就是我们要让模型学会的「世界」：一条长度为 7 的线，× 是陷阱，G 是终点，A 是自己。每一步有 20% 概率打滑、停在原地。模型从未见过这条规则，它要从轨迹里自己数出来。</div>
</div>

## 本次会得到什么

- 12 段 PixelWorld：观察 `(9, 16, 16, 3)`，动作 `(8,)`
- 一次校验失败：若 `dones` 中间出现 True，`Episode.validate` 会拒绝
- 按 0.70 / 0.15 切开的 8 / 2 / 2 段
- 四条段内窗口：观察 5 帧、动作 4 步，`episode_id` 不串段
- 从 140 段 LineWorld 数出的 \(\hat P(\cdot\mid 3,\text{left})\) 与 \(\hat P(\cdot\mid 3,\text{right})\)
- 按段切开后的一步准确率 0.831
- 一次 MPC：第一步打滑停在 3，后面三步仍然走到 6

## 怎样运行

```text
notebooks/03_data/learn-a-table-world.ipynb
```

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest tests.test_data tests.test_gridworld -v
```

教学版只依赖 NumPy，CPU 上几秒。下面数字都是 `seed=0 / seed=1` 对着源码跑出来的。

## 第一步：先把一段经历装对

第 `t` 个动作夹在第 `t` 帧和第 `t+1` 帧之间。所以观察永远比动作多一帧：

```python
from hwm.data import make_pixelworld_dataset

episodes = make_pixelworld_dataset(num_episodes=12, length=8, seed=0)
ep = episodes[0]
print(ep.episode_id)
print('obs', ep.observations.shape, 'act', ep.actions.shape)
print('dones', ep.dones.tolist())
print('actions', ep.actions.tolist())
ep.validate()
```

**运行这一步，你会看到什么？**

```
pixelworld-000
obs (9, 16, 16, 3) act (8,)
dones [False, False, False, False, False, False, False, True]
actions [0, 2, 4, 3, 3, 2, 3, 2]
```

8 步、9 帧。`dones` 只有最后一格是 True——这段走完了，没有在中间重置。`validate` 会查四件事：至少两帧、动作数是观察数减一、reward / done 与动作等长、`dones[:-1]` 必须全假。

中间若冒出一个 True，源码直接抛错：`episode 结束以后不能继续保存 transition`。那不是格式洁癖。`done=True` 之后环境已经重置，下一条记录若还接在同一段里，模型会把「从终点瞬移回起点」当成一种普通动态。

**这就是时间对齐**：后面所有 Encoder、RSSM、视频模型，第一步都在确认这件事。错开一格，训练仍能降 loss，学到的却是画面惯性，不是动作。

## 第二步：取样时不要跨过段的边界

许多段放进同一个池子，训练时从里面反复取窗口——这就是 Replay Buffer。它看起来像数组，取样规则却不能像数组。

episode A 在终点结束，episode B 从一个新的随机状态开始。若先把两段首尾相接再切窗口，模型会学到一条不存在的转移：

$$
o^{(A)}_{T} \xrightarrow{\;a\;} o^{(B)}_{0}.
$$

正确顺序：先选一段足够长的 episode，再在它**内部**切连续片段。

```python
from hwm.data import ReplayBuffer, split_by_episode

split = split_by_episode(episodes, train_ratio=0.70, val_ratio=0.15)
print({k: [e.episode_id for e in v] for k, v in split.items()})

buffer = ReplayBuffer()
for episode in split['train']:
    buffer.add(episode)

batch = buffer.sample(batch_size=4, sequence_length=4, seed=0)
for item in batch:
    print(item['episode_id'], 'start', item['start'],
          item['observations'].shape, item['actions'].shape)
```

**运行这一步，你会看到什么？**

```
train: pixelworld-000 … 007
val:   pixelworld-008, 009
test:  pixelworld-010, 011

pixelworld-006 start 3  (5, 16, 16, 3) (4,)
pixelworld-000 start 2  (5, 16, 16, 3) (4,)
pixelworld-007 start 3  (5, 16, 16, 3) (4,)
pixelworld-004 start 3  (5, 16, 16, 3) (4,)
```

观察比动作多一帧：窗口长度 4，观察就是 5。四条窗口各自来自同一段，没有一半 A、一半 B。

旧讲义里写过「200 段划成 140 / 30 / 30」。那组数字对不上源码。`split_by_episode` 用的是比例：12 段、0.70 / 0.15，就是 8 / 2 / 2。

**这就是 Replay Buffer 要保住的东西**：时间连续性。把 transition 打散再训练，RSSM 的 GRU 就在背不存在的历史。

**一个值得做的实验**：故意把两段 `np.concatenate` 再切窗口，看会不会切到段界。切到了，那条「转移」在真实环境里一次也不会发生。

## 第三步：切分按段，不按帧

相邻两帧几乎相同。若按帧随机划 train / val / test，测试集里会出现训练帧的近邻，一步准确率会被抬虚。

12 段已经按段切开了。还要记住另一件事：确定性格子里，按段切开只能防止「同一条轨迹的未来漏进测试集」。同一条物理规律——「在 3 向右，多半走到 4」——仍可能同时出现在 train 和 test。数据卡要写清你防的是哪一种泄漏。

动作条件任务还要检查：测试集里有没有训练从未覆盖过的 \((s,a)\)。没见过的键，计数模型会停在原地。那不是泛化，是承认无知。

## 第四步：从计数学出第一台世界模型

不必先上神经网络。换到 3.4 的 LineWorld：长度 7，起点 3，终点 6，陷阱 0，每一步有 \(20\%\) 概率打滑。

```python
from hwm.gridworld import LineWorld, EmpiricalDynamics

world = LineWorld()
print(world.render(world.start))

transitions, episode_ids = world.collect(
    policy=lambda state, rng: rng.choice(list(world.actions)),
    episodes=140,
    max_steps=20,
    seed=0,
)
print(len(transitions), 'mean length', round(len(transitions) / 140, 2))

model = EmpiricalDynamics().fit(transitions)
print('left ', model.distribution(3, 'left'))
print('right', model.distribution(3, 'right'))
```

**运行这一步，你会看到什么？**

```
× · · A · · G
1494 条转移，平均每段 10.67 步

left  {2: 0.792, 3: 0.208}    # 210 次走到 2，55 次停在 3
right {4: 0.810, 3: 0.190}    # 204 次走到 4，48 次停在 3
```

真实打滑是 0.20。140 段数出来大约是 0.21 / 0.19，不是教科书上的整数。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/f3-counts.png" alt="从数据里数出的转移" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">从格子 3 向左、向右的经验分布。有限计数不会恰好等于 0.20 / 0.80。这已经是一台世界模型：它保留了两种可能，也标出了没见过的格子。</div>
</div>

公式就是归一化计数：

$$
\hat{P}(s'\mid s,a)
= \frac{n(s,a,s')}{\sum_{s''} n(s,a,s'')}.
$$

规划时取众数；没见过的 \((s,a)\) 停在原地，给一步代价 \(-1\)。未知不是「随机猜」，是「承认没见过」。

按 episode 切开再测一步（前 100 段训练，后 40 段测试）：

```python
train = [t for t, e in zip(transitions, episode_ids) if e < 100]
test  = [t for t, e in zip(transitions, episode_ids) if e >= 100]
held = EmpiricalDynamics().fit(train)
acc = sum(
    held.transition(t.state, t.action).next_state == t.next_state
    for t in test
) / len(test)
print(round(acc, 3), len(train), len(test))
print(held.transition(3, 'left').next_state,
      held.transition(3, 'right').next_state)
```

**运行这一步，你会看到什么？**

```
0.831   train 1074 / test 420
left → 2    right → 4
```

准确率到不了 1。世界本身在掷骰子，众数预测必然有一部分对不上下一次抽样。向左向右的众数不同——模型读了动作。

**这就是第一台可学习世界模型**：没有梯度，只有计数。它已经能保留多种结果、标出未见过的键、被 Planner 调用。

## 第五步：只执行第一步，打滑了再看

把规划器选出的第一步放回真实 LineWorld，保存新的 transition，再规划。这就是 MPC。

```python
from hwm.gridworld import mpc_episode

traj, plans = mpc_episode(world, held, depth=4, max_steps=20, seed=1)
for step, (item, plan) in enumerate(zip(traj, plans)):
    slipped = item.state == item.next_state
    print(step, item.state, item.action, '→', item.next_state,
          'r', item.reward, 'plan', plan.actions,
          'SLIP' if slipped else '')
```

**运行这一步，你会看到什么？**

```
0  3 right → 3   r -1    plan ('right', 'right', 'right', 'down')  SLIP
1  3 right → 4   r -1    plan ('right', 'right', 'right', 'down')
2  4 right → 5   r -1    plan ('right', 'right', 'down', 'down')
3  5 right → 6   r +10   plan ('right', 'down', 'down', 'down')
```

第一步想向右，打滑停在 3。计划作废，下一步对着真实位置再搜，仍然走到 6。

注意计划里出现了 `down`。`lookahead` 默认搜 GridWorld 的四个方向，LineWorld 只有左右。未知动作在模型里会停在原地。256 = \(4^4\)，不是 \(2^4\)。把 `action_order=world.actions` 传进去，评估次数会变成 16，计划里的 `down` 也会消失。这是源码里的真坑，不是排版错误。

**这就是闭环**：一步准确率 0.83 和「走到终点」不是一回事。打滑让前者到不了 1；只要模型对动作敏感，规划仍可能稳定到终点。

**一个值得做的实验**：把训练段从 20 扫到 200，画一步准确率和 MPC 成功率。覆盖率比模型结构更先决定成败。

## 已知简化与坑

- **12 段 PixelWorld 只用来看 shape。** 真正学动态的是后面那条会打滑的线。不要把 8 / 2 / 2 写成 3.4 的切分。
- **0.831 含 20% 打滑。** 先看 `slip_probability`，再判断模型有没有学好。
- **`lookahead` 默认四个动作。** 在 LineWorld 里会搜到 `up` / `down`。
- **按段切开防的是轨迹泄漏，不是规律泄漏。** 确定性格子里，同一条 \((s,a,s')\) 仍可能同时出现在 train 和 test。
- **旧讲义的 140 / 30 / 30 对不上源码。** 以 `split_by_episode` 的比例为准。

## 展开与下一步

| 这一节刚跑过的                  | 完整展开                                                    |
| ------------------------------- | ----------------------------------------------------------- |
| `Episode.validate`、12 段 shape | 2.1                                                         |
| `ReplayBuffer.sample`、按段切   | 2.2                                                         |
| 计数表、反事实、MPC             | 2.3、本页                                                   |
| 自己留一个缺口再走一遍          | [3.6](/chapters/03-data-and-first-model/06-learnable-world) |

如果表格方法已经解决了任务，继续换上神经网络不会加分。只有当图像、历史、不确定性或动作空间确实让表格失效时，才该进入后续路线。先找到失效点，再决定加什么。

> 理论铺垫：[3.3 从经历学出转移模型](/chapters/03-data-and-first-model/03-first-learned-world) 与 [3.4 世界模型的基本检查](/chapters/03-data-and-first-model/04-basic-checks) · 下一节：[3.6 动手：重新发明一台可学习世界模型](/chapters/03-data-and-first-model/06-learnable-world)

## 小结

- [ ] 观察比动作多一帧；`dones` 中间不能为真。
- [ ] Replay Buffer 只在同一 episode 内切窗口，不跨段拼接。
- [ ] train / val / test 按段切；12 段、比例 0.70 / 0.15 是 8 / 2 / 2。
- [ ] \(\hat P(s'\mid s,a)=n(s,a,s')/n(s,a)\) 就是第一台世界模型。140 段从 3 向左约 0.79 / 0.21，不是整数 0.80 / 0.20。
- [ ] 一步准确率 0.831 到不了 1；MPC 第一步打滑，后面仍然走到 6。
