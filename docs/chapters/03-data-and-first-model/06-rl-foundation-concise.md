# 3.6　强化学习基础的简洁实现

> **本节目标**：不再使用人工写的转移表。你将面对一个「当前观察不够用、数据盖不住、规划 horizon 不够长」的小世界，自行决定什么状态值得保存、模型输出一个结果还是分布、Planner 怎样使用它。最终交出一份完整的证据链：接口图、数据卡、基线、learned dynamics、四类评价、一组稳定失败、一项最小改动。

> **本节代码**：[项目模板](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/projects/learnable-world-template.ipynb) · [`gridworld.py`](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/gridworld.py)

> **前置知识**：你已经跑过 1.6（九格世界 + 表格动态）和 3.4（LineWorld 计数动态），知道世界模型的最小闭环——预测→规划→执行→修正。本节不再给你完整转移表，你需要自己设计缺口，并让一种失败稳定出现。

---

1.6 给了你一个 3×3 的网格世界，转移表是人工写的。你只需要查表、规划、执行。一切都很干净。

3.4 给了你一条线形世界，动态是带打滑的计数。你从轨迹里数出转移概率，发现「学到的」和「真实的」可以非常接近。

但真实世界不会给你一张完美的表格。你会遇到这些情况：

- **当前观察看不到完整状态**——你需要记住过去两步才能推断位置；
- **同一个动作在不同地面上结果不同**——但地面类型暂时不可见；
- **目标在 episode 开始时改变**——你的模型需要对「目标在哪」敏感；
- **地图里有一块训练数据没覆盖的区域**——你的模型走到那里时会暴露无知。

还有第五种、也是本仓库里立刻就能复现的缺口：**one-step 全对，短 horizon 的规划仍然到不了目标**。数据盖住了局部转移，却盖不住「从起点看到终点」所需的深度。

本节的任务是：**选择其中一种变化，让一种失败稳定出现，然后用数据、接口、模型和反例把它讲清楚。**

不是造最复杂的环境，不是画最漂亮的架构图。是让一个缺口变得可见。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/world-loop.png" alt="世界模型学习循环" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">一张还算干净的 5×5 世界：左边是乱走收集到的轨迹，中间是从轨迹里数出来的转移，右边是 MPC 走出来的路。本节要你打破的，正是这种「学到的和真的可以完全一致」。</div>
</div>

## 本次会得到什么

提交时，你应该能拿出：

- 一张自己选的小世界地图，以及「一次只改了一种条件」的说明
- 一份按 episode 切分的数据卡：字段、shape、seed、覆盖率
- 一种没有学坏的基线（复制上一状态，或永远向目标走）
- 一台从 `Transition` 计数得到的 `EmpiricalDynamics`，以及它读动作的证据
- 四类数字：one-step、多步 rollout、反事实、下游任务
- 一组固定 seed 下可重复的失败，以及至少两种解释
- 一项最小下一步，而不是一张「CNN + Transformer + Diffusion + MCTS」拼图

下面用仓库里的 5×5 网格把整条证据链先走一遍。本节可以换环境，但字段、切分、四类评价和稳定失败不能缺。

## 为什么本节是整门课的分水岭

1.6 和 3.4 的世界模型是「玩具」——状态完全可观测，动态完全确定或几乎被数据盖住，规划深度刚好够用。本节第一次打破这三个假设中的至少一个。

打破之后，1.6 的那套方法会不够用：

- 转移表不能处理**部分可观测**——你需要某种记忆或历史拼接；
- 单点预测不能处理**随机动态**——你需要输出分布而不是单一结果；
- 短 horizon 贪心不能处理**未见区域或过远的目标**——你需要更深的搜索、更好的覆盖，或承认模型不知道。

本节不要求你用神经网络。表格、线性模型、小 MLP 都可以。它要求的是：**你能把一个现实缺口变成数据、接口、模型、反例和下一步设计。** 这是做研究的最小完整循环。

## 第一步：选择一种变化，先把世界画出来

从下面五种变化中选一种，**一次只选一种**：

```text
1. 部分可观测：当前观察看不到完整位置，需要最近两步历史；
2. 隐式条件：同一个动作在两种地面上结果不同，但地面类型暂时不可见；
3. 目标漂移：目标会在 episode 开始时改变；
4. 未覆盖区域：地图中有一个训练数据未覆盖的区域；
5. 规划深度不够：one-step 很准，但短 horizon 看不到终点奖励。
```

修改 3.4 的 `LineWorld`，或在 `GridWorld` 上构造一个类似的小世界。环境必须足够小，能在 CPU 上 5 分钟内完成全部实验。

本节后面的数字，全部来自这个 5×5：

```python
from hwm.gridworld import GridWorld

world = GridWorld(
    rows=5,
    cols=5,
    start=(0, 0),
    goal=(4, 4),
    walls=((2, 2),),
    traps=((1, 3),),
)
print(world.render(world.start))
```

**运行这一步，你会看到什么？**

```
A · · · ·
· · · × ·
· · ■ · ·
· · · · ·
· · · · G
```

起点在左上，目标在右下，中间一块墙，`(1, 3)` 是陷阱。动作是 `down / right / up / left`。走进目标得 `+10`，走进陷阱得 `-10`，其余每步 `-1`。这就是 1.6 九格世界放大一圈之后的样子。

**这就是「先把世界钉死」**：后面所有覆盖率、准确率、成功率，都必须对着同一张地图说话。换了陷阱位置却沿用旧数字，证据作废。

**一个值得做的实验**：先让你的变化「几乎不造成影响」——部分可观测时给很大的感受野，未覆盖区域先只挡住一格，规划深度先设到能看见终点。确认基线能通过。然后逐步收紧，直到基线稳定失败。本节的目的不是制造最复杂环境，而是让一种失败稳定出现。

## 第二步：自己收集数据，写数据卡

本节不给你数据集。`EmpiricalDynamics` 吃的是 `Transition`，不是随便三个元组。

```python
import random
from hwm.gridworld import GridWorld, EmpiricalDynamics, ACTIONS

world = GridWorld(
    rows=5, cols=5, start=(0, 0), goal=(4, 4),
    walls=((2, 2),), traps=((1, 3),),
)
rng = random.Random(0)
action_names = list(ACTIONS)

transitions, episode_ids = [], []
for ep in range(50):
    state = world.start
    for _ in range(20):
        action = rng.choice(action_names)
        item = world.step(state, action, rng)
        transitions.append(item)
        episode_ids.append(ep)
        state = item.next_state
        if item.done:
            break

train = [t for t, eid in zip(transitions, episode_ids) if eid < 40]
val   = [t for t, eid in zip(transitions, episode_ids) if 40 <= eid < 45]
test  = [t for t, eid in zip(transitions, episode_ids) if eid >= 45]
print(len(transitions), len(train), len(val), len(test))
```

`Transition` 的字段是 `(state, action, reward, next_state, done)`。第 `t` 个动作对应 `state_t → next_state_t`，不是 `t+1` 才生效。

**运行这一步，你会看到什么？** `seed=0`、50 段、每段最多 20 步时：

```
transitions: 863
mean episode length: 17.26
random reached goal: 3 / 50
random fell in trap: 15 / 50
mean reward per step: -1.12
split by episode: train 699 / val 85 / test 79
```

数据卡至少写明：

```text
- 数据怎样生成（什么策略、多少 episode、每段多长、哪个 seed）；
- transition 字段（state, action, reward, next_state, done）；
- action 对应哪两次观察（t 还是 t+1）；
- episode 在哪里结束（到达目标、掉进陷阱、步数上限）；
- train / val / test 按什么边界切分（必须按 episode）；
- 覆盖了多少 (state, action)。
```

**一个关键陷阱**：切分必须按 episode 边界，不能按步。若把 863 条转移的后 20% 直接切成 test，这 173 条与前 80% 的 `(state, action, next_state)` **全部相同**——确定性格子世界里，按步切分等于把同一条物理规律同时放进 train 和 test。按 episode 切开，只能证明「没把同一段轨迹的未来泄漏过去」，不能证明 test 里的转移没在 train 里出现过。数据卡要写清你防的是哪一种泄漏。

**这就是数据卡要挡住的事**：别人问「模型训练时看到了测试轨迹吗？」，你能用切分规则回答，而不是用感觉回答。

**一个值得做的实验**：只收集 5 段 episode。覆盖率会从 50 段的 `70 / 96 = 73%` 掉到 `35 / 96 = 36%`。200 段才能到 `87 / 96 = 91%`。后面所有「模型不会」的结论，先核对是不是根本没见过那个格子。

## 第三步：画出接口图

不用论文架构图。用你自己的变量名，写清数据怎样流动：

```text
obs_t, a_t  →  EmpiricalDynamics.counts[(s, a)]
            →  P(s' | s, a) 或最可能的 s'
Planner     →  lookahead / mpc_episode
环境        →  world.step 返回真实 Transition
比较        →  predicted.next_state 对 actual.next_state
```

这张图必须回答四个问题：

1. **模型的输入是什么？** 当前格子？最近两步观察的拼接？动作名字还是 one-hot？
2. **模型的输出是什么？** 一个确定的下一状态？一个概率分布？均值和方差？
3. **Planner 怎样使用模型？** `lookahead` 穷举？随机采样？只取 `distribution` 的众数？
4. **环境怎样返回真实结果？** `world.step` 与 `model.transition` 在哪里比较？

教学版的计数模型是：

$$
\hat{P}(s'\mid s,a)
= \frac{n(s,a,s')}{\sum_{s''} n(s,a,s'')}
$$

规划时不抽样，只取众数，并对未见过的 `(s, a)` 停在原地、给一步代价 `-1`：

$$
\hat{s}_{t+1}
=
\begin{cases}
\arg\max_{s'} \hat{P}(s'\mid s_t,a_t) & n(s_t,a_t)>0 \\
s_t & \text{otherwise}
\end{cases}
$$

这和 `EmpiricalDynamics.transition` 一致：有计数就走最常见的下一格，没计数就原地踏步。未知不是「随机猜」，是「承认没见过」。

**运行这一步，你会看到什么？** 一张手绘接口图。如果别人看了这张图，能直接复现你的数据流，说明它够清楚。

**这就是接口图的判据**：能复现，就够了；不能复现，再漂亮也是插图。

## 第四步：实现简单基线

先实现一种不学习或很少学习的办法，并且**不要故意写坏**。

```python
def copy_last_accuracy(data):
    return sum(t.state == t.next_state for t in data) / len(data)

def greedy_episode(world, max_steps=20):
    state = world.start
    path, reward = [state], 0.0
    for _ in range(max_steps):
        action = world.greedy_action(state)
        item = world.transition(state, action)
        path.append(item.next_state)
        reward += item.reward
        state = item.next_state
        if item.done:
            break
    return path, reward
```

`greedy_action` 只看下一格离终点的曼哈顿距离，不看陷阱，也不读学到的模型。

**运行这一步，你会看到什么？** 同一张 5×5、`seed=0` 的切分上：

```
copy-last accuracy
  train 0.262   val 0.282   test 0.215

greedy path (no model):
  (0,0) → (0,1) → (0,2) → (0,3) → (0,4)
        → (1,4) → (2,4) → (3,4) → (4,4)
  return 3.0    reached goal
```

复制上一状态只有两成对——格子世界里大多数动作真的会换格，这个基线应该弱。贪心在这张地图上反而能到：陷阱在 `(1, 3)`，不挡最短路。

**这就是「基线不能故意写坏」**：在这个世界里，不学习的贪心已经能到终点。后面 learned dynamics 若还到不了，就不能写成「环境太难」，只能写成「模型或 Planner 出了问题」。

若你把陷阱挪到贪心路上，例如 `traps=((0, 1), (1, 1))`，贪心第一步就会掉进 `(0, 1)`，回报变成 `-10`。那是另一种合法的本节：让无模型基线稳定失败，再看学到的模型能不能绕开。一次只选一种。

**一个值得做的实验**：先确认基线在「变化很弱」时能过，再收紧变化。基线成功率若是 0%，先检查环境；若是 100%，learned dynamics 必须在别的指标上赢，例如更短的路径、避开陷阱、或在未覆盖区域不装懂。

## 第五步：实现 learned dynamics

用 `EmpiricalDynamics` 从 train 计数。模型必须把动作当作条件。

```python
from hwm.gridworld import EmpiricalDynamics

model = EmpiricalDynamics()
model.fit(train)

def one_step_accuracy(model, data):
    return sum(
        model.transition(t.state, t.action).next_state == t.next_state
        for t in data
    ) / len(data)

keys = [
    ((r, c), a)
    for r in range(world.rows)
    for c in range(world.cols)
    if (r, c) not in world.walls
    for a in action_names
]
covered = sum(1 for key in keys if model.distribution(*key))
print('coverage', covered, '/', len(keys))
print('one-step', one_step_accuracy(model, test))
print(model.distribution((0, 0), 'right'))
print(model.transition((4, 3), 'down'))  # 训练里没见过
```

**运行这一步，你会看到什么？**

```
train coverage: 70 / 96 = 0.729
one-step accuracy: train 1.000 / val 1.000 / test 1.000
P(s' | (0,0), right) = {(0,1): 1.0}
unknown (4,3), down → stay at (4,3), reward -1.0
```

确定性世界里，见过的 `(s, a)` 会变成概率 1 的单点；没见过的 26 个键，模型按设计停在原地。one-step 在 test 上 100%，只说明随机策略走到的那些格子被数清楚了，不说明 Planner 能从 `(0, 0)` 走到 `(4, 4)`。

若你选的是 LineWorld 那种打滑动态，输出就必须是分布。`slip_probability=0.2`、140 段随机轨迹之后，位置 3 向右的计数是：

```
P(s' | 3, right) ≈ {3: 0.19, 4: 0.81}
P(s' | 3, left)  ≈ {2: 0.79, 3: 0.21}
```

真值是 0.8 / 0.2。模型学到的是频率，不是「下一次一定成功」。

**这就是「模型必须读动作」**：换动作，分布必须换。`(0, 0)` 向右到 `(0, 1)`，向上停在 `(0, 0)`。若换动作后预测不变，这台模型不能拿去规划。

**一个值得做的实验**：把表格换成「只输入状态、不输入动作」的计数器。one-step 可能还看起来不差——因为随机策略下下一格的边际分布有峰。反事实测试会立刻拆穿它。

## 第六步：四类评价

同时报告四种评价，缺一不可。

### 1. One-step

模型预测的下一状态与真实下一状态是否一致。离散格子用准确率，连续状态用 MSE。上面已经有：test 上 `1.000`，复制上一状态只有 `0.215`。

### 2. 多步 rollout

从同一起点出发，让模型连续吃自己的预测。这就是 1.6 教过的复合误差。

```python
def rollout_compare(model, world, start, actions):
    s_model, s_true = start, start
    rows = []
    for action in actions:
        predicted = model.transition(s_model, action)
        actual = world.transition(s_true, action)
        rows.append((s_model, predicted.next_state, s_true, actual.next_state))
        s_model, s_true = predicted.next_state, actual.next_state
        if actual.done:
            break
    return rows

print(rollout_compare(
    model, world, (0, 0),
    ['right', 'right', 'right', 'right', 'down', 'down', 'down', 'down'],
))
```

**运行这一步，你会看到什么？** 这条朝向目标的 8 步，模型和真值逐步重合，最后都到 `(4, 4)`。one-step 准的时候，沿已覆盖路径的多步也可以准。换一条钻进未覆盖格子的动作序列，误差会从某一步突然变成「停在原地」。

### 3. 反事实

同一起点，只替换动作。

```
at (0, 0):
  down  pred (1, 0)   true (1, 0)
  right pred (0, 1)   true (0, 1)
  up    pred (0, 0)   true (0, 0)
  left  pred (0, 0)   true (0, 0)
```

四个动作给出三个不同的下一格。模型读了动作。

### 4. 下游任务

把模型交给 `mpc_episode`：在模型里规划，在真实环境里只执行第一步。

$$
a_t^*
= \arg\max_{a_{t:t+H}}
\sum_{k=0}^{H-1} \hat r(\hat s_{t+k}, a_{t+k}),
\quad
\hat s_{t+k+1} = f(\hat s_{t+k}, a_{t+k})
$$

只执行 \(a_t^*\)，然后重新观察。教学版的 `lookahead` 穷举 \(4^H\) 条序列，取累计预测奖励最高的一条。

```python
from hwm.gridworld import mpc_episode, format_trajectory

for depth in (1, 3, 6):
    items, plans = mpc_episode(world, model, depth=depth, max_steps=20, seed=0)
    print(depth, format_trajectory(items), sum(t.reward for t in items), plans[0].actions)
```

**运行这一步，你会看到什么？**

```
depth=1  (0,0)↓(1,0)↓…↓(4,0)↓(4,0)…   return -20   first=('down',)
depth=3  同样卡在 (4,0)                 return -20   first=('down','down','down')
depth=6  (0,0)↓(1,0)↓(2,0)↓(3,0)↓(4,0)
         →(4,1)→(4,2)→(4,3)→(4,4)     return 3     first=('down',)*6

20 个 seed、depth=3：成功率 0 / 20，mean return -20
10 个 seed、depth=6：成功率 10 / 10，mean return 3
```

同一台 one-step 准确率 100% 的模型，horizon 3 全军覆没，horizon 6 全部到达。真环境本身也一样：用 `world` 当 oracle 模型，`depth=3` 仍然卡在左下角，`depth=8` 才规划出 `down×4 + right×4`、预测回报 `3.0`。

**这就是四类评价缺一不可的原因**：one-step 说「转移学会了」，多步说「沿这条路还能跟住」，反事实说「模型读了动作」，下游说「Planner 能不能用它办事」。四者可以互相打架。打架的地方，就是你要写进报告的缺口。

**一个值得做的实验**：把 `lookahead` 的评分从累计奖励改成「终点曼哈顿距离」。depth=3 有可能立刻学会往右下走。那你要诚实地写：失败来自奖励塑造和 horizon，不是来自转移表。

## 第七步：找到一组稳定失败

提交至少一组在固定 seed 下可重复的失败。不能只挑一张坏图。

上面这组就够用，而且稳定：

```
seed=0，50 段随机轨迹，按 episode 切分
one-step test accuracy = 1.000
copy-last test accuracy = 0.215
coverage = 70 / 96
greedy return = 3.0（能到）
learned MPC, depth=3, 20 seeds: 0 / 20，轨迹永远是
  (0,0) → (1,0) → (2,0) → (3,0) → (4,0) → (4,0) → …
```

从 `(0, 0)` 穷举 3 步，64 条动作序列的预测回报全部是 `-3.0`。Planner 用动作名字的固定顺序打破平局，于是永远先走 `down`。走到 `(4, 0)` 之后，短视的搜索仍然看不见右下角那个 `+10`，智能体在左下角撞墙直到超时。

然后至少给出两种解释：

```text
- 数据不足？右下角 (4,3)、(3,3)、(1,4) 在 699 条训练转移里分别只出现 2、2、2 次；
- 观察不足？若你选了部分可观测，当前格子编号可能根本不够；
- 模型能力不足？表格模型不会对未见 (s, a) 泛化，只会停在原地；
- 规划使用不当？depth=3 的累计奖励全是 -3，搜索在平局里选了「先向下」；
- 奖励太稀疏？+10 只在走进目标时出现，短 horizon 看不见。
```

本例里，后两条已经足够解释 depth=3 的崩溃；把 depth 加到 6，同一张表、同一批数据，成功率变成 10 / 10。所以这组失败更像「规划使用不当 + 稀疏奖励」，不太像「转移没学会」。

本节不要求你修好这个失败。它要求你**说清楚失败在哪里、为什么会出现、哪种解释已经被你排除**。

**这就是稳定失败**：换 20 个 seed 还是同一条左下角轨迹，不是偶尔翻车的一张图。

**一个值得做的实验**：把训练策略从纯随机改成 ε-greedy。贪心路上没有陷阱时，数据会更集中在最短路附近，覆盖率可能更低，但 depth=3 也许反而能看见 `+10`。写清你换的是数据分布，还是 Planner。

## 第八步：写下一步需求

最后只写一项最小改动。问自己：**如果只能改一件事，改什么能让这组失败消失？**

对上面这组数字，最小改动可以是：

- 把 `lookahead` 的 horizon 从 3 加到 6；或
- 把逐步 `-1` 改成「距离目标的势能」，让短视搜索也朝右下走；或
- 在未覆盖的 `(s, a)` 上加探索奖励，而不是静静停在原地。

不要直接画出「CNN + Transformer + Diffusion + MCTS」。

这一项改动，就是你进入下一章路线的起点。

## 运行与产物

本节必须可以在 CPU 上完成，运行时间目标不超过 5 分钟。

```bash
jupyter lab
# 打开 notebooks/projects/learnable-world-template.ipynb
PYTHONPATH=src python -m unittest tests.test_gridworld -v
```

跑完后，你应该有：

- **接口图**：用自己的变量名画的数据流图
- **数据卡**：字段、shape、切分方式、seed、覆盖率
- **基线结果**：复制上一状态或贪心的成功率 / 回报
- **Learned dynamics**：计数表或 loss、以及「换动作预测会变」的证据
- **四类评价**：one-step、多步、反事实、下游
- **稳定失败**：固定 seed 可重复 + 至少两种解释
- **下一步需求**：一项最小改动

## 评分

| 项目       | 分数 | 检查重点                                   |
| ---------- | ---: | ------------------------------------------ |
| 问题与接口 |   15 | 不靠模型名也能说清缺口                     |
| 数据与切分 |   20 | 时间对齐、episode 边界、覆盖率、无轨迹泄漏 |
| 基线与模型 |   20 | 基线合理，模型确实读取动作                 |
| 评价证据   |   20 | 一步、多步、反事实、下游齐全               |
| 失败诊断   |   15 | 失败稳定，解释至少有两种可能               |
| 表达与复现 |   10 | Notebook 可运行，seed 与输出完整           |

## 不接受的结论

下面这些句子，即使数字是真的，也不算交差：

- 「loss / one-step 下降了，所以世界模型学会了。」——上面 test 准确率 1.000，depth=3 的成功率仍是 0。
- 「基线是 0，我们的方法是 100%，所以模型很好。」——先检查是不是把贪心写坏了，或把测试起点全放在训练集里。
- 「随机策略有时能到，有时不能，这就是失败。」——必须固定 seed，稳定复现。
- 「下一步上 Transformer。」——只允许一项与失败直接对应的最小改动。
- 「按步切分也没关系，因为格子世界是确定的。」——你要防的是轨迹泄漏，不是转移是否确定。

## 已知简化与坑

教学版有几处刻意的约束，提交前先从这里检查：

- **环境必须足够小**。本节的目标是让一种失败稳定出现。如果需要 GPU 或超过 5 分钟，说明设计太复杂了。
- **一次只选一种变化**。不要同时让部分可观测 + 目标漂移 + 未覆盖区域。
- **基线不能故意写坏**。本例中贪心已经能到；你要解释的是为什么学到的模型反而更差，或你换了一张贪心会失败的地图。
- **数据切分必须按 episode**。按步切分会把同一段轨迹拆进 train / test。
- **模型必须读动作**。`EmpiricalDynamics` 的键是 `(state, action)`，不要改成只键状态。
- **未知转移会停在原地**。这是代码里的显式设计，不是随机 bug。把它写进失败分析。
- **`lookahead` 在平局时按动作名字顺序取第一条**。depth=3 从 `(0, 0)` 出发 64 条序列回报全是 `-3` 时，它会稳定地先走 `down`。
- **失败必须稳定**。不能只挑一张坏图。

## 扩展练习

完成基本要求后，按从便宜到昂贵的顺序推荐：

1. **换模型**：把表格换成线性模型或小 MLP，观察四类评价的变化——更复杂的模型是否真的更好？
2. **增加数据量**：把训练从 5 段提到 50 段再提到 200 段，画出覆盖率曲线。5 段是 36%，50 段是 73%，200 段是 91%。
3. **换 Planner**：把穷举换成随机采样或 CEM，观察下游成功率——Planner 的选择是否改变结论？
4. **加不确定性**：对计数很少的 `(s, a)` 输出更扁的分布。走到未覆盖区域时，方差应变大。这能否帮你检测失败？

## 选路线

完成本节以后，回答四个问题：

```text
我最希望模型交出什么？
谁会使用这个结果？
哪些信息必须保留？
哪些信息可以暂时不预测？
```

根据答案进入第 4–8 章中的一条路线：

- 如果答案是「latent 空间中的规划」→ 决策与规划路线（第 4 章，Dreamer）
- 如果答案是「可观看的视频」→ 互动视频路线（第 5 章，VQ-VAE + Transformer）
- 如果答案是「任务相关的特征」→ JEPA 路线（第 6 章，Video-JEPA）
- 如果答案是「机器人动作」→ 机器人与 VLA 路线（第 7 章，Tiny VLA）
- 如果答案是「三维空间预测」→ 空间世界路线（第 8 章）

**不要默认所有人继续学习 Dreamer。** 本节的意义是让你亲手发现「我的模型缺什么」，然后根据缺口选择路线。

## 本节小结

- **本节是整门课的分水岭**：从人工写的转移表到自行设计的 learned dynamics。
- **一次只选一种变化**：让一种失败稳定出现，而不是同时制造多个问题。
- **接口图和数据卡是基础**：如果别人看了不能复现，说明它们不够清楚。
- **基线不能故意写坏**：本例中贪心回报是 `3.0`，复制上一状态只有 `0.215`。
- **四类评价缺一不可**：one-step `1.000` 与 depth=3 成功率 `0 / 20` 可以同时成立。
- **失败诊断比修好失败更重要**：说清楚失败在哪里、为什么会出现，比盲目尝试修复更有价值。
- **下一步只改一件事**：不要画出组合拳架构图。

从 1.6 的九格世界到本节的自设计世界，核心问题从未改变：**在行动之前，先在内部预见行动的后果**。本节让你第一次面对「预见不准」或「预见准了也用不好」——观察不够、数据不足、horizon 不够、Planner 在平局里走偏。这些缺口不是 bug，是研究的起点。

## 后续工作

本节只要求你看清一个缺口。后面的路线把它放大：

Dyna 把「用模型生成想象经验」写成完整算法，每一步真实交互后都在表格里再做几步规划更新。World Models 把同一件事做到像素上：V 压缩、M 想象、C 在梦里进化。Dreamer 再把无梯度进化换成可微的 Actor-Critic。你在本节里亲手碰到的覆盖空洞、复合误差、模型被 Planner 钻空子，都会在那些更大的模型里再次出现。

若你已经决定走决策与规划路线，下一份实验是 [4.6](/chapters/04-latent-dynamics/07-rssm-scratch)，收尾是 [4.7 动手：想象训练的简洁实现](/chapters/04-latent-dynamics/08-dreamer-concise)。

## 参考文献

1. Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press. [链接](http://incompleteideas.net/book/the-book.html) —— 第 9 章讲 Dyna 与规划；本节的计数动态和 MPC 闭环是它的最小课堂版。
2. Sutton, R. S. (1991). Dyna, an Integrated Architecture for Learning, Planning, and Reacting. _ACM SIGART Bulletin_, 2(4), 160–163. [链接](https://doi.org/10.1145/122344.122377) —— 用学到的模型做想象更新的原始架构。
3. Ha, D., & Schmidhuber, J. (2018). Recurrent World Models Facilitate Policy Evolution. _NeurIPS 2018_. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) —— 在想象中训练策略；本节之后各条路线都从这里分叉。
4. Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and Acting in Partially Observable Stochastic Domains. _Artificial Intelligence_, 101(1–2), 99–134. [链接](<https://doi.org/10.1016/S0004-3702(98)00023-X>) —— 若你选了部分可观测，信念状态比「再加一层网络」更先要讲清。
5. Talvitie, E. (2014). Model Regularization for Stable Sample Rollouts. _UAI 2014_. [arXiv:1406.2315](https://arxiv.org/abs/1406.2315) —— 多步 rollout 的复合误差，以及为什么 one-step 准不等于长程能用。
