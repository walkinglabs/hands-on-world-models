# 1.1　动手：九格世界的从零实现

> **第 1 章 · 引言**
>
> 人类在行动之前，总会在脑海中预演未来：“如果我这样走，会发生什么？”世界模型（World Model）正是赋予机器这种内在想象与推演能力的核心机制。本章不急着讲理论，而是先用 [1.1 动手：九格世界的从零实现](/chapters/01-why-world-models/01-invent-a-world-model) 跑通一个最小闭环——只用 Python 标准库，训练并使用一台最简单的世界模型。之后 [1.2 观察与预测](/chapters/01-why-world-models/02-observation-and-prediction) 解释它为什么必须这样设计，[1.3 定义与判据](/chapters/01-why-world-models/03-what-is-a-world-model) 给出可检验的判据，最后 [1.4 经典世界模型](/chapters/01-why-world-models/04-classic-world-models) 对照 V-M-C、PlaNet、Dreamer 与 MuZero。
> **本节目标**：不依赖任何深度学习框架，只用 Python 标准库、整数和字典，把世界模型的最小闭环写出来。你会亲手实现一个九格网格世界、一个表格动态模型、一个规划器，亲眼看到「预测→规划→执行→修正」的完整循环。

> **本节代码**：[本节 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/01_reinvent/invent-a-world-model.ipynb) · [gridworld.py](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/gridworld.py)

> **前置知识**：无。这是课程第一份实验，不需要神经网络、不需要 PyTorch、不需要 GPU。只需要 Python 基础。

---

2018 年，David Ha 与 Jürgen Schmidhuber 在 NeurIPS 发表了论文 _Recurrent World Models Facilitate Policy Evolution_ [1]，并配套了一篇交互式文章 [World Models](https://worldmodels.github.io/) [2]。文章里有一个赛车 demo：一辆小车在赛道上飞驰，画面模糊但动作流畅。作者说，这不是录屏，是模型在「做梦」——一个 867 个参数的线性控制器，完全在想象中学会了开车。

你当时大概和我一样，第一反应是：「这怎么可能？867 个参数？线性控制器？在梦里学的？」

但在你理解那 867 个参数之前，你需要先理解一个更基本的问题：**什么是世界模型？**

不是数学定义，不是论文摘要，而是最原始的直觉：**一个能预测「如果我这样做，世界会变成什么样」的东西**。

## 先玩为敬：浏览器里的第一台世界模型

在写任何代码之前，先在下面这个像素小世界里当一次「环境」。用方向键移动小人去右下角的旗帜，避开中间的陷阱。

右边那块画布是**模型想象**：它是一台刚开始一片空白的表格世界模型，唯一的本领是数数——把你经历过的每个「格子 + 方向 → 下一格」记下来，再据此预测你下一步会到哪。刚开始它满屏问号（一无所知），你走得越多，它预测得越准。**这就是本节接下来要从零手写的那个模型，只是先让你玩到它。**

<PlayWorldModel />

玩的时候注意三件事：

1. **问号阶段**：模型对没见过的「状态-动作」只能回答「不知道」——这就是后面要讲的**数据覆盖**问题；
2. **撞墙**：在边界按方向键，小人原地不动。模型见过几次后也会预测「原地不动」——它不知道什么是墙，但它学到了墙的效果；
3. **陷阱**：故意掉进去一次，再重置。你会发现模型已经「记住」了那个坑——**经历，而不是规则，是它全部的知识来源**。

这一节，我们要从零发明这样一个东西。不用神经网络，不用梯度下降，只用整数和字典。跑完之后，你会对「世界模型」这四个字有完全不同的理解。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/nine-grid.png" alt="九格世界" style="max-width:min(400px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">这就是我们要让模型学会的「世界」：3×3 的网格，中间是陷阱（红色），右下角是目标（绿色）。智能体从左上角出发，要学会「向哪个方向走会到哪里」。模型从未见过物理定律，它要从转移数据里自己发现这些规律。</div>
</div>

## 本次会得到什么

运行结束后，你会得到：

- 一张九格地图和智能体的初始位置
- 没有模型时的贪心路径（会走进陷阱）
- 有模型后的规划路径（绕开陷阱）
- 深度为 1、3、6 时的规划结果对比
- 同一起点只替换动作的反事实比较
- 从轨迹中学习得到的概率转移表
- 模型预测与真实执行不一致时的修正记录
- MPC 的执行轨迹（只执行第一步，重新观察）

## 怎样运行

仓库中的 Notebook 位于：

```text
notebooks/01_reinvent/invent-a-world-model.ipynb
```

安装 Jupyter 后，在仓库根目录运行：

```bash
jupyter lab
```

网格世界的可复用实现位于 `src/hwm/gridworld.py`。即使暂时不运行 Notebook，也可以先执行单元测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 第一步：九格世界

1.6 的世界是一个 3×3 的网格。智能体在某个格子，可以上下左右移动。目标是右下角，陷阱在中间。

```python
# 3x3 网格，0=空地，1=陷阱，2=目标
world = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 2],
]

# 智能体初始位置
agent_pos = (0, 0)
goal_pos = (2, 2)
trap_pos = (1, 1)

# 动作空间
actions = ['up', 'down', 'left', 'right']
```

**运行这一步，你会看到什么？** Notebook 会打印网格地图和智能体初始位置：

```
Grid World (3x3):
. . .
. T .
. . G

Agent at: (0, 0)
Goal at: (2, 2)
Trap at: (1, 1)
```

这个世界的「物理定律」很简单：向上走 y 减 1，向下走 y 加 1，向左走 x 减 1，向右走 x 加 1。但不能走出边界，走进陷阱就失败。

## 第二步：没有模型的贪心

如果智能体只能看到当前位置，它会选择离目标最近的邻居。这就是贪心策略：

```python
def greedy_action(position, goal):
    """没有模型时，只能贪心选择离目标最近的方向"""
    x, y = position
    gx, gy = goal

    # 计算四个邻居到目标的距离
    candidates = []
    if y > 0: candidates.append(('up', (x, y-1)))
    if y < 2: candidates.append(('down', (x, y+1)))
    if x > 0: candidates.append(('left', (x-1, y)))
    if x < 2: candidates.append(('right', (x+1, y)))

    # 选择离目标最近的
    best = min(candidates, key=lambda c: abs(c[1][0]-gx) + abs(c[1][1]-gy))
    return best[0]
```

**运行这一步，你会看到什么？** Notebook 会打印贪心路径：

```
Greedy path (no model):
Step 0: (0,0) → right → (1,0)
Step 1: (1,0) → down → (1,1)  ← 走进陷阱！
Failed at step 1
```

你会发现，贪心策略会走进陷阱——因为它没有「如果走进陷阱会怎样」的预测能力。它只看到「向下走离目标更近」，但不知道「向下走会掉进陷阱」。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/greedy-vs-planned.png" alt="贪心 vs 规划" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">左：没有模型时的贪心策略——只看眼前，走进陷阱（红色 X）。右：有模型后的规划策略——预见未来，绕开陷阱，到达目标（绿色星）。</div>
</div>

**这就是没有世界模型的代价**：你只能根据当前观察做决策，无法预见未来。

## 第三步：加入一步转移模型

现在给智能体一个表格动态模型：`P(下一位置 | 当前位置, 动作)`。这个模型是人工写的，不是学来的：

```python
# 人工编写的转移表（确定性）
transition = {
    ((0,0), 'right'): (1,0),
    ((0,0), 'down'): (0,1),
    ((1,0), 'right'): (2,0),
    ((1,0), 'down'): (1,1),  # 陷阱！
    ((0,1), 'right'): (1,1),  # 陷阱！
    ((0,1), 'down'): (0,2),
    # ... 其他转移
}

def plan_one_step(current, goal, transition):
    """使用转移表做一步规划"""
    best_action = None
    best_next = None
    best_dist = float('inf')

    for action in actions:
        next_pos = transition.get((current, action))
        if next_pos and next_pos != (1,1):  # 避开陷阱
            dist = abs(next_pos[0]-goal[0]) + abs(next_pos[1]-goal[1])
            if dist < best_dist:
                best_dist = dist
                best_action = action
                best_next = next_pos

    return best_action, best_next
```

有了模型，智能体可以模拟「如果我向右走，会到哪里」「如果我向下走，会到哪里」。它会避开陷阱。

**运行这一步，你会看到什么？** Notebook 会打印新的规划路径：

```
Planned path (with model):
Step 0: (0,0) → right → (1,0)  [predicted: (1,0)]
Step 1: (1,0) → right → (2,0)  [predicted: (2,0)]
Step 2: (2,0) → down → (2,1)   [predicted: (2,1)]
Step 3: (2,1) → down → (2,2)   [predicted: (2,2)]
Reached goal in 4 steps!
```

智能体绕开了陷阱——**这就是世界模型的价值**。它能在行动之前预见后果，避开危险。

**一个值得做的实验**：把陷阱位置从 (1,1) 改到 (1,0)，观察贪心策略和规划策略的行为变化。贪心策略现在能成功了（因为 (1,1) 不再是陷阱），但规划策略会绕路——因为它知道原来的陷阱位置现在安全了，但它仍然保守地避开。

## 第四步：多步预测与复合误差

一步预测很准，但多步预测呢？如果每一步有 10% 的概率出错，六步之后正确的概率只有 \(0.9^6 \approx 53\%\)。

```python
def plan_multi_step(current, goal, transition, horizon=3):
    """多步规划：模拟未来多步，选择最优序列"""
    from itertools import product

    best_sequence = None
    best_final_dist = float('inf')

    # 枚举所有可能的动作序列
    for seq in product(actions, repeat=horizon):
        pos = current
        valid = True
        for action in seq:
            next_pos = transition.get((pos, action))
            if not next_pos or next_pos == (1,1):  # 走进陷阱或无效
                valid = False
                break
            pos = next_pos

        if valid:
            dist = abs(pos[0]-goal[0]) + abs(pos[1]-goal[1])
            if dist < best_final_dist:
                best_final_dist = dist
                best_sequence = seq

    return best_sequence
```

**运行这一步，你会看到什么？** Notebook 会比较深度为 1、3、6 的规划结果：

```
Horizon=1:  (0,0) → right → (1,0)
Horizon=3:  (0,0) → right, right, down → (2,1)
Horizon=6:  (0,0) → right, right, down, down, right, right → (2,2) [goal]

但 horizon=6 的预测可靠性只有 53%...
```

你会发现，horizon 越长，规划越不靠谱——**这就是复合误差**。每一步的小误差会累积，最终让长程预测变得不可信。

这也是为什么后面的 Dreamer 和 MuZero 都要花大力气解决「如何让长程预测更准」的问题。

## 第五步：从数据里学习动态

现在把人工写的转移表拿走。智能体需要从自己的轨迹里学习 `P(下一位置 | 当前位置, 动作)`：

```python
# 收集轨迹
trajectory = []
for episode in range(10):
    pos = (0, 0)
    for step in range(10):
        action = random.choice(actions)
        next_pos = true_transition(pos, action)  # 真实环境
        trajectory.append((pos, action, next_pos))
        pos = next_pos
        if pos == (2, 2): break

# 从轨迹计数学习转移概率
counts = {}
for (s, a, s_next) in trajectory:
    counts[(s, a, s_next)] = counts.get((s, a, s_next), 0) + 1

# 归一化得到概率
learned_transition = {}
for (s, a, s_next), count in counts.items():
    if (s, a) not in learned_transition:
        learned_transition[(s, a)] = {}
    total = sum(c for (ss, aa, sn), c in counts.items() if ss == s and aa == a)
    learned_transition[(s, a)][s_next] = count / total
```

**运行这一步，你会看到什么？** Notebook 会打印学习到的转移表：

```
Learned transition probabilities:
((0,0), 'right'): {(1,0): 1.0}
((0,0), 'down'): {(0,1): 1.0}
((1,0), 'right'): {(2,0): 1.0}
((1,0), 'down'): {(1,1): 1.0}  ← 学会了这是陷阱
...
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/learning-from-data.png" alt="从数据学习" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">从轨迹数据学习转移模型的过程：左——多条轨迹在网格中穿行；中——统计每个 (状态, 动作) 对的转移次数；右——归一化得到转移概率。数据足够时，学到的模型和真实规律一致。</div>
</div>

如果数据足够，它和人工写的转移表一致。但如果某些 (状态, 动作) 对没有出现在训练数据中，模型就不知道会发生什么——**这就是数据覆盖的问题**。

**一个值得做的实验**：只收集 5 段轨迹（而不是 10 段），观察学习到的转移表是否完整。你会发现某些转移概率是 0 或缺失——因为数据不够。

## 第六步：MPC——只执行第一步并重新观察

模型永远不可能完美。如果智能体完全相信模型，它会沿着模型预测的最优路径走到底。但如果模型在某一步出错了，后续所有规划都白费。

MPC（Model Predictive Control）的解决方案是：**只执行第一步，然后重新观察真实世界，重新规划**。

MPC 的数学形式是一个滚动优化问题：

$$
a_t^* = \arg\min_{a_{t:t+H}} \sum_{k=0}^{H} c\bigl(\hat{s}_{t+k},\, s_{\text{goal}}\bigr), \quad \text{s.t. } \hat{s}_{t+k+1} = f(\hat{s}_{t+k}, a_{t+k})
$$

其中 \(f\) 是学到的世界模型，\(c\) 是代价函数（如曼哈顿距离），\(H\) 是规划 horizon。关键：只执行 \(a_t^*\)，然后在 \(t+1\) 重新观察真实状态并重新优化。

```python
def mpc_loop(start, goal, transition, max_steps=20):
    """MPC 循环：每一步都重新规划"""
    pos = start
    trajectory = [pos]

    for step in range(max_steps):
        if pos == goal:
            break

        # 用模型规划（但只执行第一步）
        plan = plan_multi_step(pos, goal, transition, horizon=3)
        if not plan:
            break

        first_action = plan[0]

        # 在真实环境中执行
        next_pos = true_transition(pos, first_action)

        # 检查模型预测是否准确
        predicted_next = transition.get((pos, first_action))
        if predicted_next != next_pos:
            print(f"Model error at step {step}: predicted {predicted_next}, got {next_pos}")

        pos = next_pos
        trajectory.append(pos)

    return trajectory
```

**运行这一步，你会看到什么？** Notebook 会打印 MPC 的执行记录：

```
MPC execution:
Step 0: at (0,0), plan [right, right, down], execute right → (1,0) ✓
Step 1: at (1,0), plan [right, down, down], execute right → (2,0) ✓
Step 2: at (2,0), plan [down, down], execute down → (2,1) ✓
Step 3: at (2,1), plan [down], execute down → (2,2) ✓
Reached goal!
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/mpc-process.png" alt="MPC 过程" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">MPC 的四步过程：每一步都从当前位置重新规划（虚线），但只执行第一步（实线箭头）。这样即使模型有误差，也能用真实反馈不断修正，避免长程预测的累积误差。</div>
</div>

你会发现，即使模型有误差，MPC 仍然能到达目标——因为它不断用真实观测修正预测。每一步都重新规划，所以不会让误差累积。

**这就是 MPC 的核心思想**：不信任长程预测，只执行最确定的一步，然后用真实反馈修正。

## 已知简化与坑

教学版有几处刻意的简化，跑不通时先从这里找原因：

- **世界过于简单**。3×3 网格、4 个动作、1 个陷阱——这不是 Atari。真实世界的状态空间是连续的，动作是无限的。
- **转移是确定性的**。真实世界通常有随机性：同样的动作可能得到不同的结果。教学版假设世界是完全确定的。
- **转移表是完整的**。如果某些 (状态, 动作) 对没有出现在训练数据中，模型就不知道会发生什么。真实世界需要泛化能力。
- **没有观测噪声**。智能体完美知道自己的位置。真实世界的观测通常有噪声。
- **没有奖励函数**。我们用「到达目标」作为成功标准。真实世界需要更复杂的奖励设计。

## 扩展练习

跑通默认配置后，按从便宜到昂贵的顺序推荐：

1. **扩大世界**：把九格世界改成 5×5，陷阱数量从 1 个提到 5 个，观察贪心策略与 MPC 的成功率差异。陷阱越多，模型的价值越大。
2. **加入随机性**：让转移有 10% 的概率失败（比如想向右走，但有 10% 概率停在原地），观察 MPC 的表现。
3. **部分可观测**：让智能体只能看到周围 1 格的范围内，观察它需要怎样的记忆机制。
4. **连续状态**：把网格世界改成连续空间（坐标是浮点数），观察表格模型为什么不适用，需要什么替代方案。

## 本节小结

- **世界模型的最小闭环**：预测→规划→执行→修正，四步缺一不可。
- **没有模型的贪心策略会走进陷阱**——因为它没有「如果这样做会怎样」的预测能力。
- **一步预测很准，但多步预测有复合误差**——horizon 越长，规划越不靠谱。
- **动态模型可以从数据里学习**——计数转移、归一化概率。但数据覆盖是关键。
- **MPC 只执行第一步并重新观察**——用真实观测修正模型误差，避免长程预测的累积误差。

从九格网格到 CarRacing 赛车，从表格动态到 RSSM，从贪心策略到 Dreamer 的 Actor-Critic——世界模型的核心思想从未改变：**在行动之前，先在内部预见行动的后果**。而这一节，你亲手把这个思想从零发明出来。

## 后续工作

1.6 用最简单的方式展示了世界模型的核心思想。但真实世界的世界模型要复杂得多：

**Dyna-Q** [1] 是最经典的模型强化学习算法之一。它在每一步真实交互后，用学到的模型生成多步「想象」经验，加速学习。我们的 1.6 已经实现了这个思想的核心。

**World Models** [2] 把这个世界模型扩展到了像素级。用 VAE 压缩观测，用 MDN-RNN 预测未来，用进化算法在梦境中训练控制器。867 个参数的控制器完全在想象中学习，却在 CarRacing 赛道上表现优异。

**Dreamer** [3] 进一步改进：用 RSSM（Recurrent State-Space Model）替代 MDN-RNN，用 Actor-Critic 替代进化算法，让梯度直接在梦境中反向传播。这成为后续 DreamerV2、DreamerV3 的基础。

**MuZero** [4] 走得更远：它甚至不要求模型重建观测，只预测奖励和策略。这让 MuZero 在 Atari、Go、国际象棋、将棋上都达到超人水平。

这些方法的核心思想都来自 1.6：**先预测，再规划，再执行，再修正**。只是它们用神经网络替代了表格，用梯度下降替代了枚举，用像素替代了网格。

## 参考文献

1. Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press. [链接](http://incompleteideas.net/book/the-book.html) —— 经典 RL 教材，第 9 章讲 Dyna 与规划。
2. Ha, D., & Schmidhuber, J. (2018). Recurrent World Models Facilitate Policy Evolution. _NeurIPS 2018_. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) —— World Models：867 个参数在梦境中学会开车。
3. Hafner, D., et al. (2019). Dream to Control: Learning Behaviors by Latent Imagination. _ICLR 2020_. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) —— DreamerV1：RSSM + imagination + Actor-Critic 的原始版本。
4. Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. _Nature_. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) —— MuZero：用 learned model + MCTS 打通棋类和 Atari。
5. Tassa, Y., et al. (2012). Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization. _IROS 2012_. [链接](https://doi.org/10.1109/IROS.2012.6386025) —— MPC 在机器人控制中的经典应用。
