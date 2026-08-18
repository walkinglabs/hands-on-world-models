# PA0 · 动手：重新发明一台可学习世界模型

> **本节目标**：不再使用人工写的转移表。你将面对一个「当前观察不够用」的小世界，自行决定什么状态值得保存、模型输出一个结果还是分布、Planner 怎样使用它。最终交出一份完整的证据链：接口图、数据卡、基线、learned dynamics、四类评价、一组稳定失败、一项最小改动。

> **本节代码**：[PA0 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/assignments/PA0-template.ipynb)

> **前置知识**：你已经跑过 F0（九格世界 + 表格动态）和 F3（LineWorld 计数动态），知道世界模型的最小闭环——预测→规划→执行→修正。PA0 不再给你转移表，你需要自己设计。

---

F0 给了你一个 3×3 的网格世界，转移表是人工写的。你只需要查表、规划、执行。一切都很干净。

F3 给了你一条线形世界，动态是确定性的计数。你从轨迹里数出转移概率，发现「学到的」和「真实的」可以完全一致。

但真实世界不会给你一张完美的表格。你会遇到这些情况：

- **当前观察看不到完整状态**——你需要记住过去两步才能推断位置；
- **同一个动作在不同地面上结果不同**——但地面类型暂时不可见；
- **目标在 episode 开始时改变**——你的模型需要对「目标在哪」敏感；
- **地图里有一块训练数据没覆盖的区域**——你的模型走到那里时会暴露无知。

PA0 的任务是：**选择其中一种变化，让一种失败稳定出现，然后用数据、接口、模型和反例把它讲清楚。**

不是造最复杂的环境，不是画最漂亮的架构图。是让一个缺口变得可见。

## 为什么 PA0 是整门课的分水岭

F0 和 F3 的世界模型是「玩具」——状态完全可观测，动态完全确定，数据完全覆盖。PA0 第一次打破这三个假设中的至少一个。

打破之后，你会发现 F0 的那套方法不够用了：

- 转移表不能处理**部分可观测**——你需要某种记忆或历史拼接；
- 计数转移不能处理**随机动态**——你需要输出分布而不是单一结果；
- 贪心规划不能处理**未见区域**——你需要某种不确定性估计或回退策略。

PA0 不要求你用神经网络。表格、线性模型、小 MLP 都可以。它要求的是：**你能把一个现实缺口变成数据、接口、模型、反例和下一步设计。** 这是做研究的最小完整循环。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/pa0-world-loop.png" alt="世界模型学习循环" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">PA0 真实结果：在 5×5 GridWorld 中收集轨迹（左）、用 EmpiricalDynamics 学习转移概率（中）、MPC 闭环规划路径（右，蓝点为路径，绿圈为目标）。实际运行 hwm.gridworld 模块。</div>
</div>

## 第一步：选择一种变化

从下面四种变化中选一种，**一次只选一种**：

```text
1. 部分可观测：当前观察看不到完整位置，需要最近两步历史；
2. 隐式条件：同一个动作在两种地面上结果不同，但地面类型暂时不可见；
3. 目标漂移：目标会在 episode 开始时改变；
4. 未覆盖区域：地图中有一个训练数据未覆盖的区域。
```

修改 F3 的 LineWorld 或自己构造一个类似的小世界。环境必须足够小，能在 CPU 上 5 分钟内完成全部实验。

参考代码框架（用 `hwm.gridworld` 的 `EmpiricalDynamics` + MPC）：

```python
from hwm.gridworld import GridWorld, EmpiricalDynamics, ACTIONS
import numpy as np

# 1. 收集数据
env = GridWorld(rows=5, cols=5, goal=(4, 4), traps=[(1, 1), (3, 2)])
trajectories = []
for ep in range(50):
    state = env.reset(rng=np.random.default_rng(ep))
    for step in range(20):
        action = np.random.choice(list(ACTIONS.keys()))
        transition = env.step(state, action, rng=np.random.default_rng())
        trajectories.append((state, action, transition.next_state))
        state = transition.next_state

# 2. 学习世界模型
model = EmpiricalDynamics()
for (s, a, s_next) in trajectories:
    model.update(s, a, s_next)

# 3. MPC 闭环规划
def predict_fn(start, actions):
    s = start
    for a in actions:
        dist = model.distribution(s, a)
        if dist:
            s = max(dist, key=dist.get)
    return s

# 4. 搜索最优动作序列
best_actions = None
best_cost = float('inf')
for _ in range(100):  # 100 次随机采样
    candidate = np.random.randint(0, 4, size=5)
    predicted = predict_fn(current_state, candidate)
    cost = abs(predicted[0] - goal[0]) + abs(predicted[1] - goal[1])
    if cost < best_cost:
        best_cost = cost
        best_actions = candidate

# 5. 执行第一步，观察结果
action = list(ACTIONS.keys())[best_actions[0]]
transition = env.step(current_state, action, rng=np.random.default_rng())
```

**运行这一步，你会看到什么？** 你应该能画出你的小世界地图，标出智能体的起始位置、目标位置和变化的类型。

**一个值得做的实验**：先让你的变化「几乎不造成影响」——比如部分可观测时，给智能体一个很大的感受野。确认基线能通过。然后逐步收紧条件，直到基线稳定失败。**PA0 的目的不是制造最复杂环境，而是让一种失败稳定出现。**

## 第二步：画出接口图

不用论文架构图。用你自己的变量名，写清数据怎样流动：

```text
模型输入 → 内部状态或预测 → 模型输出
Planner 怎样调用 → 环境怎样返回真实结果
```

这张图必须回答四个问题：

1. **模型的输入是什么？** 当前观察？最近两步观察的拼接？动作的 one-hot 编码？
2. **模型的输出是什么？** 一个确定的下一状态？一个概率分布？均值和方差？
3. **Planner 怎样使用模型？** beam search？随机采样？前向模拟？
4. **环境怎样返回真实结果？** 真实转移与模型预测在哪里比较？

**运行这一步，你会看到什么？** 一张手绘接口图。如果别人看了这张图，能直接复现你的数据流，说明它够清楚。

## 第三步：写数据卡

PA0 不给你数据集。你需要自己收集。数据卡至少写明：

```text
- 数据怎样生成（什么策略、多少 episode、每段多长）；
- transition 字段和 shape（obs, action, next_obs, reward, done）；
- action 对应哪两次观察（t 还是 t+1？）；
- episode 在哪里结束（到达目标？步数上限？碰撞？）；
- train、val、test 按什么边界切分（按 episode 还是按步？）；
- seed 与数据量。
```

**一个关键陷阱**：train/val/test 的切分必须按 episode 边界，不能按步。如果同一段 episode 的前半段在 train、后半段在 test，你的模型会「看到未来」——这就是**数据泄漏**。

**运行这一步，你会看到什么？** 一份数据卡，写明每个字段的 shape 和含义。如果有人问你「你的模型训练时看到了测试数据吗？」，你能用这份数据卡证明没有。

## 第四步：实现简单基线

先实现一种不学习或很少学习的办法：

- **复制上一状态**：永远预测 `next_obs = current_obs`；
- **永远向目标移动**：不管动态如何，总是选择离目标最近的方向；
- **只使用当前观察**：不用历史，不用记忆，贪心选择。

基线在简单情况中应该能够成功。**不要把它故意写坏。** 如果基线在你的环境中完全失败，说明环境设计有问题——PA0 要的是「基线能过，但 learned dynamics 更好」。

**运行这一步，你会看到什么？** 基线的成功率或 return。如果基线成功率是 0%，收紧环境；如果是 100%，learned dynamics 没有提升空间。

## 第五步：实现 learned dynamics

实现至少一种从数据学习的转移模型。表格、线性模型或小神经网络都可以，但需要说明**为什么当前数据需要它**。

关键要求：**模型必须把动作作为输入或明确的条件。** 如果模型不读动作，它只能预测「世界自己会怎样变化」，不能预测「如果我这样做，世界会变成什么样」。

$$
\hat{s}_{t+1} = f_\theta(s_t, a_t)
$$

对于部分可观测的情况，状态可能需要拼接历史：

$$
\hat{s}_{t+1} = f_\theta([s_t; s_{t-1}; \ldots], a_t)
$$

**运行这一步，你会看到什么？** 训练 loss 曲线。如果 loss 下降到基线以下，说明模型学到了某种动态规律。但 loss 下降不等于模型有用——下一步要验证。

## 第六步：四类评价

同时报告四种评价，缺一不可：

**1. One-step 指标**

模型预测的下一状态与真实下一状态的距离。MSE、准确率或 IoU，取决于你的状态表示。

**2. 多步 rollout（5、10、20 步）**

从同一起点出发，让模型连续预测多步。观察误差怎样累积——这就是 F0 教你的**复合误差**。

**3. 反事实**

同一起点，只替换动作，观察模型预测是否随之改变。如果换动作后预测不变，模型没有学到动作条件动态。

**4. 下游任务**

使用模型前后的任务成功率或 return。模型预测好不等于决策好——Planner 可能利用模型的漏洞。

**运行这一步，你会看到什么？** 四组数字。如果 one-step 好但多步差，说明复合误差严重；如果多步好但反事实失败，说明模型不读动作；如果反事实好但下游差，说明 Planner 有问题。

## 第七步：找到一组稳定失败

提交至少一组在固定 seed 下可重复的失败。失败必须稳定出现——不能只挑一张坏图。

然后分析失败更可能来自哪里：

```text
- 数据不足？训练数据没有覆盖失败区域；
- 观察不足？当前观察不包含决策所需的全部信息；
- 模型能力不足？模型结构无法表达真实动态；
- 规划使用不当？Planner 过度信任模型的错误预测。
```

至少给出两种可能的解释。PA0 不要求你修好这个失败——它要求你**说清楚失败在哪里、为什么会出现**。

**运行这一步，你会看到什么？** 一条失败的轨迹或一组失败的预测。你能指出「从这里开始，模型走偏了」，并解释为什么。

## 第八步：写下一步需求

最后只写一项最小改动。不要直接画出「CNN + Transformer + Diffusion + MCTS」的组合图。

问自己：**如果只能改一件事，改什么能让这组失败消失？**

- 补速度状态？
- 加入动作时间？
- 缩短 horizon？
- 增加失败区域的数据？
- 加入 OOD 惩罚？

这一项改动，就是你进入下一章路线的起点。

## 运行与产物

PA0 必须可以在 CPU 上完成，运行时间目标不超过 5 分钟。

```bash
jupyter lab
# 打开 notebooks/assignments/PA0-reinvent-a-learned-world.ipynb
```

跑完后，你应该有：

- **接口图**：用自己的变量名画的数据流图
- **数据卡**：字段、shape、切分方式、seed
- **基线结果**：简单策略的成功率或 return
- **Learned dynamics**：loss 曲线、模型结构说明
- **四类评价**：one-step、多步、反事实、下游
- **稳定失败**：可重复的失败案例 + 至少两种解释
- **下一步需求**：一项最小改动

## 评分

| 项目 | 分数 | 检查重点 |
| ---- | ---: | -------- |
| 问题与接口 | 15 | 不靠模型名也能说清缺口 |
| 数据与切分 | 20 | 时间对齐、episode 边界、无泄漏 |
| 基线与模型 | 20 | 基线合理，模型确实读取动作 |
| 评价证据 | 20 | 一步、多步、反事实、下游齐全 |
| 失败诊断 | 15 | 失败稳定，解释至少有两种可能 |
| 表达与复现 | 10 | Notebook 可运行，seed 与输出完整 |

## 已知简化与坑

教学版有几处刻意的约束，提交前先从这里检查：

- **环境必须足够小**。PA0 的目标是让一种失败稳定出现，不是造一个复杂世界。如果你的环境需要 GPU 或超过 5 分钟，说明设计太复杂了。
- **一次只选一种变化**。不要同时让部分可观测 + 目标漂移 + 未覆盖区域。一次一种，让失败清晰可解释。
- **基线不能故意写坏**。如果基线在你的环境中完全失败，说明环境设计有问题。基线应该能过，但 learned dynamics 应该更好。
- **数据切分必须按 episode**。按步切分会导致数据泄漏——同一段 episode 的前半段在 train、后半段在 test。
- **模型必须读动作**。如果模型不读动作，它只能预测「世界自己会怎样变化」，不能用于规划。
- **失败必须稳定**。不能只挑一张坏图。固定 seed 下必须能重复出现。

## 扩展练习

完成基本要求后，按从便宜到昂贵的顺序推荐：

1. **换模型**：把表格模型换成线性模型或小 MLP，观察四类评价的变化——更复杂的模型是否真的更好？
2. **增加数据量**：把训练数据从 10 段 episode 提到 50 段，观察多步 rollout 的改善——数据多少开始「够用了」？
3. **换 Planner**：把 beam search 换成随机采样或 CEM，观察下游成功率的变化——Planner 的选择是否影响结论？
4. **加不确定性**：让模型输出均值和方差，而不是只输出均值。当模型走到未覆盖区域时，方差应该变大——这能否帮助你检测失败？

## 选路线

完成 PA0 以后，回答四个问题：

```text
我最希望模型交出什么？
谁会使用这个结果？
哪些信息必须保留？
哪些信息可以暂时不预测？
```

根据答案进入第 3–7 章中的一条路线：

- 如果答案是「latent 空间中的规划」→ 路线 A（Dreamer）
- 如果答案是「可观看的视频」→ 路线 B（VQ-VAE + Transformer）
- 如果答案是「任务相关的特征」→ 路线 C（Video-JEPA）
- 如果答案是「机器人动作」→ 路线 D（Tiny VLA）
- 如果答案是「三维空间预测」→ 路线 E（空间世界）

**不要默认所有人继续学习 Dreamer。** PA0 的意义是让你亲手发现「我的模型缺什么」，然后根据缺口选择路线。

## 本节小结

- **PA0 是整门课的分水岭**：从人工写的转移表到自行设计的 learned dynamics，从完全可观测到部分可观测或条件隐式。
- **一次只选一种变化**：让一种失败稳定出现，而不是同时制造多个问题。
- **接口图和数据卡是基础**：如果别人看了你的图和数据卡不能复现，说明它们不够清楚。
- **基线不能故意写坏**：基线应该能过，但 learned dynamics 应该更好。
- **四类评价缺一不可**：one-step、多步、反事实、下游——每一种暴露不同的问题。
- **失败诊断比修好失败更重要**：说清楚失败在哪里、为什么会出现，比盲目尝试修复更有价值。
- **下一步只改一件事**：不要画出「CNN + Transformer + Diffusion + MCTS」的组合图。

从 F0 的九格世界到 PA0 的自设计世界，世界模型的核心问题从未改变：**在行动之前，先在内部预见行动的后果**。但 PA0 让你第一次面对「预见不准」的现实——观察不够、数据不足、模型能力有限。这些缺口不是 bug，是研究的起点。

## 参考文献

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [链接](http://incompleteideas.net/book/the-book.html) —— 经典 RL 教材，第 8 章讲 Dyna 与规划。
2. Ha, D., & Schmidhuber, J. (2018). Recurrent World Models Facilitate Policy Evolution. *NeurIPS 2018*. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) —— World Models：在想象中训练策略的开创性工作。
3. Talvitie, E. (2014). Model Regularization for Stable Sample Rollouts. *UAI 2014*. [链接](https://arxiv.org/abs/1406.2315) —— 模型正则化：多步 rollout 的复合误差问题。
4. Jafferjee, I., et al. (2020). Model-based Reinforcement Learning for Biological Systems. *NeurIPS 2020 Workshop*. —— 部分可观测环境下的世界模型设计。
5. Gregor, K., et al. (2015). DRAW: A Recurrent Neural Network for Image Generation. *ICML 2015*. [arXiv:1502.04623](https://arxiv.org/abs/1502.04623) —— 序列生成中的注意力机制，对部分可观测问题有启发。
