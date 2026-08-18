# PA2 · 动手：设计下一台世界模型

> **本节目标**：不是提出宏大新架构，而是完成一次可以被推翻的小型研究循环。从你在 PA0 或 PA1 中发现的稳定失败出发，提出两种竞争解释，做一个最小改动，设计能证明自己错的实验，公平对照，最后提交「下一台模型」短报告。

> **本节代码**：[PA2 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/assignments/PA2-next-model-template.ipynb)

> **前置知识**：你已经跑过 PA0（自设计世界模型）和 PA1（某条路线的完整训练），发现了至少一组稳定失败。PA2 从这些失败出发，尝试修好其中一个。

---

PA0 让你亲手发现「我的模型缺什么」。PA1 让你亲手训练一台完整的世界模型，亲眼看到它在哪里崩溃。Z0 让你用六项测试审问模型，找出它的失败模式。

现在，你有一组稳定失败。PA2 的任务是：**完成一次可以被推翻的小型研究循环**。

不是提出「CNN + Transformer + Diffusion + MCTS」的宏大新架构。是问一个更小的问题：**如果只改一件事，能不能让这组失败消失？**

## 为什么 PA2 是整门课的终点

PA0–PA1 让你理解世界模型「是什么」和「怎么做」。Z0 让你学会「怎样审问」。PA2 让你做研究的最小完整循环：

```text
发现失败 → 提出解释 → 最小改动 → 设计证伪实验 → 公平对照 → 写报告
```

这个循环的核心不是「证明我对了」，而是「设计能证明自己错的实验」。如果你只能设计支持自己的指标，那不是研究，是自我安慰。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/pa2-research-cycle.png" alt="研究循环" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">PA2 真实结果：左——原始模型（hidden=16）稳定失败的损失曲线；中——两种竞争解释（容量不足 vs 数据不足）；右——改进模型（hidden=64）损失下降 + 多步误差对照（红:原始 vs 绿:改进）。实际运行 hwm.control + hwm.evaluation 模块。</div>
</div>

## 第一步：固定一个稳定失败

从你在 PA0 或 PA1 中发现的失败里，选一个**稳定出现**的。给出复现命令、数据、seed 和量化条件。

```text
Stable failure:
  reproduction command: python run_experiment.py --seed 42
  data: 50 episodes, PixelWorld
  seed: 42
  failure condition: success rate < 0.3 after 1000 training steps
  
  failure example:
    episode 23: agent reaches step 15, then loops forever
    predicted next state: (3, 5)
    actual next state: (3, 7)
    error: 2 cells
```

失败必须至少重复三次或在一组样本上稳定出现，不能只挑一张坏图。

**运行这一步，你会看到什么？** 一条可复现的失败轨迹。如果有人运行你的命令，应该能看到同样的失败。

## 第二步：提出两种竞争解释

从数据、状态、动态、训练目标、规划器与系统接口中选择。说明两种解释会预测什么不同实验结果。

```text
Explanation 1: observation insufficient
  hypothesis: current observation doesn't contain velocity information
  prediction: adding velocity to state will improve success rate by 20%
  
Explanation 2: model capacity insufficient  
  hypothesis: linear model cannot capture non-linear dynamics
  prediction: switching to MLP will improve success rate by 15%
  
Discriminating experiment:
  if adding velocity helps but MLP doesn't → Explanation 1 is correct
  if MLP helps but adding velocity doesn't → Explanation 2 is correct
  if both help → both are factors
  if neither helps → both explanations are wrong
```

**关键**：两种解释必须能预测不同的实验结果。如果两种解释预测同样的结果，你无法区分它们。

**运行这一步，你会看到什么？** 两种解释和它们的预测。如果有人问你「这个实验能区分两种解释吗？」，你能用预测回答。

## 第三步：做一个最小改动

只改一个主要因素。例如：

- 补速度状态
- 加入动作时间
- 缩短 horizon
- 改变掩码
- 增加碰撞失败数据
- 加入 OOD 惩罚

**单变量消融**的数学原则：固定其他变量 \(x_2, x_3, \ldots\)，只改变 \(x_1\)，观察输出变化：

$$
\Delta y = f(x_1', x_2, \ldots) - f(x_1, x_2, \ldots)
$$

如果 \(\Delta y\) 显著，说明 \(x_1\) 确实是因果因素；否则它可能只是相关性。

参考代码框架（单变量消融）：

```python
import numpy as np

def ablation_study(model_factory, data, seeds=[0, 1, 2]):
    """对比基线 vs 改动，多种子重复"""
    results = {'baseline': [], 'changed': []}
    
    for seed in seeds:
        # 基线
        model_base = model_factory(add_velocity=False)
        model_base.fit(data, seed=seed)
        results['baseline'].append(model_base.evaluate())
        
        # 改动：只加 velocity
        model_changed = model_factory(add_velocity=True)
        model_changed.fit(data, seed=seed)
        results['changed'].append(model_changed.evaluate())
    
    # 报告均值和标准差
    for name, scores in results.items():
        print(f'{name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}')
    
    return results
```

```text
Minimal change:
  original: state = (position,)
  changed: state = (position, velocity)
  
  rationale: Explanation 1 predicts observation insufficient
  scope: only add velocity, keep everything else the same
```

**不要同时改多个因素。** 如果你同时补速度、换 MLP、缩短 horizon，你无法知道哪个改动有效。

**运行这一步，你会看到什么？** 改动前后的对比。如果成功率上升，说明改动有效；如果不变或下降，说明改动无效或解释错误。

## 第四步：设计能证明自己错的实验

提前写下：**若出现什么结果，就说明新设计没有解决原问题。** 不得只写支持自己的指标。

```text
Falsification criteria:
  if success rate improvement < 5% → new design didn't help
  if multi-step rollout error increases → new design made things worse
  if OOD performance doesn't improve → new design doesn't generalize
  
  must report:
    - success rate (support)
    - multi-step rollout error (could go either way)
    - OOD performance (could go either way)
    - failure cases (must not increase)
```

**关键**：你必须提前写下「什么结果会证明我错了」。如果你只能写支持自己的指标，那不是研究。

**运行这一步，你会看到什么？** 一份证伪标准。如果有人问你「什么结果会让你放弃这个设计？」，你能用这份标准回答。

## 第五步：公平对照

固定数据 split、seed 集合、更新数或计算预算。报告平均结果、波动、失败样例与额外资源。

```text
Fair comparison:
  fixed:
    - data split: same 50 episodes
    - seed set: {42, 123, 456}
    - training steps: 1000
    - compute budget: same
    
  reported:
    - mean success rate: 0.45 → 0.52 (+15%)
    - std: 0.05 → 0.04
    - failure cases: 12 → 8
    - extra resources: +10% training time
```

**不能同时更换模型、数据和训练预算，再把差异归因于某一个组件。**

**运行这一步，你会看到什么？** 对照表格。如果改进显著，说明设计有效；如果改进在误差范围内，说明设计无效。

## 第六步：提交「下一台模型」短报告

回答六个问题：

**1. 它解决哪种稳定失败？**

```text
Failure: agent loops forever when velocity is ambiguous
```

**2. 新输入、状态、预测目标或规划接口是什么？**

```text
Change: add velocity to state observation
```

**3. 为什么更简单的方法不够？**

```text
Baseline (copy last state) cannot predict motion direction
Linear model cannot capture acceleration dynamics
```

**4. 结果支持了什么？**

```text
Supported:
  - success rate improved 15%
  - multi-step rollout error decreased 10%
  - failure cases decreased 33%
```

**5. 结果没有支持什么？**

```text
Not supported:
  - OOD performance didn't improve (novel obstacle layouts)
  - training time increased 10%
  - cannot claim general solution to partial observability
```

**6. 下一次最值得收集的证据是什么？**

```text
Next evidence:
  - test on novel obstacle layouts (OOD generalization)
  - ablation: velocity from ground truth vs predicted
  - longer training to see if improvement persists
```

附代码、运行清单、曲线、checkpoint 哈希和 3–5 篇直接相关论文。

**负结果可以获得完整成绩，前提是问题、对照与边界写清楚。**

## 运行与产物

```bash
jupyter lab
# 打开 notebooks/assignments/PA2-next-model-template.ipynb
```

跑完后，你应该有：

- **稳定失败的复现命令**：别人能运行并看到同样的失败
- **两种竞争解释**：能预测不同实验结果
- **最小改动**：只改一个因素
- **证伪标准**：什么结果会让你放弃这个设计
- **公平对照**：固定数据、seed、计算预算
- **短报告**：六个问题的回答

## 已知简化与坑

- **不要提出宏大新架构**。PA2 的目标是完成一次研究循环，不是发明 DreamerV4。
- **一次只改一个因素**。同时改多个因素，你无法知道哪个有效。
- **必须提前写下证伪标准**。如果你只能写支持自己的指标，那不是研究。
- **公平对照不可妥协**。固定数据 split、seed 集合、更新数或计算预算。
- **负结果也是好结果**。如果最小改动没有效果，说清楚为什么，这同样有价值。
- **不要声称解决了你没有解决的问题**。如果只在训练分布内有效，不要声称解决了 OOD 泛化。

## 扩展练习

完成基本要求后，按从便宜到昂贵的顺序推荐：

1. **换失败**：如果你修的失败是「观察不足」，试试修「模型能力不足」或「规划器使用不当」——不同失败需要不同改动。
2. **换解释**：如果你提出的两种解释都是「数据相关」，试试提出「模型结构相关」或「训练目标相关」的解释。
3. **换改动**：如果你做的最小改动是「加状态」，试试「改训练目标」或「改规划器」——不同改动暴露不同的问题。
4. **换评价**：如果你只报告了成功率，试试报告「样本效率」「计算效率」或「泛化能力」——不同评价暴露不同的优劣。

## 本节小结

- **PA2 是整门课的终点**：从发现失败到完成一次研究循环，不是提出宏大新架构。
- **稳定失败是研究的起点**：必须可复现，不能只挑一张坏图。
- **两种竞争解释必须能预测不同结果**：否则你无法区分它们。
- **最小改动只改一个因素**：同时改多个因素，你无法知道哪个有效。
- **证伪标准必须提前写下**：什么结果会让你放弃这个设计。
- **公平对照不可妥协**：固定数据、seed、计算预算。
- **负结果也是好结果**：问题、对照与边界写清楚，同样有价值。

从 F0 的九格世界到 PA2 的自设计研究，从人工写的转移表到「设计能证明自己错的实验」——你走完了世界模型的最小完整循环：理解→实现→审问→改进。这不是终点，是研究的起点。

## 参考文献

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [链接](http://incompleteideas.net/book/the-book.html) —— 经典 RL 教材，第 8 章讲 Dyna 与规划。
2. Talvitie, E. (2014). Model Regularization for Stable Sample Rollouts. *UAI 2014*. [链接](https://arxiv.org/abs/1406.2315) —— 模型正则化：多步 rollout 的复合误差问题。
3. Jafferjee, I., et al. (2020). Model-based Reinforcement Learning for Biological Systems. *NeurIPS 2020 Workshop*. —— 部分可观测环境下的世界模型设计。
4. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. *arXiv:2301.04104*. [链接](https://arxiv.org/abs/2301.04104) —— DreamerV3：一套超参打通 150+ 任务，可复现世界模型的工程标杆。
5. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview*. [链接](https://openreview.net/pdf?id=BZ5a1r-kVsf) —— JEPA 的原始提案：预测特征而非像素。
