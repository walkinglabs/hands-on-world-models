# 9.4　动手：设计新的世界模型

> **本节目标**：不是提出宏大新架构，而是完成一次可以被推翻的小型研究循环。从你在 3.6 或所选路线最后一份动手里发现的稳定失败出发，提出两种竞争解释，做一个最小改动，设计能证明自己错的实验，公平对照，最后提交「下一台模型」短报告。
>
> **本节代码**：[本节 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/projects/next-model-template.ipynb)
>
> **前置知识**：你已经跑过 [3.6 动手：重新发明一台可学习世界模型](/chapters/03-data-and-first-model/06-learnable-world)、所选路线最后一份动手，以及 [9.2 动手：世界模型的系统评测](/chapters/09-evaluate-and-invent/02-systematic-evaluation)。本节从那里找到的稳定失效出发，只改一件事。9.3 讲「为什么要这样设计实验」；本节是要交的东西。

---

3.6 让你亲手发现「我的模型缺什么」，所选路线最后一份动手让你训练一台完整的世界模型、亲眼看到它在哪里崩溃，9.2 再用六项测试把失败写成可复现的条件。

现在你有一组稳定失败。本节的任务是：**完成一次可以被推翻的小型研究循环**。

不是提出「CNN + Transformer + Diffusion + MCTS」的宏大新架构。是问一个更小的问题：**如果只改一件事，能不能让这组失败消失？**

## 为什么本节是整门课的终点

3.6 到各条路线的动手让你理解世界模型「是什么」和「怎么做」，9.2 让你学会「怎样审问」。本节让你做研究的最小完整循环：

```text
发现失败 → 提出解释 → 最小改动 → 设计证伪实验 → 公平对照 → 写报告
```

这个循环的核心不是「证明我对了」，而是「设计能证明自己错的实验」。如果你只能设计支持自己的指标，那不是研究，是自我安慰。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/research-cycle.png" alt="研究循环" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">左边是那个稳定失败，中间是两种可能的解释，右边是只改一件事之后还在不在。负结果也算数——只要你真的能把自己推翻。</div>
</div>

## 本次会得到什么

这是整门课的收尾。提交时必须交出：

- 一条可复现失败：命令、数据、checkpoint 哈希、seed 集合、量化条件、至少 3 次重复
- 两种竞争解释，以及它们分别预测的不同实验结果
- 一个最小改动：只动一个主要因素，附 diff 或对照实现
- 实验前写下的证伪标准：哪种数字组合让你放弃这个设计
- 公平对照表：同一 split、同一 seed 集合、同一更新数或计算预算；均值、标准差、失败样例、额外资源
- 3–5 页短报告，按下面六个问题组织
- Run Manifest：环境、耗时、显存、checkpoint SHA256
- 3–5 篇你真正读过、和这次改动直接相关的论文

模板在 `notebooks/projects/next-model-template.ipynb`。单元格里的 `assert True` 是占位，不是完成标志。

## 怎样运行

```bash
jupyter lab
# 打开 notebooks/projects/next-model-template.ipynb
```

对照实验应复用你自己的训练脚本和 9.2 的评价函数，不要另写一套「专属指标」。

## 第一步：固定一个稳定失败

从 3.6、所选路线最后一份动手、9.2 里选一个**稳定出现**的失败。给出复现命令、数据、seed 和量化条件。

失败必须至少重复三次，或在一组样本上稳定出现，不能只挑一张坏图。

坏的写法：「模型长视频预测效果不好。」

好的写法：「物体被连续遮挡超过 3 步、场景里同时有 3 个以上移动物体时，位置 MSE 从 0.02 升到 0.3 以上；seed=42/43/44 三次都出现；失败样例见 `fig/long_occlusion_failure.png`。」

```python
# 伪代码：把失败钉死在命令和数字上
results = []
for seed in (42, 43, 44):
    metrics = replay_failure(command, seed=seed)
    results.append(metrics["target_mse"])
print(results)
assert all(x > 0.3 for x in results)
```

**运行这一步，你会看到什么？** 一条别人能复现的失败轨迹，以及三次重复的数字。如果三次里只有一次坏，先回去补评测，别开始改模型。

## 第二步：提出两种竞争解释

从数据、状态、动态、训练目标、规划器与系统接口里选。两种解释必须预测**不同**的实验结果。

```text
解释 A（观察不足）：当前观察不含速度，遮挡后位置不可辨。
  预测：把速度补进状态，长遮挡 MSE 会降，短遮挡几乎不变。

解释 B（容量不足）：线性动力学拟合不了加速。
  预测：换成 MLP，所有运动场景一起降；只补速度则几乎不动。
```

单变量消融的形状是

$$
\Delta y = f(x_1', x_2, \ldots) - f(x_1, x_2, \ldots)
$$

如果两种解释预测同一个 \(\Delta y\)，你无法区分它们。第 9.3 节把「四种结果怎么判」写成了图；本页要求你在跑实验之前把那张表填进报告。

## 第三步：做一个最小改动

只改一个主要因素。例如：补速度状态、加入动作时间、缩短 horizon、改变掩码、增加碰撞失败数据、加入 OOD 惩罚。

```python
import numpy as np

def ablation_study(model_factory, data, seeds=(0, 1, 2)):
    """对比基线 vs 改动，多种子重复。"""
    results = {"baseline": [], "changed": []}
    for seed in seeds:
        base = model_factory(add_velocity=False)
        base.fit(data, seed=seed)
        results["baseline"].append(base.evaluate())

        changed = model_factory(add_velocity=True)
        changed.fit(data, seed=seed)
        results["changed"].append(changed.evaluate())

    for name, scores in results.items():
        print(f"{name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    return results
```

**不要同时改多个因素。** 同时补速度、换 MLP、缩短 horizon，你无法知道哪个有效。

**运行这一步，你会看到什么？** 一份只动一处的 diff，以及改动前后能跑通的同一套评价。成功率上升、不变或下降都是合法结果。

## 第四步：设计能证明自己错的实验

提前写下：**若出现什么结果，就说明新设计没有解决原问题。** 不得只写支持自己的指标。

```text
若目标场景提升 < 5%，且在误差范围内 → 新设计没帮上忙
若目标场景升了，但所有简单场景也升了同一点 → 可能只是训得更久
若多步 rollout 误差变大 → 新设计有害
若 OOD 没有改善 → 不能声称泛化
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/z86-falsify.png" alt="四种结果怎么判" style="max-width:min(880px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">跑实验之前先写下这四格。目标场景升、其它不变，才支持你选的解释；全线一起升一点，更像预算或运气。</div>
</div>

如果你答不出「什么结果会让我放弃这个设计」，先不要开训。

## 第五步：公平对照

固定数据 split、seed 集合、更新数或计算预算。报告平均、波动、失败样例与额外资源。

一次 seed 的提升不算数。至少 3 个 seed。参数量或时间变了，必须写进表，不能把「多训了一倍」说成方法赢了。

评价函数必须和 9.2 用同一份。给新模型写专属指标，该项零分。

## 第六步：提交「下一台模型」短报告

按六个问题写，3–5 页：

1. **它解决哪种稳定失败？** 用第一步的命令和数字复述。
2. **新输入、状态、预测目标或规划接口是什么？** 只写最小变化。
3. **为什么更简单的方法不够？** 加数据、调学习率、缩短 horizon、规则补丁，你试过哪几个。
4. **结果支持了什么？** 和事前预测是否一致。
5. **结果没有支持什么？** 哪些场景没升、哪些更差、哪个解释被否了。
6. **下一次最值得收集的证据是什么？** 只写一项。

附代码 diff、Run Manifest、主对比曲线、3–5 个失败样例对比、3–5 篇直接相关论文。

**负结果可以获得完整成绩，前提是问题、对照与边界写清楚。**

## 评分

| 项目       | 分数 | 检查重点                                       |
| ---------- | ---: | ---------------------------------------------- |
| 稳定失败   |   15 | 可复现；三次以上；有量化条件                   |
| 竞争解释   |   15 | 至少两种；预测的实验结果不同                   |
| 最小改动   |   20 | 只改一个因素；代码真改了；能跑                 |
| 证伪标准   |   15 | 实验前写好；含「怎样算自己错了」               |
| 公平对照   |   20 | 同 split / seed / 预算；均值和标准差；失败样例 |
| 报告与复现 |   15 | 六个问题答全；manifest；不越界声称             |

只报提升、藏下降、对照不公平、说不清为什么有效——哪怕主指标涨了，总分不超过 60。负结果且对照干净，可以拿满分。

## 已知简化与坑

- **不要提出宏大新架构。** 本节的目标是走完研究循环，不是发明 DreamerV4。
- **一次只改一个因素。** 同时改多个，分数记在「最小改动」上。
- **必须提前写下证伪标准。** 事后挑好看的曲线，证伪项零分。
- **公平对照不可妥协。** 换了数据还换了预算，就不能把差异算在那一个组件上。
- **负结果也是好结果。** 解释错了、改动有害，只要写清楚，同样完成循环。
- **不要声称解决了你没有解决的问题。** 只在训练分布内有效，就不要写 OOD 泛化。

## 扩展练习

1. **换失败**：修好「观察不足」之后，再试「规划器使用不当」。
2. **换解释**：两种解释都是数据向的，就补一个结构向或目标向的。
3. **换改动**：加状态无效，就改训练目标或改 Planner。
4. **换评价**：除了成功率，再报样本效率或计算效率。

这些加分，但不替代上面六项。

## 本节小结

- **本节是整门课的终点**：从发现失败到完成一次研究循环，不是提出宏大新架构。
- **稳定失败是研究的起点**：必须可复现，不能只挑一张坏图。
- **两种竞争解释必须能预测不同结果**，否则你无法区分它们。
- **最小改动只改一个因素**；同时改多个，你不知道哪个有效。
- **证伪标准必须提前写下**。
- **公平对照不可妥协**：固定数据、seed、计算预算。
- **负结果也是好结果**：问题、对照与边界写清楚，同样有价值。

从 1.6 的九格世界到本节的自设计研究，从人工写的转移表到「设计能证明自己错的实验」——你走完了世界模型的最小完整循环：理解→实现→审问→改进。这不是终点，是研究的起点。

## 参考文献

1. Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press. [链接](http://incompleteideas.net/book/the-book.html) —— 第 9 章讲 Dyna 与规划，是「先有模型再改模型」的老家。
2. Talvitie, E. (2014). Model Regularization for Stable Sample Rollouts. _UAI 2014_. [arXiv:1406.2315](https://arxiv.org/abs/1406.2315) —— 多步 rollout 的复合误差，为什么「看起来一步很准」不够。
3. Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). When to Trust Your Model: Model-Based Policy Optimization. _NeurIPS 2019_. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253) —— 模型只在可信区间里用，是很多「最小改动」的原型。
4. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. _arXiv:2301.04104_. [链接](https://arxiv.org/abs/2301.04104) —— DreamerV3：一套超参打通 150+ 任务，对照实验该有多干净。
5. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. _OpenReview_. [链接](https://openreview.net/pdf?id=BZ5a1r-kVsf) —— JEPA 提案：预测特征而不是像素，常被选作「改预测目标」的那一项最小改动。
