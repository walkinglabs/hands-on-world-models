# 9.2　动手：世界模型的系统评测

> **本节目标**：不增加新网络，系统检查多步 horizon、反事实动作、单变量 OOD、不确定性校准和 Planner 漏洞。完成后把同一套评测迁移到自己的路线模型。

> **本节代码**：[本节 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/09_evaluation/test-a-world-model.ipynb) · [evaluation.py](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/evaluation.py)

> **前置知识**：你已经跑过至少一条路线，有一台训练好的世界模型。9.2 不训练新模型，只评测你已经有的模型。最好刚读完 [4.6 动手：World Models 的复现](/chapters/04-decision-and-planning/06-reproduce-world-models)——那里已经出现过「梦境分数高不等于真实分数高」。

---

你花了几小时（甚至几天）训练了一台世界模型。loss 下降了，reconstruction 看起来不错，token accuracy 也挺高。

但你的模型真的「理解」了世界吗？还是只是记住了训练数据的统计规律？

**审问（interrogate）世界模型**，就是设计一系列测试，暴露模型的失败模式。不是证明它有多好，而是找出它在哪里会出错。

这一节，我们要审问一台世界模型。六项测试，覆盖多步预测、反事实推理、分布外泛化、不确定性估计、Planner 漏洞和复现规范。跑完之后，你会对「我的模型到底学到了什么」有完全不同的理解。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/interrogation.png" alt="六项审问测试" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">审问世界模型的六项测试。数字来自本节同一套解析 toy：horizon 1/5/12 的 MSE 是 0.01 / 0.17 / 0.96，反事实差异是 0 / 1.8 / 1.8，OOD 误差从 0.1 跳到 0.9，Planner 选出 a = 3.25。不是证明模型有多好，而是找出它在哪里会出错。</div>
</div>

## 这就是要审问的东西

9.2 不用赛车、不用 PixelWorld。它用一个故意有偏差、可以完全算清楚的一维世界：

$$
x_{t+1} = x_t + a_t
\qquad\text{（真实）}
$$

$$
\hat x_{t+1} = \hat x_t + 0.9\, a_t
\qquad\text{（模型）}
$$

真实世界每次动作完整移动；模型只预测 0.9 倍位移。一步误差是 \(0.1^2 = 0.01\)，几乎看不见。多步以后，模型会逐渐落后。

三条测试轨迹从 \(x\in\{0, 1, -1\}\) 出发，分别一直向右、一直向左、左右交替，各滚 12 步。评价代码不训练任何网络，只对这个解析 toy 提问。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/toy-world.png" alt="一维有偏世界" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">从 \(x=0\) 一直向右走 12 步。真实世界走到 12.0，模型只走到 10.8。一步几乎看不出，多步才会拉开——这就是后面所有测试要审问的对象。</div>
</div>

这个玩具足够小，每一步的对错你都能手算。它也足够坏：一步看不出问题，规划器却会主动钻空子。把同一套问题换成你自己训练的模型，接口不用改。

## 本次会得到什么

运行结束后，你会得到：

- 一条 12 步的 horizon 曲线：MSE 从 0.01 走到 0.96
- 同一起点上三条反事实轨迹，相对停留的差异是 0 / 1.8 / 1.8
- 一个事先写死的 OOD 条件：动作从 1 改成 3，绝对误差从 0.1 跳到 0.9
- 三个校准箱子：低箱偏高估，高箱略低估
- Planner 选出的动作 \(a=3.25\)，落在训练支持 \([-1, 1]\) 外面
- 一份可复现的评测记录，字段规范见[附录 B](/appendices/data-compute-delivery)

## 怎样运行

仓库中的 Notebook 位于：

```text
notebooks/09_evaluation/test-a-world-model.ipynb
```

安装 Jupyter 后，在仓库根目录运行：

```bash
jupyter lab
```

跨路线共用的评价函数位于 `src/hwm/evaluation.py`。即使暂时不运行 Notebook，也可以先执行单元测试：

```bash
PYTHONPATH=src python -m unittest tests.test_evaluation -v
```

9.2 使用解析 toy，CPU 即可。完成后把同一套检查迁移到自己的路线模型。

## 测试一：多步 horizon 曲线

**问题**：模型的一步预测很准，但多步预测呢？

一步误差相当于考试时老师把前 99 步答案都给你了，只让你算第 100 步。真正部署时，模型只能把自己上一步的输出当成下一步的输入。开环滚动是：

$$
\hat s_1 = f_\theta(s_0, a_0),\quad
\hat s_2 = f_\theta(\hat s_1, a_1),\quad
\ldots,\quad
\hat s_H = f_\theta(\hat s_{H-1}, a_{H-1})
$$

评价函数不把多步藏进一个平均数，而是对每个 horizon 单独报误差：

$$
\mathrm{err}(h)
= \frac{1}{N}\sum_{n=1}^{N}
\bigl\| \hat s^{(n)}_h - s^{(n)}_h \bigr\|^2,
\qquad h = 1,\ldots,H
$$

这正是 `horizon_errors` 在做的事：先对特征维求均方，再对样本求平均，留下一条长度为 \(H\) 的曲线。

```python
import numpy as np

def rollout(start, actions, scale=1.0):
    result, state = [], float(start)
    for action in actions:
        state += scale * action
        result.append(state)
    return np.asarray(result)

def horizon_errors(predict, starts, action_sequences, true_rollouts):
    """每个 horizon 一个数，不把多步藏进平均数。"""
    true_rollouts = np.asarray(true_rollouts, dtype=np.float32)
    predictions = np.stack(
        [predict(start, actions) for start, actions in zip(starts, action_sequences)]
    ).astype(np.float32)
    axes = tuple(range(2, predictions.ndim))
    squared = (predictions - true_rollouts) ** 2
    per_step = squared.mean(axis=axes) if axes else squared
    return per_step.mean(axis=0)

starts = [0.0, 1.0, -1.0]
action_sequences = [[1.0] * 12, [-1.0] * 12, [1.0, -1.0] * 6]
truth = np.stack([rollout(s, a, 1.0) for s, a in zip(starts, action_sequences)])
errors = horizon_errors(lambda s, a: rollout(s, a, 0.9), starts, action_sequences, truth)
print('horizon 1/5/12 MSE:', [round(float(errors[i]), 4) for i in [0, 4, 11]])
```

**运行这一步，你会看到什么？**

```
horizon 1/5/12 MSE: [0.01, 0.17, 0.96]
```

完整曲线是

```
0.01, 0.027, 0.063, 0.107, 0.17, 0.24, 0.33, 0.427, 0.543, 0.667, 0.81, 0.96
```

horizon 越长，误差越大——**这就是复合误差**。第 1 步几乎看不见，第 12 步已经差了两个数量级。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/horizon-errors.png" alt="多步 horizon 曲线" style="max-width:min(720px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">三条轨迹平均后的 horizon 曲线。h = 1 / 5 / 12 的 MSE 分别是 0.01 / 0.17 / 0.96。一步还像及格，十二步已经接近 1.0。</div>
</div>

曲线是平均。拆开看，三条轨迹并不一样：

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/horizon-process.png" alt="三条轨迹各自的复合误差" style="max-width:min(830px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">从各自起点滚 12 步。一直向右、一直向左，第 12 步本条 MSE 都是 1.44（真实 12 对模型 10.8，或 −11 对 −9.8）；左右交替在第 12 步碰巧重合，本条误差是 0。三条平均才是 0.96。</div>
</div>

**判断**：如果曲线在 10 步后急剧上升，说明模型的多步一致性很差，不能拿来做长 horizon 规划。如果某条动作序列的误差始终接近 0，先检查它是不是在对消，而不是模型突然变准了。

**一个值得做的实验**：再加一条「复制上一状态」的基线，\(\hat s_{t+1}=s_t\)。一步误差会很难看，但曲线不会继续涨。你的模型必须在某个 horizon 窗口里同时打过复制基线和匀速外推，才算真的学到了动态。

## 测试二：固定起点的反事实动作

**问题**：模型真的理解动作因果关系吗？还是只是记住了「通常会发生什么」？

反事实的意思很朴素：**锁死整个世界，只改动作**。同一起点、同一历史、同一随机源，分别喂三条动作，看未来是否分开：

$$
\text{同一起点 } s_0 \Rightarrow
\begin{cases}
a^{(0)}=[0,0,0] & \Rightarrow \hat s^{(0)} \\
a^{(1)}=[1,1,1] & \Rightarrow \hat s^{(1)} \\
a^{(2)}=[-1,-1,-1] & \Rightarrow \hat s^{(2)}
\end{cases}
$$

敏感度以第一条轨迹为参照：

$$
\Delta_i
= \mathrm{mean}\bigl| \hat s^{(i)} - \hat s^{(0)} \bigr|
$$

这正是 `counterfactual_sensitivity`：对每条动作序列做一次预测，减去参照，再对剩余维取平均绝对值。\(\Delta=0\) 是危险信号——换动作，未来完全没动。

```python
def counterfactual_sensitivity(predict, start, action_sequences):
    """固定起点，只换动作；结果完全相同是危险信号。"""
    outputs = np.stack([predict(start, actions) for actions in action_sequences])
    reference = outputs[0]
    axes = tuple(range(1, outputs.ndim))
    return np.mean(np.abs(outputs - reference), axis=axes)

candidate_actions = [[0, 0, 0], [1, 1, 1], [-1, -1, -1]]
sensitivity = counterfactual_sensitivity(
    lambda s, a: rollout(s, a, 0.9), 0.0, candidate_actions
)
print('相对 stay 的未来差异:', np.round(sensitivity, 3))
```

**运行这一步，你会看到什么？**

```
相对 stay 的未来差异: [0.  1.8 1.8]
```

向右三条：\(0.9, 1.8, 2.7\)；向左三条：\(-0.9, -1.8, -2.7\)。相对停留，平均绝对差恰好都是 1.8。方向对，幅度对——这个有偏模型至少还在听动作。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/counterfactual.png" alt="反事实轨迹" style="max-width:min(780px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">同一起点 \(x=0\)，只换动作。灰线停留不动，蓝线向右，红线向左。相对停留的差异是 0 / 1.8 / 1.8。如果三条线叠在一起，模型没有学到动作条件动态。</div>
</div>

**判断**：\(\Delta_i>0\) 只说明「换动作后轨迹变了」，还要看方向是否和动作一致。视频世界里不要要求像素逐点相同，比较的是结果的分布。

**一个值得做的实验**：把模型改成完全忽略动作，\(\hat x_{t+1}=\hat x_t + 0.9\)。敏感度会变成什么？三条轨迹平行平移，相对差异不再反映动作方向——这就是「只在抄惯性」。

### 动作影响度：把敏感度变成一个数

只看"换动作后轨迹变了"还不够，还要问"变了多少是动作的功劳"。**动作影响度**（action influence）把反事实敏感度收成一个标量：对所有候选动作的 $\Delta_i$ 取平均，再和"换初始状态"的敏感度做比。

$$
I_{\text{action}}=\frac{\overline{\Delta_{\text{action}}}}{\overline{\Delta_{\text{state}}}}
$$

分子是换动作带来的预测差异，分母是换起点带来的预测差异。比值接近 $0$，说明模型主要在抄惯性，动作几乎不影响预测；比值接近 $1$，说明动作和起点一样重要，模型真的把动作当回事。learn-world-model 的 P06 把这个指标当成反事实世界模型的核心评测——它比单看 $\Delta_i$ 更难自欺，因为分母是一个参照系。

## 测试三：单变量 OOD

**问题**：模型在训练分布内表现良好，但在分布外呢？

神经网络有个坏毛病：哪怕输入完全没见过，它也会自信满满地吐出一个数字。OOD 测试就是故意给它看训练时没见过的东西。条件必须**事先写死**，不能等跑完再挑那个错得最离谱的样例当证据。

本节只改一个变量：动作幅度。训练支持是 \(|a|\le 1\)。分布内用 \(a=1\)、模型尺度 0.9；分布外用 \(a=3\)，并且故意把模型尺度掉到 0.7——模拟「没见过的输入上，偏差还会变大」。

$$
e_{\mathrm{ID}} = \bigl| 0.9\cdot 1 - 1 \bigr| = 0.1,
\qquad
e_{\mathrm{OOD}} = \bigl| 0.7\cdot 3 - 3 \bigr| = 0.9
$$

```python
in_distribution = abs(rollout(0, [1], 0.9)[0] - rollout(0, [1], 1.0)[0])
ood = abs(rollout(0, [3], 0.7)[0] - rollout(0, [3], 1.0)[0])
print('ID/OOD absolute error:', round(float(in_distribution), 3), round(float(ood), 3))
```

**运行这一步，你会看到什么？**

```
ID/OOD absolute error: 0.1 0.9
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/ood.png" alt="单变量 OOD" style="max-width:min(820px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">左边是训练内：\(a=1\)，误差 0.1。右边是事先写死的 OOD：\(a=3\)，误差 0.9。模型两边都给出了「下一步位置」，但右边那个数字已经不能拿来规划。</div>
</div>

误差跳了 9 倍。更麻烦的是：模型仍然给出一个看起来合理的 2.1，并不会抛异常、不会说「我没见过」。这就是泛化边界——也是下一节校准要抓的东西。

**判断**：OOD 误差明显高于分布内，本身不是事故，是信号。真正危险的是误差升高了，模型的不确定性却没有升高。

**一个值得做的实验**：把 OOD 动作扫一遍 \(\{1.5, 2, 3, 5\}\)，画误差对 \(|a|\) 的曲线。边界通常不是一刀切，而是从训练支持边缘开始慢慢坏掉。

## 测试四：不确定性校准

**问题**：模型说「我有 90% 的把握」时，真的 90% 正确吗？

校准（calibration）问的不是准不准，而是诚实不诚实。把预测置信度 \(p\) 切成若干箱子，每个箱子里算两个数：平均置信度 \(\bar p_b\)，真实事件频率 \(\bar f_b\)。

$$
\bar p_b = \mathbb{E}\bigl[p \mid p\in B_b\bigr],
\qquad
\bar f_b = \mathbb{E}\bigl[y \mid p\in B_b\bigr]
$$

`calibration_bins` 按 \([0,1]\) 均匀切箱，返回每个非空箱子的 `count`、`confidence`、`frequency`。理想情况下点落在对角线上；点在线下方，是过度自信。

```python
def calibration_bins(probabilities, outcomes, num_bins=5):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    outcomes = np.asarray(outcomes, dtype=np.float32)
    edges = np.linspace(0, 1, num_bins + 1)
    result = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        include = (probabilities >= lower) & (
            probabilities <= upper if upper == 1 else probabilities < upper
        )
        if include.any():
            result.append({
                "lower": float(lower),
                "upper": float(upper),
                "count": int(include.sum()),
                "confidence": float(probabilities[include].mean()),
                "frequency": float(outcomes[include].mean()),
            })
    return result

probabilities = np.array([0.1, 0.2, 0.35, 0.65, 0.8, 0.95])
outcomes = np.array([0, 0, 1, 0, 1, 1])
bins = calibration_bins(probabilities, outcomes, num_bins=3)
for item in bins:
    print(item)
```

**运行这一步，你会看到什么？**

```
{'lower': 0.0, 'upper': 0.333, 'count': 2, 'confidence': 0.15, 'frequency': 0.0}
{'lower': 0.333, 'upper': 0.667, 'count': 2, 'confidence': 0.50, 'frequency': 0.50}
{'lower': 0.667, 'upper': 1.0, 'count': 2, 'confidence': 0.875, 'frequency': 1.0}
```

低箱：模型平均说 15%，实际一次都没发生。中箱刚好校准。高箱：模型平均说 87.5%，实际全中，略低估。三个箱子的平均 \(|\bar p-\bar f|\) 大约 0.09。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/calibration.png" alt="校准曲线" style="max-width:min(720px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">三个箱子，各两个样本。灰线是完美校准。低箱落在线下方（过度自信），高箱略偏上。六个点只能示范分箱算法，不能给校准下结论。</div>
</div>

Guo 等人在 _On Calibration of Modern Neural Networks_ 里指出：现代网络常常过度自信，softmax 的最大概率并不能直接当可靠度用。世界模型的方差、ensemble 分歧、扩散多样性，都只是不确定性的线索；**没经过分箱，就不算校准**。

**判断**：没有不确定性输出的模型，这一步可以跳过，但报告里必须写明：「我的模型不提供不确定性估计，无法判断什么时候它不可靠。」

**一个值得做的实验**：把样本从 6 个加到 200 个，再画一次。箱子里的点数不够时，\(\bar f_b\) 会在 0 和 1 之间乱跳，看起来像校准很差，其实是估计方差太大。

## 测试五：Planner 漏洞

**问题**：Planner 会不会利用模型的漏洞，找到一条只在模型里畅通的路线？

普通测试随机采样动作。Planner 不一样，它在解

$$
a^\star = \arg\max_a \; R\bigl(f_\theta(s, a)\bigr)
$$

会**主动**走到模型最乐观、也往往最不可靠的地方。这就是模型利用（model exploitation）：Ha 与 Schmidhuber 用温度 \(\tau\) 做过对照——梦境太确定时，控制器会过拟合那些只在想象里成立的捷径；Talvitie 则从复合误差的角度说明，开环 rollout 会把自己送进从未见过的区域。

本节的玩具 Planner 以为「下一步应该靠近 3」，又错误地以为位移只有 0.9 倍，于是去最大化 \(-\bigl(0.9a-3\bigr)^2\)：

```python
candidates = np.linspace(-4, 4, 33)
model_scores = -(0.9 * candidates - 3.0) ** 2
chosen = float(candidates[model_scores.argmax()])
print('planner chosen action:', chosen, 'training support: [-1, 1]')
print('是否 OOD:', abs(chosen) > 1)
```

**运行这一步，你会看到什么？**

```
planner chosen action: 3.25 training support: [-1, 1]
是否 OOD: True
```

33 个候选均匀铺在 \([-4, 4]\)，步长 0.25。模型分数在 \(a=3.25\) 处最高（\(0.9\times 3.25=2.925\)，离 3 只差 0.075）。训练里从未见过 \(|a|>1\)。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/planner-exploit.png" alt="Planner 漏洞" style="max-width:min(820px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">浅蓝带是训练支持 \([-1, 1]\)。红线是模型分数，灰线是真实分数。Planner 沿着红线走到 \(a=3.25\)，已经出了训练分布。4.6 里那句「梦境分数高不等于真实分数高」，就是这件事。</div>
</div>

如果在真实世界里执行 \(a=3.25\)，下一步是 3.25，不是模型以为的 2.925。差距看起来不大，因为这个玩具的偏差是线性的。像素世界、接触动力学里，同样的「走出支持」通常直接撞墙。

**判断**：Planner 选中的动作落在训练支持外面，就要单独拿真实环境复核。只看模型里的回报，等于让考生自己给自己打分。

**一个值得做的实验**：给规划目标加一项不确定性惩罚，或把候选裁回 \([-1, 1]\)，再看选中的动作还在不在外面。限幅是最便宜的补丁；补数据、缩短 horizon、真实闭环复核，是更稳的做法。

## 测试六：run manifest 与硬件证据

**问题**：你的实验能复现吗？

配置文件里写「预计 21GB」只是计划。只有完整跑完一次、留下命令、种子、耗时、显存峰值和 checkpoint 哈希，才叫证据。`RunManifest` 把这些字段收成一份可保存的记录；`RunTimer` 量墙钟时间；`runtime_summary` 记下 Python 与机器。

```python
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time
from typing import Optional

@dataclass
class RunManifest:
    experiment: str
    route: str
    seed: int
    dataset: str
    split: str
    command: str
    started_at: str
    wall_time_seconds: float
    device: str = "cpu"
    gpu: str = "not-recorded"
    cuda: str = "not-recorded"
    peak_allocated_mb: Optional[float] = None
    peak_reserved_mb: Optional[float] = None
    checkpoint_sha256: Optional[str] = None
    notes: str = ""

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

class RunTimer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.seconds = time.perf_counter() - self.start

def runtime_summary():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

with RunTimer() as timer:
    _ = sum(range(1000))

manifest = RunManifest(
    experiment="evaluation-smoke",
    route="evaluation",
    seed=0,
    dataset="analytic-toy",
    split="test",
    command="run test-a-world-model notebook",
    started_at="2026-08-12T00:00:00Z",
    wall_time_seconds=timer.seconds,
    notes="CPU 教学 smoke，不是 24GB 证据",
)
print(runtime_summary())
print("peak_reserved_mb:", manifest.peak_reserved_mb)
```

**运行这一步，你会看到什么？**

```
{'python': '3.12.2', 'platform': 'macOS-26.5.2-arm64-arm-64bit', 'processor': 'arm'}
peak_reserved_mb: None
```

耗时大约 \(10^{-5}\) 秒量级，只是一次 `sum(range(1000))`。`gpu`、`cuda`、`checkpoint_sha256`、两个显存峰值全部空着。这份清单证明的是「教学 smoke 跑通了」，不是「24GB 能训完」。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/run-manifest.png" alt="Run Manifest" style="max-width:min(860px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">左边是一次完整运行必须写清的字段，右边是这次教学运行实际留下的 JSON。peak reserved 是空的，所以不能拿它当 24GB 证据。显存要同时报 allocated 和 reserved，只报小的那个不够。</div>
</div>

**判断**：缺字段就等于让后来的人猜。失败运行也要留——OOM、NaN、发散，分别指向配方太大、数值不稳和代码写错，三种问题修法完全不同。

**一个值得做的实验**：给自己的训练脚本接上 `RunManifest.save`，把 checkpoint 做一次 sha256，再在另一台机器上只凭这份 JSON 重跑。缺哪一项，复现就会在哪一项卡住。

## 与仿真基准协议对标

六项审问是给自己模型做体检；投稿级的机器人评测另有事实标准，以 LIBERO 为代表：每任务 $10$ 条 episode、$4$ 个套件共 $400$ 条、$3$ 个种子取平均、固定数据集 revision、给出官方复现基线（如 $\pi_{0.5}$ 的 $97.5\%$）。LeRobot 的 `lerobot-eval` 已把这套协议做成一行命令。

本书协议在两处更严，迁移到基准时不得放松：

1. **置信区间必报**。基准社区习惯报种子均值；本书要求同时给出区间——$20$ 次试验的半宽约 $\pm 0.20$（见 [7.7](/chapters/07-robot-vla/07-simulators-and-sim2real)）。
2. **失败分类必报**。成功率均值之外，必须给出失败模式分布；排行榜不替你查这个。

反过来，基准协议有本书 toy 评测没有的东西：任务数足够多、扰动足够系统（LIBERO-plus 有上万个扰动变体）。正确姿势是：用基准的**任务广度**，加本书的**统计深度**。

## 已知简化与坑

教学版有几处刻意的简化，跑不通或数字对不上时，先从这里找原因：

- **9.2 的 toy 过于简单**。一维位移、线性偏差，失败模式干净，但不等于你的路线模型会以同样方式失败。
- **OOD 测试只改变一个变量**。真实世界的 OOD 是多变量叠在一起的，单变量测试只是起点。
- **校准曲线需要大量样本**。六个点、三个箱子，只能示范分箱，不能给校准下结论。
- **Planner 漏洞测试依赖 Planner 质量**。如果 Planner 本身很弱，可能根本走不到模型的漏洞上。
- **教学 manifest 不是硬件证据**。`peak_reserved_mb is None` 必须写进报告，不能把 smoke 说成 24GB 已验证。
- **平均会掩盖对消**。左右交替那条轨迹第 12 步误差是 0，拉低了整条 horizon 曲线。先看分轨迹，再看平均。

## 扩展练习

完成 9.2 后，按从便宜到昂贵的顺序推荐：

1. **迁移到自己训练的模型**：把六项测试的接口换成你的 `predict(start, actions)`，观察失败模式落在哪一项。
2. **加上复制上一帧 / 匀速外推基线**：确认模型在哪个 horizon 窗口里真正赢过傻基线。
3. **多变量 OOD**：同时改变动作幅度和起点偏移，看误差是相加还是相乘。
4. **对抗测试**：故意构造让 Planner 回报最高、真实回报最低的动作，作为 9.4 的稳定失败。

## 本节小结

- **多步 horizon 曲线暴露复合误差**：本节 toy 上，1 / 5 / 12 步 MSE 是 0.01 / 0.17 / 0.96。
- **反事实动作测试因果理解**：同一起点只换动作，差异必须非零且方向对；这里是 0 / 1.8 / 1.8。
- **单变量 OOD 测试泛化边界**：动作从 1 改到 3，绝对误差从 0.1 跳到 0.9，模型照样吐数字。
- **不确定性校准测试置信度可靠性**：分箱之后才谈诚实；六个样本不够下结论。
- **Planner 漏洞测试模型利用**：规划器选出 \(a=3.25\)，落在训练支持 \([-1, 1]\) 外面。
- **run manifest 是复现的基础**：没有清单，实验无法复现；空着的显存字段不能当 24GB 证据。

从 4.6 的 World Models 到这一节的审问测试，评价在不断加深：loss 下降 → 多步一致 → 反事实推理 → OOD 泛化 → 不确定性校准 → Planner 漏洞。每一种测试暴露一种失败模式。你的任务是在具体场景里选择最相关的那几项，而不是把六张图都贴上再宣布成功。

## 后续工作

9.2 用解析 toy 把系统检查跑通。真正的世界模型评价，要回答的是同一组问题，只是对象换成了像素、latent 和真实回报。

**Dreamer** 的评价设定不再报单步重建，而是报真实环境的回合回报，并与 model-free 基线画在同一张样本效率曲线上。想象中的回报只是训练信号，验收看的是关上梦境之后还能不能得分。

**World Models** 已经用温度实验写下模型利用的原型：\(\tau\) 太小，梦境确定、进化容易，控制器却可能过拟合那些只在想象里成立的路线。4.6 里 `dream_score` 与 `real_score` 的差距，就是这张账单。

**9.4** 从这里出发。不要同时改架构、数据和规划器。从六项里挑一个**稳定可复现的失败**——比如 horizon 10 之后必定超过复制基线，或 Planner 稳定选出 \(|a|>1\) 的动作——只改一件事，再把同一套审问跑一遍。

## 参考文献

1. Ha, D., & Schmidhuber, J. (2018). Recurrent World Models Facilitate Policy Evolution. _NeurIPS 2018_. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) —— 温度实验与梦境 / 真实分数的差距，是模型利用的经典演示。
2. Talvitie, E. (2014). Model Regularization for Stable Sample Rollouts. _UAI 2014_. [arXiv:1406.2315](https://arxiv.org/abs/1406.2315) —— 开环多步 rollout 的复合误差，以及用正则把模型按回数据支持。
3. Talvitie, E. (2017). Self-Correcting Models for Model-Based Reinforcement Learning. _AAAI 2017_. [arXiv:1604.02212](https://arxiv.org/abs/1604.02212) —— 让模型学会在自己的预测上做下一步，减轻训练 / 部署的分布偏移。
4. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. _ICML 2017_. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) —— 可靠性图与分箱校准；现代网络常常过度自信。
5. Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. _ICLR 2020_. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) —— 验收看真实环境回报与样本效率，而不是单步重建。
6. Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering Diverse Domains through World Models. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) —— DreamerV3：同一套超参、同一套评价协议打通 150+ 任务。
