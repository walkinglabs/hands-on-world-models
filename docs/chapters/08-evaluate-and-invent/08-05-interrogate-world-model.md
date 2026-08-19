# 8.5　动手：审问世界模型

> **本节目标**：不训练新网络。用一个能完全算清的一维玩具，把六项审问——多步 horizon、反事实动作、单变量 OOD、不确定性校准、Planner 漏洞、Run Manifest——写成带公式和真实数字的判据。做完后把同一套问题搬到你的 PA1 模型上。
>
> **本节代码**：[8.7 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/08_evaluation/Z0-test-a-world-model.ipynb) · [evaluation.py](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/evaluation.py)
>
> **前置知识**：8.1–8.4 已经讲过基线、反事实、硬件证据和失败分析。这一节把它们接到可运行的函数上。具体单元格在 [8.7 实验页](/chapters/08-evaluate-and-invent/08-07-test-a-world-model)；本页写清「为什么这样问、怎样才算过」。

---

你花了几小时甚至几天训练了一台世界模型。loss 下降了，重建看起来不错，token accuracy 也挺高。

但模型真的理解了世界，还是只记住了训练数据的统计？

**审问（interrogate）** 不是证明它有多好，而是设计一组测试，让失败模式自己站出来。8.7 不训练新架构。它先用一个解析玩具把六项检查跑通，再要求你把同一套检查原封不动搬到 PA1。

玩具故意有偏差：真实世界每次动作完整移动，模型只预测 0.9 倍位移。一步几乎看不见，多步以后会慢慢落后。这不是实现 bug，是后面所有公式的活标本。

## 本次会得到什么

跑完 8.7，你应能亲手算出并解释这些数字：

- horizon 1 / 5 / 12 的 MSE：**0.01 / 0.17 / 0.96**
- 固定起点、三组动作相对「停留」的差异：**0.0 / 1.8 / 1.8**
- 训练范围内动作 1 的误差 **0.1**，动作 3 且尺度再偏一点时误差 **0.9**
- 三个校准箱：置信度 0.15 / 0.50 / 0.88，对应频率 0.00 / 0.50 / 1.00
- Planner 在支持区间 \([-1,1]\) 之外选中 **\(a=3.25\)**
- 一份 `RunManifest`，峰值显存在 CPU smoke 上是 `None`

把同一套函数接到 PA1 模型后，再交：基线对照的 horizon 曲线、反事实图、至少 3 个 OOD 失败、校准图（若模型有不确定性）、Planner 漏洞样例、「什么行 / 什么不行」一页总结。

## 怎样运行

```text
notebooks/08_evaluation/Z0-test-a-world-model.ipynb
```

CPU 即可。先把玩具跑明白，再评你的 PA1 模型。

```bash
jupyter lab
PYTHONPATH=src python -m unittest tests.test_evaluation -v
```

核心函数在 `hwm.evaluation`：`horizon_errors`、`counterfactual_sensitivity`、`calibration_bins`、`RunManifest`。8.7 的接口是「你提供 `predict(start, actions)`」，不是「再写一套评测」。

## 第一步：先立基线，再画 horizon

一步误差是开卷考试——每一步都从真实状态出发，只猜下一步：

$$
\hat s_{t+1}=f_\theta(s_t, a_t),\qquad
\mathrm{MSE}_1=\mathbb{E}\bigl[\|\hat s_{t+1}-s_{t+1}\|^2\bigr]
$$

部署时没有真实状态可白嫖。开环 rollout 只能把自己的输出喂回去：

$$
\hat s_1=f_\theta(s_0,a_0),\;
\hat s_{k+1}=f_\theta(\hat s_k,a_k),\;
\mathrm{err}(H)=\mathbb{E}\bigl[\|\hat s_H-s_H\|^2\bigr]
$$

8.7 的模型是 \(s_{t+1}=s_t+0.9\,a_t\)，世界是 \(s_{t+1}=s_t+a_t\)。三个起点 \(0,1,-1\)，三段 12 步动作。`horizon_errors` 先对每个起点算逐步平方误差，再对起点取平均，**不把多步藏进一个平均数**。

```python
from hwm.evaluation import horizon_errors
import numpy as np

def rollout(start, actions, scale=1.0):
    result, state = [], float(start)
    for action in actions:
        state += scale * action
        result.append(state)
    return np.asarray(result)

starts = [0.0, 1.0, -1.0]
action_sequences = [[1.0] * 12, [-1.0] * 12, [1.0, -1.0] * 6]
truth = np.stack([rollout(s, a, 1.0) for s, a in zip(starts, action_sequences)])
errors = horizon_errors(lambda s, a: rollout(s, a, 0.9), starts, action_sequences, truth)
print([round(float(errors[i]), 4) for i in (0, 4, 11)])
# [0.01, 0.17, 0.96]
```

**真实判据**：`errors[-1] > errors[0]`。12 个点是 0.01, 0.027, 0.063, 0.107, 0.17, 0.24, 0.33, 0.427, 0.543, 0.667, 0.81, 0.96，几乎按 \(H^2\) 涨——单步偏差 0.1，\(H\) 步累积偏差 \(0.1H\)，平方以后就是 \(0.01 H^2\)。

还要画两条傻子基线：复制当前状态、匀速外推 \(\hat s_{t+1}=s_t+v\Delta t\)。复制基线在这个玩具上第 12 步会到几十，图里不画，免得把 0.96 那根曲线压扁。但它有用：如果到某个 \(H\) 你的模型已经不如复制，那根 horizon 就不能交给 Planner。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/z85-horizon.png" alt="0.9 倍位移的 horizon 曲线" style="max-width:min(760px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">8.7 玩具：一步 MSE 0.01，五步 0.17，十二步 0.96。复合误差不是比喻，是可以手算的 \(0.01 H^2\)。</div>
</div>

接到 PA1 时，至少报 1 / 5 / 10 / 30 步，并写明「从第几步开始不如复制」。挑 3 条 rollout 可视化：一条还行、一条开始漂、一条崩了。别只挑成功的。

## 第二步：固定起点，只换动作

反事实问的是因果关系，不是「训练集里左转视频看起来像左转」。公式是：同一 \(s_0\)、同一随机源，只换动作序列，看轨迹是否分开。

$$
\Delta_i=\mathrm{mean}\bigl|\,f(s_0,a^{(i)})-f(s_0,a^{(0)})\,\bigr|
$$

`counterfactual_sensitivity` 把第一条序列当参考，对其余序列做平均绝对差。参考选「全 0 / 停留」，差异才好解释。

```python
from hwm.evaluation import counterfactual_sensitivity

candidate_actions = [[0, 0, 0], [1, 1, 1], [-1, -1, -1]]
sensitivity = counterfactual_sensitivity(
    lambda s, a: rollout(s, a, 0.9), 0.0, candidate_actions
)
print(np.round(sensitivity, 3))
# [0.  1.8 1.8]
```

**真实判据**：停留相对自己是 0；持续 +1 和持续 −1 都是 **1.8**。三步位移分别是 \(0.9+1.8+2.7=5.4\) 的均值绝对差摊到三步，得到 1.8。方向相反、幅度相同，对称性本身也是一种检查。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/z85-counterfactual.png" alt="同一起点换动作" style="max-width:min(760px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">黑线停在原点，蓝线往正方向走，绿线往负方向走。若三条重合，动作条件就没学到，视频再清楚也是白搭。</div>
</div>

接到 PA1 时锁死 seed，至少测：全 0、正向最大、反向最大。随机环境不要要求像素逐点相同，比较的是分布。\(\Delta\approx 0\) 就是失败，不是「模型很稳」。

## 第三步：单变量 OOD

神经网络有个坏毛病：没见过的输入也会吐一个看起来合理的数。OOD 测试必须在跑之前写好维度，一次只改一个变量。

8.7 的训练支持是 \(|a|\le 1\)。动作 1、尺度 0.9 时绝对误差 0.1；动作 3、尺度再改成 0.7 时绝对误差 0.9。后者不是「同一个模型的外推」，而是故意把偏差加大，模拟「分布外动力学更不准」。

```python
in_distribution = abs(rollout(0, [1], 0.9)[0] - rollout(0, [1], 1.0)[0])
ood = abs(rollout(0, [3], 0.7)[0] - rollout(0, [3], 1.0)[0])
# 0.1 vs 0.9
```

**真实判据**：`ood > in_distribution`。接到 PA1，按路线选维度，每维至少 20 个样本：

| 路线 | 建议改的那一个变量 |
| ---- | ------------------ |
| 通用 | 动作幅度超出训练范围 |
| 视觉 | 颜色打乱，或亮度 \(\pm 50\%\) |
| 物理 | 摩擦或速度乘一个未见过的系数 |
| 机器人 | 物体位置偏移，或未见过的组合 |
| 空间 | 视角偏 15°，或点云降采样 |

记录三件事：误差涨了多少、不确定性有没有升、它是「知道自己不知道」还是在瞎编。挑 3 个最离谱的失败可视化。

## 第四步：不确定性校准

模型说「80% 会撞」，所有打 80 分的样本里大约就该有 80% 真撞。Guo 等人把现代网络的校准问题写成 reliability diagram [1]：把预测概率分箱，比较箱内平均置信度 \(\bar p_b\) 和真实频率 \(\bar f_b\)。

$$
\mathrm{ECE}=\sum_{b=1}^{B}\frac{n_b}{N}\,\bigl|\bar p_b-\bar f_b\bigr|
$$

8.7 用 6 个点、3 个箱，只为看清接口，不是为了得到可靠的 ECE。

```python
from hwm.evaluation import calibration_bins

probabilities = np.array([0.1, 0.2, 0.35, 0.65, 0.8, 0.95])
outcomes = np.array([0, 0, 1, 0, 1, 1])
for item in calibration_bins(probabilities, outcomes, num_bins=3):
    print(item)
```

**真实判据**：

| 箱 | 置信度 | 频率 | 怎么读 |
| -- | -----: | ---: | ------ |
| \([0, 1/3)\) | 0.15 | 0.00 | 略偏高估 |
| \([1/3, 2/3)\) | 0.50 | 0.50 | 碰巧准 |
| \([2/3, 1]\) | 0.875 | 1.00 | 略偏低估 |

没有不确定性输出的模型可以跳过，但必须在报告里写：「我的模型不提供不确定性，无法判断什么时候它不可靠。」ensemble、高斯方差、扩散多样样本，都可以进这个函数；没分箱的「看起来不确定」不算校准。

## 第五步：让 Planner 去找漏洞

随机测试采样训练分布里的动作。Planner 解的是另一件事：在模型里找高回报动作。若模型错误地认为超大动作没有代价，最优解会落在训练支持外面。

8.7 的模型分数是 \(-(0.9a-3)^2\)。连续区间 \([-4,4]\) 上的最大值在 \(a=3/0.9=3.\overline{3}\)，33 个网格点里最近的是 **3.25**。训练支持是 \([-1,1]\)。

```python
candidates = np.linspace(-4, 4, 33)
model_scores = -(0.9 * candidates - 3.0) ** 2
chosen = float(candidates[model_scores.argmax()])
# 3.25 ，且 abs(chosen) > 1
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/z85-calibration-planner.png" alt="校准分箱与 Planner 选中 OOD 动作" style="max-width:min(900px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">左：三个校准箱。右：浅绿是训练支持 \([-1,1]\)，黑线是 Planner 选中的 \(a=3.25\)。模型利用不是抽象词，是一个落在支持集外面的动作。</div>
</div>

**真实判据**：选中动作必须标成 OOD，并在真实动态里复核——这里真实最优其实在 \(a=3\)，模型的 0.9 倍尺度把峰值挪歪了。接到 PA1：固定起点，让 CEM / MPC / 你的规划器跑至少 10 次，记录动作幅值、是否出支持集、真实执行回不回报得上。限制动作范围是防护，不是作弊；不复核真实环境，就是在给模型的幻觉打分。

这正是 Janner 等人问过的问题：什么时候该信任模型 [2]。World Models 原文里低温梦境 2086、真实环境 193，也是同一类账单。

## 第六步：留下运行证据

`RunManifest` 要能让别人复现：实验名、路线、seed、数据、split、命令、开始时间、墙钟、设备、峰值显存、checkpoint SHA256。8.7 的 CPU smoke 里 `peak_reserved_mb is None`——这不是漏填，是诚实：没有 GPU 就不要写「24GB 能跑」。

```python
from hwm.evaluation import RunManifest, RunTimer, runtime_summary

with RunTimer() as timer:
    _ = sum(range(1000))
manifest = RunManifest(
    experiment="Z0-smoke",
    route="Z",
    seed=0,
    dataset="analytic-toy",
    split="test",
    command="run Z0 notebook",
    started_at="2026-08-12T00:00:00Z",
    wall_time_seconds=timer.seconds,
    notes="CPU 教学 smoke，不是 24GB 证据",
)
```

接到 PA1 时再补：git commit、PyTorch 版本、GPU 型号、评测样本数、全部曲线路径。状态页不能只写「在 24GB 上能跑」。

## 第七步：写清「什么行，什么不行」

一页，写具体条件，不写空话。

玩具版可以这样写：

- 做到了：horizon 1 的误差 0.01；换动作后轨迹按 \(\Delta=1.8\) 分开；校准箱能算出来。
- 没做到：horizon 12 误差 0.96；动作 3 的误差是动作 1 的 9 倍；Planner 选 \(a=3.25\)。
- 边界：只在 \(|a|\le 1\)、一步预测、确定性一维位移时可靠。

PA1 版把数字换成你的模型。下一节 8.6 / PA2 从这里列出的稳定失败里只挑一个。

## 和 8.7 实验页怎么分工

| 内容 | 本页 8.5 | [8.7](/chapters/08-evaluate-and-invent/08-07-test-a-world-model) |
| ---- | -------- | -------------- |
| 为什么要问这六件事 | 公式、判据、真实数字 | 简写动机 |
| 怎么点单元格 | 指向 Notebook | 逐步跟做 |
| 接到 PA1 | 列出必须迁移的证据 | 作为扩展练习 |

不要两页都写成检查清单。本页的清单只出现在「本次会得到什么」和最后的总结里。

## 已知简化与坑

- **玩具是确定性一维位移。** 失败模式干净，不代表你的 PA1 会以同样方式崩。
- **校准只用了 6 个点。** ECE 在这个样本量下没有统计意义，它只教接口。
- **OOD 的 0.9 是把模型和尺度一起改了。** 接到真实模型时，只改输入，不要偷偷改权重。
- **Planner 漏洞依赖 Planner 够贪。** 规划器本身很弱，可能根本走不到支持集外面。
- **复制基线在无界位移上会炸。** 比的是「从哪一步开始不如它」，不是和它比终点绝对值。

## 本节小结

- **一步误差是开卷，多步才是真用。** 0.9 倍位移给出 0.01 → 0.17 → 0.96。
- **反事实必须锁死起点。** \(\Delta=0\) 就是没学到动作，不是模型很稳。
- **OOD 一次只改一个变量**，并事先写好维度。
- **没分箱的不确定性不算校准。** 没有不确定性输出，就明确写做不到。
- **Planner 是漏洞的最佳猎手。** \(a=3.25\) 落在 \([-1,1]\) 外面，必须回真实动态复核。
- **没有 manifest，实验不算完成。**

从 3.6 的 World Models 到这一节的审问，评价在加深：loss 下降 → 多步一致 → 反事实 → OOD → 校准 → Planner 漏洞。每一种测试暴露一种失败。下一节从其中一个稳定失败出发，只改一件事。

## 参考文献

1. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) —— reliability diagram 与 ECE 的经典写法。
2. Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). When to Trust Your Model: Model-Based Policy Optimization. *NeurIPS 2019*. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253) —— 模型只在可信区间里用，Planner 漏洞的直接动机。
3. Ha, D., & Schmidhuber, J. (2018). Recurrent World Models Facilitate Policy Evolution. *NeurIPS 2018*. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) —— 低温梦境高分、真实环境崩溃，是模型利用的经典数字。
4. Talvitie, E. (2014). Model Regularization for Stable Sample Rollouts. *UAI 2014*. [arXiv:1406.2315](https://arxiv.org/abs/1406.2315) —— 多步 rollout 复合误差的早期分析。
5. Hafner, D., et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR 2020*. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) —— 想象中的回报必须回到真实环境核对，和本节第五步是同一句话。
