# 7.13　动手：VLA 与动作后果检查

> **本节目标**：完成一次具身智能的完整实验——先用行为克隆训练 Tiny VLA，再训练一个 World-Model Checker 在动作执行前预测后果。不是证明 VLA 有多好，而是用证据回答「在动作执行前检查后果，真的能减少碰撞吗？」
>
> **本节代码**：[搭一台小型 VLA](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_robot/build-a-tiny-vla.ipynb) · [行动前检查动作](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_robot/check-actions-before-moving.ipynb) · [robot.py](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/robot.py)
>
> **前置知识**：你已经跑过机器人与 VLA 路线（第 7 章）的两份 Notebook——第一份是 Tiny VLA 的 smoke，第二份是 outcome model 加重排的 smoke——知道行为克隆、action chunk、outcome model。本节把它们扩展成完整训练，并补上闭环评价与失败分析。

---

第一份 Notebook 用 160 个样本确认接口连通：state-only BC 能下降、加入图像和语言后还能再降、action chunk 形状是 \((160, 3, 2)\)。第二份用随机候选动作训练后果模型，再在「直达必撞」场景里重排。

但 smoke 不是实验。160 个样本、50–80 步更新——这些数字离「VLA 真的能抓取」还差很远。默认渲染是 \(32\times 32\)，不是旧讲义里写过的 \(16\times 16\)。指令也只有两条中文：「移动到红色目标」「移动到绿色目标」。

本节的任务是：**用完整训练回答「World-Model Checker 真的能减少碰撞吗？」** 你会亲眼看到 VLA 的动作 MSE 下降但闭环成功率仍低、换指令后动作确实改变、Checker 把碰撞压下去但目标进展也可能一起掉。这些失败不是 bug，是具身智能的核心账本。

## 为什么本节是机器人与 VLA 路线的小整机

这条路线的叙事是：行为克隆模仿专家 → VLA 加入视觉和语言 → World-Model Checker 在动作执行前预测后果。两份跟做 Notebook 确认这套管线在接口层面可行。本节要确认它在训练层面可评价。

```text
数据收集 → state-only BC 基线 → VLA 训练 → 反事实测试 →
outcome model 训练 → 碰撞场景构造 → reranking → 真实闭环检查
```

每一步的输出是下一步的输入。如果 VLA 只在训练分布内拟合单步示范，OOD 场景会崩溃；如果 outcome model 把碰撞概率估反，reranking 会选错动作；如果候选集合里根本没有安全动作，Checker 再准也救不了。

目标不是打破这些问题——教学版的数据量和计算量不够。目标是**让你亲眼看到这些问题的存在**，并用证据回答「Checker 在哪里有效、在哪里失效」。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/tabletop.png" alt="桌面状态与直达碰撞" style="max-width:min(860px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">第二份 Notebook 手挑的那个状态：抓手在 (0.20, 0.50)，红色目标在 (0.85, 0.50)，障碍挡在 (0.31, 0.50)。直达动作 \(a=[1,0]\) 一步就会撞；斜向更远，却可能绕开。</div>
</div>

## 本次会得到什么

这一节要交证据，不是演示。提交时必须交出下面这份清单，缺一项就按该项零分：

- 按**场景 seed** 切分的数据卡：样本数、字段 shape、train/val/test 边界、seed
- state-only BC 的 loss 曲线，以及第一步动作 MSE
- Tiny VLA 的 chunk loss 曲线，参数量，以及同一张图换指令后的动作差
- 闭环指标：成功率、平均碰撞次数、初始距离、最终距离；测试集不得与训练集同 seed
- outcome model 的总 loss、状态 MSE、碰撞 BCE；以及训练数据里的碰撞比例
- 至少 64 个「直达必撞」场景上的对照：`direct_collision_rate`、`reranked_collision_rate`、`reranked_mean_progress`
- 至少一组 Checker 失败的逐候选拆解：真实碰撞、预测概率、被选中的动作
- 资源清单：设备、步数、耗时、checkpoint SHA256

## 怎样运行

```bash
python -m pip install -r requirements-neural.txt
jupyter lab
# notebooks/07_robot/build-a-tiny-vla.ipynb
# notebooks/07_robot/check-actions-before-moving.ipynb
```

可复用实现在 `src/hwm/robot.py`。即使暂时不跑 Notebook，也可以先跑测试：

```bash
PYTHONPATH=src python -m unittest tests.test_routes_de -v
```

默认渲染 \(32\times 32\)、chunk 长度 3、状态 8 维。本节要求你把样本量和更新次数加到能画出稳定曲线，而不是停在 smoke。

## 第一步：生成桌面数据，按场景 seed 切分

`make_tabletop_dataset` 在同一时刻对齐图片、语言、本体感觉和未来三个动作。默认 160 条、seed=0 时，字段是：

```text
images         (160, 3, 32, 32)
states         (160, 8)
instructions   (160,)
action_chunks  (160, 3, 2)
next_states    (160, 8)
collisions     (160,)
```

8 维状态依次是抓手、红色目标、绿色目标、障碍的 \((x, y)\)。专家策略先朝目标走；直走会进障碍半径 0.18 时，改走左右两个垂直方向中空隙更大的那个。环境步进时障碍半径是 0.13、步长 0.12，所以示范里的碰撞标签经常是 0——专家本来就会躲。

```python
from hwm.robot import INSTRUCTIONS, make_tabletop_dataset

data = make_tabletop_dataset(num_samples=160, chunk_size=3, seed=0)
print(INSTRUCTIONS[int(data["instructions"][0])])
# 移动到绿色目标
```

**切分必须按场景 seed，不能按步。** 同一场景的前半段进 train、后半段进 test，等于让模型看见未来。本节至少再单独生成一个测试集，例如 `seed=17`。

**运行这一步，你会看到什么？** 一张 \(32\times 32\) 的桌面图、一条中文指令、一个 8 维状态、一段长度 3 的动作。如果你看到的是英文 "Pick up the red block" 或 \(16\times 16\) 图片，说明你在看过期注释，不是当前代码。

## 第二步：state-only BC 基线

先不使用图像，只检查监督学习能否从状态和指令预测专家第一步。行为克隆的目标是

$$
\mathcal{L}_{\mathrm{BC}} = \mathbb{E}_{(s, \ell, a^*)}\bigl[\| a^* - \pi_\theta(s, \ell) \|_2^2\bigr]
$$

第一份 Notebook 的最小实现把 2 维指令做成 one-hot，再拼到 8 维状态上：

```python
state_policy = torch.nn.Sequential(
    torch.nn.Linear(8 + 2, 32), torch.nn.ReLU(),
    torch.nn.Linear(32, 2), torch.nn.Tanh(),
)
state_input = torch.cat(
    (data["states"], torch.nn.functional.one_hot(data["instructions"], 2).float()),
    dim=-1,
)
target = data["action_chunks"][:, 0]
```

**真实判据（第一份 Notebook 默认 50 步、Adam \(3\times 10^{-3}\)、seed=0）**：state-only BC loss 从 **0.498 降到 0.377**。它会降，但不会降到接近 0——8 维状态看不见障碍的像素形状，也看不见「红在左、绿在右」的图像布局。

如果这条基线已经能在闭环里拿到很高成功率，后面的图像和语言就没有增加决策价值。先把基线跑诚实，再谈 VLA。

## 第三步：训练 Tiny VLA 的 action chunk

CNN 读 \(32\times 32\) 桌面，8 维 language embedding 区分红/绿目标，本体感觉告诉模型抓手精确位置。一次输出未来 \(k=3\) 步：

$$
\mathcal{L}_{\mathrm{chunk}} = \sum_{i=0}^{k-1} \bigl\| a_{t+i}^* - \hat a_{t+i} \bigr\|_2^2
$$

`TinyVLA` 大约 **11,062** 个参数。视觉塔是两层 stride-2 卷积，再 AdaptiveAvgPool 到 \(2\times 2\)，和语言、状态拼成 96 维后进 MLP。

```python
from hwm.robot import TinyVLA

model = TinyVLA(chunk_size=3)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for _ in range(60):
    opt.zero_grad()
    chunks = model(data["images"], data["instructions"], data["states"])
    loss = torch.nn.functional.mse_loss(chunks, data["action_chunks"])
    loss.backward()
    opt.step()
```

**真实判据（第一份 Notebook 默认 60 步）**：multimodal chunk loss 从 **0.514 降到 0.270**。注意：这里的终点 0.270 比 state-only 的 0.377 低，但两者监督的不是同一个目标——前者是 3 步 \(\times\) 2 维，后者只是第一步。比较时要同时报「第一步 MSE」和「整段 chunk MSE」，不要混在一张表里假装赢了。

**一个值得做的实验**：把 `chunk_size` 从 3 提到 5，看后几步 MSE 怎样涨。chunk 越长，模型要一次猜更远的未来，这是 action chunk 自己的复合误差。

## 第四步：换指令，再回到环境

监督 loss 下降只说明拟合了示范。两个更硬的问题是：文字有没有进决策？闭环能不能到目标？

```python
from hwm.robot import evaluate_vla

same_image = data["images"][:1].expand(2, -1, -1, -1)
same_state = data["states"][:1].expand(2, -1)
with torch.no_grad():
    two_goals = model(same_image, torch.tensor([0, 1]), same_state)
print(float((two_goals[0] - two_goals[1]).abs().mean()))

test = make_tabletop_dataset(32, chunk_size=3, seed=17)
metrics = evaluate_vla(model, test["states"], test["instructions"], max_steps=12)
```

`evaluate_vla` 每步重新渲染，只执行 chunk 的第一步，成功半径 0.15。

**真实判据（第一份 Notebook 默认权重、测试 seed=17、12 步）**：

| 指标                 |  数值 |
| -------------------- | ----: |
| 换指令后的平均动作差 | 0.208 |
| `success_rate`       | 0.156 |
| `mean_collisions`    | 3.562 |
| `initial_distance`   | 0.375 |
| `final_distance`     | 0.302 |

动作差大于 0，说明语言头不是摆设。成功率只有 0.156、最终距离 0.302 仍大于成功半径，说明 **loss 下降不等于会做任务**。这正是本节要你放大样本量之后继续盯住的缺口。

OOD 至少做三类，一次只改一个变量：换未见过的障碍布局、把目标颜色对调但不改指令、把抓手初始位置推到训练范围外。每类报告成功率和碰撞，不要只贴一张好看的图。

## 第五步：训练 outcome model

专家示范几乎全是安全动作。只用示范训练碰撞头，标签会塌成常数。所以第二份 Notebook 用 `make_outcome_dataset`：一半样本把障碍故意放在随机动作前方。

400 条、seed=1 时，碰撞比例是 **0.527**。损失把下一状态 MSE 和碰撞 BCE 加在一起：

$$
\mathcal{L} = \| \hat s' - s' \|_2^2 + \mathrm{BCEWithLogits}(\hat c, c)
$$

```python
from hwm.robot import TabletopOutcomeModel, make_outcome_dataset, outcome_loss

odata = make_outcome_dataset(400, seed=1)
model = TabletopOutcomeModel()   # 约 5,449 个参数
```

**真实判据（80 步、Adam \(3\times 10^{-3}\)、seed=1）**：总 loss 从 **0.977 降到 0.538**；拆开看，状态 MSE 已经到 **0.013**，碰撞 BCE 仍有 **0.525**。状态好学，碰撞难学——后面重排选错，多半先查碰撞头，而不是查动力学头。

## 第六步：重排候选，并报告安全—进展账本

重排不是「选预测最安全的动作」这么简单。实现里的分数是

$$
\mathrm{score}(a) = -\| \hat s'(a)_{\mathrm{grip}} - g \|_2 - \lambda \,\sigma(\hat c(a))
$$

默认 \(\lambda=2\)；批量评价 `evaluate_reranker` 用 \(\lambda=4\)。候选是四个方向：直达、左垂直、右垂直、后退。

第二份 Notebook 那个手挑状态上，重排选中了候选 0（直达）。真实碰撞是 `[True, True, True, False]`，模型给出的碰撞概率却是 `[0.098, 0.138, 0.186, 0.756]`——它把唯一安全的后退当成最危险。手挑样例会骗人，所以必须批量评价。

```python
from hwm.robot import evaluate_reranker

safety = evaluate_reranker(model, num_cases=64, seed=23, collision_weight=4.0)
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/action-rerank.png" alt="重排前后碰撞与进展" style="max-width:min(860px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">64 个直达必撞场景：碰撞率从 1.000 降到 0.328，平均进展却是 -0.036。安全了，并不等于更接近目标。</div>
</div>

**真实判据（64 场景、seed=23、\(\lambda=4\)）**：

| 指标                      |   数值 |
| ------------------------- | -----: |
| `direct_collision_rate`   |  1.000 |
| `reranked_collision_rate` |  0.328 |
| `reranked_mean_progress`  | -0.036 |

碰撞确实少了，平均每步反而离目标更远。这就是「安全地停住」和「安全地完成任务」的差别。本节必须把这两本账分开写，不能只报碰撞下降。

失败分析至少归到三类：候选集合里没有好动作、后果预测把安全/危险估反、\(\lambda\) 太大导致永远选后退。指出「从哪一个候选开始选错」，比再报一次平均数有用。

## 24GB 目标

小视觉 Encoder、小语言 Encoder、chunk 8–16、单卡 reserved 设计目标不超过 22GB。这是课程预算，不是实测。当前仓库没有本节的 24GB 完整运行记录；在日志、曲线和 checkpoint 齐全前，不得标成「24GB 已验证」。

## 评分

| 项目         | 分数 | 检查重点                                                  |
| ------------ | ---: | --------------------------------------------------------- |
| 数据与切分   |   15 | 字段 shape 与代码一致；按场景 seed 切分；无泄漏           |
| 基线与 VLA   |   20 | state-only 先报；chunk loss 与第一步 MSE 分开；参数量写明 |
| 反事实与闭环 |   20 | 换指令动作差；成功率/碰撞/距离齐全；测试 seed 独立        |
| Checker 对照 |   25 | 直达 vs 重排；碰撞与进展分开；至少 64 个场景              |
| 失败诊断     |   10 | 能指出选错来自候选、预测还是权重                          |
| 表达与复现   |   10 | Notebook 可运行；seed、曲线、checkpoint 哈希完整          |

负结果可以拿满分：只要对照公平，并写清 Checker 为什么没有帮上忙。

## 已知简化与坑

- **桌面是 \(32\times 32\) 的合成图**，两个圆盘目标、一个圆盘障碍、两条指令。这不是真实机器人，VLA 在这里很容易把 loss 做下去。
- **专家示范几乎无碰撞**。用示范训练碰撞头会得到常数分类器；后果数据必须自己造失败。
- **Outcome model 只做一步**。多步绕障、把 chunk 整段送进 Checker，是扩展，不是默认能力。
- **手挑样例会骗人**。上面那个状态里，模型把唯一不撞的后退打成 0.756。批量数字才能立项。
- **\(\lambda\) 过大会学会原地不动**。碰撞降、进展变负，不是实现成功，是规划目标偏了。
- **24GB 是设计目标**，当前未完整实测。

## 本节小结

- **本节是机器人与 VLA 路线的小整机**：从 smoke 扩到可评价的训练，用证据回答 Checker 有没有减少碰撞。
- **State-only BC 是必须先报的基线**。第一份 Notebook 里它从 0.498 降到 0.377，用来判断视觉和语言是否真的加了价值。
- **动作 MSE 下降不等于闭环成功**。同一套默认权重上，chunk loss 到 0.270，成功率只有 0.156。
- **换指令后动作差 0.208**，说明语言条件进了决策；OOD 布局仍可能把这条差打回 0。
- **Checker 能减碰撞，也可能减进展**：1.000 → 0.328 的碰撞，配上 -0.036 的进展，必须分开写。
- **失败分析比平均指标更值钱**：说清楚选错来自候选、预测还是 \(\lambda\)。

从第一份 Notebook 的 160 条样本到本节的完整训练，从第二份 Notebook 的 smoke 到批量闭环——规模变化让你看见具身智能的三本账：分布内拟合、OOD 崩溃、安全与进展的权衡。

## 参考文献

1. Brohan, A., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. _CoRL 2023_. [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) —— 把视觉—语言骨干接到机器人动作上的代表性工作。
2. Kim, M. J., et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. _CoRL 2024_. [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) —— 开源 VLA 基线，讨论数据混合与动作解码。
3. Chi, C., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. _RSS 2023_. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) —— 用扩散生成动作序列，和 action chunk 是同一类接口。
4. Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). When to Trust Your Model: Model-Based Policy Optimization. _NeurIPS 2019_. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253) —— 模型只在可信区间里用：Checker 的直接祖先。
5. Shridhar, M., Manuelli, L., & Fox, D. (2022). CLIPort: What and Where Pathways for Robotic Manipulation. _CoRL 2022_. [arXiv:2109.12098](https://arxiv.org/abs/2109.12098) —— 语言条件操作里，空间通路和语义通路为什么要分开。
