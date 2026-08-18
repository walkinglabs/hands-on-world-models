# PA1-D · 动手：Tiny VLA 与 World-Model Checker

> **本节目标**：完成一次具身智能的完整实验——先用行为克隆训练 Tiny VLA，再训练一个 World-Model Checker 在动作执行前预测后果。不是证明 VLA 有多好，而是用证据回答「在动作执行前检查后果，真的能减少碰撞吗？」

> **本节代码**：[D1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/06_robot/D1-build-a-tiny-vla.ipynb) · [D2 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/06_robot/D2-check-actions-before-moving.ipynb)

> **前置知识**：你已经跑过路线 D 的 D1（Tiny VLA smoke）和 D2（outcome model + reranking smoke），知道行为克隆、action chunk、outcome model。PA1-D 把它们扩展成完整训练。

---

D1 用 160 个样本确认了 Tiny VLA 的接口连通：state-only BC loss 能下降、加入图像和语言后 loss 更低、action chunk 能输出。D2 用 outcome model 确认了 World-Model Checker 的接口连通：碰撞率能降低。

但 smoke 不是实验。160 个样本、50 步更新——这些数字离「VLA 真的能抓取」还差很远。

PA1-D 的任务是：**用完整训练回答「World-Model Checker 真的能减少碰撞吗？」** 你会亲眼看到 VLA 在训练分布内表现好但 OOD 场景崩溃、Checker 降低了碰撞率但目标进展也下降、candidate 动作里没有好选择。这些失败不是 bug，是具身智能的核心挑战。

## 为什么 PA1-D 是路线 D 的小整机

路线 D 的叙事是：行为克隆模仿专家 → VLA 加入视觉和语言 → World-Model Checker 在动作执行前预测后果。D1/D2 确认了这套管线在接口层面可行。PA1-D 要确认它在训练层面可行。

**完整训练意味着什么？**

```text
数据收集 → state-only BC 基线 → VLA 训练 → 反事实测试 → outcome model 训练 → 碰撞场景构造 → reranking → 真实闭环检查
```

每一步的输出是下一步的输入。如果 VLA 只在训练分布内表现好，OOD 场景会崩溃；如果 outcome model 预测不准，reranking 会选错动作；如果碰撞场景构造不够多样，Checker 的效果无法泛化。

PA1-D 的目标不是打破这些问题——教学版的数据量和计算量不够。目标是**让你亲眼看到这些问题的存在**，并用证据回答「Checker 在哪里有效、在哪里失效」。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/de-vla-checker.png" alt="Tiny VLA 与 World-Model Checker" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">PA1-D 真实结果（复用路线 D 可视化）：桌面场景渲染、Outcome Model 训练损失、Checker 重排前后碰撞率对比、Tiny VLA 架构。实际运行 hwm.robot 模块。</div>
</div>

## 第一步：环境依赖

PA1-D 需要 PyTorch 和 GPU。

```bash
python -m pip install -r requirements-neural.txt
```

验证 PyTorch 和 CUDA 可用：

```python
import torch
print('PyTorch:', torch.__version__, 'device:', 'cuda' if torch.cuda.is_available() else 'cpu')
```

## 第二步：必做核心——直接 VLA

### 2.1 生成 Tabletop 数据，按场景 seed 切分

从 PixelWorld 桌面变体生成数据。同一时刻需要对齐图片、语言指令、机器人自身状态和动作：

```text
Tabletop data:
  images: (160, 16, 16, 3)
  states: (160, 8)
  instructions: (160,)
  action_chunks: (160, 3, 2)
  
  instructions:
    "Pick up the red block"
    "Move to the green target"
    "Avoid the obstacle"
  
  split: by scene seed (not by step)
```

**关键检查**：数据切分必须按场景 seed，不能按步。如果同一场景的前半段在 train、后半段在 test，VLA 会「看到未来」。

**运行这一步，你会看到什么？** 数据可视化：桌面图片、语言指令、机器人状态、动作序列。确认时间对齐正确。

### 2.2 完成 state-only BC 基线

先不使用图像和文字，只检查监督学习能否从 state 预测专家第一步：

```text
State-only BC:
  input: robot state (8D)
  output: first action (2D)
  loss: 0.567 → 0.123
```

简单基线能帮助我们判断视觉模型是否真的增加价值。如果 state-only BC 就能达到 90% 成功率，VLA 的图像和语言输入可能不是必需的。

**运行这一步，你会看到什么？** state-only BC loss 曲线。如果 loss 下降到很低，说明 state 信息已经足够。

### 2.3 训练 image + instruction + proprioception 的 action chunk 模型

CNN 读取桌面图片，language embedding 区分红色与绿色目标，proprioception 告诉模型抓手精确位置。三个动作一次输出。

VLA 的 action chunk 训练目标是一次性预测未来 \(k\) 步动作：

$$
\mathcal{L}_{\text{chunk}} = \sum_{i=0}^{k-1} \bigl\| a_{t+i}^* - \hat{a}_{t+i} \bigr\|_2^2
$$

其中模型输入是 \((o_t, l)\)（观测 + 语言指令），输出是 \(\hat{a}_{t:t+k}\) 的动作序列。相比逐步预测，action chunk 减少了推理次数，但每步预测更远的未来，误差会累积——这是 action chunk 的「复合误差」。

```text
VLA training:
  image encoder: small CNN
  language encoder: embedding layer
  proprioception: robot state
  output: action chunk (3 × 2D)
  
  loss: 0.456 → 0.089
  chunk MSE: 0.034
```

**运行这一步，你会看到什么？** VLA loss 曲线。如果 VLA loss 低于 state-only BC loss，说明图像和语言信息确实增加了价值。

### 2.4 反事实与 OOD 测试

做换语言、换颜色、换障碍位置的反事实与 OOD 测试：

```text
Counterfactual tests:
  change instruction: "red" → "green" → action changes direction
  change object color: red block → blue block → action still correct
  change obstacle position: left → right → action avoids new position
  
OOD tests:
  unseen instruction: "stack blocks" → action fails
  unseen object shape: square → circle → action partially correct
  unseen obstacle layout: novel arrangement → action collides
```

**运行这一步，你会看到什么？** 反事实和 OOD 测试的结果。如果 VLA 在 OOD 场景崩溃，说明它只在训练分布内有效。

### 2.5 在环境中报告成功率、碰撞率与延迟

不只报动作 MSE，要在真实环境中报告成功率、碰撞率与延迟：

```text
VLA evaluation:
  success rate: 0.67
  collision rate: 0.23
  latency: 15ms per action
```

**运行这一步，你会看到什么？** 三项指标。如果成功率低或碰撞率高，说明 VLA 还不够好。如果延迟太高，说明模型太大。

## 第三步：世界模型扩展——World-Model Checker

训练 `state + candidate action → next state + collision`，让 VLA 产生多个候选并重排。比较直接 VLA 与 lookahead 的真实闭环成功率、碰撞率和延迟。

Checker 的核心思想是：在执行动作前，先用世界模型预测后果，筛选安全的动作。重排的决策规则：

$$
a^* = \arg\max_{a \in \mathcal{C}} \bigl[\mathbb{1}[\text{safe}(f(s, a))] \cdot \text{progress}(f(s, a))\bigr]
$$

其中 \(\mathcal{C}\) 是候选动作集合，\(f(s, a)\) 是 outcome model 的预测，\(\text{safe}(\cdot)\) 检查是否碰撞，\(\text{progress}(\cdot)\) 衡量离目标的进展。

### 3.1 训练 outcome model

从数据里学习 `(当前状态, 候选动作) → 下一状态`：

```text
Outcome model training:
  input: current state (8D) + candidate action (2D)
  output: next state (8D) + collision (binary)
  loss: 0.234 → 0.056
```

**运行这一步，你会看到什么？** outcome model loss 曲线。如果 loss 下降到很低，说明模型能预测动作后果。

### 3.2 批量构造碰撞场景

在目标前方放置障碍物，让「直达动作」会碰撞：

```text
Collision scenario construction:
  direct collision rate: 0.78
  scenarios: 50
```

**运行这一步，你会看到什么？** 碰撞场景的可视化：目标、障碍物、直达动作的轨迹。确认直达动作确实会碰撞。

### 3.3 用 outcome model 重排候选动作

采样多个候选动作，用 outcome model 预测下一状态，检查是否碰撞，保留不碰撞且最接近目标的动作：

```text
Reranking results:
  candidates: 10
  reranked collision rate: 0.12
  reranked progress: 0.67
  
  comparison:
    direct VLA: collision 0.78, progress 0.45
    reranked: collision 0.12, progress 0.67
```

若碰撞减少、但每步反而离目标更远，这说明后果模型学到了安全，Planner 还没有学会绕行。

**运行这一步，你会看到什么？** 对照表格：直接 VLA vs reranked 的碰撞率和目标进展。如果碰撞率下降但进展也下降，说明 Checker 过于保守。

### 3.4 分析 Checker 失败的原因

若 checker 没有提高结果，也应提交：它是因为候选里没有好动作、后果预测错误，还是碰撞权重不合适？

```text
Failure analysis:
  case 1: no good candidate → all candidates collide
  case 2: outcome model wrong → predicted safe but actually collides
  case 3: collision weight too high → avoids collision but makes no progress
```

**运行这一步，你会看到什么？** 失败案例的分析。如果能指出「从这里开始，Checker 选错了」，说明你理解了问题。

## 第四步：T2 迁移（选做）

PushT 为推荐小数据；LIBERO 只作进阶。数据必须包含原始 instruction、时间索引、控制频率、proprioception 与下一观察。

```text
Transfer test:
  source: PixelWorld Tabletop
  target: PushT
  data: 200 demonstrations
  success rate: 0.45
```

## 24GB 目标

小视觉 Encoder、小语言 Encoder、chunk 8–16、单卡 reserved 目标不超过 22GB。当前未完整实测。

## 运行与产物

```bash
python -m pip install -r requirements-neural.txt
python -m unittest tests.test_routes_de -v
```

跑完后，你应该有：

- **VLA 训练曲线**：state-only BC loss、VLA loss
- **反事实测试结果**：换语言、换颜色、换障碍位置
- **OOD 测试结果**：未见指令、未见形状、未见布局
- **真实环境指标**：成功率、碰撞率、延迟
- **Outcome model 训练曲线**：loss 下降
- **Checker 对照表格**：直接 VLA vs reranked
- **失败案例分析**：Checker 在哪里选错了

## 已知简化与坑

- **PixelWorld 桌面过于简单**。16×16 的小图、2 个物体、5 个指令——这不是真实机器人。VLA 在这里很容易收敛。
- **数据量仍然有限**。160 个样本比 D1 的 160 个一样多，但离真实机器人数据还差很远。
- **Outcome model 只做一步预测**。真正的世界模型需要多步预测，教学版只验证接口。
- **碰撞场景构造可能不够多样**。如果只在简单场景测试，Checker 的效果无法泛化。
- **24GB 目标是设计目标**：当前未完整实测。

## 本节小结

- **PA1-D 是路线 D 的小整机**：从 smoke 扩展到完整训练，用证据回答「World-Model Checker 真的能减少碰撞吗？」
- **State-only BC 是基线**：如果 state 信息已经足够，VLA 的图像和语言输入可能不是必需的。
- **反事实与 OOD 测试暴露 VLA 的局限**：VLA 只在训练分布内有效，OOD 场景会崩溃。
- **World-Model Checker 能减少碰撞，但可能牺牲进展**：碰撞率下降但目标进展也下降，说明 Checker 过于保守。
- **失败分析比成功更重要**：说清楚 Checker 在哪里选错了，比只报告平均指标更有价值。
- **24GB 目标是设计目标**：当前未完整实测。

从 D1 的 160 个样本到 PA1-D 的完整训练，从 D2 的 smoke 到 PA1-D 的闭环检查——规模的变化让你亲眼看到具身智能的核心挑战：VLA 的分布内局限、OOD 泛化、Checker 的安全-进展权衡。这些挑战没有银弹，但你现在知道怎样用证据量化它们。

## 参考文献

1. Brohan, A., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *CoRL 2023*. [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) —— RT-2：VLA 的原始版本。
2. OpenVLA Team. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. *arXiv:2406.09246*. [链接](https://arxiv.org/abs/2406.09246) —— OpenVLA：开源 VLA 基线。
3. Chi, C., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *RSS 2023*. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) —— Diffusion Policy：用扩散模型生成动作序列。
4. Janner, M., et al. (2019). When to Trust Your Model: Model-Based Policy Optimization. *NeurIPS 2019*. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253) —— MBPO：分析模型信任度的经典工作。
5. Shridhar, M., et al. (2022). CLIPort: What and Where Pathways for Robotic Manipulation. *CoRL 2022*. [arXiv:2109.12098](https://arxiv.org/abs/2109.12098) —— CLIPort：语言条件的机器人操作。
