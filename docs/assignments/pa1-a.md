# PA1-A · 动手：做出一台 Dreamer-lite

> **本节目标**：完成一条有证据的真实循环——从环境收集数据，在 RSSM 中想象，用 Actor-Critic 学习，再回到环境检查。不是复现 DreamerV3 的排行榜，而是亲手把 A1/A2 的 smoke 扩展成一次完整训练，亲眼看到「接口连通」与「策略收敛」之间的巨大鸿沟。

> **本节代码**：[A1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/03_decision/A1-learn-a-latent-world.ipynb) · [A2 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/03_decision/A2-act-in-imagination.ipynb)

> **前置知识**：你已经跑过路线 A 的 A1（RSSM 接口 smoke）和 A2（位置模型 + Actor-Critic smoke），知道 RSSM 的 prior/posterior 结构、imagination rollout、TD-λ 目标。PA1-A 把它们扩展成完整训练。

---

A1 用 4 段 PixelWorld episode 确认了 RSSM 接口连通：loss 能下降、shape 正确、梯度能流。A2 用可解释的位置模型证明了 learned dynamics 真的能帮助行动，然后把位置换成 RSSM latent，做了一次 Actor-Critic 更新。

但 smoke 不是训练。4 段 episode、15 次更新、一次 Actor 梯度——这些数字离「策略收敛」还差几个数量级。

PA1-A 的任务是：**把 smoke 扩展成一次完整训练，然后诚实地报告结果**。你会亲眼看到 loss 下降但策略没有改善、Actor 利用模型漏洞、imagination rollout 在 OOD 区域崩溃。这些失败不是 bug，是 Dreamer 架构设计的核心动机。

## 为什么 PA1-A 是路线 A 的小整机

路线 A 的叙事是：VAE 的 latent 空间不够稳 → RSSM 用双路结构替代；CMA-ES 没有梯度信号 → Actor-Critic 用反向传播替代。A1/A2 确认了这些替代方案在接口层面可行。PA1-A 要确认它们在训练层面可行。

**完整训练意味着什么？**

```text
数据收集 → 训练 RSSM → imagination rollout → 更新 Actor-Critic → 回到环境收集更多数据 → 重复
```

这是一个闭环。每一步的输出是下一步的输入。如果 RSSM 的预测不准，imagination 里的 Actor 会学到错误的策略；如果 Actor 的策略错误，收集到的新数据会引入偏差；如果数据有偏差，RSSM 的训练会更不准。

PA1-A 的目标不是打破这个恶性循环——教学版的数据量和计算量不够。目标是**让你亲眼看到这个循环的存在**，并诚实地报告它在哪里开始崩溃。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/pa1a-dreamer-lite.png" alt="Dreamer-lite 完整训练循环" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">PA1-A 真实数据流：数据收集（8 episodes，PixelWorld）→ RSSM 训练损失下降 → imagination rollout（从 posterior 出发，prior 推演 5 步）→ Actor-Critic 更新（REINFORCE + MSE）。实际运行 hwm.neural 模块。</div>
</div>

## 第一步：环境依赖

PA1-A 需要 PyTorch 和 GPU。

```bash
python -m pip install -r requirements-neural.txt
```

验证 PyTorch 和 CUDA 可用：

```python
import torch
print('PyTorch:', torch.__version__, 'device:', 'cuda' if torch.cuda.is_available() else 'cpu')
```

教学版 smoke 在 CPU 上运行（A1/A2），但 PA1-A 的完整训练建议用单张 24GB GPU。如果没有 GPU，可以减少数据量和更新次数，但需要明确标注「CPU 缩减版」。

## 第二步：阶段一——PixelWorld 完整训练

复用项目内 PixelWorld 生成器，增加可学习策略与明确目标。完成以下完整循环：

### 2.1 数据收集与 Replay Buffer

从环境中收集 episode，存入 Replay Buffer。Buffer 支持连续序列采样——不是随机打散单个 transition，而是保留时间连续性：

```text
Replay Buffer:
  episodes: 50 段
  sequence_length: 32
  batch_size: 16
  sample: (16, 32, ...)  # [batch, time, ...]
```

**运行这一步，你会看到什么？** Buffer 填满后，采样一个 batch，检查 shape 是否正确。sequence 内的 transition 必须来自同一段 episode，保持时间连续。

### 2.2 RSSM 训练

CNN Encoder 把 64×64 RGB 压成特征向量，RSSM 的 prior/posterior 双路结构提取隐状态，三个预测 head 分别重建观测、预测奖励、预测 episode 是否结束：

```text
RSSM training:
  observation loss: reconstruction MSE
  reward loss: reward prediction MSE
  continue loss: continue prediction BCE
  KL loss: prior vs posterior KL divergence
  total loss: obs + reward + continue + KL_weight * KL
```

**24GB 目标配方**：

| 项目 | 目标值 |
| ---- | ------: |
| 观察 | `64×64 RGB` |
| batch | 16 |
| sequence length | 32 |
| deterministic state | 256 |
| stochastic state | 32 |
| imagination horizon | 15 |
| mixed precision | 可选 |
| peak reserved | 目标不超过 22GB |

**运行这一步，你会看到什么？** loss 曲线。前 100 步 reconstruction loss 快速下降，然后趋于平稳。KL loss 可能先升后降——posterior 看到了更多样的观测，与 prior 的差距先变大，然后 prior 逐渐追上。

### 2.3 Imagination 与 Actor-Critic

从 posterior state 出发，让 Actor 采样动作，RSSM prior 推演 latent，Critic 给出 value，计算 TD-λ target：

$$
G_t^{\lambda} = \sum_{l=0}^{H-1} (\gamma\lambda)^l \bigl[\hat{r}_{t+l} + \gamma(1-\lambda)\hat{v}_{t+l+1}\bigr] + (\gamma\lambda)^H \hat{v}_{t+H}
$$

想象训练的核心循环：

```python
# Imagination 训练伪代码
states = buffer.sample_sequences(batch_size=16, seq_len=32)
latent = world_model.encode(states)  # posterior

for horizon_step in range(15):  # imagination horizon H=15
    action = actor(latent)                    # Actor 采样动作
    latent = world_model.imagine(latent, action)  # prior 推演
    reward = reward_head(latent)              # 预测奖励
    value = critic(latent)                    # Critic 评估

# Actor loss: 最大化想象回报
actor_loss = -lambda_return(rewards, values)
# Critic loss: 逼近 TD-λ target
critic_loss = mse(values, td_lambda_target(rewards, values))
```

Actor 的梯度沿「动作 → latent → reward/value」反向传播；Critic 的梯度沿「latent → value → TD target」反向传播。

**运行这一步，你会看到什么？** Actor loss 和 Critic loss 的曲线。如果 Actor loss 下降但真实环境 return 没有改善，说明 Actor 在利用模型漏洞——它在 imagination 里找到了「捷径」，但这些捷径在真实环境里不存在。

### 2.4 真实 episode 收集与再次训练

用更新后的 Actor 回到环境收集新 episode，加入 Replay Buffer，再次训练 RSSM 和 Actor-Critic。这是闭环的关键一步。

**运行这一步，你会看到什么？** 新 episode 的质量可能比初始随机 episode 好——Actor 学到了某种策略。但也可能更差——Actor 利用了模型漏洞，在真实环境里表现糟糕。

## 第三步：阶段二——DMC Cartpole 小迁移（选做）

使用 DeepMind Control Suite Cartpole 的固定小配置。若本地不方便安装 MuJoCo，可以只提交 PixelWorld 必做结果，并把 DMC 标为未运行。

**不能使用他人曲线代替。** 如果标为未运行，就标为未运行。

## 第四步：必交证据

PA1-A 要求八项证据，缺一不可：

**1. 数据卡与环境 wrapper**

写明数据来源、episode 数量、sequence 长度、切分方式。环境 wrapper 的接口必须与 A1/A2 一致。

**2. 世界模型各项 loss 曲线**

reconstruction、reward、continue、KL 四条曲线。标注训练步数和 epoch。

**3. One-step 与多步预测指标**

one-step reconstruction MSE；5/15/30-step latent 预测的 MSE 或相关性。多步指标应该随 horizon 增加而恶化——这就是复合误差。

**4. 基线**

至少一个基线：随机策略、PlaNet/CEM 或无模型策略。基线的 return 是 learned dynamics 的下限。

**5. 真实环境 return 对真实交互步数**

这是最关键的曲线。横轴是真实交互步数（不是训练步数），纵轴是 episode return。如果这条曲线上升，说明模型确实在帮助学习。

**6. 同一起点替换动作的反事实**

固定初始状态，替换不同动作序列，观察 imagination 预测是否随之改变。如果换动作后预测不变，模型没有学到动作条件动态。

**7. 一组 Actor 或 Planner 利用模型漏洞的失败**

找到 imagination 预测与真实结果差异最大的案例。分析失败来自：RSSM 预测不准？Actor 过度信任模型？imagination rollout 进入 OOD 区域？

**8. 资源清单与 checkpoint 哈希**

GPU 型号、CUDA 版本、peak reserved memory、总训练时间、checkpoint 文件哈希。确保结果可复现。

## 第五步：替代选题——Mini-MuZero

若更喜欢棋类搜索，可以使用四子棋完成 Mini-MuZero，替代 Dreamer-lite。它必须包含：

- **Representation**：把棋盘状态编码为隐状态
- **Dynamics**：预测下一隐状态和奖励
- **Prediction**：从隐状态预测策略和价值
- **MCTS**：用 dynamics 做搜索改进策略
- **搜索改进前后的胜率**

不能同时把两个项目各做一半。选择一种，完整提交数据—模型—规划—真实检查。

## 运行与产物

```bash
python -m pip install -r requirements-neural.txt
python -m unittest tests.test_control tests.test_neural -v
```

跑完后，你应该有：

- **完整的训练曲线**：RSSM loss、Actor loss、Critic loss
- **真实环境 return 曲线**：对真实交互步数
- **多步预测指标**：5/15/30-step 的 MSE 或相关性
- **反事实测试**：同一起点替换动作的预测差异
- **失败案例**：Actor 利用模型漏洞的具体证据
- **资源清单**：GPU、显存、时间、checkpoint 哈希

## Smoke 与 PA1-A 的区别

| 项目 | A1/A2 smoke | PA1-A |
| ---- | ----------- | ----- |
| 数据 | 4 段 PixelWorld | 50+ 段 PixelWorld；选做 DMC |
| 训练 | 15–100 步 | 直到形成稳定曲线 |
| 目的 | 检查控制增益、接口与梯度 | 检查完整 latent policy 的 return 和样本效率 |
| 资源 | CPU | 单张 24GB GPU |
| 结论 | 可解释 dynamics 能帮助行动；RSSM 接口可运行 | latent 模型与策略是否形成稳定闭环 |

## 已知简化与坑

教学版有几处刻意的简化，跑不通时先从这里找原因：

- **PixelWorld 仍然简单**。64×64 RGB、5 个动作、红色方块——这不是 Atari。RSSM 在这里学到的「世界」非常有限。
- **数据量仍然有限**。50 段 episode 比 A1 的 4 段多很多，但离 DreamerV3 的数千段还差很远。
- **没有 world model gradient**。教学版冻结 RSSM 只训练 Actor-Critic，不反向传播通过 dynamics。完整 Dreamer 需要 world model gradient。
- **imagination rollout 可能进入 OOD 区域**。Actor 采样动作后，prior 推演的 latent 可能远离训练分布，导致预测崩溃。
- **24GB 目标是设计目标，不是实测结果**。只有完整训练结束并提交 GPU/CUDA、peak allocated、peak reserved、总时间、曲线和 checkpoint 哈希以后，才可以在状态页改成「24GB 已验证」。

## 扩展练习

跑通默认配置后，按从便宜到昂贵的顺序推荐：

1. **增加数据量**：把 episode 数量从 50 提到 200，观察真实环境 return 的变化——数据多少开始「够用了」？
2. **改变 imagination horizon**：把 horizon 从 15 提到 30，观察 Actor loss 和真实 return 的变化——horizon 越长，复合误差越严重？
3. **加入 world model gradient**：让 Actor 的梯度反向传播通过 RSSM dynamics，观察样本效率的提升。
4. **可视化 imagination rollout**：把 latent imagination 的轨迹画出来，对比真实轨迹——Actor 在梦境里走了什么路线？

## 研究问题

完成训练后，回答：**模型最稳定的失败来自哪一项？**

```text
状态没有保存重要信息
动态在长 horizon 漂移
reward/value 误导 Actor
Planner 进入 OOD 区域
真实数据没有覆盖策略访问状态
```

至少提出两种解释，再选一项最小改动带到第 8 章。

## 本节小结

- **PA1-A 是路线 A 的小整机**：从 smoke 扩展到完整训练，亲眼看到「接口连通」与「策略收敛」之间的鸿沟。
- **完整训练是一个闭环**：数据收集 → RSSM 训练 → imagination → Actor-Critic 更新 → 回到环境 → 重复。
- **八项证据缺一不可**：数据卡、loss 曲线、多步指标、基线、真实 return、反事实、失败案例、资源清单。
- **Actor 会利用模型漏洞**：imagination 里的「捷径」在真实环境里可能不存在——这是 learned dynamics 的核心风险。
- **24GB 目标是设计目标**：只有完整训练并提交实测数据后，才能标为「已验证」。
- **Mini-MuZero 是替代选题**：四子棋 + Representation + Dynamics + Prediction + MCTS，完整提交数据—模型—规划—真实检查。

从 A1 的 4 段 episode 到 PA1-A 的 50 段 episode，从 A2 的一次 Actor 更新到 PA1-A 的数千次更新——规模的变化不是量变，是质变。你会亲眼看到 Dreamer 的承诺和代价：在想象中训练是可能的，但想象不是现实。

## 参考文献

1. Hafner, D., et al. (2019). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR 2020*. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) —— DreamerV1：RSSM + imagination + Actor-Critic 的原始版本。
2. Hafner, D., et al. (2021). Mastering Atari with Discrete World Models. *ICLR 2021*. [arXiv:2010.02193](https://arxiv.org/abs/2010.02193) —— DreamerV2：离散 latent + 更大规模，Atari 上超过 model-free。
3. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. *arXiv:2301.04104*. [链接](https://arxiv.org/abs/2301.04104) —— DreamerV3：一套超参打通 150+ 任务。
4. Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. *Nature*. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) —— MuZero：用 learned model + MCTS 打通棋类和 Atari。
5. Tassa, Y., et al. (2018). DeepMind Control Suite. *arXiv:1801.00330*. [链接](https://arxiv.org/abs/1801.00330) —— DMC：连续控制的标准 benchmark。
