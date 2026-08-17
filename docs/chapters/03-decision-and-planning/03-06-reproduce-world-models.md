# 3.6 动手复现 World Models

> **本节目标**：在 CarRacing 赛道上，用不到原文百分之一的规模，完整跑通 World Models 的 V-M-C 管线。从随机数据开始，训练一个压缩器、一个想象器、一个进化出来的控制器，最后把它们接回真实环境，亲眼看到「在想象中训练」是怎样发生的。

> **本节代码**：[训练脚本](https://github.com/walkinglabs/hands-on-world-models/blob/main/scripts/run_carracing.py)

> **前置知识**：你已经读过绪论，知道 V-M-C 三组件各自做什么。这一节把它们真跑一遍。

---

2018 年，David Ha 与 Jürgen Schmidhuber 在 NeurIPS 发表了论文 *Recurrent World Models Facilitate Policy Evolution* [1]，并配套了一篇交互式文章 [World Models](https://worldmodels.github.io/) [2]。文章里有一个赛车 demo：一辆小车在赛道上飞驰，画面模糊但动作流畅。作者说，这不是录屏，是模型在「做梦」——一个 867 个参数的线性控制器，完全在想象中学会了开车。

你当时大概和我一样，第一反应是：「这怎么可能？867 个参数？线性控制器？在梦里学的？」

这一节，我们要亲手把这件事再做一遍。规模打折，原理不打折。跑完之后，你会对「在想象中训练」这句话有完全不同的理解。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/carracing-initial.png" alt="CarRacing 目标世界" style="max-width:400px; border:1px solid #ddd; border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">这就是我们要让模型学会的「世界」：红色小车在灰色赛道上行驶，周围是绿色草地。模型从未见过物理定律，它要从像素流里自己发现「踩油门车会加速、打方向盘车会转弯」。</div>
</div>

## 为什么选 CarRacing

World Models 原文有两个实验：CarRacing 赛车和 VizDoom 躲火球。我们选赛车，原因有两个：

**第一，观测就是像素。** 赛道画面是 96×96×3 的图像，V 必须学会把它压成 32 维 latent。压缩的意义立刻可见——12,288 个像素变成 32 个数，信息量压缩了 384 倍。

**第二，随机策略就能收集数据。** 世界模型学的是「世界如何响应动作」，不是「如何开好车」。乱开乱撞的轨迹同样是合法的物理样本。你不需要先有一个会开车的策略，只需要让世界告诉你「踩油门之后画面怎么流动」。

动作空间是三维连续向量：方向盘 \([-1, 1]\)、油门 \([0, 1]\)、刹车 \([0, 1]\)。每局最多 1,000 步（约 20 秒），冲出赛道或长时间停滞会提前结束。

原文收集了 10,000 次随机 rollout、合计约 1,000 万帧。课程脚本默认 400 次，普通笔记本 CPU 上约 1–3 小时可完成全流程。

## 第一步：收集随机数据

世界模型学的是物理规律，不是驾驶技术。数据不需要来自专家——乱开乱撞的轨迹同样是合法的物理样本。

脚本用一个「温和随机」策略收集数据：方向盘均匀随机，目标速度每 20 步左右重采一次（让车偶尔沿直线走一段，而不是原地打转），刹车以 10% 概率踩下。每步把观测缩放为 64×64×3（area 插值，与原文一致），连同动作与奖励一起存档。

```python
# 简化的数据收集循环
for episode in range(num_rollouts):
    observation, _ = env.reset()
    for step in range(max_steps):
        action = np.array([
            np.random.uniform(-1, 1),      # 方向盘
            target_speed,                   # 油门
            float(np.random.rand() < 0.1)  # 刹车
        ])
        observation, reward, terminated, truncated, _ = env.step(action)
        observation = resize_frame(observation)  # 96x96 → 64x64
        # 存储 (observation, action, reward)
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/random-rollout.png" alt="随机策略数据" style="max-width:800px; border:1px solid #ddd; border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">图 0：随机策略 rollout 的一小段。上排：5 帧画面——车在赛道上乱开，有时冲出赛道；中排：动作条（方向盘红=左/绿=右，油门绿色高度，刹车红色块）；下排：奖励条（绿=正奖励/红=负奖励）。这就是喂给 V 和 M 的全部数据：没有专家示范，只有乱开乱撞的轨迹。</div>
</div>

默认 400 局约收集 40 万帧。这些数据唯一的用途，是让 V 和 M 看清这个世界：路面长什么样、踩油门以后画面如何流动、冲出赛道之前发生了什么。

**运行这一步，你会看到什么？** 脚本打印 `== 1/5 收集数据：随机策略 rollout`，然后逐局输出。400 局大约需要 5-10 分钟。收集完成后，`episodes` 列表里装着 400 条轨迹，每条轨迹是一串 (帧, 动作, 奖励) 的序列。

## 第二步：训练 V，一个 ConvVAE

V 的任务是把每一帧观测压成 32 维 latent。脚本用四层卷积做编码器、四层转置卷积做解码器，中间是均值与对数方差两个线性头——结构与 Kingma & Welling 的 Auto-Encoding Variational Bayes [3] 一致。

训练目标是 ELBO（Evidence Lower Bound）：

$$
\mathcal{L}_{V} = \mathbb{E}_{q(z \mid x)}\bigl[-\log p(x \mid z)\bigr] + \beta \cdot \mathrm{KL}\bigl(q(z \mid x)\,\|\,\mathcal{N}(0, I)\bigr)
$$

第一项是重建误差，第二项把后验拉向标准正态。KL 权重取 \(\beta = 0.001\)，是一个很温和的正则：latent 不必完全像白噪声，但要足够平滑——后面 M 要在这个空间里想象，C 要在里面采样行动，坑坑洼洼的 latent 会让两者一起摔跤。

采样用重参数化技巧，把随机性从参数路径上挪走：

$$
z = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

这样一来 \(z\) 对 \(\mu, \sigma\) 可微，重建误差的梯度就能一路流回编码器。

**运行这一步，你会看到什么？** 脚本打印 `== 2/5 训练 V：ConvVAE`，然后逐 epoch 输出重建损失。10 个 epoch 大约需要 10-20 分钟。训练完成后，`vae.pt` 保存了模型权重。

学成后，一帧 12,288 个像素被压成 32 个数。解码出来的画面是模糊的——赛道线、车身、弯道大致可辨，细节丢失殆尽。这正是绪论说的「只保留决策需要的信息」：M 和 C 关心的从来不是草地的纹理。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/vae-reconstruction.png" alt="VAE 重建对比" style="max-width:600px; border:1px solid #ddd; border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">图 1：VAE 重建对比。左：原图（96×96）；右：编码再解码后的重建。赛道、车身、草地大致可辨，但细节丢失——这正是「只保留决策需要的信息」。</div>
</div>

**一个值得做的实验**：把训练好的 VAE 接上随机 latent，看看解码出来的画面长什么样。你会发现，随机采样的 latent 解码出来的画面几乎是噪声——因为 VAE 学到的 latent 空间是连续的、平滑的，随机采样大概率落在训练分布之外。这正是 M 要在里面想象的空间。

## 第三步：训练 M，一个 MDN-RNN

M 建模转移分布 \(P(z_{t+1}, r_{t+1} \mid a_t, z_t, h_t)\)。结构是一个 GRU 单元加上混合高斯输出头：\(z_t\) 与 \(a_t\) 拼接后送入 GRU 得到新隐状态 \(h_{t+1}\)，再由三个线性头输出五个高斯分量的均值、方差和权重，另有一个单高斯头预测奖励。混合密度网络（Mixture Density Network）的思想来自 Bishop (1994) [4]。

对下一时刻的 latent，预测的不是一个值而是一个混合分布：

$$
P(z_{t+1} \mid a_t, z_t, h_t) = \sum_{i=1}^{5} \pi_i \, \mathcal{N}\bigl(z_{t+1}; \mu_i, \sigma_i^2\bigr)
$$

为什么必须用混合高斯？因为未来本质上是多峰的：弯道前既可以左拐也可以右拐，火球可能向左也可能向右。单一高斯的最优解站在两个峰的中间，而「中间」恰恰是永远不会发生的轨迹。混合高斯允许模型同时举起多种未来。

训练损失是混合密度的负对数似然加上奖励的高斯负对数似然：

$$
\mathcal{L}_{M} = -\log \sum_{i=1}^{5} \pi_i \, \mathcal{N}\bigl(z_{t+1}; \mu_i, \sigma_i^2\bigr) \;+\; \bigl[-\log \mathcal{N}\bigl(r_{t+1}; \mu_r, \sigma_r^2\bigr)\bigr]
$$

训练方式是 teacher forcing：每一步都从零隐状态出发、喂真实的 \(z_t\)，只监督单步转移；多步记忆的延续留给部署时的梦境 rollout。M 约 31 万参数（原文约 42 万）。

**温度**控制采样时的确定性：分量权重按 \(\pi_i^{1/\tau}\) 重新归一，高斯标准差乘上 \(\tau\)。\(\tau\) 越小，梦境越确定、越「敢于自信」；\(\tau\) 越大，梦境越接近训练数据里的真实噪声。原文的 τ 温度实验（梦境 2086 vs 193、918 vs 1092，见绪论）就是在这根旋钮上做出来的。

**运行这一步，你会看到什么？** 脚本打印 `== 3/5 训练 M：MDN-RNN`，然后逐 epoch 输出负对数似然。10 个 epoch 大约需要 5-10 分钟。训练完成后，`mdn.pt` 保存了模型权重。

**一个值得做的实验**：用训练好的 M 做 free-running rollout——从一帧真实画面开始，让 M 自己预测下一步，再把预测的下一步喂回去，循环 100 步。你会发现，前 20 步的预测还算合理，但越往后画面越扭曲，最终变成完全无法辨认的噪声。这就是**复合误差**：每一步的微小偏差滚进下一步，越滚越大。M 的单步预测很准，但多步一致性全靠 V 与 z 分布的稳定性硬撑。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/mdn-free-running.png" alt="M 的 free-running rollout" style="max-width:800px; border:1px solid #ddd; border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">图 2：M 的 free-running rollout。从左到右：第 0、10、30、60、99 步。前几步还算合理，越往后越扭曲——这就是<strong>复合误差</strong>：每一步的微小偏差滚进下一步，越滚越大。</div>
</div>

## 第四步：进化 C，一个 867 参数的线性控制器

C 小到令人意外：

$$
a_t = \mathrm{clip}\bigl(W\,[z_t;\, h_t] + b,\; -1,\; 1\bigr), \qquad W \in \mathbb{R}^{3 \times 288}
$$

参数一共 \((32 + 256) \times 3 + 3 = 867\) 个。这是原文的核心设计：把所有复杂度押进 M，让 C 小到可以用无梯度方法优化。

脚本用几十行实现了一个教学版 CMA-ES（Covariance Matrix Adaptation Evolution Strategy）[5]。它在 867 维参数空间里维护一个高斯分布，逐代做三件事：

1. **采样**：从 \(\mathcal{N}(m, \sigma^2 C)\) 采出一个种群（默认 32 个个体）。
2. **评估**：每个个体去 M 的梦里开 4 次车、每次 200 步，fitness 是累计预测奖励的平均。
3. **更新**：按 fitness 排序，靠前的个体加权重构均值，并沿它们的方向更新协方差矩阵——搜索分布就这样一代代学会「朝好参数的方向倾斜」。

$$
m \leftarrow \sum_i w_i \, x_{(i)}, \qquad C \leftarrow (1 - c)\,C + c \sum_i w_i \, (x_{(i)} - m)(x_{(i)} - m)^{\top}
$$

梦境评估的细节：从一帧真实画面编码出的 \(z\) 出发，C 出动作、M 想象下一状态并预测奖励，如此滚动 200 步；C 的每个动作还注入强度 0.1 的高斯噪声，相当于一点探索。整个过程不碰真实环境——这正是「在想象中训练」的最原始形态。

**运行这一步，你会看到什么？** 脚本打印 `== 4/5 进化 C：CMA-ES`，然后每 25 代输出当前最佳 fitness。300 代进化大约需要 30-60 分钟（这是最耗时的一步）。训练完成后，`controller.npy` 保存了 867 个参数。

**一个值得做的实验**：把进化出的 `controller.npy` 换成全零参数，再跑一次真实评估。你会发现，全零控制器的真实分数远低于随机基线——这说明 C 确实学到了东西，而不是一切来自 V 和 M。

## 第五步：回到现实

把进化出的 C 接回真实赛道闭环：V 编码真实帧 → C 出动作（此时不加噪声）→ 环境反馈 → M 更新记忆。脚本同时报告三个分数：

| 指标                  | 含义                                       |
| --------------------- | ------------------------------------------ |
| `dream_score`         | M 想象中的累计预测奖励——C 以为自己能开多好 |
| `real_score`          | 真实环境 8 局平均回报——它实际能开多好      |
| `random_policy_score` | 随机策略的真实回报——及格线                 |

注意一个危险的直觉：**梦境分数高不等于真实分数高**。C 完全可能钻 M 的空子，找到一条只在想象中畅通的路线——这就是绪论说的**模型利用（model exploitation）**。判断复现是否成功，看的是 `real_score` 是否超过 `random_policy_score`；两者与 `dream_score` 的差距，正是复合误差与模型利用的合计账单。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/real-evaluation.png" alt="真实环境评估" style="max-width:800px; border:1px solid #ddd; border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">图 3：真实环境评估的关键帧。进化出的 C 在真实赛道上闭环：V 编码真实帧 → C 出动作 → 环境反馈 → M 更新记忆。</div>
</div>

**运行这一步，你会看到什么？** 脚本打印 `== 5/5 真实环境评估`，然后输出三个分数和完整的 `metrics.json`。如果一切顺利，你应该看到 `real_score` 明显高于 `random_policy_score`——这意味着，在梦里进化出来的策略，确实能在真实环境里开得更好。

**一个值得做的实验**：分别用 `--no-memory` 和 `--temperature 0.1` 完整跑一遍，对比三个分数的变化。`--no-memory` 去掉了 M 的记忆，控制器退化为只看当前帧，你应该看到 `real_score` 下降——这是「记忆」价值的直接证据。`--temperature 0.1` 让梦境更确定，进化更容易，但控制器可能过拟合梦境的自信，你应该看到 `dream_score` 上升但 `real_score` 下降——这是「梦境与现实差距」的直接证据。

## 运行与产物

运行：

```bash
python scripts/run_carracing.py --output runs/carracing-world-model
```

跑完在输出目录生成五个文件：

- `vae.pt`：训练好的 ConvVAE 权重
- `mdn.pt`：训练好的 MDN-RNN 权重
- `controller.npy`：进化出的 867 个参数
- `metrics.json`：梦境分数、真实分数、随机基线、VAE 重建损失、MDN 负对数似然等全部指标
- `manifest.json`：实验元数据（命令、种子、耗时、checkpoint 的 sha256——第 8 章的运行证据规范）

完整参数：400 次数据收集、VAE 与 MDN 各训练 10 个 epoch、300 代进化、每代 32 个个体、每个个体 4 条 200 步的梦境轨迹。

## 对照实验与原文结论

脚本的每个旗标都对应原文的一个结论：

| 命令                | 对照什么          | 预期现象                                                          |
| ------------------- | ----------------- | ----------------------------------------------------------------- |
| 默认（带记忆）      | 完整 V-M-C        | 真实分数明显高于随机基线                                          |
| `--no-memory`       | 去掉 M 的记忆 h_t | 控制器退化为只看当前帧，分数下降——对应原文消融 632 vs 906         |
| `--temperature 0.1` | 低温梦境          | 梦境确定、进化容易，但控制器可能过拟合梦境的自信——对应原文 τ 实验 |

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/comparison.png" alt="官方 vs 复现对比" style="max-width:800px; border:1px solid #ddd; border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">图：CarRacing 累计奖励对比。蓝色为原文数据（CarRacing-v0 / gym 0.9），橙色为我们的复现（CarRacing-v3 / gymnasium 1.x，50 rollouts、30 代进化）。环境版本不同导致奖励尺度不可直接比较——v3 的奖励更稀疏、负奖励更多，但相对趋势一致：有记忆 > 随机 > 无记忆。</div>
</div>

## 已知简化与坑

教学版有几处刻意的简化，复现结果与原文数值对不上时，先从这里找原因：

- **M 训练时隐状态恒为零**。梦境里 GRU 的记忆在延续，但训练从未监督过「利用记忆做多步预测」，多步一致性全靠 V 与 z 分布的稳定性硬撑。这是复合误差的一个直接来源，也是 Dreamer 要用可微梦境端到端训练的动机之一。
- **数据来自温和随机策略**，而非纯粹的均匀随机；数据分布与真实驾驶轨迹有别。
- **梦境只有 200 步**，原文最长 2,100 步——进化出的控制器只学会了短期行为。
- **环境版本不同**：原文用 gym 0.9 的 CarRacing-v0，课程用 gymnasium 1.x 的 CarRacing-v3，物理细节与奖励有细微差别。原文的具体分数（906、210）应视为参考值，不是精确靶子。

## 扩展练习

跑通默认配置后，按从便宜到昂贵的顺序推荐：

1. **全零控制器对照**：把 `controller.npy` 换成全零参数再跑一次真实评估——亲眼确认 C 确实学到了东西，而不是一切来自 V 和 M。
2. **温度扫描**：分别用 `--temperature 0.1` 与 `--temperature 1.5` 完整跑一遍，对比梦境分数与真实分数的差距如何变化。
3. **数据量曲线**：把 `--rollouts` 提到 1,000，观察 V 的重建损失与真实分数的关系——数据多少开始「够用了」？
4. **换 MLP 控制器**：把线性控制器换成一层 MLP（几十行改动），看 867 参数的预算是否是瓶颈。

跑通之后，推荐接着做 [PA1-A · 做出一台 Dreamer-lite](/assignments/pa1-a)：那台模型不再进化 C，而是用可微的想象直接训练策略，你会亲身体会「进化」与「梯度下降」两条路线手感的不同。

## 本节小结

- **World Models 的完整管线**：随机数据 → V 压缩 → M 想象 → C 进化 → 真实部署，五步缺一不可，每一步都有明确的数学形状。
- **V 用重参数化技巧训练 ELBO**，β 控制 latent 的平滑程度；**M 用混合高斯建模多峰未来**，并同时预测奖励。
- **C 的 867 个参数由 CMA-ES 在梦境中进化**，fitness 是 M 预测的累计奖励，全程不碰真实环境。
- **温度 τ 是梦境确定性的旋钮**：低温易进化、易过拟合梦境，高温更接近真实噪声。
- **带记忆（[z, h] 输入）显著优于只看当前帧**，这是「记忆」价值的直接证据。
- **梦境分数高不等于真实分数高**——两者的差距就是复合误差与模型利用的合计账单。

从 2018 年的交互式文章，到你亲手跑通的训练脚本，World Models 的核心思想从未改变：**在行动之前，先在内部预见行动的后果**。而这条思想的后来的发展——Dreamer 的可微想象 [6]、MuZero 的隐式搜索 [7]、Genie 的可玩世界——都将在接下来的章节里，由你亲手实现。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/dream-generation.png" alt="梦境生成的世界" style="max-width:800px; border:1px solid #ddd; border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">训练好的世界模型生成的「梦境世界」：C 在 M 的想象中开了 200 步，全程未接触真实环境。从左到右，画面从清晰逐渐模糊——复合误差在累积，但赛道、车身、草地的结构始终可辨。这就是 M 学到的「世界」：不完美，但足够让 C 在里面学会开车。</div>
</div>

原文与训练细节见 Ha & Schmidhuber (2018) [1][2]，官方训练代码见 [WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments)（MIT 协议）。

## 后续工作：从 World Models 出发，世界模型走了多远

World Models 的 V-M-C 管线证明了一件事：**可以在想象中学会控制**。但它留下了三个明显的短板，后来的工作逐一攻克了它们。

### 短板一：CMA-ES 进化太慢、太贵

CMA-ES 在 867 维空间里盲目搜索，300 代 × 32 个体 × 4 条梦境轨迹 = 几万次梦境 rollout。每次 rollout 还要跑 200 步 MDN-RNN，计算量巨大。而且进化策略没有梯度信号，不知道「往哪个方向改参数会更好」，只能靠种群统计量慢慢摸索。

**Dreamer 系列**（Hafner et al. 2019–2023）[6][8] 把 C 换成了可微的策略网络，直接在 M 的梦境里做梯度下降。M 的 GRU 被替换成更强大的 RSSM（Recurrent State Space Model），能同时维护确定性隐状态和随机 latent；策略网络从隐状态里读出动作，价值网络评估隐状态的好坏，两者通过梦境 rollout 的梯度联合优化。DreamerV1 在 DeepMind Control Suite 上超过了当时所有的 model-free 方法；DreamerV2 用离散 latent 打通了 Atari；DreamerV3 一套超参跑遍 150+ 任务，成为「可复现世界模型」的标杆。

关键区别：World Models 的 C 是**无梯度进化**，Dreamer 的 C 是**可微想象 + 策略梯度**。前者像蒙眼摸索，后者像看着地图走。

### 短板二：像素重建是负担，不是帮助

World Models 的 V 必须把 96×96 像素压成 32 维再重建回来。但 M 和 C 关心的从来不是「草地的纹理」，而是「弯道在哪里、车在什么位置」。强迫 V 重建像素，相当于让模型把算力浪费在决策无关的细节上。

**MuZero**（Schrittwieser et al. 2020）[7] 直接扔掉了 V 的解码器。它的 M 不预测像素，只预测隐状态转移和奖励；它的「规划」不在像素空间里做，而是在隐空间里跑蒙特卡洛树搜索（MCTS）。 Atari、围棋、国际象棋、将棋，一套架构全部打通，发在 *Nature* 上。MuZero 的核心洞察是：**世界模型不需要重建世界，只需要重建决策需要的信息**。

**SimPLe**（Kaiser et al. 2020）[9] 走了另一条路：保留像素预测，但用视频预测的视角来做——把 M 的输出当成「下一帧视频」，用更复杂的视频生成架构（卷积 LSTM + 残差连接）来提升多步一致性。它在 Atari 上证明了：如果像素预测足够准，model-based RL 可以超过 model-free。

### 短板三：确定性模型撑不住长 rollout

World Models 的 M 用混合高斯建模多峰未来，但 GRU 的隐状态是确定性的。训练时 teacher forcing 只监督单步转移，部署时 free-running 的复合误差会迅速累积——你在图 2 里已经看到了：100 步后画面变成噪声。

**Stochastic MuZero**（Schrittwieser et al. 2021）[10] 给隐状态加入了随机性：每一步的隐状态转移不再是确定函数，而是从后验分布里采样。这让模型能表达「同样的动作可能导致不同的结果」，长 rollout 的稳定性显著提升。

**Genie**（Bruce et al. 2024）[11] 走得更远：它用视频生成模型（ViT + 离散 token + 扩散/自回归解码）直接生成可交互的像素世界。用户按一个键，Genie 生成下一帧；再按一个键，再生成一帧。它不再区分 V、M、C——整个系统就是一个「按动作条件生成视频」的大模型。Genie 从 YouTube 游戏视频里无监督学出了 2D 平台游戏的物理规律，用户可以在生成的世界里真正「玩」起来。

**GameNGen**（Valevski et al. 2024）[12] 在 Genie 的基础上做到了实时：用扩散模型 + 自回归 token 的混合架构，以 20 FPS 生成 DOOM 游戏画面。世界模型从「辅助决策的工具」变成了「可以玩的游戏引擎」。

### 从赛车到自动驾驶

CarRacing 是自动驾驶的玩具版。真正的驾驶世界模型要处理多视角相机、3D 几何、长时序一致性、安全约束。几个代表性工作：

- **GAIA-1**（Hu et al. 2023）[13]：Wayve 的驾驶世界模型，用多视角相机输入 + 离散 token + 自回归生成，能生成逼真的驾驶视频，并支持动作条件控制。
- **DriveDreamer**（Wang et al. 2023）[14]：用世界模型做驾驶数据增强——在模型生成的「梦境驾驶视频」里训练自动驾驶策略，再迁到真实环境。
- **UniSim**（Yan et al. 2024）[15]：Google 的通用模拟器，从真实传感器数据里学出可交互的 3D 世界，支持自动驾驶、机器人导航等多种任务的仿真。

### 一句话总结这条线

World Models 提出了问题：**能不能在想象中学会控制？**
Dreamer 回答了「能，而且用梯度比进化更高效」；MuZero 回答了「连像素都不用重建，隐空间就够了」；Genie 回答了「世界模型本身就可以是一个可玩的世界」。从 867 个参数的线性控制器，到几十亿参数的生成式世界引擎，核心思想从未改变——**在行动之前，先在内部预见行动的后果**。

## 参考文献

1. Ha, D., & Schmidhuber, J. (2018). Recurrent World Models Facilitate Policy Evolution. *NeurIPS 2018*. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
2. Ha, D., & Schmidhuber, J. (2018). World Models. 交互式文章：[worldmodels.github.io](https://worldmodels.github.io/) —— 本篇正文中的 V-M-C 框架、CarRacing 实验设置与图 1 均来自此文（图表为 CC-BY 4.0 授权）。配套代码：[WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments)。
3. Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR 2014*. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
4. Bishop, C. M. (1994). Mixture Density Networks. *Aston University Technical Report NCRG/94/004*. [链接](https://publications.aston.ac.uk/id/eprint/373/)
5. Hansen, N. (2001). Completely Derandomized Self-Adaptation in Evolution Strategies. *Evolutionary Computation*, 9(2), 159–195. [链接](https://doi.org/10.1162/106365601750190398)
6. Hafner, D., et al. (2019). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR 2020*. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) —— 用可微梦境端到端训练策略，替代 CMA-ES 进化。
7. Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. *Nature*, 588, 604–609. [链接](https://doi.org/10.1038/s41586-020-03051-4) —— MuZero：不重建像素，只在隐空间里做蒙特卡洛树搜索。
8. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. *arXiv:2301.04104*. [链接](https://arxiv.org/abs/2301.04104) —— DreamerV3：一套超参打通 150+ 任务，可复现世界模型的工程标杆。
9. Kaiser, Ł., et al. (2020). Model-Based Reinforcement Learning for Atari. *ICLR 2020*. [arXiv:1903.00374](https://arxiv.org/abs/1903.00374) —— SimPLe：用视频预测做 model-based RL，Atari 上超过 model-free。
10. Schrittwieser, J., et al. (2021). Offline AlphaZero Implicitly Models Planning for Actions. *arXiv:2109.09179*. [链接](https://arxiv.org/abs/2109.09179) —— Stochastic MuZero：隐状态随机化，提升长 rollout 稳定性。
11. Bruce, J., et al. (2024). Genie: Generative Interactive Environments. *ICML 2024*. [arXiv:2402.15391](https://arxiv.org/abs/2402.15391) —— 从 YouTube 视频无监督学出可交互的 2D 游戏世界。
12. Valevski, D., et al. (2024). GameNGen: Auto-regressive Neural Engine for Interactive Generation of DOOM. *arXiv:2406.13843*. [链接](https://arxiv.org/abs/2406.13843) —— 20 FPS 实时生成 DOOM 游戏画面。
13. Hu, A., et al. (2023). GAIA-1: A Generative World Model for Autonomous Driving. *arXiv:2309.17080*. [链接](https://arxiv.org/abs/2309.17080) —— Wayve 的驾驶世界模型。
14. Wang, X., et al. (2023). DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. *arXiv:2309.09777*. [链接](https://arxiv.org/abs/2309.09777) —— 用梦境驾驶视频做数据增强。
15. Yan, Y., et al. (2024). UniSim: Learning Interactive Real-World Simulators. *ICLR 2024*. [arXiv:2310.20520](https://arxiv.org/abs/2310.20520) —— Google 的通用可交互世界模拟器。
