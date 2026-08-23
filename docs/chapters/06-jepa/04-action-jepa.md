# 6.4　动作条件 JEPA

同一个机器人状态，向左推和向右推会得到截然不同的下一刻。被动 JEPA 只看视频，无从知道"动作"究竟是变化的原因，还是一段顺带录下来的伴随信号。

要让特征的未来听从动作，就得把动作显式地喂给 Predictor。这就是 **Action-JEPA**。

## 把动作加进预测

设历史特征为 $h = f_\theta(c)$，候选动作为 $a$，未来特征目标为 $\tilde{y} = \mathrm{sg}(f_{\bar\theta}(y))$。Action-JEPA 把 Predictor 写成 $g_\phi(h, a)$，损失：

$$
\mathcal{L}_{\text{Action-JEPA}}
= \frac{1}{N}\sum_{i=1}^{N}\Big\|g_\phi\big(h_i,\,a_i\big) - \tilde{y}_i\Big\|_2^{2}.
$$

对比 5.1 的 JEPA 损失，唯一的差别是 Predictor 多吃了 $a$ 这一项。但这一项的含义不小：它要求模型把动作当成变化的原因，而不是画面里的另一个像素。

动作通常先过一个 embedding 层 $e$，得到 $e(a)\in\mathbb{R}^{d_a}$，再和 $h$ 拼起来或相加，交给 Predictor：

```text
history feature h ─┐
                    ├─ concat/add ─► Predictor g_φ ─► predicted future feature
action embedding e(a) ─┘
```

## 数据要求

要训上式，一条样本至少得有四件事：观察 $o_t$、动作 $a_t$、下一观察 $o_{t+1}$、时间戳 $t$。机器人数据还要补上 proprioception（本体感觉）、控制频率、执行延迟。

这是被动预训练与动作条件训练的分界。UCF101-mini 只有视频、没有动作标签，适合做被动预训练。要训 Action-JEPA，得换成 PixelWorld 这种带动作记录的合成数据，或真实的机器人轨迹。C2 正是在带动作的 PixelWorld 上训练的。

## 反事实检查：动作有没有进动态

Action-JEPA 是不是真把动作当回事，不能只看 loss。最直接的检查是**反事实**：固定同一历史和环境随机源，只替换动作 $a$。

设候选动作集合 $\{a^{(1)},\dots,a^{(K)}\}$，对应的预测特征为 $\hat{z}^{(k)} = g_\phi(h, a^{(k)})$。考察它们之间的差异：

$$
\Delta_{k} = \big\|\hat{z}^{(k)} - \hat{z}^{(1)}\big\|_2^{2}.
$$

如果所有 $\Delta_k$ 都接近 $0$，说明换动作几乎不改变预测——动作根本没进动态。这正是 C2 第 3 节检查的事情。

但只看 $\Delta_k$ 还不够，因为模型可能对动作敏感、却敏感在无关维度上。我们常常再训一个位置 probe $W$，把 $\hat{z}^{(k)}$ 映射成可解释的方块位置 $\hat{p}^{(k)}$，再看 $\hat{p}^{(k)}$ 是否随动作方向合理移动。

## 一个最小规划器

有了能区分动作的预测，就能做最朴素的规划：对每个候选动作 $a$ 预测一步特征 $\hat{z}=g_\phi(h,a)$，再用 probe 或 reward head 打分，选最高分：

$$
a^{*} = \arg\min_{a\in\mathcal{A}}\;\big\|\,W\,g_\phi(h, a) - g_{\text{goal}}\big\|_2^{2}.
$$

$g_{\text{goal}}$ 是目标位置。这就是一步 lookahead 的动作选择，C2 第 4 节把它实现成了一个 `argmin` 接口。

一个重要的诚实声明：probe 的训练集与测试集按 episode seed 分开。若 probe 只在训练 feature 上拟合得好，我们只能说表示记住了样本，不能说它保留了可迁移的位置。短期动作选择用的也是同一个 held-out probe——表示不可靠，选择就没有依据。

## 与 Dreamer 的边界

Action-JEPA 和 Dreamer（第 4 章）都能在特征空间预测未来，区别在侧重点：

- **Action-JEPA**：核心是非生成的特征预测目标，关注表示质量。它一般不训练 reward、continue 和 Actor-Critic。
- **Dreamer**：在 latent 里预测未来的同时，进一步训练 reward head、continue function、Actor 和 Critic，目标是提高真实回报。

换句话说，Action-JEPA 是世界模型的"表示半成品"，Dreamer 把这种表示推到了完整的决策管线。加入 Action-JEPA 之后，如果一步规划没能优于"保持原动作"基线，就要怀疑表示里没保留任务需要的可控信息。

## 小结

- Action-JEPA 把动作 $a$ 作为 Predictor 的条件项，让特征未来随动作分叉。
- 被动预训练与动作条件训练需要不同数据，前者无动作、后者有动作。
- 反事实检查与下游动作选择，才能说明特征真正支持控制。
- JEPA 重表示，Dreamer 重回报；二者在 latent 预测上接壤、目标不同。

[上一篇 5.3 视频 JEPA](./03-video-jepa.md) · [回到第 6 章](./index.md) · [动手：C2 动作条件特征预测](/chapters/06-jepa/06-jepa)
