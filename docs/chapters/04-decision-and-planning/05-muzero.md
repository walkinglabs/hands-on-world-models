# 4.5　MuZero 与蒙特卡洛树搜索

四子棋、围棋、国际象棋有共同特点：规则完全确定，没有随机风，也不需要重画棋盘纹理。Dreamer 用 decoder 帮助训练状态，但棋类规划根本不需要还原棋子的颜色和质感，只需要知道「这一步之后的 reward、合法动作概率和价值」。

MuZero 因此学一种 **value-equivalent** 表示：状态不必还原完整观察，只要保留搜索需要的信息。

## 三个网络，各管一件事

MuZero 把世界模型拆成三个函数，都作用在隐状态上：

$$
\begin{aligned}
\text{Representation:}&\quad s_0 = h(o_1, o_2, \dots, o_t) \\
\text{Dynamics:}&\quad s_{t+1},\, r_t = g(s_t,\, a_t) \\
\text{Prediction:}&\quad \mathbf{p}_t,\, v_t = f(s_t)
\end{aligned}
$$

- representation 把观察历史压成初始隐状态 $s_0$。
- dynamics 给定 $s_t$ 和动作 $a_t$，在隐空间推出 $s_{t+1}$，并预测这一步的 reward $r_t$。
- prediction 给出策略先验 $\mathbf{p}_t$（每个合法动作的概率）和价值 $v_t$。

动作是搜索树的边。dynamics 在隐空间展开下一节点，prediction 为搜索提供先验和值。

## 训练目标：三个预测头的加权和

MuZero 不重建观察，三个监督信号都来自搜索和自我对弈的真实结果：

$$
\mathcal{L} = \sum_{t}\bigl[\,\ell_{\text{reward}}(r_t, u_t) + \ell_{\text{value}}(v_t, z_t) + \ell_{\text{policy}}(\mathbf{p}_t, \boldsymbol{\pi}_t)\,\bigr]
$$

其中 $u_t$ 是真实 reward，$z_t$ 是从这一步开始的最终回报，$\boldsymbol{\pi}_t$ 是 MCTS 搜索后得到的改进策略（见下节）。reward 和 value 通常用 cross-entropy 形式的 twohot 编码，policy 用普通交叉熵：

$$
\ell_{\text{policy}}(\mathbf{p}_t, \boldsymbol{\pi}_t) = -\sum_a \boldsymbol{\pi}_t(a)\,\log \mathbf{p}_t(a)
$$

注意 $\boldsymbol{\pi}_t$ 不是人类标注，而是搜索本身产出的更优动作分布——模型在学「自己搜索出来的结果」。

## MCTS：在隐空间长出一棵搜索树

蒙特卡洛树搜索（MCTS）反复执行四步：选择、扩展、评估、回传。

**选择**从根节点沿树下行，用 PUCT 公式决定走哪条边：

$$
a = \arg\max_a\Bigl(Q(s,a) + c_{\text{puct}}\, \mathbf{p}(a)\,\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}\Bigr)
$$

$Q(s,a)$ 是这条边的平均价值，$N(s,a)$ 是访问次数，$\mathbf{p}(a)$ 是 prediction 给的先验。后半项像一个「探索奖金」：访问少的动作、先验高的动作，更容易被选去试。

到达叶子节点后，**扩展**用 dynamics 推出 $s_{t+1}$，**评估**用 prediction 得到 $v$，**回传**把 $v$ 沿路径加回每个祖先的 $Q$。

重复几百次后，根节点的访问次数 $\boldsymbol{\pi}_t(a) \propto N(s_t, a)$ 形成一个比原始 $\mathbf{p}$ 更集中的分布。这个 $\boldsymbol{\pi}_t$ 既是要执行的动作分布，也是上面的训练目标。

## 与 Dreamer 的不同

| 维度       | Dreamer                     | MuZero                     |
| ---------- | --------------------------- | -------------------------- |
| 怎么用模型 | 可微想象，训练 actor-critic | MCTS 在隐空间搜索          |
| 典型动作   | 连续控制                    | 离散动作                   |
| 状态约束   | decoder 重建观察            | 只预测 reward/policy/value |

Dreamer 的状态受观察重建约束，保留了大量画面细节。MuZero 主动舍弃与 reward、policy、value 无关的信息。代价是隐状态更难用人眼解释，未来想加新任务（比如「描述棋面」）时，可能发现相关信息早被丢掉了——这正是 value-equivalent 表示的固有风险。

## Mini-MuZero 的课程范围

选修实验用四子棋。棋盘规则提供准确环境，我们实现 representation、dynamics、prediction 和一个小型 MCTS，比较「直接 policy」与「加搜索」的胜率。

这不等于复现工业规模的 MuZero。课程目标是看清一件事：**改变预测目标以后，模型保留的信息和评价方式也随之改变**。

## 小结

- [ ] MuZero 不重建观察，只预测 reward、policy 和 value，学的是 value-equivalent 表示。
- [ ] MCTS 用 PUCT 在隐空间展开树，访问次数形成比原策略更集中的改进策略，作为训练目标。
- [ ] 这种表示适合当前规划目标，但不保证保留所有世界信息。

到这里，决策路线的五种思路——潜在状态、RSSM、PlaNet/CEM、Dreamer、MuZero——已经串成一条线。动手把这套接口跑通，见 [A2：在想象中规划与行动](/chapters/04-decision-and-planning/07-decision-and-planning)。
