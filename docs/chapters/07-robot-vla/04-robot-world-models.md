# 7.4　机器人世界模型

直接 VLA 给出动作就结束了。它可以很快，却不一定在执行前比较过“从左侧接近”和“从右侧接近”各自会发生什么。

我们再加一个小型后果模型，让 VLA 先提出几个候选动作，再由后果模型预测各自的下一步。这就是**世界模型检查器**（world-model checker）。

## 两个模块，各管一件事

所谓检查器，就是把“想动作”和“想后果”拆成两个模型：

```text
VLA：          image + instruction + proprio → candidate actions
World model：  current state + candidate action → next state / collision
```

世界模型只学一件事：在当前状态下执行某个动作，下一状态会变成什么、会不会碰撞。语言负责指定目标，不进入这个转移。

## 动作后果的转移方程

设当前状态为 $s_t$，候选动作为 $a_t$，世界模型预测下一状态 $\hat s_{t+1}$ 和碰撞概率 $\hat c_{t+1}$。最朴素的写法是确定性转移加上一个分类头：

$$
\hat s_{t+1}=f_\theta(s_t,a_t),\qquad \hat c_{t+1}=\sigma\bigl(g_\phi(s_t,a_t)\bigr)\in(0,1),
$$

其中 $\sigma$ 是 sigmoid。状态部分用回归损失，碰撞部分用二分类交叉熵：

$$
\mathcal{L}_{\text{state}}=\lVert \hat s_{t+1}-s_{t+1}\rVert_2^2,\qquad
\mathcal{L}_{\text{coll}}=-\,c_{t+1}\log \hat c_{t+1}-(1-c_{t+1})\log(1-\hat c_{t+1}).
$$

两者加权相加就是一步后果模型的训练目标：

$$
\mathcal{L}_{\text{outcome}}=\mathcal{L}_{\text{state}}+\lambda\,\mathcal{L}_{\text{coll}}.
$$

权重 $\lambda$ 决定碰撞要多大力度去学。

## 想象 rollout：多步往后看

把一步后果接起来，就能在脑子里走 $H$ 步，这就是**想象 rollout**（imagined rollout）。从一个真实状态 $s_t$ 出发，反复喂入候选动作序列：

$$
\hat s_{t+1}=f_\theta(s_t,a_t),\quad \hat s_{t+2}=f_\theta(\hat s_{t+1},a_{t+1}),\quad \ldots,\quad \hat s_{t+H}=f_\theta(\hat s_{t+H-1},a_{t+H-1}).
$$

每一步还累积一个碰撞概率。整条想象轨迹的总风险，可以近似成各步不碰撞概率的乘积：

$$
P_{\text{safe}}=\prod_{k=1}^{H}\bigl(1-\hat c_{t+k}\bigr).
$$

有了 $\hat s_{t+H}$ 和 $P_{\text{safe}}$，planner 或 reranker 就能比较“安全地靠近目标”与“安全却原地不动”两条路线，而不是只看一步。

## 为什么数据要包含坏动作

专家示范大多是成功动作。若后果模型只见过好动作，它从未见过碰撞长什么样，也就无法判断一个动作坏不坏，只会在训练分布之外自信外推。

桌面生成器可以从同一状态分别执行安全与不安全动作，记录 $s_{t+1}$、成功与否、碰撞与否。坏动作是 checker 的必要监督。要让数据里的碰撞比例既不是 0 也不是 1，模型才有东西可学。

## 直接执行与 lookahead 的公平比较

固定同一个候选生成器和相同的计算预算，比较三种策略：

1. **直接执行**最高概率动作；
2. 用一步后果模型重新排序，即 **learned checker**；
3. 用真实模拟器后果排序，作为**上限参考**。

设三者的真实碰撞率分别为 $R_{\text{direct}}$、$R_{\text{learned}}$、$R_{\text{oracle}}$。我们希望看到：

$$
R_{\text{oracle}}\;\le\;R_{\text{learned}}\;\le\;R_{\text{direct}}.
$$

如果 learned checker 没有靠近 oracle 上限，要检查模型误差和 OOD，而不是只增加候选数量。

## Checker 也会被利用

候选生成器可能找到后果模型错判为安全的动作——模型说不会撞，真实却撞了。这叫 checker 被利用。

对策是不确定性阈值、动作约束、真实反馈和失败数据回流。世界模型不是安全保证。真实机器人仍需要独立的速度、力矩、工作空间和急停约束。

还有一种容易忽略的失败：一步 checker 可以一直选择不会碰撞、却离目标更远的动作。安全但毫无进展。课程实验因此同时记录真实碰撞率和每步目标进展，只有安全没有任务进展，说明规划目标或 horizon 还不够。

## 小结

- [ ] VLA 输出动作，世界模型预测动作后果，两者可以级联。
- [ ] 想象 rollout 把一步后果接成 $H$ 步轨迹，并累积碰撞风险。
- [ ] 后果数据必须包含失败与坏动作。
- [ ] learned checker 要与直接执行和真实模拟器上限比较。
- [ ] 世界模型检查不能代替独立的安全约束。

本章动手实验见 [7.8 动手：从零实现 VLA 与世界模型检查器](/chapters/07-robot-vla/08-robot-vla)：第一份 Notebook 搭一台小型的视觉-语言-动作模型，第二份 Notebook 用后果模型在行动前检查候选动作。空间与驾驶部分见 [8.7 动手：三维重建与占用预测](/chapters/08-spatial-worlds/07-spatial-world)。
