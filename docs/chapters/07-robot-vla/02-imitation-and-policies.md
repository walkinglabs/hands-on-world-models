# 7.2　模仿学习与生成策略

一条抓取示范有 400 步。假设策略每一步有 $1\%$ 的概率给出一个稍微偏离示范的动作，那么整条轨迹里至少偏离一次的概率是 $1-0.99^{400}\approx 0.982$。也就是说，几乎每一次执行都会走到示范之外。

**行为克隆**（behavior cloning, BC）把示范当成监督学习：训练时它只见过专家附近的状态，执行时却必须为自己犯下的错误负责。这一节先说清这个缺口，再看动作分块与生成式策略各自补上了哪一块。

## 复合误差：偏差不会自己消失

设专家的状态访问分布是 $d_{\pi^*}$，学到的策略在单步上的期望误差是

$$
\epsilon=\mathbb{E}_{s\sim d_{\pi^*}}\bigl[\mathbb{1}\{\pi_\theta(s)\neq \pi^*(s)\}\bigr].
$$

这个量只在专家附近有定义，这正是问题所在。策略一旦偏离，后续状态就来自 $d_{\pi_\theta}$ 而不是 $d_{\pi^*}$，误差没有被测量过。经典分析给出的总代价上界是 $O(\epsilon H^2)$：一次偏离要赔的不是一步，而是剩下的 $H-t$ 步。

这就是**分布偏移**（covariate shift）与**复合误差**（compounding error）。代入数值检查一下：$\epsilon=0.01$、$H=400$ 时 $\epsilon H^2=1600$，而整条轨迹的最大代价只有 $400$——这个上界已经空了。它的价值不在数字，而在于指出哪些量可以动：$\epsilon$ 靠数据和容量降，$H$ 由任务给定，剩下唯一能改的是**策略被查询的次数**。

## 动作不是唯一的：MSE 会取平均

杯子摆在桌中间，正前方有一块障碍。示范里 $60\%$ 从左侧绕、$40\%$ 从右侧绕，两条都成功。均方误差的最优解是条件均值：

$$
a_{\text{MSE}}=\arg\min_{a}\;\mathbb{E}_{a^*\sim p(\cdot\mid s)}\bigl\lVert a-a^*\bigr\rVert_2^2=\mathbb{E}\bigl[a^*\mid s\bigr].
$$

把切向分量代进去：$0.6\times(+1)+0.4\times(-1)=+0.2$。模型输出一个幅度只有 $0.2$ 的向左动作，既没绕开左边，也没绕开右边，直接撞上障碍。

```python
import numpy as np

rng = np.random.default_rng(0)

# 同一个状态下，60% 的示范向左绕，40% 向右绕，切向分量正好相反
left = rng.normal(loc=[0.0, +1.0], scale=0.05, size=(600, 2))
right = rng.normal(loc=[0.0, -1.0], scale=0.05, size=(400, 2))
demos = np.concatenate([left, right])

# 障碍占据 |a_y| < 0.5 的通道，落进去就撞
def feasible(a):
    return abs(a[1]) >= 0.5

mse_action = demos.mean(axis=0)
print("MSE 最优动作:", mse_action.round(3), "可行:", feasible(mse_action))
print("示范可行比例:", round(float(np.mean([feasible(a) for a in demos])), 3))

# 生成式策略做的是从分布里采样，而不是把两个峰平均掉
idx = rng.integers(0, len(demos), size=1000)
print("采样可行比例:", round(float(np.mean([feasible(a) for a in demos[idx]])), 3))
```

输出是 `MSE 最优动作: [-0.001 0.199] 可行: False`，而示范与采样的可行比例都是 $1.0$。数据里没有任何一条不可行的动作，模型却输出了一个不可行的动作——这不是拟合不足，是损失函数选错了。

## 动作分块：把决策频率降下来

**动作分块**（action chunking）一次输出未来 $K$ 步：

$$
\hat A_t=\bigl[\hat a_t,\;\hat a_{t+1},\;\ldots,\;\hat a_{t+K-1}\bigr]\in\mathbb{R}^{K\times A}.
$$

ACT（action chunking with transformers, Zhao et al., 2023）用一个 CVAE 加 Transformer 做这件事。收益直接落在上一节那个"可以动的量"上：策略被查询的次数从 $H$ 降到 $H/K$，期望偏离次数从 $\epsilon H=4$ 降到 $\epsilon H/K$，取 $K=50$ 就是 $0.08$。

代价同样明确：chunk 内部是**开环**的。$50\ \mathrm{Hz}$ 控制下 $K=50$ 意味着整整 $1$ 秒不看世界。这一秒里杯子被人碰倒，机器人照旧执行原计划。所以 $K$ 是复合误差和反应延迟之间的取舍，不是越大越好。

重叠 chunk 的**时序集成**（temporal ensembling）把多个预测加权平均以求平滑，但它把上一节刚解决的多模态又平均回去了。只有在动作分布已经单峰时才安全。

## 扩散策略与流匹配：直接建模多峰

Diffusion Policy（Chi et al., 2023）把整段 chunk 当成要去噪的对象。给定观察 $o$，训练一个噪声预测网络：

$$
\mathcal{L}_{\text{DP}}=\mathbb{E}_{k,A_0,\epsilon}\bigl\lVert \epsilon_\theta\bigl(\sqrt{\bar\alpha_k}A_0+\sqrt{1-\bar\alpha_k}\,\epsilon,\;k,\;o\bigr)-\epsilon\bigr\rVert_2^2 .
$$

因为采样保留随机性，同一个观察多次采样可以分别落到左绕和右绕，而不是落在中间。

**流匹配**（flow matching）换一条路：在噪声 $\epsilon$ 和真实动作 $A_0$ 之间画一条直线 $A^{(\tau)}=(1-\tau)\epsilon+\tau A_0$，学习它的速度场：

$$
\mathcal{L}_{\text{FM}}=\mathbb{E}_{\tau,A_0,\epsilon}\bigl\lVert v_\theta\bigl((1-\tau)\epsilon+\tau A_0,\;\tau,\;o\bigr)-\bigl(A_0-\epsilon\bigr)\bigr\rVert_2^2 .
$$

采样就是把 $\dot A=v_\theta(A,\tau,o)$ 从 $\tau=0$ 积到 $\tau=1$。$\pi_0$（Black et al., 2024）用的就是这条路线：目标路径是直的，欧拉积分 $10$ 步往往够用，而扩散常需要几十步。延迟差别很实在——单步前向 $8\ \mathrm{ms}$ 时，$10$ 步是 $80\ \mathrm{ms}$，$50$ 步是 $400\ \mathrm{ms}$。

要写清楚它们解决了什么、没解决什么：生成式策略解决的是**表达**，让多个可行模式都留在输出分布里；它没有解决分布偏移。训练数据还是专家示范，复合误差还在。

## 实时分块：把推理延迟写进采样约束

$50\ \mathrm{Hz}$ 控制的周期是 $20\ \mathrm{ms}$。若一次采样要 $120\ \mathrm{ms}$，那么新 chunk 算完时，机器人已经又走了 $d=\lceil 120/20\rceil=6$ 步。两种朴素处理都不好看：执行完旧 chunk 再等新 chunk，机器人会停顿；算完就硬切，衔接处速度跳变。

**实时分块**（real-time chunking, RTC；Black et al., 2025）把已经承诺执行的前 $d$ 步当作已知，只对剩下的部分采样，相当于在动作序列上做 inpainting：

$$
\hat A_{t+d}\sim p_\theta\bigl(\cdot\mid o_t\bigr)\quad\text{s.t.}\quad \hat a_{t+j}=a^{\text{prev}}_{t+j},\;\; j<d .
$$

约束里藏着一个硬条件：必须有 $K>d$，否则没有可改的自由度。$K=50$、$d=6$ 时还剩 $44$ 步可以重新规划。另一个条件是 $d$ 必须**实测**——相机曝光、USB 传输、图像预处理、动作后处理都要计入。假设一个延迟值，等于把误差从时间轴搬到动作轴上。

## 小结

- [ ] BC 的误差只在专家分布上被测量，复合误差上界 $O(\epsilon H^2)$ 提示唯一可改的量是策略查询次数。
- [ ] MSE 的最优解是条件均值；示范里全是可行动作，平均出来的动作可以不可行。
- [ ] 动作分块把查询次数降到 $H/K$，代价是 chunk 内开环，$K$ 是复合误差与反应延迟的取舍。
- [ ] 扩散与流匹配保住了动作的多峰表达，但不改变分布偏移；RTC 把实测推理延迟变成采样约束，要求 $K>d$。

下一篇把图像、语言和自身状态一起接进策略，看 VLA 如何让同一台机器人听懂不同指令。动手实验见 [7.8 动手：从零实现 VLA 与世界模型检查器](/chapters/07-robot-vla/08-robot-vla)。
