# 4.4　Dreamer：在想象中训练

PlaNet 每做一个真实动作都要重跑一遍 CEM。CRAFTER、DMC Cartpole 这类任务要求高频控制，搜索开销可能压垮整个系统。

Dreamer 的思路是：既然世界模型已经能展开未来，干脆把「在想象里反复试、找到好动作」这件事，直接练进 actor。部署时只做一次前向，不再搜索。

## 从真实状态出发，在 prior 里想象

想象不能凭空开始。我们先从 replay buffer 采样真实序列，用 posterior 得到可靠的起始状态 $s_t$。随后冻结世界模型，只用 prior 一路向前展开想象轨迹：

$$
a_{\tau}\sim \pi_\theta(\,\cdot\mid s_{\tau}),\qquad s_{\tau+1}\sim p(\,\cdot\mid h_{\tau+1})
$$

其中 $h_{\tau+1}$ 由 GRU 用 $h_\tau, z_\tau, a_\tau$ 算出。每一步还用预测 head 读出想象中的 reward 和 continue：

$$
r_{\tau} = R_\rho(s_{\tau}),\qquad c_{\tau} = C_\kappa(s_{\tau})
$$

整条想象轨迹只活在模型内部，不接触真实环境。想象 horizon 通常比任务总长度短：过长会积累模型误差，过短则看不见延迟后果。

## λ 回报：在偏差和方差之间折中

评价一条想象轨迹的好坏，需要一个回报目标。直接用真实未来 reward 的累加 $\sum r_\tau$ 方差大；只用一步 bootstrap 又偏差大。$\lambda$ 回报把不同长度的回报估计加权混合：

$$
G^\lambda_{\tau} = r_{\tau} + \gamma c_{\tau}\Bigl((1-\lambda)\, V(s_{\tau+1}) + \lambda\, G^\lambda_{\tau+1}\Bigr)
$$

这里 $\gamma$ 是折扣因子，$c_\tau$ 是 continue 概率（终止时未来被截断），$\lambda\in[0,1]$ 控制混合比例。$\lambda=0$ 退化为一步 TD，$\lambda=1$ 退化为蒙特卡洛回报。$\lambda$ 取中间值时，偏差和方差都不大。

## Critic：拟合想象状态的 value

critic $V_\psi$ 学习预测「从某个想象状态出发，还能拿多少回报」。训练目标是让 $V_\psi(s_\tau)$ 逼近刚才算出的 $G^\lambda_\tau$：

$$
\mathcal{L}_{\text{critic}} = \frac{1}{2}\,\mathbb{E}\!\left[\bigl(V_\psi(s_\tau) - \mathrm{sg}(G^\lambda_\tau)\bigr)^2\right]
$$

$\mathrm{sg}$ 表示 stop-gradient，回报作为固定目标，不回传给世界模型。这一项就是 TD-λ 的均方误差。

## Actor：提高高回报动作的概率

actor $\pi_\theta$ 输出一个动作分布。连续动作常用经过 `tanh` 压缩的正态分布，保证动作落在合法范围 $[-1,1]$：

$$
a = \tanh(\mu_\theta(s)+\sigma_\theta(s)\odot\varepsilon),\qquad \varepsilon\sim\mathcal{N}(0, I)
$$

教学版 actor 用最朴素的 REINFORCE 形式：想象轨迹上每一步都算出 $\log\pi_\theta(a_\tau\mid s_\tau)$，再用 $G^\lambda_\tau$ 加权：

$$
\mathcal{L}_{\text{actor}} = -\,\mathbb{E}\!\left[\log\pi_\theta(a_\tau\mid s_\tau)\, \mathrm{sg}(G^\lambda_\tau)\right]
$$

意思是「回报高的动作，提高它的对数概率」。完整 Dreamer 还会回传 dynamics gradient，让 actor 也利用「这条轨迹为什么好」的信息。

## 一轮训练循环

把世界模型、critic、actor 的更新串起来：

```text
真实环境收集 episode
→ replay buffer 采样序列
→ 训练 encoder、RSSM、预测 heads
→ 从 posterior 状态开始想象 H 步
→ 算 λ 回报，训练 critic
→ 用 λ 回报加权，训练 actor
→ 回到真实环境用新 actor 收集数据
```

三者各有优化器和目标。日志要分开记录世界模型 loss、critic loss、actor loss 和真实 return，避免策略退化被总 loss 遮住。

## DreamerV3 的稳定工具

真实任务里 reward 和 value 跨度可能从 0.01 到 100，直接学很难。DreamerV3 加了几样工具：symlog 压缩大跨度数值；twohot 把标量目标分到相邻 bin，把回归变成分类；unimix 防止离散概率过早变成绝对值；free bits 保护随机状态。

它们解决的是数值尺度、分类饱和和 KL 使用问题，是「让训练跑稳」的工程件。错误的动作对齐或模型漏洞，不能靠它们修好。

## 小结

- [ ] Dreamer 从真实 posterior 状态出发，在 prior 里展开想象轨迹，不接触真实环境。
- [ ] λ 回报在一步 TD 和蒙特卡洛之间折中，critic 拟合它，actor 朝它提高动作概率。
- [ ] 世界模型、actor、critic 各自优化，最终结果由真实 return 和样本效率检查。

Dreamer 用想象训练可微的 actor，适合连续控制。下一篇看 MuZero 怎样换一条路：不重建观察，而是用树搜索改进 policy。动手实验见 [A2：在想象中规划与行动](/chapters/04-decision-and-planning/07-decision-and-planning)。
