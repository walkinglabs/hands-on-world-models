# 4.2　循环状态空间模型（RSSM）

MiniGrid 里agent钻进一条走廊，前方拐角后面可能是一堵墙，也可能是出口。从当前画面看，两种情况完全一样。模型如果只输出一个确定的状态，就会把「墙」和「出口」平均掉，得到一个两边都不像的结果。

RSSM（循环状态空间模型）的做法是：把状态拆成两部分，一部分记住长期线索，另一部分专门描述「现在还说不准」的东西。

## 确定状态与随机状态

RSSM 把隐状态写成两段拼接：

$$
s_t = (h_t,\, z_t)
$$

- $h_t\in\mathbb{R}^{d_h}$ 是**确定状态**，由 GRU 逐 步更新，负责长期记忆。它像一条稳定的「主干」，把过去的动作和观察串起来。
- $z_t\in\mathbb{R}^{d_z}$ 是**随机状态**，是一个概率分布上的随机变量，负责当前的不确定性。拐角后是墙还是出口，可以用 $z_t$ 取不同值来表达。

为什么要把两者分开？只用确定状态，多种可能被压成一个平均；只用随机状态，长期线索又难稳定传递。分开存放，既保住记忆，又保留「多种可能」。

## 一步状态更新

确定状态由 GRU 更新，输入是上一步的确定状态、随机状态和动作：

$$
h_t = \text{GRU}\bigl(h_{t-1},\, \text{concat}(z_{t-1},\, a_{t-1})\bigr)
$$

随机状态则有两种来源，这就是 RSSM 的关键设计。

## Prior 与 posterior：同一时刻的两个分布

想象未来时还没有真实观察，模型只能根据 $h_t$ 猜 $z_t$。这个分布叫 **prior**：

$$
\bar z_t \sim p(\,\cdot\mid h_t),\qquad p(z_t\mid h_t)=\mathcal{N}(\mu_{\text{pr}},\, \sigma_{\text{pr}}^2)
$$

训练时能看到当前帧的 embedding $e_t=f_\theta(o_t)$，信息更充分。这个分布叫 **posterior**：

$$
z_t \sim q(\,\cdot\mid h_t,\, e_t),\qquad q(z_t\mid h_t, e_t)=\mathcal{N}(\mu_{\text{po}},\, \sigma_{\text{po}}^2)
$$

两者描述同一个 $z_t$，区别只在信息来源：prior 只有过去，posterior 还看到了当前观察。部署 rollout 时观察尚未发生，只能用 prior；训练时有观察，可以用 posterior。

KL 损失让 prior 学会接近 posterior：

$$
\mathcal{L}_{\text{KL}} = \mathrm{KL}\bigl(q(z_t\mid h_t, e_t)\,\|\, p(z_t\mid h_t)\bigr)
$$

这样训练久了，部署时即使没有观察，单凭 prior 采样也能得到接近真相的 $z_t$。

## 四类训练目标

RSSM 的总损失由四项组成，每一项都盯着状态的一个侧面：

$$
\mathcal{L} = \mathcal{L}_{\text{obs}} + \mathcal{L}_{\text{reward}} + \mathcal{L}_{\text{cont}} + \mathcal{L}_{\text{KL}}
$$

1. 观察损失：状态能否重建当前观察，$\hat o_t = d_\psi(h_t, z_t)$。
2. reward 损失：状态能否预测动作结果，$\hat r_t = R_\rho(h_t, z_t)$。
3. continue 损失：状态能否判断任务是否继续，$\hat c_t = C_\kappa(h_t, z_t)$。
4. KL 损失：只凭过去的 prior 能否逼近看过观察的 posterior。

各项数值尺度不同。训练日志要分别画曲线，不能只报告相加后的总 loss——某一项崩了，可能被其他项的增长盖住。

## KL 坍缩与 Free Bits

decoder 太强时，模型可能学会「绕开 $z_t$」直接重建观察。于是 posterior 和 prior 越长越像，随机状态不再携带信息，这叫 **KL 坍缩**。

Free Bits 给一小段 KL 免税额度：每个维度上，KL 低于阈值 $\lambda$ 的部分不计入损失：

$$
\mathcal{L}_{\text{KL}}^{(i)} = \max\bigl(\lambda,\, \mathrm{KL}^{(i)}\bigr)
$$

这样 $z_t$ 有机会先学会携带必要内容，再慢慢收紧约束。离散 RSSM 还会用直通估计、Unimix 等技巧处理具体的数值问题，它们应通过消融检查，而不是当作固定装饰。

## 多步检查：open-loop rollout

训练时每一步都能看到真实观察（posterior），部署 rollout 却要不断用自己的 prior。这两者差距会随步数累积。

4.7 的第一份 Notebook 因此同时画两条曲线：teacher-forced 的一步 loss，以及从某一步开始**只用 prior** 推演的 open-loop 多步 loss。后者更接近部署条件，也更容易暴露「prior 偷懒」的问题。

## 小结

- [ ] RSSM 用确定状态 $h_t$ 保存长期线索，用随机状态 $z_t$ 表达当前不确定性。
- [ ] posterior 在训练时读取当前观察，prior 在想象时只依赖过去和动作，KL 让两者靠近。
- [ ] 四项损失分别约束观察、reward、continue 和 KL，要分开记录。
- [ ] open-loop 多步 rollout 比 teacher-forced 一步结果更接近部署条件。

状态能预测了，下一步是让它替我们选动作。下一篇看 PlaNet 怎样在这个 latent 世界里用 CEM 现场搜索。动手训练见 [4.7 动手：学出一个潜在世界](/chapters/04-decision-and-planning/07-decision-and-planning)。
