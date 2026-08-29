# 8.4　特权蒸馏与现实迁移（Sim2Real）

同一份策略，在训练用的模拟器里跑出 $0.94$ 的成功率，换一个物理后端评测掉到 $0.71$，搬到真机只剩 $0.38$。现实差距（Reality Gap）是具身智能逃不开的坎。

## 现实差距从哪里来

**现实差距**（reality gap）要拆成可分别诊断的四项，否则只会得到一句"迁移不好"：

| 来源   | 典型表现                 | 最小诊断实验                                 |
| ------ | ------------------------ | -------------------------------------------- |
| 视觉   | 换光照或背景，成功率骤降 | 用真机图像做离线动作预测，比较误差与仿真图像 |
| 动力学 | 轨迹缓慢偏离，误差累积   | 开环回放同一动作序列，比较关节轨迹           |
| 延迟   | 高频振荡、过冲、极限环   | 发阶跃指令，实测从下发到编码器响应的时间     |
| 接触   | 抓取滑落、插接卡死       | 同一动作换求解器或时间步，比较成功率差异     |

四项要分开测。若开环回放的关节轨迹在 $2$ 秒内就偏出 $5^\circ$，问题在动力学，换多少视觉增强都没用。

## 域随机化与系统辨识

**域随机化**（domain randomization, DR）把仿真参数当随机变量，优化其上的期望回报：

$$
\max_\theta\;\mathbb{E}_{\xi\sim p(\xi)}\bigl[J(\pi_\theta;\;\xi)\bigr],\qquad
\xi=(\mu,\;m,\;k_p,\;\tau_{\text{delay}},\ldots).
$$

例如摩擦 $\mu\sim U(0.4,1.2)$、连杆质量 $\pm 20\%$、电机增益 $\pm 25\%$。代价是明确的：区间越宽，策略越保守。典型形状是仿真内成功率随区间变宽单调下降，真机成功率先升后降——因为过宽的区间里包含大量真机永远不会出现的物理，策略为它们付出了保守性。

**系统辨识**（system identification）从另一头收紧：用真机的开环回放数据拟合 $\xi$，把随机化区间围在实测值附近。

$$
\hat\xi=\arg\min_{\xi}\;\sum_{t}\bigl\lVert q_t^{\text{real}}-q_t^{\text{sim}}(\xi)\bigr\rVert_2^2 .
$$

两者是互补的：辨识给中心，随机化给宽度。只做随机化会得到一个平庸而稳健的策略；只做辨识会得到一个对单台机器人过拟合的策略，换一台就失效。

## 延迟建模与动作重复

延迟要逐段实测再相加。一个典型分解是传感 $5\ \mathrm{ms}$、推理 $20\ \mathrm{ms}$、通信 $3\ \mathrm{ms}$、执行器响应 $10\ \mathrm{ms}$，合计 $38\ \mathrm{ms}$。若控制频率是 $200\ \mathrm{Hz}$（周期 $5\ \mathrm{ms}$），机器人执行的是 $7.6$ 步以前算出的动作。仿真里不建模这一项，训出的策略会依赖一个真机不存在的即时反馈。

```python
from collections import deque

class DelayedActionRepeat:
    """在仿真环境外面套一层：动作延迟 d 个控制周期，并重复执行 r 次。"""

    def __init__(self, env, delay_steps: int, action_repeat: int = 1):
        self.env, self.delay, self.repeat = env, delay_steps, action_repeat

    def reset(self):
        obs = self.env.reset()
        self.buffer = deque([self.env.zero_action()] * self.delay,
                            maxlen=self.delay + 1)
        return obs

    def step(self, action):
        self.buffer.append(action)
        applied = self.buffer[0]          # 真正被执行的是 d 步以前的动作
        total_reward, done = 0.0, False
        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step(applied)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

# 38 ms 延迟、200 Hz 控制：delay = ceil(38 / 5) = 8
env = DelayedActionRepeat(base_env, delay_steps=8, action_repeat=1)
```

`action_repeat` 是另一件事：它降低有效决策频率，让策略更平滑，也和 [7.2](/chapters/07-robot-policy/04-behavior-cloning) 的动作分块同源。两者都用开环片段换稳定性。延迟量本身也应当随机化，因为真机的延迟会抖动。

## 人在回路的样本效率

纯仿真 RL 便宜但不诚实；纯真机 RL 诚实但太贵。**HIL-SERL** 一类方法让人在真机上介入：策略在跑，人随时接管并留下那一段示范，再用这些数据做 off-policy 更新。它改的是**数据从哪来**，不是世界模型的接口。和 [7.3](/chapters/07-robot-policy/07-vla-rtx) 数据引擎的差别是：人给的是真机标签，生成模型给的是合成标签。两者都可以进同一 replay。课程不要求复现 HIL-SERL；若真机选题用了介入，必须在数据卡里写清介入比例和谁在何时接管。

成功判据也不一定由人给。2025 年以来的做法是让视觉语言模型当奖励模型：把 rollout 视频喂给 VLM，问"任务成了吗"，用回答当奖励信号。它省掉了人工标注，但把奖励对齐问题换成了"VLM 的判断可不可信"——所以用 VLM 奖励时，必须抽一部分由人复核，报告一致率。

## 真机评测必须报告什么

$20$ 次试验里成功 $14$ 次，$\hat p=0.70$。正态近似的 $95\%$ 区间是

$$
0.70\pm 1.96\sqrt{\frac{0.70\times 0.30}{20}}=0.70\pm 0.20 .
$$

也就是 $[0.50,\,0.90]$。在 $20$ 次试验下，$0.70$ 和 $0.55$ 根本不可区分。要把半宽压到 $0.05$，需要 $n\approx 323$ 次——真机上这通常意味着几天。这不是可以绕过的统计学，只能诚实地报告 $n$ 和区间。

一份可信的真机报告至少包含：试验次数与置信区间；初始条件的分布（物体位置、光照、干扰的采样方式）；**失败分类**（未抓住、抓住后滑落、碰撞、超时、卡死各占多少）；每类失败的视频；以及权重与随机种子。只报一个成功率均值的结果不构成证据。

## 小结

- [ ] 现实差距要拆成视觉、动力学、延迟、接触四项分别诊断；开环回放是最便宜的第一道检查。
- [ ] 系统辨识给随机化区间定中心，域随机化给宽度；过宽的区间会用保守性换稳健性。
- [ ] 真机结论必须带 $n$、置信区间、失败分类和视频，$20$ 次试验的半宽约 $\pm 0.20$。
- [ ] 人在回路（HIL-SERL）改的是真机数据从哪来；介入比例必须写进数据卡。

下一篇进入本章第一个动手实验：把一份示范数据写成标准格式，训练扩散策略并对比。

[上一篇 7.6　物理引擎与仿真构建](/chapters/08-robot-sim/01-physics-mujoco) · [下一篇 → 7.8　动手：扩散策略的从零开始实现](/chapters/07-robot-policy/10-diffusion-policy-scratch)
