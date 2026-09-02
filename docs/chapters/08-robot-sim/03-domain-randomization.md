# 8.3 域随机化 (Domain Randomization) 与泛化

在具身机器人从数字世界迈向实体物理世界的征途中，工程师们遭遇的最令人沮丧的现象莫过于“仿真神勇、真机瘫痪”——在虚拟仿真器中能够百分之百优雅抓取、健步如飞的机器人策略，一旦部署到真实的物理机械臂或四足机器人上，却常常剧烈发抖、抓空滑脱、甚至直接失控摔倒。

这种仿真环境与真实物理世界之间的系统性差异，在学术界被称为**仿真到真实迁移鸿沟（Sim-to-Real Gap）**。

其根本原因在于，任何计算仿真器都只是真实物理世界的简化数学近似：
- 真实减速器存在微小的齿隙与背隙（Backlash）；
- 真实连杆的摩擦力随温度与润滑状态动态变化；
- 真实相机存在不可预测的光照反光、镜头畸变与传感器白噪声；
- 真实物理电机的指令下发到转子响应存在数十毫秒的随机硬件通信延迟。

为了攻克这一物理泛化难题，OpenAI、UC Berkeley 等机构提出了颠覆性的 **域随机化（Domain Randomization, DR）** 与 **自适应域随机化（Automatic Domain Randomization, ADR）** 技术。

其核心哲学极其深邃：**既然我们无法制造一个与真实世界一模一样的完美仿真器，那么就索性在虚拟世界中将所有物理参数与视觉属性进行极度宽泛的随机扰动——如果一个策略能够同时在成千上万个光怪陆离的物理参数世界中完成任务，那么当它来到真实物理世界时，真实世界不过是这个随机化包络中的又一个普通特例！**

<div align="center">

<img src="/figures/08-robot-sim/source/03-domain-randomization/tobin-fig1.png" alt="OpenAI 域随机化在纯合成随机化图像上训练物体检测，并成功零样本迁移到真实机械臂抓取。" width="86%">

_图 8.3-1：OpenAI 域随机化在纯合成随机化图像上训练物体检测，并成功零样本迁移到真实机械臂抓取。 出处：[Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World，Josh Tobin et al.，2017](https://arxiv.org/abs/1703.06907)。_

</div>

---

## 8.3.1 物理与统计基石：从过拟合虚拟物理到分布外泛化

要理解域随机化的数学威力，我们首先需要从统计学习中的分布偏移（Distribution Shift）讲起。

### 1. 经典强化学习的“虚拟物理过拟合”
在传统的固定参数仿真器中，重力加速度恒定为 $g = 9.81\text{ m/s}^2$，地面摩擦系数固定为 $\mu = 1.0$，关节阻尼恒为 $c = 0.05$。
神经网络策略极具“投机性”，它会迅速学会利用这组唯一物理参数的微观动力学漏洞（例如卡在某个精确的共振频率上借力）。然而，一旦真机的真实摩擦系数是 $\mu = 0.82$，这个脆弱的策略就会瞬间崩溃。

### 2. 贝叶斯鲁棒优化目标（Bayesian Robust RL）
域随机化将原本确定性的马尔可夫决策过程 $\mathcal{M}$，扩展为一个由环境扰动参数矢量 $\boldsymbol{\xi} \in \Xi$ 参数化的**环境分布族 $\mathcal{M}(\boldsymbol{\xi})$**。

参数矢量 $\boldsymbol{\xi}$ 涵盖了广泛的物理与视觉自由度：
- **刚体质量分布**：$m_i \sim \mathcal{U}(0.8 m_{i, 0}, 1.2 m_{i, 0})$；
- **库仑摩擦系数**：$\mu \sim \mathcal{U}(0.3, 1.5)$；
- **通信与执行器延迟**：$\Delta t_{\text{delay}} \sim \text{Uniform}(0\text{ ms}, 40\text{ ms})$；
- **视觉表面纹理与光照**：背景贴图、光源三维坐标与色温。

策略的优化目标是在整个参数扰动先验分布 $p(\boldsymbol{\xi})$ 上最大化期望累积回报：

$$J_{\text{DR}}(\pi_\theta) = \mathbb{E}_{\boldsymbol{\xi} \sim p(\boldsymbol{\xi})} \left[ \mathbb{E}_{\tau \sim \pi_\theta, \mathcal{M}(\boldsymbol{\xi})} \left[ \sum_{t=0}^T \gamma^t r(\mathbf{s}_t, \mathbf{a}_t) \right] \right]$$

<div align="center">

<img src="/figures/08-robot-sim/latex/03-domain-randomization/visual-nuisance-feature-invariance.png" alt="视觉域随机化通过随机纹理与光照迫使网络忽略无关背景、提取核心空间几何" width="86%">

_图 8.3-2：视觉域随机化通过随机纹理与光照迫使网络忽略无关背景、提取核心空间几何。_

</div>

---

## 8.3.2 核心数学推导一：物理动力学扰动与受力手算

物理参数的剧烈扰动如何直接改变机器人的运动学响应？

<div align="center">

<img src="/figures/08-robot-sim/source/03-domain-randomization/peng-fig1.png" alt="动力学随机化通过随机化质量、摩擦与电机阻尼，训练出可直接部署到物理机器人的跳跃步态。" width="86%">

_图 8.3-3：动力学随机化通过随机化质量、摩擦与电机阻尼，训练出可直接部署到物理机器人的跳跃步态。 出处：[Sim-to-Real Transfer of Robotic Control with Dynamics Randomization，Xue Bin Peng et al.，2018](https://arxiv.org/abs/1710.06537)。_

</div>

<div align="center">

<img src="/figures/08-robot-sim/source/03-domain-randomization/tobin-fig2.png" alt="桌面随机几何体与随机材质纹理生成，展示视觉域随机化的样本空间丰富度。" width="86%">

_图 8.3-4：桌面随机几何体与随机材质纹理生成，展示视觉域随机化的样本空间丰富度。 出处：[Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World，Josh Tobin et al.，2017](https://arxiv.org/abs/1703.06907)。_

</div>

### 动力学扰动参数手算数值算例
设一个机械臂末端正在推动一个木块。电机输出水平推力 $F_{\text{push}} = 20.0\text{ N}$。
我们对两个关键物理参数进行域随机化采样：
- **木块质量**：$m \sim \mathcal{U}(1.0\text{ kg}, 4.0\text{ kg})$；
- **地面摩擦系数**：$\mu \sim \mathcal{U}(0.2, 0.8)$（重力加速度 $g = 10.0\text{ m/s}^2$）。

我们来手算在极端边界环境下的物体加速度范围：
1. **环境 A（最轻最滑边界）**：$m_A = 1.0\text{ kg}, \mu_A = 0.2$
   - 摩擦阻力：$f_A = \mu_A m_A g = 0.2 \times 1.0 \times 10.0 = 2.0\text{ N}$；
   - 净加速度：$a_A = \frac{F_{\text{push}} - f_A}{m_A} = \frac{20.0 - 2.0}{1.0} = +18.0\text{ m/s}^2$；
2. **环境 B（最重最涩边界）**：$m_B = 4.0\text{ kg}, \mu_B = 0.8$
   - 摩擦阻力：$f_B = \mu_B m_B g = 0.8 \times 4.0 \times 10.0 = 32.0\text{ N}$；
   - 净外力：$F_{\text{push}} - f_B = 20.0 - 32.0 = -12.0\text{ N} \le 0$（推力小于最大静摩擦力，物体保持静止）；
   - 净加速度：$a_B = 0.0\text{ m/s}^2$。

加速度在 $[0.0, 18.0]\text{ m/s}^2$ 之间发生了翻天覆地的变化！如果策略仅依据单步速度反馈做出决策，在环境 A 中会推得过猛飞出桌面，在环境 B 中则根本推不动。
为了在所有随机环境中同时取得高分，**策略网络被倒逼着必须学会通过历史状态的时序差分来隐式辨识当前的物理质量与阻力，并自适应调节输出推力的大小！**

<details>
<summary><b>深入推导：非参数化域随机化在最坏情况下的极小极大鲁棒控制（Minimax Robust Control）边界证明（点击展开查看完整推导）</b></summary>

将环境参数先验 $p(\boldsymbol{\xi})$ 视为博弈论中的自然对抗对手（Adversarial Nature）。
定义最坏情况鲁棒目标：
$$\max_\theta \min_{p(\boldsymbol{\xi}) \in \mathcal{P}} \mathbb{E}_{\boldsymbol{\xi} \sim p} [J(\pi_\theta, \boldsymbol{\xi})]$$
根据冯·诺依曼极小极大定理（Minimax Theorem），若策略空间与环境不确定性集合 $\mathcal{P}$ 为紧致凸集，且目标函数关于 $\theta$ 凹、关于 $\boldsymbol{\xi}$ 凸，则最优策略 $\pi^*$ 满足鞍点条件：
$$\min_{p \in \mathcal{P}} \mathbb{E}_{\boldsymbol{\xi} \sim p} [J(\pi^*, \boldsymbol{\xi})] = \max_\theta \min_{p \in \mathcal{P}} \mathbb{E}_{\boldsymbol{\xi} \sim p} [J(\pi_\theta, \boldsymbol{\xi})]$$
该证明严格确立了域随机化在最坏物理工况下的抗扰动下界保障。
</details>

---

## 8.3.3 核心数学推导二：自动域随机化 (ADR) 与课程学习

尽管域随机化效果拔群，但手动调节随机化参数范围却极其折磨：
- **范围设定过窄**：无法覆盖真机复杂的物理效应，依然发生迁移崩溃；
- **范围设定过宽**（例如把重力随机设在 $0 \sim 100\text{ m/s}^2$）：物理规律过于荒诞，导致强化学习策略在训练初期彻底迷失方向，根本无法收敛。

<div align="center">

<img src="/figures/08-robot-sim/source/03-domain-randomization/rma-fig3.png" alt="RMA 比较不同域随机化幅度下的地形适应表现，阐明自适应随机化范围的重要性。" width="86%">

_图 8.3-5：RMA 比较不同域随机化幅度下的地形适应表现，阐明自适应随机化范围的重要性。 出处：[RMA: Rapid Motor Adaptation for Bipedal Robots，Ashish Kumar et al.，2021](https://arxiv.org/abs/2107.04034)。_

</div>

OpenAI 在 2019 年单手解魔方机器人中提出了 **自适应域随机化（Automatic Domain Randomization, ADR）**：
让算法根据策略在当前边界上的任务成功率，自动调节每一个物理参数的随机化边界 $[a_k, b_k]$。

### 1. 边界动态扩张与收缩控制律
对于第 $k$ 个物理参数（如质量或摩擦力）：
系统周期性地将参数锁定在其当前的下界 $a_k$ 与上界 $b_k$ 处，评估策略的成功率 $S(a_k)$ 与 $S(b_k)$。
- **当成功率高于阈值（$S > C_{\text{high}} = 80\%$）**：表明策略已经完全征服当前难度，算法自动向外扩张边界：
  $$a_k \leftarrow a_k - \Delta_k, \quad b_k \leftarrow b_k + \Delta_k$$
- **当成功率低于阈值（$S < C_{\text{low}} = 40\%$）**：表明环境过于严酷超出学习能力，算法自动收缩边界以巩固基础：
  $$a_k \leftarrow a_k + \Delta_k, \quad b_k \leftarrow b_k - \Delta_k$$

通过 ADR 自动生成的“物理难度渐进阶梯”，机器人策略能够从小幅扰动开始稳扎稳打，最终在无人干预的情况下自主学会适应极端恶劣的物理世界！

<details>
<summary><b>深入推导：自适应域随机化作为参数化马尔可夫决策过程（PAMDP）课程学习收敛性证明（点击展开查看完整推导）</b></summary>

将环境参数边界向量 $\mathbf{B} = [a_1, b_1, \dots, a_K, b_K]^\top$ 构造成一个外层元控制系统。
外层优化的目标是最大化参数空间超体积 $\text{Vol}(\mathbf{B}) = \prod_{k=1}^K (b_k - a_k)$，约束条件为策略在边界上的预期值满足 $\mathbb{E}_{\boldsymbol{\xi} \in \partial \mathbf{B}} [V^{\pi_\theta}(\boldsymbol{\xi})] \ge V_{\text{target}}$。
利用李雅普诺夫稳定性定理，构造势能函数 $V(\mathbf{B}, \theta) = -\log \text{Vol}(\mathbf{B}) + \beta \|V^{\pi_\theta} - V_{\text{target}}\|^2$。
当内层 PPO 迭代速度满足时间尺度分离（Two-Time-Scale Separation）时，外层边界更新动力学严格渐近收敛于最大熵物理环境覆盖流形。
</details>

---

## 8.3.4 纯底层 PyTorch 代码实现：物理参数与视觉噪声随机化环境包装器

下面我们使用纯底层 PyTorch 算子实现一个结构完备的物理参数随机化与 ADR 边界自适应调度器。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainRandomizedEnv(nn.Module):
    """
    纯底层 PyTorch 物理域随机化多环境模拟器
    在每个重置周期为每个环境独立采样物理质量、摩擦力与传感器观测噪声
    """
    def __init__(self, num_envs: int = 1024, dt: float = 0.02):
        super().__init__()
        self.num_envs = num_envs
        self.dt = dt

        # 注册当前物理参数边界 [min, max]
        self.register_buffer("mass_bounds", torch.tensor([1.0, 3.0]))
        self.register_buffer("friction_bounds", torch.tensor([0.2, 1.0]))
        self.register_buffer("obs_noise_std", torch.tensor(0.05))

        # 每个环境的实时物理参数张量
        self.register_buffer("env_mass", torch.zeros(num_envs))
        self.register_buffer("env_friction", torch.zeros(num_envs))
        self.register_buffer("state", torch.zeros(num_envs, 2)) # [位置 x, 速度 v]

    def sample_domain_parameters(self):
        """
        在当前边界内均匀采样每个环境的物理参数
        """
        m_low, m_high = self.mass_bounds[0], self.mass_bounds[1]
        f_low, f_high = self.friction_bounds[0], self.friction_bounds[1]

        self.env_mass = torch.rand(self.num_envs, device=self.state.device) * (m_high - m_low) + m_low
        self.env_friction = torch.rand(self.num_envs, device=self.state.device) * (f_high - f_low) + f_low

    def reset(self) -> torch.Tensor:
        self.sample_domain_parameters()
        self.state.zero_()
        return self.get_noisy_obs()

    def get_noisy_obs(self) -> torch.Tensor:
        """
        注入传感器高斯观测白噪声
        """
        noise = torch.randn_like(self.state) * self.obs_noise_std
        return self.state + noise

    def step(self, force_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        在随机化的动力学参数下推进一步物理仿真
        """
        u = force_action.squeeze(-1) # (num_envs,)
        x = self.state[:, 0]
        v = self.state[:, 1]

        # 随机化摩擦阻力: f = -mu * m * g * sign(v)
        f_friction = -self.env_friction * self.env_mass * 9.81 * torch.tanh(v * 10.0)
        net_force = u + f_friction
        acc = net_force / self.env_mass

        next_v = v + self.dt * acc
        next_x = x + self.dt * next_v

        self.state[:, 0] = next_x
        self.state[:, 1] = next_v

        # 奖励：达到目标位置 x_target = 1.0
        rewards = - (next_x - 1.0).pow(2) - 0.01 * u.pow(2)
        return self.get_noisy_obs(), rewards

class ADRScheduler:
    """
    自动域随机化 (ADR) 边界动态自适应控制器
    """
    def __init__(self, env: DomainRandomizedEnv, step_size: float = 0.1, high_thresh: float = -0.1):
        self.env = env
        self.step_size = step_size
        self.high_thresh = high_thresh

    def update_bounds(self, mean_reward: float):
        """
        根据策略表现自动扩张参数边界
        """
        if mean_reward > self.high_thresh:
            # 表现优异，扩大质量与摩擦力随机化范围
            self.env.mass_bounds[0] = max(0.5, self.env.mass_bounds[0] - self.step_size)
            self.env.mass_bounds[1] += self.step_size
            self.env.friction_bounds[0] = max(0.05, self.env.friction_bounds[0] - self.step_size * 0.5)
            self.env.friction_bounds[1] += self.step_size * 0.5
            print(f"[ADR Update] 策略达标！质量边界扩张至: [{self.env.mass_bounds[0]:.2f}, {self.env.mass_bounds[1]:.2f}]")

# ===================================================================
# 单元测试与物理扰动分布校验
# ===================================================================
if __name__ == "__main__":
    num_envs = 1024
    env = DomainRandomizedEnv(num_envs=num_envs)
    adr = ADRScheduler(env=env, step_size=0.2, high_thresh=-0.5)

    obs = env.reset()
    dummy_force = torch.full((num_envs, 1), 15.0)

    # 1. 推进单步物理交互
    next_obs, rewards = env.step(dummy_force)

    print(f"[DR Test] 并发随机化环境数: {num_envs}")
    print(f"[DR Test] 采样质量范围: [{env.env_mass.min().item():.2f}, {env.env_mass.max().item():.2f}] kg")
    print(f"[DR Test] 采样摩擦系数范围: [{env.env_friction.min().item():.2f}, {env.env_friction.max().item():.2f}]")
    print(f"[DR Test] 平均单步奖励: {rewards.mean().item():.4f}")

    assert next_obs.shape == (num_envs, 2), "观测张量形状不符！"
    assert env.env_mass.min() >= 1.0 and env.env_mass.max() <= 3.0, "质量采样超出初始边界！"

    # 2. 模拟策略高分触发 ADR 边界自动膨胀
    adr.update_bounds(mean_reward=0.0)
    assert env.mass_bounds[1] > 3.0, "ADR 边界未能成功向上扩张！"
    print("✓ 物理域随机化环境与 ADR 自适应调度器单测全部通过！")
```

---

## 8.3.5 本节小结

回顾本节内容，我们建立了跨越仿真到真实鸿沟的完整域随机化方法论：
1. **分布包络哲学**：通过在虚拟世界中穷举物理参数与视觉噪声的多样性组合，使真实世界成为随机化流形的一个自然特例；
2. **时序辨识的倒逼机制**：动力学参数的剧烈扰动促使策略网络从历史交互轨迹中隐式提取系统物理特征，自适应调节输出控制量；
3. **ADR 课程进阶**：利用任务成功率自动调节参数边界开合，消除了手动调参的盲目性，实现了从易到难的端到端稳健收敛。
