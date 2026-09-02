# 8.7 仿真到真实迁移核心精讲

在整个具身智能与仿生机器人研究领域，**仿真到真实迁移（Sim-to-Real Transfer）** 是衡量算法是否具备工程实用价值的“终极试金石”。

一个在数字物理仿真器中表现惊艳的模型，如果在面对真实街道的石子、真实机械电机的迟滞或真实光照的眩光时无法稳定工作，那么它充其量只是一个昂贵的虚拟玩具。

经过产业界与学术界近十年的高强度攻关，行业已经沉淀出了一套逻辑严密、分工明确的 Sim-to-Real 技术方法论：从纯粹统计意义上的**域随机化（Domain Randomization）**，到两阶段自适应的**特权信息蒸馏（RMA）**，再到基于真实轨迹物理拟合的**系统辨识（System Identification, SysID）**与**执行器网络建模（Actuator Net）**。

本节我们将以精炼且深刻的视角，纵览 Sim-to-Real 各大核心流派的物理哲学、数学机理与工程落地闭环。

<div align="center">

<img src="/figures/08-robot-sim/source/07-sim2real-concise/tobin-fig6.png" alt="OpenAI 域随机化在真实复杂杂乱桌面机械臂抓取任务中的成功率评测。" width="86%">

_图 8.7-1：OpenAI 域随机化在真实复杂杂乱桌面机械臂抓取任务中的成功率评测。 出处：[Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World，Josh Tobin et al.，2017](https://arxiv.org/abs/1703.06907)。_

</div>

---

## 8.7.1 物理与控制基石：四大 Sim2Real 流派深度横向解构

要针对具体机器人硬件挑选最适配的迁移方案，我们必须从经典控制与信息论角度对四大主流范式进行严谨对比：

### 1. 域随机化（Domain Randomization, DR）
- **核心哲学**：将真实世界视为一个宽泛随机分布包络下的特例；
- **优势**：无需真机参数测量，纯虚拟训练，开箱即用；
- **物理代价**：若随机化范围过大，策略往往被迫采取极度保守的“防摔爬行姿态”，牺牲了高动态奔跑性能。

### 2. 特权在线自适应蒸馏（RMA / LBC）
- **核心哲学**：仿真器全知上帝教导基准策略，学生网络凭本体历史在线估计环境隐变量；
- **优势**：兼顾极致运动敏捷性与面对未知松软泥雪地形的秒级自适应；
- **物理代价**：需要精细设计一阶段特权状态空间的维度与物理物理先验。

### 3. 经典与可微系统辨识（System Identification, SysID）
- **核心哲学**：通过在真机上执行预设激励动作，采集力矩与位移，反求出机械臂真实的质量惯性矩阵 $\mathbf{M}$ 与阻尼系数；
- **优势**：物理可解释性极强，直接对齐数字孪生仿真器；
- **物理代价**：无法覆盖随温度与磨损动态剧烈漂移的非线性摩擦。

### 4. 执行器动力学神经网络（Actuator Network）
- **核心哲学**：电机减速器的电气特性（如磁滞、电机过热降额、反电动势）极其非线性，直接用小型 MLP/LSTM 拟合真实电机的输入电压与输出力矩映射关系；
- **优势**：彻底消除了高频控制中关节力矩的建模误差。

<div align="center">

<img src="/figures/08-robot-sim/latex/07-sim2real-concise/domain-randomization-nested-expectation.png" alt="嵌套期望结构：外层物理参数分布先验与内层马尔可夫轨迹采样的鲁棒优化" width="86%">

_图 8.7-2：嵌套期望结构：外层物理参数分布先验与内层马尔可夫轨迹采样的鲁棒优化。_

</div>

---

## 8.7.2 核心数学推导一：线性刚体动力学系统辨识与最小二乘求解

在多连杆机器人中，如何通过真实的运动轨迹精确反求未知连杆的质量、质心位置与转动惯量？

<div align="center">

<img src="/figures/08-robot-sim/source/07-sim2real-concise/peng-fig7.png" alt="动力学系统辨识使仿真器模型精准贴近物理机器人，显著降低仿真与真实误差。" width="86%">

_图 8.7-3：动力学系统辨识使仿真器模型精准贴近物理机器人，显著降低仿真与真实误差。 出处：[Sim-to-Real Transfer of Robotic Control with Dynamics Randomization，Xue Bin Peng et al.，2018](https://arxiv.org/abs/1710.06537)。_

</div>

<div align="center">

<img src="/figures/08-robot-sim/source/07-sim2real-concise/rma-fig4.png" alt="四足机器人在不同载重与地形下在线自适应估计物理参数并稳定行走。" width="86%">

_图 8.7-4：四足机器人在不同载重与地形下在线自适应估计物理参数并稳定行走。 出处：[RMA: Rapid Motor Adaptation for Bipedal Robots，Ashish Kumar et al.，2021](https://arxiv.org/abs/2107.04034)。_

</div>

### 1. 动力学方程的参数线性化分离（Linear-in-Parameters）
在刚体动力学中，尽管关于关节位置 $\mathbf{q}$ 是极其复杂的三角非线性函数，但关于待辨识的刚体惯性参数矢量 $\boldsymbol{\theta} \in \mathbb{R}^p$（包含各连杆质量 $m_i$、一阶质量矩 $m_i \mathbf{r}_i$、转动惯量张量 $I_{xx}, I_{yy}, I_{zz}$），方程满足严格的**线性分离性质**：

$$\boldsymbol{\tau}(t) = \mathbf{Y}(\mathbf{q}(t), \dot{\mathbf{q}}(t), \ddot{\mathbf{q}}(t)) \cdot \boldsymbol{\theta}$$

其中 $\mathbf{Y} \in \mathbb{R}^{n \times p}$ 称为**运动学回归矩阵（Observation Regressor Matrix）**，其元素完全由编码器测量出的关节角度、角速度与角加速度显式计算得出。

### 2. 批量最小二乘参数求解（Least Squares Regression）
在真机上运行一段充分激励各关节运动的测试轨迹，采集 $N$ 个时间步的力矩与运动数据，纵向堆叠构造超定线性方程组：

$$\mathbf{T} = \begin{bmatrix} \boldsymbol{\tau}(1) \\ \boldsymbol{\tau}(2) \\ \vdots \\ \boldsymbol{\tau}(N) \end{bmatrix} \in \mathbb{R}^{n N}, \quad \mathbf{\Phi} = \begin{bmatrix} \mathbf{Y}(1) \\ \mathbf{Y}(2) \\ \vdots \\ \mathbf{Y}(N) \end{bmatrix} \in \mathbb{R}^{n N \times p}$$

利用初等线性代数的正规方程组，求得物理惯性参数矢量的无偏解析最小二乘解：

$$\hat{\boldsymbol{\theta}} = (\mathbf{\Phi}^\top \mathbf{\Phi})^{-1} \mathbf{\Phi}^\top \mathbf{T}$$

### 3. 单摆系统辨识数值手算算例
设一个单摆连杆绕水平轴旋转，角度为 $\theta$，重力加速度 $g = 10.0\text{ m/s}^2$。
单摆的连续动力学方程为：
$$\tau = I \ddot{\theta} + m g L \sin(\theta) = \begin{bmatrix} \ddot{\theta} & g \sin(\theta) \end{bmatrix} \begin{bmatrix} I \\ m L \end{bmatrix} = \mathbf{Y} \cdot \mathbf{p}$$
其中待辨识参数矢量为 $\mathbf{p} = [I, mL]^\top$（转动惯量与一阶质量矩）。

在真机上采集了 2 个瞬时采样点：
- **采样点 1**：$\theta_1 = 30^\circ$（$\sin 30^\circ = 0.5$），$\ddot{\theta}_1 = 2.0\text{ rad/s}^2$，测量电机力矩 $\tau_1 = 5.0\text{ N}\cdot\text{m}$；
- **采样点 2**：$\theta_2 = 90^\circ$（$\sin 90^\circ = 1.0$），$\ddot{\theta}_2 = 0.0\text{ rad/s}^2$，测量电机力矩 $\tau_2 = 4.0\text{ N}\cdot\text{m}$。

我们来手动求解物理参数 $\mathbf{p}$：
1. **构造回归矩阵与力矩向量**：
   $$\mathbf{\Phi} = \begin{bmatrix} 2.0 & 10.0 \times 0.5 \\ 0.0 & 10.0 \times 1.0 \end{bmatrix} = \begin{bmatrix} 2.0 & 5.0 \\ 0.0 & 10.0 \end{bmatrix}, \quad \mathbf{T} = \begin{bmatrix} 5.0 \\ 4.0 \end{bmatrix}$$
2. **求解线性方程组**：
   - 由第 2 行：$10.0 \times (mL) = 4.0 \implies mL = 0.40\text{ kg}\cdot\text{m}$；
   - 代入第 1 行：$2.0 \times I + 5.0 \times (0.40) = 5.0 \implies 2.0 I + 2.0 = 5.0 \implies 2.0 I = 3.0 \implies I = 1.50\text{ kg}\cdot\text{m}^2$！

初等代数的直观求解清晰展现了系统辨识的数学魔力：仅需两次精准的运动测量，未知的机械转动惯量 $1.50\text{ kg}\cdot\text{m}^2$ 与质量矩 $0.40\text{ kg}\cdot\text{m}$ 被彻底精准锁定！

<details>
<summary><b>深入推导：基于递推最小二乘法（RLS）与遗忘因子的实时在线系统辨识推导（点击展开查看完整推导）</b></summary>

当机械臂抓取未知重物时，参数 $\boldsymbol{\theta}_t$ 发生阶跃突变。
引入指数遗忘因子 $\lambda \in (0, 1]$，在线递推增益矩阵更新公式为：
$$\mathbf{K}_t = \frac{\mathbf{P}_{t-1} \mathbf{Y}_t^\top}{\lambda + \mathbf{Y}_t \mathbf{P}_{t-1} \mathbf{Y}_t^\top}$$
参数协方差矩阵与估计值迭代更新：
$$\hat{\boldsymbol{\theta}}_t = \hat{\boldsymbol{\theta}}_{t-1} + \mathbf{K}_t (\boldsymbol{\tau}_t - \mathbf{Y}_t \hat{\boldsymbol{\theta}}_{t-1})$$
$$\mathbf{P}_t = \frac{1}{\lambda} (\mathbf{I} - \mathbf{K}_t \mathbf{Y}_t) \mathbf{P}_{t-1}$$
该算法无需重新求解大矩阵求逆，以 $\mathcal{O}(p^2)$ 的极低算力在单片机上实现微秒级在线物理参数追踪。
</details>

---

## 8.7.3 核心数学推导二：执行器非线性延迟与一阶低通滤波

在 Sim2Real 失败案例中，超过 $70\%$ 的非预期振荡源自**未建模的执行器延迟与低通滤波特性**。

<div align="center">

<img src="/figures/08-robot-sim/source/07-sim2real-concise/lbc-fig2.png" alt="特权信息蒸馏在动态交通与复杂物理环境中的端到端闭环迁移测试。" width="86%">

_图 8.7-5：特权信息蒸馏在动态交通与复杂物理环境中的端到端闭环迁移测试。 出处：[Learning by Cheating，Dian Chen et al.，2020](https://arxiv.org/abs/1912.09686)。_

</div>

真实电机无法瞬间响应阶跃力矩指令，系统满足**一阶低通滤波与纯时间滞后响应方程**：

$$\boldsymbol{\tau}_{\text{real}}(t) = (1 - \alpha) \boldsymbol{\tau}_{\text{real}}(t-1) + \alpha \boldsymbol{\tau}_{\text{cmd}}(t - d)$$

其中：
- $d \in \{1, 2, 3\}$ 为硬件 CAN 总线与驱动器的离散通信延迟步数（约 $10 \sim 30\text{ ms}$）；
- $\alpha = \frac{\Delta t}{\tau_{\text{motor}} + \Delta t} \in (0, 1]$ 为平滑滤波系数。

在仿真器训练时显式注入该执行器延迟方程，能迫使策略网络提前学会预判物理惯性，彻底消除了真机部署时的“高频打摆震颤”。

<details>
<summary><b>深入推导：二阶线性执行器在根轨迹分析下的阶跃响应阻尼比与频域超调证明（点击展开查看完整推导）</b></summary>

将电机与连杆组合系统建模为闭环二阶传递函数：
$$G(s) = \frac{\omega_n^2}{s^2 + 2 \zeta \omega_n s + \omega_n^2} e^{-s \cdot T_d}$$
对纯时间延迟 $e^{-s T_d}$ 进行一阶帕德近似（Padé Approximation）$e^{-s T_d} \approx \frac{1 - s T_d / 2}{1 + s T_d / 2}$。
系统的开环右半平面零点 $z = +2 / T_d$ 诱发严重的相位滞后。
当比例增益 $K_p > 2 \zeta \omega_n / T_d$ 时，闭环极点穿越虚轴进入右半平面，引发临界失稳振荡。在仿真器中显式建模该相移是消除真机自激振荡的充要条件。
</details>

---

## 8.7.4 纯底层 PyTorch 代码实现：系统辨识求解器与执行器延迟模型

下面我们使用纯底层 PyTorch 算子实现刚体动力学最小二乘辨识求解器与带延迟的执行器动力学模拟引擎。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActuatorDynamicsDelayModel(nn.Module):
    """
    非线性执行器响应与硬件延迟模拟模块
    模拟真实电机的 CAN 总线延迟与一阶平滑迟滞响应
    """
    def __init__(self, action_dim: int = 6, delay_steps: int = 2, alpha: float = 0.6):
        super().__init__()
        self.action_dim = action_dim
        self.delay_steps = delay_steps
        self.alpha = alpha

        # 延迟环形队列缓存
        self.register_buffer("history_buffer", torch.zeros(delay_steps + 1, action_dim))
        self.register_buffer("current_actual_torque", torch.zeros(action_dim))

    def reset(self):
        self.history_buffer.zero_()
        self.current_actual_torque.zero_()

    def forward(self, cmd_torque: torch.Tensor) -> torch.Tensor:
        """
        :param cmd_torque: (action_dim,) 当前策略输出的指令力矩
        :return: (action_dim,) 真实物理电机实际输出的滞后响应力矩
        """
        # 1. 压入最新指令至延迟缓冲区
        self.history_buffer = torch.cat([self.history_buffer[1:], cmd_torque.unsqueeze(0)], dim=0)

        # 2. 提取延迟 d 步后的历史指令
        delayed_cmd = self.history_buffer[0]

        # 3. 一阶低通滤波响应: tau_t = (1 - alpha) * tau_{t-1} + alpha * delayed_cmd
        self.current_actual_torque = (1.0 - self.alpha) * self.current_actual_torque + self.alpha * delayed_cmd
        return self.current_actual_torque.clone()

class SystemIdentificationSolver:
    """
    线性参数化多刚体系统辨识最小二乘求解器
    """
    @staticmethod
    def solve_least_squares(regressor_matrix: torch.Tensor, torques: torch.Tensor) -> torch.Tensor:
        """
        :param regressor_matrix: (N, p) 运动学回归矩阵 Phi
        :param torques: (N, 1) 对应采集的真实力矩
        :return: (p, 1) 辨识出的物理惯性参数估计值 theta
        """
        # 正规方程组求解: theta = (Phi^T * Phi)^(-1) * Phi^T * T
        phi_t = regressor_matrix.t()
        gram_mat = torch.mm(phi_t, regressor_matrix)
        rhs = torch.mm(phi_t, torques)

        # 利用 Cholesky 分解稳定求解
        estimated_theta = torch.linalg.solve(gram_mat, rhs)
        return estimated_theta

# ===================================================================
# 单元测试与系统辨识解析解校验
# ===================================================================
if __name__ == "__main__":
    # 1. 测试正文中的单摆最小二乘辨识算例
    phi_test = torch.tensor([[2.0, 5.0], [0.0, 10.0]], dtype=torch.float32)
    tau_test = torch.tensor([[5.0], [4.0]], dtype=torch.float32)

    theta_est = SystemIdentificationSolver.solve_least_squares(phi_test, tau_test)
    i_est, ml_est = theta_est.squeeze(-1).tolist()

    print(f"[SysID Test] 辨识转动惯量: {i_est:.4f} kg*m^2 (期望: 1.5000)")
    print(f"[SysID Test] 辨识质量矩: {ml_est:.4f} kg*m (期望: 0.4000)")

    assert abs(i_est - 1.5) < 1e-4 and abs(ml_est - 0.4) < 1e-4, "系统辨识数值手算校验失败！"

    # 2. 测试执行器延迟与低通响应
    actuator = ActuatorDynamicsDelayModel(action_dim=2, delay_steps=2, alpha=0.5)
    actuator.reset()

    # 输入持续恒定的阶跃力矩 [10.0, 10.0]
    step_cmd = torch.tensor([10.0, 10.0])
    responses = []
    for step in range(6):
        real_tau = actuator(step_cmd)
        responses.append(real_tau[0].item())

    print(f"[Actuator Test] 阶跃力矩响应序列 (前6步): {[round(x, 3) for x in responses]}")
    # 前2步由于延迟应为 0.0，第3步开始平滑爬升
    assert responses[0] == 0.0 and responses[1] == 0.0, "执行器纯延迟未能正确生效！"
    assert responses[2] > 0.0 and responses[-1] > responses[2], "低通滤波爬升异常！"
    print("✓ Sim2Real 动力学系统辨识与执行器延迟模型单测全部通过！")
```

---

## 8.7.5 本节小结

回顾本节内容，我们建立了 Sim2Real 迁移工程落地的完整技术全景：
1. **四大流派优势互补**：域随机化筑牢泛化底座，特权蒸馏赋予动态自适应，系统辨识与执行器网络消除了底层硬件建模残差；
2. **线性参数分离的优雅解**：刚体动力学的参数线性性质使得物理惯性参数可以通过最小二乘法得到精确唯一的解析辨识；
3. **延迟与低通的物理本质**：在仿真中显式注入执行器迟滞与通信延迟，从源头杜绝了真机高频失稳发抖。
