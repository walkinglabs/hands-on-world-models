# 2.1 张量、时间与轨迹

> **第 2 章 · 预备知识**
> 
> 网格世界只有几个整数。现实中的一次观察可能包含图片、相机、语言和机器人状态，一段经历还要包含动作、奖励与时间顺序。
> 
> 这一章建立后续五条世界模型路线共用的基础语言。每个组件只讲到足以判断它的输入、输出和用途；完整实现留到真正使用它的路线章节。
> 
> 核心实验在 [2.8 动手：基础实验](/chapters/02-foundations/08-basic-experiments) 中串联看见、视觉、记忆、压缩、空间与规划。学完本章后，先明确模型最需要输出的形式（潜在向量、连续画面、特征表示、机器人动作或三维占用），再从第 4–8 章选择对应的设计路线——各路线彼此独立。

---

## 本节导读

世界模型旨在构建物理环境的内在仿真器。要让神经网络成功学习物理演化法则，首要前提并非设计精巧的注意力机制或庞大的参数规模，而是建立**严密无歧义的数据时空几何与因果表示**。在纯文本大模型中，输入是简单的一维 Token 序列；而在具身智能与物理世界模型中，输入是高度多模态、跨越连续时间且交织着因果控制动作的时空张量流。

- **核心内容**：多维张量几何与时空布局规范（$B, T, C, H, W$ 与 $B, T, H, W, C$ 内存步长与连续性）；时序因果对齐与“差一律”（Off-by-One Law，$T+1$ 观测对齐 $T$ 动作与奖励）；转移元组 $(o_t, a_t, r_t, o_{t+1}, d_t)$ 与 Episode 截断/终止拓扑（Termination vs Truncation）；时间连续性、控制频率 $\Delta t$ 与硬件延迟（Phase Lag）；轨迹批处理机制（Padding/Masking 与 Sliding Chunking）。
- **核心问题**：为什么简单的时序对齐错开 1 步会导致世界模型学到虚假因果并使规划崩溃？在离散轨迹中如何正确处理环境重置（Termination vs Truncation）以避免模型产生凭空瞬移的“虫洞陷阱”？硬件延迟与采样周期 $\Delta t$ 是如何改变物理动力学方程的？
- **核心概念**：维度语义契约（Dimension Semantic Contract）、时序因果交织（Temporal Interleaving）、差一律（Off-by-One Law）、转移元组（Transition Tuple）、阶段截断与自然终止（Truncation vs Termination）、控制周期与时间步长（$\Delta t$）、硬件时延相位滞后（Phase Lag）、轨迹切片与掩码（Trajectory Slicing & Masking）。
- **核心公式**：
  $$\tau = (o_0, a_0, r_0, o_1, a_1, r_1, \dots, o_T), \quad \hat{v}_t = \frac{\Delta x}{\Delta t}, \quad d_t = \text{terminated}_t \lor \text{truncated}_t, \quad \mathcal{M}_{i,t} = \mathbb{I}(t < T_i)$$

```text
+-----------------------------------------------------------------------------------------------+
|                                世界模型多模态时空轨迹张量流                                     |
|                                                                                               |
| 观测序列 (T+1 步):  o_0 ------------> o_1 ------------> o_2 ----------> ... ---------> o_T   |
|                      |                 ^ |               ^ |                              ^   |
| 动作与奖励 (T 步):   +---> [ a_0, r_0 ] -+ +-> [ a_1, r_1 ] -+   ...    +-> [ a_{T-1}, r_{T-1} ] -+   |
|                                                                                               |
| 状态重置信号:        d_0 = False       d_1 = False             ...       d_{T-1} = Term/Trunc |
+-----------------------------------------------------------------------------------------------+
```

---

## 1. 多维张量的时空布局与语义契约

在强化学习与具身物理交互中，单次观测可能包含车载相机图像、机器人关节角编码、末端六维力矩以及触觉阵列信号。在输入神经网络之前，所有异构感知流必须被规范化为多维张量（Tensor）。

### 1.1 维度语义契约（Dimension Semantic Contract）

一个无语义标注的张量形状（如 `[32, 10, 5]`）在工程上极其危险：它可能代表“32 个样本、时间步长 10、5 维机器人状态”，也可能代表“批大小 32、特征通道 10、5 维隐向量”，甚至是“32 个相机、10 个时间步、5 个目标框”。

在世界模型工程中，必须建立严格的**维度语义契约**：

| 模态类型 | 常见张量形状 | 各维度数学与物理语义 |
| :--- | :--- | :--- |
| **单帧彩色图像** | $[C, H, W]$ 或 $[H, W, C]$ | $C$: 颜色通道 (RGB=3); $H$: 像素高度; $W$: 像素宽度 |
| **视频序列 (PyTorch)** | $[B, T, C, H, W]$ | $B$: 批大小; $T$: 时间步数; $C$: 通道数; $H$: 高度; $W$: 宽度 |
| **视频序列 (NumPy/OpenCV)**| $[B, T, H, W, C]$ | 通道轴置于末尾（存储布局与图像编解码器原生对齐） |
| **机器人本体状态 (Proprio)**| $[B, T, P]$ | $P$: 包含关节位置 $\boldsymbol{\theta}$、速度 $\dot{\boldsymbol{\theta}}$、末端位姿 $\mathbf{x}_{\text{ee}} \in \mathrm{SE}(3)$、夹爪开合度 $g$ |
| **控制动作 (Action)** | $[B, T, A]$ | $A$: 关节期望力矩 $\boldsymbol{\tau}$、目标角位置 $\boldsymbol{\theta}_{\text{des}}$ 或末端笛卡尔速度 $\dot{\mathbf{x}}_{\text{des}}$ |
| **标量奖励 / 终止信号** | $[B, T]$ | 单时间步环境反馈标量 $r_t \in \mathbb{R}$ 与布尔终止标志 $d_t \in \{0, 1\}$ |

```text
[B, T, C, H, W] 视频张量分层结构:
┌─────────────────────────────────────────────────────────┐  Batch 维 (b=0 ... B-1)
│  ┌───────────────────────────────────────────────────┐  │  Time 维  (t=0 ... T-1)
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  │  │
│  │  │ Channel R (0)│  │ Channel G (1)│  │ Chan B(2)│  │  │  Channel 维 (c=0 ... C-1)
│  │  │  H x W 像素  │  │  H x W 像素  │  │ H x W   │  │  │  Spatial 维 (H 高, W 宽)
│  │  └──────────────┘  └──────────────┘  └─────────┘  │  │
│  │                 t = 0 瞬时观测画面                │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │                 t = 1 瞬时观测画面                │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 1.2 内存步长（Strides）与连续性（Contiguity）的底层机制

计算机物理内存（RAM / VRAM）是一维线性连续寻址空间。一个 $D$ 维张量在物理内存中的寻址由**形状（Shape）**与**步长（Strides）**共同决定。对于索引坐标 $\mathbf{i} = (i_0, i_1, \dots, i_{D-1})$，其在一维内存中的绝对线性偏移量为：

$$\text{offset}(\mathbf{i}) = \sum_{d=0}^{D-1} i_d \times \text{stride}_d$$

在标准的 C 连续（C-contiguous, 行优先）存储中，最右侧维度的步长为 1，相邻元素的物理地址连续：

$$\text{stride}_{D-1} = 1, \quad \text{stride}_d = \text{stride}_{d+1} \times \text{shape}_{d+1} \quad (d = D-2, D-3, \dots, 0)$$

例如，一个形状为 $[2, 3, 4, 4, 3]$ 的 NumPy 视频张量（`[B, T, H, W, C]`）：
- 步长元组为：$(3\times4\times4\times3, 4\times4\times3, 4\times3, 3, 1) = (144, 48, 12, 3, 1)$。

当使用 PyTorch 的 `tensor.permute(0, 1, 4, 2, 3)` 将其转换为 `[B, T, C, H, W]` 时：
- **形状变为**：$[2, 3, 3, 4, 4]$；
- **步长变为**：$(144, 48, 1, 12, 3)$。

```text
[B, T, H, W, C] 原始物理内存分布 (C-contiguous):
内存地址: | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | ...
像素数据: |R00|G00|B00|R01|G01|B01|R02|G02|B02|R03|G03|B03| ...
          └─── 像素 (0,0) ───┘ └─── 像素 (0,1) ───┘

经过 permute(0, 1, 4, 2, 3) 调整视图后 (非连续 Non-contiguous):
逻辑形状: [B, T, C, H, W]
寻址通道 R: 逻辑上连续访问 R00 -> R01 -> R02 -> R03
物理跨度:   内存索引 0 -> 3 -> 6 -> 9 (步长为 3，内存物理地址发生跳跃!)
```

由于物理内存中的字节并没有被真正移动，该张量变成了**非连续张量（Non-contiguous Tensor）**。若直接将非连续张量送入底层依赖高效连续访存的 C++/CUDA 算子（如 `conv3d`、cuDNN 或 Transformer 的 FlashAttention），或调用 `.view()`，程序将立即崩溃并抛出：

```text
RuntimeError: view size is not compatible with input tensor's size and stride 
(at least one dimension spans across two contiguous subspaces). Use .reshape(...) or .contiguous().
```

因此，在任何跨维度转置操作之后，如果下游算子依赖连续内存，必须显式调用 `.contiguous()`。这会在底层分配一块全新的连续显存并执行物理数据拷贝：

```python
# 安全转换规范
video_torch = video_np_tensor.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, H, W, C] -> [B, T, C, H, W]
```

### 1.3 具身智能多模态字典张量容器

在现代多模态世界模型（如 DreamerV3、Robomimic、OpenVLA）中，单一样本通常包含多路视觉相机、力觉传感器与语言指令。最稳健的工程容器是**统一时序字典张量（Multimodal Tensor Dict）**：

```python
from typing import Dict
import torch

BatchDict = Dict[str, torch.Tensor]

# 标准 BatchDict 结构示例：
batch: BatchDict = {
    # 观测组 (均具备 T+1 时间步)
    "camera_wrist": torch.randn(16, 11, 3, 128, 128),  # 腕部视角: [B, T+1, C, H, W]
    "camera_front": torch.randn(16, 11, 3, 256, 256),  # 广角视角: [B, T+1, C, H, W]
    "proprio":      torch.randn(16, 11, 14),            # 机械臂双臂 14-DoF 关节位姿: [B, T+1, P]
    # 控制与动力学组 (均具备 T 时间步)
    "actions":      torch.randn(16, 10, 7),             # 控制动作: [B, T, A]
    "rewards":      torch.zeros(16, 10),                # 环境奖励: [B, T]
    "terminated":   torch.zeros(16, 10, dtype=torch.bool), # 自然终止标志: [B, T]
    "truncated":    torch.zeros(16, 10, dtype=torch.bool), # 步数截断标志: [B, T]
    "masks":        torch.ones(16, 11, dtype=torch.bool),  # 有效步掩码: [B, T+1]
}
```

---

## 2. 时序因果交织与“差一律”（Off-by-One Law）

在世界模型与强化学习中，时序索引的处理极易滋生隐蔽且致命的 Bug。其中最核心的法则是：**时序因果交织下的差一律（Off-by-One Law）**。

### 2.1 物理因果链与差一律推导

在马尔可夫决策过程（MDP）中，时间以离散脉冲推进。一个完整的轨迹片段包含环境与智能体的交替交互：

$$o_0 \xrightarrow{a_0} (r_0, o_1) \xrightarrow{a_1} (r_1, o_2) \xrightarrow{a_2} \dots \xrightarrow{a_{T-1}} (r_{T-1}, o_T)$$

从数学因果律推导：
1. 在 $t=0$ 刻，智能体首先接收到环境的**初始观测 $o_0$**；
2. 智能体基于 $o_0$ 输出控制动作 $a_0$；
3. 环境物理引擎接收 $a_0$，推进一个物理步长 $\Delta t$，产生反馈标量 $r_0$ 并转移至**后继观测 $o_1$**；
4. 这一过程重复 $T$ 次。

```text
时间轴 (Time Axis):
  t = 0             t = 1             t = 2                      t = T-1           t = T
    |                 |                 |                          |                 |
  [o_0]             [o_1]             [o_2]       ...          [o_{T-1}]           [o_T]  <- 共 T+1 个观测
    |                 |                 |                          |
    +---- a_0 --------+---- a_1 --------+---- a_2 ...             +---- a_{T-1} -----+      <- 共 T 个动作
    |     r_0         |     r_1         |                          |     r_{T-1}
```

**差一律的核心结论**：
在一个包含 $T$ 个动作决策步骤（$T$ 个转移步）的时间跨度内，**观测序列天然包含 $T+1$ 帧**（起点 $o_0$ 到终点 $o_T$），而**动作序列、单步奖励序列与终止标志序列各包含 $T$ 个元素**。

$$\text{Length}(\mathbf{o}) = T+1, \quad \text{Length}(\mathbf{a}) = T, \quad \text{Length}(\mathbf{r}) = T, \quad \text{Length}(\mathbf{d}) = T$$

### 2.2 致命的时序错配：因果反转与动力学延迟

在实现时序批处理时，若开发者机械地将所有变量截断为相同的长度 $T$ 并直接在同一循环中对齐，将引发灾难性的模型崩溃：

```text
【错误对齐方式 A：前向错配 (Forward Shift)】
观测:   o_0      o_1      o_2      ...  o_{T-1}
动作:   a_0      a_1      a_2      ...  a_{T-1}
模型:   试图用 a_t 预测从 o_{t-1} 到 o_t 的变化 (用未来的动作预测过去的观测!)
后果:   因果反转 (Acausal Leakage)。世界模型在推演未来时彻底忽略输入的动作。

【错误对齐方式 B：后向错配 (Backward Shift)】
观测:   o_1      o_2      o_3      ...  o_T
动作:   a_0      a_1      a_2      ...  a_{T-1}
模型:   试图用 a_t 预测从 o_{t+1} 到 o_{t+2} 的变化 (动作作用滞后了一个周期!)
后果:   动力学虚假时延。基于该世界模型规划的策略在实机部署时将发生剧烈超调与发散。
```

#### 因果有向无环图（DAG）证明

根据 Pearl 因果图理论：
- **真实因果 DAG**：$a_t \to o_{t+1}$，且 $o_t \to o_{t+1}$。在给定 $(o_t, a_t)$ 的条件下，$o_{t+1}$ 条件独立于未来动作 $a_{t+1}$：
  $$I(o_{t+1}; a_{t+1} \mid o_t, a_t) = 0$$
- **前向错配下的伪模型**：若将数据集强行重构为 $(o_t, a_t) \to o_t$，由于 $o_t$ 发生于 $a_t$ 施加之前，在物理真实分布中 $o_t \perp a_t$（在给定历史策略前提下无直接因果）。模型为了最小化预测误差，会将 $a_t$ 的权重梯度直接优化为 0，退化为纯自回归视频生成器（Action-free Video Predictor）。

在闭环推演与模型预测控制（MPC）中，一个动作无关的世界模型对任何规划候选路径都会给出相同的未来想象，导致**规划器彻底失效**。

---

## 3. Episode、Transition 与环境重置拓扑

世界模型训练依赖大量交互数据。我们需要从最微观的单步转移逐步构建宏观的轨迹流，并精确处理重置边界。

### 3.1 Transition 元组与 Episode 序列

- **Transition（单步转移）**：强化学习经验回放（Replay Buffer）的原子单元：
  $$e_t = (o_t, a_t, r_t, o_{t+1}, d_t)$$
  其中 $d_t \in \{0, 1\}$ 为环境终止布尔指示变量。
- **Episode（完整轨迹回合）**：从环境初始分布 $o_0 \sim \rho_0(\cdot)$ 开始，直到环境发出重置信号为止的不可分割时序链：
  $$\tau = (o_0, a_0, r_0, o_1, a_1, r_1, \dots, o_{T-1}, a_{T-1}, r_{T-1}, o_T)$$

### 3.2 自然终止（Termination）vs 步数截断（Truncation）

在早期的强化学习库（如 OpenAI Gym v0.21 之前）中，环境只返回单个布尔值 `done`。这一粗糙设计掩盖了两种截然不同的物理与拓扑语义。现代环境规范（Gymnasium API）将 `done` 严格拆分为 `terminated` 与 `truncated`：

$$d_t = \text{terminated}_t \lor \text{truncated}_t$$

```text
                     ┌─────────────────── 环境重置信号 d_t ───────────────────┐
                     │                                                        │
          自然终止 (Termination)                                   步数截断 (Truncation)
  ┌──────────────────────────────────────┐                ┌──────────────────────────────────────┐
  │ 物理意义: 触碰吸收态或任务终结       │                │ 物理意义: 达到人为设定的时间上限     │
  │ 典型场景: 倒立摆倒地、机械臂撞坏     │                │ 典型场景: TimeLimit.max_steps=1000   │
  │ 贝尔曼目标: 吸收态未来价值 V*(s) = 0  │                │ 贝尔曼目标: 物理未终结, 需 Bootstrap │
  │ 世界模型: 物理演化在该步真正停止     │                │ 世界模型: 物理规律未变, 仅观测截断   │
  └──────────────────────────────────────┘                └──────────────────────────────────────┘
```

#### 数学差异与贝尔曼价值估计

在利用世界模型进行价值学习（Actor-Critic）时，两者的 Bellman 目标存在本质不同：

1. **若发生自然终止 ($\text{terminated}_t = \text{True}$)**：
   系统转移至终止吸收态（Absorbing State），该状态下未来累积收益恒为 0：
   $$y_t = r_t + \gamma (1 - \text{terminated}_t) V_\phi(o_{t+1}) = r_t + 0 = r_t$$

2. **若发生步数截断 ($\text{truncated}_t = \text{True}$ 且 $\text{terminated}_t = \text{False}$)**：
   系统并未进入物理死亡，仅因数据记录窗口耗尽而停止。后继状态 $o_{t+1}$ 仍具有物理延续性与未来价值，**必须保留价值自举（Bootstrapping）**：
   $$y_t = r_t + \gamma V_\phi(o_{t+1})$$

若错误地将截断视为自然终止，世界模型与策略在临近最大步数（如 $t=990$ 到 $1000$）时，会误认为世界即将“毁灭”且未来收益归零，诱发灾难性的**临界恐慌行为（Horizon-Induced Panic）**。

### 3.3 虫洞陷阱（Wormhole Trap）

在构建多轨迹批处理缓冲区时，若将多条不同 Episode 的轨迹无间隔地展平拼接在同一个一维物理数组中：

```text
内存线性排布:  [... Episode A 的最后一步 o_T^{(A)} ]  [ Episode B 的初始步 o_0^{(B)} ...]
                                               \      /
                                                \    /
                                              跨界转移
```

当世界模型在训练时对该数组执行步长为 1 的滑动窗口采样时，会强行采样到如下非法转移：

$$(o_T^{(A)}, a_T^{(A)}, r_T^{(A)}, o_0^{(B)})$$

**物理后果**：
世界模型被迫学习一个反常的动力学函数：在执行动作 $a_T^{(A)}$ 后，画面瞬间从 Episode A 的终点状态“超距瞬移”到了 Episode B 的起点状态。这种现象被称为**虫洞陷阱（Wormhole Trap）**。在基于世界模型的自回归长程规划中，模型会产生强烈的幻觉，试图通过某些动作组合“穿越虫洞”直接瞬移到目标位置。

**工程防御准则**：在轨迹缓冲区中，不同 Episode 必须以独立的列表/容器存储；若拼接入连续张量，必须在边界处通过布尔有效掩码（Mask）显式隔断因果自注意力与时序损失计算。

---

## 4. 连续物理时间、采样率 $\Delta t$ 与硬件时延

真实世界是连续时间动力学系统，而数字控制器与深度模型只能在离散时钟步长下运行。理解连续时间到离散时序的映射，是构建物理保真世界模型的核心基础。

### 4.1 微分动力学与离散数值积分

物理世界的运动受一阶/二阶常微分方程（ODE）控制：

$$\frac{\mathrm{d}\mathbf{s}(t)}{\mathrm{d}t} = \dot{\mathbf{s}}(t) = f_{\text{phy}}(\mathbf{s}(t), \mathbf{a}(t))$$

数字采样以固定周期 $\Delta t$（即采样频率 $f_s = 1/\Delta t$）对连续系统进行离散化：
- **一阶前向欧拉积分（Forward Euler）**：
  $$\mathbf{s}_{k+1} = \mathbf{s}_k + \Delta t \cdot f_{\text{phy}}(\mathbf{s}_k, \mathbf{a}_k) + \mathcal{O}(\Delta t^2)$$
- **四阶龙格-库塔积分（Runge-Kutta 4th Order, RK4）**：
  $$\mathbf{s}_{k+1} = \mathbf{s}_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4) + \mathcal{O}(\Delta t^5)$$

由此可见，离散状态转移算子 $g_\theta(\mathbf{s}_k, \mathbf{a}_k) \approx \mathbf{s}_{k+1}$ 的数学形式**强依赖于采样周期 $\Delta t$**。如果在采集数据时混用了不同控制频率（例如一部分轨迹来自 $10\text{ Hz}$ 采样，另一部分来自 $50\text{ Hz}$ 采样），而未将 $\Delta t$ 作为显式条件输入网络，世界模型将面临严重的动力学多义性，无法收敛到统一的物理规律。

### 4.2 采样周期 $\Delta t$ 的工程权衡

选择控制与世界模型预测的时间步长 $\Delta t$ 涉及经典的工程权衡：

```text
                  控制采样频率 f_s = 1 / Δt
  0.5 Hz (Δt=2.0s)                                   500 Hz (Δt=0.002s)
  ◄───────────────────────────────────────────────────────────────────►
  【低频采样区域】                                   【高频采样区域】
  • 接触碰撞严重跳跃与穿模                           • 单步像素差分 Δo ≈ 0 (淹没于噪声)
  • 丢失高频动力学特征                               • 规划长程目标需要过大的时间步数 T
  • 动力学呈现剧烈非线性                             • 显存与自回归计算开销爆炸
```

1. **Nyquist-Shannon 采样定理**：控制频率 $f_s$ 必须至少大于系统最高机械动态频宽 $f_{\max}$ 的 2 倍。例如，抓取刚体碰撞过程的动态频宽通常在 $20\text{ Hz} \sim 50\text{ Hz}$，因此机器人世界模型常用的控制周期为 $\Delta t \in [0.02\text{ s}, 0.1\text{ s}]$（即 $10\text{ Hz} \sim 50\text{ Hz}$）。
2. **感知信噪比约束**：若 $\Delta t$ 过小（如 $1\text{ ms}$），连续两帧图像的像素位移几乎为零，模型反向传播计算的梯度将主要用于拟合相机 CCD 的高斯热噪声与光照抖动。

### 4.3 硬件时延（Hardware Latency）与相位滞后（Phase Lag）

在真实物理机器人系统中，时间并非理想的阶跃时钟。从相机光子感光到电机产生力矩，存在不可忽略的**物理时延链路**：

$$\delta_{\text{total}} = \delta_{\text{cam}} + \delta_{\text{infer}} + \delta_{\text{comm}} + \delta_{\text{act}}$$

- **相机曝光与传输延迟** $\delta_{\text{cam}} \approx 20 \sim 35\text{ ms}$；
- **神经网络大模型推理延迟** $\delta_{\text{infer}} \approx 20 \sim 50\text{ ms}$；
- **总线通信与主控分发延迟** $\delta_{\text{comm}} \approx 2 \sim 5\text{ ms}$；
- **电机驱动器响应与上升时间** $\delta_{\text{act}} \approx 10 \sim 20\text{ ms}$。

```text
理想无延迟系统:
时间点 t:         观测 o_t ──> 策略立即输出 a_t ──> 物理世界立即执行 a_t

实际物理时延链路 (总延迟 δ ≈ 60 ms):
时间 t = 0 ms:   相机快门触发拍摄 o_0
时间 t = 25 ms:  图像解包并传输到 GPU 显存
时间 t = 60 ms:  世界模型与策略推理完成，发出动作指令 a_0
时间 t = 80 ms:  底层电机力矩达到目标值 (此时物理世界早已演化至 t = 80 ms 的新状态!)
```

#### 相位滞后引发的控制发散

在控制理论中，纯时延 $\delta$ 会在频域引入相位滞后 $\Delta \phi = -\omega \delta$。若世界模型简单地假设 $a_t$ 作用于 $o_t$ 并产生 $o_{t+1}$，在实机闭环控制中，策略输出的控制指令将永远落后于物理状态半个周期，引发严重的**机械共振与控制发散（Phase Lag Oscillation）**。

#### 解决方案：动作队列与时延状态增广

为了让具有延迟的系统重新恢复严格的马尔可夫性质，世界模型必须引入**时延增广状态空间（Latency-Augmented State Space）**：

$$\tilde{\mathbf{s}}_t = \big(o_t, \underbrace{a_{t-1}, a_{t-2}, \dots, a_{t-K}}_{\text{飞行中动作 (In-flight Actions)}}\big)$$

世界模型显式地以当前滞后观测 $o_t$ 与历史上已发出但尚未完全生效的动作队列 $(a_{t-K}, \dots, a_{t-1})$ 作为联合输入，预演并预测在未来实际生效时刻 $t + \delta$ 的真实状态。

---

## 5. 变长轨迹批处理：切片抽取与填充掩码

在离线轨迹数据集（如 D4RL、Open X-Embodiment）中，每条 Episode 的交互步数通常是变长的（例如成功任务耗时 45 步，失败任务耗时 200 步）。GPU 矩阵乘法与批处理张量要求维度规整，因此需要两种经典的批处理策略。

```text
原始变长轨迹数据集:
Episode 0: [====== 6 步 ======]
Episode 1: [================== 12 步 ==================]
Episode 2: [========= 8 步 =========]

───────────────────────────────────────────────────────────────────────────
策略 A: 滑动窗口切片 (Sliding Window Chunking, 固定窗口 H = 4 步)
Chunk 0.0: [====]            Chunk 1.0: [====]            Chunk 2.0: [====]
Chunk 0.1:   [====]          Chunk 1.1:   [====]          Chunk 2.1:   [====]
(特点: 显存利用率 100%, 适合定长自回归与 RSSM 训练, 严格满足差一律 H+1 vs H)

───────────────────────────────────────────────────────────────────────────
策略 B: 动态填充与掩码 (Padding & Masking, 统一填充至 H_max = 12 步)
Episode 0: [====== 6 步 ====== | 0 0 0 0 0 0 (Padding)]  Mask: [1 1 1 1 1 1 1 0 0 0 0 0 0]
Episode 1: [================== 12 步 ==================]  Mask: [1 1 1 1 1 1 1 1 1 1 1 1 1]
Episode 2: [========= 8 步 ========= | 0 0 0 0 (Padding)] Mask: [1 1 1 1 1 1 1 1 1 0 0 0 0]
(特点: 保留整条轨迹完整全局信息, 需在损失函数中引入有效掩码归一化)
```

### 5.1 滑动窗口切片（Sliding Window Chunking）

从一条长度为 $L_i$ 的轨迹中，随机截取长度为 $H$（Horizon）的连续子轨迹：
- **截取观测**：$o_{t : t+H+1}$（长度为 $H+1$）；
- **截取动作**：$a_{t : t+H}$（长度为 $H$）；
- **截取奖励与标志**：$r_{t : t+H}, d_{t : t+H}$（长度各为 $H$）。

**适用场景**：Dreamer、RSSM、以及绝大多数基于固定上下文窗口的时序世界模型。它的显存利用效率极高，完全避免了无效计算。

### 5.2 填充与有效掩码（Padding & Masking）

当必须对完整 Episode 进行整轨批处理（例如训练全局 Decision Transformer 或变长扩散生成）时，将一个 Batch 内的所有样本沿时间维度对齐填充至最大长度 $H_{\max}$：

- **张量填充**：使用零值对观测与动作在末尾补齐；
- **有效掩码张量** $\mathcal{M} \in \{0, 1\}^{B \times (H_{\max}+1)}$：
  $$\mathcal{M}_{i, t} = \mathbb{I}(t \le T_i)$$

#### 掩码损失函数的正确归一化

在计算世界模型的重构与预测损失时，必须通过掩码消除 Padding 填充区域的伪梯度：

$$\mathcal{L}_{\text{batch}} = \frac{\sum_{i=1}^B \sum_{t=0}^{H_{\max}-1} \mathcal{M}_{i, t+1} \cdot \ell\big(\hat{o}_{i, t+1}, o_{i, t+1}\big)}{\sum_{i=1}^B \sum_{t=0}^{H_{\max}-1} \mathcal{M}_{i, t+1} + \epsilon}$$

**关键细节**：损失归一化的分母必须是**有效步数的实际总和** $\sum \mathcal{M}$，而绝对不能除以张量总尺寸 $B \times H_{\max}$。若除以 $B \times H_{\max}$，有效步数较少的批次其有效梯度会被大幅稀释，导致短轨迹样本欠拟合。

---

## 6. 简洁实现：TrajectoryBuffer 与时序因果校验器

下面提供一份工业级、完全自包含且可直接执行的 Python/PyTorch 代码。该模块包含多模态轨迹缓冲区 `TrajectoryBuffer` 与时序因果自动化校验器 `TimeInterleavedValidator`。

```python
"""
动手学世界模型 - 基础模块 2.1
TrajectoryBuffer: 多模态轨迹回放缓冲区与变长切片批处理器
TimeInterleavedValidator: 时序因果、差一律与虫洞陷阱自动化校验器
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch


class TrajectoryBuffer:
    """多模态轨迹回放缓冲区，支持存储变长 Episode 并提供 Chunking 与 Padding 采样。"""

    def __init__(
        self,
        capacity_episodes: int,
        obs_spec: Dict[str, Tuple[int, ...]],
        act_dim: int,
    ):
        """
        参数:
            capacity_episodes: 缓冲区容纳的最大 Episode 数量
            obs_spec: 观测模态字典, 如 {'image': (3, 64, 64), 'proprio': (7,)}
            act_dim: 控制动作维度
        """
        self.capacity = capacity_episodes
        self.obs_spec = obs_spec
        self.act_dim = act_dim
        self.episodes: List[Dict[str, np.ndarray]] = []
        self.total_transitions = 0

    def add_episode(self, episode: Dict[str, np.ndarray]) -> None:
        """存入完整 Episode 并严格执行差一律与数据规范校验。"""
        actions = episode["actions"]
        T = actions.shape[0]

        # 1. 验证观测维度是否严格为 T + 1
        for key in self.obs_spec:
            assert key in episode, f"缺少观测键名: {key}"
            assert episode[key].shape[0] == T + 1, (
                f"观测 '{key}' 长度 ({episode[key].shape[0]}) 必须等于 T + 1 ({T + 1})"
            )

        # 2. 验证动作、奖励与终止标志长度是否严格为 T
        assert actions.shape == (T, self.act_dim), f"动作形状错误: {actions.shape}"
        assert episode["rewards"].shape == (T,), f"奖励长度错误: {episode['rewards'].shape}"
        assert episode["terminated"].shape == (T,), f"terminated 长度错误: {episode['terminated'].shape}"
        assert episode["truncated"].shape == (T,), f"truncated 长度错误: {episode['truncated'].shape}"

        if len(self.episodes) >= self.capacity:
            removed = self.episodes.pop(0)
            self.total_transitions -= removed["actions"].shape[0]

        self.episodes.append(episode)
        self.total_transitions += T

    def sample_chunks(
        self, batch_size: int, chunk_length: int
    ) -> Dict[str, torch.Tensor]:
        """
        定长滑动窗口采样 (Sliding Window Chunking)
        返回张量形状严格符合差一律:
            - 观测: [B, H + 1, ...]
            - 动作: [B, H, A]
            - 奖励: [B, H]
            - 标志: [B, H]
            - 掩码: [B, H + 1] (全 1)
        """
        H = chunk_length
        valid_episodes = [ep for ep in self.episodes if ep["actions"].shape[0] >= H]
        if not valid_episodes:
            raise ValueError(f"缓冲区中没有长度 >= {H} 的轨迹可供采样!")

        sampled: Dict[str, List[np.ndarray]] = {k: [] for k in self.obs_spec}
        sampled["actions"] = []
        sampled["rewards"] = []
        sampled["terminated"] = []
        sampled["truncated"] = []
        sampled["masks"] = []

        for _ in range(batch_size):
            ep = valid_episodes[np.random.randint(0, len(valid_episodes))]
            T = ep["actions"].shape[0]
            start = np.random.randint(0, T - H + 1)
            end = start + H

            for k in self.obs_spec:
                sampled[k].append(ep[k][start : end + 1])
            sampled["actions"].append(ep["actions"][start:end])
            sampled["rewards"].append(ep["rewards"][start:end])
            sampled["terminated"].append(ep["terminated"][start:end])
            sampled["truncated"].append(ep["truncated"][start:end])
            sampled["masks"].append(np.ones(H + 1, dtype=bool))

        return {k: torch.from_numpy(np.stack(v)) for k, v in sampled.items()}

    def sample_padded_batch(
        self, batch_size: int, max_len: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """变长填充批采样 (Padding & Masking)"""
        indices = np.random.randint(0, len(self.episodes), size=batch_size)
        batch_eps = [self.episodes[i] for i in indices]
        actual_max_t = max(ep["actions"].shape[0] for ep in batch_eps)
        H = max_len if max_len is not None else actual_max_t

        sampled: Dict[str, List[np.ndarray]] = {k: [] for k in self.obs_spec}
        sampled["actions"] = []
        sampled["rewards"] = []
        sampled["terminated"] = []
        sampled["truncated"] = []
        sampled["masks"] = []

        for ep in batch_eps:
            T = ep["actions"].shape[0]
            cur_t = min(T, H)

            for k, shape in self.obs_spec.items():
                pad_obs = np.zeros((H + 1, *shape), dtype=ep[k].dtype)
                pad_obs[: cur_t + 1] = ep[k][: cur_t + 1]
                sampled[k].append(pad_obs)

            pad_act = np.zeros((H, self.act_dim), dtype=ep["actions"].dtype)
            pad_act[:cur_t] = ep["actions"][cur_t * 0 : cur_t]
            sampled["actions"].append(pad_act)

            pad_rew = np.zeros(H, dtype=ep["rewards"].dtype)
            pad_rew[:cur_t] = ep["rewards"][:cur_t]
            sampled["rewards"].append(pad_rew)

            pad_term = np.zeros(H, dtype=bool)
            pad_term[:cur_t] = ep["terminated"][:cur_t]
            sampled["terminated"].append(pad_term)

            pad_trunc = np.zeros(H, dtype=bool)
            pad_trunc[:cur_t] = ep["truncated"][:cur_t]
            sampled["truncated"].append(pad_trunc)

            mask = np.zeros(H + 1, dtype=bool)
            mask[: cur_t + 1] = True
            sampled["masks"].append(mask)

        return {k: torch.from_numpy(np.stack(v)) for k, v in sampled.items()}


class TimeInterleavedValidator:
    """时序因果对齐与数据拓扑自动化校验器"""

    @staticmethod
    def validate_off_by_one(batch: Dict[str, torch.Tensor]) -> bool:
        """严格校验差一律: 观测与掩码长度为 T+1, 动作/奖励/终止为 T"""
        T_act = batch["actions"].shape[1]
        for key in ["rewards", "terminated", "truncated"]:
            if key in batch:
                assert batch[key].shape[1] == T_act, (
                    f"控制信号 '{key}' 长度 ({batch[key].shape[1]}) != 动作长度 ({T_act})"
                )

        for key, tensor in batch.items():
            if key not in ["actions", "rewards", "terminated", "truncated", "masks"]:
                assert tensor.shape[1] == T_act + 1, (
                    f"观测张量 '{key}' 长度 ({tensor.shape[1]}) 必须等于动作长度 + 1 ({T_act + 1})"
                )

        if "masks" in batch:
            assert batch["masks"].shape[1] == T_act + 1, (
                f"掩码长度 ({batch['masks'].shape[1]}) 必须等于动作长度 + 1 ({T_act + 1})"
            )
        return True

    @staticmethod
    def validate_no_wormholes(batch: Dict[str, torch.Tensor]) -> bool:
        """校验是否存在跨 Episode 边界无断点拼接的虫洞陷阱"""
        dones = batch["terminated"] | batch["truncated"]
        masks = batch["masks"]
        B, T = dones.shape
        for b in range(B):
            done_indices = torch.where(dones[b])[0].tolist()
            if len(done_indices) > 0:
                first_done = done_indices[0]
                # 在第一个 done 发生之后，后继观测若存在，其后的掩码必须全部置 False
                if first_done + 2 < masks.shape[1]:
                    invalid_active = torch.any(masks[b, first_done + 2 :])
                    assert not invalid_active, (
                        f"样本 {b} 在步骤 {first_done} 触发 done 后存在非法活跃转移 (虫洞陷阱)!"
                    )
        return True

    @staticmethod
    def inspect_memory_strides(tensor_dict: Dict[str, torch.Tensor]) -> None:
        """打印张量的内存步长与连续性状态"""
        print("-" * 75)
        print(f"{'张量键名':<16} | {'形状 (Shape)':<20} | {'步长 (Strides)':<22} | {'连续性'}")
        print("-" * 75)
        for k, v in tensor_dict.items():
            print(f"{k:<16} | {str(list(v.shape)):<20} | {str(v.stride()):<22} | {v.is_contiguous()}")
        print("-" * 75)


# -------------------------------------------------------------------------
# 自动化执行验证
# -------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. 实例化缓冲区 (图像 + 7维机械臂关节)
    obs_spec = {"camera_rgb": (3, 64, 64), "joint_states": (7,)}
    buffer = TrajectoryBuffer(capacity_episodes=50, obs_spec=obs_spec, act_dim=4)

    # 2. 模拟写入 10 条变长轨迹 (每条 15~30 步)
    for ep_id in range(10):
        T_ep = np.random.randint(15, 30)
        episode_data = {
            "camera_rgb": np.random.randn(T_ep + 1, 3, 64, 64).astype(np.float32),
            "joint_states": np.random.randn(T_ep + 1, 7).astype(np.float32),
            "actions": np.random.randn(T_ep, 4).astype(np.float32),
            "rewards": np.random.randn(T_ep).astype(np.float32),
            "terminated": np.zeros(T_ep, dtype=bool),
            "truncated": np.zeros(T_ep, dtype=bool),
        }
        # 最后一步设定为自然终止
        episode_data["terminated"][-1] = True
        buffer.add_episode(episode_data)

    print(f"成功存入 {len(buffer.episodes)} 条 Episode，总转移步数: {buffer.total_transitions}")

    # 3. 采样定长滑动窗口 Chunk (H = 8)
    chunk_batch = buffer.sample_chunks(batch_size=4, chunk_length=8)
    print("\n[1] 定长切片采样 (Chunk Batch, H=8):")
    TimeInterleavedValidator.inspect_memory_strides(chunk_batch)

    # 4. 执行严密性校验
    assert TimeInterleavedValidator.validate_off_by_one(chunk_batch)
    assert TimeInterleavedValidator.validate_no_wormholes(chunk_batch)
    print(">> 定长切片差一律与虫洞校验 100% 通过!")

    # 5. 采样变长 Padding 批次 (最大长度对齐至 H_max = 20)
    pad_batch = buffer.sample_padded_batch(batch_size=3, max_len=20)
    print("\n[2] 变长填充采样 (Padded Batch, H_max=20):")
    TimeInterleavedValidator.inspect_memory_strides(pad_batch)

    assert TimeInterleavedValidator.validate_off_by_one(pad_batch)
    assert TimeInterleavedValidator.validate_no_wormholes(pad_batch)
    print(">> 变长填充掩码与拓扑校验 100% 通过!")
```

运行输出：

```text
成功存入 10 条 Episode，总转移步数: 221

[1] 定长切片采样 (Chunk Batch, H=8):
---------------------------------------------------------------------------
张量键名             | 形状 (Shape)         | 步长 (Strides)         | 连续性
---------------------------------------------------------------------------
camera_rgb       | [4, 9, 3, 64, 64]    | (110592, 12288, 4096, 64, 1) | True
joint_states     | [4, 9, 7]            | (63, 7, 1)             | True
actions          | [4, 8, 4]            | (32, 4, 1)             | True
rewards          | [4, 8]               | (8, 1)                 | True
terminated       | [4, 8]               | (8, 1)                 | True
truncated        | [4, 8]               | (8, 1)                 | True
masks            | [4, 9]               | (9, 1)                 | True
---------------------------------------------------------------------------
>> 定长切片差一律与虫洞校验 100% 通过!

[2] 变长填充采样 (Padded Batch, H_max=20):
---------------------------------------------------------------------------
张量键名             | 形状 (Shape)         | 步长 (Strides)         | 连续性
---------------------------------------------------------------------------
camera_rgb       | [3, 21, 3, 64, 64]   | (258048, 12288, 4096, 64, 1) | True
joint_states     | [3, 21, 7]           | (147, 7, 1)            | True
actions          | [3, 20, 4]           | (80, 4, 1)             | True
rewards          | [3, 20]              | (20, 1)                | True
terminated       | [3, 20]              | (20, 1)                | True
truncated        | [3, 20]              | (20, 1)                | True
masks            | [3, 21]              | (21, 1)                | True
---------------------------------------------------------------------------
>> 变长填充掩码与拓扑校验 100% 通过!
```

---

## 7. 练习与思考

### 习题 1：张量步长与视图计算证明

给定一个形状为 `[4, 16, 64, 64, 3]` 的 PyTorch 视频张量 $\mathbf{X}$（符合 NHWC 顺序，数据类型为 `float32`，每元素占用 4 字节）：
1. 写出该张量在标准 C-contiguous 存储下的步长元组 $\text{stride}(\mathbf{X})$；
2. 执行 $\mathbf{Y} = \mathbf{X}.\text{permute}(0, 1, 4, 2, 3)$ 得到 NCHW 格式，写出 $\mathbf{Y}$ 的形状与步长元组；
3. 证明：若直接对 $\mathbf{Y}$ 调用 `.view(4, 16, 3 * 64 * 64)`，为何底层硬件指针寻址公式 $\text{offset}(\mathbf{i}) = \sum i_d \times \text{stride}_d$ 无法通过单组连续基地址常数展开，从而必然触发运行时异常？

::: details 思考与解析
1. 初始形状为 $[4, 16, 64, 64, 3]$。C-contiguous 步长为：
   $$\text{stride}(\mathbf{X}) = (16 \times 64 \times 64 \times 3, \; 64 \times 64 \times 3, \; 64 \times 3, \; 3, \; 1) = (196608, 12288, 192, 3, 1)$$
2. 经过 `permute(0, 1, 4, 2, 3)` 后，维度由 $(0, 1, 2, 3, 4)$ 映射为 $(0, 1, 4, 2, 3)$：
   - 形状为：$[4, 16, 3, 64, 64]$；
   - 步长对应原步长的置换：$(196608, 12288, 1, 192, 3)$。
3. **证明**：`.view()` 要求被合并的维度在物理内存中构成单调递增且物理相邻的一维内存切片。在 $\mathbf{Y}$ 中，空间高度 $H$ 维的步长为 192，通道 $C$ 维的步长为 1，宽度 $W$ 维的步长为 3。物理内存中每跨越一个像素的 $C$ 通道，物理地址只加 1；但要跨越到下一行高度 $H$，物理地址跳跃 192。这导致空间维度与通道维度的元素在底层物理内存中是交错离散存放的。要在逻辑上合并为单一的一维长度 $3 \times 64 \times 64 = 12288$，无法找到一个恒定的线性基步长 $S$ 满足 $i_{\text{flat}} \times S = \text{offset}$。因此物理内存非连续，必须通过 `.contiguous()` 重新物理拷贝数据。
:::

### 习题 2：时序差一律与因果图 D-分离证明

设真实物理世界转移概率为 $P(o_{t+1} \mid o_t, a_t)$，智能体策略为 $a_t \sim \pi(a_t \mid o_t)$。若在训练数据管道中发生了**前向错配（Forward Shift）**，即将动作张量向左平移 1 步，构建了虚假转移样本 $(o_t, a_{t+1}, o_{t+1})$：
1. 画出包含节点 $o_t, a_t, o_{t+1}, a_{t+1}$ 的真实时序因果有向无环图（DAG）；
2. 利用 D-分离（D-Separation）准则，证明在给定当前观测 $o_t$ 的条件下，虚假输入动作 $a_{t+1}$ 与转移目标 $o_{t+1}$ 之间的条件互信息恒为 0（即世界模型将彻底退化为动作无关模型）：
   $$I(o_{t+1}; a_{t+1} \mid o_t) = 0 \quad (\text{在开环被动策略下})$$

::: details 思考与解析
1. 因果 DAG 路径为：$o_t \to a_t \to o_{t+1} \to a_{t+1} \to o_{t+2}$，同时存在物理惯性直接边 $o_t \to o_{t+1}$。
2. **证明**：考虑节点 $o_{t+1}$ 与 $a_{t+1}$。从 $o_{t+1}$ 到 $a_{t+1}$ 唯一的因果边是 $o_{t+1} \to a_{t+1}$（因策略依赖观测）。这是一个标准的顺向因果链（Chain: $o_t \to o_{t+1} \to a_{t+1}$）。
   根据 D-分离准则：因果链中 $o_{t+1}$ 是中间节点。但在前向错配模型中，目标是用 $(o_t, a_{t+1})$ 预测 $o_{t+1}$。
   因为 $a_{t+1}$ 是 $o_{t+1}$ 的因果后继（Effect / Descendant），在物理时间上 $a_{t+1}$ 发生在 $o_{t+1}$ 之后。在给定历史 $o_t, a_t$ 的前提下，$o_{t+1}$ 的真实生成分布完全由系统物理动力学决定，与未来尚未发生的动作 $a_{t+1}$ 条件独立。因此模型反向传播时无法从 $a_{t+1}$ 提取出关于 $o_{t+1}$ 的额外动力学解释力，导致动作权重退化为 0。
:::

### 习题 3：终止与截断的价值估计偏差推导

设环境单步奖励恒为常数 $r_t = +1$，折扣因子 $\gamma = 0.9$。环境设置了最大时间上限 $T_{\max} = 100$ 步，物理过程在 100 步后实际上可以无限延续：
1. 计算真实物理系统下的无限步真实状态价值 $V^*(s_t)$；
2. 若算法错误地将 $t=100$ 处的 Truncation 视为 Termination（将 $V(s_{100})$ 强制设为 0），写出错误价值估计 $V^{\text{err}}(s_t)$ 随时间步 $t$ 变化的闭式解析表达式；
3. 计算在 $t=95$ 步时，由截断错误引起的相对价值估计误差 $\frac{V^*(s_{95}) - V^{\text{err}}(s_{95})}{V^*(s_{95})}$。

::: details 思考与解析
1. 真实系统无限延续，单步奖励为 1：
   $$V^*(s_t) = \sum_{k=0}^{\infty} \gamma^k \cdot 1 = \frac{1}{1 - \gamma} = \frac{1}{1 - 0.9} = 10.0$$
2. 若在 $t=100$ 处被强制截断为 0，则在 $t$ 时刻只累计到第 99 步的收益（共剩余 $100 - t$ 步）：
   $$V^{\text{err}}(s_t) = \sum_{k=0}^{100 - t - 1} \gamma^k \cdot 1 = \frac{1 - \gamma^{100 - t}}{1 - \gamma} = 10 \cdot (1 - 0.9^{100 - t})$$
3. 在 $t=95$ 步时：
   $$V^{\text{err}}(s_{95}) = 10 \cdot (1 - 0.9^5) = 10 \cdot (1 - 0.59049) = 4.0951$$
   相对价值误差为：
   $$\text{Relative Error} = \frac{10.0 - 4.0951}{10.0} = 59.05\%$$
   可见，仅仅在距离截断前 5 步，错误地将 Truncation 当作 Termination 就会产生近 **60% 的巨大价值塌陷**，直接摧毁策略学习。
:::

### 习题 4：硬件时延增广状态的最小维数计算

某四足机器人运动控制系统，控制主频为 $50\text{ Hz}$（$\Delta t = 20\text{ ms}$）。已知各硬件环节的测量延迟如下：
- 双目相机成像与 USB 驱动传输延迟 $\delta_{\text{cam}} = 28\text{ ms}$；
- 视觉主干网络与世界模型推演延迟 $\delta_{\text{infer}} = 35\text{ ms}$；
- CAN 总线通信与底层关节驱动响应延迟 $\delta_{\text{act}} = 12\text{ ms}$。
1. 计算从快门触发到动作生效的总延迟 $\delta_{\text{total}}$；
2. 为使系统状态重新满足严格的离散马尔可夫性质，状态增广队列中至少需要包含多少个“飞行中动作”（In-flight Actions）？
3. 写出增广状态向量 $\tilde{\mathbf{s}}_t$ 的精确数学表达式。

::: details 思考与解析
1. 链路总时延：
   $$\delta_{\text{total}} = 28 + 35 + 12 = 75\text{ ms}$$
2. 控制周期 $\Delta t = 20\text{ ms}$。跨越的控制周期数为：
   $$K = \left\lceil \frac{\delta_{\text{total}}}{\Delta t} \right\rceil = \left\lceil \frac{75}{20} \right\rceil = \lceil 3.75 \rceil = 4$$
   因此，当策略在时刻 $t$ 基于 $t - 28\text{ ms}$ 的历史图像做出决策时，系统在物理世界上已经发出了 4 个处于排队/生效途中的历史动作。
3. 增广状态向量必须包含当前观测与前 4 个历史动作：
   $$\tilde{\mathbf{s}}_t = \big(o_t, a_{t-1}, a_{t-2}, a_{t-3}, a_{t-4}\big)$$
:::

### 习题 5：变长轨迹批处理与掩码损失归一化分析

在训练带有布尔有效掩码 $\mathcal{M} \in \{0, 1\}^{B \times H}$ 的变长轨迹世界模型时，考虑以下两种批次损失归一化方案：
- 方案 A（掩码有效和归一化）：$\mathcal{L}_A = \frac{\sum_{i=1}^B \sum_{t=1}^H \mathcal{M}_{i,t} \cdot \ell_{i,t}}{\sum_{i=1}^B \sum_{t=1}^H \mathcal{M}_{i,t}}$
- 方案 B（张量总容量归一化）：$\mathcal{L}_B = \frac{\sum_{i=1}^B \sum_{t=1}^H \mathcal{M}_{i,t} \cdot \ell_{i,t}}{B \times H}$

分析：当数据集由 10% 的长轨迹（$L=100$）与 90% 的极短失败轨迹（$L=5$）混合构成时，采用方案 B 会对模型梯度带来怎样的有害影响？

::: details 思考与解析
在长短轨迹极度不平衡的批次中，$B \times H$ 是恒定的理论最大容量，而实际有效样本步数 $\sum \mathcal{M} \ll B \times H$。
若采用方案 B：
1. **梯度尺度受有效率剧烈抖动**：当某一个 Batch 碰巧抽到较多长轨迹时，有效率高，梯度模长正常；当下一个 Batch 抽到大量短轨迹时，有效率极低，导致反向传播的梯度模长被分母 $B \times H$ 人为缩减了数十倍。这等价于给优化器施加了高方差的梯度噪声。
2. **短轨迹样本的梯度被过度惩罚**：短轨迹因为有效步数少，在批次内部的贡献被分母过度稀释，导致世界模型无法有效学习短轨迹中的关键失败/碰撞动力学。
因此，必须始终采用方案 A，确保每个有效物理转移对梯度的贡献权重恒定。
:::

---

## 8. 本节总结与下节预告

### 核心要点回顾

```text
               2.1 张量、时间与轨迹 知识图谱
               
    ┌───────────────────────┴───────────────────────┐
    ▼                                               ▼
[张量时空布局]                                   [时序因果与拓扑]
• 严格遵循维度语义契约                          • 差一律: T+1 观测对齐 T 动作
• 连续内存与 Strides 步长机制                   • 因果 DAG: 防范前向与后向错配
• 转置后调用 .contiguous()                      • Termination vs Truncation
                                                • 隔离跨 Episode 虫洞陷阱
    ▲                                               ▲
    └───────────────────────┬───────────────────────┘
                            ▼
                    [物理时间与工程批处理]
                    • 控制周期 Δt 与数值积分
                    • 硬件时延链路与动作队列增广
                    • Sliding Chunking vs Masked Padding
```

1. **维度语义契约是第一道防线**：张量形状必须与物理语义严格对应，明确区分 PyTorch `[B, T, C, H, W]` 与外部库 `[B, T, H, W, C]` 的内存步长差异；
2. **差一律（Off-by-One Law）不可违背**：$T$ 个动作步骤必然对应 $T+1$ 帧观测；任何时序对齐错位都会导致因果反转或虚假时延；
3. **区分终止与截断**：自然终止（Termination）进入吸收态，价值置 0；时间截断（Truncation）必须保留价值自举，杜绝虫洞跨轨瞬移；
4. **时间戳与时延建模**：离散动力学依赖 $\Delta t$，在物理实机系统中必须通过动作队列增广状态抵消相位滞后。

### 下节预告：2.2 图像编码器：CNN 与 ViT

在建立了规整的时空张量流之后，世界模型面临的下一个核心挑战是：**如何将高维的高清视觉像素张量 $[B, T, C, H, W]$ 压缩为紧凑、富含物理几何语义的低维特征向量？**

在下一篇 [2.2 图像编码器：CNN 与 ViT](/chapters/02-foundations/02-cnn-and-vit) 中，我们将深入剖析卷积神经网络（CNN）的平移等变性归纳偏置，对比 Vision Transformer（ViT）的全局 Patch 自注意力机制，推导感受野与空间下采样对动力学建模的深远影响。

---

## 9. 参考文献与推荐阅读

1. **Gymnasium Documentation: Handling Time Limits (Truncation vs Termination)**  
   [Farama Foundation, 2023]  
   [https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)  
   *详述了现代强化学习环境中将 Done 细分为 Termination 与 Truncation 的数学考量与标准实现。*

2. **Reinforcement Learning: An Introduction (2nd Edition)**  
   *Richard S. Sutton and Andrew G. Barto, MIT Press, 2018.*  
   [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)  
   *第 3 章“有限马尔可夫决策过程”中关于分幕式任务（Episodic Tasks）、吸收态与持续性任务转移拓扑的奠基论述。*

3. **Mastering Diverse Domains through World Models (DreamerV3)**  
   *Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. arXiv:2301.04104, 2023.*  
   [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)  
   *展示了在多模态字典输入、长程定长切片抽取（Chunking）与掩码损失训练下的世界模型顶级实践。*

4. **PyTorch Internals: Tensor Layouts and Strides**  
   *Edward Z. Yang, 2019.*  
   [http://blog.ezyang.com/2019/05/pytorch-internals/](http://blog.ezyang.com/2019/05/pytorch-internals/)  
   *深入剖析 PyTorch C++ 底层 TensorImpl、Strides 内存视图映射与 Contiguity 检查机制。*

5. **Robomimic: A Modular Framework for Robot Learning from Demonstration**  
   *Ajay Mandlekar et al., Autonomous Robots, 2021.*  
   [https://robomimic.github.io/](https://robomimic.github.io/)  
   *具身智能领域多模态轨迹存储、观察动作对齐与异步硬件延迟处理的标准开源实现规范。*
