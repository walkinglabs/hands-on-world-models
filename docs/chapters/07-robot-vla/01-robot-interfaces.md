# 7.1 机器人学习接口

在前面的章节中，我们看到了世界模型在虚拟环境和模拟器中的各种应用。无论是网格世界还是赛车游戏，计算机都可以在内部建立一个虚拟世界，并在想象中规划出下一步该怎么走。

现在，我们要把目光投向物理世界。我们希望让计算机走出虚拟的模拟环境，真正操控一台机械臂去完成现实生活中的任务，比如拿起桌上的杯子、收拾散落的积木。这类结合了物理身体与环境交互的技术，通常被称为**具身智能**（Embodied AI）。

说到让计算机理解世界，我们可能会想到视觉语言模型（VLM）。现在的模型不仅能看懂图片，还能用流利的人类语言描述“桌上放着一只红色杯子”。但是，对机器人来说，光会“看”和“说”是远远不够的。机器人必须伸出手臂、张开夹爪、调整姿态，最终把杯子平稳地拿起来。换句话说，计算机不仅要理解环境，还必须输出可以直接执行的**物理动作**。

那么，如何让计算机学会操作机械臂呢？要让计算机控制机械臂，首先必须把物理世界中发生的事情转化为计算机能够处理的数据。本节我们将系统梳理机器人学习的数据接口、常见的示教数据采集方式、多传感器时间同步算法、数据清洗与归一化实战，以及工业界主流的数据集格式规范。

---

## 机器人数据的构成

在图像识别中，输入是一张图片，输出是一个类别标签（比如“猫”或“狗”）。而在机器人操作中，数据要丰富得多。

想象一台桌面机械臂正在执行“抓取红杯”的任务。在这一过程中，我们需要记录下机械臂看到了什么、自身处于什么姿势、听到了什么指令，以及在每一瞬间执行了什么动作。

把这一完整过程从头到尾记录下来，就得到了一条数据。在机器人学习中，这样一条包含完整操作过程的数据记录通常被称为一条 **Episode**（演示轨迹）。

一条典型的 Episode 主要包含以下 4 类信息：

- **图像（Images）**：安装在桌面上方的全局相机或机械臂手腕上的微型相机拍摄的连续画面。
- **本体状态（Proprioception）**：机器人自身的状态，如各个关节当前的角度、末端夹爪在空间中的三维坐标以及夹爪当前的开合度。
- **任务指令（Instruction）**：人类给出的自然语言文本，比如 `"拿起红色的杯子"`。
- **动作（Actions）**：在这一时刻发送给电机的控制指令，比如各个关节需要转动的角度增量。

下面，我们通过图 7-1 来直观地看一下一条 Episode 中各个数据的时间对应关系。

![图 7-1 一条 Episode 中观察与动作的时序对应关系](/figures/7-1-episode-timeline.svg)

从图 7-1 可以清楚地看到：在一个包含 $T$ 步动作的 Episode 中，**观察数据有 $T+1$ 帧，而动作数据只有 $T$ 个**。

为什么会多出一帧观察呢？这是因为在最开始的 $t=0$ 时刻，机械臂尚未执行任何动作，但相机已经拍摄到了初始画面 $o_0$；当机械臂执行完最后一步动作 $a_{T-1}$ 之后，环境会演化到最终状态，相机记录下终点画面 $o_T$。

因此，**动作 $a_t$ 永远夹在当前观察 $o_t$ 与下一时刻观察 $o_{t+1}$ 之间**。

如果在组织数据时不小心把动作平移错位，让模型误把后一帧画面当成前一帧的输入，离线训练时看似准确率很高，但放到真机上机器人就会因为无法感知动作引起的因果变化而失控。

---

## 动作空间的表示

搞清楚了数据的时序关系后，接下来我们需要确定动作（Action）的具体物理含义。也就是说，神经网络在每一步输出的一组浮点数数组，到底控制电机的什么物理量？

为了把这个问题看清楚，我们来看一个具体的真实操作场景：

假设一台桌面 6 自由度机械臂（6 个旋转关节电机 + 1 个末端开合夹爪）当前正停在桌面上方，末端夹爪位于坐标 $(x=0.30, y=0.00, z=0.20)$ 米处。

现在人类发出了一个简单指令：**“将机械臂向前平移 5 厘米（沿 X 轴正方向 $+0.05\ \text{m}$），姿态保持不变，夹爪保持闭合捏紧。”**

面对同一个物理动作，不同动作空间的表示方式和实际数值截然不同：

### 1. 笛卡尔末端空间表示（Cartesian Space）

在末端空间中，网络直接输出机械臂末端夹爪在三维笛卡尔坐标系中的位姿增量和夹爪状态：

$$\Delta x = [\Delta x, \Delta y, \Delta z, \Delta\text{roll}, \Delta\text{pitch}, \Delta\text{yaw}, g] \in \mathbb{R}^7$$

在 Python 中打印出来的实际数组如下：

```python
import numpy as np

# 笛卡尔末端动作向量（单位：米 m，弧度 rad，夹爪 -1~1）
action_cartesian = np.array(
    [+0.05, 0.00, 0.00, 0.00, 0.00, 0.00, -1.0], dtype=np.float32
)
```

这个向量的含义对人类和神经网络来说一目了然：
- 平移分量 $(\Delta x, \Delta y, \Delta z) = [+0.05, 0.00, 0.00]$：仅沿 $X$ 轴向前走 $5\ \text{cm}$，$Y$ 和 $Z$ 轴不动；
- 旋转分量 $(\Delta\text{roll}, \Delta\text{pitch}, \Delta\text{yaw}) = [0.00, 0.00, 0.00]$：三个旋转角均不改变；
- 夹爪分量 $g = -1.0$：保持夹爪闭合（通常 $+1.0$ 表示完全张开，$-1.0$ 表示完全捏合）。

**为什么视觉大模型偏爱末端空间？**  
因为相机画面中的三维几何与末端笛卡尔坐标系高度契合。神经网络从图像中看到杯子在正前方 5 厘米处，直接输出 $\Delta x = +0.05$ 是非常自然、容易学习的。更重要的是，无论是 Google 机械臂、Franka 机械臂还是小型的廉价桌面臂，**“向前移动 5 厘米”的空间语义完全通用**，这使得跨不同机器人本体预训练模型成为可能（如 OpenVLA、RT-2）。

**工程代价与奇异点风险：**  
然而，机械臂底层的电机并不知道什么是“三维坐标系”，电机只听得懂“转动多少角度”。控制器必须通过**逆运动学**（Inverse Kinematics, 简称 IK），利用机械臂的雅可比矩阵 $J(q)$ 实时把末端位移翻译成各个关节的旋转量：

$$\Delta q \approx J^\dagger(q) \cdot \Delta x$$

当机械臂伸展得太直（例如大臂与小臂夹角接近 $180^\circ$）时，雅可比矩阵会发生数学上的**奇异点**（Singularity）。在奇异点附近，哪怕末端只向前挪动 $1\ \text{mm}$（$\Delta x = 0.001$），逆运动学求出的关节角速度可能瞬间飙升到 $\Delta\theta_3 > 15\ \text{rad/s}$（远超物理电机承受极限），导致机械臂剧烈抽搐甚至触发底层驱动器的紧急停机保护（E-Stop）。

---

### 2. 关节空间表示（Joint Space）

在关节空间中，网络直接输出机械臂上每一个电机需要旋转的角度增量：

$$\Delta q = [\Delta\theta_1, \Delta\theta_2, \Delta\theta_3, \Delta\theta_4, \Delta\theta_5, \Delta\theta_6, g] \in \mathbb{R}^7$$

同样是完成“向前平移 5 厘米”这个动作，在 Python 中打印出来的关节动作数组可能是这样的：

```python
# 关节角增量动作向量（单位：弧度 rad，夹爪 -1~1）
action_joint = np.array(
    [-0.012, +0.038, -0.055, +0.021, -0.043, +0.008, -1.0], dtype=np.float32
)
```

仔细看这组数字：**为了让末端在空间中走出一条纯粹的直线，机械臂的 6 个关节电机必须全部以不同的角度和方向协同转动！** 底座微转、肩部抬起、肘部下压、手腕倾斜补偿。

**为什么精细操作与硬件工程师更偏爱关节空间？**  
因为关节角度增量可以直接打包发送给底层的电机驱动芯片（如 CANopen 或 EtherCAT 总线）。由于完全绕过了逆运动学求解，**它绝对不会遇到奇异点爆炸的问题，电机的运动极其平稳、可控**。在 Stanford 著名的双臂遥操作系统 ACT（Zhao et al., 2023）与 Mobile ALOHA 中，正是因为低成本舵机的结构形变较大、逆运动学解算误差大，作者全面采用了 14 维关节位置直接控制。

**代价与难点：**  
对于神经网络而言，想要学会这 6 个看似杂乱无章的关节弧度数值，必须在隐层中硬生生拟合出整套复杂的三角函数运动学方程。而且，一旦换了一台连杆长度稍有不同的机械臂，先前学到的关节控制规律就会全部报废，无法直接泛化。

---

### 实际研究中的选型总结

| 动作空间类型 | 典型代表模型 | 典型应用场景 | 核心权衡结论 |
| :--- | :--- | :--- | :--- |
| **笛卡尔末端位姿增量** | OpenVLA, RT-1, RT-2, Octo | 跨本体通用操作、大模型多任务预训练 | 视觉语义直观、便于跨本体迁移；但依赖逆运动学，需在代码中设置奇异点速度截断保护 |
| **关节位置 / 角度增量** | ACT, Mobile ALOHA, Diffusion Policy (部分真机) | 双臂精细装配、低成本硬件桌面操作 | 硬件执行绝对稳定、无奇异点风险；但跨机器人本体泛化难度极高 |

---

## 经典控制栈：为什么还要学习

在学习方法上场之前，工业机器人已经能用三件套干活：**正向运动学**（FK）由关节角算出末端位姿 $p=\mathrm{FK}(q)$；**逆运动学**（IK）反过来求 $q$；**PD 控制**用 $\tau=K_p(q_d-q)+K_d(\dot q_d-\dot q)$ 把关节跟踪到 $q_d$。Stanford CS 123 的前几周 lab、HF Robotics Course Unit 2，教的就是这一层。

这一层必须会读，否则后面的动作空间选择没有物理着落。它不够用，是因为三件事它解不了：

1. **感知**：FK/IK 假设你已经知道杯子在哪。从像素到 $p^*$ 不是解析几何。
2. **接触**：PD 把位置误差乘上很大的 $K_p$，插接和捏纸杯会直接超出力限。见 [7.5](/chapters/07-robot-vla/05-manipulation-and-touch)。
3. **语言与长时程**：没有「拿蓝杯再放到架子上」这种符号接口。

所以学习不是替代电机驱动，而是替代「人手写状态估计 + 人手写技能切换」。数据采集仍然常常靠**遥操作**：人动领导臂或手柄，从臂复现，得到 $(o_t, a_t)$ 示范。LeRobot 的 SO-101、ALOHA 都是这条流水线的硬件版。没有示范、没有仿真，后面的 VLA 没有监督。

---

## 怎么做数据？常见数据采集方案

了解了数据接口后，下一个最核心的实践问题是：**如果我们要训练一台属于自己的机械臂，这些人类示教数据究竟是怎么采集出来的？**

目前在学术界与工业界，主流的人类示教数据采集方案主要有以下三种：

```
                    ┌─────────────────────────────────────────┐
                    │       机器人示教数据采集主流方案           │
                    └────────────────────┬────────────────────┘
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
  主从同构臂遥操作 (Puppet Arm)      空间手柄与 VR 遥操作 (6-DoF)     人类介入协助式示教 (Intervention)
  • ALOHA, GELLO, Koch 硬件        • Quest 3 / SpaceMouse          • 策略自主运行 + 人工脚踏板纠错
  • 零力矩拖动，硬件 1:1 映射       • 适合大范围笛卡尔末端控制        • 专门采集长尾失败与纠偏数据
  • 精细装配首选，手感真实         • 需逆运动学解算，存在手眼延迟   • 解决行为克隆分布偏移的利器
```

### 1. 主从同构臂遥操作（Leader-Follower / Puppet Arm）
这是目前桌面双臂精细操作（如 Stanford ALOHA、GELLO、Koch Arm）最常用的方案：
- **原理**：搭建一台与从动机械臂（Follower/Puppet）自由度与连杆尺寸完全一致的“主臂”（Leader/Master）。人类操作者直接用手捏住主臂末端拖动，主臂关节上的编码器实时将转动的弧度值通过 USB/CAN 发送给从臂；
- **优点**：操作手感极其直观，物理 1:1 映射，关节空间数据天然对齐，能够完成穿针引线、剥香蕉皮等毫米级超精细操作；
- **成本与开源方案**：早期工业级主从臂动辄数十万元，而如今基于 Dynamixel 舵机或 3D 打印的 Koch/GELLO 开源主臂成本已降至数千元人民币，极大降低了具身智能的数据门槛。

### 2. 空间手柄与 VR 遥操作（VR / 6-DoF Spatial Controller）
在需要大范围移动或末端控制的场景中，VR 头显（如 Meta Quest 3、Apple Vision Pro）与 3Dconnexion 空间鼠标成为主流：
- **原理**：通过头显的视觉定位或手柄空间传感器，实时追踪人类手掌在三维空间中的绝对坐标 $(x, y, z, \text{roll}, \text{pitch}, \text{yaw})$，再将其转换为末端增量发送给机械臂底层 IK 解算器；
- **优点**：无需制造昂贵的同构机械臂，支持跨不同型号机械臂的统一遥操作；
- **局限**：缺乏物理力反馈阻尼，人类在空气中挥舞手柄时容易产生高频手抖，且手眼标定（Hand-Eye Calibration）误差和 Wi-Fi 传输延迟会导致操作精度下降。

### 3. 人类介入协助式示教（Human-in-the-Loop Intervention）
这是针对行为克隆“分布偏移”问题的高级采集策略（类似 DAgger 思想）：
- **原理**：让初步训练好的神经网络策略自主控制机械臂运行。当机械臂即将偏离轨迹或发生碰撞时，旁边的人类工程师踩下脚踏板或拨动手柄切入控制权，手动将手臂纠正回正常轨道，随后松开踏板让模型继续运行；
- **核心价值**：**专门收集“从错误偏离状态恢复到正确状态”的纠偏数据**。这类数据是常规成功示教中极度匮乏的，能有效提升策略在面对突发扰动时的鲁棒性。

---

## 端到端实战：编写多线程数据录制器

在真实采集时，相机捕获画面（20~30ms）、电机读取角度（2ms）与遥操作手柄（10ms）处于不同的线程中。如果用单线程串行阻塞读取，相机的读帧耗时会严重拖慢控制循环。

工业级数据采集系统必须采用**多线程异步生产者-消费者架构**。下面我们编写一个可以直接运行、结构清晰的多线程机器人数据录制器：

```python
import threading
import time
import h5py
import numpy as np


class RobotDataRecorder:
    """多线程异步机器人示教数据录制器（支持按键触发与因果对齐）"""

    def __init__(self, fps=50):
        self.fps = fps
        self.dt = 1.0 / fps
        self.is_recording = False
        self.lock = threading.Lock()

        # 硬件最新缓存 (由各传感器独立采集线程实时更新)
        self.latest_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.latest_proprio = np.zeros(7, dtype=np.float32)
        self.latest_action = np.zeros(7, dtype=np.float32)

        # 当前录制轨迹数据缓冲区
        self.buffer_images = []
        self.buffer_proprio = []
        self.buffer_actions = []
        self.buffer_timestamps = []

    def update_camera_frame(self, frame: np.ndarray):
        """相机采集线程回调接口"""
        with self.lock:
            self.latest_image = frame.copy()

    def update_robot_state(
        self, joint_angles: np.ndarray, cmd_action: np.ndarray
    ):
        """电机/手柄高频控制线程回调接口"""
        with self.lock:
            self.latest_proprio = joint_angles.copy()
            self.latest_action = cmd_action.copy()

    def start_episode(self):
        """开始录制一条新轨迹"""
        self.buffer_images.clear()
        self.buffer_proprio.clear()
        self.buffer_actions.clear()
        self.buffer_timestamps.clear()

        # 记录 t=0 初始观察 o_0
        with self.lock:
            self.buffer_images.append(self.latest_image.copy())
            self.buffer_proprio.append(self.latest_proprio.copy())
            self.buffer_timestamps.append(time.time())

        self.is_recording = True
        print("🔴 开始录制 Episode...")

    def record_step(self):
        """主控制周期触发点 (50Hz 定时器调用)"""
        if not self.is_recording:
            return

        with self.lock:
            # 记录执行动作 a_t
            self.buffer_actions.append(self.latest_action.copy())
            # 记录执行后观察 o_{t+1}
            self.buffer_images.append(self.latest_image.copy())
            self.buffer_proprio.append(self.latest_proprio.copy())
            self.buffer_timestamps.append(time.time())

    def stop_and_save(self, save_path: str, instruction: str = "拿起红色杯子"):
        """停止录制并持久化保存为标准 HDF5 格式"""
        self.is_recording = False
        num_obs = len(self.buffer_images)
        num_acts = len(self.buffer_actions)

        # 严格验证因果时序一致性
        assert (
            num_obs == num_acts + 1
        ), f"时序错位！观察帧数 {num_obs} 不等于 动作步数+1 ({num_acts + 1})"

        with h5py.File(save_path, "w") as f:
            f.create_dataset(
                "observations/images",
                data=np.array(self.buffer_images),
                compression="gzip",
            )
            f.create_dataset(
                "observations/proprio", data=np.array(self.buffer_proprio)
            )
            f.create_dataset("actions", data=np.array(self.buffer_actions))
            f.create_dataset("timestamps", data=np.array(self.buffer_timestamps))
            f.attrs["instruction"] = instruction
            f.attrs["fps"] = self.fps

        print(
            f"💾 成功保存轨迹至 {save_path}！(包含 {num_obs} 帧观察, {num_acts} 步动作)"
        )
```

---

## 多传感器的时间对齐

在确定了动作空间后，另一个关键的工程问题是：**物理世界中的各个硬件传感器并不是按同一个时钟节奏采样的**。

为了看清异步数据是如何在时间轴上发生错位的，我们来看一组真实桌面遥操作系统在采集数据时的原始时间戳日志：

- **外部 USB 相机（30 Hz）**：理论上每 $33.3\ \text{ms}$ 产生一帧，但受曝光与 USB 传输影响，帧到达电脑的实际时间戳可能为：`[0ms, 35ms, 68ms, 104ms, ...]`；
- **机械臂电机编码器（500 Hz）**：通过高速 CAN 总线每 $2\ \text{ms}$ 刷新一次关节角度，时间戳为：`[0ms, 2ms, 4ms, 6ms, 8ms, ...]`；
- **遥操作手柄（100 Hz）**：每 $10\ \text{ms}$ 采集一次人类指令动作，时间戳为：`[0ms, 10ms, 20ms, 30ms, ...]`。

如果直接按列表索引把第 $k$ 张图片、第 $k$ 个关节角度和第 $k$ 个动作强行拼在一起，相机刚走到第 3 帧（对应时间约 $100\ \text{ms}$），电机已经读到了第 50 帧（对应时间同样是 $100\ \text{ms}$），但两者的数组下标整整差了 17 倍！时间轴会彻底崩塌。

---

### 因果零阶保持（Zero-Order Hold）对齐实例

解决异步多速率采样的工业标准方案是：**设定统一的控制主频（如 50 Hz，即每隔 $\Delta t = 20\ \text{ms}$ 一个控制步），并使用因果零阶保持（Zero-Order Hold, ZOH）对齐各个传感器**。

所谓因果零阶保持，规则极其纯粹：**在统一时间网格点 $t_k$ 处，严格只读取在 $t_k$ 之前刚刚到达系统的最新数据，绝不向后透支未来。**

我们通过表 7-1 来查看前 $100\ \text{ms}$ 内各个传感器实际对齐后的数值映射：

| 统一控制时间网格 $t_k$ | 机械臂关节角（取最新 2ms 读数） | 外部相机画面（取已到达的最新帧） | 人类控制动作（取当前 10ms 读数） | 状态说明 |
| :--- | :--- | :--- | :--- | :--- |
| **$t_0 = 0\ \text{ms}$** | `q(0ms) = [0.00, 0.00, ...]` | **图像帧 0**（到达时间 0ms） | `a(0ms) = [+0.05, 0.0, ...]` | 初始静止状态 |
| **$t_1 = 20\ \text{ms}$** | `q(20ms) = [0.01, 0.02, ...]` | **图像帧 0**（帧 1 尚未到达，保持帧 0） | `a(20ms) = [+0.05, 0.0, ...]` | 图像保持，关节更新 |
| **$t_2 = 40\ \text{ms}$** | `q(40ms) = [0.02, 0.05, ...]` | **图像帧 1**（帧 1 于 35ms 到达） | `a(40ms) = [+0.05, 0.0, ...]` | 图像刷新至帧 1 |
| **$t_3 = 60\ \text{ms}$** | `q(60ms) = [0.03, 0.07, ...]` | **图像帧 1**（帧 2 于 68ms 到达，当前仍为帧 1）| `a(60ms) = [+0.05, 0.0, ...]` | 图像保持，关节更新 |
| **$t_4 = 80\ \text{ms}$** | `q(80ms) = [0.04, 0.09, ...]` | **图像帧 2**（帧 2 于 68ms 到达） | `a(80ms) = [+0.05, 0.0, ...]` | 图像刷新至帧 2 |
| **$t_5 = 100\ \text{ms}$** | `q(100ms) = [0.05, 0.12, ...]` | **图像帧 2**（帧 3 尚未到达） | `a(100ms) = [+0.05, 0.0, ...]` | 图像保持，关节更新 |

<div align="center">表 7-1 异步多传感器在 50 Hz（20ms）统一时钟下的因果零阶保持对齐</div>

从表 7-1 可以清楚地观察到：
1. **画面重复是正常且真实的物理现象**：在 30 Hz 相机与 50 Hz 控制器的组合下，相机画面在某些控制步中会保持上一帧。这是因为物理相机曝光与传输需要时间，控制器必须面对这种“视觉延迟”；
2. **绝对杜绝未来插值**：如果有人在离线处理时，试图用 $35\ \text{ms}$ 的帧 1 和 $68\ \text{ms}$ 的帧 2 线性插值生成 $60\ \text{ms}$ 的“伪画面”，神经网络就会偷偷学会在 $60\ \text{ms}$ 时利用尚未发生的未来物理信息，导致实机闭环部署时彻底失效。

---

### Python 实现：多传感器因果时间同步器

下面我们编写一个可以直接处理带时间戳的原始异步数据流的对齐函数：

```python
def synchronize_multimodal_streams(
    raw_images,
    cam_timestamps,
    raw_proprios,
    proprio_timestamps,
    raw_actions,
    action_timestamps,
    control_freq=50.0,  # 目标控制频率 50 Hz
):
    """使用因果零阶保持（Causal Zero-Order Hold）将异步多传感器数据对齐到统一时间网格"""
    dt = 1.0 / control_freq
    start_time = max(
        cam_timestamps[0], proprio_timestamps[0], action_timestamps[0]
    )
    end_time = min(
        cam_timestamps[-1], proprio_timestamps[-1], action_timestamps[-1]
    )

    # 生成均匀的统一控制时间点: t_0, t_1, t_2, ...
    grid_times = np.arange(start_time, end_time, dt)

    synced_images = []
    synced_proprio = []
    synced_actions = []

    # 1. 对齐 T+1 帧观察数据
    for t in grid_times:
        # np.searchsorted(..., side='right') - 1 确保严格只取在 t 之前或刚好等于 t 的最新样本
        cam_idx = np.searchsorted(cam_timestamps, t, side="right") - 1
        proprio_idx = (
            np.searchsorted(proprio_timestamps, t, side="right") - 1
        )

        synced_images.append(raw_images[max(0, cam_idx)])
        synced_proprio.append(raw_proprios[max(0, proprio_idx)])

    # 2. 对齐 T 步控制动作
    for t in grid_times[:-1]:
        act_idx = np.searchsorted(action_timestamps, t, side="right") - 1
        synced_actions.append(raw_actions[max(0, act_idx)])

    return {
        "images": np.array(synced_images),  # 形状: [T + 1, H, W, C]
        "proprio": np.array(synced_proprio),  # 形状: [T + 1, P]
        "actions": np.array(synced_actions),  # 形状: [T, A]
        "timestamps": grid_times,  # 形状: [T + 1]
    }
```

---

## 数据清洗、标注与质检细节

采集录制好原始示教数据后，**绝对不能直接扔进神经网络训练**。人类在遥操作时会不可避免地引入停顿、手抖、多余试探甚至操作失误。如果未经清洗直接训练，策略会把这些坏习惯全部学去。

```
原始录制数据 ──► [ 1. 静止发呆段截断 ] ──► [ 2. 手抖平滑滤波 ] ──► [ 3. 全局统计量归一化 ] ──► 送入训练
                   剔除开始与结束的静止帧       消除高频电机震颤          防止梯度爆炸 / 统一量纲
```

### 1. 静止发呆段自动截断（Trimming Idle Steps）
示教者在听到“开始录制”到双手真正握住主臂开始移动之间，通常有 $1\sim 2$ 秒的反应延迟；任务完成后到按下按键停止录制之间也有延迟。这会导致轨迹两端充满大量“观察画面不变、动作几乎为 0”的静止帧。
- **危害**：如果数据集中有大量静止帧，策略在遇到未知状态时极易学会在原地“发呆保持不动”的懒惰局部最优；
- **处理方式**：计算机械臂末端或关节速度标量 $v_t = \|\Delta q_t\|_2$。若 $v_t$ 小于静止阈值 $\epsilon_{\text{idle}}$，自动裁剪掉两端的静态片段。

```python
def trim_idle_steps(actions, observations, threshold=0.005):
    """自动裁剪轨迹首尾由于人类反应延迟引起的静止发呆帧"""
    # 计算每一步的动作绝对幅度
    action_magnitudes = np.linalg.norm(actions, axis=-1)
    moving_indices = np.where(action_magnitudes > threshold)[0]

    if len(moving_indices) == 0:
        return None  # 全程静止的废轨迹

    start_idx = moving_indices[0]
    end_idx = moving_indices[-1] + 1

    # 观察对应保留 [start_idx : end_idx + 1]
    trimmed_actions = actions[start_idx:end_idx]
    trimmed_obs = observations[start_idx : end_idx + 1]
    return trimmed_obs, trimmed_actions
```

### 2. 全局动作归一化统计量（Normalization）
神经网络对输入输出的数值范围极其敏感。关节角在 $[-3.14, +3.14]$ 之间，夹爪在 $[0, 1]$ 之间，末端位移在 $[-0.05, +0.05]$ 之间。如果不做归一化，大范围的关节角会彻底主导损失函数。

> **严禁踩坑：切勿使用局部单条轨迹归一化**  
> 归一化统计量必须在**全训练集的所有 Episode 上计算全局统计量**（全局均值/方差或分位数）。如果对单条轨迹做局部归一化，会将微小的抖动强行放大为最大位移，完全破坏物理速度的真实量纲。

```python
def compute_dataset_action_stats(all_actions_list):
    """计算全数据集的全局动作归一化参数 (支持分位数裁剪，防止离群异常值)"""
    stacked_actions = np.concatenate(all_actions_list, axis=0)  # [N_total, A]

    # 采用 1% 与 99% 分位数代替简单 min/max，防止个别离群噪点破坏尺度
    q01 = np.percentile(stacked_actions, 1, axis=0)
    q99 = np.percentile(stacked_actions, 99, axis=0)

    stats = {
        "mean": np.mean(stacked_actions, axis=0),
        "std": np.std(stacked_actions, axis=0) + 1e-6,
        "q01": q01,
        "q99": q99,
    }
    return stats


def normalize_action(action, stats, mode="quantile"):
    """将动作归一化至 [-1, 1] 区间供模型训练"""
    if mode == "quantile":
        # 缩放到 [-1, 1] 并截断超界值
        norm_a = 2.0 * (action - stats["q01"]) / (stats["q99"] - stats["q01"] + 1e-6) - 1.0
        return np.clip(norm_a, -1.0, 1.0)
    elif mode == "gaussian":
        return (action - stats["mean"]) / stats["std"]
```

---

## 真实世界的机器人数据格式与接口

在实际科研与工程落地中，我们一般不会自己从零发明数据存储格式，而是基于社区成熟的机器人数据集标准进行开发。

目前具身智能与机器人学习领域形成了三大主流的数据存储格式与接口规范：

1. **HuggingFace LeRobot 格式**（现代 PyTorch 与高效视频压缩生态）
2. **Open X-Embodiment / RLDS 格式**（Google 主导的多机器人跨本体通用格式，OpenVLA / Octo 原生采用）
3. **Robomimic / ACT HDF5 格式**（Stanford ACT 与 Diffusion Policy 桌面精细操作标杆）

下面我们分别拆解这三种真实世界接口的底层数据结构与 Python 读取方式。

---

### 1. HuggingFace LeRobot 数据接口

LeRobot 是 HuggingFace 专为机器人学习打造的开源库。它的核心设计理念是**极致轻量与高吞吐**：视频帧采用硬件级 MP4（H.264/AV1）深度压缩以节省磁盘空间，动作与关节数值采用 Safetensors / Arrow 存储，原生无缝对接 PyTorch `DataLoader`。

一条典型的 LeRobot 样本字典在送入神经网络时的张量结构如下：

```python
import torch

# LeRobot 数据加载器吐出的单个 Batch 字典结构
batch = {
    # 外部第三人称俯视视角图像，已归一化至 [0, 1]
    "observation.images.laptop": torch.Tensor,  # 形状: [B, 3, 480, 640], dtype=torch.float32
    # 机械臂手腕上的第一人称视角微型相机
    "observation.images.phone": torch.Tensor,  # 形状: [B, 3, 480, 640], dtype=torch.float32
    # 机械臂自身当前状态 (6 关节角度 + 1 夹爪开合度)
    "observation.state": torch.Tensor,  # 形状: [B, 7], dtype=torch.float32
    # 控制动作 (未来动作，若使用动作分块则为 [B, K, A])
    "action": torch.Tensor,  # 形状: [B, 7] 或 [B, K, 7], dtype=torch.float32
    # 轨迹与时间元数据
    "timestamp": torch.Tensor,  # 形状: [B], dtype=torch.float32 (秒)
    "episode_index": torch.Tensor,  # 形状: [B], dtype=torch.int64
    "frame_index": torch.Tensor,  # 形状: [B], dtype=torch.int64
    "next.done": torch.Tensor,  # 形状: [B], dtype=torch.bool
}
```

在 Python 中加载并查看 LeRobot 开源数据集非常简洁：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 加载 HuggingFace 社区开源的真实机械臂示教数据集（如 Koch 双臂抓取）
dataset = LeRobotDataset("lerobot/koch_pick_place_lego")

print(f"轨迹总数: {dataset.num_episodes}, 帧总数: {dataset.num_frames}")
print(f"控制频率: {dataset.fps} Hz")

# 读取第 0 帧观察
sample = dataset[0]
print("观察状态形状:", sample["observation.state"].shape)  # 输出: (6,)
print("动作维度形状:", sample["action"].shape)  # 输出: (6,)
```

---

### 2. Open X-Embodiment / RLDS 数据接口

**RLDS**（Robot Learning Dataset Standard）是基于 TensorFlow Datasets（TFDS）构建的标准格式，是 Google 汇聚全球 22 个机器人实验室、跨几十种本体的 **Open X-Embodiment** 数据集所采用的核心标准。OpenVLA、RT-1、RT-2、Octo 等大模型均基于此接口进行分布式流式训练。

RLDS 将数据组织为嵌套的阶梯流（Step Sequence），每一个 Step 的典型字典结构如下：

```python
# RLDS / Open X-Embodiment 单步数据字典
step = {
    "is_first": bool,  # 是否为轨迹的起始第一帧 (t=0)
    "is_last": bool,  # 是否为轨迹的结束帧 (t=T)
    "is_terminal": bool,  # 任务是否自然达成终止
    "observation": {
        "image": np.ndarray,  # 主视角相机, [H, W, 3], uint8
        "wrist_image": np.ndarray,  # 手腕相机, [H, W, 3], uint8
        "natural_language_instruction": str,  # 任务文本指令, 如 "pick up yellow block"
        "state": np.ndarray,  # 机械臂本体状态, [P], float32
    },
    # 7 维末端动作 [dx, dy, dz, droll, dpitch, dyaw, gripper_open]
    "action": np.ndarray,  # [7], float32
    "discount": float,  # 折扣因子 (通常 1.0)
    "reward": float,  # 单步奖励 (0 或 1)
}
```

在 Python 中通过 `tensorflow_datasets` 直接流式读取：

```python
import tensorflow_datasets as tfds

# 远程流式加载 Open X-Embodiment 中的开源数据集（如 Fractal / RT-1 数据）
dataset = tfds.load("fractal20220817_data", split="train")

for episode in dataset.take(1):
    for step in episode["steps"]:
        instruction = (
            step["observation"]["natural_language_instruction"]
            .numpy()
            .decode("utf-8")
        )
        img_shape = step["observation"]["image"].shape
        action = step["action"].numpy()
        print(
            f"指令: {instruction} | 图像形状: {img_shape} | 动作: {action.round(3)}"
        )
        break
```

---

### 3. Robomimic / ACT HDF5 数据接口

在学术界和本地单机训练中，**HDF5** 是使用最广泛的层级二进制存储格式。Stanford 的 ACT（ALOHA 双臂系统）和 Columbia 的 Diffusion Policy 均原生采用这一组织形式。

一个典型的 HDF5 数据集文件在磁盘上按层级组织：

```text
dataset.hdf5
├── data/
│   ├── demo_0/
│   │   ├── obs/
│   │   │   ├── top_image            # [T+1, 480, 640, 3], uint8, 俯视相机
│   │   │   ├── left_wrist_image     # [T+1, 480, 640, 3], uint8, 左手腕相机
│   │   │   ├── right_wrist_image    # [T+1, 480, 640, 3], uint8, 右手腕相机
│   │   │   └── qpos                 # [T+1, 14], float32, 双臂 14 个关节当前角度
│   │   ├── actions                  # [T, 14], float32, 双臂目标关节角度动作
│   │   └── rewards                  # [T], float32
│   ├── demo_1/
│   │   └── ...
```

在 Python 中使用 `h5py` 读取 HDF5 数据极其直观：

```python
import h5py

# 打开 ALOHA 双臂 HDF5 示教文件
with h5py.File("aloha_bimanual_demo.hdf5", "r") as root:
    demo = root["data/demo_0"]

    # 提取第 0 条示教中的完整序列
    top_images = np.array(demo["obs/top_image"])  # [T + 1, H, W, 3]
    joint_positions = np.array(demo["obs/qpos"])  # [T + 1, 14]
    actions = np.array(demo["actions"])  # [T, 14]

    print(
        f"示教观察帧数 (T+1): {len(top_images)}, 动作执行步数 (T): {len(actions)}"
    )
    print(
        f"双臂关节状态维度: {joint_positions.shape[1]}, 动作控制维度: {actions.shape[1]}"
    )
```

---

### 三大主流格式选型对比

| 特性维度 | HuggingFace LeRobot | Open X-Embodiment (RLDS) | Robomimic / ACT (HDF5) |
| :--- | :--- | :--- | :--- |
| **底层存储格式** | Arrow + MP4 视频 + Safetensors | TFRecords / Apache Beam | 层级 HDF5 / Zarr 二进制 |
| **磁盘空间开销** | **极小**（MP4 高压缩比，约为原始图像的 1/20） | 中等（TFRecord 序列化压缩） | 较大（通常存储未压缩或 PNG 序列） |
| **开发生态** | **PyTorch 原生**，开箱即用 | **TensorFlow / JAX**，支持多机分布式流式加载 | **NumPy / PyTorch**，单机切片读取速度极快 |
| **典型适用模型** | LeRobot 策略基准、桌面单/双臂操作 | OpenVLA, Octo, RT-2 等大规模跨本体预训练 | ACT, Diffusion Policy 算法复现与单任务调优 |

---

## 闭环在线推理接口设计

搞清楚了离线数据的存储与读取后，我们最后来看一下：**训练好的策略模型在部署到真实机械臂上时，在线闭环交互接口（Inference Loop）是如何运转的？**

标准具身智能系统的部署架构通常遵循类似于 Gym 的环境封装：

```python
class RealRobotEnv:
    """真实机械臂硬件环境封装接口 (符合 Gym 标准 API)"""

    def __init__(self, camera_driver, robot_driver, control_fps=50):
        self.cam = camera_driver
        self.robot = robot_driver
        self.dt = 1.0 / control_fps

    def reset(self) -> dict:
        """重置机械臂回初始准备位姿，并返回初始观察 o_0"""
        self.robot.move_to_home()
        time.sleep(1.0)
        return self._get_observation()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """向物理电机下发动作 a_t，并等待物理时间片演化，返回新观察 o_{t+1}"""
        t_start = time.time()

        # 1. 安全限幅与物理指令下发
        safe_action = np.clip(action, -1.0, 1.0)
        self.robot.send_command(safe_action)

        # 2. 精确延时以稳定控制周期 (如 20ms)
        elapsed = time.time() - t_start
        if elapsed < self.dt:
            time.sleep(self.dt - elapsed)

        # 3. 读取执行后产生的新观察
        next_obs = self._get_observation()
        done = self._check_termination()
        return next_obs, 0.0, done, {}

    def _get_observation(self) -> dict:
        return {
            "image": self.cam.read_latest_frame(),
            "proprio": self.robot.read_joint_states(),
        }
```

在主控程序中，策略以标准闭环方式被持续调用：

```python
# 闭环实机推理主循环
env = RealRobotEnv(camera_driver=..., robot_driver=...)
policy = load_trained_policy("checkpoint.pt")
stats = load_action_stats("action_stats.json")

obs = env.reset()
instruction = "拿起红色的水杯"

for step in range(400):
    # 1. 输入当前观察与语言，策略前向推理输出归一化动作
    norm_action = policy.predict(obs, instruction)

    # 2. 反归一化为物理动作
    real_action = unnormalize_action(norm_action, stats)

    # 3. 物理执行并获取下一步观察
    obs, reward, done, info = env.step(real_action)

    if done:
        print("🎉 任务成功完成！")
        break
```

---

## 本节小结

在本节中，我们系统学习了机器人学习的数据表示、采集与接口设计：

- 一条标准的机器人 Episode 记录了图像、本体状态、语言指令和动作序列，严格遵循 **$T+1$ 帧观察对应 $T$ 步动作**的时序关系。
- 动作 $a_t$ 永远夹在观察 $o_t$ 与 $o_{t+1}$ 之间，保持严格的物理因果时序。
- 动作空间可选用笛卡尔末端位姿增量或关节角度增量，分别对应了跨本体语义通用性与底层电机执行稳定性的权衡。
- FK / IK / PD 是底层执行栈；学习替代的是感知与技能切换，不是电机驱动。
- 人类示教数据可通过主从同构臂（ALOHA/GELLO）、VR 空间手柄或人机介入式（Intervention）方案进行多线程高频录制。
- 原始数据需经过发呆静止段截断、手抖滤波与全数据集全局归一化清洗后，方可送入模型训练。
- 在实际工程中，可直接基于 LeRobot、RLDS 或 HDF5 等开源通用标准格式进行高效组织与读取，并通过标准 Env 闭环接口完成实机部署。

现在，我们已经把数据和接口准备好了。接下来的核心问题是：**如何让神经网络从人类示范数据中学会输出动作？如果我们直接用监督学习去拟合示范，会遇到什么困难？**

在下一节中，我们将深入讨论模仿学习以及现代生成式策略的核心原理。

---

## 参考资料

1. [LeRobot: State-of-the-art Machine Learning for Real-World Robotics (HuggingFace)](https://github.com/huggingface/lerobot) —— 开源机器人学习库，包含标准的数据集格式与硬件接口定义。
2. [ALOHA: A Low-cost Open-source Hardware System (Tony Zhao)](https://tonyzhaozh.github.io/aloha/) —— 低成本双臂遥操作硬件与数据采集标准开源方案。
3. [RT-1: Robotics Transformer for Real-World Control at Scale (Brohan et al., 2022)](https://arxiv.org/abs/2212.06817) —— 大规模机器人数据采集规范与模型设计经典论文。
4. [GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework (Wu et al., 2023)](https://wuphilipp.github.io/gello_site/) —— 通用低成本主从臂遥操作开源硬件与数据采集系统。
