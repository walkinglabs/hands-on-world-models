# 7.7 视觉-语言-动作模型 (VLA) 与 RT-X

我们在前面的章节中深入讨论了行为克隆（BC）、扩散策略（Diffusion Policy）与动作分块（ACT）等基于模仿学习的控制方法。在特定的实验台面或抓取任务上，这些算法能让机械臂展现出如行云流水般的流畅反应。

然而，若我们尝试用一句日常的人类自然语言——比如“请把桌子左边盘子里的红苹果递给坐在右边的客人”——去指挥这些传统策略时，它们会立刻无所适从。传统策略本质上是一组“快速的条件反射神经回路”：它们擅长处理固定的像素输入，却完全无法理解什么是“红苹果”、什么是“客人”，更缺少对空间方位与常识逻辑的通用认知。

如何将拥有万亿级互联网常识图文先验的大语言模型（LLM）与多模态大模型（VLM），与机械臂在物理世界中的毫米级动作控制深度融为一体？这催生了具身智能领域最重要的前沿路线——**视觉-语言-动作模型（Vision-Language-Action Models, VLA）**。

<div align="center">

<img src="/figures/07-robot-policy/source/07-vla-rtx/rt2-fig1.png" alt="RT-2 将机器人动作表示为文本词元，直接在大模型中联合训练。" width="86%">

_图 7.7-1：RT-2 将机器人动作表示为文本词元，直接在大模型中联合训练。 出处：[RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control，Anthony Brohan et al.，2023](https://arxiv.org/abs/2307.15818)。_

</div>

---

## 7.7.1 物理与认知基石：符号接地困境与跨模态具身统一

要理解 VLA 模型的诞生逻辑，我们首先需要从人类语言的“抽象符号”与物理世界的“连续受力”之间的认知鸿沟讲起。

### 1. 经典工程探索与“符号接地”硬性极限
在日常语言与符号系统中，知道语言文字是由一个个抽象离散的“符号”组成的（例如“易碎的玻璃杯”这六个汉字）。而在经典物理实验中测量电流或弹簧弹力时，面对的则是平滑连续的“物理量”（例如电压 $2.35\text{ V}$、拉力 $4.9\text{ N}$）。

计算机科学家在很早就尝试让人工智能控制机械臂，但经历过漫长的困境：
- **20 世纪 70 年代（符号积木世界）**：麻省理工学院 Terry Winograd 开发了著名的 **SHRDLU** 系统。在纯虚拟的微观积木世界中，系统可以通过语法树解析人类指令“把绿色的圆锥体放到红色方块上”，并规划虚拟机械臂移动。但当科学家尝试把 SHRDLU 部署到真实世界的物理摄像头与机械臂时，系统立刻崩溃——真实摄像头的噪点、光照反光、物体的任意摆放无法被直接转化为离散的逻辑谓词 `is_green(block_A)`；
- **20 世纪 90 年代的“感知-规划-执行”流水线**：经典的具身系统尝试将任务拆解为串联的三个步骤：第一步用目标检测算法在图像中框出杯子坐标；第二步由几何规划器计算机械臂各关节要转过的角度；第三步由电机控制器输出电流力矩。这种流水线遭遇了认知科学中著名的**符号接地困境（Symbol Grounding Problem）**——如果机器人只在字典里查到“玻璃杯是易碎品”，却不知道伸出手臂时电机应该输出几牛·米的力矩才能刚好拿稳而不捏碎，那么抽象的文字符号就始终无法“稳稳地落在物理地表上”。流水线中任何一个前置模块产生 $1\text{ cm}$ 的几何误差，级联放大后就会导致打翻水杯或碰撞桌台。

### 2. 现代大模型带来的认知降维与大一统
2020 年代以来，基于海量互联网图文训练的多模态大模型展现出了令人震撼的通用世界常识：它们能够看懂图像中的幽默漫画、理解不同餐具的用途、甚至推断出“倾斜的水杯即将掉落”等物理趋势。

机器人学家们由此萌生了一个极其大胆的想法：**我们是否可以直接打破‘先识别文字、再计算几何坐标、最后算电机角度’的繁琐分工，让大模型在看懂画面的同时，像‘写作文接龙’一样直接写出下一步机械臂要走的空间位移？**

这一构想直接催生了 Google 研发的 **RT-1**（Robotics Transformer 1, 2022）与 **RT-2**（Robotics Transformer 2, 2023）。

<div align="center">

<img src="/figures/07-robot-policy/source/07-vla-rtx/rt2-fig2.png" alt="RT-1 采用 EfficientNet、FiLM 与 TokenLearner 压缩视觉标记，送入因果 Transformer 生成离散动作。" width="86%">

_图 7.7-2：RT-1 采用 EfficientNet、FiLM 与 TokenLearner 压缩视觉标记，送入因果 Transformer 生成离散动作。 出处：[RT-1: Robotics Transformer for Real-World Control at Scale，Anthony Brohan et al.，2022](https://arxiv.org/abs/2212.06817)。_

</div>

---

## 7.7.2 核心数学推导一：动作空间离散化与词元化（Action Tokenization）

在经典物理力学中研究物体的空间运动时，最熟悉的工具是**三维直角坐标系** $(x, y, z)$ 和**量角器**。机械臂在空间中伸展抓取，本质上就是在连续改变它在空间中的前后、左右、上下毫米级距离（三维平移）以及手腕的翻转角度（三维旋转）。

在机器人学中，这一套用来完全确定末端夹爪位置与朝向的实数集合，被称为**空间位姿（Pose）**。真实世界里的物理位移是平滑连续的——机械臂可以向前伸展 $1.0\text{ 毫米}$，也可以向前伸展 $1.001\text{ 毫米}$，其间存在无数种可能。

然而，大语言模型（如 ChatGPT、LLaMA）内部处理的信息，本质上是一本由整数编号构成的“离散字典”——在计算机术语中，字典里的每一个词或字符对应的整数编号被称为**词元（Token）**（例如在文本词表中，编号 101 代表“苹果”，编号 102 代表“香蕉”）。

这就引出了一个核心的技术矛盾：**如何把物理世界中平滑连续的空间位移与旋转，转化为大模型看得懂、能自回归输出的离散词元编号（Action Tokens），同时还不丢失毫米级的控制精度？**

### 1. 机械臂末端 7 维动作空间
在标准操作任务中，机械臂末端夹爪在一步控制周期内的相对运动，可以用一个 7 维实数向量来完整描述：

$$\mathbf{a}_t = [\Delta x, \Delta y, \Delta z, \Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw}, a_{\text{gripper}}]^\top \in \mathbb{R}^7$$

> **公式符号逐一拆解**：
> - $\Delta x, \Delta y, \Delta z$：末端夹爪在直角坐标系中沿前后、左右、上下方向的平移位移增量（单位：米 $\text{m}$）；
> - $\Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw}$：手腕围绕三个坐标轴的旋转角度增量（单位：弧度 $\text{rad}$）；
> - $a_{\text{gripper}} \in [0, 1]$：夹爪开合度（$0$ 表示完全张开手掌，$1$ 表示完全并拢夹紧）。

### 2. 均匀分桶离散化（Uniform Binning）
在基础数学中，我们熟悉用区间分割来逼近连续曲线。对于第 $j$ 维动作，假设它物理允许的安全移动范围在 $[a_{\min, j}, a_{\max, j}]$ 之间。我们可以像用直尺刻度一样，把这个区间等距离切分成 $B$ 个互不重叠的离散小格子（在计算机中称为“桶”，通常取 $B = 256$，恰好对应计算机中 1 个字节 8-bit 的存储范围）。

<div align="center">

<img src="/figures/07-robot-policy/source/07-vla-rtx/rt1-fig5.png" alt="连续动作区间被均匀分割为 256 个桶" width="86%">

_图 7.7-3：连续动作区间被均匀分割为 256 个桶；连续值映射到对应桶中心，还原时通过反归一化取回连续估计。_

</div>

连续动作值 $a_j$ 变成整数词元（Token）的过程分为三步：
1. **归一化到 $[0, 1]$ 标量区间**：
   $$\tilde{a}_j = \frac{a_j - a_{\min, j}}{a_{\max, j} - a_{\min, j}}$$
2. **截断防越界（Clamping）**：
   $$\hat{a}_j = \min(\max(\tilde{a}_j, 0.0), 1.0)$$
3. **线性缩放并就近四舍五入取整（Rounding）**：
   $$\text{Token}_j = \text{round}\left(\hat{a}_j \times (B - 1)\right) \in \{0, 1, 2, \dots, B-1\}$$

当模型预测出这个整数编号 $\text{Token}_j$ 后，底层控制程序通过反比例映射，立刻将其还原为真实的物理位移：

$$a_j^{\text{continuous}} = a_{\min, j} + \frac{\text{Token}_j}{B - 1} (a_{\max, j} - a_{\min, j})$$

**手算代入算例**：
设机械臂在 $x$ 轴方向单步允许的最大移动范围在 $[-0.5\text{ m}, +0.5\text{ m}]$（总跨度为 $1.0\text{ 米}$），量化桶数取 $B = 256$。假定策略期望下发的连续物理位移为 $a_x = +0.10\text{ m}$。

1. 计算归一化比例：
   $$\tilde{a}_x = \frac{0.10 - (-0.50)}{0.50 - (-0.50)} = \frac{0.60}{1.00} = 0.60$$
2. 转换为离散词元编号：
   $$\text{Token}_x = \text{round}(0.60 \times 255) = \text{round}(153.0) = 153$$
3. 反向还原物理位移：
   $$a_x^{\text{recon}} = -0.50 + \frac{153}{255} \times 1.00 = -0.50 + 0.60 = +0.10\text{ m}$$

在 256 个分桶下，相邻两格之间的最大误差（量化物理分辨率）为：

$$\Delta a_{\text{res}} = \frac{a_{\max} - a_{\min}}{2 \times (B - 1)} = \frac{1.0\text{ m}}{2 \times 255} \approx 0.00196\text{ m} = 1.96\text{ mm}$$

这个手算结果直观证明：**在长达 $1\text{ 米}$ 的操作范围内，仅用 256 个离散词元编号，就能把量化误差锁死在 $2\text{ 毫米}$ 以内**，完全足以支撑日常绝大多数端茶倒水、抓取工具的精细作业！

<details>
<summary><b>深入推导：动作连续概率密度与自回归离散交叉熵损失的变分推导（点击展开查看完整推导）</b></summary>

设真实动作服从连续条件概率密度 $p(\mathbf{a} \mid \mathbf{o})$，其中 $\mathbf{o} = (I, \text{lang})$ 为多模态观测。
采用分桶量化后，第 $j$ 维动作落在第 $k$ 个区间的概率为：
$$P(\text{Token}_j = k \mid \mathbf{o}) = \int_{\text{bin}_k} p(a_j \mid \mathbf{o}) da_j$$
模型通过参数为 $\theta$ 的 Transformer 输出各类别的 Logits $\mathbf{z}_{j} \in \mathbb{R}^B$，并通过 Softmax 计算预测概率：
$$\hat{P}_\theta(\text{Token}_j = k \mid \mathbf{o}) = \frac{\exp(z_{j, k})}{\sum_{m=0}^{B-1} \exp(z_{j, m})}$$
训练时的自回归监督目标为多维动作的负对数似然（交叉熵损失）：
$$\mathcal{L}_{\text{VLA}}(\theta) = -\sum_{t=1}^T \sum_{j=1}^7 \log \hat{P}_\theta\left(\text{Token}_{t, j} = k_{t, j}^* \mid \mathbf{o}_t, \mathbf{a}_{t, <j}^*\right)$$
相比于均方误差（MSE Loss），交叉熵损失天然具备拟合多峰分布（Multimodal Distribution）的能力，当面对“向左绕开障碍”与“向右绕开障碍”两个同等合理的动作时，模型不会输出导致正中碰撞的折中平均值。
</details>

---

## 7.7.3 核心数学推导二：跨模态特征融合与 FiLM 机制

在 RT-1 架构中，如何让抽象的语言指令（例如“抓取红色的可乐罐”）去指导视觉卷积网络（CNN）在画面中找到目标？RT-1 采用了一种基于一次函数思想的调制机制——**特征仿射调制（Feature-wise Linear Modulation, FiLM）**。

### 1. 仿射变换的基础几何直觉
在初等代数中，我们最熟悉的基础函数是一次函数 $y = kx + b$。
- 斜率 $k$ 负责将输入信号按比例放大或缩小（称为**缩放 Scale**）；
- 截距 $b$ 负责将信号整体向上或向下平移（称为**偏置 Shift**）。

在高等数学中，这种“乘一个系数再加上一个偏置”的线性变换被称为**仿射变换（Affine Transformation）**。

FiLM 机制把这一思想直接搬到了多维图像特征图上：自然语言指令经过文本编码器提取为一个语义向量 $\mathbf{e}_{\text{lang}}$ 后，网络用两组全连接层分别计算出每一个视觉特征通道专属的“缩放因子 $\gamma$”与“偏置项 $\beta$”。

<div align="center">

<img src="/figures/07-robot-policy/latex/07-vla-rtx/film-channel-broadcast.png" alt="语言生成的通道缩放和平移参数广播到全部空间位置" width="86%">

_图 7.7-4：语言嵌入生成每个通道的 γ 与 β；固定通道 c 后，同一对标量会广播到全部 H×W 位置。_

</div>

设卷积神经网络提取出的中间视觉特征张量为 $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$（包含 $C$ 个特征通道，图像高 $H$，宽 $W$）。

由语言生成的逐通道仿射参数为：

$$\boldsymbol{\gamma} = \mathbf{W}_\gamma \mathbf{e}_{\text{lang}} + \mathbf{b}_\gamma \in \mathbb{R}^C$$

$$\boldsymbol{\beta} = \mathbf{W}_\beta \mathbf{e}_{\text{lang}} + \mathbf{b}_\beta \in \mathbb{R}^C$$

对第 $c$ 个特征通道的所有像素位置 $(h, w)$ 执行标准的一次函数仿射调制：

$$\mathbf{F}'_{c, h, w} = \gamma_c \cdot \mathbf{F}_{c, h, w} + \beta_c$$

> **物理直觉**：如果人类指令提到了“红色”，语言模块就会主动调大负责识别红色特征图的通道斜率 $\gamma_{\text{red}} \gg 1$；若与当前指令无关（例如“蓝色桌面”），则将对应通道的响应压低 $\gamma_{\text{blue}} \approx 0$。这使得视觉网络在特征提取的最早期，就具备了“带着人类目的去观察画面”的注意力。

<details>
<summary><b>深入推导：FiLM 逐通道张量广播机制与梯度反向传播链式法则（点击展开查看完整推导）</b></summary>

将调制公式写为完整的张量批处理形式。设输入特征为 $\mathbf{F} \in \mathbb{R}^{B \times C \times H \times W}$，语言参数 $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^{B \times C}$。
通过张量维度扩展（Unsqueeze）与广播（Broadcasting）：
$$\mathbf{F}' = \boldsymbol{\gamma}_{[:, :, \text{None}, \text{None}]} \odot \mathbf{F} + \boldsymbol{\beta}_{[:, :, \text{None}, \text{None}]}$$
在反向传播时，下游损失 $\mathcal{L}$ 对语言调制参数的梯度由所有空间位置聚合求和得到：
$$\frac{\partial \mathcal{L}}{\partial \gamma_{b, c}} = \sum_{h=1}^H \sum_{w=1}^W \frac{\partial \mathcal{L}}{\partial F'_{b, c, h, w}} F_{b, c, h, w}$$
$$\frac{\partial \mathcal{L}}{\partial \beta_{b, c}} = \sum_{h=1}^H \sum_{w=1}^W \frac{\partial \mathcal{L}}{\partial F'_{b, c, h, w}}$$
这种全局空间梯度的反向汇聚，促使语言编码器能够稳定地学会跨模态空间注意力调节。
</details>

---

## 7.7.4 从 RT-1 到 RT-2：大语言模型赋能的涌现与泛化

虽然 RT-1 实现了多任务指令控制，但其底层从零训练的 Transformer 参数量较小（约 3500 万参数），泛化能力受限于机器人采集的几万条真机轨迹。

2023 年，Google 推出了里程碑式的 **RT-2**。RT-2 不再单独从零训练小模型，而是直接利用拥有数百亿参数的预训练多模态大模型（如 PaLI-X 55B 和 PaLM-E 12B），进行**动作-文本联合微调（Co-fine-tuning）**。

### 1. 动作即文本（Actions as Tokens）
在 RT-2 中，动作词元 `<0>` 到 `<255>` 直接被加入到了大模型的常用文本词表中，与“你好”、“苹果”等词语享受完全同等的地位。

例如，对于人类指令“把恐龙玩具放到正方形盒子里”，RT-2 的输入与输出格式就像写一段问答对话：
- **输入序列**：`[工作台图像] + "指令: 把恐龙玩具放到正方形盒子里"`
- **大模型自回归输出文本**：`"128 140 102 128 128 135 255"`

这 7 个数字字符串被直接解析为末端机械臂在直角坐标系中的 7 维物理控制量。

### 2. 互联网级常识的“物理涌现”
由于 RT-2 继承了预训练大模型阅览全球互联网图文后获得的常识，它在物理世界中涌现出了惊人的推理能力：
1. **多语言零样本理解**：即使训练数据中只有英文操作轨迹，用户用德语、中文发出指令，机器人依然能准确执行；
2. **常识推理抓取**：如果向机器人发出指令“把能用来敲钉子的工具递给我”，即便现场没有锤子，RT-2 也能推理出画面中的坚硬石块可以充当替代工具并准确抓取；
3. **空间隐喻与安全意识**：能够理解“把即将掉落桌面的易碎物扶正”，自动识别处于桌沿边缘倾斜的水杯。

> **想一想**
>
> 既然把动作当成文字直接输出这么强大，为什么我们不能直接用部署在云端的大型 VLM（如 GPT-4V 或 Gemini）以 $1000\text{ Hz}$（每秒 1000 次）的高频直接控制机器人的每一个关节电机？
>
> **解答**：这受限于巨大的计算延迟。数百亿参数的大模型单次推理需要耗费 $100 \sim 500\text{ 毫秒}$（即输出频率仅有 $2 \sim 10\text{ Hz}$）。而机器人与物体接触时的物理防滑、抗冲击调节必须在 $1\text{ 毫秒}$（$1000\text{ Hz}$）内作出响应。因此，现代具身架构普遍采用**分层控制**：由机载/云端大模型以低频（$3 \sim 10\text{ Hz}$）规划宏观的末端目标，再由底层的硬件控制器（如前文讲过的 WBC 或 PD 控制器）以 $1000\text{ Hz}$ 超高频确保物理接触的安全与平衡。

---

## 7.7.5 RT-X 与 Open X-Embodiment 跨具身大一统

在具身智能的发展历程中，最大的数据瓶颈在于“孤岛效应”——每家实验室使用不同构型、不同关节数、不同尺寸的机器人（如单臂 Franka、双臂 ALOHA、移动底盘 Google Robot）。

2023 年底，全球 34 间顶尖机器人实验室联合推出了 **Open X-Embodiment (RT-X)** 数据集与模型。

<div align="center">

<img src="/figures/07-robot-policy/source/07-vla-rtx/rtx-fig3.png" alt="RT-1-X 与 RT-2-X 在统一跨具身数据上分别延续机器人 Transformer 与 VLA 路线。" width="86%">

_图 7.7-5：RT-1-X 与 RT-2-X 在统一跨具身数据上分别延续机器人 Transformer 与 VLA 路线。 出处：[Open X-Embodiment: Robotic Learning Datasets and RT-X Models，Open X-Embodiment Collaboration，2023](https://arxiv.org/abs/2310.08864)。_

</div>

RT-X 通过构建统一的 7 维末端执行器控制协议，汇聚了跨越 22 种不同机器人形态、超过 100 万条真实操作轨迹（包含了 500 多种技能）。

实验表明，在多形态跨具身数据集上联合预训练的 **RT-1-X** 与 **RT-2-X**，其在新机器人平台上的零样本迁移成功率比仅在单一机器人数据上训练的模型平均提升了 **$50\%$ 以上**！这首次在物理机器人领域确立了类似于自然语言处理的“数据 Scaling Law（缩放定律）”。

---

## 7.7.6 纯底层 PyTorch 代码实现：从零搭建微型 VLA 策略网络

下面我们使用纯底层 PyTorch 算子手写实现一个结构严密的微型 VLA 模型，包含动作离散化器、FiLM 跨模态调制层与因果自回归解码器。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionTokenizer:
    """
    动作离散化与去离散化处理器 (Action Tokenizer)
    将 7 维连续空间动作映射为 [0, vocab_size - 1] 的整数词元索引。
    """
    def __init__(self, action_min: list[float], action_max: list[float], vocab_size: int = 256):
        self.action_min = torch.tensor(action_min, dtype=torch.float32)
        self.action_max = torch.tensor(action_max, dtype=torch.float32)
        self.vocab_size = vocab_size

    def tokenize(self, action: torch.Tensor) -> torch.Tensor:
        """
        连续物理位移 -> 离散词元编号
        :param action: (Batch, action_dim) 连续动作向量
        :return: (Batch, action_dim) 长整型词元编号张量
        """
        self.action_min = self.action_min.to(action.device)
        self.action_max = self.action_max.to(action.device)

        # 1. 线性归一化到 [0, 1]
        norm_action = (action - self.action_min) / (self.action_max - self.action_min + 1e-8)
        # 2. 裁剪可能越界的极端数值
        clamped_action = torch.clamp(norm_action, 0.0, 1.0)
        # 3. 缩放到桶区间并就近取整
        tokenized = torch.round(clamped_action * (self.vocab_size - 1)).long()
        return tokenized

    def detokenize(self, tokenized: torch.Tensor) -> torch.Tensor:
        """
        离散词元编号 -> 连续物理位移还原
        :param tokenized: (Batch, action_dim) 词元编号
        :return: (Batch, action_dim) 连续动作估计值
        """
        self.action_min = self.action_min.to(tokenized.device)
        self.action_max = self.action_max.to(tokenized.device)

        norm_action = tokenized.float() / (self.vocab_size - 1)
        action = norm_action * (self.action_max - self.action_min) + self.action_min
        return action

class FiLMLayer(nn.Module):
    """
    特征仿射调制层 (Feature-wise Linear Modulation, FiLM)
    利用语言特征动态生成通道级缩放因子 gamma (斜率) 与偏置 beta (截距)。
    """
    def __init__(self, lang_dim: int, channels: int):
        super().__init__()
        self.fc_gamma = nn.Linear(lang_dim, channels)
        self.fc_beta = nn.Linear(lang_dim, channels)

    def forward(self, x: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        """
        :param x: 视觉特征图 (B, C, H, W)
        :param lang_emb: 语言嵌入向量 (B, lang_dim)
        :return: 经过一次函数仿射调制后的视觉特征图 (B, C, H, W)
        """
        # 计算逐通道的缩放与偏置，形状扩展为 (B, C, 1, 1)
        gamma = self.fc_gamma(lang_emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.fc_beta(lang_emb).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class TinyVLAModel(nn.Module):
    """
    微型视觉-语言-动作 (Tiny VLA) 策略模型
    结构：CNN 视觉前端 + FiLM 跨模态融合 + 因果 Transformer 解码器 + 动作预测头
    """
    def __init__(
        self,
        action_dim: int = 7,
        lang_dim: int = 128,
        img_channels: int = 64,
        vocab_size: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.vocab_size = vocab_size

        # 1. 视觉特征提取与 FiLM 调制
        self.vision_conv = nn.Conv2d(3, img_channels, kernel_size=8, stride=4, padding=2)
        self.film = FiLMLayer(lang_dim, img_channels)
        self.vision_proj = nn.Linear(img_channels, d_model)

        # 2. 动作词元嵌入层
        self.action_emb = nn.Embedding(vocab_size, d_model)

        # 3. 因果自回归 Transformer 解码网络
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 动作离散分类输出头
        self.action_head = nn.Linear(d_model, vocab_size)

    def forward(
        self, image: torch.Tensor, lang_emb: torch.Tensor, action_tokens: torch.Tensor = None
    ) -> torch.Tensor:
        """
        前向计算
        :param image: (B, 3, H, W) RGB 视觉张量
        :param lang_emb: (B, lang_dim) 语言指令嵌入
        :param action_tokens: (B, action_len) 动作词元序列 (自回归输入)
        :return: (B, seq_len, vocab_size) 动作词元的分类对数几率 (Logits)
        """
        # 1. 提取视觉特征并进行语言调制
        v_feat = F.relu(self.vision_conv(image)) # (B, img_channels, H', W')
        v_feat = self.film(v_feat, lang_emb)

        # 展平为视觉序列: (B, H'*W', img_channels) -> (B, seq_vis, d_model)
        v_seq = v_feat.flatten(2).transpose(1, 2)
        v_seq = self.vision_proj(v_seq)

        # 2. 拼接视觉词元与动作词元
        if action_tokens is not None:
            a_seq = self.action_emb(action_tokens) # (B, action_len, d_model)
            seq = torch.cat([v_seq, a_seq], dim=1) # (B, seq_vis + action_len, d_model)
        else:
            seq = v_seq

        # 3. 构造下三角因果掩码 (Causal Mask)，防止注意力读取未来词元
        seq_len = seq.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=image.device), diagonal=1
        )

        # 4. Transformer 自回归编码与分类输出
        hidden_states = self.transformer(seq, mask=causal_mask)
        logits = self.action_head(hidden_states) # (B, seq_len, vocab_size)
        return logits

# ===================================================================
# 单元测试与张量演变追踪
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    action_dim = 7
    vocab_size = 256
    img_h, img_w = 64, 64

    # 1. 测试动作离散化器
    tokenizer = ActionTokenizer(
        action_min=[-1.0] * action_dim, action_max=[1.0] * action_dim, vocab_size=vocab_size
    )
    dummy_continuous_action = torch.tensor(
        [[0.10, -0.50, 0.80, 0.00, -0.90, 0.20, 1.00], [0.00, 0.50, -0.20, 0.30, -0.10, 0.00, 0.00]]
    )
    tokens = tokenizer.tokenize(dummy_continuous_action)
    reconstructed_action = tokenizer.detokenize(tokens)
    max_recon_error = (reconstructed_action - dummy_continuous_action).abs().max().item()

    print(f"[Tokenizer Test] 输入连续动作:\n{dummy_continuous_action.numpy().round(3)}")
    print(f"[Tokenizer Test] 量化词元索引:\n{tokens.numpy()}")
    print(f"[Tokenizer Test] 最大重构量化误差: {max_recon_error:.6f} (小于 2mm/格精度)")
    assert max_recon_error < 0.01, "动作离散化重构精度不足！"

    # 2. 测试 VLA 模型前向推理
    model = TinyVLAModel(action_dim=action_dim, lang_dim=128, vocab_size=vocab_size)
    model.eval()

    dummy_image = torch.randn(batch_size, 3, img_h, img_w)
    dummy_lang = torch.randn(batch_size, 128)
    dummy_action_tokens = torch.randint(0, vocab_size, (batch_size, action_dim))

    with torch.no_grad():
        output_logits = model(dummy_image, dummy_lang, dummy_action_tokens)

    print(f"[VLA Test] 视觉输入张量形状: {dummy_image.shape}")
    print(f"[VLA Test] 语言嵌入张量形状: {dummy_lang.shape}")
    print(f"[VLA Test] 模型输出 Logits 形状: {output_logits.shape}")

    assert output_logits.shape[-1] == vocab_size, "分类输出维度与词表大小不符！"
    print("✓ 微型 VLA 策略模型单测全部通过！")
```

---

## 7.7.7 本节小结

回顾本节内容，我们建立了一条从经典符号接地走向端到端视觉-语言-动作大模型的完整认知演进：
1. **符号接地的时代演进**：经典流水线因级联误差与语义鸿沟而难以应对非结构化世界，VLA 将物理控制无缝融入预训练多模态大模型的自回归生成体系；
2. **动作离散化的数学与工程权衡**：通过 256 桶的均匀分段，连续空间位移被离散化为紧凑的整数词元，既天然支持多模态动作分布，又将物理量化误差控制在毫米级；
3. **跨模态特征调制的直观法则**：FiLM 机制利用语言特征生成仿射参数 $\gamma$ 与 $\beta$，在视觉早期以通道级拉伸平移实现了“意图导向”的特征聚焦；
4. **跨具身统一与 Scaling Law 落地**：RT-X 证明了在统一 7 维末端动作空间下，汇聚多机器人形态的海量异构轨迹能够激发跨具身的正向知识迁移。
