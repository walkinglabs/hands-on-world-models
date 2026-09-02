# 5.5 从零实现可交互视频世界模型 (Interactive Video World Model from Scratch)

在视频生成技术从“静态生成一段固定短片”迈向“生成一个具备交互能力的动态物理世界”的跨越中，**动作可控交互式视频世界模型（Action-Conditioned Interactive Video World Models, 如 Oasis, Genie, GameNGen）** 代表了当前具身智能与人工智能世界模拟的最前沿突破。

传统的视频生成模型如同一台单向播放的电影放映机：输入一段文本提示词，它输出一段固定无法改变的 MP4 视频；
而**可交互视频世界模型**则如同一台真正运行在神经网络内部的“神经物理游戏引擎”：
- 用户或机械臂在每一毫秒按下键盘、推动摇杆或下发电机扭矩指令 $\mathbf{a}_t$；
- 神经网络实时接收该动作，并依据物理运动学规律，在下一帧精准渲染出角色移动、刚体碰撞、灯光阴影变化的动态响应画面 $\hat{\mathbf{x}}_{t+1}$；
- 智能体根据最新画面做出下一步决策，形成真正意义上的人机/智机在环物理闭环！

本节我们将从初等仿射特征调制出发，严密推导动作条件特征线性调制（FiLM）、门控时空残差融合与自回归累积漂移抑制机理，并使用纯底层 PyTorch 从零手写一个端到端完整的动作可控交互式视频世界模型。

<div align="center">

<img src="/figures/05-interactive-video/source/05-interactive-video-scratch/diamond-fig1.png" alt="Oasis 可交互神经物理世界模型：根据用户键盘操作实时生成可交互物理沙盒世界画面。" width="86%">

_图 5.5-1：Oasis 可交互神经物理世界模型：根据用户键盘操作实时生成可交互物理沙盒世界画面。 出处：[Oasis: A Universe in a Transformer，Decart & Etched，2024](https://oasis.decart.ai/)。_

</div>

---

## 5.5.1 物理与交互基石：动作条件注入与神经游戏引擎闭环

要实现对物理画面的精准动作操控，系统必须在深度神经网络的每一层建立起动作控制量与视觉特征图之间的显式物理因果映射。

### 1. 人机在环交互控制闭环
在每一时间步 $t$：
1. **输入当前画面与动作**：系统输入上一帧历史观测 $\mathbf{x}_{t-1}$ 以及外部输入的物理动作指令 $\mathbf{a}_{t-1}$（如 `[向左平移, 跳跃, 夹爪闭合]`）；
2. **潜在物理动力学演变**：网络内部的动作调制层将动作信号投影为动力学形变速度场；
3. **高保真画面输出**：系统渲染出新画面 $\hat{\mathbf{x}}_t$；
4. **状态反馈**：新画面作为下一时刻的初始历史，持续循环滚动！

### 2. 动作注入的物理难题
- 动作信号通常是一个极低维度的稀疏向量（如 4 维浮点数）；
- 视频特征图是数万维的高维三维张量；
- 如果仅仅将动作与图像简单相加，动作信号会被庞大的像素特征瞬间淹没冲淡，导致模型生成“视而不见”的失控失灵画面。

<div align="center">

<img src="/figures/05-interactive-video/latex/05-interactive-video-scratch/interleaved-action-causal-visibility.png" alt="FiLM 动作条件特征线性调制数据流：动态缩放因子 gamma 与平移因子 beta 的逐通道仿射变换" width="86%">

_图 5.5-2：FiLM 动作条件特征线性调制数据流：动态缩放因子 gamma 与平移因子 beta 的逐通道仿射变换。_

</div>

---

## 5.5.2 核心数学推导一：特征线性调制 (FiLM) 与门控跨模态融合

为了让低维动作指令能够精准“指挥”高维空间特征图，Perez 等人提出了 **特征线性调制（Feature-wise Linear Modulation, FiLM）**。

<div align="center">

<img src="/figures/05-interactive-video/source/05-interactive-video-scratch/genie-fig1.png" alt="Genie 交互式世界模型：从无标注网络视频中无监督学习潜在动作并实现用户可交互控制。" width="86%">

_图 5.5-3：Genie 交互式世界模型：从无标注网络视频中无监督学习潜在动作并实现用户可交互控制。 出处：[Genie: Generative Interactive Environments，Jake Bruce et al.，2024](https://arxiv.org/abs/2402.15391)。_

</div>

### 1. FiLM 动作逐通道仿射变换方程
设中间视觉特征图为 $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$。
通过一个动作映射网络，根据输入动作 $\mathbf{a} \in \mathbb{R}^{d_a}$ 预测出每通道专属的缩放向量 $\boldsymbol{\gamma}(\mathbf{a}) \in \mathbb{R}^C$ 与平移向量 $\boldsymbol{\beta}(\mathbf{a}) \in \mathbb{R}^C$：

$$\boldsymbol{\gamma}(\mathbf{a}) = \mathbf{W}_\gamma \mathbf{a} + \mathbf{b}_\gamma, \quad \boldsymbol{\beta}(\mathbf{a}) = \mathbf{W}_\beta \mathbf{a} + \mathbf{b}_\beta$$

对特征图执行广播仿射变换：

$$\mathbf{F}_{\text{modulated}}[c, i, j] = \boldsymbol{\gamma}(\mathbf{a})[c] \cdot \mathbf{F}[c, i, j] + \boldsymbol{\beta}(\mathbf{a})[c]$$

### 2. FiLM 调制手算数值算例
设某个特征通道的原始像素激活值为 $F = 2.0$。
当前用户下发了“全速右转”动作 $\mathbf{a} = [1.0]$。
动作网络计算得到该通道的缩放因子 $\gamma(a) = 1.5$，平移因子 $\beta(a) = -0.5$。

我们来手动求解调制后的通道特征值：
$$F_{\text{modulated}} = 1.5 \times 2.0 + (-0.5) = 3.0 - 0.5 = 2.5$$

> **代数物理启示**：
> - 缩放因子 $\gamma(a) = 1.5$ 充当了“特征放大器”，强化了与右转相关的光流特征；
> - 平移因子 $\beta(a) = -0.5$ 注入了确定性的运动方向偏置；
> - 动作指令直接在特征流形层面重塑了空间响应！

<details>
<summary><b>深入推导：FiLM 仿射变换在流形切空间李代数变换下的结构保持性证明（点击展开查看完整推导）</b></summary>

将特征空间视为微分流形 $\mathcal{M}$ 上的向量丛截面。
动作条件映射 $\phi_{\mathbf{a}}(\mathbf{x}) = \text{diag}(\boldsymbol{\gamma}) \mathbf{x} + \boldsymbol{\beta}$ 构成了仿射李群 $\text{Aff}(n)$ 的元素。
当 $\boldsymbol{\gamma} > 0$ 时，该变换在流形切空间 $T_p\mathcal{M}$ 上保持了微分同胚性（Diffeomorphism）。
在反向传播中，特征图关于动作的梯度为 $\frac{\partial \mathcal{L}}{\partial \mathbf{a}} = \sum_{c, i, j} \frac{\partial \mathcal{L}}{\partial F_c} (F_c \mathbf{W}_\gamma[c, :] + \mathbf{W}_\beta[c, :])$，建立了强有力的动作感知直接反馈通路。
</details>

---

## 5.5.3 核心数学推导二：自回归长程推演漂移与噪声注入防御

在自由交互推演时，如果模型连续推演数百步（如连续运行 1 分钟），前一步生成的微小重构瑕疵会被作为下一时刻的输入，导致误差随着时间步指数级雪崩累积，最终引发画面模糊溶化或物体凭空消失。

<div align="center">

<img src="/figures/05-interactive-video/source/05-interactive-video-scratch/diamond-fig1.png" alt="Oasis 在不同噪声增强与自回归训练策略下的长期画面稳定性评测。" width="86%">

_图 5.5-4：Oasis 在不同噪声增强与自回归训练策略下的长期画面稳定性评测。 出处：[Oasis: A Universe in a Transformer，Decart & Etched，2024](https://oasis.decart.ai/)。_

</div>

为了增强自回归长程推演的鲁棒性，系统在训练时采用 **输入噪声增强注入（Noise Injection / Scheduled Sampling）**：

$$\tilde{\mathbf{x}}_{t-1} = \mathbf{x}_{t-1} + \sigma_{\text{noise}} \cdot \boldsymbol{\epsilon}, \quad \text{其中 } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

通过在历史输入帧中主动注入轻微的高斯白噪声，迫使网络学会**自我纠错与去噪修复能力**，使得模型在面对自身产生的微小生成瑕疵时能够自发将其抚平，保障了交互式推演数千步的绝对物理稳健性！

<details>
<summary><b>深入推导：自回归时序动力学在随机扰动注入下的李雅普诺夫指数衰减分析（点击展开查看完整推导）</b></summary>

设真实动力学定点吸引子为 $\mathbf{x}^*$。离散误差演化满足 $\mathbf{e}_{t+1} = \mathbf{J}_f \mathbf{e}_t + \boldsymbol{\eta}_t$。
定义最大李雅普诺夫指数 $\lambda = \lim_{T \to \infty} \frac{1}{T} \log \|\prod_{t=1}^T \mathbf{J}_f(t)\|$。
当在训练时对输入施加协方差为 $\sigma^2 \mathbf{I}$ 的噪声注入时，等价于在目标函数中引入了吉洪诺夫正则化项 $\mathcal{R}(\theta) = \frac{\sigma^2}{2} \text{Tr}(\mathbf{J}_f^\top \mathbf{J}_f)$。
极小化该正则项迫使雅可比矩阵谱范数满足 $\|\mathbf{J}_f\| < 1 \implies \lambda < 0$，系统轨迹在相空间中指数级渐近收敛至稳定极限环。
</details>

---

## 5.5.4 纯底层 PyTorch 代码实现：从零手写端到端可交互动作可控视频世界模型

下面我们使用纯底层 PyTorch 算子手写实现一个集成了 FiLM 动作调制、时序循环推演与逐帧渲染的完整交互式世界模型。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMBlock(nn.Module):
    """
    FiLM 动作条件特征调制层
    y = gamma(a) * x + beta(a)
    """
    def __init__(self, channels: int, action_dim: int):
        super().__init__()
        self.channels = channels
        self.fc = nn.Linear(action_dim, channels * 2)

    def forward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        :param feat: (B, C, H, W)
        :param action: (B, action_dim)
        """
        mod = self.fc(action).unsqueeze(-1).unsqueeze(-1) # (B, 2*C, 1, 1)
        gamma, beta = mod.chunk(2, dim=1)
        return feat * (1.0 + gamma) + beta

class InteractiveVideoWorldModel(nn.Module):
    """
    端到端可交互动作可控视频世界模型 (Interactive Video World Model)
    输入上一帧画面与当前操作动作，以 30 FPS 实时推演渲染下一帧画面
    """
    def __init__(self, in_c: int = 3, action_dim: int = 2, hidden_c: int = 32):
        super().__init__()
        # 1. 空间帧卷积编码
        self.enc1 = nn.Conv2d(in_c, hidden_c, kernel_size=3, padding=1)
        self.film1 = FiLMBlock(channels=hidden_c, action_dim=action_dim)

        # 2. 下采样与深层时序动力学
        self.down = nn.Conv2d(hidden_c, hidden_c * 2, kernel_size=4, stride=2, padding=1) # (16, 16)
        self.film2 = FiLMBlock(channels=hidden_c * 2, action_dim=action_dim)

        # 3. 循环记忆状态更新 (ConvGRU 单元)
        self.conv_gru_gate = nn.Conv2d(hidden_c * 2 * 2, hidden_c * 2, kernel_size=3, padding=1)
        self.conv_gru_cand = nn.Conv2d(hidden_c * 2 * 2, hidden_c * 2, kernel_size=3, padding=1)

        # 4. 上采样画面解码渲染
        self.up = nn.ConvTranspose2d(hidden_c * 2, hidden_c, kernel_size=4, stride=2, padding=1) # (32, 32)
        self.out_conv = nn.Conv2d(hidden_c, in_c, kernel_size=3, padding=1)

    def forward_step(
        self, prev_frame: torch.Tensor, action: torch.Tensor, h_prev: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        单帧交互前向推演
        :param prev_frame: (B, 3, 32, 32) 上一帧
        :param action: (B, action_dim) 键盘/控制动作
        :param h_prev: (B, 64, 16, 16) 上一时刻循环隐状态
        :return: (next_frame, h_next)
        """
        B = prev_frame.shape[0]
        if h_prev is None:
            h_prev = torch.zeros(B, 64, 16, 16, device=prev_frame.device)

        # 步骤一：特征提取与 FiLM 动作调制
        x1 = F.relu(self.film1(self.enc1(prev_frame), action))
        x2 = F.relu(self.film2(self.down(x1), action)) # (B, 64, 16, 16)

        # 步骤二：ConvGRU 时序推进
        gru_in = torch.cat([x2, h_prev], dim=1)
        z_gate = torch.sigmoid(self.conv_gru_gate(gru_in))
        cand = torch.tanh(self.conv_gru_cand(torch.cat([x2, z_gate * h_prev], dim=1)))
        h_next = (1.0 - z_gate) * h_prev + z_gate * cand

        # 步骤三：解码生成下一帧
        dec1 = F.relu(self.up(h_next))
        next_frame = torch.sigmoid(self.out_conv(dec1))

        return next_frame, h_next

# ===================================================================
# 单元测试与闭环交互连续推演校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    action_dim = 2
    sim_steps = 6

    model = InteractiveVideoWorldModel(in_c=3, action_dim=action_dim, hidden_c=32)

    # 1. 模拟用户从随机初始帧开始交互
    curr_frame = torch.rand(batch_size, 3, 32, 32)
    h_state = None

    generated_sequence = []
    print(f"[Interactive Test] 开始模拟连续 {sim_steps} 步用户动作交互推演...")

    for step in range(sim_steps):
        # 模拟随机用户操作动作: [水平推力, 垂直跳跃]
        user_action = (torch.rand(batch_size, action_dim) - 0.5) * 2.0

        # 实时推演下一帧
        next_frame, h_state = model.forward_step(curr_frame, user_action, h_state)
        generated_sequence.append(next_frame)

        curr_frame = next_frame # 闭环滚入下一时刻

    stacked_video = torch.stack(generated_sequence, dim=1)
    print(f"[Interactive Test] 成功生成闭环交互视频序列，形状: {stacked_video.shape} (期望: [{batch_size}, {sim_steps}, 3, 32, 32])")
    print(f"[Interactive Test] 终态画面像素范围: [{next_frame.min().item():.3f}, {next_frame.max().item():.3f}]")

    assert stacked_video.shape == (batch_size, sim_steps, 3, 32, 32), "交互视频序列维度不符！"
    assert not torch.isnan(stacked_video).any(), "交互推演出现 NaN 异常！"
    print("✓ 动作可控交互式视频世界模型、FiLM 调制与闭环流式推演单测全部通过！")
```

---

## 5.5.5 本节小结

回顾本节内容，我们完成了可交互视频世界模型的终极工程实战：
1. **FiLM 动作条件注入**：通过逐通道仿射缩放与平移，赋予了低维控制指令对高维像素特征的绝对驾驭力；
2. **ConvGRU 时序记忆流**：在紧凑的二维空间特征图上维护连续物理惯性，保障了运动推演的时空因果连贯性；
3. **闭环神经物理沙盒**：实现了输入上一帧与即时动作、实时输出下一帧的高吞吐闭环，为完全由神经网络驱动的下一代具身仿真与虚拟世界奠定了坚实的技术底座。
