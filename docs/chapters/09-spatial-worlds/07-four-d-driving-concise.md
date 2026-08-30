## 9.7 4D 自动驾驶预测模型的简洁实现

自动驾驶系统在本质上是一个在极其复杂的动态时空环境中做出连续决策的智能体。在早期的研究中，研究人员通常将这一任务分解为多个独立的子模块：感知（识别车辆与行人）、预测（估计其他物体的未来轨迹）以及规划（生成自车的行驶路线）。然而，这种级联式架构不可避免地会导致误差累积，并且在处理不规则物体或未见过的场景时显得捉襟见肘。

为了解决这一问题，学术界逐渐转向端到端的“世界模型”（World Models）架构，其中最具代表性的方向便是4D（3D空间 + 1D时间）占据网络预测（Occupancy Forecasting）。自 FIERY `[Hu et al., 2021]` 首次在鸟瞰图（BEV）空间中引入时空预测以来，以及后续 UniAD `[Hu et al., 2023]` 等工作对自动驾驶全栈端到端架构的探索，基于4D空间的未来状态预测已经成为自动驾驶领域的核心技术。在这一范式中，模型不再仅仅输出离散物体的边界框，而是直接预测整个三维物理空间在未来多个时间步内的状态演化。

在本节中，我们将从最基础的物理运动学原理出发，逐步推导出4D时空世界模型的核心方程，并使用深度学习框架提供一个高度模块化、且包含丰富张量操作细节的简洁实现。

### 9.7.1 从基础物理学到时空状态转移

在高中物理中，我们学习过匀加速直线运动的最基本规律。假设一个物体在时刻 $t$ 的位置为 $p_t$，速度为 $v_t$，加速度为 $a_t$，那么在极小的时间间隔 $\Delta t$ 后，其下一时刻 $t+1$ 的位置可以表示为：

$$p_{t+1} = p_t + v_t \Delta t + \frac{1}{2} a_t \Delta t^2$$

如果我们将其中的速度和加速度视为物体内在的“隐状态”（Hidden State），而位置视为观测到的“显状态”，那么 :eqref:`eq_kinematics_scalar` 实际上描述了一个动态系统随时间演化的过程。

在自动驾驶的4D预测任务中，我们面对的不再是单一质点，而是一个被离散化为三维体素（Voxel）的物理空间。设三维空间的尺寸为 $X \times Y \times Z$。对于空间中的每一个体素坐标 $(x, y, z)$，它不仅包含当前是否被占据的状态，还隐含了该位置上物体的运动趋势（即高维语义特征）。

因此，我们需要将标量方程 :eqref:`eq_kinematics_scalar` 推广到高维张量空间。设 $\mathbf{S}_t \in \mathbb{R}^{C \times X \times Y \times Z}$ 为时刻 $t$ 的三维空间特征图，其中 $C$ 为特征通道数。系统在时刻 $t$ 的状态转移不再是简单的线性叠加，而是由一个非线性的深度神经网络 $f_\theta$ 来建模：

$$\mathbf{S}_{t+1} = f_\theta(\mathbf{S}_t, \mathbf{a}_t)$$

其中，$\mathbf{a}_t$ 为自车在时刻 $t$ 采取的动作（如转向、加速），$\theta$ 为网络参数。这里的 $f_\theta$ 需要能够同时捕获空间上的几何结构以及时间上的运动连续性。

### 9.7.2 核心机制：时空卷积与注意力机制

为了实现 :eqref:`eq_tensor_state_transition` 中的状态转移，模型需要同时理解“物体在哪里（空间）”以及“物体要去哪里（时间）”。

在最复杂的自动驾驶场景中，车辆和行人的交互会产生高度非线性的时空演变。> 我们可以将其类比为气象台预测台风：气象雷达扫描出的并非孤立的点，而是一个随着时间翻滚、变形、甚至分裂的三维云图矩阵；为了预测下一时刻某片云的形状，我们不仅需要看这一时刻这片云周围的空气湿度（空间关联），还必须追溯过去几十分钟内这片云是如何随风移动的（时间关联）。这种同时跨越空间和时间的双重聚合，正是时空世界模型的核心。

在数学上，我们可以使用三维卷积（3D Convolution）结合循环神经网络（如 GRU），或者直接使用时空自注意力机制（Spatiotemporal Attention）来实现。在本节的简洁实现中，我们将采用一种经典的卷积门控循环单元（ConvGRU）变体。不同于标准 GRU 处理一维向量，3D ConvGRU 的隐状态和门控信号都是四维张量（Batch Size, Channel, X, Y, Z），其更新方程如下：

首先，计算更新门（Update Gate）$\mathbf{Z}_t$ 和重置门（Reset Gate）$\mathbf{R}_t$。对于输入特征 $\mathbf{X}_t$ 和上一时刻的隐状态 $\mathbf{H}_{t-1}$：

$$\mathbf{Z}_t = \sigma(\mathbf{W}_z * [\mathbf{X}_t, \mathbf{H}_{t-1}] + \mathbf{b}_z)$$

$$\mathbf{R}_t = \sigma(\mathbf{W}_r * [\mathbf{X}_t, \mathbf{H}_{t-1}] + \mathbf{b}_r)$$

在这里，$*$ 代表三维空间卷积操作，$[\cdot, \cdot]$ 代表在通道维度上的拼接（Concatenation），$\sigma$ 为 Sigmoid 激活函数。接着，我们计算候选隐状态 $\tilde{\mathbf{H}}_t$：

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{W}_h * [\mathbf{X}_t, (\mathbf{R}_t \odot \mathbf{H}_{t-1})] + \mathbf{b}_h)$$

其中，$\odot$ 为逐元素乘法（Hadamard Product）。最后，通过更新门进行隐状态的平滑过渡：

$$\mathbf{H}_t = (1 - \mathbf{Z}_t) \odot \mathbf{H}_{t-1} + \mathbf{Z}_t \odot \tilde{\mathbf{H}}_t$$

方程 :eqref:`eq_final_hidden` 极其优雅地将历史记忆（$\mathbf{H}_{t-1}$）与当前观测（$\tilde{\mathbf{H}}_t$）在三维空间中逐体素地融合，从而完成了真正的4D时空演化。

### 9.7.3 网络架构的简洁实现

现在，我们将上述严格的数学推导转化为代码实现。我们将构建一个名为 `FourDPredictor` 的模型。为了保持代码的简洁性，我们将三维空间的 Z 轴高度维度进行压缩，退化为高度通道（Height Channels），从而在实现时可以使用更为高效的 2D 卷积来近似 3D 卷积的运算能力，这也是目前业界基于 BEV（Bird's-Eye-View）世界模型的主流做法。

(**首先，我们导入必要的库并定义时空转移模块。**)

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

class ConvGRUCell(nn.Module):
    """一个简化的 2D ConvGRU 单元，用于时空状态演化"""
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # 我们将输入和隐状态在通道维度拼接，因此输入通道数为二者之和
        self.conv_z = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_r = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        
    def forward(self, x, h_prev):
        # x 形状: (batch_size, input_channels, height, width)
        # h_prev 形状: (batch_size, hidden_channels, height, width)
        
        # 在通道维度上进行拼接
        x_and_h = torch.cat([x, h_prev], dim=1)
        
        # 严格对应方程 9.7.3 和 9.7.4
        z = torch.sigmoid(self.conv_z(x_and_h))
        r = torch.sigmoid(self.conv_r(x_and_h))
        
        # 严格对应方程 9.7.5
        x_and_r_h = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_h(x_and_r_h))
        
        # 严格对应方程 9.7.6
        h_current = (1.0 - z) * h_prev + z * h_tilde
        return h_current
```

在具备了核心的时空记忆单元之后，我们可以构建完整的前向预测模型。该模型接收过去的感知特征序列，并自回归地生成未来 $T$ 个时间步的占据栅格（Occupancy Grid）概率图。

(**接下来，我们实现完整的 4D 预测世界模型。**)

```{.python .input}
#@tab pytorch
class FourDWorldModel(nn.Module):
    """4D 自动驾驶世界模型的简洁实现"""
    def __init__(self, in_channels, hidden_channels, out_channels, future_steps):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.future_steps = future_steps
        
        # 特征编码器：将感知输入压缩到隐状态空间
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        )
        
        # 时空状态转移模块
        self.dynamics_model = ConvGRUCell(hidden_channels, hidden_channels)
        
        # 解码器：将隐状态投影回三维占据概率（通道代表高度或不同类别的占据情况）
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )
        
    def forward(self, past_features):
        """
        past_features 形状: (batch_size, past_steps, in_channels, H, W)
        返回 futures 形状: (batch_size, future_steps, out_channels, H, W)
        """
        batch_size, past_steps, _, height, width = past_features.shape
        device = past_features.device
        
        # 初始化隐状态 h_0 为全零张量
        h_t = torch.zeros((batch_size, self.hidden_channels, height, width), device=device)
        
        # 1. 历史状态预热 (Burn-in)
        for t in range(past_steps):
            x_t = self.encoder(past_features[:, t, ...])
            h_t = self.dynamics_model(x_t, h_t)
            
        # 2. 未来状态自回归预测 (Rollout)
        future_predictions = []
        # 在预测阶段，我们假设没有新的观测输入，使用零向量或特定的动作向量作为驱动
        dummy_action = torch.zeros_like(x_t)
        
        for t in range(self.future_steps):
            # 使用动力学模型在隐空间中推演未来
            h_t = self.dynamics_model(dummy_action, h_t)
            
            # 解码当前时间步的隐状态为物理空间占据预测
            pred_t = self.decoder(h_t)
            future_predictions.append(pred_t)
            
        # 在时间维度上堆叠预测结果
        return torch.stack(future_predictions, dim=1)
```

在上述代码中，历史状态预热阶段使模型充分吸收了过去的运动趋势，而未来状态的自回归生成则完全依赖于 `dynamics_model` 在隐状态中对物理规律的内在推演。这正是“世界模型”的精髓所在。

### 9.7.4 损失函数与训练过程

在自动驾驶中，占据网格（Occupancy Grid）通常是一个二值分布：空间中的体素要么被占据（车辆、建筑），要么为空（可行驶区域）。因此，我们自然地选用二元交叉熵损失（Binary Cross Entropy, BCE）来约束模型在每一时刻、每一个空间位置上的预测概率。

假设真实的未来多帧占据网格张量为 $\mathbf{Y} \in \{0, 1\}^{B \times T \times C \times H \times W}$，模型的预测逻辑值（Logits）为 $\hat{\mathbf{Y}}$，则整个时空预测损失 $\mathcal{L}$ 为：

$$\mathcal{L} = \frac{1}{B \cdot T \cdot C \cdot H \cdot W} \sum_{b,t,c,h,w} \text{BCE}(\sigma(\hat{Y}_{b,t,c,h,w}), Y_{b,t,c,h,w})$$

(**下面，我们构造合成数据并演示一轮前向传播和损失计算。**)

```{.python .input}
#@tab pytorch
# 超参数设置
batch_size, past_steps, future_steps = 2, 3, 4
channels, height, width = 16, 64, 64
out_channels = 4 # 例如：代表 4 种不同的高度层级或语义类别

# 实例化模型与优化器
model = FourDWorldModel(channels, 32, out_channels, future_steps)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# 构造模拟的过去特征序列和真实的未来占据网格标签
past_features = torch.randn(batch_size, past_steps, channels, height, width)
# 真实标签通常为 0 或 1，这里使用随机二值张量模拟
future_labels = torch.randint(0, 2, (batch_size, future_steps, out_channels, height, width)).float()

# 前向传播
future_logits = model(past_features)

# 计算时空预测损失，对应方程 9.7.7
loss = criterion(future_logits, future_labels)

print(f"输入特征形状: {past_features.shape}")
print(f"预测输出形状: {future_logits.shape}")
print(f"时空预测损失: {loss.item():.4f}")

# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

通过这一简单的框架，模型不仅能够在二维的鸟瞰平面上，更能在垂直的高度维度和向前延展的时间轴上，建立起一幅动态的四维物理世界蓝图。

### 9.7.5 小结

* 4D 自动驾驶预测要求模型不仅理解当前的三维空间结构，还必须对未来的时间演化进行精准推断。
* 我们通过类比基础运动学，导出了世界模型中隐状态转移的一般性数学表达。
* ConvGRU 是一种天然契合时空动力学建模的网络结构，其复杂的门控机制允许模型逐体素地遗忘冗余信息并更新运动趋势。
* 在实际实现中，出于显存和算力的考量，通常会将 Z 轴投影至通道维度，在 2D BEV 空间内完成准 3D 的时空张量运算。

### 9.7.6 练习

1. 在 `FourDWorldModel` 的预测阶段，我们使用了全零的 `dummy_action`。如果我们能够获得自车规划的未来动作指令（例如，方向盘转角和加速度），你将如何修改网络结构和代码以将自车动作注入到动态模型中？（**提示**：考虑使用多层感知机（MLP）将动作向量扩展到与隐状态相同的空间维度并相加或拼接。）
2. 方程 :eqref:`eq_bce_loss` 中的 BCE 损失平等地对待了每一个空间网格。然而在现实中，远处的预测误差可能比近处的误差更可接受。你能否给出一个带空间衰减权重的修正版 BCE 损失公式？
3. 阅读 FIERY `[Hu et al., 2021]` 论文中关于前瞻预测（Future Instance Prediction）的部分，其在预测占据网格之外，还预测了实例级别的中心度（Centerness）和偏移量（Offset）。在我们的简洁代码中，需要如何扩展 `decoder` 来同时输出这三个分支？
