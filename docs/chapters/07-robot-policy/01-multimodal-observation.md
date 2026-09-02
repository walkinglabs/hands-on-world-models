# 7.1 机器人多模态感知与观测建模

在具身智能（Embodied AI）与机器人学的研究中，智能体并不是生存在纯粹抽象的文字或数学符号中，而是作为一个物理实体，沉浸在由光线、重力、接触力与空间几何交织而成的真实世界里。

正如人类在拿取桌上的一杯热咖啡时，不仅需要双眼观察咖啡杯的位置与边缘，还需要手臂肌肉与关节处的本体感觉感知手臂的伸展角度，并通过指尖的触觉感受杯壁的温度与滑移阻力。

对于机器人而言，单一模态的数据往往存在严重的物理局限性：
- 视觉传感器容易受到强光反射、阴影遮挡或动态模糊的干扰；
- 内部关节编码器虽然精准，却完全无法获知外部物理世界的空间结构。

如何将高维度的**视觉观测（Exteroception）**、低维度的**本体状态（Proprioception）**以及时序动作历史在统一的潜在数学空间中进行对齐与融合，构成了构建具身世界模型与策略控制的第一道基石。

<div align="center">

<img src="/figures/07-robot-policy/source/01-multimodal-observation/levine-fig1.png" alt="基于视觉的端到端机器人操作：从多视角摄像头和关节传感器到低级电机力矩。" width="86%">

_图 7.1-1：基于视觉的端到端机器人操作：从多视角摄像头和关节传感器到低级电机力矩。 出处：[End-to-End Training of Deep Visuomotor Policies，Sergey Levine et al.，2016](https://arxiv.org/abs/1504.00702)。_

</div>

---

## 7.1.1 物理与生理基石：外感知与本体感受的生物映射

要深刻理解多模态融合的数学建模，我们首先需要从经典物理力学与生物感知系统的对应关系讲起。

在经典力学与生理学中，机器人的感知通道清晰地划分为两大阵营：

### 1. 外感受器（Exteroception）：构建外部几何世界
人类的视网膜包含上亿个感光细胞，能以高达 $60\text{ Hz}$ 以上的频率捕捉外界环境的光子流分布。在机器人硬件上，这对应着：
- **RGB 相机**：提供稠密的二维色彩与纹理信息（表征“物体是什么”）；
- **深度相机（RGB-D）与激光雷达（LiDAR）**：利用飞行时间法（Time-of-Flight, ToF）或双目立体视觉，直接测量物体表面距离相机的物理深度 $Z$（表征“物体在哪里”）。

### 2. 本体感受器（Proprioception）：感知自躯体运动状态
人类的肌腱与肌肉中分布着大量的肌梭（Muscle Spindles）与高尔基腱器官（Golgi Tendon Organs），用于感知肌肉的拉伸长度与收缩张力。在机械臂与人形机器人上，这对应着：
- **关节光学编码器（Encoders）**：以 $0.001^\circ$ 的超高精度实时读取机械臂各关节的旋转角度 $\mathbf{q} \in \mathbb{R}^n$ 与旋转角速度 $\dot{\mathbf{q}}$；
- **末端六维力/力矩传感器（F/T Sensor）**：实时监测夹爪在接触物体时受到的沿三轴的作用力 $(F_x, F_y, F_z)$ 及绕三轴的力矩 $(\tau_x, \tau_y, \tau_z)$。

如果机器人仅仅依靠视觉，当夹爪伸入狭窄的抽屉时，视觉视线会被机械臂自身严重遮挡；若没有本体感觉的即时反馈，机器人将无法判断夹爪是否已经触底，极易发生机械过载导致电机烧毁。

<div align="center">

<img src="/figures/07-robot-policy/source/01-multimodal-observation/rt2-fig1.png" alt="RT-2 将高分辨率图像打散为图像补丁词元，与文本指令和本体状态在多模态骨干网中统一融合。" width="86%">

_图 7.1-2：RT-2 将高分辨率图像打散为图像补丁词元，与文本指令和本体状态在多模态骨干网中统一融合。 出处：[RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control，Anthony Brohan et al.，2023](https://arxiv.org/abs/2307.15818)。_

</div>

---

## 7.1.2 核心数学推导一：异构模态的高维嵌入与时空对齐

视觉图像属于高维连续网格数据（例如一张 $224 \times 224 \times 3$ 的 RGB 图像包含 $150,528$ 个浮点数），而关节角度仅是一个极其紧凑的 7 维向量（$\mathbf{q} \in \mathbb{R}^7$）。如果直接将这两个维度相差悬殊的张量强行拼接，低维的关节物理量在反向传播时会被高维图像的巨大梯度彻底淹没。

因此，系统必须先将不同维度的异构物理量投影到**相同维度的隐藏语义流形 $\mathbb{R}^D$**（通常取隐藏层维度 $D = 512$ 或 $D = 768$）。

### 1. 视觉特征线性分块投影（ViT Patch Projection）
设输入图像为 $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$。我们将其均匀划分为 $N_p = \frac{H \cdot W}{P^2}$ 个大小为 $P \times P$ 的局部小图像块（Patches，例如 $P = 16$）。每一个图像块被展平为向量 $\mathbf{x}_p^{(i)} \in \mathbb{R}^{P^2 C}$。

通过可学习的线性投影矩阵 $\mathbf{E}_{\text{vis}} \in \mathbb{R}^{(P^2 C) \times D}$ 并叠加空间位置编码 $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{N_p \times D}$，图像被转化为一系列规范的视觉词元序列：

$$\mathbf{v}_i = \mathbf{x}_p^{(i)} \mathbf{E}_{\text{vis}} + \mathbf{e}_{\text{pos}}^{(i)}, \quad i \in \{1, 2, \dots, N_p\}$$

### 2. 本体感觉的非线性升维映射（Proprioceptive MLP）
对于 7 自由度机械臂的关节角度向量 $\mathbf{q}_t \in \mathbb{R}^7$ 与夹爪开合度 $g_t \in [0, 1]$，我们组合为本体状态向量 $\mathbf{s}_t = [\mathbf{q}_t^\top, g_t]^\top \in \mathbb{R}^8$。

利用带有激活函数的两层多层感知机（MLP）将其升维投影为单个本体词元 $\mathbf{z}_{\text{prop}} \in \mathbb{R}^D$：

$$\mathbf{z}_{\text{prop}} = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{s}_t + \mathbf{b}_1) + \mathbf{b}_2$$

通过这一步骤，原本信息密度悬殊的视觉与关节状态被严格归一化为同等长度的 $D$ 维特征向量，为后续注意力机制中的对称点积交互铺平了道路。

---

## 7.1.3 核心数学推导二：跨模态交叉注意力（Cross-Attention）融合机制

在得到视觉词元序列 $\mathbf{V}_{\text{seq}} = [\mathbf{v}_1, \dots, \mathbf{v}_{N_p}]^\top \in \mathbb{R}^{N_p \times D}$ 与本体状态词元 $\mathbf{z}_{\text{prop}} \in \mathbb{R}^{1 \times D}$ 后，如何让机械臂的当前姿态去“寻找”图像中最相关的物理交互区域？

系统采用了**跨模态交叉注意力机制（Cross-Attention Mechanism）**。

<div align="center">

<img src="/figures/07-robot-policy/source/01-multimodal-observation/rt1-fig13.png" alt="RT-1 的注意力可视化展示不同层和头如何聚焦任务相关图像区域。" width="86%">

_图 7.1-3：RT-1 的注意力可视化展示不同层和头如何聚焦任务相关图像区域。 出处：[RT-1: Robotics Transformer for Real-World Control at Scale，Anthony Brohan et al.，2022](https://arxiv.org/abs/2212.06817)。_

</div>

<div align="center">

<img src="/figures/07-robot-policy/latex/01-multimodal-observation/cross-attention-row-softmax.png" alt="单个本体查询沿视觉 patch 维做行 Softmax，再汇聚 Value" width="86%">

_图 7.1-4：单个本体查询沿视觉 patch 维做行 Softmax，再汇聚 Value。_

</div>

### 1. 四步严密交叉注意力推导流程
#### 步骤一：生成查询向量（Query）与键值对（Key, Value）
- **Query（查询向量 $\mathbf{Q}$）**：由本体感觉词元 $\mathbf{z}_{\text{prop}}$ 线性变换得到，代表机器人根据自身当前关节姿态发出的询问：
  $$\mathbf{Q} = \mathbf{z}_{\text{prop}} \mathbf{W}_Q \in \mathbb{R}^{1 \times D}$$
- **Key 与 Value（键矩阵 $\mathbf{K}$ 与值矩阵 $\mathbf{V}$）**：由所有视觉图像块 $\mathbf{V}_{\text{seq}} \in \mathbb{R}^{N_p \times D}$ 线性映射得到：
  $$\mathbf{K} = \mathbf{V}_{\text{seq}} \mathbf{W}_K \in \mathbb{R}^{N_p \times D}, \quad \mathbf{V} = \mathbf{V}_{\text{seq}} \mathbf{W}_V \in \mathbb{R}^{N_p \times D}$$

#### 步骤二：缩放点积相似度计算
计算本体 Query 与每一个视觉 Patch Key 的点积内积，并除以维度缩放因子 $\sqrt{D}$ 以稳定方差：
$$S_j = \frac{\mathbf{Q} \mathbf{K}_j^\top}{\sqrt{D}} = \frac{1}{\sqrt{D}} \sum_{d=1}^D Q_d K_{j, d}, \quad j \in \{1, 2, \dots, N_p\}$$

#### 步骤三：Softmax 行概率归一化
$$\mathbf{A}_j = \frac{\exp(S_j)}{\sum_{k=1}^{N_p} \exp(S_k)} \in (0, 1), \quad \text{满足 } \sum_{j=1}^{N_p} \mathbf{A}_j = 1$$

#### 步骤四：多模态加权值特征汇聚
最终的跨模态融合特征为所有视觉 Value 向量按注意力权重的凸组合求和：
$$\mathbf{z}_{\text{fused}} = \mathbf{A} \mathbf{V} = \sum_{j=1}^{N_p} \mathbf{A}_j \mathbf{V}_j \in \mathbb{R}^{1 \times D}$$

### 2. 交叉注意力详细手算代入算例
设特征隐藏维度 $D = 2$。
- 本体感觉 Query 向量为 $\mathbf{Q} = [1.0, 2.0]$；
- 视觉端仅有两个图像块：$\mathbf{K}_1 = [1.0, 2.0], \mathbf{V}_1 = [10.0, 0.0]$（图像块 1 为把手位置）；$\mathbf{K}_2 = [0.0, 1.0], \mathbf{V}_2 = [0.0, 10.0]$（图像块 2 为桌角背景）。

我们来执行完整的数值代入计算：
1. **计算点积内积**：
   $$S_1 = \frac{\mathbf{Q} \cdot \mathbf{K}_1}{\sqrt{2}} = \frac{1.0 \times 1.0 + 2.0 \times 2.0}{\sqrt{2}} = \frac{1.0 + 4.0}{1.414} = \frac{5.0}{1.414} \approx 3.536$$
   $$S_2 = \frac{\mathbf{Q} \cdot \mathbf{K}_2}{\sqrt{2}} = \frac{1.0 \times 0.0 + 2.0 \times 1.0}{\sqrt{2}} = \frac{0.0 + 2.0}{1.414} = \frac{2.0}{1.414} \approx 1.414$$
2. **计算 Softmax 权重**：
   $$\exp(S_1) = \exp(3.536) \approx 34.33, \quad \exp(S_2) = \exp(1.414) \approx 4.11$$
   $$A_1 = \frac{34.33}{34.33 + 4.11} = \frac{34.33}{38.44} \approx 0.893 \quad (89.3\%)$$
   $$A_2 = \frac{4.11}{38.44} \approx 0.107 \quad (10.7\%)$$
3. **加权汇聚 Value 特征**：
   $$\mathbf{z}_{\text{fused}} = 0.893 \times [10.0, 0.0] + 0.107 \times [0.0, 10.0] = [8.93, 1.07]$$

初等代数的直观计算生动证实：本体状态主动将高达 **$89.3\%$** 的注意力权重精准聚焦在把手图像块上，使策略在复杂混乱的操作台上能够精准“凝视”关键物理交互部件！

<details>
<summary><b>深入推导：缩放点积注意力方差稳定性证明与 Softmax 梯度反向传播（点击展开查看完整推导）</b></summary>

假设查询与键向量的分量服从独立同分布的标准正态分布 $q_i, k_i \sim \mathcal{N}(0, 1)$。
两个 $D$ 维向量的点积为 $S = \sum_{j=1}^D q_j k_j$。
根据均值与方差性质：
$$\mathbb{E}[S] = 0, \quad \text{Var}(S) = \sum_{j=1}^D \text{Var}(q_j k_j) = D \times 1 = D$$
若不除以 $\sqrt{D}$，当特征维度 $D$ 较大（如 $D = 512$）时，点积方差高达 512，导致 Softmax 函数进入极度饱和区（梯度几乎处处为 0，引发梯度消失）。
引入缩放因子后：
$$\text{Var}\left(\frac{S}{\sqrt{D}}\right) = \frac{1}{D} \text{Var}(S) = 1$$
方差严格稳定为 1，确保了反向传播时梯度的健康流动。
</details>

---

## 7.1.4 纯底层 PyTorch 代码实现：多模态感知对齐与交叉注意力引擎

下面我们使用纯底层 PyTorch 张量算子实现一个结构完整的机器人多模态观测特征提取与交叉注意力融合模块。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalObservationEncoder(nn.Module):
    """
    机器人多模态观测编码器 (Multimodal Observation Encoder)
    融合高维 RGB 视觉图像与低维关节本体感觉向量。
    """
    def __init__(self, num_joints: int = 7, d_model: int = 128, img_channels: int = 3):
        super().__init__()
        self.d_model = d_model

        # 1. 视觉卷积特征提取器 (将 64x64 图像提取为 16 个 4x4 的局部特征补丁)
        self.vision_backbone = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1), # (B, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),            # (B, 64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, d_model, kernel_size=4, stride=4),                  # (B, d_model, 4, 4)
            nn.ReLU()
        )

        # 2. 关节本体感觉 MLP 投影器
        self.proprio_projector = nn.Sequential(
            nn.Linear(num_joints, 64),
            nn.GELU(),
            nn.Linear(64, d_model)
        )

        # 3. 跨模态注意力投影矩阵
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # 4. 融合输出全连接层
        self.fusion_head = nn.Linear(d_model * 2, d_model)

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向计算
        :param image: (B, 3, H, W) RGB 视觉张量
        :param proprio: (B, num_joints) 关节角度向量
        :return: (fused_feature, attention_weights)
        """
        batch_size = image.size(0)

        # 1. 提取视觉特征补丁序列: (B, d_model, 4, 4) -> (B, 16, d_model)
        vis_map = self.vision_backbone(image)
        vis_seq = vis_map.flatten(2).transpose(1, 2) # (B, N_patches=16, d_model)

        # 2. 提取本体感觉词元: (B, num_joints) -> (B, 1, d_model)
        prop_emb = self.proprio_projector(proprio).unsqueeze(1)

        # 3. 交叉注意力计算: Q 来自本体，K, V 来自视觉
        q = self.w_q(prop_emb) # (B, 1, d_model)
        k = self.w_k(vis_seq)  # (B, 16, d_model)
        v = self.w_v(vis_seq)  # (B, 16, d_model)

        # 计算缩放点积注意力权重
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_model ** 0.5) # (B, 1, 16)
        attn_weights = F.softmax(scores, dim=-1)

        # 汇聚视觉特征: (B, 1, 16) * (B, 16, d_model) -> (B, 1, d_model)
        attended_vis = torch.bmm(attn_weights, v).squeeze(1) # (B, d_model)

        # 4. 拼接本体特征与汇聚视觉特征
        fused = self.fusion_head(torch.cat([prop_emb.squeeze(1), attended_vis], dim=-1))
        return fused, attn_weights

# ===================================================================
# 单元测试与注意力权重分布校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    num_joints = 7
    img_h, img_w = 64, 64
    d_model = 128

    encoder = MultimodalObservationEncoder(num_joints=num_joints, d_model=d_model)
    encoder.eval()

    dummy_images = torch.randn(batch_size, 3, img_h, img_w)
    dummy_joints = torch.randn(batch_size, num_joints)

    with torch.no_grad():
        fused_out, attn_weights = encoder(dummy_images, dummy_joints)

    print(f"[Observation Test] 输入图像张量形状: {dummy_images.shape}")
    print(f"[Observation Test] 输入关节张量形状: {dummy_joints.shape}")
    print(f"[Observation Test] 融合表征向量形状: {fused_out.shape}")
    print(f"[Observation Test] 视觉注意力权重形状: {attn_weights.shape}")
    print(f"[Observation Test] 单样本注意力权重和: {attn_weights[0].sum().item():.4f}")

    assert fused_out.shape == (batch_size, d_model), "融合输出维度不符！"
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, 1)), "注意力概率归一化不满足！"
    print("✓ 多模态观测编码器与交叉注意力单测全部通过！")
```

---

## 7.1.5 本节小结

回顾本节内容，我们建立了机器人多模态感知与观测对齐的完整认知框架：
1. **生物感知映射**：机器人的外部视觉与内部关节编码器对应着人类的视网膜与本体肌腱感受器，二者的融合是闭环动作控制的前提；
2. **异构特征对齐**：高维图像被切分为离散词元序列，低维关节向量经由非线性 MLP 映射升维至统一隐藏层维度；
3. **交叉注意力聚焦**：通过缩放点积注意力，本体感觉 Query 能够实时在所有图像块中检索最相关的物理交互区域，实现“目标导向”的主动空间感知。
