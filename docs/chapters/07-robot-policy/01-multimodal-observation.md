# 7.1 机器人多模态感知与观测建模

在探索机器人如何自主规划与执行物理动作之前，我们必须首先解决一个最基础的物理问题：**机器人究竟是如何“观察”并“理解”周围物理世界的？**

与部署在云端服务器上的纯文本或纯图像大模型不同，一台真实的机器人置身于充满光影变化、重力加速度与物体接触碰撞的真实物理世界之中。它不仅需要通过安装在头部或机械臂末端的摄像头“看”到工作台上的物体，还需要通过关节内部的高精度传感器“感应”自身机械臂当前的弯曲角度与运动速度，甚至通过指尖的压力阵列“触摸”物体的软硬与滑脱趋势。

这一将来自异构物理传感器的数据流转化为策略网络可处理特征的过程，被称为**多模态观测建模（Multimodal Observation Modeling）**。

<div align="center">

<img src="/figures/07-robot-policy/source/01-multimodal-observation/levine-fig1.png" alt="相机画面与机械臂构型共同进入视觉运动策略，输出直接驱动真实机器人。" width="86%">

_图 7.1-1：相机画面与机械臂构型共同进入视觉运动策略，输出直接驱动真实机器人。 出处：[End-to-End Training of Deep Visuomotor Policies，Sergey Levine et al.，2016](https://arxiv.org/abs/1504.00702)。_

</div>

---

## 7.1.1 物理与生理基石：人类感觉器官与机器人多模态传感器

要构建机器人的感知系统，我们首先需要从人类的感觉生理学与经典物理测量原理中汲取灵感。

### 1. 生物感知系统的多通道协同
在生物学中，人类之所以能够闭着眼睛准确摸到自己的鼻尖，或者在黑暗中稳稳端起水杯，依赖于两套高度协同的感觉网络：
- **外部感受（Exteroception）**：眼睛的视网膜感光细胞捕捉环境中的可见光光子（提供物体几何轮廓、色彩与空间相对距离）；
- **本体感受（Proprioception）**：肌肉内部的**肌梭（Muscle Spindle）**感应肌肉纤维的伸缩长度与拉伸速率，关节囊中的**高尔基腱器官（Golgi Tendon Organ）**测量肌腱承受的张力大小，内耳前庭器官测量头部的倾斜与重力加速度。

正是视网膜光流与深层肌腱受力感知的毫秒级协同融合，构成了人类小脑实时调控动作的感知基石。

### 2. 机器人世界中的多模态传感器映射
对应到物理机器人系统中，我们拥有三种核心的物理传感器数据流：
1. **外部视觉张量（Visual RGB Images）**：安装于环境操作台上方（Eye-to-Hand，全局上帝视角）或机械臂手腕处（Eye-in-Hand，随动局部视角）的相机，以 $30\text{ Hz}$ 采集空间三维色彩阵列 $I_t \in \mathbb{R}^{3 \times H \times W}$；
2. **关节本体感觉向量（Joint Proprioception）**：安装于每个电机轴端的光电编码器（Encoder），以 $1000\text{ Hz}$ 高频采集各关节当前的实际旋转角度 $\mathbf{q}_t \in \mathbb{R}^N$ 与角速度 $\dot{\mathbf{q}}_t \in \mathbb{R}^N$；
3. **触觉与末端力矩（Tactile & Wrench Sensors）**：安装于夹爪指尖的凝胶触觉传感器（如 GelSight）或六维腕力传感器，实时测量法向正压力 $F_z$ 与切向摩擦力剪切分布。

<div align="center">

<img src="/figures/07-robot-policy/source/01-multimodal-observation/rt2-fig1.png" alt="RT-2 把机器人动作表示为语言 token，连接视觉语言推理与低层控制。" width="86%">

_图 7.1-2：RT-2 把机器人动作表示为语言 token，连接视觉语言推理与低层控制。 出处：[RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control，Anthony Brohan et al.，2023](https://arxiv.org/abs/2307.15818)。_

</div>

---

## 7.1.2 核心数学推导一：视觉与本体感觉的特征对齐与投影

高维图像（如 $224 \times 224 \times 3 \approx 15\text{ 万}$ 个像素数值）与低维本体感觉向量（如 7 个浮点数）在信息密度与数学结构上存在巨大的不对称性。

### 1. 视觉图像块嵌入与线性投影
在现代视觉架构（如 Vision Transformer, ViT）中，输入的二维图像 $I \in \mathbb{R}^{3 \times H \times W}$ 被划分为 $N_p$ 个互不重叠的小方块（Patches，尺寸通常为 $P \times P = 14 \times 14$）。图像块的总数量为：

$$N_p = \frac{H \times W}{P^2}$$

每个图像块被展平并通过线性投影矩阵 $\mathbf{W}_{\text{vis}}$ 映射为 $D$ 维的连续视觉词元（Visual Token）：

$$\mathbf{v}_i = \mathbf{W}_{\text{vis}} \text{vec}(\text{Patch}_i) + \mathbf{b}_{\text{vis}} \in \mathbb{R}^D, \quad \forall i \in \{1, \dots, N_p\}$$

### 2. 本体感觉状态的多层感知机（MLP）特征对齐
对于包含 $N$ 个关节角度的本体感觉向量 $\mathbf{q} \in \mathbb{R}^N$，如果直接与图像词元拼接，微弱的几个标量很容易被数以百计的视觉词元淹没。

系统通过两层带有 GELU 激活函数的多层感知机，将本体感觉升维并映射至相同的特征维度 $D$：

$$\mathbf{z}_{\text{prop}} = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{q} + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^D$$

> **公式符号逐一拆解**：
> - $\mathbf{W}_1 \in \mathbb{R}^{D_{\text{mid}} \times N}, \mathbf{b}_1 \in \mathbb{R}^{D_{\text{mid}}}$：第一层特征升维权重与偏置（例如 $D_{\text{mid}} = 128$）；
> - $\text{GELU}(x) = x \Phi(x)$：高斯误差线性单元激活函数；
> - $\mathbf{W}_2 \in \mathbb{R}^{D \times D_{\text{mid}}}, \mathbf{b}_2 \in \mathbb{R}^D$：对齐至 Transformer 隐藏层主维度的投影权重；
> - $\mathbf{z}_{\text{prop}} \in \mathbb{R}^D$：生成的单条本体感觉词元。

**手算代入算例**：
设某单臂机械臂拥有 $N = 7$ 个关节，当前角度为 $\mathbf{q} = [0.0, 0.5, -0.2, 0.0, 0.8, -0.4, 1.0]^\top$。
第一层线性层权重偏置经过矩阵乘法后得到中间特征，经过 GELU 非线性激活，最终输出一个与图像词元长度完全一致的 256 维向量 $\mathbf{z}_{\text{prop}}$。
此时，本体感觉不再是微不足道的 7 个标量，而是拥有与图像块等量齐观表达能力的结构化语义词元！

<details>
<summary><b>深入推导：多模态异构特征跨通道互信息最大化与几何流形对齐证明（点击展开查看完整推导）</b></summary>

设视觉特征流形为 $\mathcal{M}_{\text{vis}} \subset \mathbb{R}^{N_p \times D}$，本体感觉流形为 $\mathcal{M}_{\text{prop}} \subset \mathbb{R}^D$。
多模态联合表征学习的目标是最大化两者与真实物理动作 $\mathbf{a}$ 的互信息下界：
$$I(\mathbf{Z}_{\text{vis}}, \mathbf{z}_{\text{prop}}; \mathbf{a}) = H(\mathbf{a}) - H(\mathbf{a} \mid \mathbf{Z}_{\text{vis}}, \mathbf{z}_{\text{prop}})$$
当仅有视觉输入时，由于相机视角遮挡（Occlusion），后验熵 $H(\mathbf{a} \mid \mathbf{Z}_{\text{vis}})$ 较高；引入本体感觉后，条件熵由于马尔可夫决策过程的充分统计量性质严格递减：
$$H(\mathbf{a} \mid \mathbf{Z}_{\text{vis}}, \mathbf{z}_{\text{prop}}) \le H(\mathbf{a} \mid \mathbf{Z}_{\text{vis}})$$
两层 MLP 投影充当了李群流形上的微分同胚映射，保证了低维欧氏关节空间向高维注意力超球面的拓扑平滑嵌入。
</details>

---

## 7.1.3 核心数学推导二：跨注意力机制（Cross-Attention）与视觉聚焦

如何让机器人知道“根据当前的关节姿态，应该重点看画面的哪个局部”？系统采用了经典的**跨模态交叉注意力机制（Cross-Attention）**。

<div align="center">

<img src="/figures/07-robot-policy/source/01-multimodal-observation/rt1-fig13.png" alt="RT-1 的注意力可视化展示不同层和头如何聚焦任务相关图像区域。" width="86%">

_图 7.1-3：RT-1 的注意力可视化展示不同层和头如何聚焦任务相关图像区域。 出处：[RT-1: Robotics Transformer for Real-World Control at Scale，Anthony Brohan et al.，2022](https://arxiv.org/abs/2212.06817)。_

</div>

<div align="center">

<img src="/figures/07-robot-policy/latex/01-multimodal-observation/cross-attention-row-softmax.png" alt="单个本体查询沿视觉 patch 维做行 Softmax，再汇聚 Value" width="86%">

_图 7.1-4：单个本体查询沿视觉 patch 维做行 Softmax，再汇聚 Value。_

</div>

### 1. 查询向量（Query）与键值对（Key, Value）
- **Query（查询向量 $\mathbf{Q}$）**：由本体感觉词元 $\mathbf{z}_{\text{prop}}$ 线性变换得到，代表机器人当前的内部状态发出的询问：“我现在的夹爪位置，对应的目标物体在哪里？”
  $$\mathbf{Q} = \mathbf{z}_{\text{prop}} \mathbf{W}_Q \in \mathbb{R}^{1 \times D}$$
- **Key 与 Value（键矩阵 $\mathbf{K}$ 与值矩阵 $\mathbf{V}$）**：由所有视觉图像块 $\mathbf{V}_{\text{seq}} \in \mathbb{R}^{N_p \times D}$ 线性映射得到：
  $$\mathbf{K} = \mathbf{V}_{\text{seq}} \mathbf{W}_K \in \mathbb{R}^{N_p \times D}, \quad \mathbf{V} = \mathbf{V}_{\text{seq}} \mathbf{W}_V \in \mathbb{R}^{N_p \times D}$$

### 2. 缩放点积注意力与行归一化
计算本体 Query 与每一个视觉 Patch Key 的点积相似度，除以缩放因子 $\sqrt{D}$，并通过 Softmax 归一化为概率权重分布：

$$\mathbf{A} = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{D}} \right) \in \mathbb{R}^{1 \times N_p}$$

最终的跨模态融合特征为所有视觉 Value 向量的加权和：

$$\mathbf{z}_{\text{fused}} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{1 \times D}$$

> **初等代数直觉**：
> 点积 $\mathbf{Q} \cdot \mathbf{K}_i^\top$ 衡量了向量之间的夹角余弦相似度。如果第 $i$ 个图像块正是机械臂夹爪当前接触的把手，内积数值就会极大，Softmax 分配给它的权重 $A_i \approx 0.9$；其余背景区域权重趋近于 0。这使策略在复杂混乱的操作台上能够精准“凝视”关键物体。

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
