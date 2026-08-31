# 灵巧手与灵巧操作

平行夹爪拿起杯子时，主要控制开合宽度和夹持力。若要在手内旋转杯子，多根手指必须轮流建立和释放接触，并协调各关节的运动。后一类问题称为**灵巧操作**（Dexterous Manipulation）：机器人利用多指末端执行器完成抓取、转动和重新定位等接触丰富的动作。

<div align="center">

<img src="/figures/07-robot-policy/source/02-dexterous-manipulation/rubik-fig1.png" alt="Shadow Hand 在真实系统中连续重定向魔方，呈现灵巧操作的接触丰富性。" width="86%">

_图 7.2-1：Shadow Hand 在真实系统中连续重定向魔方，呈现灵巧操作的接触丰富性。 出处：[Solving Rubik's Cube with a Robot Hand，OpenAI et al.，2019](https://arxiv.org/abs/1910.07113)。_

</div>

## 历史脉络与学术背景

早期工业抓取多在结构化环境中使用夹爪。Salisbury 和 Craig 在 1982 年描述了一种 3 指、9 自由度的关节手及其控制设计，为多指手研究提供了一个早期实例 [[Salisbury & Craig, 1982]](https://doi.org/10.1177/027836498200100102)。后来的 Shadow Hand 等平台提供更多可动关节，也带来了更高维的控制与感知问题。

灵巧手具有高维动作空间、频繁接触切换和复杂动力学，因此控制与数据收集都很困难。Rajeswaran 等人展示了用无模型深度强化学习训练多指手完成转笔、开门和锤击等任务 [[Rajeswaran et al., 2017]](https://arxiv.org/abs/1709.10087)。OpenAI 随后结合强化学习与自动域随机化，让 Shadow Hand 在真实系统上完成魔方复原 [[Akkaya et al., 2019]](https://arxiv.org/abs/1910.07113)。这些结果证明了学习方法在论文所测试任务中的可行性，但并不意味着传统模型式控制已被完全取代。

<div align="center">

<img src="/figures/07-robot-policy/source/02-dexterous-manipulation/raj-fig1.png" alt="多指手完成转笔、开门、锤击等不同技能，说明任务和接触模式的多样性。" width="86%">

_图 7.2-2：多指手完成转笔、开门、锤击等不同技能，说明任务和接触模式的多样性。 出处：[Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations，Aravind Rajeswaran et al.，2017](https://arxiv.org/abs/1709.10087)。_

</div>

## 物理基础：从滑动摩擦到多点接触

先从两指夹持的受力分析开始。

假设一个木块静止在水平桌面上，我们用两根手指分别从左右两侧夹住它，然后将其提起。此时木块受到的力包括：重力 $G$、两根手指提供的正压力 $N_1$ 和 $N_2$、以及向上的静摩擦力 $f_1$ 和 $f_2$。
为了让木块不掉落，根据摩擦定律，必须满足：

$$f_1 + f_2 \ge G$$

同时，静摩擦力受限于正压力与静摩擦因数 $\mu$：

$$f_i \le \mu N_i, \quad i \in \{1, 2\}$$

这个不等式只说明当前重力方向上的抗滑条件。更强的**力封闭**（Force Closure）要求接触力能够抵抗任意方向的小外部力和力矩；判断它通常需要把所有接触点的力通过抓取映射组合到物体的六维力旋量空间中。因此，“两侧夹紧且不下滑”并不自动等于三维力封闭。

多指灵巧手包含更多关节和接触切换。在三维空间中，每个接触点的可行力受到**摩擦锥（Friction Cone）**约束。设接触点 $k$ 的单位法向量为 $\mathbf{n}_k$（$\|\mathbf{n}_k\|_2=1$），接触力为 $\mathbf{f}_k$；为了不滑动，接触力需要位于由摩擦系数 $\mu$ 决定的圆锥体内。

用向量不等式可以严格地表示为：

$$ \sqrt{|\mathbf{f}_k|^2 - (\mathbf{n}_k^\top \mathbf{f}_k)^2} \le \mu (\mathbf{n}_k^\top \mathbf{f}_k)
$$

<div align="center">

<img src="/figures/07-robot-policy/latex/02-dexterous-manipulation/friction-cone-decomposition.png" alt="接触力分解为法向投影与切向分量，并与摩擦锥边界比较" width="86%">

_图 7.2-3：法向投影给出正压力，剩余切向分量不超过其 μ 倍时，接触力才落在摩擦锥内。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

在这个公式中，$\mathbf{n}_k^\top \mathbf{f}_k$ 表示接触力在法线方向的投影大小（即正压力），而左侧的根式则计算了接触力在切平面上的分量大小（即摩擦力大小）。这个几何约束意味着，灵巧手的每一个指尖都必须精确控制施加力的方向和大小。一旦任何一个手指施力偏差，接触点就会在表面打滑（滑动摩擦取代静摩擦），导致物体意外旋转或脱落。这便是灵巧操作在动力学层面极难控制的根本原因之一。

## 状态空间与维度灾难

接下来，我们将视角从物理力学转移到机器人的控制理论。要用深度强化学习或模仿学习来控制灵巧手，我们必须首先定义系统的**状态（State）**。

对于一个有 $N$ 个自由度的灵巧手（例如 Shadow Hand 的 $N=24$），其本体感觉（Proprioception）状态 $\mathbf{q} \in \mathbb{R}^N$ 描述了所有关节的当前角度。其运动学状态不仅包括位置 $\mathbf{q}$，还包括关节角速度 $\dot{\mathbf{q}} \in \mathbb{R}^N$。因此，仅仅是灵巧手本身的运动学状态维度就已经达到了 $2N$。

但这还不够。灵巧操作的目的是操纵物体。假设目标物体是一个刚体，其位姿在三维空间中需要 6 个维度来描述（3 个平移坐标，加上 3 个欧拉角或 4 个四元数分量）。设物体的位姿为 $\mathbf{p} \in \mathbb{R}^7$ （使用四元数），线速度和角速度为 $\mathbf{v} \in \mathbb{R}^6$。

如果把手和物体放在同一个状态中，并暂时假设这些量都能准确测得，状态向量 $\mathbf{s}$ 可以写成：

$$\mathbf{s} = \left[ \mathbf{q}^\top, \dot{\mathbf{q}}^\top, \mathbf{p}^\top, \mathbf{v}^\top \right]^\top$$

当 $N=24$ 时，这个结构化状态已有 $2N+7+6=61$ 个标量。现实系统通常还不能直接获得精确的 $\mathbf{p}$ 和 $\mathbf{v}$，需要从图像、触觉或历史观测中估计；困难由此从有限维控制扩展到部分可观测的状态估计。

## 策略网络与多模态感知

视觉提供物体与手指的相对位置，本体感觉提供关节状态，触觉则直接反映接触是否稳定。策略网络需要把这些信号对齐到同一控制时刻；只做特征拼接并不能解决传感器延迟和缺失观测问题。

假设在每一个时间步 $t$，机器人获取到当前视角的图像 $I_t \in \mathbb{R}^{3 \times H \times W}$，以及灵巧手的关节状态 $q_t \in \mathbb{R}^{N}$。我们的目标是学习一个确定性策略函数 $\pi_\theta$，它输出下一步的关节动作 $a_t \in \mathbb{R}^N$（通常是目标关节角度，由底层的PD控制器转化为力矩）：

$$a_t = \pi_\theta(I_t, q_t)$$

为了处理这种多模态输入，我们通常会构建一个双流（Two-Stream）网络。视觉流通过卷积神经网络（CNN）或视觉变换器（Vision Transformer, ViT）将高维图像降维为一个紧凑的视觉特征向量 $\mathbf{z}_{vis} \in \mathbb{R}^D$。本体感觉流则可能直接通过一个多层感知机（MLP）提取特征 $\mathbf{z}_{prop} \in \mathbb{R}^{D'}$。随后，两部分特征被拼接（Concatenation），送入最终的动作输出网络。

下面我们将用代码详细实现这样一个多模态策略网络。

下面实现一个仅包含图像与关节状态的最小策略网络。它用于说明张量接口，不代表完整的触觉控制系统。

```python
import torch
from torch import nn

class DexterousPolicy(nn.Module):
    def __init__(self, num_joints=24, action_dim=24, visual_feature_dim=128):
        """
        初始化灵巧操作策略网络。
        假定输入为 64x64 的 RGB 图像和 num_joints 维度的关节状态。
        """
        super().__init__()

        # 视觉编码器：经典的简单卷积网络架构
        # 输入维度：(Batch, 3, 64, 64)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, visual_feature_dim),
            nn.ReLU()
        )

        # 本觉感知编码器：简单的MLP
        # 输入维度：(Batch, num_joints)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(num_joints, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 融合与输出层
        # 输入维度：视觉特征和本体特征拼接 (visual_feature_dim + 64)
        self.action_head = nn.Sequential(
            nn.Linear(visual_feature_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh() # 限制动作输出在 [-1, 1] 区间
        )

    def forward(self, image, proprioception):
        """
        前向传播函数。
        image: 张量维度 (B, 3, H, W)
        proprioception: 张量维度 (B, num_joints)
        """
        # 提取视觉特征，维度 (B, visual_feature_dim)
        vis_feat = self.visual_encoder(image)

        # 提取本体特征，维度 (B, 64)
        prop_feat = self.proprio_encoder(proprioception)

        # 在特征维度进行拼接，维度 (B, visual_feature_dim + 64)
        fused_feat = torch.cat([vis_feat, prop_feat], dim=1)

        # 输出动作，维度 (B, action_dim)
        action = self.action_head(fused_feat)

        return action
```

在上面的代码中，值得注意的是我们使用了 `Tanh()` 作为最后一层的激活函数，将输出严格限制在 $[-1, 1]$ 之间。在实际操作中，这代表了归一化后的关节角度控制量。在下发给机器人底层控制器时，我们再将其反归一化（Denormalize）到每个关节真实的机械限位（Joint Limits）区间内。

## 从单一技能到泛化

上述网络只给出了策略接口。要把仿真中训练的策略迁移到真实手上，还要处理质量、摩擦系数、执行器延迟和相机位姿等差异。

一种常用方法是**域随机化**（Domain Randomization）：训练时在合理范围内改变摩擦、质量、延迟和视觉参数，使策略不能依赖单一仿真配置。它可以提高对参数变化的容忍度，但范围设置过窄会漏掉真实差异，过宽也可能让学习变得困难。

<div align="center">

<img src="/figures/07-robot-policy/source/02-dexterous-manipulation/rubik-fig10.png" alt="ADR 闭环根据边界环境表现自动扩张随机化分布。" width="86%">

_图 7.2-4：ADR 闭环根据边界环境表现自动扩张随机化分布。 出处：[Solving Rubik's Cube with a Robot Hand，OpenAI et al.，2019](https://arxiv.org/abs/1910.07113)。_

</div>

## 小结

- **灵巧操作**同时涉及多点接触、摩擦约束和接触切换，模型式控制与学习式策略各有适用范围。
- 灵巧系统的状态空间维度庞大，融合高维视觉图像与低维本体感觉信号是设计策略网络的关键挑战。
- 从仿真迁移到真实系统时，域随机化是一种常见工具，但仍需配合系统辨识、真实数据校准和闭环安全测试。

$$
$$
