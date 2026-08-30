# 6.4 动作条件的 JEPA（Action-conditional JEPA）
:label:sec_action_jepa

在深度学习的早期发展中，研究者们曾试图通过像素级的重构来理解世界。然而，现实世界包含了大量不可预测且往往无关紧要的细节——例如微风中树叶的随机摆动，或是背景中随机的纹理变化。在这一背景下，Yann LeCun 在其经典论文《迈向自主机器智能之路》（*A Path Towards Autonomous Machine Intelligence*, [LeCun, 2022]）中提出了联合嵌入预测架构（Joint Embedding Predictive Architecture, JEPA）。与传统的自编码器不同，JEPA 放弃了在像素空间进行重构，转而在抽象的特征（隐变量）空间中进行预测，从而强制模型学习世界的高阶语义。

然而，原始的 JEPA 更多聚焦于对静态空间特征或被动视频流的补全，它缺乏与世界交互的关键要素：**动作（Action）**。为了构建一个真正的“世界模型”（World Model），智能体必须能够回答反事实的问题：“如果我采取了动作 $A$ 而不是动作 $B$，世界将会发生怎样的改变？”。在此驱动下，动作条件的 JEPA（Action-conditional JEPA）应运而生。它不仅保留了 JEPA 在抽象空间预测的优势，更将动作变量显式地注入预测器中，使其成为智能体在复杂环境中进行规划和决策的强大引擎。

本节我们将从最基础的物理运动学原理出发，严谨地推导动作条件 JEPA 的数学架构，深入分析其防止表征坍塌（Representation Collapse）的内在机制，并给出详尽的工业级代码实现。

## 6.4.1 从基础运动学到隐空间的非线性演化

为了透彻理解动作条件下的预测机制，我们首先回到高中物理中最基础的匀速直线运动模型。假设一个物体在时刻 $t$ 的位置为 $x_t \in \mathbb{R}$，在时间间隔 $\Delta t$ 内，它受到了速度为 $v_t$ 的动作输入。那么，它在时刻 $t+1$ 的位置可以通过以下标量方程精确描述：

$$x_{t+1} = x_t + v_t \cdot \Delta t$$
:eqlabel:eq_kinematics_scalar

在这个简单的物理系统中，状态 $x_t$ 是完全可观测的（例如物体在坐标轴上的位置），而 $v_t$ 则是我们主动施加的“动作”。如果我们将其推广到多维空间，状态变为向量 $\mathbf{x}_t \in \mathbb{R}^D$，动作变为控制向量 $\mathbf{a}_t \in \mathbb{R}^A$，我们便得到了现代控制理论中的线性离散时间状态方程：

$$\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{a}_t$$
:eqlabel:eq_kinematics_vector

其中矩阵 $\mathbf{A}$ 和 $\mathbf{B}$ 描述了系统固有的物理规律。然而，在自动驾驶或机器人控制等真实场景中，我们无法直接获取像“坐标”这样干净的状态向量 $\mathbf{x}_t$。我们能获取的，往往是包含数百万像素的高维图像或高频传感器阵列数据（例如 $\mathbf{o}_t \in \mathbb{R}^{H \times W \times C}$）。更严重的是，这些原始观测数据内部的演化规律是高度非线性的。

为了解决高维观测数据的预测难题，动作条件 JEPA 采用了**降维与抽象**的核心思想。它不直接去预测复杂的 $\mathbf{o}_{t+1}$，而是引入一个**编码器（Encoder）** $E_\theta$，将高维的观测数据映射到一个低维的隐特征空间（Latent Space）中：

$$\mathbf{s}_t = E_\theta(\mathbf{o}_t), \quad \mathbf{s}_t \in \mathbb{R}^d$$
:eqlabel:eq_encoder_state

在这个抽象的隐空间中，我们再引入一个由神经网络参数化的非线性**预测器（Predictor）** $P_\phi$，利用当前时刻的隐状态 $\mathbf{s}_t$ 和动作 $\mathbf{a}_t$ 来预测下一时刻的隐状态：

$$\hat{\mathbf{s}}_{t+1} = P_\phi(\mathbf{s}_t, \mathbf{a}_t)$$
:eqlabel:eq_predictor_state

通过这种方式，动作条件 JEPA 将复杂的“像素级演化”转换为了纯粹的“语义级演化”，极大地降低了预测环境动态的难度。

## 6.4.2 架构解析与严密的数学表达

动作条件 JEPA 的整体架构由三个核心神经网络组件构成。为了保证数学上的严谨性，我们将精确定义每个组件的张量输入与输出。

1. **上下文编码器（Context Encoder） $E_\theta$**：
   负责处理当前时刻 $t$ 的观测数据 $\mathbf{o}_t$。参数为 $\theta$。其输出被称为上下文表征（Context Representation） $\mathbf{s}_t \in \mathbb{R}^d$。
   
2. **目标编码器（Target Encoder） $E_{\bar{\theta}}$**：
   负责处理未来时刻 $t+1$ 的真实观测数据 $\mathbf{o}_{t+1}$，以生成预测的“基准真相”（Ground Truth）。为了防止表征坍塌，其参数 $\bar{\theta}$ 并非通过梯度下降更新，而是上下文编码器参数 $\theta$ 的指数移动平均（Exponential Moving Average, EMA）。其输出被称为目标表征（Target Representation） $\mathbf{y}_{t+1} \in \mathbb{R}^d$。

3. **动作条件预测器（Action-conditional Predictor） $P_\phi$**：
   接收上下文表征 $\mathbf{s}_t$ 和动作向量 $\mathbf{a}_t \in \mathbb{R}^k$，预测未来状态。参数为 $\phi$。输出为预测表征 $\hat{\mathbf{y}}_{t+1} \in \mathbb{R}^d$。

在给定的批次大小 $B$（Batch Size）下，我们可以将损失函数定义为预测表征 $\hat{\mathbf{Y}}$ 与目标表征 $\mathbf{Y}$ 之间的均方误差（MSE）。设批次中的第 $i$ 个样本的第 $j$ 个特征维度分别为 $\hat{y}_{t+1}^{(i,j)}$ 和 $y_{t+1}^{(i,j)}$，标量形式的损失函数展开如下：

$$\mathcal{L}_{JEPA}(\theta, \phi) = \frac{1}{B \cdot d} \sum_{i=1}^B \sum_{j=1}^d \left( \hat{y}_{t+1}^{(i,j)} - y_{t+1}^{(i,j)} \right)^2$$
:eqlabel:eq_jepa_loss_scalar

将其写为紧凑的矩阵形式（即 Frobenius 范数的平方）：

$$\mathcal{L}_{JEPA}(\theta, \phi) = \frac{1}{B \cdot d} \left\| P_\phi(E_\theta(\mathbf{O}_t), \mathbf{A}_t) - E_{\bar{\theta}}(\mathbf{O}_{t+1}) \right\|_F^2$$
:eqlabel:eq_jepa_loss_matrix

请严格注意，在反向传播计算梯度时，梯度**只流向**参数 $\theta$ 和 $\phi$。目标编码器的参数 $\bar{\theta}$ 被视为常数（Stop-Gradient），其更新严格遵循以下 EMA 规则：

$$\bar{\theta} \leftarrow \tau \bar{\theta} + (1 - \tau) \theta$$
:eqlabel:eq_ema_update

其中 $\tau \in [0, 1)$ 是动量系数（Momentum），通常取值非常接近 $1$（如 $0.99$ 或 $0.996$）。

## 6.4.3 为什么需要 EMA：表征坍塌的几何分析

初学者经常会问：为什么不能让目标编码器和上下文编码器共享参数，并同时更新它们？

假设我们令 $\bar{\theta} = \theta$，并在优化过程中同时对它们求梯度以最小化 :eqref:eq_jepa_loss_matrix。在这种情况下，神经网络会寻找一条“捷径”来完美地将损失降为零，即：
**令所有的权重全部坍缩为零，或者映射到一个不随输入变化的常数向量。**

当 $E_\theta(\mathbf{O}) = \mathbf{0}$ 且 $P_\phi(\cdot) = \mathbf{0}$ 时，无论输入什么图像和动作，预测值和目标值永远为 $\mathbf{0}$，损失函数完美等于 $0$。这就是自监督学习中臭名昭著的**表征坍塌（Representation Collapse）**。

引入不对称的 EMA 机制，从几何动力学的角度来看，相当于在优化空间中为目标函数设置了一个“缓慢移动的锚点（Anchor）”。
(1) 预测器 $P_\phi$ 被迫努力去拟合目标编码器当前产生的特征分布。
(2) 由于 $\bar{\theta}$ 接收不到直接使两者靠近的梯度，目标编码器不会主动向预测器“妥协”（即不会向零点坍缩）。
(3) 上下文编码器 $\theta$ 虽然接收到了梯度，但它的目的是为了提取能够预测未来目标特征的信息，而不是变成零。
通过这种不对称的动态平衡，模型被迫在隐空间中保留对环境演化至关重要的信息，同时忽略无法预测的噪声。

> [!NOTE]
> 在遇到极其复杂的自监督学习架构时，我们可以将这种不对称性类比为“老师与学生”的指导机制：目标编码器是老师，它基于过去的经验（EMA 权重）给出现阶段的“标准答案”；上下文编码器和预测器是学生，必须努力根据当前的线索去猜测老师的答案。老师不会因为学生做错题就改变答案，从而保证了知识体系不坍塌。

## 6.4.4 多步预测的自回归展开

真实的规划往往需要预测未来多步的状态。动作条件 JEPA 可以自然地扩展为自回归（Autoregressive）模式。给定初始观测 $\mathbf{o}_t$ 以及一个动作序列 $\mathbf{a}_t, \mathbf{a}_{t+1}, \dots, \mathbf{a}_{t+K-1}$，我们可以递归地展开预测：

1. $\hat{\mathbf{s}}_{t+1} = P_\phi(E_\theta(\mathbf{o}_t), \mathbf{a}_t)$
2. $\hat{\mathbf{s}}_{t+2} = P_\phi(\hat{\mathbf{s}}_{t+1}, \mathbf{a}_{t+1})$
3. ...
4. $\hat{\mathbf{s}}_{t+K} = P_\phi(\hat{\mathbf{s}}_{t+K-1}, \mathbf{a}_{t+K-1})$

相应的，总损失函数将是这 $K$ 步预测误差的累加：

$$\mathcal{L}_{multi} = \sum_{k=1}^K \lambda_k \left\| \hat{\mathbf{s}}_{t+k} - E_{\bar{\theta}}(\mathbf{o}_{t+k}) \right\|_F^2$$
:eqlabel:eq_multistep_loss

其中 $\lambda_k$ 为不同时间步的权重系数。通过多步展开训练，预测器被迫学习长期的环境动态，而不仅仅是下一步的细微变化。

## 6.4.5 代码实现

下面我们以工业级代码的标准，使用 PyTorch 和 TensorFlow 分别实现一个多层感知机（MLP）版本的动作条件 JEPA 核心架构。我们将明确处理梯度的截断（Stop-Gradient）以及 EMA 参数的更新。

(**首先，我们定义基础的编码器和预测器模块。**)

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
import copy

class Encoder(nn.Module):
    """一个简化的基于 MLP 的编码器，用于提取隐状态。"""
    def __init__(self, obs_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        # x: (Batch_Size, obs_dim)
        return self.net(x)

class ActionPredictor(nn.Module):
    """动作条件预测器：结合当前隐状态和动作，预测下一步隐状态。"""
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        # 在这里，我们将隐状态和动作在特征维度上拼接 (Concatenation)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, state, action):
        # state: (Batch_Size, latent_dim)
        # action: (Batch_Size, action_dim)
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models

class Encoder(tf.keras.Model):
    """一个简化的基于 MLP 的编码器，用于提取隐状态。"""
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.net = models.Sequential([
            layers.Dense(hidden_dim),
            layers.LayerNormalization(),
            layers.ReLU(),
            layers.Dense(latent_dim)
        ])

    def call(self, x):
        # x: (Batch_Size, obs_dim)
        return self.net(x)

class ActionPredictor(tf.keras.Model):
    """动作条件预测器：结合当前隐状态和动作，预测下一步隐状态。"""
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.net = models.Sequential([
            layers.Dense(hidden_dim),
            layers.LayerNormalization(),
            layers.ReLU(),
            layers.Dense(latent_dim)
        ])

    def call(self, state, action):
        # state: (Batch_Size, latent_dim)
        # action: (Batch_Size, action_dim)
        x = tf.concat([state, action], axis=-1)
        return self.net(x)
```

(**接下来，我们组装完整的 Action-conditional JEPA 模型，并实现 EMA 更新逻辑。**)

```{.python .input}
#@tab pytorch
class ActionConditionalJEPA(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=256, hidden_dim=512, ema_tau=0.99):
        super().__init__()
        self.ema_tau = ema_tau
        
        # 1. 上下文编码器
        self.context_encoder = Encoder(obs_dim, hidden_dim, latent_dim)
        
        # 2. 动作条件预测器
        self.predictor = ActionPredictor(latent_dim, action_dim, hidden_dim)
        
        # 3. 目标编码器 (与上下文编码器结构完全一致，但参数独立)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        
        # 目标编码器的参数不参与梯度更新，冻结它们
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """执行目标编码器的指数移动平均 (EMA) 更新"""
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            # \bar{\theta} = \tau * \bar{\theta} + (1 - \tau) * \theta
            param_k.data.mul_(self.ema_tau).add_(param_q.data, alpha=1.0 - self.ema_tau)

    def forward(self, obs_t, action_t, obs_t_plus_1):
        """
        前向传播计算单步预测损失
        """
        # (1) 提取当前上下文隐状态: (Batch_Size, latent_dim)
        s_t = self.context_encoder(obs_t)
        
        # (2) 预测下一时刻的隐状态: (Batch_Size, latent_dim)
        s_t_plus_1_pred = self.predictor(s_t, action_t)
        
        # (3) 使用目标编码器获取真实的下一时刻目标隐状态
        with torch.no_grad():
            s_t_plus_1_target = self.target_encoder(obs_t_plus_1)
            # 严格确保在此处进行 Stop-Gradient (尽管 no_grad 已经保证了这一点)
            s_t_plus_1_target = s_t_plus_1_target.detach()
            
        # (4) 计算 MSE 损失
        # 这里使用了平滑且严谨的均方误差公式
        loss = nn.functional.mse_loss(s_t_plus_1_pred, s_t_plus_1_target)
        
        return loss, s_t_plus_1_pred
```

```{.python .input}
#@tab tensorflow
class ActionConditionalJEPA(tf.keras.Model):
    def __init__(self, action_dim, latent_dim=256, hidden_dim=512, ema_tau=0.99):
        super().__init__()
        self.ema_tau = ema_tau
        
        # 1. 上下文编码器
        self.context_encoder = Encoder(hidden_dim, latent_dim)
        
        # 2. 动作条件预测器
        self.predictor = ActionPredictor(hidden_dim, latent_dim)
        
        # 3. 目标编码器
        self.target_encoder = Encoder(hidden_dim, latent_dim)
        # 初始化时需要调用一次以建立权重
        dummy_input = tf.zeros((1, 100)) # 假设 obs_dim=100 的占位符，可根据实际调整
        self.context_encoder(dummy_input)
        self.target_encoder(dummy_input)
        
        # 将上下文编码器的权重硬拷贝给目标编码器
        self.target_encoder.set_weights(self.context_encoder.get_weights())
        
        # 目标编码器的参数不参与梯度更新
        self.target_encoder.trainable = False

    def update_target_encoder(self):
        """执行目标编码器的指数移动平均 (EMA) 更新"""
        for param_q, param_k in zip(self.context_encoder.weights, self.target_encoder.weights):
            # \bar{\theta} = \tau * \bar{\theta} + (1 - \tau) * \theta
            param_k.assign(self.ema_tau * param_k + (1.0 - self.ema_tau) * param_q)

    def call(self, inputs):
        """
        前向传播计算单步预测损失
        inputs: (obs_t, action_t, obs_t_plus_1)
        """
        obs_t, action_t, obs_t_plus_1 = inputs
        
        # (1) 提取当前上下文隐状态
        s_t = self.context_encoder(obs_t)
        
        # (2) 预测下一时刻的隐状态
        s_t_plus_1_pred = self.predictor(s_t, action_t)
        
        # (3) 获取真实目标并应用 tf.stop_gradient
        s_t_plus_1_target = self.target_encoder(obs_t_plus_1)
        s_t_plus_1_target = tf.stop_gradient(s_t_plus_1_target)
            
        # (4) 计算 MSE 损失
        loss = tf.reduce_mean(tf.square(s_t_plus_1_pred - s_t_plus_1_target))
        
        return loss, s_t_plus_1_pred
```

(**为了观察模型如何避免表征坍塌，我们编写一个简短的训练循环，并监控损失和隐状态的方差。如果模型坍塌，隐状态的方差会迅速趋近于零。**)

```{.python .input}
#@tab pytorch
# 初始化模型和优化器
obs_dim, action_dim = 128, 4
jepa = ActionConditionalJEPA(obs_dim=obs_dim, action_dim=action_dim)
optimizer = torch.optim.Adam(jepa.parameters(), lr=1e-3)

# 模拟一个批次的随机观测和动作数据
batch_size = 32
obs_t = torch.randn(batch_size, obs_dim)
action_t = torch.randn(batch_size, action_dim)
# 模拟环境的演化：下一时刻的观测，这里加入了一些噪声来模拟不可预测性
obs_t_plus_1 = obs_t + 0.1 * torch.randn(batch_size, obs_dim)

for step in range(50):
    optimizer.zero_grad()
    
    # 计算损失
    loss, pred_state = jepa(obs_t, action_t, obs_t_plus_1)
    
    # 反向传播并更新 \theta 和 \phi
    loss.backward()
    optimizer.step()
    
    # 【关键步骤】手动更新目标编码器 \bar{\theta}
    jepa.update_target_encoder()
    
    if (step + 1) % 10 == 0:
        # 监控隐特征维度的方差。方差远离 0 说明没有发生表征坍塌。
        state_variance = pred_state.var(dim=0).mean().item()
        print(f"Step {step+1}: Loss = {loss.item():.4f}, 隐特征均方差 = {state_variance:.4f}")
```

## 6.4.6 小结与实践指导

在构建大规模世界模型时，动作条件 JEPA 提供了一种极具数学优雅性的解决方案。通过在隐空间中进行预测，并利用不对称的 EMA 架构冻结目标梯度，它巧妙地在“学习世界动态”与“忽略无关噪声”之间找到了平衡。

在实际训练中，你需要注意以下几点：
1. **EMA 动量参数（$\tau$）的选择**：如果 $\tau$ 太小，目标网络更新过快，很容易陷入表征坍塌；如果 $\tau$ 太大（非常接近 $1.0$），目标网络更新过于缓慢，导致训练收敛极慢。一种常见的策略是采用“余弦退火（Cosine Annealing）”，在训练过程中逐渐将 $\tau$ 从 $0.99$ 提升至 $1.0$。
2. **多步预测的稳定性**：在执行多步自回归展开时，由于每一次预测都建立在前一次的输出之上，误差会呈指数级累积。通常需要在预测器中加入 Layer Normalization，并在训练早期限制预测的步数 $K$。

## 练习

1. **数学推导**：如果我们将损失函数中的均方误差（MSE）替换为余弦相似度损失（Cosine Similarity Loss），目标编码器的 EMA 机制依然是必须的吗？为什么？
   *提示：考虑余弦相似度优化空间下是否存在平凡解（即无论输入如何，输出总是固定的单位向量）。*
2. **代码扩展**：修改上述 PyTorch 代码中的 `ActionConditionalJEPA.forward` 方法，使其能够接受一个形状为 `(Batch_Size, K, action_dim)` 的动作序列，并返回累加的 $K$ 步自回归损失（参考 :eqref:eq_multistep_loss）。
   *提示：你需要在一个循环中多次调用 `self.predictor`，并将前一步的预测 `s_pred` 作为下一步的输入状态。注意只在最终时刻计算或者在每一步都计算与真实目标的损失。*

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
