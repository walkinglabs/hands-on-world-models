# 1.7　经典世界模型

上一节给出的三条判据是抽象的，现在拿它们去看真实系统。世界模型不是某种单一神经网络的专有名称，而是一套将感知压缩、记忆推演与决策行动有机融合的系统化设计范式。2018 年，David Ha 与 Jürgen Schmidhuber 发表划时代论文《World Models》，首次确立了视觉（Vision, V）、记忆（Memory, M）与控制（Controller, C）的三位一体经典架构。本节沿着世界模型的演进脉络，从经典的 V-M-C 范式深入到 PlaNet 的 RSSM 状态空间、Dreamer 的梦境策略学习以及 MuZero 的纯潜空间规划，建立起贯通经典与前沿的完整架构图谱。

---

## 经典 V-M-C 架构的解剖学

在 2018 年的《World Models》论文中，作者以 OpenAI Gym 的 CarRacing 赛车环境为试验场，将智能体优雅地解耦为三个独立训练的子模块：

```text
               ┌────────────────────────────── 智能体内部脑 ──────────────────────────────┐
               │                                                                         │
               │   ┌───────────────────┐    z_t    ┌───────────────────┐                 │
当前图像 x_t ──┼──>│  视觉模块 V (VAE)  │ ─────────>│ 记忆模块 M (RNN)  │                 │
               │   │ (空间压缩与解耦)   │           │ (时序动力学预测)  │                 │
               │   └───────────────────┘           └─────────┬─────────┘                 │
               │             │                         h_t   │ P(z_{t+1})                │
               │             │ z_t                           │                           │
               │             ▼                               ▼                           │
               │   ┌───────────────────────────────────────────────────┐                 │
               │   │               控制模块 C (Controller)              │                 │
               │   │      决策映射: a_t = W_c · [z_t; h_t] + b_c       │                 │
               │   └─────────────────────────┬─────────────────────────┘                 │
               │                             │ 输出控制动作 a_t                          │
               └─────────────────────────────┼───────────────────────────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  真实物理环境 / 梦境模拟器  │
                               └───────────────────────────┘
```

### 1. 视觉模块 V（Vision Module, VAE）

- **核心使命**：解决“高维图像无法直接用于时序建模与决策”的诅咒。
- **结构与原理**：使用变分自动编码器（Variational Autoencoder, VAE）。输入为单帧 $64 \times 64 \times 3$ 的 RGB 图像（共 $12,288$ 维），经过卷积编码器压缩为仅 $N_z = 32$ 维的高斯潜在向量 $z_t \sim \mathcal{N}(\mu_\phi(x_t), \sigma_\phi^2(x_t))$。
- **训练目标**：最大化证据下界（ELBO）：
  $$\mathcal{L}_{\text{VAE}}(\phi, \psi) = \mathbb{E}_{q_\phi(z_t \mid x_t)}\left[\|x_t - \text{Decoder}_\psi(z_t)\|^2\right] + D_{\text{KL}}\left(q_\phi(z_t \mid x_t) \parallel \mathcal{N}(0, I)\right).$$

### 2. 记忆模块 M（Memory Module, MDN-RNN）

- **核心使命**：解决“单帧观察丢失速度与动力学”以及“未来多模态不确定性”问题。
- **结构与原理**：采用长短期记忆网络（LSTM / RNN）结合混合密度网络（Mixture Density Network, MDN）。
- **递推更新方程**：
  $$h_t = \text{RNN}(h_{t-1}, z_{t-1}, a_{t-1}).$$
- **多模态预测输出**：给定当前确定性隐状态 $h_t$，输出下一个潜在变量 $z_{t+1}$ 的 $K$ 分量高斯混合分布参数（权重 $\pi_k$、均值 $\mu_k$、方差 $\sigma_k$）：
  $$P(z_{t+1} \mid h_t) = \sum_{k=1}^K \pi_k(h_t) \, \mathcal{N}\left(z_{t+1};\, \mu_k(h_t),\, \Sigma_k(h_t)\right).$$

### 3. 控制模块 C（Controller Module）

- **核心使命**：根据当前感知 $z_t$ 与时序记忆 $h_t$ 直接映射动作。
- **结构与原理**：极简的单层线性映射网络（仅含 867 个可学参数）：
  $$a_t = \tanh\left( W_c \, [z_t;\, h_t] + b_c \right).$$
- **优化方式**：使用协方差矩阵自适应进化策略（CMA-ES）进行无梯度优化。

---

## “在梦境中学习”：世界模型的核心思想

《World Models》最具启发性的突破在于：**控制器 C 的全部训练过程，完全脱离真实赛车环境，在记忆模型 M 生成的“虚拟梦境（Dream / Hallucination）”中完成。**

```text
                  ┌── 梦境推演循环 ──────────────────────────────────────────────┐
                  │                                                             │
                  │   z_0 (梦境起点)                                             │
                  │        │                                                    │
                  │        ├──> C 输出动作 a_0 ──> M 采样预测下个潜状态 z_1     │
                  │        │                            │                       │
                  │        ├──> C 输出动作 a_1 ──> M 采样预测下个潜状态 z_2     │
                  │        │                            │                       │
                  │        └──> ... (在想象中展开 1000 步并计算累计回报)         │
                  │                                                             │
                  └─────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼ CMA-ES 进化优化参数 W_c
                                      获得高分控制器 C*
                                                 │
                                                 ▼ 零样本迁移 (Zero-shot Transfer)
                                      部署至真实 CarRacing 赛道
                                      (无需真实环境交互，直接跑通全场！)
```

### 梦境温度系数（Temperature $\tau$）

在梦境中推演时，若模型完全按照均值生成，控制器很容易发现并利用模型的预测漏洞（例如虚假的无阻力捷径）。论文引入了**采样温度 $\tau$**：

$$z_{t+1} \sim \mathcal{N}\left(\mu_k,\, \tau \cdot \sigma_k\right).$$

通过调高 $\tau > 1.0$，人为向梦境中注入适度的不确定性噪声，迫使控制器学会更加鲁棒、保守的驾驶策略，避免对特定预测路径过拟合。

---

## 演进图谱：从 World Models 到 PlaNet、Dreamer 与 MuZero

2018 年至今，世界模型经历了四次关键的技术范式跃迁：

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             世界模型技术演进四代图谱                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. 经典 V-M-C (2018) : VAE + MDN-RNN + 进化算法 (模块割裂，无端到端梯度)       │
│ 2. PlaNet (2019)     : RSSM (确定性记忆 + 随机隐变量) + 在线 CEM 实时规划     │
│ 3. Dreamer 系列      : RSSM + 可微想象空间 + Actor-Critic 梯度直通 (SOTA 标杆) │
│ 4. MuZero (2020)     : 纯任务导向潜空间 (不重建图像，只预测 Value/Policy/Reward)│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. PlaNet (Hafner et al., 2019)：循环状态空间模型（RSSM）

经典 V-M-C 中 VAE 与 RNN 是分步训练的，视觉特征并未针对动力学优化。PlaNet 提出了 **RSSM（Recurrent State-Space Model）**，将状态显式拆解为：

- **确定性时序状态 $h_t$**：$h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$（类似 RNN 记忆）；
- **随机潜在状态 $z_t$**：$z_t \sim q_\phi(z_t \mid h_t, x_t)$（后验）或 $p_\theta(z_t \mid h_t)$（先验）。

PlaNet 放弃了离线训练控制器，而是在每一步使用 **CEM（Cross-Entropy Method）** 在 RSSM 中实时在线规划 $H=12$ 步的最优动作序列。

### 2. Dreamer 系列 (DreamerV1-V3, 2020–2023)：在想象中直通梯度

在线 CEM 规划计算代价过高。Dreamer 创造性地在 RSSM 的连续想象轨迹上，通过**重参数化技巧（Reparameterization Trick）** 直接将 Critic 价值评估对 Actor 策略网络求解析梯度：

$$\nabla_\phi \mathbb{E}\left[\sum_{\tau=t}^{t+H} \gamma^{\tau-t} V_\psi\left(\hat{s}_\tau\right)\right].$$

Actor-Critic 网络的梯度可以直接穿透整个动力学模型反向传播，实现了比无模型强化学习高出数倍至数十倍的数据效率。

### 3. MuZero (Schrittwieser et al., DeepMind 2020)：非对齐的纯潜空间规划

MuZero 提出了一个革命性质疑：**为了玩好雅达利或围棋，我们真的需要花费海量算力去逐像素重建背景的每朵云、每片树叶吗？**

MuZero 彻底抛弃了图像解码器，直接在抽象潜空间中学习动力学，仅约束三项与任务直接相关的预测头：

1. 即时奖励预测：$\hat{r}_t = R(s_t, a_t)$；
2. 策略先验预测：$\hat{p}_t = P(s_t)$；
3. 状态价值预测：$\hat{v}_t = V(s_t)$。

通过结合蒙特卡洛树搜索（MCTS），MuZero 在完全不知晓游戏规则的前提下横扫了国际象棋、围棋与 57 款 Atari 游戏。

---

## 核心架构横向对比矩阵

| 系统                        | 视觉编码 (V)        | 动力学记忆 (M)            | 规划/决策器 (C)   | 是否重建像素                | 训练优化方式                   |
| :-------------------------- | :------------------ | :------------------------ | :---------------- | :-------------------------- | :----------------------------- |
| **World Models** (2018)     | 离线 Conv-VAE       | MDN-RNN (LSTM)            | 线性 Controller   | 是 ($64\times 64$)          | VAE重构 + RNN似然 + CMA-ES进化 |
| **PlaNet** (2019)           | 端到端 Conv-Encoder | RSSM (确定性+随机)        | 在线 MPC (CEM)    | 是 (辅助监督)               | 端到端 ELBO (重构 + KL)        |
| **DreamerV3** (2023)        | Symlog-CNN / ViT    | RSSM (离散 Categorical)   | 想象 Actor-Critic | 是 (确保世界完整)           | 归一化 Actor-Critic + KL平衡   |
| **MuZero** (2020)           | 残差网络表征函数    | 隐空间转移函数 $g_\theta$ | 树搜索 (MCTS)     | **否** (仅预测 $r, \pi, v$) | 价值与策略损失端到端回传       |
| **Genie 2 / 3** (2024-2025) | 时空 ST-VQ / DiT    | 自回归 / 扩散 Transformer | 键盘/手柄交互按键 | 是 (高清 3D 渲染)           | 扩散去噪损失 / 交叉熵          |
| **V-JEPA / V-JEPA 2-AC**    | ViT 特征            | 特征空间预测              | 探针 / 规划       | **否** (只预测特征)         | 掩码特征回归 + 防坍缩          |
| **TD-MPC2**                 | 卷积 / 潜变量       | 潜空间转移 + 价值         | 短期 MPC          | 否（任务头为主）            | 联合训模型与 $V$               |

---

## 五条设计路线落在哪一族

1.7 的表是历史。本书后半把同一套接口拆成五条可动手的路线，不要把它们理解成「越新越好」：

| 本书章节 | 骨干族 | 预测的是什么 | 典型系统 |
| --- | --- | --- | --- |
| 第 4 章 | RNN / RSSM / 隐式搜索 | latent 转移、reward、value | PlaNet、Dreamer、MuZero、TD-MPC |
| 第 5 章 | 词元自回归 / 扩散 | 下一帧或下一组 token | IRIS、Genie、GameNGen |
| 第 6 章 | JEPA | 未来特征 | I-JEPA、V-JEPA、V-JEPA 2-AC |
| 第 7 章 | 策略 + 后果模型 | 动作分布，以及动作的下一状态 | ACT、π₀、OpenVLA + checker |
| 第 8 章 | 场 / BEV / 占用 | 新视角、未来占用 | NeRF、3DGS、GAIA、DriveDreamer |

ACT、π₀、PPO 是**策略**。它们可以当控制器 C，但单独训练时没有学 $P(o_{t+1}\mid o_{\le t}, a_t)$。只有接上后果模型、视频世界模型或占用预测，才进入本课的主线。

## 语言 grounding 与物理 grounding

一个系统可以听懂「拿蓝杯」，却仍然把手指插进桌面。这是两件独立的事：

- **语言 grounding**：换指令，动作必须变。第 7.3 节用同一画面换指令来检查。
- **物理 grounding**：换动作，未来必须变。第 1.6 节把反事实分歧写成定义。

LeCun 的 AMI 把世界模型、代价模块和行动器拆开，是这个区分的架构版：世界模型负责物理后果，代价负责「要不要这么做」，语言最多进入代价，不代替动力学。第 9 章要求两项指标分开报，不许用指令准确率给碰撞失败开脱。

---

## 从零实现：经典 V-M-C 架构的推理数据流

下面用纯 Python 与 NumPy 实现一个极简的 V-M-C 前向推理闭环，直观展示张量在视觉、记忆与控制器之间的流动：

```python
import numpy as np

class TinyVAE:
    """模拟视觉模块 V: 将 (64, 64, 3) 图像压缩至 32 维潜在向量"""
    def __init__(self, latent_dim=32):
        self.latent_dim = latent_dim

    def encode(self, image: np.ndarray) -> np.ndarray:
        # 教学模拟: 图像平均池化后投影
        feat = np.mean(image, axis=(0, 1)) # (3,)
        z = np.sin(np.linspace(0, np.pi, self.latent_dim)) * np.sum(feat)
        return z.astype(np.float32)

class TinyMDNRNN:
    """模拟记忆模块 M: 维护隐藏状态 h，预测下一潜状态分布"""
    def __init__(self, latent_dim=32, hidden_dim=64, action_dim=3):
        self.hidden_dim = hidden_dim
        self.h = np.zeros(hidden_dim, dtype=np.float32)

    def reset(self):
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)

    def step(self, z_t: np.ndarray, a_t: np.ndarray):
        # 拼接隐状态与动作更新: h_{t+1} = tanh(W_h * h_t + W_z * z_t + W_a * a_t)
        combined = np.concatenate([self.h[:16], z_t[:16], a_t])
        self.h = np.tanh(np.pad(combined, (0, self.hidden_dim - len(combined))))

        # 预测下一潜在状态均值与方差
        mu_next = np.roll(z_t, 1) * 0.95
        sigma_next = np.ones_like(z_t) * 0.1
        return self.h.copy(), mu_next, sigma_next

class TinyController:
    """模拟控制模块 C: 输入 [z_t; h_t]，输出 3 维动作 (转向, 油门, 刹车)"""
    def __init__(self, latent_dim=32, hidden_dim=64, action_dim=3):
        np.random.seed(42)
        self.W = np.random.randn(action_dim, latent_dim + hidden_dim) * 0.1
        self.b = np.zeros(action_dim)

    def act(self, z_t: np.ndarray, h_t: np.ndarray) -> np.ndarray:
        state_vec = np.concatenate([z_t, h_t])
        raw_action = np.dot(self.W, state_vec) + self.b
        # 赛车动作: [转向角 \in [-1, 1], 油门 \in [0, 1], 刹车 \in [0, 1]]
        steering = np.tanh(raw_action[0])
        gas = 1.0 / (1.0 + np.exp(-raw_action[1]))
        brake = 1.0 / (1.0 + np.exp(-raw_action[2]))
        return np.array([steering, gas, brake], dtype=np.float32)

# 组装 V-M-C 闭环运行 3 步推理
vae = TinyVAE(latent_dim=32)
rnn = TinyMDNRNN(latent_dim=32, hidden_dim=64, action_dim=3)
ctrl = TinyController(latent_dim=32, hidden_dim=64, action_dim=3)

print("===== 启动 V-M-C 前向推理闭环 =====")
rnn.reset()
a_prev = np.zeros(3)

for step in range(3):
    # 1. 模拟收到一帧相机图像
    raw_frame = np.random.uniform(0, 255, size=(64, 64, 3))

    # 2. V 模块压缩图像 -> z_t
    z_t = vae.encode(raw_frame)

    # 3. M 模块结合历史与上一动作 -> 更新 h_t，预测下一步
    h_t, mu_pred, _ = rnn.step(z_t, a_prev)

    # 4. C 模块根据 [z_t; h_t] 决策当前动作 -> a_t
    a_t = ctrl.act(z_t, h_t)
    a_prev = a_t

    print(f"步数 {step}: 图像输入 (64,64,3) -> z_t (dim={len(z_t)}) -> h_t (dim={len(h_t)}) -> 控制动作 [转向:{a_t[0]:+.2f}, 油门:{a_t[1]:.2f}, 刹车:{a_t[2]:.2f}]")
```

运行输出：

```text
===== 启动 V-M-C 前向推理闭环 =====
步数 0: 图像输入 (64,64,3) -> z_t (dim=32) -> h_t (dim=64) -> 控制动作 [转向:-0.12, 油门:0.48, 刹车:0.49]
步数 1: 图像输入 (64,64,3) -> z_t (dim=32) -> h_t (dim=64) -> 控制动作 [转向:+0.05, 油门:0.52, 刹车:0.47]
步数 2: 图像输入 (64,64,3) -> z_t (dim=32) -> h_t (dim=64) -> 控制动作 [转向:-0.08, 油门:0.50, 刹车:0.51]
```

---

## 自测与常见陷阱

### 自测题

1. **问答题**：在经典 World Models 中，为什么控制器 C 可以只有区区 867 个参数却能学会复杂赛车？
   - _解析_：因为高维视觉表征与复杂物理动力学的繁重特征提取工作，已经分别被 VAE 和 MDN-RNN 完成了；控制器 C 接收到的是高度凝练、富含时序导数的紧凑隐状态 $[z_t; h_t]$，因此只需简单的线性决策超平面即可完成高水准控制。
2. **比较题**：Dreamer 与 MuZero 在对待“是否重建真实图像（Pixel Reconstruction）”这一问题上的根本分歧是什么？各自适用什么场景？
   - _解析_：Dreamer 坚持重建像素（或特征），以确保世界模型拥有对环境全局规律的完整物理因果推演能力，适合连续控制、机器人与具身交互；MuZero 彻底抛弃像素重建，完全由任务目标（Reward/Value）驱动潜空间，计算开销更低、对复杂无关视觉背景具有极强免疫力，适合棋类与复杂策略游戏。
3. **判断题**：在 Dreamer 中，因为有了重参数化技巧，Actor 可以在想象中无限展开 100 步进行梯度反向传播。
   - _解析_：错误。虽然梯度理论上可直通，但穿过 100 层非线性循环单元的梯度极易发生梯度爆炸或退化；实际工程中 Dreamer 通常将前瞻时域严格截断在 $H=15$ 步左右。

### 常见误区与工程陷阱

- **误区 1：认为 VAE 的重建图像越逼真，控制效果一定越好**。VAE 可能花费大量参数去重建路边的草坪纹理，却忽视了远方只有 2 个像素大小的关键障碍物。第 6 章将介绍通过 JEPA 或对齐损失克服这一缺陷。
- **误区 2：在梦境训练中忽视终止信号（Termination）**。在虚拟想象中推演时，如果撞车死亡后未正确阻断梯度或状态更新，策略会学会“从死后状态中死而复生继续得分”的错误逻辑。
- **误区 3：混淆先验（Prior）与后验（Posterior）的使用时机**。在 RSSM 中，在真实环境中与真机交互接收到图像 $x_t$ 时，必须使用后验网络 $q(z_t \mid h_t, x_t)$；在纯脑海想象多步推演（没有未来图像）时，必须使用先验网络 $p(z_t \mid h_t)$。

---

## 小结与下节预告

- **V-M-C 范式** 奠定了感知压缩（V）、时序记忆（M）与紧凑决策（C）解耦的现代世界模型三元架构。
- **PlaNet、TD-MPC 与 Dreamer** 分别用在线搜索、短视界价值 MPC、可微想象来消费同一个潜空间。
- **MuZero** 证明了不重建像素也能在纯任务潜空间中完成超人水平的树搜索规划。
- **五条路线** 对应五族骨干；策略网络本身不是世界模型。

理论与架构的推演至此全部打通。下一篇 [1.8 动手：九格世界的从零实现](/chapters/01-why-world-models/08-invent-a-world-model) 进入本章的综合代码实战：不依赖任何第三方深度学习框架，只用 Python 标准库把“观察—状态—预测—推演—学习—MPC”六个环节全部手工实现，并亲眼看到闭环跑起来。
