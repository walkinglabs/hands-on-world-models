# 7.9　世界动作模型（VLA-JEPA / WAM）

桌面上同时放着红杯和蓝杯。只看图片，小臂知道物体在哪里，却不知道这次任务要抓哪一个；只看“拿起蓝杯”这句话，又不知道杯子位于桌上的哪个角落。

**VLA**（vision-language-action）模型要做的，就是把视觉、语言和机器人状态三种输入，共同映射到一个动作。所谓 VLA，就是一台同时会看、会听、会动的策略网络。

![图 7-20 Google RT-2：直接将网络规模预训练的视觉语言模型转化为机器人动作生成策略 (Brohan et al., 2023)](/figures/rt2-architecture.png)

![图 7-21 Google RT-1：面向真实厨房与移动操作的大规模 Robotics Transformer (Brohan et al., 2022)](/figures/rt1-model-teaser.png)

---

## 三类输入，三种 token

先把三种输入各自变成模型能消化的表示。图像交给 CNN 或 ViT，得到一组视觉 token；指令交给文本编码器，得到一组语言 token；自身状态交给一个小 MLP，得到一个状态 token。

$$
\text{image}\xrightarrow{\,\text{ViT/CNN}\,}Z_v,\qquad
\text{instruction}\xrightarrow{\,\text{encoder}\,}Z_\ell,\qquad
s_t\xrightarrow{\,\text{MLP}\,}z_s.
$$

其中 $Z_v\in\mathbb{R}^{N_v\times d}$ 是 $N_v$ 个视觉 token，$Z_\ell\in\mathbb{R}^{N_\ell\times d}$ 是 $N_\ell$ 个语言 token，$z_s\in\mathbb{R}^{d}$ 是单个状态 token，三者维度都对齐到 $d$。

接着把这些 token 拼成一串，交给 Transformer。每个 token 都可以注意到另外两类，于是语言可以指向图像里某个物体，图像也可以反过来约束语言。最后由一个 **action head** 输出动作 $\hat a_t$。

```text
[ Z_v ; Z_ℓ ; z_s ] → Transformer → action head → â_t
```

![图 7-22 OpenVLA：基于 Llama 2 与 DINOv2 / SigLIP 骨干的 7B 通用视觉-语言-动作基础模型 (Kim et al., 2024)](/figures/openvla-model.jpg)

---

## 多模态融合：交叉注意与拼接

三种 token 互相影响的方式有很多种。最直接的办法是全部拼成一条长序列，让标准自注意力去混合。这叫**早期融合**。

也可以分开处理：视觉 token 自己做几层自注意，再通过**交叉注意**（cross-attention）去查询语言 token。设视觉查询为 $Q_v$、语言键值为 $K_\ell,V_\ell$，单层交叉注意是：

$$
\text{Attn}(Q_v,K_\ell,V_\ell)=\mathrm{softmax}\!\left(\frac{Q_v K_\ell^{\top}}{\sqrt{d}}\right)V_\ell.
$$

无论哪种融合，关键是语言必须真的参与决定动作。如果删掉语言 token、输出几乎不变，模型就只是把图像和状态又用了一遍，并没有“听”这句话。

![图 7-23 Octo：基于 Transformer 与扩散动作头的跨本体通用机器人策略架构 (Octo Model Team, 2024)](/figures/octo-model.png)

---

## 同一画面换指令

判断语言是否生效，最小反事实是固定图片和状态，把“拿红杯”换成“拿蓝杯”。设两次输出的动作分别为 $\hat a_t^{(\text{红})}$ 和 $\hat a_t^{(\text{蓝})}$，我们希望两者有可测量的差异：

$$
\Delta a=\bigl\lVert \hat a_t^{(\text{红})}-\hat a_t^{(\text{蓝})}\bigr\rVert_2\;\gg\;0.
$$

如果训练数据里红杯总在左边、蓝杯总在右边，模型可能只学到位置捷径，根本不读语言。测试集需要交换颜色与位置，才能逼出语言是否真在被使用。

---

## 预训练视觉语言模型提供什么

预训练的 **VLM**（vision-language model）见过海量图文，可以提供物体与语言的通用表示，减少从零学习语义的开销。但它通常不输出符合机器人坐标系、控制频率和安全约束的关节动作。

换句话说，VLM 给的是一个会看会说的底座。VLA 还要补上动作示范、机器人自身状态和控制接口。会看会说，不等于已经会动。

![图 7-24 Google RT-2 在未见物体识别、符号推理与常识理解中展现出的零样本动作泛化能力 (Brohan et al., 2023)](/figures/rt2-generalization.png)

---

## π₀ 的双专家与轻量 VLA

$\pi_0$ 不是「把 VLM 后面接一个 MLP」。它是一个混合专家 Transformer：大的视觉语言骨干处理图像和指令，小的 **action expert** 处理本体感觉与动作块，两边用自注意力通信、权重分开。连续动作由流匹配生成，所以 7.2 的直线流是它的动作头，不是另一台模型。

**SmolVLA** 把同一接口收到大约 $2.5\times 10^8$ 参数，目标是消费级 GPU 上能微调。**GR00T** 一类通才模型再把人形、全身和跨本体数据加进去。课程动手用小型 VLA；读论文时要能指出：冻结的是哪一段、动作头是离散 token、扩散还是流匹配、有没有独立的 action expert。

![图 7-25 HuggingFace LeRobot 端到端开源 VLA 训练、评测与部署闭环流程](/figures/lerobot-vla-architecture.jpg)

---

## 长时程：先选技能再动手

「热一杯咖啡」不能只靠 50 Hz 的 chunk。**SayCan** 用语言模型给技能打分，再用价值或可达性扔掉身体做不到的技能。它改的是技能选择，不是像素动力学。VLA 把技能和低层焊在一起；分层系统把它们拆开。世界模型可以出现在两层：高层预测倒水后台面会不会湿，低层预测这一段 chunk 会不会撞。评价必须分开报技能选择错误和技能执行错误，见 [附录 D](/appendices/neighboring-fields)。

![图 7-26 SayCan：结合大语言模型常识推理与机器人可执行度（Affordance）的分层长程规划框架 (Ahn et al., 2022)](/figures/saycan-architecture.png)

![图 7-27 RT-1 在真实世界多任务长程评测与基线对比结果 (Brohan et al., 2022)](/figures/rt1-real-evals.png)

---

## 冻结还是微调

数据少的时候，可以冻结视觉或语言编码器，只训练融合层与 action head。设可训练参数为 $\theta_{\text{fuse}}$ 和 $\theta_{\text{head}}$，冻结参数为 $\theta_{\text{enc}}$，则一步更新只动前者：

$$
\theta_{\text{fuse}},\theta_{\text{head}}\leftarrow \theta_{\text{fuse}},\theta_{\text{head}}-\eta\,\nabla_{\theta_{\text{fuse}},\theta_{\text{head}}}\,\mathcal{L}_{\text{BC}}.
$$

这样省显存、也减少过拟合。若相机视角和预训练图像差异很大，再逐步解冻后层。公平对照应当依次比较 state-only、image+state、image+language+state 三档，而不是一次加全部模态后只报最好的结果。

---

## 机器人世界模型的三种用法

直接 VLA 给出动作就结束了。它可以很快，却不一定在执行前比较过「从左侧接近」和「从右侧接近」各自会发生什么。

机器人世界模型不是一种网络，而是三种用法。它们共用「动作条件预测」，下游完全不同。

| 用法           | 世界模型输出什么               | 谁来消费                               | 代表系统                                                |
| -------------- | ------------------------------ | -------------------------------------- | ------------------------------------------------------- |
| **后果检查器** | 下一状态、碰撞、是否更接近目标 | VLA 或规划器，执行前重排候选           | 1X 一类部署；本节转移方程                               |
| **数据引擎**   | 可交互的未来视频或轨迹         | 用来合成示范、做域随机、少真机试错     | UniSim、DreamGen、Cosmos                                |
| **零样本策略** | 点轨迹、未来视频或动作 token   | 直接当策略，或把预测点当成要跟踪的目标 | Track2Act、World Action Models、LeRobot v0.6 的 FastWAM |

三种用法的反事实检查是同一句：固定当前观察，只换候选动作，预测必须分开。分不开，就还不是世界模型，只是视频生成或行为克隆。

### 用法一：后果检查器与转移方程

把「想动作」和「想后果」拆成两个模型：

```text
VLA：          image + instruction + proprio → candidate actions
World model：  current state + candidate action → next state / collision
```

世界模型只学一件事：在当前状态下执行某个动作，下一状态会变成什么、会不会碰撞。语言负责指定目标，不进入这个转移。

设当前状态为 $s_t$，候选动作为 $a_t$，世界模型预测下一状态 $\hat s_{t+1}$ 和碰撞概率 $\hat c_{t+1}$。最朴素的写法是确定性转移加上一个分类头：

$$
\hat s_{t+1}=f_\theta(s_t,a_t),\qquad \hat c_{t+1}=\sigma\bigl(g_\phi(s_t,a_t)\bigr)\in(0,1),
$$

其中 $\sigma$ 是 sigmoid。状态部分用回归损失，碰撞部分用二分类交叉熵：

$$
\mathcal{L}_{\text{state}}=\lVert \hat s_{t+1}-s_{t+1}\rVert_2^2,\qquad
\mathcal{L}_{\text{coll}}=-c_{t+1}\log \hat c_{t+1}-(1-c_{t+1})\log(1-\hat c_{t+1}).
$$

两者加权相加就是一步后果模型的训练目标：

$$
\mathcal{L}_{\text{outcome}}=\mathcal{L}_{\text{state}}+\lambda\,\mathcal{L}_{\text{coll}}.
$$

把一步后果接起来，就能在脑子里走 $H$ 步，这就是**想象 rollout**。从一个真实状态 $s_t$ 出发，反复喂入候选动作序列：

$$
\hat s_{t+1}=f_\theta(s_t,a_t),\quad \hat s_{t+2}=f_\theta(\hat s_{t+1},a_{t+1}),\quad \ldots,\quad \hat s_{t+H}=f_\theta(\hat s_{t+H-1},a_{t+H-1}).
$$

整条想象轨迹的总风险，可以近似成各步不碰撞概率的乘积：

$$
P_{\text{safe}}=\prod_{k=1}^{H}\bigl(1-\hat c_{t+k}\bigr).
$$

专家示范大多是成功动作。若后果模型只见过好动作，它从未见过碰撞长什么样，也就无法判断一个动作坏不坏。坏动作是 checker 的必要监督。

公平比较要固定同一个候选生成器和相同计算预算，对照三种策略：直接执行最高概率动作、用一步后果模型重排（learned checker）、用真实模拟器后果排序（上限参考）。设三者真实碰撞率分别为 $R_{\text{direct}}$、$R_{\text{learned}}$、$R_{\text{oracle}}$，我们希望看到

$$
R_{\text{oracle}}\;\le\;R_{\text{learned}}\;\le\;R_{\text{direct}}.
$$

候选生成器可能找到后果模型错判为安全的动作——这叫 checker 被利用。对策是不确定性阈值、动作约束、真实反馈和失败数据回流。世界模型不是安全保证；真实机器人仍需要独立的速度、力矩、工作空间和急停约束。还有一种失败：一步 checker 可以一直选择不会碰撞、却离目标更远的动作。评价因此必须同时记录真实碰撞率和每步目标进展。

### 用法二：世界模型当数据引擎

真机试错贵，仿真又对接触不诚实。第二条用法是：用世界模型生成看起来像真机、并且动作可干预的轨迹，拿去训 VLA 或补稀有失败。

合法的数据引擎必须通过和检查器相同的反事实门槛。否则生成的只是「更像训练视频的视频」，策略会在幻觉里过拟合。UniSim 从真实传感器学可交互模拟；DreamGen 一类用视频世界模型扩示范；NVIDIA Cosmos 把同一想法做成可条件化的世界基础模型。它们改的是**数据从哪来**，不是检查器的接口。

评价不能只报生成 FVD。至少还要：合成数据训出的策略，在真机或保留仿真上的成功率；以及换动作后生成轨迹是否分叉。第 5 章的按键视频是这条用法的最小课堂版。

### 用法三：视频世界模型当零样本策略

第三条用法更激进：不另训 VLA，把未来预测直接当成要执行的计划。

Track2Act 从互联网视频预测物体和手的点轨迹，再让真机去跟踪这些点。World Action Models 把动作本身写成世界模型的一种输出。它们和检查器的差别是：**预测的消费者是跟踪控制器，不是重排模块**。失败模式也不同：检查器错了，最多选错候选；零样本策略错了，手会去抓视频里的幻觉点。所以必须报跟踪误差和真机任务成功率，不能用视频 PSNR 代替。

LeRobot v0.6 把这条路线做成了产品：VLA-JEPA 把第 6 章的 Action-JEPA 直接当策略——不重建像素、不另训 Actor-Critic，预测出的特征就是动作的依据。Track2Act 预测像素空间里的点，VLA-JEPA 预测特征空间里的未来，再由一个轻量头译成动作。这正好是 6.4「Action-JEPA 是世界模型的表示半成品」那句的落地版。

---

## 动手入口：Tiny VLA 与后果检查

完整的配方级实验在 [7.6 动手：扩散策略的从零开始实现](/chapters/07-robot-policy/10-diffusion-policy-scratch)；把检查器做成可验证证据，是 [7.8 动手：把世界模型接上身体](/chapters/07-robot-policy/11-world-model-body-loop) 的毕业设计。这里只给最小入口：两份教学 Notebook，CPU 上几分钟跑完。

- [搭一台小型 VLA](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_robot/build-a-tiny-vla.ipynb)：160 条 $32\times 32$ 桌面示范，state-only BC 损失 $0.498\rightarrow 0.377$，Tiny VLA chunk 损失 $0.525\rightarrow 0.277$，同一张图换指令后动作平均差 $0.120$。
- [行动前检查动作](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_robot/check-actions-before-moving.ipynb)：后果模型重排候选。64 个必撞场景里，直达碰撞 $1.000$，重排后 $0.328$，平均进展却是 $-0.036$——安全了，并不等于更接近目标。

实现见 `src/hwm/robot.py`。烟雾不是完整训练：目标是检查图片、指令、状态、动作是否对齐，以及换指令后动作是否真的变。

---

## 小结

- [ ] VLA 用 Transformer 把图像、语言、proprio 三种 token 融合成一个动作。
- [ ] 交叉注意是让语言查询图像、图像约束语言的标准工具。
- [ ] 同一画面换指令，是检查语言条件是否生效的最小反事实。
- [ ] VLM 表示不能替代动作数据与控制接口。
- [ ] π₀ 用视觉语言骨干加独立 action expert 和流匹配；SmolVLA 是同一接口的小算力版。
- [ ] 长时程任务可以先选技能（SayCan）再执行；世界模型要声明自己在哪一层。
- [ ] 机器人世界模型有三种用法：后果检查、数据引擎、零样本策略。不要把其中一种写成定义。
- [ ] 后果数据必须包含失败与坏动作；learned checker 要与直接执行和真实模拟器上限比较；FVD / PSNR 不能代替真机任务数字。

下一篇把接触问题从像素搬到力：指尖的摩擦锥与整台身体的浮动基，是同一套互补条件在两个尺度上的展开。

[上一篇 7.2　行为克隆与扩散策略](/chapters/07-robot-policy/04-behavior-cloning) · [下一篇 → 7.4　接触力与全身控制](/chapters/07-robot-policy/02-dexterous-manipulation)
