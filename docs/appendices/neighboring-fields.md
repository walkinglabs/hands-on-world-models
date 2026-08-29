# 附录 D　邻近课程与覆盖对照

本附录做两件事：把 Hugging Face、Datawhale 和高校课里出现过、但不宜再开新章的问题收进来；并给出一张对照表，说明每个主题落在本书哪一页。

正文的主线仍然是世界模型。下面这些主题必须能讲清楚它们**改的是哪一段接口**，以及为什么不单独成章。

## 覆盖对照

| 外来课的主题                                                                                                                   | 本书落点                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF Robotics Course Unit 0–2：LeRobot、经典控制极限                                                                             | [7.1](/chapters/07-robot-policy/01-multimodal-observation)                                                                                                                                                                   |
| HF _Robot Learning Tutorial_：ACT、Diffusion、Flow、RTC、π₀、SmolVLA                                                           | [7.2](/chapters/07-robot-policy/04-behavior-cloning)、[7.3](/chapters/07-robot-policy/07-vla-rtx)                                                                                                                            |
| LeRobot v0.4–v0.6：teleop→record→train→rollout 闭环、DAgger 回流、HIL-SERL、LIBERO 评测协议、世界模型策略（VLA-JEPA、FastWAM） | [7.8](/chapters/07-robot-policy/10-diffusion-policy-scratch)、[7.7](/chapters/08-robot-sim/01-physics-mujoco)、[9.2](/chapters/10-evaluate-and-invent/02-systematic-evaluation)、[7.4](/chapters/07-robot-policy/07-vla-rtx) |
|                                                                                                                                | EVA-Client 类部署框架：遥操作采集、质检回放、checkpoint 下发、延迟补偿、轨迹平滑、日志对比                                                                                                                                   | [7.8](/chapters/07-robot-policy/10-diffusion-policy-scratch) 的采集与部署两节 |
| HF Deep RL：DQN / PPO / SAC、model-based bonus                                                                                 | [2.6](/chapters/03-data-and-first-model/03-value-policy-gradient)、第 4 章                                                                                                                                                   |
| HF ML for 3D：多视图扩散、3DGS、mesh                                                                                           | [8.3](/chapters/09-spatial-worlds/03-nerf-3dgs)                                                                                                                                                                              |
| Datawhale L01–L05 / P01–P06：四代历史、RSSM、Dreamer、换骨干、反事实                                                           | [1.4](/chapters/01-why-world-models/03-classic-world-models)、第 4 章、[9.2](/chapters/10-evaluate-and-invent/02-systematic-evaluation)、本附录的换骨干条目                                                                  |
| Datawhale every-embodied：VAE/DDPM 代码、LeWM 复现、SO-101 遥操作                                                              | [4.6](/chapters/04-latent-dynamics/07-rssm-scratch)、[6.5](/chapters/06-jepa/05-jepa-scratch)、[7.8](/chapters/07-robot-policy/10-diffusion-policy-scratch)                                                                  |
| Datawhale dive-into-embodied-ai：CS123 中文仿真版、VLA 十二讲                                                                  | [7.1](/chapters/07-robot-policy/01-multimodal-observation)、[7.6](/chapters/07-robot-policy/02-dexterous-manipulation)、[7.3](/chapters/07-robot-policy/07-vla-rtx)                                                          |
| MIT 6.s953：POMDP、少数据、Dyna、UniSim、好奇心、元学习、MARL                                                                  | [1.2](/chapters/01-why-world-models/01-observation-and-state)、[4.6](/chapters/04-latent-dynamics/07-rssm-scratch)、本附录                                                                                                   |
| Berkeley CS 285：模仿学习、model-based、offline RL、LLM-RL                                                                     | [7.2](/chapters/07-robot-policy/04-behavior-cloning)、第 4 章、本附录                                                                                                                                                        |
| Berkeley CS 294-277：触觉、腿式、视频世界模型当策略、长时程语言                                                                | [7.4](/chapters/07-robot-policy/07-vla-rtx)–[7.6](/chapters/07-robot-policy/02-dexterous-manipulation)、[7.3](/chapters/07-robot-policy/07-vla-rtx)                                                                          |
| Stanford CS 123：PD、FK/IK、四足 RL、foundation model lab                                                                      | [7.1](/chapters/07-robot-policy/01-multimodal-observation)、[7.6](/chapters/07-robot-policy/02-dexterous-manipulation)、[7.9](/chapters/07-robot-policy/10-diffusion-policy-scratch)                                         |
| 北大《具身智能导论》：三维抓取、Sim2Real、多模态大模型                                                                         | [7.5](/chapters/07-robot-policy/02-dexterous-manipulation)、[7.7](/chapters/08-robot-sim/01-physics-mujoco)、[7.3](/chapters/07-robot-policy/07-vla-rtx)                                                                     |

动手配方仍只出现在各章「动手：」页。本附录不另开 Notebook。

---

## 无模型强化学习家族

世界模型课默认读者见过「用回报更新策略」。下面这张表把无模型方法钉在接口上，避免和第 4 章的 model-based 路线混为一谈。

| 方法        | 学什么                            | 不学什么             | 典型用途                                             |
| ----------- | --------------------------------- | -------------------- | ---------------------------------------------------- |
| DQN         | $Q(s,a)$，离散动作取 $\arg\max$   | 转移 $P(s'\mid s,a)$ | Atari、网格                                          |
| PPO / A2C   | 策略 $\pi(a\mid s)$ 与价值 $V(s)$ | 转移                 | 连续控制、腿式教师                                   |
| SAC         | 随机策略 + 熵正则                 | 转移                 | 操作、接触丰富的仿真                                 |
| 行为克隆    | $\pi(a\mid o)$ 拟合专家           | 回报与转移           | [7.2](/chapters/07-robot-policy/04-behavior-cloning) |
| Offline RL  | 只在固定数据集上改进策略          | 新的真实交互         | 下面一节                                             |
| Model-based | 转移，再用规划或想象更新策略      | —                    | 第 4 章                                              |

判断句：若实验没有学 $P(s'\mid s,a)$ 或等价的动作条件预测，它就不是世界模型实验，即使环境是机器人。

---

## 离线强化学习

**离线 RL**（offline RL）的约束是：策略改进只能使用已经采集好的数据集 $\mathcal{D}$，不能再向环境要新的 $(s,a,s')$。这和「用离线数据训世界模型」相邻，但目标不同。

- 世界模型：学 $\hat P(s'\mid s,a)$，评价看反事实与多步漂移。
- 离线策略：学 $\pi$，评价看在 $\mathcal{D}$ 支持内的回报；超出支持时 Q 函数会高估未见过的动作。

常见对策是约束策略不要远离行为策略（CQL、IQL 一类）。对世界模型课的含义是：只用成功示范训后果模型，会把碰撞区当成 OOD 并自信外推——这正是 [7.4](/chapters/07-robot-policy/07-vla-rtx) 要求混入坏动作的原因。

---

## 分层决策与语言规划

长时程任务（「热一杯咖啡」）不能只靠 50 Hz 的关节指令。常见拆法：

```text
语言 / 任务 → 技能序列（拿杯、倒水、递出）→ 技能内的低层策略或 VLA chunk
```

**SayCan** 用语言模型对技能打分，再用价值函数或可达性检查「身体现在做不做得了」。它改的是**技能选择**，不是像素动力学。VLA 把技能和低层动作焊在同一个网络里；分层系统把它们拆开，世界模型可以出现在每一层：高层预测「倒水之后台面会不会湿」，低层预测「这一段 chunk 会不会撞」。

评价分层系统时，必须分开报：技能选择错误、技能执行错误、以及两者耦合后的任务失败。只报最终成功率，看不出世界模型帮的是哪一层。

---

## 元学习与「极少数据」

MIT 6.s953 把「数据极少」单列为一周。**元学习**（MAML、RL$^2$）的目标是：在一簇任务上训练，使得新任务只需几条轨迹就能适应。

它和世界模型的关系是竞争，不是包含。世界模型用过去的转移加速当前任务；元学习用过去的**任务分布**加速新任务。两者可以叠：先在世界模型里适应，再把适应后的策略搬到真机。课程不把 MAML 当必做实验；若选题碰到「每个物体只有 5 条示范」，先写清你改的是动力学、策略初始化，还是数据增强。

---

## 从单智能体到多智能体世界模型（MARL与人类交互）

当环境里还有其他学习主体（其他机器人、人类）时，转移不再是平稳的 $P(s'\mid s,a)$：别人的策略也在变。这就是 **MARL** 和非平稳。自对弈（AlphaZero、Hide and Seek）把对手当作环境的一部分一起学。

对世界模型的直接后果：你学到的动力学里混进了别人当时的策略。换一个对手或人类，同一动作的后果会变。诚实的做法是把其他主体的动作也写进条件，比如 **Gamma-World**（多智能体世界模型）或 **WALL-WM**（事件级动作模型）：

$$
\hat s_{t+1}=f_\theta(s_t, a_t^{\text{self}}, a_t^{\text{others}}),
$$

而不是假装世界只有自己。同时，引入人类不仅是作为环境障碍，还包括 **RLHF（基于人类反馈的强化学习）**，让人类告诉世界模型“哪些生成未来是危险的或偏好的”。

本书实验默认单智能体；若做对抗、协作或人类交互，必须把「谁的动作进了条件」写进数据卡。

---

## 生物运动控制在提供什么

Berkeley CS 294-277 用生物力学开场，不是为了讲解剖，而是给出三条工程约束：

1. **延迟**：视觉约 $100\,\mathrm{ms}$、本体感觉更快；控制回路必须在延迟下稳定。见 [7.2](/chapters/07-robot-policy/04-behavior-cloning) 的 RTC 与 [7.7](/chapters/08-robot-sim/01-physics-mujoco) 的动作延迟。
2. **欠驱动与接触**：没有电机连着地面，力只能通过摩擦锥传递。见 [7.6](/chapters/07-robot-policy/02-dexterous-manipulation)。
3. **发育式课程**：婴儿先稳定头和躯干，再走、再抓。对应工程上的课程学习：先状态观察的教师，再视觉学生。

生物启发到此为止。本书不把肌肉模型或中枢模式发生器当成可打分实验。

---

## 物体中心表示

把整张图压成一个向量，杯子和背景挤在一起。**物体中心**（object-centric）表示先把场景拆成物体槽，再对每个槽预测位姿与接触。抓取规划、插接、语言里的「那个红杯」都依赖这种拆分。

它改的是状态 $s_t$ 的内部结构，不是决策接口。第 8 章的占用网格是空间版的物体中心；第 7.5 节的接触后果是物体版。若线性探针能从表示里读出物体坐标，而换物体颜色后坐标仍对，才算拆开了，而不是记了纹理捷径。

---

## LLM 策略与「会说不会动」

用语言模型直接输出技能名或代码（Code as Policies），是 CS 285 近年的 LLM-RL 线。它解决的是长时程离散选择，不解决 $50\,\mathrm{Hz}$ 的连续控制。

和世界模型的接法只有一种合法形式：语言模型提出候选，世界模型或价值函数否决做不到的候选。只把 LLM 的文本当成功证据，不能代替 [7.4](/chapters/07-robot-policy/07-vla-rtx) 的动作后果检查。

---

## 换动力学骨干：RSSM 与因果 Transformer

Datawhale 的 P04 要求：同一套数据上，把 RSSM 换成小型因果 Transformer（STORM 一类），并并排比较。

合法对照只有一条：编码器、预测头、数据、horizon 和 seed 集合保持不变，只换 $s_t = g(s_{t-1}, e_t, a_{t-1})$ 的实现。

- RSSM：GRU 维护 $h_t$，随机 $z_t$ 走 KL。
- 因果 Transformer：把 $(z_{\le t}, a_{\le t})$ 当成 token 序列，用遮罩预测 $z_{t+1}$。

必须同时报一步损失、多步漂移、反事实差异。只报 Transformer 的 next-token loss 更低，不能声称规划变好。第 4.7 节的简洁档允许把这个对照当作 Dreamer-lite 的替代选题；第 5 章的视频 Transformer 是另一条路线，不要把像素 token 和 RSSM latent 混在一张表里。

---

## 语言 grounding 与物理 grounding

Datawhale L05 的争论可以收成一个判断：

- **语言 grounding**：表示能否对齐「红杯 / 蓝杯」这类符号，换指令则换动作。见 [7.3](/chapters/07-robot-policy/07-vla-rtx)。
- **物理 grounding**：表示能否保留接触、碰撞、可达性，换动作则换未来。见 [1.3](/chapters/01-why-world-models/02-what-is-a-world-model) 的反事实条件。

两者可以同时失败：画面清晰、指令对、手还是插进桌子。第 9 章要求两项分开报，不许用 FID 或指令准确率互相替代。LeCun 的 AMI 把世界模型、代价、演员拆开，是这个争论的架构版本，不是新的损失函数名词。
