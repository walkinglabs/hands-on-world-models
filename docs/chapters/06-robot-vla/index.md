# 第 6 章　具身智能与机器人

视觉语言模型可以描述桌上的杯子，机器人却需要输出能执行的动作。本章先实现一台能动的动作模型，再加入一个小型世界模型，在执行以前检查候选动作的后果。

## 本章文章

1. [机器人数据与行为克隆](./06-01-robot-data-and-bc.md)：一条 episode 里有什么，以及为什么动作 MSE 不等于成功。
2. [视觉-语言-动作模型（VLA）](./06-02-vision-language-action.md)：图像、语言、proprio 三种 token 怎样融合成一个动作。
3. [动作分块与多模态动作](./06-03-action-chunk.md)：一次输出一段动作，并允许同一上下文有多条可行路线。
4. [世界模型检查器](./06-04-world-model-checker.md)：让 VLA 提出候选，再用想象 rollout 检查各自后果。
5. [6.5 动手：机器人与 VLA 实验](./06-05-robot-vla.md)

直接 VLA 的输出是动作；世界模型的输出是动作后果。两者可以连接，但不能用同一个名字代替。动作 MSE 下降也不保证闭环成功，本章会一直保留这种诚实差距。

## 参考资料

### 实践博客（5 篇）

1. [RT-2: New model translates vision and language into action (DeepMind)](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/) —— 官方博客，用大量真机案例展示 VLA 的泛化与失败模式。
2. [ALOHA: A Low-cost Open-source Hardware System (Tony Zhao)](https://tonyzhaozh.github.io/aloha/) —— ALOHA 项目页：低成本双臂遥操作硬件与 ACT 策略的配套说明，配 6.1、6.3。
3. [Mobile ALOHA (Fu et al.)](https://mobile-aloha.github.io/) —— 项目页：移动底盘 + 50 条演示学出全身操作，展示小数据配方的效果。
4. [Diffusion Policy 项目页 (Chi et al.)](https://diffusion-policy.cs.columbia.edu/) —— 扩散策略的项目页，含大量对比可视化，配 6.3 的多模态动作。
5. [OpenVLA: Open-Source Vision-Language-Action Model](https://openvla.github.io/) —— 开源 VLA 项目页：配方、数据与评测全公开，适合动手复现。

### 原始论文（5 篇）

1. [RT-1: Robotics Transformer for Real-World Control at Scale (Brohan et al., 2022)](https://arxiv.org/abs/2212.06817) —— 首个大规模真机数据训出的通用机器人模型，本章数据与 BC 部分的参照。
2. [RT-2: Vision-Language-Action Models (Zitkovich et al., 2023)](https://arxiv.org/abs/2307.15818) —— 把动作写成 token 接进 VLM 的原始论文，配 6.2。
3. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware: ACT (Zhao et al., 2023)](https://arxiv.org/abs/2304.13705) —— 动作分块（action chunking）的来源，ALOHA 硬件与 CVAE 策略，配 6.3。
4. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (Chi et al., 2023)](https://arxiv.org/abs/2303.04137) —— 用扩散模型输出多模态动作序列，是“同一上下文多条可行路线”的代表实现。
5. [OpenVLA: An Open-Source Vision-Language-Action Model (Kim et al., 2024)](https://arxiv.org/abs/2406.09246) —— 开源 VLA 的配方与评测，配 6.2 的可复现基线。
