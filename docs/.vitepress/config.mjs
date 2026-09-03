import { defineConfig } from "vitepress";
import {
  pythonSemanticTransformer,
  setupPythonSemanticStyles,
} from "./semantic-highlighting.mjs";

export default defineConfig({
  lang: "zh-CN",
  title: "动手学世界模型",
  description: "从看见、记住和预测，到在想象中规划与行动",
  base: process.env.BASE || "/hands-on-world-models/",
  cleanUrls: false,
  lastUpdated: true,
  markdown: {
    math: true,
    lineNumbers: true,
    shikiSetup: setupPythonSemanticStyles,
    codeTransformers: [pythonSemanticTransformer()],
    theme: {
      light: "light-plus",
      dark: "dark-plus",
    },
  },
  head: [
    ["meta", { name: "theme-color", content: "#25636a" }],
    ["meta", { name: "author", content: "Walking Labs" }],
  ],
  themeConfig: {
    logo: { light: "/logo-light.png", dark: "/logo-dark.png" },
    siteTitle: "动手学世界模型",
    outline: { level: [2, 3], label: "本页内容" },
    lastUpdated: { text: "最后更新" },
    docFooter: { prev: "上一篇", next: "下一篇" },
    returnToTopLabel: "回到顶部",
    sidebarMenuLabel: "目录",
    darkModeSwitchLabel: "外观",
    nav: [
      {
        text: "绪论",
        items: [
          {
            text: "绪论",
            link: "/guide/world-model-intro",
          },
          {
            text: "世界模型八十年",
            link: "/guide/world-model-history",
          },
        ],
      },
      {
        text: "共同基础",
        items: [
          {
            text: "为什么需要世界模型",
            link: "/chapters/01-why-world-models/01-observation-and-state",
          },
          {
            text: "深度学习与表征基础",
            link: "/chapters/02-foundations/01-cnn-and-vit",
          },
          {
            text: "数据与强化学习基础",
            link: "/chapters/03-data-and-first-model/01-episodes-and-transitions",
          },
        ],
      },
      {
        text: "技术路线",
        items: [
          {
            text: "潜在动力学模型",
            link: "/chapters/04-latent-dynamics/01-world-models",
          },
          {
            text: "交互式视频生成",
            link: "/chapters/05-interactive-video/01-video-prediction-svg",
          },
          {
            text: "联合嵌入预测架构",
            link: "/chapters/06-jepa/01-jepa-foundation",
          },
          {
            text: "具身策略大模型",
            link: "/chapters/07-robot-policy/01-multimodal-observation",
          },
          {
            text: "具身仿真与现实迁移",
            link: "/chapters/08-robot-sim/01-physics-mujoco",
          },
          {
            text: "空间世界与自动驾驶",
            link: "/chapters/09-spatial-worlds/01-camera-geometry",
          },
        ],
      },
      {
        text: "评测与研究",
        items: [
          {
            text: "评测与研究设计",
            link: "/chapters/10-evaluate-and-invent/01-evaluate-principles",
          },
        ],
      },
      {
        text: "GitHub",
        link: "https://github.com/walkinglabs/hands-on-world-models",
      },
    ],
    sidebar: [
      {
        text: "导览",
        collapsed: false,
        items: [
          {
            text: "绪论",
            link: "/guide/world-model-intro",
          },
          {
            text: "世界模型八十年",
            link: "/guide/world-model-history",
          },
        ],
      },
      {
        text: "1. 为什么需要世界模型",
        collapsed: false,
        items: [
          {
            text: "1.1. 观察、状态与变化",
            link: "/chapters/01-why-world-models/01-observation-and-state",
          },
          {
            text: "1.2. 什么是世界模型",
            link: "/chapters/01-why-world-models/02-what-is-a-world-model",
          },
          {
            text: "1.3. 经典世界模型",
            link: "/chapters/01-why-world-models/03-classic-world-models",
          },
          {
            text: "1.4. 体验世界模型（交互式驾驶）",
            link: "/chapters/01-why-world-models/04-imagine-driving",
          },
        ],
      },
      {
        text: "2. 深度学习与表征基础",
        collapsed: false,
        items: [
          {
            text: "2.1. 视觉基础模型（CNN与ViT）",
            link: "/chapters/02-foundations/01-cnn-and-vit",
          },
          {
            text: "2.2. 序列模型（RNN与Transformer）",
            link: "/chapters/02-foundations/02-rnn-and-transformer",
          },
          {
            text: "2.3. 空间离散化（VAE与VQ）",
            link: "/chapters/02-foundations/03-vae-and-vq",
          },
          {
            text: "2.4. 生成模型（自回归与扩散）",
            link: "/chapters/02-foundations/04-autoregressive-and-diffusion",
          },
          {
            text: "2.5. 基础模块的从零开始实现",
            link: "/chapters/02-foundations/05-basic-components-scratch",
          },
          {
            text: "2.6. 基础模块的简洁实现",
            link: "/chapters/02-foundations/06-basic-components-concise",
          },
        ],
      },
      {
        text: "3. 数据与强化学习基础",
        collapsed: false,
        items: [
          {
            text: "3.1. 经历与状态转移",
            link: "/chapters/03-data-and-first-model/01-episodes-and-transitions",
          },
          {
            text: "3.2. 经验回放池",
            link: "/chapters/03-data-and-first-model/02-replay-buffer-and-splits",
          },
          {
            text: "3.3. 价值函数与策略梯度",
            link: "/chapters/03-data-and-first-model/03-value-policy-gradient",
          },
          {
            text: "3.4. 模型预测控制与交叉熵方法",
            link: "/chapters/03-data-and-first-model/04-mpc-and-cem",
          },
          {
            text: "3.5. 强化学习基础的从零开始实现",
            link: "/chapters/03-data-and-first-model/05-rl-foundation-scratch",
          },
          {
            text: "3.6. 强化学习基础的简洁实现",
            link: "/chapters/03-data-and-first-model/06-rl-foundation-concise",
          },
        ],
      },
      {
        text: "4. 潜在动力学模型",
        collapsed: false,
        items: [
          {
            text: "4.1. 变分自编码器与循环网络的结合（World Models）",
            link: "/chapters/04-latent-dynamics/01-world-models",
          },
          {
            text: "4.2. 循环状态空间模型（RSSM）",
            link: "/chapters/04-latent-dynamics/02-rssm",
          },
          {
            text: "4.3. 潜空间规划（PlaNet）",
            link: "/chapters/04-latent-dynamics/03-planet",
          },
          {
            text: "4.4. 想象中的策略梯度（DreamerV1）",
            link: "/chapters/04-latent-dynamics/04-dreamer-v1",
          },
          {
            text: "4.5. 分类分布与离散化（DreamerV2/V3）",
            link: "/chapters/04-latent-dynamics/05-dreamer-v2-v3",
          },
          {
            text: "4.6. 隐式树搜索（MuZero）",
            link: "/chapters/04-latent-dynamics/06-muzero",
          },
          {
            text: "4.7. 循环状态空间模型的从零开始实现",
            link: "/chapters/04-latent-dynamics/07-rssm-scratch",
          },
          {
            text: "4.8. 想象训练的简洁实现",
            link: "/chapters/04-latent-dynamics/08-dreamer-concise",
          },
        ],
      },
      {
        text: "5. 交互式视频生成",
        collapsed: false,
        items: [
          {
            text: "5.1. 动作条件视频预测（SVG）",
            link: "/chapters/05-interactive-video/01-video-prediction-svg",
          },
          {
            text: "5.2. 视频词元化与自回归生成（VideoPoet）",
            link: "/chapters/05-interactive-video/02-tokenizer-videopoet",
          },
          {
            text: "5.3. 基于扩散的物理模拟（Sora）",
            link: "/chapters/05-interactive-video/03-diffusion-sora",
          },
          {
            text: "5.4. 实时交互与键值缓存",
            link: "/chapters/05-interactive-video/04-kv-cache-realtime",
          },
          {
            text: "5.5. 交互式视频模型的从零开始实现",
            link: "/chapters/05-interactive-video/05-interactive-video-scratch",
          },
          {
            text: "5.6. 受动作控制的视频小世界",
            link: "/chapters/05-interactive-video/06-controllable-video-concise",
          },
        ],
      },
      {
        text: "6. 联合嵌入预测架构",
        collapsed: false,
        items: [
          {
            text: "6.1. 联合嵌入与特征预测（I-JEPA）",
            link: "/chapters/06-jepa/01-jepa-foundation",
          },
          {
            text: "6.2. 掩码机制与表示坍缩",
            link: "/chapters/06-jepa/02-mask-collapse",
          },
          {
            text: "6.3. 目标网络（EMA）",
            link: "/chapters/06-jepa/03-target-network-ema",
          },
          {
            text: "6.4. 动作条件特征预测（Action-JEPA）",
            link: "/chapters/06-jepa/04-action-jepa",
          },
          {
            text: "6.5. 联合嵌入预测架构的从零开始实现",
            link: "/chapters/06-jepa/05-jepa-scratch",
          },
          {
            text: "6.6. 视频 JEPA 的简洁实现",
            link: "/chapters/06-jepa/06-video-jepa-concise",
          },
        ],
      },
      {
        text: "7. 具身策略大模型",
        collapsed: false,
        items: [
          {
            text: "7.1. 多模态异构观测（视觉、触觉与本体感觉）",
            link: "/chapters/07-robot-policy/01-multimodal-observation",
          },
          {
            text: "7.2. 灵巧手与灵巧操作（Dexterous Manipulation）",
            link: "/chapters/07-robot-policy/02-dexterous-manipulation",
          },
          {
            text: "7.3. 双足人形与全身控制（Humanoid & WBC）",
            link: "/chapters/07-robot-policy/03-humanoid-wbc",
          },
          {
            text: "7.4. 模仿学习基线（行为克隆）",
            link: "/chapters/07-robot-policy/04-behavior-cloning",
          },
          {
            text: "7.5. 扩散策略（Diffusion Policy）",
            link: "/chapters/07-robot-policy/05-diffusion-policy",
          },
          {
            text: "7.6. 动作分块预测（ACT）",
            link: "/chapters/07-robot-policy/06-action-chunking-act",
          },
          {
            text: "7.7. 视觉语言动作模型（RT-X）",
            link: "/chapters/07-robot-policy/07-vla-rtx",
          },
          {
            text: "7.8. 原生具身大模型（OpenVLA）",
            link: "/chapters/07-robot-policy/08-openvla",
          },
          {
            text: "7.9. 世界动作模型（VLA-JEPA / WAM）",
            link: "/chapters/07-robot-policy/09-vla-jepa-wam",
          },
          {
            text: "7.10. 扩散策略的从零开始实现",
            link: "/chapters/07-robot-policy/10-diffusion-policy-scratch",
          },
          {
            text: "7.11. 把世界模型接上身体",
            link: "/chapters/07-robot-policy/11-world-model-body-loop",
          },
        ],
      },
      {
        text: "8. 具身仿真与现实迁移",
        collapsed: false,
        items: [
          {
            text: "8.1. 物理引擎与接触力模拟（MuJoCo）",
            link: "/chapters/08-robot-sim/01-physics-mujoco",
          },
          {
            text: "8.2. 并行仿真与合成数据（Isaac）",
            link: "/chapters/08-robot-sim/02-simulation-isaac",
          },
          {
            text: "8.3. 现实差距与域随机化（Domain Randomization）",
            link: "/chapters/08-robot-sim/03-domain-randomization",
          },
          {
            text: "8.4. 特权蒸馏与现实迁移（Sim2Real）",
            link: "/chapters/08-robot-sim/04-privilege-distill-sim2real",
          },
          {
            text: "8.5. 想象强化学习与自我改进（RISE）",
            link: "/chapters/08-robot-sim/05-imagination-rl-rise",
          },
          {
            text: "8.6. 遥操作与人在回路（HIL-SERL）",
            link: "/chapters/08-robot-sim/06-teleop-hil-serl",
          },
          {
            text: "8.7. 现实迁移的简洁实现",
            link: "/chapters/08-robot-sim/07-sim2real-concise",
          },
          {
            text: "8.8. 具身规划的从零开始实现",
            link: "/chapters/08-robot-sim/08-embodied-planning-scratch",
          },
        ],
      },
      {
        text: "9. 空间世界与自动驾驶",
        collapsed: false,
        items: [
          {
            text: "9.1. 相机几何与多视角投影",
            link: "/chapters/09-spatial-worlds/01-camera-geometry",
          },
          {
            text: "9.2. 鸟瞰图与三维占用网格（BEV）",
            link: "/chapters/09-spatial-worlds/02-bev-occupancy",
          },
          {
            text: "9.3. 神经辐射场与三维高斯（NeRF/3DGS）",
            link: "/chapters/09-spatial-worlds/03-nerf-3dgs",
          },
          {
            text: "9.4. 时空四维场景预测",
            link: "/chapters/09-spatial-worlds/04-four-dimensional-worlds",
          },
          {
            text: "9.5. 自动驾驶世界模型",
            link: "/chapters/09-spatial-worlds/05-driving-world-models",
          },
          {
            text: "9.6. 占用网格预测的从零开始实现",
            link: "/chapters/09-spatial-worlds/06-occupancy-scratch",
          },
          {
            text: "9.7. 驾驶场景下的四维世界模型",
            link: "/chapters/09-spatial-worlds/07-four-d-driving-concise",
          },
        ],
      },
      {
        text: "10. 评测与研究设计",
        collapsed: false,
        items: [
          {
            text: "10.1. 怎样评价世界模型",
            link: "/chapters/10-evaluate-and-invent/01-evaluate-principles",
          },
          {
            text: "10.2. 世界模型的系统评测",
            link: "/chapters/10-evaluate-and-invent/02-systematic-evaluation",
          },
          {
            text: "10.3. 失效分析与模型改进",
            link: "/chapters/10-evaluate-and-invent/03-failure-analysis",
          },
          {
            text: "10.4. 设计新的世界模型",
            link: "/chapters/10-evaluate-and-invent/04-next-world-model",
          },
        ],
      },
      {
        text: "附录",
        collapsed: false,
        items: [
          {
            text: "A. 数学底座与术语速查",
            link: "/appendices/math-code-glossary",
          },
          {
            text: "B. 数据集、算力与交付标准",
            link: "/appendices/data-compute-delivery",
          },
          {
            text: "C. 经典文献与产业地图",
            link: "/appendices/papers-benchmarks-industry",
          },
          {
            text: "D. 相关领域对照",
            link: "/appendices/neighboring-fields",
          },
        ],
      },
    ],
  },
});
