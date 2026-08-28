import { defineConfig } from "vitepress";

export default defineConfig({
  lang: "zh-CN",
  title: "动手学世界模型",
  description: "从看见、记住和预测，到在想象中规划与行动",
  base: process.env.BASE || "/hands-on-world-models/",
  cleanUrls: false,
  lastUpdated: true,
  markdown: {
    math: true,
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
        link: "/guide/world-model-intro",
      },
      {
        text: "预备知识",
        link: "/chapters/02-foundations/01-tensors-and-trajectories",
      },
      {
        text: "五个方向",
        link: "/chapters/04-decision-and-planning/01-latent-world-model",
      },
      {
        text: "评价与研究",
        link: "/chapters/09-evaluate-and-invent/01-perception-and-utility",
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
          { text: "绪论", link: "/guide/world-model-intro" },
          { text: "世界模型八十年", link: "/guide/world-model-history" },
        ],
      },
      {
        text: "1. 引言",
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
            text: "1.4. 动手：在想象中驾驶（交互式体验）",
            link: "/chapters/01-why-world-models/04-imagine-driving",
          },
        ],
      },
      {
        text: "2. 预备知识",
        collapsed: false,
        items: [
          {
            text: "2.1. 张量与轨迹",
            link: "/chapters/02-foundations/01-tensors-and-trajectories",
          },
          {
            text: "2.2. 卷积神经网络与视觉 Transformer",
            link: "/chapters/02-foundations/02-cnn-and-vit",
          },
          {
            text: "2.3. 循环神经网络与注意力机制",
            link: "/chapters/02-foundations/03-memory-and-dynamics",
          },
          {
            text: "2.4. 变分自编码器与向量量化",
            link: "/chapters/02-foundations/04-compression-and-generation",
          },
          {
            text: "2.5. 坐标系与三维表示",
            link: "/chapters/02-foundations/05-space-representations",
          },
          {
            text: "2.6. 价值函数与策略梯度",
            link: "/chapters/02-foundations/06-value-policy-planner",
          },
          {
            text: "2.7. 动手：核心组件的简洁实现",
            link: "/chapters/02-foundations/07-basic-experiments",
          },
        ],
      },
      {
        text: "3. 数据与第一个世界模型",
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
            text: "3.3. 转移概率与极大似然估计",
            link: "/chapters/03-data-and-first-model/03-first-learned-world",
          },
          {
            text: "3.4. 多步预测与累积误差",
            link: "/chapters/03-data-and-first-model/04-basic-checks",
          },
          {
            text: "3.5. 动手：表格型世界模型的从零开始实现",
            link: "/chapters/03-data-and-first-model/05-learn-a-table-world",
          },
          {
            text: "3.6. 动手：神经网络世界模型的简洁实现",
            link: "/chapters/03-data-and-first-model/06-learnable-world",
          },
        ],
      },
      {
        text: "4. 决策与规划",
        collapsed: false,
        items: [
          {
            text: "4.1. 潜在动力学模型",
            link: "/chapters/04-decision-and-planning/01-latent-world-model",
          },
          {
            text: "4.2. 循环状态空间模型（RSSM）",
            link: "/chapters/04-decision-and-planning/02-rssm-training",
          },
          {
            text: "4.3. 交叉熵方法与模型预测控制",
            link: "/chapters/04-decision-and-planning/03-planet-and-cem",
          },
          {
            text: "4.4. 想象训练（Dreamer）",
            link: "/chapters/04-decision-and-planning/04-dreamer-imagination",
          },
          {
            text: "4.5. 蒙特卡洛树搜索（MuZero）",
            link: "/chapters/04-decision-and-planning/05-muzero",
          },
          {
            text: "4.6. 动手：循环状态空间模型的从零开始实现",
            link: "/chapters/04-decision-and-planning/06-reproduce-world-models",
          },
          {
            text: "4.7. 动手：想象训练的简洁实现",
            link: "/chapters/04-decision-and-planning/07-decision-and-planning",
          },
        ],
      },
      {
        text: "5. 交互式视频",
        collapsed: false,
        items: [
          {
            text: "5.1. 自回归视频生成",
            link: "/chapters/05-interactive-video/01-video-data",
          },
          {
            text: "5.2. 图像与视频词元化（Tokenizer）",
            link: "/chapters/05-interactive-video/02-vq-tokenizer",
          },
          {
            text: "5.3. 动作条件注入",
            link: "/chapters/05-interactive-video/03-action-conditioning",
          },
          {
            text: "5.4. 键值缓存（KV Cache）与实时生成",
            link: "/chapters/05-interactive-video/04-memory-drift-realtime",
          },
          {
            text: "5.5. 动手：交互式视频模型的从零开始实现",
            link: "/chapters/05-interactive-video/05-interactive-video",
          },
          {
            text: "5.6. 动手：受动作控制的视频小世界",
            link: "/chapters/05-interactive-video/06-controllable-video",
          },
        ],
      },
      {
        text: "6. 联合嵌入预测架构",
        collapsed: false,
        items: [
          {
            text: "6.1. 联合嵌入与特征预测",
            link: "/chapters/06-jepa/01-feature-prediction",
          },
          {
            text: "6.2. 掩码机制与表示坍缩",
            link: "/chapters/06-jepa/02-mask-ema-collapse",
          },
          {
            text: "6.3. 目标网络（EMA）",
            link: "/chapters/06-jepa/03-video-jepa",
          },
          {
            text: "6.4. 动作条件特征预测",
            link: "/chapters/06-jepa/04-action-jepa",
          },
          {
            text: "6.5. 动手：联合嵌入预测架构的从零开始实现",
            link: "/chapters/06-jepa/05-jepa",
          },
          {
            text: "6.6. 动手：视频 JEPA 的简洁实现",
            link: "/chapters/06-jepa/06-video-jepa",
          },
        ],
      },
      {
        text: "7. 具身智能与机器人",
        collapsed: false,
        items: [
          {
            text: "7.1. 机器人数据与异构观测",
            link: "/chapters/07-robot-vla/01-robot-interfaces",
          },
          {
            text: "7.2. 行为克隆与扩散策略",
            link: "/chapters/07-robot-vla/02-imitation-and-policies",
          },
          {
            text: "7.3. 视觉语言动作模型（VLA）",
            link: "/chapters/07-robot-vla/03-vision-language-action",
          },
          {
            text: "7.4. 接触力与全身控制",
            link: "/chapters/07-robot-vla/04-contact-and-whole-body",
          },
          {
            text: "7.5. 模拟器与现实迁移",
            link: "/chapters/07-robot-vla/05-simulators-and-sim2real",
          },
          {
            text: "7.6. 动手：扩散策略的从零开始实现",
            link: "/chapters/07-robot-vla/06-data-to-generative-policy",
          },
          {
            text: "7.7. 动手：现实迁移的简洁实现",
            link: "/chapters/07-robot-vla/07-sim2real",
          },
          {
            text: "7.8. 动手：把世界模型接上身体（毕业设计）",
            link: "/chapters/07-robot-vla/08-world-model-meets-body",
          },
        ],
      },
      {
        text: "8. 空间世界与自动驾驶",
        collapsed: false,
        items: [
          {
            text: "8.1. 相机模型与多视角投影",
            link: "/chapters/08-spatial-worlds/01-camera-geometry",
          },
          {
            text: "8.2. 鸟瞰图（BEV）与三维占用网格",
            link: "/chapters/08-spatial-worlds/02-bev-and-occupancy",
          },
          {
            text: "8.3. 神经辐射场（NeRF）与三维高斯",
            link: "/chapters/08-spatial-worlds/03-nerf-3dgs-mesh",
          },
          {
            text: "8.4. 时空四维场景预测",
            link: "/chapters/08-spatial-worlds/04-four-dimensional-worlds",
          },
          {
            text: "8.5. 自动驾驶世界模型",
            link: "/chapters/08-spatial-worlds/05-driving-world-models",
          },
          {
            text: "8.6. 动手：占用网格预测的从零开始实现",
            link: "/chapters/08-spatial-worlds/06-occupancy-prediction",
          },
          {
            text: "8.7. 动手：驾驶场景下的四维世界模型",
            link: "/chapters/08-spatial-worlds/07-four-d-driving",
          },
        ],
      },
      {
        text: "9. 评测与研究设计",
        collapsed: false,
        items: [
          {
            text: "9.1. 怎样评价世界模型",
            link: "/chapters/09-evaluate-and-invent/01-perception-and-utility",
          },
          {
            text: "9.2. 动手：世界模型的系统评测",
            link: "/chapters/09-evaluate-and-invent/02-systematic-evaluation",
          },
          {
            text: "9.3. 动手：失效分析与模型改进",
            link: "/chapters/09-evaluate-and-invent/03-failure-to-next-model",
          },
          {
            text: "9.4. 动手：设计新的世界模型",
            link: "/chapters/09-evaluate-and-invent/04-next-world-model",
          },
        ],
      },
      {
        text: "附录",
        collapsed: false,
        items: [
          {
            text: "A. 数学、代码与术语速查",
            link: "/appendices/math-code-glossary",
          },
          {
            text: "B. 数据、算力与交付标准",
            link: "/appendices/data-compute-delivery",
          },
          {
            text: "C. 论文、榜单与产业地图",
            link: "/appendices/papers-benchmarks-industry",
          },
          {
            text: "D. 邻近课程与覆盖对照",
            link: "/appendices/neighboring-fields",
          },
        ],
      },
    ],
    socialLinks: [
      {
        icon: "github",
        link: "https://github.com/walkinglabs/hands-on-world-models",
      },
    ],
    editLink: {
      pattern:
        "https://github.com/walkinglabs/hands-on-world-models/edit/main/docs/:path",
      text: "在 GitHub 上改进本页",
    },
    footer: {
      message: "以问题为起点，以可重复实验为证据。",
      copyright: "CC BY-NC-SA 4.0 · Walking Labs",
    },
    search: {
      provider: "local",
      options: {
        translations: {
          button: { buttonText: "搜索", buttonAriaLabel: "搜索" },
          modal: {
            noResultsText: "没有找到结果",
            resetButtonTitle: "清除搜索",
            footer: {
              selectText: "选择",
              navigateText: "切换",
              closeText: "关闭",
            },
          },
        },
      },
    },
  },
});
