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
    logo: "/logo.png",
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
          { text: "课程总纲", link: "/课程总纲" },
          { text: "世界模型简史", link: "/guide/world-model-history" },
        ],
      },
      {
        text: "1. 世界模型的基本问题",
        collapsed: false,
        items: [
          {
            text: "1.1. 观察、状态与历史",
            link: "/chapters/01-why-world-models/01-current-observation",
          },
          {
            text: "1.2. 动作条件预测",
            link: "/chapters/01-why-world-models/02-action-conditioned-future",
          },
          {
            text: "1.3. 多步预测与规划",
            link: "/chapters/01-why-world-models/03-rollout-planning-policy",
          },
          {
            text: "1.4. 从经历学习动态",
            link: "/chapters/01-why-world-models/04-learned-dynamics",
          },
          {
            text: "1.5. 经典世界模型",
            link: "/chapters/01-why-world-models/05-classic-world-models",
          },
          {
            text: "1.6. 动手：从零重新发明世界模型",
            link: "/chapters/01-why-world-models/06-invent-a-world-model",
          },
        ],
      },
      {
        text: "2. 预备知识",
        collapsed: false,
        items: [
          {
            text: "2.1. 张量、时间与轨迹",
            link: "/chapters/02-foundations/01-tensors-and-trajectories",
          },
          {
            text: "2.2. 图像编码器",
            link: "/chapters/02-foundations/02-cnn-and-vit",
          },
          {
            text: "2.3. 记忆与动态",
            link: "/chapters/02-foundations/03-memory-and-dynamics",
          },
          {
            text: "2.4. 压缩与生成",
            link: "/chapters/02-foundations/04-compression-and-generation",
          },
          {
            text: "2.5. 空间表示",
            link: "/chapters/02-foundations/05-space-representations",
          },
          {
            text: "2.6. 决策接口",
            link: "/chapters/02-foundations/06-value-policy-planner",
          },
          {
            text: "2.7. 经验回放与第一个模型",
            link: "/chapters/02-foundations/07-data-and-first-model",
          },
          {
            text: "2.8. 动手：基础实验",
            link: "/chapters/02-foundations/08-basic-experiments",
          },
        ],
      },
      {
        text: "3. 数据与第一个模型",
        collapsed: false,
        items: [
          {
            text: "3.1. 经验的存储",
            link: "/chapters/03-data-and-first-model/01-episodes-and-transitions",
          },
          {
            text: "3.2. Replay Buffer 与数据切分",
            link: "/chapters/03-data-and-first-model/02-replay-buffer-and-splits",
          },
          {
            text: "3.3. 从经验学习第一个模型",
            link: "/chapters/03-data-and-first-model/03-first-learned-world",
          },
          {
            text: "3.4. 动手：第一台可学习世界模型",
            link: "/chapters/03-data-and-first-model/04-learn-a-table-world",
          },
          {
            text: "PA0 · 动手：重新发明一台可学习世界模型",
            link: "/assignments/pa0",
          },
        ],
      },
      {
        text: "4. 决策与规划",
        collapsed: false,
        items: [
          {
            text: "4.1. 潜在状态世界模型",
            link: "/chapters/04-decision-and-planning/01-latent-world-model",
          },
          {
            text: "4.2. RSSM：记忆与不确定性",
            link: "/chapters/04-decision-and-planning/02-rssm-training",
          },
          {
            text: "4.3. PlaNet 与 CEM",
            link: "/chapters/04-decision-and-planning/03-planet-and-cem",
          },
          {
            text: "4.4. Dreamer：在想象中训练",
            link: "/chapters/04-decision-and-planning/04-dreamer-imagination",
          },
          {
            text: "4.5. MuZero 与蒙特卡洛树搜索",
            link: "/chapters/04-decision-and-planning/05-muzero",
          },
          {
            text: "4.6. 动手：复现 World Models",
            link: "/chapters/04-decision-and-planning/06-reproduce-world-models",
          },
          {
            text: "4.7. 动手：决策与规划实验",
            link: "/chapters/04-decision-and-planning/07-decision-and-planning",
          },
          {
            text: "PA1-A · 动手：做出一台 Dreamer-lite",
            link: "/assignments/pa1-a",
          },
        ],
      },
      {
        text: "5. 交互式视频",
        collapsed: false,
        items: [
          {
            text: "5.1. 从视频生成到视频世界模型",
            link: "/chapters/05-interactive-video/01-video-data",
          },
          {
            text: "5.2. 先决定预测什么",
            link: "/chapters/05-interactive-video/02-vq-tokenizer",
          },
          {
            text: "5.3. AR、Diffusion 与 Diffusion Forcing",
            link: "/chapters/05-interactive-video/03-action-transformer",
          },
          {
            text: "5.4. 动作、记忆、长时生成与评价",
            link: "/chapters/05-interactive-video/04-diffusion-and-evaluation",
          },
          {
            text: "5.5. 动手：交互视频实验",
            link: "/chapters/05-interactive-video/05-interactive-video",
          },
          {
            text: "PA1-B · 动手：做出一个听从按键的视频小世界",
            link: "/assignments/pa1-b",
          },
        ],
      },
      {
        text: "6. JEPA：特征空间预测",
        collapsed: false,
        items: [
          {
            text: "6.1. 预测特征而非像素",
            link: "/chapters/06-jepa/01-feature-prediction",
          },
          {
            text: "6.2. 掩码、EMA 与表示坍缩",
            link: "/chapters/06-jepa/02-mask-ema-collapse",
          },
          {
            text: "6.3. 视频 JEPA",
            link: "/chapters/06-jepa/03-video-jepa",
          },
          {
            text: "6.4. 动作条件 JEPA",
            link: "/chapters/06-jepa/04-action-jepa",
          },
          {
            text: "6.5. 动手：JEPA 实验",
            link: "/chapters/06-jepa/05-jepa",
          },
          {
            text: "PA1-C · 动手：训练并审问一个 Tiny Video-JEPA",
            link: "/assignments/pa1-c",
          },
        ],
      },
      {
        text: "7. 具身智能与机器人",
        collapsed: false,
        items: [
          {
            text: "7.1. 机器人数据与行为克隆",
            link: "/chapters/07-robot-vla/01-robot-data-and-bc",
          },
          {
            text: "7.2. 视觉-语言-动作模型",
            link: "/chapters/07-robot-vla/02-vision-language-action",
          },
          {
            text: "7.3. 动作分块与多模态动作",
            link: "/chapters/07-robot-vla/03-action-chunk",
          },
          {
            text: "7.4. 世界模型检查器",
            link: "/chapters/07-robot-vla/04-world-model-checker",
          },
          {
            text: "7.5. 动手：机器人与 VLA 实验",
            link: "/chapters/07-robot-vla/05-robot-vla",
          },
          {
            text: "PA1-D · 动手：Tiny VLA 与 World-Model Checker",
            link: "/assignments/pa1-d",
          },
        ],
      },
      {
        text: "8. 空间世界与自动驾驶",
        collapsed: false,
        items: [
          {
            text: "8.1. 相机几何与投影",
            link: "/chapters/08-spatial-worlds/01-camera-geometry",
          },
          {
            text: "8.2. BEV、占用网格与 LSS",
            link: "/chapters/08-spatial-worlds/02-bev-and-occupancy",
          },
          {
            text: "8.3. NeRF、3DGS 与网格",
            link: "/chapters/08-spatial-worlds/03-nerf-3dgs-mesh",
          },
          {
            text: "8.4. 四维世界（4D）",
            link: "/chapters/08-spatial-worlds/04-four-dimensional-worlds",
          },
          {
            text: "8.5. 驾驶世界模型与未来占用",
            link: "/chapters/08-spatial-worlds/05-driving-world-models",
          },
          {
            text: "8.6. 动手：空间世界实验",
            link: "/chapters/08-spatial-worlds/06-spatial-world",
          },
          {
            text: "PA1-E · 动手：空间世界二选一",
            link: "/assignments/pa1-e",
          },
        ],
      },
      {
        text: "9. 评测与研究设计",
        collapsed: false,
        items: [
          {
            text: "9.1. 感知质量与功能效用",
            link: "/chapters/09-evaluate-and-invent/01-perception-and-utility",
          },
          {
            text: "9.2. 动手：世界模型的系统评测",
            link: "/chapters/09-evaluate-and-invent/02-systematic-evaluation",
          },
          {
            text: "9.3. 动手：从失效到下一台世界模型",
            link: "/chapters/09-evaluate-and-invent/03-failure-to-next-model",
          },
          {
            text: "PA2 · 动手：设计下一台世界模型",
            link: "/assignments/pa2",
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
