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
        link: "/chapters/01-foundations/01-01-tensors-and-trajectories",
      },
      {
        text: "五个方向",
        link: "/chapters/03-decision-and-planning/03-01-latent-world-model",
      },
      {
        text: "评价与研究",
        link: "/chapters/08-evaluate-and-invent/08-01-baselines-and-horizons",
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
          { text: "世界模型简史", link: "/guide/world-model-history" },
        ],
      },
      {
        text: "0. 世界模型的基本问题",
        collapsed: false,
        items: [
          {
            text: "0.1. 观察、状态与历史",
            link: "/chapters/00-why-world-models/00-01-current-observation",
          },
          {
            text: "0.2. 动作条件预测",
            link: "/chapters/00-why-world-models/00-02-action-conditioned-future",
          },
          {
            text: "0.3. 多步预测与规划",
            link: "/chapters/00-why-world-models/00-03-rollout-planning-policy",
          },
          {
            text: "0.4. 从经验学习动态",
            link: "/chapters/00-why-world-models/00-04-learned-dynamics",
          },
          {
            text: "0.5. 经典世界模型",
            link: "/chapters/00-why-world-models/00-05-classic-world-models",
          },
          { text: "0.6. 从零实现世界模型", link: "/labs/f0" },
          {
            text: "PA0 · 重新发明一台可学习世界模型",
            link: "/assignments/pa0",
          },
        ],
      },
      {
        text: "1. 预备知识",
        collapsed: false,
        items: [
          {
            text: "1.1. 张量、时间与轨迹",
            link: "/chapters/01-foundations/01-01-tensors-and-trajectories",
          },
          {
            text: "1.2. 图像编码器",
            link: "/chapters/01-foundations/01-02-cnn-and-vit",
          },
          {
            text: "1.3. 记忆与动态",
            link: "/chapters/01-foundations/01-03-memory-and-dynamics",
          },
          {
            text: "1.4. 压缩与生成",
            link: "/chapters/01-foundations/01-04-compression-and-generation",
          },
          {
            text: "1.5. 空间表示",
            link: "/chapters/01-foundations/01-05-space-representations",
          },
          {
            text: "1.6. 决策接口",
            link: "/chapters/01-foundations/01-06-value-policy-planner",
          },
          {
            text: "1.7. 经验回放与第一个模型",
            link: "/chapters/01-foundations/01-07-data-and-first-model",
          },
          { text: "1.8. 基础实验", link: "/labs/foundations" },
        ],
      },
      {
        text: "2. 数据与第一个模型",
        collapsed: false,
        items: [
          {
            text: "2.1. 经验的存储",
            link: "/chapters/02-data-and-first-model/02-01-episodes-and-transitions",
          },
          {
            text: "2.2. Replay Buffer 与数据切分",
            link: "/chapters/02-data-and-first-model/02-02-replay-buffer-and-splits",
          },
          {
            text: "2.3. 从经验学习第一个模型",
            link: "/chapters/02-data-and-first-model/02-03-first-learned-world",
          },
        ],
      },
      {
        text: "3. 决策与规划",
        collapsed: false,
        items: [
          {
            text: "3.1. 潜在状态世界模型",
            link: "/chapters/03-decision-and-planning/03-01-latent-world-model",
          },
          {
            text: "3.2. RSSM：记忆与不确定性",
            link: "/chapters/03-decision-and-planning/03-02-rssm-training",
          },
          {
            text: "3.3. PlaNet 与 CEM",
            link: "/chapters/03-decision-and-planning/03-03-planet-and-cem",
          },
          {
            text: "3.4. Dreamer：在想象中训练",
            link: "/chapters/03-decision-and-planning/03-04-dreamer-imagination",
          },
          {
            text: "3.5. MuZero 与蒙特卡洛树搜索",
            link: "/chapters/03-decision-and-planning/03-05-muzero",
          },
          { text: "3.6. 决策与规划实验", link: "/labs/route-a" },
          {
            text: "PA1-A · 做出一台 Dreamer-lite",
            link: "/assignments/pa1-a",
          },
        ],
      },
      {
        text: "4. 交互式视频",
        collapsed: false,
        items: [
          {
            text: "4.1. 动作条件视频数据",
            link: "/chapters/04-interactive-video/04-01-video-data",
          },
          {
            text: "4.2. VQ-VAE：离散图像 token",
            link: "/chapters/04-interactive-video/04-02-vq-tokenizer",
          },
          {
            text: "4.3. 动作条件 Transformer：自回归视频生成",
            link: "/chapters/04-interactive-video/04-03-action-transformer",
          },
          {
            text: "4.4. 扩散视频生成与多步漂移",
            link: "/chapters/04-interactive-video/04-04-diffusion-and-evaluation",
          },
          { text: "4.5. 交互视频实验", link: "/labs/route-bc" },
          {
            text: "PA1-B · 做出一个听从按键的视频小世界",
            link: "/assignments/pa1-b",
          },
        ],
      },
      {
        text: "5. JEPA：特征空间预测",
        collapsed: false,
        items: [
          {
            text: "5.1. 预测特征而非像素",
            link: "/chapters/05-jepa/05-01-feature-prediction",
          },
          {
            text: "5.2. 掩码、EMA 与表示坍缩",
            link: "/chapters/05-jepa/05-02-mask-ema-collapse",
          },
          {
            text: "5.3. 视频 JEPA",
            link: "/chapters/05-jepa/05-03-video-jepa",
          },
          {
            text: "5.4. 动作条件 JEPA",
            link: "/chapters/05-jepa/05-04-action-jepa",
          },
          { text: "5.5. JEPA 实验", link: "/labs/route-bc" },
          {
            text: "PA1-C · 训练并审问一个 Tiny Video-JEPA",
            link: "/assignments/pa1-c",
          },
        ],
      },
      {
        text: "6. 具身智能与机器人",
        collapsed: false,
        items: [
          {
            text: "6.1. 机器人数据与行为克隆",
            link: "/chapters/06-robot-vla/06-01-robot-data-and-bc",
          },
          {
            text: "6.2. 视觉-语言-动作模型",
            link: "/chapters/06-robot-vla/06-02-vision-language-action",
          },
          {
            text: "6.3. 动作分块与多模态动作",
            link: "/chapters/06-robot-vla/06-03-action-chunk",
          },
          {
            text: "6.4. 世界模型检查器",
            link: "/chapters/06-robot-vla/06-04-world-model-checker",
          },
          {
            text: "6.5. 机器人与 VLA 实验",
            link: "/labs/route-de",
          },
          {
            text: "PA1-D · Tiny VLA 与 World-Model Checker",
            link: "/assignments/pa1-d",
          },
        ],
      },
      {
        text: "7. 空间世界与自动驾驶",
        collapsed: false,
        items: [
          {
            text: "7.1. 相机几何与投影",
            link: "/chapters/07-spatial-worlds/07-01-camera-geometry",
          },
          {
            text: "7.2. BEV、占用网格与 LSS",
            link: "/chapters/07-spatial-worlds/07-02-bev-and-occupancy",
          },
          {
            text: "7.3. NeRF、3DGS 与网格",
            link: "/chapters/07-spatial-worlds/07-03-nerf-3dgs-mesh",
          },
          {
            text: "7.4. 四维世界（4D）",
            link: "/chapters/07-spatial-worlds/07-04-four-dimensional-worlds",
          },
          {
            text: "7.5. 驾驶世界模型与未来占用",
            link: "/chapters/07-spatial-worlds/07-05-driving-world-models",
          },
          { text: "7.6. 空间世界实验", link: "/labs/route-de" },
          {
            text: "PA1-E · 空间世界二选一",
            link: "/assignments/pa1-e",
          },
        ],
      },
      {
        text: "8. 评价与研究设计",
        collapsed: false,
        items: [
          {
            text: "8.1. 基线与多步评价",
            link: "/chapters/08-evaluate-and-invent/08-01-baselines-and-horizons",
          },
          {
            text: "8.2. 反事实、分布外与鲁棒性",
            link: "/chapters/08-evaluate-and-invent/08-02-counterfactual-and-ood",
          },
          {
            text: "8.3. 运行证据与复现",
            link: "/chapters/08-evaluate-and-invent/08-03-hardware-evidence",
          },
          {
            text: "8.4. 失败分析与下一个世界模型",
            link: "/chapters/08-evaluate-and-invent/08-04-next-world-model",
          },
          {
            text: "8.5. 审问世界模型",
            link: "/chapters/08-evaluate-and-invent/08-05-interrogate-world-model",
          },
          {
            text: "8.6. 实现自己的世界模型",
            link: "/chapters/08-evaluate-and-invent/08-06-next-model-proposal",
          },
          {
            text: "PA2 · 设计下一台世界模型",
            link: "/assignments/pa2",
          },
          { text: "Z0 · 审问一台世界模型", link: "/labs/z0" },
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
