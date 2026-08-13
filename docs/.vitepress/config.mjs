import { defineConfig } from "vitepress";

export default defineConfig({
  lang: "zh-CN",
  title: "动手学世界模型",
  description: "从看见、记住和预测，到在想象中规划与行动",
  base: process.env.BASE || "/hands-on-world-models/",
  cleanUrls: false,
  lastUpdated: true,
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
      { text: "开始学习", link: "/guide/start" },
      { text: "课程总纲", link: "/课程总纲" },
      { text: "第 0 章", link: "/chapters/00-why-world-models/" },
      {
        text: "GitHub",
        link: "https://github.com/walkinglabs/hands-on-world-models",
      },
    ],
    sidebar: [
      {
        text: "课程",
        items: [
          { text: "怎样使用本书", link: "/guide/start" },
          { text: "完整课程总纲", link: "/课程总纲" },
          { text: "数据与实验状态", link: "/data-status" },
          { text: "数据使用指南", link: "/data-guide" },
          { text: "教师指南", link: "/teacher-guide" },
        ],
      },
      {
        text: "第一部分 · 重新发明",
        collapsed: false,
        items: [
          {
            text: "第 0 章 · 为什么需要世界模型",
            link: "/chapters/00-why-world-models/",
            items: [
              {
                text: "0.1 · 一张照片为什么不够",
                link: "/chapters/00-why-world-models/00-01-current-observation",
              },
              {
                text: "0.2 · 没执行的动作会怎样",
                link: "/chapters/00-why-world-models/00-02-action-conditioned-future",
              },
              {
                text: "0.3 · 从一步到多步规划",
                link: "/chapters/00-why-world-models/00-03-rollout-planning-policy",
              },
              {
                text: "0.4 · 从经历学习转移",
                link: "/chapters/00-why-world-models/00-04-learned-dynamics",
              },
              {
                text: "0.5 · 经典 World Models",
                link: "/chapters/00-why-world-models/00-05-classic-world-models",
              },
            ],
          },
          { text: "F0 · 九格世界（Notebook）", link: "/labs/f0" },
        ],
      },
      {
        text: "第二部分 · 共同基础",
        collapsed: false,
        items: [
          {
            text: "第 1 章 · 表示世界并学出模型",
            link: "/chapters/01-foundations/",
            items: [
              {
                text: "1.1 · 张量、时间与轨迹",
                link: "/chapters/01-foundations/01-01-tensors-and-trajectories",
              },
              {
                text: "1.2 · CNN 与 ViT",
                link: "/chapters/01-foundations/01-02-cnn-and-vit",
              },
              {
                text: "1.3 · 记忆与动态",
                link: "/chapters/01-foundations/01-03-memory-and-dynamics",
              },
              {
                text: "1.4 · 压缩与生成",
                link: "/chapters/01-foundations/01-04-compression-and-generation",
              },
              {
                text: "1.5 · 空间表示",
                link: "/chapters/01-foundations/01-05-space-representations",
              },
              {
                text: "1.6 · Value、Policy 与 Planner",
                link: "/chapters/01-foundations/01-06-value-policy-planner",
              },
              {
                text: "1.7 · 数据与第一台模型",
                link: "/chapters/01-foundations/01-07-data-and-first-model",
              },
            ],
          },
          { text: "F1–F3 · 共同基础实验", link: "/labs/foundations" },
          { text: "PA0 · 第一台可学习世界", link: "/assignments/pa0" },
        ],
      },
      {
        text: "第三部分 · 五选一",
        collapsed: true,
        items: [
          {
            text: "第 2 章 · 决策与规划",
            link: "/chapters/02-decision-and-planning/",
            items: [
              {
                text: "2.1 · Latent World Model",
                link: "/chapters/02-decision-and-planning/02-01-latent-world-model",
              },
              {
                text: "2.2 · RSSM 训练",
                link: "/chapters/02-decision-and-planning/02-02-rssm-training",
              },
              {
                text: "2.3 · PlaNet 与 CEM",
                link: "/chapters/02-decision-and-planning/02-03-planet-and-cem",
              },
              {
                text: "2.4 · Dreamer 想象学习",
                link: "/chapters/02-decision-and-planning/02-04-dreamer-imagination",
              },
              {
                text: "2.5 · MuZero",
                link: "/chapters/02-decision-and-planning/02-05-muzero",
              },
            ],
          },
          { text: "A1–A2 · 路线实验", link: "/labs/route-a" },
          { text: "PA1-A · Dreamer-lite", link: "/assignments/pa1-a" },
          {
            text: "第 3 章 · 可交互视频",
            link: "/chapters/03-interactive-video/",
            items: [
              {
                text: "3.1 · 视频与动作数据",
                link: "/chapters/03-interactive-video/03-01-video-data",
              },
              {
                text: "3.2 · VQ Tokenizer",
                link: "/chapters/03-interactive-video/03-02-vq-tokenizer",
              },
              {
                text: "3.3 · 动作条件 Transformer",
                link: "/chapters/03-interactive-video/03-03-action-transformer",
              },
              {
                text: "3.4 · Diffusion 与评价",
                link: "/chapters/03-interactive-video/03-04-diffusion-and-evaluation",
              },
            ],
          },
          {
            text: "第 4 章 · JEPA 抽象预测",
            link: "/chapters/04-jepa/",
            items: [
              {
                text: "4.1 · 特征预测",
                link: "/chapters/04-jepa/04-01-feature-prediction",
              },
              {
                text: "4.2 · Mask、EMA 与坍缩",
                link: "/chapters/04-jepa/04-02-mask-ema-collapse",
              },
              {
                text: "4.3 · Video-JEPA",
                link: "/chapters/04-jepa/04-03-video-jepa",
              },
              {
                text: "4.4 · Action-JEPA",
                link: "/chapters/04-jepa/04-04-action-jepa",
              },
            ],
          },
          { text: "B1–C2 · 路线实验", link: "/labs/route-bc" },
          { text: "PA1-B · 互动视频", link: "/assignments/pa1-b" },
          { text: "PA1-C · Tiny JEPA", link: "/assignments/pa1-c" },
          {
            text: "第 5 章 · VLA 与机器人",
            link: "/chapters/05-robot-vla/",
            items: [
              {
                text: "5.1 · 数据与行为克隆",
                link: "/chapters/05-robot-vla/05-01-robot-data-and-bc",
              },
              {
                text: "5.2 · Vision-Language-Action",
                link: "/chapters/05-robot-vla/05-02-vision-language-action",
              },
              {
                text: "5.3 · Action Chunk",
                link: "/chapters/05-robot-vla/05-03-action-chunk",
              },
              {
                text: "5.4 · World Model Checker",
                link: "/chapters/05-robot-vla/05-04-world-model-checker",
              },
            ],
          },
          {
            text: "第 6 章 · 空间世界",
            link: "/chapters/06-spatial-worlds/",
            items: [
              {
                text: "6.1 · 相机几何",
                link: "/chapters/06-spatial-worlds/06-01-camera-geometry",
              },
              {
                text: "6.2 · BEV 与 Occupancy",
                link: "/chapters/06-spatial-worlds/06-02-bev-and-occupancy",
              },
              {
                text: "6.3 · NeRF、3DGS 与 Mesh",
                link: "/chapters/06-spatial-worlds/06-03-nerf-3dgs-mesh",
              },
              {
                text: "6.4 · 4D 世界",
                link: "/chapters/06-spatial-worlds/06-04-four-dimensional-worlds",
              },
              {
                text: "6.5 · 驾驶世界模型",
                link: "/chapters/06-spatial-worlds/06-05-driving-world-models",
              },
            ],
          },
          { text: "D1–E2 · 路线实验", link: "/labs/route-de" },
          { text: "PA1-D · Tiny VLA", link: "/assignments/pa1-d" },
          { text: "PA1-E · 空间二选一", link: "/assignments/pa1-e" },
        ],
      },
      {
        text: "第四部分 · 设计下一台模型",
        collapsed: false,
        items: [
          {
            text: "第 7 章 · 评价与研究",
            link: "/chapters/07-evaluate-and-invent/",
            items: [
              {
                text: "7.1 · 基线与 Horizon",
                link: "/chapters/07-evaluate-and-invent/07-01-baselines-and-horizons",
              },
              {
                text: "7.2 · 反事实与 OOD",
                link: "/chapters/07-evaluate-and-invent/07-02-counterfactual-and-ood",
              },
              {
                text: "7.3 · 24GB 运行证据",
                link: "/chapters/07-evaluate-and-invent/07-03-hardware-evidence",
              },
              {
                text: "7.4 · 设计下一台模型",
                link: "/chapters/07-evaluate-and-invent/07-04-next-world-model",
              },
            ],
          },
          { text: "Z0 · 审问世界模型", link: "/labs/z0" },
          { text: "PA2 · 下一台模型", link: "/assignments/pa2" },
          { text: "运行证据与 24GB", link: "/run-evidence" },
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
