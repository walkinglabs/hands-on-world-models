import { defineConfig } from "vitepress";

export default defineConfig({
  lang: "zh-CN",
  title: "动手学世界模型",
  description:
    "世界模型不只识别眼前的画面；它从连续观察中推测看不见的状态，并学习时间与行动会让世界怎样变化。",
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
            text: "第 0 章 · 观察为什么不等于世界",
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
            text: "第 1 章 · 世界模型的常用组件",
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
            ],
          },
          {
            text: "第 2 章 · 数据与第一台模型",
            link: "/chapters/02-data-and-first-model/",
            items: [
              {
                text: "2.1 · Episode 与 Transition",
                link: "/chapters/02-data-and-first-model/02-01-episodes-and-transitions",
              },
              {
                text: "2.2 · Replay Buffer 与切分",
                link: "/chapters/02-data-and-first-model/02-02-replay-buffer-and-splits",
              },
              {
                text: "2.3 · 第一台可学习模型",
                link: "/chapters/02-data-and-first-model/02-03-first-learned-world",
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
            text: "第 7 章 · 空间世界",
            link: "/chapters/07-spatial-worlds/",
            items: [
              {
                text: "7.1 · 相机几何",
                link: "/chapters/07-spatial-worlds/07-01-camera-geometry",
              },
              {
                text: "7.2 · BEV 与 Occupancy",
                link: "/chapters/07-spatial-worlds/07-02-bev-and-occupancy",
              },
              {
                text: "7.3 · NeRF、3DGS 与 Mesh",
                link: "/chapters/07-spatial-worlds/07-03-nerf-3dgs-mesh",
              },
              {
                text: "7.4 · 4D 世界",
                link: "/chapters/07-spatial-worlds/07-04-four-dimensional-worlds",
              },
              {
                text: "7.5 · 驾驶世界模型",
                link: "/chapters/07-spatial-worlds/07-05-driving-world-models",
              },
            ],
          },
          { text: "E1–E2 · 空间路线实验", link: "/labs/route-de" },
          { text: "PA1-E · 空间二选一", link: "/assignments/pa1-e" },
          {
            text: "第 4 章 · 互动视频",
            link: "/chapters/04-interactive-video/",
            items: [
              {
                text: "4.1 · 从视频到世界模型",
                link: "/chapters/04-interactive-video/04-01-video-data",
              },
              {
                text: "4.2 · 先决定预测什么",
                link: "/chapters/04-interactive-video/04-02-vq-tokenizer",
              },
              {
                text: "4.3 · AR 与 Diffusion",
                link: "/chapters/04-interactive-video/04-03-action-transformer",
              },
              {
                text: "4.4 · 动作、记忆与评价",
                link: "/chapters/04-interactive-video/04-04-diffusion-and-evaluation",
              },
            ],
          },
          { text: "B1–B2 · 互动视频实验", link: "/labs/route-bc" },
          { text: "PA1-B · 互动视频", link: "/assignments/pa1-b" },
          {
            text: "第 3 章 · 决策与规划",
            link: "/chapters/03-decision-and-planning/",
            items: [
              {
                text: "3.1 · Latent World Model",
                link: "/chapters/03-decision-and-planning/03-01-latent-world-model",
              },
              {
                text: "3.2 · RSSM 训练",
                link: "/chapters/03-decision-and-planning/03-02-rssm-training",
              },
              {
                text: "3.3 · PlaNet 与 CEM",
                link: "/chapters/03-decision-and-planning/03-03-planet-and-cem",
              },
              {
                text: "3.4 · Dreamer 想象学习",
                link: "/chapters/03-decision-and-planning/03-04-dreamer-imagination",
              },
              {
                text: "3.5 · MuZero",
                link: "/chapters/03-decision-and-planning/03-05-muzero",
              },
            ],
          },
          { text: "A1–A2 · 决策路线实验", link: "/labs/route-a" },
          { text: "PA1-A · Dreamer-lite", link: "/assignments/pa1-a" },
          {
            text: "第 5 章 · JEPA 抽象预测",
            link: "/chapters/05-jepa/",
            items: [
              {
                text: "5.1 · 特征预测",
                link: "/chapters/05-jepa/05-01-feature-prediction",
              },
              {
                text: "5.2 · Mask、EMA 与坍缩",
                link: "/chapters/05-jepa/05-02-mask-ema-collapse",
              },
              {
                text: "5.3 · Video-JEPA",
                link: "/chapters/05-jepa/05-03-video-jepa",
              },
              {
                text: "5.4 · Action-JEPA",
                link: "/chapters/05-jepa/05-04-action-jepa",
              },
            ],
          },
          { text: "C1–C2 · JEPA 路线实验", link: "/labs/route-bc" },
          { text: "PA1-C · Tiny JEPA", link: "/assignments/pa1-c" },
          {
            text: "第 6 章 · VLA 与机器人",
            link: "/chapters/06-robot-vla/",
            items: [
              {
                text: "6.1 · 数据与行为克隆",
                link: "/chapters/06-robot-vla/06-01-robot-data-and-bc",
              },
              {
                text: "6.2 · Vision-Language-Action",
                link: "/chapters/06-robot-vla/06-02-vision-language-action",
              },
              {
                text: "6.3 · Action Chunk",
                link: "/chapters/06-robot-vla/06-03-action-chunk",
              },
              {
                text: "6.4 · World Model Checker",
                link: "/chapters/06-robot-vla/06-04-world-model-checker",
              },
            ],
          },
          { text: "D1–D2 · 机器人路线实验", link: "/labs/route-de" },
          { text: "PA1-D · Tiny VLA", link: "/assignments/pa1-d" },
        ],
      },
      {
        text: "第四部分 · 设计下一台模型",
        collapsed: false,
        items: [
          {
            text: "第 8 章 · 评价与研究",
            link: "/chapters/08-evaluate-and-invent/",
            items: [
              {
                text: "8.1 · 基线与 Horizon",
                link: "/chapters/08-evaluate-and-invent/08-01-baselines-and-horizons",
              },
              {
                text: "8.2 · 反事实与 OOD",
                link: "/chapters/08-evaluate-and-invent/08-02-counterfactual-and-ood",
              },
              {
                text: "8.3 · 24GB 运行证据",
                link: "/chapters/08-evaluate-and-invent/08-03-hardware-evidence",
              },
              {
                text: "8.4 · 设计下一台模型",
                link: "/chapters/08-evaluate-and-invent/08-04-next-world-model",
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
