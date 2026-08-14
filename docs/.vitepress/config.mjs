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
      { text: "开始学习", link: "/guide/start" },
      { text: "世界模型导论", link: "/chapters/00-why-world-models/00-01-current-observation" },
      { text: "基础组件与模型训练", link: "/chapters/01-foundations/01-01-tensors-and-trajectories" },
      { text: "方向选修", link: "/chapters/03-decision-and-planning/03-01-latent-world-model" },
      { text: "模型评价与研究", link: "/chapters/08-evaluate-and-invent/08-01-baselines-and-horizons" },
      {
        text: "GitHub",
        link: "https://github.com/walkinglabs/hands-on-world-models",
      },
    ],
    sidebar: [
      {
        items: [
          { text: "导读", link: "/guide/start" },
        ],
      },
      {
        text: "导论",
        collapsed: false,
        items: [
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
            ],
          },
        ],
      },
      {
        text: "基础组件与训练",
        collapsed: false,
        items: [
          {
            text: "1. 常用组件",
            collapsed: false,
            items: [
              {
                text: "1.1. 张量、时间与轨迹",
                link: "/chapters/01-foundations/01-01-tensors-and-trajectories",
              },
              {
                text: "1.2. 图像编码器：CNN 与 ViT",
                link: "/chapters/01-foundations/01-02-cnn-and-vit",
              },
              {
                text: "1.3. 记忆与动态：RNN、Transformer 与 RSSM",
                link: "/chapters/01-foundations/01-03-memory-and-dynamics",
              },
              {
                text: "1.4. 压缩与生成：VAE、VQ-VAE 与扩散",
                link: "/chapters/01-foundations/01-04-compression-and-generation",
              },
              {
                text: "1.5. 空间表示：BEV 与占用网格",
                link: "/chapters/01-foundations/01-05-space-representations",
              },
              {
                text: "1.6. 决策接口：价值、策略与规划器",
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
            text: "2. 经验回放与第一个世界模型",
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
              { text: "2.3. 从经验学习第一个模型", link: "/chapters/02-data-and-first-model/02-03-first-learned-world" },
            ],
          },
        ],
      },
      {
        text: "五个方向",
        collapsed: false,
        items: [
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
              { text: "7.6. 机器人与空间实验", link: "/labs/route-de" },
            ],
          },
          {
            text: "4. 可交互视频世界",
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
              { text: "4.5. 互动视频与 JEPA 实验", link: "/labs/route-bc" },
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
                text: "3.3. PlaNet 与交叉熵方法（CEM）",
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
                text: "5.4. 动作条件 JEPA（Action-JEPA）",
                link: "/chapters/05-jepa/05-04-action-jepa",
              },
              { text: "5.5. 互动视频与 JEPA 实验", link: "/labs/route-bc" },
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
                text: "6.2. 视觉-语言-动作模型（VLA）",
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
              { text: "6.5. 机器人与空间实验", link: "/labs/route-de" },
            ],
          },
        ],
      },
      {
        text: "评价与研究",
        collapsed: false,
        items: [
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
            ],
          },
        ],
      },
    ],
    socialLinks: [
      { icon: "github", link: "https://github.com/walkinglabs/hands-on-world-models" },
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
