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
        text: "1. 引言",
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
            text: "1.3. 多步推演与规划",
            link: "/chapters/01-why-world-models/03-rollout-planning-policy",
          },
          {
            text: "1.4. 从经历学习动力学",
            link: "/chapters/01-why-world-models/04-learned-dynamics",
          },
          {
            text: "1.5. 观察与动作的联合分布",
            link: "/chapters/01-why-world-models/05-joint-distribution",
          },
          {
            text: "1.6. 世界模型的定义",
            link: "/chapters/01-why-world-models/06-what-is-a-world-model",
          },
          {
            text: "1.7. 经典世界模型",
            link: "/chapters/01-why-world-models/07-classic-world-models",
          },
          {
            text: "1.8. 动手：九格世界的从零实现",
            link: "/chapters/01-why-world-models/08-invent-a-world-model",
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
            text: "2.3. 记忆与动力学",
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
            text: "2.7. 训练稳定性",
            link: "/chapters/02-foundations/07-training-stability",
          },
          {
            text: "2.8. 动手：组件接口的简洁实现",
            link: "/chapters/02-foundations/08-basic-experiments",
          },
        ],
      },
      {
        text: "3. 数据与第一个世界模型",
        collapsed: false,
        items: [
          {
            text: "3.1. 经历与转移",
            link: "/chapters/03-data-and-first-model/01-episodes-and-transitions",
          },
          {
            text: "3.2. 经验回放与数据切分",
            link: "/chapters/03-data-and-first-model/02-replay-buffer-and-splits",
          },
          {
            text: "3.3. 从经历学出转移模型",
            link: "/chapters/03-data-and-first-model/03-first-learned-world",
          },
          {
            text: "3.4. 世界模型的基本检查",
            link: "/chapters/03-data-and-first-model/04-basic-checks",
          },
          {
            text: "3.5. 动手：表格世界模型的从零开始实现",
            link: "/chapters/03-data-and-first-model/05-learn-a-table-world",
          },
          {
            text: "3.6. 动手：重新发明一台可学习世界模型",
            link: "/chapters/03-data-and-first-model/06-learnable-world",
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
            text: "4.2. 循环状态空间模型（RSSM）",
            link: "/chapters/04-decision-and-planning/02-rssm-training",
          },
          {
            text: "4.3. 交叉熵方法与模型预测控制",
            link: "/chapters/04-decision-and-planning/03-planet-and-cem",
          },
          {
            text: "4.4. Dreamer 与想象训练",
            link: "/chapters/04-decision-and-planning/04-dreamer-imagination",
          },
          {
            text: "4.5. 蒙特卡洛树搜索（MuZero）",
            link: "/chapters/04-decision-and-planning/05-muzero",
          },
          {
            text: "4.6. 动手：World Models 的复现",
            link: "/chapters/04-decision-and-planning/06-reproduce-world-models",
          },
          {
            text: "4.7. 动手：Dreamer 的简化实现",
            link: "/chapters/04-decision-and-planning/07-decision-and-planning",
          },
          {
            text: "4.8. 动手：Dreamer 的完整闭环",
            link: "/chapters/04-decision-and-planning/08-dreamer-loop",
          },
        ],
      },
      {
        text: "5. 交互式视频",
        collapsed: false,
        items: [
          {
            text: "5.1. 视频世界模型",
            link: "/chapters/05-interactive-video/01-video-data",
          },
          {
            text: "5.2. 词元化与预测目标",
            link: "/chapters/05-interactive-video/02-vq-tokenizer",
          },
          {
            text: "5.3. 自回归与扩散",
            link: "/chapters/05-interactive-video/03-action-transformer",
          },
          {
            text: "5.4. 动作条件与可控性",
            link: "/chapters/05-interactive-video/04-action-conditioning",
          },
          {
            text: "5.5. 记忆、漂移与实时生成",
            link: "/chapters/05-interactive-video/05-memory-drift-realtime",
          },
          {
            text: "5.6. 动手：动作条件视频模型的从零实现",
            link: "/chapters/05-interactive-video/06-interactive-video",
          },
          {
            text: "5.7. 动手：听从按键的视频小世界",
            link: "/chapters/05-interactive-video/07-controllable-video",
          },
        ],
      },
      {
        text: "6. 联合嵌入预测架构（JEPA）",
        collapsed: false,
        items: [
          {
            text: "6.1. 特征预测",
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
            text: "6.5. 动手：视频 JEPA 的从零实现",
            link: "/chapters/06-jepa/05-jepa",
          },
          {
            text: "6.6. 动手：审问一个视频 JEPA",
            link: "/chapters/06-jepa/06-video-jepa",
          },
        ],
      },
      {
        text: "7. 具身智能与机器人",
        collapsed: false,
        items: [
          {
            text: "7.1. 机器人学习接口",
            link: "/chapters/07-robot-vla/01-robot-interfaces",
          },
          {
            text: "7.2. 模仿学习与生成策略",
            link: "/chapters/07-robot-vla/02-imitation-and-policies",
          },
          {
            text: "7.3. 视觉语言动作模型（VLA）",
            link: "/chapters/07-robot-vla/03-vision-language-action",
          },
          {
            text: "7.4. 机器人世界模型",
            link: "/chapters/07-robot-vla/04-robot-world-models",
          },
          {
            text: "7.5. 操作、接触与触觉",
            link: "/chapters/07-robot-vla/05-manipulation-and-touch",
          },
          {
            text: "7.6. 腿式与全身控制",
            link: "/chapters/07-robot-vla/06-legged-and-whole-body",
          },
          {
            text: "7.7. 模拟器与 Sim-to-Real",
            link: "/chapters/07-robot-vla/07-simulators-and-sim2real",
          },
          {
            text: "7.8. 动手：从零实现 VLA 与世界模型检查器",
            link: "/chapters/07-robot-vla/08-robot-vla",
          },
          {
            text: "7.9. 动手：机械臂的仿真与真机迁移",
            link: "/chapters/07-robot-vla/09-arm-sim2real",
          },
          {
            text: "7.10. 动手：灵巧手的视触觉控制",
            link: "/chapters/07-robot-vla/10-dexhand-visuotactile",
          },
          {
            text: "7.11. 动手：全身策略的仿真与真机迁移",
            link: "/chapters/07-robot-vla/11-whole-body-sim2real",
          },
          {
            text: "7.12. 动手：VLA 与动作后果检查",
            link: "/chapters/07-robot-vla/12-vla-checker",
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
            text: "8.2. 鸟瞰图与占用网格",
            link: "/chapters/08-spatial-worlds/02-bev-and-occupancy",
          },
          {
            text: "8.3. 神经辐射场与三维高斯",
            link: "/chapters/08-spatial-worlds/03-nerf-3dgs-mesh",
          },
          {
            text: "8.4. 四维场景",
            link: "/chapters/08-spatial-worlds/04-four-dimensional-worlds",
          },
          {
            text: "8.5. 驾驶世界模型",
            link: "/chapters/08-spatial-worlds/05-driving-world-models",
          },
          {
            text: "8.6. 物理先验",
            link: "/chapters/08-spatial-worlds/06-physics-priors",
          },
          {
            text: "8.7. 动手：三维重建与占用预测",
            link: "/chapters/08-spatial-worlds/07-spatial-world",
          },
          {
            text: "8.8. 动手：空间世界二选一",
            link: "/chapters/08-spatial-worlds/08-spatial-world",
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
            text: "9.4. 动手：设计下一台世界模型",
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
