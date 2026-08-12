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
    logo: "/logo.svg",
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
      { text: "第 0 章", link: "/chapters/00-why-world-models" },
      {
        text: "GitHub",
        link: "https://github.com/walkinglabs/hands-on-world-models",
      },
    ],
    sidebar: [
      {
        text: "课程",
        items: [
          { text: "首页", link: "/" },
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
            link: "/chapters/00-why-world-models",
          },
          { text: "F0 · 九格世界（Notebook）", link: "/labs/f0" },
        ],
      },
      {
        text: "第二部分 · 共同基础",
        collapsed: false,
        items: [
          {
            text: "第 1 章 · 表示、记忆与推演",
            link: "/chapters/01-components",
          },
          {
            text: "第 2 章 · 从经历学出小世界",
            link: "/chapters/02-data-and-first-model",
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
            text: "第 3 章 · 决策与规划",
            link: "/chapters/03-decision-and-planning",
          },
          { text: "A1–A2 · 路线实验", link: "/labs/route-a" },
          { text: "PA1-A · Dreamer-lite", link: "/assignments/pa1-a" },
          {
            text: "第 4 章 · 可交互视频",
            link: "/chapters/04-interactive-video",
          },
          {
            text: "第 5 章 · JEPA 抽象预测",
            link: "/chapters/05-jepa",
          },
          { text: "B1–C2 · 路线实验", link: "/labs/route-bc" },
          { text: "PA1-B · 互动视频", link: "/assignments/pa1-b" },
          { text: "PA1-C · Tiny JEPA", link: "/assignments/pa1-c" },
          {
            text: "第 6 章 · VLA 与机器人",
            link: "/chapters/06-robot-vla",
          },
          {
            text: "第 7 章 · 空间世界",
            link: "/chapters/07-spatial-worlds",
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
            text: "第 8 章 · 评价与研究",
            link: "/chapters/08-evaluate-and-invent",
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
