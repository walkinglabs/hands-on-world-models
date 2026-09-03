import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const chapters = [
  {
    directory: "01-why-world-models",
    section: "01-observation-and-state",
    title: "第 1 章 · 为什么需要世界模型",
    route: "从不完备观测，到可交互的预测与验证闭环",
    nodes: [
      ["1.1", "观察、状态与变化", "历史聚合为马尔可夫状态"],
      ["1.2", "什么是世界模型", "转移、奖励与规则形式化"],
      ["1.3", "经典世界模型", "从控制论到神经世界模型"],
      ["1.4", "交互式驾驶", "观察—预测—选择—验证"],
    ],
  },
  {
    directory: "02-foundations",
    section: "01-cnn-and-vit",
    title: "第 2 章 · 深度学习与表征基础",
    route: "从空间特征与时序记忆，到隐空间生成与标准化实现",
    nodes: [
      ["2.1", "CNN 与 ViT", "局部归纳偏置与全局注意力"],
      ["2.2", "RNN 与 Transformer", "循环记忆与因果时序建模"],
      ["2.3", "VAE 与 VQ", "连续隐变量与离散码本"],
      ["2.4", "自回归与扩散", "次词元预测与逆向去噪"],
      ["2.5", "基础组件从零实现", "纯张量前向与反向闭环"],
      ["2.6", "基础组件简洁实现", "归一化与残差骨干统一接口"],
    ],
  },
  {
    directory: "03-data-and-first-model",
    section: "01-episodes-and-transitions",
    title: "第 3 章 · 数据与强化学习基础",
    route: "从交互数据组织，到价值学习、模型预测控制与策略实现",
    nodes: [
      ["3.1", "经历与状态转移", "MDP、奖励与累积回报"],
      ["3.2", "经验回放池", "优先采样与数据切分"],
      ["3.3", "价值与策略梯度", "贝尔曼方程与对数导数"],
      ["3.4", "MPC 与 CEM", "滚动时域规划与精英重拟合"],
      ["3.5", "强化学习从零实现", "Double DQN 与软更新"],
      ["3.6", "强化学习简洁实现", "连续动作与压缩高斯策略"],
    ],
  },
  {
    directory: "04-latent-dynamics",
    section: "01-world-models",
    title: "第 4 章 · 潜在动力学模型",
    route: "从视觉—记忆—控制解耦，到隐空间规划、梦境学习与价值等价",
    nodes: [
      ["4.1", "World Models", "视觉、记忆与控制三元解耦"],
      ["4.2", "RSSM", "确定性与随机隐状态双轨"],
      ["4.3", "PlaNet", "潜空间 CEM 在线规划"],
      ["4.4", "DreamerV1", "梦境 Actor–Critic 与 Lambda 回报"],
      ["4.5", "DreamerV2 / V3", "离散隐变量与统一稳定化"],
      ["4.6", "MuZero", "价值等价表示与潜在树搜索"],
      ["4.7", "RSSM 从零实现", "时序张量流水线与解码"],
      ["4.8", "想象训练的简洁实现", "误差界、泛化与控制平滑度"],
    ],
  },
  {
    directory: "05-interactive-video",
    section: "01-video-prediction-svg",
    title: "第 5 章 · 交互式视频生成",
    route: "从随机视频预测，到时空词元、扩散建模与动作控制闭环",
    nodes: [
      ["5.1", "动作条件视频预测", "SVG 像素时序与随机未来分叉"],
      ["5.2", "Tokenizer 与 VideoPoet", "时空压缩与离散自回归"],
      ["5.3", "基于扩散的物理模拟", "Sora 时空 Patch 与扩散去噪"],
      ["5.4", "KV-Cache 与流式生成", "增量注意力与实时推演"],
      ["5.5", "交互视频模型从零实现", "FiLM 动作调制与 ConvGRU 记忆"],
      ["5.6", "受动作控制的视频小世界", "动作引导强度与范式对比"],
    ],
  },
  {
    directory: "06-jepa",
    section: "01-jepa-foundation",
    title: "第 6 章 · 联合嵌入预测架构",
    route: "从非生成式特征预测，到防坍塌、动量自举与特征空间规划",
    nodes: [
      ["6.1", "联合嵌入与特征预测", "I-JEPA 抛弃像素重构"],
      ["6.2", "掩码机制与表示坍塌", "方差—协方差撑开表示空间"],
      ["6.3", "目标网络", "EMA 动量自举与跨 Patch 预测"],
      ["6.4", "动作条件特征预测", "Action-JEPA 时空动力学"],
      ["6.5", "JEPA 从零实现", "Gather 索引与动量自举闭环"],
      ["6.6", "视频 JEPA 简洁实现", "特征空间可微 MPC 轨迹优化"],
    ],
  },
  {
    directory: "07-robot-policy",
    section: "01-multimodal-observation",
    title: "第 7 章 · 具身策略大模型",
    route: "从多模态状态与身体动力学，到生成式策略、VLA-JEPA 与真机闭环",
    nodes: [
      ["7.1", "多模态异构观测", "视觉、触觉与本体感觉融合"],
      ["7.2", "灵巧手与灵巧操作", "接触、摩擦锥与微观滑移"],
      ["7.3", "双足人形与全身控制", "质心动量与 WBC 零空间"],
      ["7.4", "模仿学习基线", "行为克隆的协变量偏移与纠偏"],
      ["7.5", "扩散策略", "多模态轨迹分布与去噪动作"],
      ["7.6", "ACT 动作分块", "高频时序对齐与动作平滑"],
      ["7.7", "视觉语言动作模型", "RT-X 从静态策略到状态预测"],
      ["7.8", "原生具身大模型", "OpenVLA 开源基座与空间对齐"],
      ["7.9", "世界动作模型", "VLA-JEPA / WAM 隐空间具身推演"],
      ["7.10", "扩散策略从零实现", "可微去噪动作生成与策略优化"],
      ["7.11", "把世界模型接上身体", "端到端真机物理控制闭环"],
    ],
  },
  {
    directory: "08-robot-sim",
    section: "01-physics-mujoco",
    title: "第 8 章 · 具身仿真与现实迁移",
    route: "从接触物理与 GPU 并行仿真，到域随机化、人在回路与 Sim2Real 规划",
    nodes: [
      ["8.1", "物理引擎与接触力模拟", "MuJoCo 多刚体动力学求解"],
      ["8.2", "并行仿真与合成数据", "Isaac GPU 巨量物理步进"],
      ["8.3", "现实差距与域随机化", "自适应参数分布边界"],
      ["8.4", "特权蒸馏与现实迁移", "历史观测推断物理隐变量"],
      ["8.5", "想象学习与自我改进", "RISE 梦境中的策略演化"],
      ["8.6", "遥操作与人在回路", "长尾故障介入与自愈重加权"],
      ["8.7", "现实迁移的简洁实现", "系统辨识、延时与滤波"],
      ["8.8", "具身规划从零实现", "GPU 并行 MPPI 轨迹优化"],
    ],
  },
  {
    directory: "09-spatial-worlds",
    section: "01-camera-geometry",
    title: "第 9 章 · 空间世界与自动驾驶",
    route: "从二维成像几何，到三维占用、连续场、四维预测与驾驶规划",
    nodes: [
      ["9.1", "相机几何与多视角投影", "内参、外参与齐次透视除法"],
      ["9.2", "BEV 与三维占用", "视锥提升与空间交叉注意力"],
      ["9.3", "神经辐射场与三维高斯", "NeRF 体积渲染与 3DGS 光栅化"],
      ["9.4", "时空四维场景预测", "动态速度场与因果演进"],
      ["9.5", "自动驾驶世界模型", "多视角预测与规划闭环"],
      ["9.6", "占用预测从零实现", "视锥提升与 3D 卷积预测"],
      ["9.7", "驾驶场景下的四维世界模型", "感知—预测—规划一体化"],
    ],
  },
  {
    directory: "10-evaluate-and-invent",
    section: "01-evaluate-principles",
    title: "第 10 章 · 评测与研究设计",
    route: "从科学指标与系统基准，到失效归因、可证伪改进与新模型设计",
    nodes: [
      ["10.1", "怎样评价世界模型", "单步误差、长程漂移与任务效用"],
      ["10.2", "世界模型的系统评测", "开环推演、闭环控制与 OOD 鲁棒性"],
      ["10.3", "失效分析与模型改进", "幻觉、复合误差与协变量偏移"],
      ["10.4", "设计新的世界模型", "可证伪假设与多模态新范式"],
    ],
  },
];

const palette = [
  ["EAF6F7", "25636A", "194E54"],
  ["FFF3E8", "D97745", "9A4B26"],
  ["EDF7EF", "4F8A5B", "315D39"],
  ["F5EEFA", "8E5AA7", "613774"],
  ["FCEEF2", "C45D78", "87384E"],
  ["EAF4FA", "4A7FA0", "2D5872"],
];

function texEscape(value) {
  return value
    .replaceAll("\\", "\\textbackslash{}")
    .replaceAll("&", "\\&")
    .replaceAll("%", "\\%")
    .replaceAll("#", "\\#")
    .replaceAll("_", "\\_")
    .replaceAll("$", "\\$")
    .replaceAll("{", "\\{")
    .replaceAll("}", "\\}");
}

function renderTikz(chapter) {
  const count = chapter.nodes.length;
  const columns = count <= 4 ? count : Math.ceil(count / 2);
  const topCount = count <= 4 ? count : columns;
  const bottomCount = count - topCount;
  const gap = columns >= 6 ? 0.22 : 0.34;
  const cardWidth = (23 - gap * (columns - 1)) / columns;
  const topY = bottomCount ? 6.35 : 4.75;
  const bottomY = 2.55;
  const centers = [];

  for (let index = 0; index < count; index += 1) {
    if (index < topCount) {
      centers.push([1 + cardWidth / 2 + index * (cardWidth + gap), topY]);
    } else {
      const bottomIndex = index - topCount;
      centers.push([
        1 + cardWidth / 2 + (columns - 1 - bottomIndex) * (cardWidth + gap),
        bottomY,
      ]);
    }
  }

  const cards = chapter.nodes
    .map(([number, title, description], index) => {
      const [x, y] = centers[index];
      const compact = columns >= 6;
      const titleSize = compact ? 7.2 : 8.3;
      const bodySize = compact ? 5.8 : 6.6;
      return `\\node[card, fill=col${index}fill, draw=col${index}stroke, text width=${(cardWidth - 0.38).toFixed(2)}cm] (s${index + 1}) at (${x.toFixed(3)},${y}) {%
  {\\fontsize{${titleSize}}{${titleSize + 1.7}}\\selectfont\\bfseries\\color{col${index}text} ${texEscape(number)}\\quad ${texEscape(title)}}\\par\\vspace{2pt}
  {\\fontsize{${bodySize}}{${bodySize + 1.7}}\\selectfont\\color{Slate} ${texEscape(description)}}
};`;
    })
    .join("\n");

  const arrows = chapter.nodes
    .slice(0, -1)
    .map((_, index) => {
      const from = centers[index];
      const to = centers[index + 1];
      if (Math.abs(from[1] - to[1]) < 0.01) {
        return to[0] > from[0]
          ? `\\draw[flow] (s${index + 1}.east) -- (s${index + 2}.west);`
          : `\\draw[flow] (s${index + 1}.west) -- (s${index + 2}.east);`;
      }
      return `\\draw[flow] (s${index + 1}.south) -- ++(0,-0.48) -| (s${index + 2}.north);`;
    })
    .join("\n");

  const definitions = chapter.nodes
    .map((_, index) => {
      const [fill, stroke, text] = palette[index % palette.length];
      return `\\definecolor{col${index}fill}{HTML}{${fill}}\n\\definecolor{col${index}stroke}{HTML}{${stroke}}\n\\definecolor{col${index}text}{HTML}{${text}}`;
    })
    .join("\n");

  return `\\documentclass[tikz,border=0pt]{standalone}
\\usepackage[UTF8,fontset=none]{ctex}
\\setCJKmainfont[BoldFont={Hiragino Sans GB W6}]{Hiragino Sans GB W3}
\\setCJKsansfont[BoldFont={Hiragino Sans GB W6}]{Hiragino Sans GB W3}
\\usetikzlibrary{arrows.meta,calc}
\\definecolor{Canvas}{HTML}{FBFCFD}
\\definecolor{Ink}{HTML}{172B2F}
\\definecolor{Slate}{HTML}{52666A}
\\definecolor{Brand}{HTML}{25636A}
${definitions}
\\begin{document}
\\begin{tikzpicture}[
  x=1cm,
  y=1cm,
  every node/.style={font=\\sffamily},
  card/.style={rounded corners=7pt, minimum height=2.15cm, inner sep=6pt, align=center, line width=0.8pt},
  flow/.style={-{Stealth[length=5pt,width=4pt]}, line width=1.2pt, draw=Brand!75, rounded corners=5pt}
]
\\path[use as bounding box] (0,0) rectangle (25,10);
\\fill[Canvas] (0,0) rectangle (25,10);
\\fill[Brand] (0,9.78) rectangle (25,10);
\\node[anchor=west, text=Ink] at (1,9.20) {\\fontsize{18}{21}\\selectfont\\bfseries ${texEscape(chapter.title)}};
\\node[anchor=west, text=Slate] at (1,8.55) {\\fontsize{9.2}{11}\\selectfont ${texEscape(chapter.route)}};
\\draw[Brand!18, line width=0.7pt] (1,8.13) -- (24,8.13);
${cards}
${arrows}
\\node[anchor=west, rounded corners=4pt, fill=Brand!8, text=Brand, inner xsep=7pt, inner ysep=4pt] at (1,0.62) {\\fontsize{7.2}{9}\\selectfont\\bfseries 学习路径};
\\node[anchor=west, text=Slate] at (3.25,0.62) {\\fontsize{7.2}{9}\\selectfont 概念建模 → 核心方法 → 工程实现 → 系统验证};
\\node[anchor=east, text=Slate!72] at (24,0.62) {\\fontsize{6.6}{8}\\selectfont HANDS-ON WORLD MODELS};
\\end{tikzpicture}
\\end{document}
`;
}

function run(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, encoding: "utf8" });
  if (result.status !== 0) {
    process.stderr.write(result.stdout || "");
    process.stderr.write(result.stderr || "");
    throw new Error(`${command} failed with exit code ${result.status}`);
  }
}

for (const chapter of chapters) {
  const outputDirectory = path.join(
    root,
    "docs",
    "public",
    "figures",
    chapter.directory,
    "latex",
    chapter.section,
  );
  fs.mkdirSync(outputDirectory, { recursive: true });
  const basename = "chapter-overview";
  const texPath = path.join(outputDirectory, `${basename}.tex`);
  const pdfPath = path.join(outputDirectory, `${basename}.pdf`);
  const pngPath = path.join(outputDirectory, `${basename}.png`);

  fs.writeFileSync(texPath, renderTikz(chapter));
  run(
    "xelatex",
    [
      "-interaction=nonstopmode",
      "-halt-on-error",
      "-file-line-error",
      `-output-directory=${outputDirectory}`,
      texPath,
    ],
    root,
  );
  run(
    "pdftocairo",
    [
      "-png",
      "-singlefile",
      "-scale-to-x",
      "2560",
      "-scale-to-y",
      "1024",
      pdfPath,
      path.join(outputDirectory, basename),
    ],
    root,
  );
  run("sips", ["--resampleHeightWidth", "1024", "2560", pngPath], root);
  for (const extension of ["aux", "log"]) {
    fs.rmSync(path.join(outputDirectory, `${basename}.${extension}`), {
      force: true,
    });
  }
  console.log(path.relative(root, pngPath));
}
