<div align="center">
  <img src="docs/public/readme/logo.png" alt="动手学世界模型 · Hands-on World Models" width="760" />

  <p>
    <strong>机器看见的只是一帧画面，世界模型学会的是整个世界如何运转。</strong><br />
    从连续观察中推测看不见的状态，学习时间、行动会让世界怎样变化——然后，在想象中规划未来。
  </p>

  <p>
    <a href="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License: CC BY-NC-SA 4.0" /></a>
    <a href="https://walkinglabs.github.io/hands-on-world-models/"><img src="https://img.shields.io/badge/在线文档-Hands--on%20World%20Models-blue" alt="在线文档" /></a>
    <a href="https://github.com/walkinglabs/hands-on-world-models/issues"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome" /></a>
  </p>

  <p>
    <a href="https://walkinglabs.github.io/hands-on-world-models/guide/world-model-intro.html"><strong>📖 开始阅读</strong></a> ·
    <a href="#全书结构">🗂️ 查看目录</a> ·
    <a href="#实验代码">🧪 运行实验</a> ·
    <a href="#快速开始">🚀 快速开始</a> ·
    <a href="#读者交流群微信">💬 读者交流群</a>
  </p>
</div>

> [!CAUTION]
> **课程正在集中重写，内容仍在快速演进。** 章节与实验可能发生较大变化，建议在 **2026 年 8 月 24 日之后**再开始系统学习。在此之前，欢迎浏览结构、提出建议。

## ✨ 本书特色

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-reinvent.webp" alt="在九格地图中预测动作的结果" width="100%" />
      <br />
      <strong>亲手训练自己的世界模型</strong>
      <br />
      <sub>从一个可以逐格检查的九宫格世界出发，每一步只引入一个新问题：图像、记忆、预测、规划。</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-components.webp" alt="公式与 PyTorch 代码逐项对应" width="100%" />
      <br />
      <strong>公式旁边就是代码</strong>
      <br />
      <sub>每学一个公式，就在 Notebook 里找到对应的 PyTorch 实现——运行它、修改它、观察模型如何变化。</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-routes.webp" alt="五种世界模型分别交出不同结果" width="100%" />
      <br />
      <strong>五条路线，任选一条走通</strong>
      <br />
      <sub>共同基础完成后，从空间、视频、决策、JEPA、机器人中选择一条路线，完整训练并检验一个小模型。</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-notebooks.webp" alt="十九份可以直接运行的 Notebook" width="100%" />
      <br />
      <strong>19 份实验，全部可以直接运行</strong>
      <br />
      <sub>从基础实验一路排到路线大作业，每份 Notebook 只解决一个问题，开箱即跑。</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-evaluation.webp" alt="课程中的世界模型评价实验" width="100%" />
      <br />
      <strong>系统学习模型评价</strong>
      <br />
      <sub>从单步误差到多步预测、动作响应与陌生场景测试，讲清楚每一种检查能回答什么问题。</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-research.webp" alt="从失败分析走向模型改进" width="100%" />
      <br />
      <strong>从复现经典走向独立研究</strong>
      <br />
      <sub>复现经典方法、分析一次稳定的失败、完成对照实验，最终提出你自己的模型改进。</sub>
    </td>
  </tr>
</table>

## 目录

- [本书特色](#本书特色)
- [本书介绍](#本书介绍)
- [最新动态](#-最新动态-news)
- [演进路线图](#️-演进路线图-roadmap)
- [全书结构](#全书结构)
- [实验代码](#实验代码)
- [推荐学习路径](#推荐学习路径)
- [快速开始](#快速开始)
- [参与贡献](#参与贡献)
- [其他课程](#其他课程)
- [引用](#引用)
- [致谢](#致谢)
- [开源协议](#开源协议)

## 📖 本书介绍

世界模型首先要解决的是**观察问题**。机器看见的只是一张图片、一段声音或某个相机视角，真实世界却在观察之外继续存在：物体会被遮挡，速度藏在前后帧的变化里，换一个视角，同一张桌子也不该变成另一个世界。世界模型尝试从连续观察中构建一个内部世界——记住暂时看不见的状态，并学习这个世界如何随时间演化。

**动作**随后加入。如果数据中记录了按键、方向盘或机器人关节指令，模型就能进一步学习"做了这个动作，世界会怎样改变"。规划器可以用这些预测比较不同做法，策略也可以从比较结果中学习。当然，预测动作的后果只是世界模型的一种用途，而非它的全部定义。

**《动手学世界模型》**从一个可以直接检查的九格世界开始：先认识观察、状态与变化，再把手写的规律换成从经历中学出来的模型。随后逐步引入图像、历史、遮挡、不确定性和三维空间——CNN、ViT、GRU、Transformer、VAE、RSSM 等组件在对应问题出现时进入实现；CEM、MCTS 与 Actor-Critic 等到真正需要选择动作时再加入。

完成共同基础后，课程分为五条路线：

| 路线            | 核心问题                               |
| :-------------- | :------------------------------------- |
| 🌌 空间世界     | 如何跨视角保持三维结构？               |
| 🎮 互动视频     | 如何生成连续、真正听从动作的画面？     |
| 🧠 决策与规划   | 如何用内部模拟减少现实中的试错？       |
| 🔮 JEPA         | 哪些未来信息值得保留？                 |
| 🤖 VLA 与机器人 | 如何把观察、语言、动作和后果接成闭环？ |

五条路线形式各异，但都回答同三个问题：**机器观察到了什么，内部保留了什么，学到的变化规律经得起哪些检查。**

### 如何讲解

每章遵循"**问题 → 方法 → 实验 → 反思**"的节奏：一个具体任务先暴露困难，再引入解决困难所需的概念与公式；可运行代码、训练曲线和评测指标用来检验方法；章节最后交代方法的假设、失败情形与适用范围。

代码尽量保留算法的主数据流：轨迹如何采集、观察如何编码、历史状态如何更新、动作如何进入 dynamics、模型如何生成 rollout、planner 或 policy 如何使用预测结果。工程技巧在真正需要时引入，不单独堆成术语表。

### 适合谁读

本书面向希望**系统学习世界模型**的学生、研究者与工程师。你只需要具备基础 Python 能力、能读懂简单的 PyTorch 代码；必要的向量、概率与梯度知识会在用到时现场复习，不要求你先修完一整套数学或强化学习课程。

读完共同基础与一条路线后，你将能够：

- 说清楚一个世界模型的输入、输出和使用方式；
- 解释 CNN、ViT、GRU、RSSM、VAE、VQ-VAE、Diffusion、MPC、CEM、MCTS 在系统中的位置；
- 从零实现一个小型 dynamics model，并检查 shape、梯度、一步预测与多步 rollout；
- 区分 Dreamer、MuZero、互动视频、JEPA、VLA 与空间世界模型的训练目标；
- 为连续轨迹设计数据切分、训练目标、反事实测试与下游评价；
- 根据失败样例提出改动，并用受控实验比较结果。

### 当前状态

本仓库是一个持续建设的中文开源课件，追求三件事：**概念正确、代码可运行、实验状态透明**。

- 课程正文：[`docs/chapters/`](docs/chapters/)
- 教学 Notebook：[`notebooks/`](notebooks/)
- 大作业：[`docs/assignments/`](docs/assignments/)
- 运行证据规范：[`docs/run-evidence.md`](docs/run-evidence.md)
- 本地验证：`npm run verify`
- 开源协议：[CC BY-NC-SA 4.0](LICENSE)

所有 Notebook 已纳入自动 smoke 测试；共同基础到路线 A 已打通一条可解释的 PixelWorld 闭环——从图片测量状态、学习动态、在模型中规划，再与真实环境的随机动作对照。尚未提交神经 PA 的 24GB 真机训练记录，外部数据的 loader、切分与校验值正在逐项补齐（详见[演进路线图](#️-演进路线图-roadmap)）。

> [!NOTE]
> **需要帮助？** 课程正在征集单张 24GB 显卡上的完整训练记录、失败样例与外部小数据 loader。提交时请附上环境、随机种子、曲线与 checkpoint 信息，规范见 [`docs/run-evidence.md`](docs/run-evidence.md)。

## 🔥 最新动态 (News)

- **[2026-08-13]** 🎉 发布课程初版：九个大章、40 篇中文小章、五条路线的教学 Notebook、PA0–PA2 任务书与项目内数据生成器。

> [!NOTE]
> 本课程有 AI 协助整理，目前尚未完成全面人工审稿，可能存在事实错误、解释不清或代码边界未覆盖的情况。欢迎通过 Issue 与 Pull Request 指正——每一条都会被认真处理。

## 🗺️ 演进路线图 (Roadmap)

课程处于活跃开发中，当前计划如下：

- [x] 建立九个大章、40 篇小章与五条选修路线
- [x] 发布 0.6–2.4、路线 Notebook、PA0–PA2 与统一评价
- [x] 将 Notebook 代码格纳入自动 smoke 测试
- [x] 建立数据状态与运行证据格式
- [ ] 完成首批 PA1 的单张 24GB 完整训练与 checkpoint
- [ ] 完成外部 T2 数据的固定 loader、切分与 SHA256 校验
- [ ] 补充各路线的失败样例与同预算对照实验
- [ ] 发布第一版稳定课程网站与 PDF

## 🗂️ 全书结构

全书共四部分、九个大章。每个大章有一页导读，下面再分成若干可独立阅读的小章：第一部分用小环境建立世界模型的基本问题；第二部分接上常用组件、数据与第一台可学习模型；第三部分提供五条可独立选择的路线；第四部分讨论评价与研究设计。

### 第一部分：世界模型的基本问题

课程先用可直接检查的表格环境，讲清楚状态、动作、转移、rollout、planner 与 policy 的关系。

| 章  | 主题                                                             | 本章主线                                             |
| :-: | :--------------------------------------------------------------- | :--------------------------------------------------- |
|  0  | [机器看见的为什么不等于世界](docs/chapters/00-why-world-models/) | 从有限观察出发，逐步得到内部状态、变化、动作与规划。 |

### 第二部分：共同基础与第一台模型

表格环境处理不了图像、历史与不确定性。这一部分介绍后续路线共用的组件，并从连续经历中学出第一个小型 dynamics model。

| 章  | 主题                                                                       | 本章主线                                                           |
| :-: | :------------------------------------------------------------------------- | :----------------------------------------------------------------- |
|  1  | [世界模型的常用组件](docs/chapters/01-foundations/)                        | 认识视觉、时序、压缩、空间与决策组件，知道每一类组件解决什么问题。 |
|  2  | [把经历变成数据，并学出第一台模型](docs/chapters/02-data-and-first-model/) | 接起 episode、transition、Replay Buffer、数据切分与表格 dynamics。 |

### 第三部分：五条世界模型路线

完成第 0–2 章后，从下面五条路线中任选其一。侧栏按 3–7 排列，彼此没有先修关系。

| 章  | 路线                                                                      | 主要内容                                                               |
| :-: | :------------------------------------------------------------------------ | :--------------------------------------------------------------------- |
|  3  | [决策与规划：怎样少在现实中试错](docs/chapters/03-decision-and-planning/) | RSSM、想象 rollout、PlaNet、Dreamer-lite，以及作为对照的 Mini-MuZero。 |
|  4  | [互动视频：怎样让画面真正听从动作](docs/chapters/04-interactive-video/)   | VQ-VAE、动作条件 Transformer、多步视频预测与 tiny diffusion 对照。     |
|  5  | [JEPA：怎样只预测有用的未来](docs/chapters/05-jepa/)                      | 视频 masking、EMA target encoder、feature prediction 与 Action-JEPA。  |
|  6  | [VLA 与机器人：怎样把理解变成动作](docs/chapters/06-robot-vla/)           | 行为克隆、动作预测、action chunk 与 world-model checker。              |
|  7  | [空间世界：怎样保持三维结构与运动一致](docs/chapters/07-spatial-worlds/)  | 相机几何、NeRF、3DGS、occupancy 与 future BEV。                        |

### 第四部分：评价与研究设计

最后一部分把各路线放回统一的测试框架：一步预测、多步预测、动作条件、分布外样本与下游任务，逐项比较。

| 章  | 主题                                                                        | 本章主线                                           |
| :-: | :-------------------------------------------------------------------------- | :------------------------------------------------- |
|  8  | [证明模型有用，并设计下一台世界模型](docs/chapters/08-evaluate-and-invent/) | 完成 8.6 统一评价，并根据 PA1 的失败样例设计 PA2。 |

### 附录：随学随查的工具箱

| 内容                                           | 用途                                                |
| :--------------------------------------------- | :-------------------------------------------------- |
| [运行证据规范](docs/run-evidence.md)           | 记录环境、显存、时间、随机种子、曲线与 checkpoint。 |
| [Notebook 索引与依赖说明](notebooks/README.md) | 每份实验的输入输出、依赖与运行方式。                |

## 🧪 实验代码

[`notebooks/`](notebooks/) 目录包含与章节一一对应的可运行实验。每条路线配备两到三份 Notebook，方便独立阅读、运行与修改。

| 领域         | 代码路径                                                             | 代表性实验                                            |
| :----------- | :------------------------------------------------------------------- | :---------------------------------------------------- |
| 世界模型入门 | [`notebooks/00_reinvent/`](notebooks/00_reinvent/)                   | 在九格世界中实现 transition、rollout 与 planner。     |
| 共同组件     | [`notebooks/01_foundations/`](notebooks/01_foundations/)             | 检查 CNN、GRU、压缩、空间与规划组件。                 |
| 数据与小模型 | [`notebooks/02_data/`](notebooks/02_data/)                           | 整理连续经历，并学出一台表格 dynamics。               |
| 决策与规划   | [`notebooks/03_decision/`](notebooks/03_decision/)                   | 学习 latent dynamics，并在想象轨迹中训练动作选择。    |
| 可交互视频   | [`notebooks/04_interactive_video/`](notebooks/04_interactive_video/) | 压缩 PixelWorld 画面，训练动作条件多步预测。          |
| JEPA         | [`notebooks/05_jepa/`](notebooks/05_jepa/)                           | 训练特征预测器，检查表示坍缩与动作敏感性。            |
| VLA 与机器人 | [`notebooks/06_robot/`](notebooks/06_robot/)                         | 从行为克隆起步，再用后果模型比较候选动作。            |
| 空间与驾驶   | [`notebooks/07_spatial/`](notebooks/07_spatial/)                     | 建立相机与空间表示，预测 4D 场景或 future occupancy。 |
| 统一评价     | [`notebooks/08_evaluation/`](notebooks/08_evaluation/)               | 比较多步误差、反事实动作、OOD 与规划增益。            |
| 大作业       | [`notebooks/assignments/`](notebooks/assignments/)                   | PA0、PA1 路线项目与 PA2 研究设计模板。                |

完整文件索引与依赖说明见 [`notebooks/README.md`](notebooks/README.md)。

## 🎯 推荐学习路径

**第一次系统学习**：按顺序完成第 0–2 章与 PA0。第 0 章提出问题，第 1 章认识组件，第 2 章把连续经历整理成数据、学出第一台模型。

**然后选择一条路线**：决策与规划、互动视频、JEPA、VLA 与机器人、空间世界——选哪一条取决于你想让模型输出什么，而不取决于侧栏里的先后位置。完成路线 Notebook 与 PA1 后，再进入第 8 章并完成 PA2。

每章建议完成四件事：**说明本章要解决的问题；画出模型的输入与输出；运行至少一个实验；改变一个条件并解释结果。** 数学或工程细节可在需要时查阅附录，不必先顺序读完。

## 🚀 快速开始

### 在线阅读

课程正文可以直接在 GitHub 中阅读：

```text
https://github.com/walkinglabs/hands-on-world-models/tree/main/docs/chapters
```

### 本地运行文档网站

环境要求：Node.js >= 18.0.0 与 npm。

```bash
git clone https://github.com/walkinglabs/hands-on-world-models.git
cd hands-on-world-models
npm install
npm run dev
```

然后在浏览器中打开终端显示的 VitePress 地址（通常是 `http://localhost:5173`）。

修改正文、导航、主题或构建脚本后，运行 `npm run verify` 检查格式并构建静态网站。

### 运行课程代码

创建 Python 环境并安装 Notebook 与神经网络依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural,notebook]'
python -m unittest discover -s tests -v
```

从第一份 Notebook 开始：

```bash
jupyter lab notebooks/00_reinvent/F0-invent-a-world-model.ipynb
```

查看并生成项目内数据：

```bash
hwm-data list
hwm-data generate pixelworld --seed 0 --num-samples 12
```

各路线的额外依赖与数据状态见 [`notebooks/README.md`](notebooks/README.md)。

## 仓库结构

```text
hands-on-world-models/
├── docs/                      # VitePress 课程正文与课程资料
│   ├── .vitepress/            # 网站配置和导航
│   ├── chapters/              # 9 个大章目录与正式小章
│   ├── assignments/           # PA0–PA2 任务书
│   └── public/                # 网站与 README 静态资源
├── notebooks/                 # 共同基础、五条路线、评价与 PA 模板
├── src/hwm/                   # 环境、数据生成器和教学模型组件
├── data/                      # 数据 registry 与本地生成数据
├── tests/                     # 代码与 Notebook smoke 测试
├── package.json               # 文档网站命令与依赖
├── pyproject.toml             # Python 包、CLI 与可选依赖
├── CONTRIBUTING.md            # 贡献指南
└── README.md                  # 项目总览
```

## 开发命令

```bash
npm run dev           # 启动本地文档服务器
npm run build         # 构建静态网站
npm run preview       # 预览构建结果
npm run format        # 使用 Prettier 格式化仓库
npm run format:check  # 检查格式
npm run verify        # 检查格式并构建网站
```

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v  # 运行 Python 与 Notebook 测试
hwm-data list                                            # 查看数据状态
hwm-data generate pixelworld --seed 0                    # 生成项目内小数据
```

## 🤝 参与贡献

让课程**更清楚、更准确、更容易复现、更容易使用**的改动，都是好贡献：

- 修复概念错误、公式、图表、失效链接或错别字；
- 改进已有章节的解释与例子；
- 添加用于说明现有方法的小型可复现实验；
- 补充数据 loader、固定切分、checksum 与许可信息；
- 提交单张 24GB 显卡上的完整运行日志、曲线与 checkpoint；
- 改进测试、构建脚本、导航与可访问性；
- 补充论文、官方文档与开源实现的原始出处。

请保持 Pull Request 范围明确：一个 PR 通常只处理一个章节、一个实验、一条数据链或一个基础设施问题。

添加内容时：

1. 将课程正文放在 [`docs/`](docs/) 目录；
2. 将可运行实验放在 [`notebooks/`](notebooks/) 或 [`src/hwm/`](src/hwm/)；
3. 为实验提供数据来源、随机种子、输入输出 shape 与最小 smoke；
4. 添加可导航页面时更新 VitePress 配置；
5. 提交前运行 `npm run verify` 与 Python 测试；
6. 使用 Conventional Commits，例如 `docs: clarify rssm state` 或 `fix: align action timestamps`。

详细规范见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 📚 其他课程

WalkingLabs 还制作了以下开源课程：

- [**Hands-On Modern RL**](https://github.com/walkinglabs/hands-on-modern-rl) — 从经典序贯决策与策略优化，进入大模型后训练、Agentic RL 与多模态系统。
- [**Modern LLM Notebook**](https://github.com/walkinglabs/modern-llm-notebook) — 使用 Jupyter Notebook 从零实现 Tokenizer、Transformer、训练、推理与对齐。

## 💬 读者交流群（微信）

有任何建议或反馈，欢迎扫码加入读者交流群：

<img
  src="https://github.com/walkinglabs/.github/raw/main/profile/wechat.png"
  alt="读者交流群"
  style="width: 100%; max-width: 520px; height: auto;"
/>

## 引用

如果在教学材料、学习笔记或衍生的非商业教育作品中使用本课程，请引用本仓库：

```bibtex
@misc{hands_on_world_models,
  title        = {Hands-On World Models: From First Principles to Prediction, Planning, and Action},
  author       = {WalkingLabs},
  year         = {2026},
  howpublished = {\url{https://github.com/walkinglabs/hands-on-world-models}},
  note         = {Chinese open courseware repository}
}
```

## 致谢

本课程的组织方式参考了 [Hands-On Modern RL](https://github.com/walkinglabs/hands-on-modern-rl)、[《动手学深度学习》](https://zh.d2l.ai/)、[Stanford CS336](https://stanford-cs336.github.io/spring2025/) 与[南京大学计算机系统基础课程实验](https://nju-projectn.github.io/ics-pa-gitbook/ics2024/)。

课程内容参考了 World Models、PlaNet、Dreamer、MuZero、IRIS、DIAMOND、V-JEPA、VLA、NeRF、3D Gaussian Splatting、BEV 与 Occupancy 等研究及开源实现，具体论文与代码来源列在相应章节中。

感谢每一位论文作者、开源实现维护者、课程试读者与贡献者。

## 开源协议

本课程资料采用 [CC BY-NC-SA 4.0](LICENSE)（署名-非商业性使用-相同方式共享 4.0 国际）协议发布：你可以出于非商业目的共享与修改本材料，前提是给出适当署名，并让衍生作品继续使用相同协议。

---

<div align="center">
  <p>如果这门课程对你有帮助，欢迎点一颗 Star ⭐️——这是对我们最大的鼓励。</p>
  <sub>Maintained with ❤️ by WalkingLabs & Contributors.</sub>
</div>
