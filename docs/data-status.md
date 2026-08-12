# 数据与实验状态

这张表只记录已经有证据的状态。知道数据集名字，不等于 loader 已经完成；写出 24GB 配方，也不等于真机训练已经通过。

## 状态含义

| 状态        | 含义                                           |
| ----------- | ---------------------------------------------- |
| 设计中      | 已确定教学目的，代码或数据尚未提供             |
| 可生成      | 仓库内有生成器，可从 seed 重新产生             |
| 可运行      | 代码和 smoke 测试可以完整执行                  |
| 已训练      | 提交了曲线、指标和 checkpoint                  |
| 24GB 已验证 | 完整记录峰值显存、时间、环境和 checkpoint 哈希 |

## 当前清单

| 内容                   | 数据       | 实现     | 硬件               | 证据                                                 |
| ---------------------- | ---------- | -------- | ------------------ | ---------------------------------------------------- |
| F0 九格世界            | 可生成     | 可运行   | CPU 已测试         | `tests/test_gridworld.py`、`tests/test_notebooks.py` |
| F1 PixelWorld 组件     | 可生成     | 可运行   | CPU 已测试         | `src/hwm/data.py`、Notebook smoke                    |
| F2 相机、空间与规划    | 内嵌小数据 | 可运行   | CPU 已测试         | `src/hwm/foundations.py`、Notebook smoke             |
| F3 LineWorld 动态      | 可生成     | 可运行   | CPU 已测试         | `src/hwm/gridworld.py`、Notebook smoke               |
| PA0                    | 任务书发布 | 学生实现 | CPU 目标，未收作业 | `docs/assignments/pa0.md`                            |
| 路线 A PyTorch smoke   | 可生成     | 可运行   | CPU 已测试         | `tests/test_neural.py`、A1/A2 Notebook smoke         |
| PA1-A Dreamer-lite     | 任务书发布 | 学生实现 | 24GB 目标，未测试  | `docs/assignments/pa1-a.md`                          |
| 路线 B 互动视频        | 设计中     | 设计中   | 24GB 目标，未测试  | 课程总纲                                             |
| 路线 C Tiny Video-JEPA | 设计中     | 设计中   | 24GB 目标，未测试  | 课程总纲                                             |
| 路线 D Tiny VLA        | 设计中     | 设计中   | 24GB 目标，未测试  | 课程总纲                                             |
| 路线 E 空间世界        | 设计中     | 设计中   | 24GB 目标，未测试  | 课程总纲                                             |

F0–F3 已形成“正文—代码—项目内数据—Notebook—测试”共同基础路径。它们只证明规则与 NumPy 小世界可以运行，不证明后续神经网络路线已经完成。
