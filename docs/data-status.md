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
| 路线 B 互动视频 smoke  | 可生成     | 可运行   | CPU 已测试         | `tests/test_routes_bc.py`、B1/B2 Notebook smoke      |
| PA1-B 互动视频         | 任务书发布 | 学生实现 | 24GB 目标，未测试  | `docs/assignments/pa1-b.md`                          |
| 路线 C Tiny Video-JEPA | 可生成     | 可运行   | CPU 已测试         | `tests/test_routes_bc.py`、C1/C2 Notebook smoke      |
| PA1-C Tiny Video-JEPA  | 任务书发布 | 学生实现 | 24GB 目标，未测试  | `docs/assignments/pa1-c.md`                          |
| 路线 D Tiny VLA smoke  | 可生成     | 可运行   | CPU 已测试         | `tests/test_routes_de.py`、D1/D2 Notebook smoke      |
| PA1-D Tiny VLA         | 任务书发布 | 学生实现 | 24GB 目标，未测试  | `docs/assignments/pa1-d.md`                          |
| 路线 E 空间世界 smoke  | 可生成     | 可运行   | CPU 已测试         | `tests/test_routes_de.py`、E1/E2 Notebook smoke      |
| PA1-E 空间世界         | 任务书发布 | 学生实现 | 24GB 目标，未测试  | `docs/assignments/pa1-e.md`                          |
| Z0 统一评价            | 内嵌小数据 | 可运行   | CPU 已测试         | `tests/test_evaluation.py`、Z0 Notebook smoke        |
| PA2 下一台模型         | 任务书发布 | 学生实现 | 随路线而定         | `docs/assignments/pa2.md`                            |

F0–F3 与路线 A–E 的 smoke 已形成“正文—代码—项目内数据—Notebook—测试”路径。Smoke 只证明最小接口和训练路径可以运行，不证明 PA 完成，也不证明 24GB 完整训练已经通过。
