# 参与课程建设

《动手学世界模型》仍在建设中。课程更需要能让初学者看见问题的例子，而不是更长的模型清单。

## 提交内容以前

先回答三个问题：

1. 这项内容从哪个具体失败出发？
2. 学生运行以后会看见什么结果？
3. 它是否增加了一份不必要的 Notebook？

若一个组件无法对应到可见失败，先不要把它加入共同基础。若一篇论文只提供大规模结果，而课程没有小实验覆盖它的核心主张，把它放进“论文拔高”，不要伪装成已经复现。

## 教材正文

正文遵循同一条顺序：

```text
具体处境
→ 最简单的旧办法
→ 加入一个条件
→ 展示失败
→ 用普通话说出缺少的能力
→ 引入组件
→ 小实验
→ 新边界
```

请使用“我们”，一个自然段推进一件事。中文解释先于英文缩写。论文名称放在问题出现以后。

## Notebook

一份 Notebook 围绕一个完整结果，通常包含三到五轮紧密相连的“失败—修补”过程。

每个代码格应该产生表、图、断言或一个能回答当前问题的行为。不保留只打印变量类型、没有教学作用的单元格。

提交前运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 刷新 Python 语义高亮

文档站在 Shiki 的 TextMate 高亮之上合并 Pylance semantic tokens。修改 Markdown 中的
Python 代码块后，请确保本机 VS Code 已安装 Pylance，然后运行：

```bash
npm run semantic:refresh
npm run semantic:check
```

刷新命令会在隔离的 VS Code 扩展宿主中调用 Pylance，并更新
`docs/.vitepress/python-semantic-tokens.json`。构建和 CI 只读取这份缓存，不会启动 VS Code。

若加入神经网络训练，还必须提供 smoke 配置、数据来源、切分方法和资源记录。没有完整真机运行证据时，只能写“设计预算”。

## 网站

安装 Node.js 18 或更高版本，然后运行：

```bash
npm install
npm run dev
```

提交前执行：

```bash
npm run format:check
npm run build
npm audit
```

## 提交状态

请使用明确状态：

- 草案：结构存在，实验尚未配齐；
- 可教：正文、代码、数据和 smoke 已经形成一条路径；
- 可学：新学习者独立跑通过，并根据反馈修改；
- 可发布：资源证据、引用、链接和完整验收均已通过。

不要只写“已完成”。
