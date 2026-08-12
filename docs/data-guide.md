# 数据使用指南

## 先看状态，不要先下载

```bash
python -m pip install -e '.[neural]'
hwm-data list
```

`data/registry.json` 为每份数据记录路线、层级、状态、schema、来源与边界。

三种状态不能混用：

- `artifact-ready`：仓库生成器可按 seed 生成 artifact 与 SHA256；
- `source-known`：知道来源，但固定子集、loader 或校验尚未发布；
- `registration-required`：还要在官网注册并同意数据条款。

## 生成项目内数据

```bash
hwm-data generate lineworld --seed 0 --num-samples 30
hwm-data generate pixelworld --seed 0 --num-samples 12
hwm-data generate tabletop --seed 0 --num-samples 256
hwm-data generate occupancy --seed 0 --num-samples 96
```

默认输出到 `artifacts/data/`，同时生成 `.npz` 与 `.json`。JSON 中包含 seed、样本数和 artifact SHA256。

Tabletop 与 Occupancy 生成器依赖 PyTorch；PixelWorld 只依赖 NumPy。

## 怎样切分

- 动态视频与 RL：按 episode/seed 切，不随机拆帧；
- 机器人：按 scene/task/episode 切，保留 instruction 与时间字段；
- 3D：按 scene 或 camera trajectory 切，避免近邻视角泄漏；
- 驾驶：按 scene/log 切，保留相机标定、ego motion 与时间戳；
- 数字任务若后续扩展：按数据库 seed 和 task template 切。

同一个 PA 的 baseline 与新方法必须使用相同 split。

## 外部数据为什么暂不自动下载

DMC、CarRacing、UCF101、PushT、Lego 和 nuScenes 的安装、许可与版本不同。有些需要注册，有些由环境运行时生成。Registry 先诚实记录来源和未完成项；只有固定 loader、子集与校验真正加入仓库后，状态才会升级。

学生自己准备的外部数据必须在数据卡中写明：来源 URL、许可、版本、生成/下载日期、split、checksum、时间与动作字段。
