# 7.3　怎样记录 24GB 训练证据

配置文件写着“预计使用 21GB”，只是一份预算。只有完整训练真正运行结束，并留下显存、时间、曲线与 checkpoint，才能称为 24GB 已验证。

## 三种配方

每个训练实验提供：

1. smoke：数分钟内检查数据、shape、前反向与保存路径；
2. 24GB target：计划在单张 24GB 显卡完成的小模型；
3. research：用于更大资源的选做配置。

Smoke 通过不表示目标配置已经完成训练。

## Run Manifest

一次完整运行至少记录：

```text
git commit
GPU / CUDA / driver
Python 与依赖版本
数据 artifact 与 checksum
seed
完整超参数
peak allocated / peak reserved
wall time
主指标与曲线
checkpoint hash
```

reserved memory 更接近训练过程中实际向设备保留的显存，课程验收上限设为 22GB，为系统和波动留出余量。

## 失败运行也要保留

OOM、NaN、训练发散和数据中断都记录到 manifest。它们能帮助区分“配方过大”“数值不稳”和“实现错误”。

删除失败日志，只留下偶然成功的一次，会使后续学习者无法复现真实过程。

## 当前课程状态

数据状态使用独立轴标记 schema、loader、artifact；实现状态区分 spec、code、smoke 与 reference-trained；硬件状态区分 budget-only、smoke-measured 与 24gb-measured。

没有完整证据时一律写“目标配方”或“尚未验证”。

## 小结

- [ ] 配方预算不是硬件实测。
- [ ] 24GB 证据包含环境、显存、时间、曲线和 checkpoint。
- [ ] Smoke、目标训练与研究配置必须明确区分。
