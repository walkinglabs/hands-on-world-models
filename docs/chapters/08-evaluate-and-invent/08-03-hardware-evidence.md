# 7.3　运行证据与复现

配置文件写着"预计使用 21GB"，那只是一份预算。只有一次完整训练真正跑完，并留下显存、时间、曲线和 checkpoint，才能称为"24GB 已验证"。

所谓证据（evidence），就是拿到这份记录的人能重跑出同样的结果；预算则只是一个计划数字。

## 三种配方

每个训练实验要分清三种配方：

1. smoke：几分钟内验证数据管线、shape、前向反向和保存路径对不对；
2. 24GB target：计划在单张 24GB 显卡上完整跑完的目标配置；
3. research：用到更大资源、可选做。

smoke 跑通，绝不等于目标配置已经完成训练。这三种状态必须分别标出，不能混用同一个词。

## Run Manifest

一次完整运行至少记录下面这些字段。少了任何一项，复现都会卡住：

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

显存有两个常用口径。`peak allocated` 是框架实际申请到的量，`peak reserved` 是进程向设备保留的量，后者更接近训练真正吃掉的显存。课程验收上限设为 $22\,\text{GB}$（即 $\approx 22\times1024\,\text{MiB}$），给系统和波动留出余量。只报 allocated 而隐瞒 reserved，是把波动藏起来的常见做法。

## 失败运行也要留

OOM、NaN、训练发散、数据中断，都要记进 manifest。它们能帮我们区分三种不同的病：配方过大、数值不稳、还是实现写错。

删掉失败日志、只留偶然成功的一次，会让后来的学习者无法看到真实过程，也就无法判断"那次成功"是不是运气。

## 当前课程状态怎么标

数据状态用独立轴区分 schema、loader、artifact；实现状态区分 spec、code、smoke 与 reference-trained；硬件状态区分 budget-only、smoke-measured 与 24gb-measured。

没有完整证据时，一律写"目标配方"或"尚未验证"，不要用"能跑"这种模糊词。

## 小结

- [ ] 配方预算不是硬件实测，三者（smoke / target / research）必须分开标。
- [ ] 24GB 证据包含环境、显存（allocated 与 reserved）、时间、曲线和 checkpoint。
- [ ] 失败运行同样要留，它们区分配方过大、数值不稳和实现错误。

> 👉 动手实验：[动手：世界模型评测（基线、多步与反事实）](/labs/z0)
