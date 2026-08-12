# 2.2　RSSM 怎样学习记忆与不确定性

一辆小车进入遮挡区域。GRU 可以根据进入前的速度继续估计位置，但地面打滑会使真实位置出现多种可能。

RSSM 同时维护确定记忆 `h_t` 和随机状态 `z_t`。

## Prior 与 posterior

想象未来时没有真实观察，模型只能根据过去和动作预测 prior：

```text
p(z_t | h_t)
```

训练时可以看到当前观察，再得到信息更充分的 posterior：

```text
q(z_t | h_t, embedding_t)
```

KL 损失让 prior 学会接近 posterior。这样部署时即使未来观察尚未发生，模型也能从 prior 采样。

## 状态更新

一个简化的时间步包含：

```text
h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
prior_t = p(z_t | h_t)
posterior_t = q(z_t | h_t, embedding_t)
```

训练使用 posterior 状态预测观察、reward 和 continue；想象使用 prior 状态前进。

## 四类训练目标

1. observation loss：状态能否说明当前观察；
2. reward loss：状态能否说明动作结果；
3. continue loss：状态能否说明任务是否继续；
4. KL loss：只凭过去的 prior 能否接近看过观察的 posterior。

各项数值尺度不同。训练日志应分别画曲线，不能只报告相加后的总 loss。

## KL 坍缩与 Free Bits

Decoder 太强时，模型可能忽略 `z_t`，让 posterior 与 prior 变得没有信息。Free Bits 允许一小段 KL 不受惩罚，使随机状态有机会携带必要内容。

离散 RSSM 还会使用直通估计、Unimix 等技巧。它们解决具体数值问题，应通过消融检查，而不是作为固定装饰。

## 多步检查

训练时每一步都能看到真实观察，部署 rollout 却会不断使用自己的 prior。A1 因此同时画 one-step 与 open-loop horizon 曲线，并保存从哪一步开始漂移的样例。

## 小结

- [ ] posterior 在训练时读取当前观察，prior 在想象时只依赖过去和动作。
- [ ] RSSM 同时学习观察、reward、continue 和 KL。
- [ ] 多步 rollout 比 teacher-forced 一步结果更接近部署条件。
