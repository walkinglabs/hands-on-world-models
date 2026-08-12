# 4.2　Mask、EMA 与表示坍缩

如果 Context Encoder 和 Target Encoder 都输出全零向量，预测误差可以很小，但表示没有保留任何内容。这种情况称为表示坍缩。

JEPA 需要在没有负样本和像素重建的情况下，避免这个平凡答案。

## Mask 在问什么

图像被切成 patch。我们遮住一部分区域，只把可见 patch 交给 Context Encoder，让 Predictor 猜被遮区域的 target feature。

若目标区域紧邻可见区域，模型可以依靠局部纹理；若目标更大、更远，就需要对象与场景结构。mask 的形状实际上定义了学习问题。

## 为什么使用两个 Encoder

在线 Context Encoder 通过梯度更新。Target Encoder 不接收预测损失的梯度，而由在线参数的指数移动平均（EMA）更新：

```text
θ_target ← m θ_target + (1-m) θ_online
```

目标变化得较慢，Predictor 不会追逐一个同时剧烈移动的标签。

## stop-gradient 做了什么

target feature 从计算图中截断梯度。否则两个 Encoder 可能一起移动到容易匹配、但没有信息的表示。

stop-gradient 与 EMA 只规定优化路径，并不保证特征适合任何任务。数据、mask 和架构仍然决定表示内容。

## 怎样检查坍缩

至少查看：

1. 各特征维度的标准差；
2. 不同样本之间的余弦相似度；
3. 特征协方差的有效秩；
4. linear probe 是否优于常数基线。

预测 loss 下降而特征方差接近零，是明显警报。

## 小结

- [ ] mask 决定模型必须从哪些上下文推断哪些目标。
- [ ] stop-gradient 阻止目标分支被预测损失直接拉动。
- [ ] EMA 提供缓慢变化的 target encoder。
- [ ] 防坍缩要用表示统计和下游 probe 检查。
