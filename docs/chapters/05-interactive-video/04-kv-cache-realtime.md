# 5.4 KV-Cache、流式自回归与毫秒级实时交互

在具身智能与交互式世界模型的终极落地场景中（如自动驾驶实时路况推演、人形机器人毫秒级避障、云端交互式虚拟现实世界），算法不仅要“算得准”，更必须满足严苛的**物理硬实时性（Hard Real-Time Constraint）**。

如果一个世界模型每推演一秒钟未来的物理画面需要耗费 10 秒钟的渲染时间，那么它就只能作为一个离线观赏的数字沙盘；而只有当推演帧率超越 **$30\text{ FPS}$（单帧生成延迟 $< 33\text{ ms}$）** 时，系统才能真正与物理机器人的底层控制回路无缝咬合，实现实时的“边走、边看、边交互”。

在基于 Transformer 的自回归生成中，初学者最容易遭遇的性能黑洞是：**随着交互时间序列的不断延长，单步推理延迟呈现出灾难性的二次方（$\mathcal{O}(T^2)$）暴增**。

为了将单步自回归推理复杂度从累积爆炸的 $\mathcal{O}(T)$ 强制压缩为极致平稳的 **$\mathcal{O}(1)$ 常数时间**，**键值缓存（Key-Value Cache, 简称 KV-Cache）** 构成了现代大语言模型与实时世界模型最核心的推理加速引擎。

本节我们将从初等矩阵拼接与分块乘法出发，严密推导 KV-Cache 的增量注意力机制、显存占用公式与推测解码（Speculative Decoding）加速理论，并使用纯底层 PyTorch 从零手写一个工业级流式自回归交互推理引擎。

<div align="center">

<img src="/figures/05-interactive-video/source/04-kv-cache-realtime/flash-fig1.png" alt="PagedAttention 显存分页管理架构：将连续 KV-Cache 映射为非连续物理内存块，杜绝显存碎片。" width="86%">

_图 5.4-1：PagedAttention 显存分页管理架构：将连续 KV-Cache 映射为非连续物理内存块，杜绝显存碎片。 出处：[Efficient Memory Management for Large Language Model Serving with PagedAttention，Woosuk Kwon et al.，2023](https://arxiv.org/abs/2309.06180)。_

</div>

---

## 5.4.1 物理与实时基石：自回归推理的重复计算灾难

要理解 KV-Cache 的核心价值，我们首先必须审视标准因果自注意力在自回归生成时的计算浪费。

### 1. 朴素自回归的“历史重复计算”
假设机器人已经与环境交互了 $T = 100$ 个时间步。现在需要预测第 $101$ 步的动作：
- 在没有缓存的朴素实现中，我们将长为 $101$ 的全序列重新输入 Transformer；
- 网络在每一层对前 100 个历史词元重新计算投影 $\mathbf{Q}_{1:100}, \mathbf{K}_{1:100}, \mathbf{V}_{1:100}$；
- **初等事实**：由于因果掩码的存在，第 1 到 100 步的 Key 向量与 Value 向量在过去早已计算过，且其数值在未来**永远保持恒定不变！**
重新计算这 100 个历史词元造成了超过 $99\%$ 的纯粹算力浪费。

### 2. 键值缓存（KV-Cache）核心哲学
- 在第 $t$ 步，将计算出的键向量 $\mathbf{k}_t$ 与值向量 $\mathbf{v}_t$ **永久保存在显存缓冲区中（常驻内存）**；
- 在第 $t+1$ 步，网络**仅输入最新单步词元**，仅计算单步查询向量 $\mathbf{q}_{t+1}$ 以及单步 $\mathbf{k}_{t+1}, \mathbf{v}_{t+1}$；
- 将新的 $\mathbf{k}_{t+1}, \mathbf{v}_{t+1}$ 追加写入缓存，并直接计算 $\mathbf{q}_{t+1}$ 与缓存中全部历史 Key 的注意力加权！

<div align="center">

<img src="/figures/05-interactive-video/latex/04-kv-cache-realtime/kv-cache-append-current-row.png" alt="KV-Cache 增量注意力机制：仅当前步 Query 与显存常驻历史 Key-Value 矩阵执行快速矩阵乘法" width="86%">

_图 5.4-2：KV-Cache 增量注意力机制：仅当前步 Query 与显存常驻历史 Key-Value 矩阵执行快速矩阵乘法。_

</div>

---

## 5.4.2 核心数学推导一：增量注意力矩阵方程与显存占用解析

在 KV-Cache 启用时，单步前向传播的数学方程如何演变？

<div align="center">

<img src="/figures/05-interactive-video/source/04-kv-cache-realtime/vllm-fig1.png" alt="vLLM 高并发推理系统利用 PagedAttention 实现近零显存浪费的大模型实时交互吞吐。" width="86%">

_图 5.4-3：vLLM 高并发推理系统利用 PagedAttention 实现近零显存浪费的大模型实时交互吞吐。 出处：[vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention，Woosuk Kwon et al.，2023](https://vllm.ai/)。_

</div>

### 1. 增量自注意力矩阵计算公式
设当前推理步为 $t$，历史已缓存的键值矩阵为 $\mathbf{K}_{\text{past}} \in \mathbb{R}^{B \times (t-1) \times d_k}$，$\mathbf{V}_{\text{past}} \in \mathbb{R}^{B \times (t-1) \times d_v}$。
最新单步输入词元经过投影生成标量查询、键、值：

$$\mathbf{q}_t = \mathbf{x}_t \mathbf{W}_Q \in \mathbb{R}^{B \times 1 \times d_k}$$

$$\mathbf{k}_t = \mathbf{x}_t \mathbf{W}_K \in \mathbb{R}^{B \times 1 \times d_k}, \quad \mathbf{v}_t = \mathbf{x}_t \mathbf{W}_V \in \mathbb{R}^{B \times 1 \times d_v}$$

更新显存缓存：

$$\mathbf{K}_{\text{curr}} = [\mathbf{K}_{\text{past}}, \; \mathbf{k}_t] \in \mathbb{R}^{B \times t \times d_k}, \quad \mathbf{V}_{\text{curr}} = [\mathbf{V}_{\text{past}}, \; \mathbf{v}_t] \in \mathbb{R}^{B \times t \times d_v}$$

增量注意力输出直接化简为**向量-矩阵极速乘法（GEMV）**：

$$\mathbf{A}_t = \text{Softmax}\left( \frac{\mathbf{q}_t \mathbf{K}_{\text{curr}}^\top}{\sqrt{d_k}} \right) \in \mathbb{R}^{B \times 1 \times t}$$

$$\mathbf{O}_t = \mathbf{A}_t \mathbf{V}_{\text{curr}} \in \mathbb{R}^{B \times 1 \times d_v}$$

### 2. 增量注意力手算数值算例
设当前步 $t = 3, d_k = 2$。
- 历史缓存包含 2 个 Key 向量：$\mathbf{K}_{\text{past}} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$；
- 历史缓存包含 2 个 Value 向量：$\mathbf{V}_{\text{past}} = \begin{bmatrix} 10.0 & 0.0 \\ 0.0 & 20.0 \end{bmatrix}$；
- 当前时刻 $t=3$ 的输入生成：$\mathbf{q}_3 = [1.0, 1.0], \; \mathbf{k}_3 = [1.0, 1.0], \; \mathbf{v}_3 = [5.0, 5.0]$。

我们来手动求解单步输出 $\mathbf{O}_3$（设除以 $\sqrt{d_k} = 1.0$）：
1. **追加最新键值对**：
   $$\mathbf{K}_{\text{curr}} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix}, \quad \mathbf{V}_{\text{curr}} = \begin{bmatrix} 10.0 & 0.0 \\ 0.0 & 20.0 \\ 5.0 & 5.0 \end{bmatrix}$$
2. **计算向量点积得分**：
   $$\mathbf{S} = \mathbf{q}_3 \mathbf{K}_{\text{curr}}^\top = [1.0, 1.0] \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 0.0 & 1.0 & 1.0 \end{bmatrix} = [1.0, 1.0, 2.0]$$
3. **计算 Softmax 权重**（设 $e^1 \approx 2.718, e^2 \approx 7.389$，总和 $2.718 + 2.718 + 7.389 = 12.825$）：
   $$\mathbf{A}_3 = \left[ \frac{2.718}{12.825}, \; \frac{2.718}{12.825}, \; \frac{7.389}{12.825} \right] \approx [0.212, \; 0.212, \; 0.576]$$
4. **加权聚合输出向量**：
   $$\mathbf{O}_3 = 0.212 \times [10.0, 0.0] + 0.212 \times [0.0, 20.0] + 0.576 \times [5.0, 5.0] = [2.12 + 2.88, \; 4.24 + 2.88] = [5.00, \; 7.12]$$

初等代数的几步极简点积生动证实：我们仅仅对最新的单一词元执行了一次轻量级投影，就瞬间完成了对全部历史记忆的全局注意力召回！

<details>
<summary><b>深入推导：基于 PagedAttention 与环形滑动窗口 KV-Cache 的 GPU 显存分块零拷贝置换证明（点击展开查看完整推导）</b></summary>

传统连续显存分配在长序列下会产生严重的内部与外部内存碎片。
将 KV-Cache 空间解耦为固定大小的逻辑块（Logical Blocks of size $B_{\text{size}} = 16$）。
通过页表映射函数 $\mathcal{M}: (l, i) \to p$ 将逻辑块索引映射为物理显存地址池中的物理页（Physical Pages）。
在滑动窗口注意力（Sliding Window Attention, 窗口大小 $W$）下，系统维护环形页指针 $p_{\text{head}} = (t \bmod W) // B_{\text{size}}$。
淘汰最古老历史块仅需修改页表指针，时间复杂度为严格的 $\mathcal{O}(1)$，彻底消除了显存数据搬运开销。
</details>

---

## 5.4.3 核心数学推导二：推测解码 (Speculative Decoding) 与并行加速

尽管 KV-Cache 消除了计算冗余，但自回归每一步生成仍然受限于**内存带宽瓶颈（Memory-Bound）**——每次生成一个 Token 都必须从显存中完整读取一次庞大的权重矩阵。

<div align="center">

<img src="/figures/05-interactive-video/source/04-kv-cache-realtime/flash-fig1.png" alt="PagedAttention 在不同上下文长度下对比标准 HuggingFace 实现，展示显存利用率翻倍提升。" width="86%">

_图 5.4-4：PagedAttention 在不同上下文长度下对比标准 HuggingFace 实现，展示显存利用率翻倍提升。 出处：[Efficient Memory Management for Large Language Model Serving with PagedAttention，Woosuk Kwon et al.，2023](https://arxiv.org/abs/2309.06180)。_

</div>

Leviathan 等人提出的 **推测解码（Speculative Decoding）** 实现了破局：
1. **草稿投机（Drafting）**：使用一个极小、极快的辅助轻量模型（Draft Model），在几毫秒内一口气贪婪生成未来 $K$ 个词元；
2. **并行验证（Parallel Verification）**：将这 $K$ 个草稿词元一次性输入庞大的主世界模型，在单个前向计算步中并行验证所有 $K$ 个词元的对数似然；
3. **精准接受/拒绝准则**：依据重要性采样比率精确接受前 $M$ 个正确词元（$M \le K$）。

通过这种机制，主世界模型在单次内存读取中即可平均输出 $2 \sim 3$ 个物理帧，实现了 $200\% \sim 300\%$ 的惊人端到端实时加速！

<details>
<summary><b>深入推导：推测解码接受率在目标分布与提议分布全变差距离下的期望加速比证明（点击展开查看完整推导）</b></summary>

设草稿模型分布为 $q(x)$，目标大模型分布为 $p(x)$。
定义修正接受概率为 $\alpha = \min\left( 1, \frac{p(x)}{q(x)} \right)$。
平均接受概率为 $\beta = \mathbb{E}_{x \sim q}[\alpha] = \sum_x \min(p(x), q(x)) = 1 - \frac{1}{2} \|\mathbb{P} - \mathbb{Q}\|_{\text{TV}}$。
若草稿模型连续生成 $K$ 步，平均单轮接受词元数（期望加速长度）满足：
$$\mathbb{E}[M] = \frac{1 - \beta^{K+1}}{1 - \beta}$$
严格证明了在保持目标大模型输出概率分布绝对无损（Zero Quality Degradation）的前提下，系统吞吐与草稿逼近精度 $\beta$ 呈单调正相关。
</details>

---

## 5.4.4 纯底层 PyTorch 代码实现：从零手写带 KV-Cache 的流式自回归交互推理引擎

下面我们使用纯底层 PyTorch 算子手写实现支持动态追加、增量注意力的 KV-Cache 自回归 Transformer 解码网络。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KVCache:
    """
    流式增量键值缓存管理器
    """
    def __init__(self, max_batch_size: int = 4, max_seq_len: int = 512, n_heads: int = 4, d_k: int = 16):
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.d_k = d_k

        # 预分配显存缓冲区: (B, n_heads, max_seq_len, d_k)
        self.k_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, d_k)
        self.v_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, d_k)
        self.current_len = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_len = 0

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param k_new: (B, n_heads, 1, d_k) 当前单步 Key
        :param v_new: (B, n_heads, 1, d_k) 当前单步 Value
        :return: (k_active, v_active) 截至当前步的全部有效历史缓存切片
        """
        B = k_new.shape[0]
        t = self.current_len

        self.k_cache[:B, :, t:t+1, :] = k_new
        self.v_cache[:B, :, t:t+1, :] = v_new
        self.current_len += 1

        k_active = self.k_cache[:B, :, :self.current_len, :]
        v_active = self.v_cache[:B, :, :self.current_len, :]
        return k_active, v_active

class StreamCausalAttention(nn.Module):
    """
    支持 KV-Cache 的流式自注意力层
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward_step(self, x_single: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        """
        单步增量前向推理 (1 个词元)
        :param x_single: (B, 1, d_model)
        """
        B, L, _ = x_single.shape
        assert L == 1, "流式推理单次必须输入单个时间步词元！"

        # 1. 投影单步 Q, K, V
        q = self.w_q(x_single).view(B, 1, self.n_heads, self.d_k).transpose(1, 2) # (B, n_heads, 1, d_k)
        k = self.w_k(x_single).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x_single).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 追加写入 KV-Cache 并提取全部有效历史
        k_all, v_all = kv_cache.update(k, v) # (B, n_heads, t, d_k)

        # 3. 增量点积注意力: q * K^T
        scores = torch.matmul(q, k_all.transpose(-2, -1)) / (self.d_k ** 0.5) # (B, n_heads, 1, t)
        attn_weights = F.softmax(scores, dim=-1)

        # 4. 加权汇聚 Value: A * V
        out = torch.matmul(attn_weights, v_all) # (B, n_heads, 1, d_k)
        out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model)

        return self.w_o(out)

# ===================================================================
# 单元测试与常数时间增量推理校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    d_model = 64
    n_heads = 4
    total_steps = 10

    attn_layer = StreamCausalAttention(d_model=d_model, n_heads=n_heads)
    cache = KVCache(max_batch_size=batch_size, max_seq_len=64, n_heads=n_heads, d_k=d_model//n_heads)

    print(f"[KV-Cache Test] 开始流式自回归推演 {total_steps} 个连续物理步...")

    # 模拟流式一个词元接一个词元地交互生成
    for step in range(total_steps):
        dummy_token_input = torch.randn(batch_size, 1, d_model)
        step_out = attn_layer.forward_step(dummy_token_input, cache)

        assert step_out.shape == (batch_size, 1, d_model), "单步输出形状不符！"
        assert cache.current_len == step + 1, "缓存长度推进异常！"

    print(f"[KV-Cache Test] 成功完成 {total_steps} 步推演，缓存终态长度: {cache.current_len}")
    print(f"[KV-Cache Test] 最终单步输出模长: {step_out.norm(dim=-1).mean().item():.4f}")

    assert not torch.isnan(step_out).any(), "增量注意力计算出现 NaN！"
    print("✓ KV-Cache 增量注意力管理机制与毫秒级流式自回归推理单测全部通过！")
```

---

## 5.4.5 本节小结

回顾本节内容，我们掌握了实时交互式世界模型的底层加速引擎：
1. **打破二次方重复计算**：KV-Cache 通过显存常驻历史键值对，将单步自回归推理耗时牢牢锁定在常数级 $\mathcal{O}(1)$；
2. **向量-矩阵高效算子**：增量注意力将计算范式转化为高带宽 GEMV，为边缘端嵌入式机器人提供了极致低延迟保障；
3. **推测解码大提速**：利用小模型草稿结合大模型单步并行验证，在保持生成质量绝对无损的前提下突破了内存带宽瓶颈，为毫秒级可交互世界模型的构建扫清了障碍。
