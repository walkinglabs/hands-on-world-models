# 10.2 系统级评测基准与榜单

世界模型同时包含观测预测、状态转移、奖励建模和决策用途。单帧画得清楚，不代表动作真的改变未来；单步误差很小，也不代表长时程展开稳定。因此，本节把评测拆成图像保真、特征分布、动作条件展开和闭环任务四层。

<div align="center">
<img src="/figures/10-evaluate-and-invent/source/02-systematic-evaluation/crafter-fig1.png" alt="Crafter 的程序生成世界同时包含地形、资源与生物，展示系统级榜单实际要求智能体处理的开放式环境。" width="86%">

_图 10.2-1：Crafter 的程序生成世界同时包含地形、资源与生物，展示系统级榜单实际要求智能体处理的开放式环境。 出处：Danijar Hafner，[Benchmarking the Spectrum of Agent Capabilities](https://arxiv.org/abs/2109.06780)（2022），Figure 1。_
</div>

这些层次不宜被简单加成一个总分。更稳妥的报告方式是同时给出指标、数据切分、展开长度、动作协议和置信区间，让读者看见模型在哪一层开始失效。

## 10.2.1 历史演进与学术追溯

在深入具体的数学推导之前，我们有必要追溯这些评测指标诞生的历史脉络。评测标准的发展，本质上是深度学习模型能力边界不断扩张的倒影。

ImageNet 为大规模静态图像识别提供了数据集与评测基准 [[Deng et al., 2009]](https://doi.org/10.1109/CVPR.2009.5206848)。GAN [[Goodfellow et al., 2014]](https://arxiv.org/abs/1406.2661) 与 VAE [[Kingma & Welling, 2013]](https://arxiv.org/abs/1312.6114) 推动了高维图像生成后，研究者又提出 Inception Score（IS）[[Salimans et al., 2016]](https://arxiv.org/abs/1606.03498) 与 Fréchet Inception Distance（FID）[[Heusel et al., 2017]](https://arxiv.org/abs/1706.08500)，用预训练网络的特征来比较生成样本的质量、多样性或特征分布。这里的 Inception 是网络名称，不应译作“初始”。

进入视频生成与预测时代后，静态特征的分布距离被进一步延展到时空维度。通过利用 3D 卷积神经网络（如 I3D），弗雷歇视频距离（Fréchet Video Distance, FVD）被提出以衡量视频序列的时空一致性 [[Unterthiner et al., 2018]](https://arxiv.org/abs/1812.01717)。

Ha 和 Schmidhuber 的 World Models 同时报告了视觉重构、潜在动力学与控制任务回报 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)；这是一种具体的深度世界模型实现，并不是“世界模型理论到 2018 年才被提出”的证据。Dreamer 进一步以环境回报和数据效率评估潜在想象训练出的策略 [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603)。这些论文说明，世界模型评测除了观测预测质量，还应覆盖动作条件动力学与下游控制效果；“必须采用某套唯一基准”则是本文的评测建议，不是论文原结论。

## 10.2.2 像素与结构的低维映射：基础视觉评测

当我们希望评估世界模型的观测模型（Observation Model）或者解码器（Decoder）重建当前环境的能力时，最直观的第一步，是衡量生成的预测图像与真实的观测图像之间的相似度。我们首先从高中阶段最基础的几何距离切入。

### 均方误差与峰值信噪比 (PSNR)

假设我们要比较两张分辨率完全相同的图像，真实的图像为 $I$，模型生成的图像为 $K$。在数字图像处理中，每张图像可以被展开为一个长度为 $m \times n$ 的一维向量（此处暂且忽略通道数）。衡量两个向量之间最朴素的方法，便是计算它们在多维欧几里得空间中的欧式距离的平方。

我们定义这两张图像之间的均方误差（Mean Squared Error, MSE）为每个对应像素点差值的平方的平均数：

$$ \text{MSE} = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i, j) - K(i, j)]^2 $$

然而，均方误差存在一个明显的缺陷：它的绝对数值缺乏一致的物理尺度。为了更符合人类对“信号质量”的直观感受，学术界引入了物理学和信号处理领域中的信噪比概念，提出了峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）。

我们可以将真实的图像 $I$ 视为纯净的“信号”，而模型重建产生的偏差 $I - K$ 视为注入的“噪声”。PSNR 通过对数尺度来衡量最大可能信号能量与噪声能量之间的比值：

$$ \text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right) = 20 \cdot \log_{10} \left( \frac{\text{MAX}_I}{\sqrt{\text{MSE}}} \right) $$

其中，$\text{MAX}_I$ 是图像像素可能的最大取值（例如对于 8-bit 的图像，$\text{MAX}_I = 255$）。PSNR 的单位为分贝（dB）。由于对数函数的单调递增特性，PSNR 的值越大，表示重建图像由于误差带来的“噪声”越小，模型的保真度也就越高。

### 结构相似性指数 (SSIM)

PSNR 逐像素汇总误差，没有显式建模邻域结构。两张图即使具有相似的边缘和轮廓，只要发生轻微平移，PSNR 也可能明显下降；反过来，较高 PSNR 也不保证局部结构自然。

为了弥补这一缺陷，结构相似性指数（Structural Similarity Index, SSIM）被提出。SSIM 巧妙地将图像块的比较解耦为三个独立的统计物理量：亮度（Luminance）、对比度（Contrast）和结构（Structure）。

给定两个局部图像窗口 $x$ 和 $y$：

1. **亮度比较**：我们用均值 $\mu_x = \frac{1}{N} \sum_{i=1}^N x_i$ 来近似局部区域的亮度。
2. **对比度比较**：我们用方差（或标准差） $\sigma_x = \left( \frac{1}{N-1} \sum_{i=1}^N (x_i - \mu_x)^2 \right)^{1/2}$ 来近似局部区域的对比度。
3. **结构比较**：在排除了亮度和对比度的影响后，我们利用协方差 $\sigma_{xy} = \frac{1}{N-1} \sum_{i=1}^N (x_i - \mu_x)(y_i - \mu_y)$ 来严格衡量二者结构形状的线性相关性。

通过引入极小的常数 $c_1, c_2$ 以防止分母为零，这三个维度被组合为最终的局部 SSIM 计算公式：

$$ \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)} $$

在常见参数与非负图像范围下，SSIM 通常不超过 1；两张局部图像完全一致时取 1。通过滑动窗口计算局部 SSIM 再平均，可以同时反映亮度、对比度和局部相关结构，但它仍不是人类偏好的完整替代。

<div align="center">
<img src="/figures/10-evaluate-and-invent/source/02-systematic-evaluation/ssim-fig2.png" alt="SSIM 原论文让多种失真保持相同 MSE，却呈现显著不同的结构质量，说明逐像素误差无法区分所有视觉退化。" width="86%">

_图 10.2-2：SSIM 原论文让多种失真保持相同 MSE，却呈现显著不同的结构质量，说明逐像素误差无法区分所有视觉退化。 出处：Zhou Wang；Alan C. Bovik；Hamid R. Sheikh；Eero P. Simoncelli，[Image Quality Assessment: From Error Visibility to Structural Similarity](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf)（2004），Figure 2。_
</div>

## 10.2.3 从单样本到概率分布：特征空间的几何度量

虽然 PSNR 和 SSIM 能够衡量确定的、一一对齐的重建误差，但世界模型对未来的预测往往面临着环境的内禀随机性（Aleatoric Uncertainty）。在高度动态的环境中，模型生成的未来图像在绝对像素排列上可能与某次特定的真实观测结果在空间上并不完全重合，但在物理规律和宏观语义分布上却是完全合理的。

此时，我们不能再局限于“衡量单一样本之间的欧氏距离”，而是必须跨越到更高的概率抽象层级：“衡量两个整体概率分布之间的距离”。

### 降维视角的分布距离推导：1D 高斯模型

在开始复杂的矩阵推导前，让我们先考虑最简单的情况：假设真实的特征分布 $P$ 和模型生成的特征分布 $Q$ 都是一维实数空间上的正态（高斯）分布。即 $P \sim \mathcal{N}(\mu_1, \sigma_1^2)$，且 $Q \sim \mathcal{N}(\mu_2, \sigma_2^2)$。

我们希望用一个指标来量化这两个分布有多“不同”。严格的直觉告诉我们，这种概率分布的差异来源于两部分：
其一，是它们期望（即概率密度中心位置）的平移偏移，反映为均值之差的平方 $(\mu_1 - \mu_2)^2$；
其二，是它们散布程度（概率密度的胖瘦程度）的差异，反映为标准差之差的平方 $(\sigma_1 - \sigma_2)^2$。

对一维高斯分布，Wasserstein-2 距离平方有解析解，恰好分成均值偏移和标准差偏移两项：

$$ W_2^2(P, Q) = (\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2 = (\mu_1 - \mu_2)^2 + (\sigma_1^2 + \sigma_2^2 - 2\sigma_1\sigma_2) $$

<div align="center">
<img src="/figures/10-evaluate-and-invent/latex/02-systematic-evaluation/wasserstein-mean-scale-plane.png" alt="两个一维高斯在均值和标准差参数平面中的差异构成直角三角形，Wasserstein 距离是其斜边" width="86%">

_图 10.2-3：两个一维高斯的均值差与标准差差是参数平面上的两条正交分量，W2 距离由二者共同决定。本文根据上式绘制。_
</div>

> 想像你是一个城市规划者，你需要将一堆沙子（代表模型生成的特征分布）搬运去填补一个特定形状的坑（代表真实的特征分布）。不仅沙子的总量必须对等，你还需要考虑搬运沙子所耗费的距离与精力。如果我们将这种“最小化总体搬运成本”的几何直觉用数学语言严密地表达出来，并假设这些沙堆都服从多维正态分布，那么我们计算出的最优传输成本，正是弗雷歇距离（Fréchet Distance）。

### 弗雷歇 Inception 距离（FID）与时空扩展（FVD）

现在，我们将一维空间的方差延展推演至高维特征空间。此时，标量均值 $\mu$ 成为了一个多维列向量，而标量方差 $\sigma^2$ 则升级为了协方差矩阵 $\Sigma$。对于多维高斯分布，标准差的乘积项 $\sigma_1\sigma_2$ 无法直接通过简单的矩阵乘法建立对应关系。严密的黎曼流形数学推导表明，这一交叉项被矩阵乘积平方根的迹（Trace，即矩阵主对角线元素之和）所替代。

由此得到两组多维高斯分布间的 $W_2^2$ 距离。对一般协方差矩阵，更明确的对称写法是：

$$ d^2(P, Q) = \|\mu_p - \mu_q\|_2^2 + \operatorname{Tr}\left(\Sigma_p + \Sigma_q - 2\left(\Sigma_p^{1/2}\Sigma_q\Sigma_p^{1/2}\right)^{1/2}\right) $$

FID 的实现常把迹项写成 $\operatorname{Tr}(\Sigma_p+\Sigma_q-2(\Sigma_p\Sigma_q)^{1/2})$；数值库还需处理有限样本造成的非对称和微小复数误差。

在实际工程操作中，直接在原始像素空间计算协方差矩阵不仅计算复杂度极其高昂，而且像素空间的分布既不符合高斯假设，也不符合人类的语义认知。因此，我们引入一个在 ImageNet 上预训练的 Inception-v3 网络作为特征空间映射器。我们将大量真实图像与生成图像均前向传播至该网络的深层（通常截取分类全连接层之前的 2048 维全局平均池化层），以此特征向量集合来计算统计量 $\mu$ 和 $\Sigma$。这就是广泛采用的弗雷歇 Inception 距离（FID）。

<div align="center">
<img src="/figures/10-evaluate-and-invent/source/02-systematic-evaluation/fid-fig3.png" alt="FID 原论文逐步增强噪声、模糊、遮挡与数据污染，展示特征分布距离如何响应不同扰动强度。" width="86%">

_图 10.2-4：FID 原论文逐步增强噪声、模糊、遮挡与数据污染，展示特征分布距离如何响应不同扰动强度。 出处：Martin Heusel et al.，[GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)（2017），Figure 3。_
</div>

FVD 沿用“提取特征后拟合高斯并比较统计量”的思路，但使用视频分类网络的时空特征。它对运动与片段结构比逐帧 FID 更敏感，却仍受特征网络、预处理、样本量和实现细节影响；低 FVD 也不能单独证明动作因果正确。

<div align="center">
<img src="/figures/10-evaluate-and-invent/source/02-systematic-evaluation/fvd-fig1.png" alt="FVD 原论文把 BAIR 生成视频按分数排序，展示时空特征度量如何区分视频生成模型的整体质量。" width="86%">

_图 10.2-5：FVD 原论文把 BAIR 生成视频按分数排序，展示时空特征度量如何区分视频生成模型的整体质量。 出处：Thomas Unterthiner et al.，[Towards Accurate Generative Models of Video: A New Metric & Challenges](https://arxiv.org/abs/1812.01717)（2018），Figure 1。_
</div>

## 10.2.4 世界模型的专属核心：动作条件下的动力学评测

FID 和 FVD 比较样本或视频特征分布，本身不会验证同一初始状态下更换动作是否得到相应未来。用于控制的世界模型还需要动作条件评测：固定历史，替换动作，并检查预测变化是否与环境变化一致。

假设智能体在时间步 $t$ 处于内部潜在隐状态 $s_t$，执行了动作 $a_t$。世界模型的动力学模型（Dynamics Model）进行一步状态转移 $s_{t+1} = f_\theta(s_t, a_t)$。这要求我们在评测时，必须将历史动作序列作为前置条件进行强干预验证。

### 动作条件下的长程累积误差 (Action-Conditioned Long-term Rollout Error)

世界模型的长期想象（Long-horizon Imagination）能力至关重要，因为强化学习中的策略规划往往需要看透未来几十乃至上百步。我们在评测中不仅要衡量单步前向预测的准确率，更要衡量连续多步自回归生成时的误差累积速度。

给定一个来自于真实环境收集的初始状态观测 $x_0$ （被编码为隐状态 $s_0$）和一段长度为 $T$ 的真实历史动作序列 $\mathcal{A} = \{a_0, a_1, \dots, a_{T-1}\}$。世界模型在此完全断开与外界真实观测的连接闭环，仅凭借初始状态 $s_0$ 和给定动作序列，在潜在向量空间中进行连续 $T$ 步的纯自回归展开（Rollout），生成一系列预测的未来状态 $\hat{s}_1, \dots, \hat{s}_T$，以及随之解码出的预测观测 $\hat{x}_1, \dots, \hat{x}_T$。

我们将这串预测序列与真实观测序列 $x_1, \dots, x_T$ 逐步比较。随着 $t$ 增大，单步偏差可能累积、保持有界，或在不稳定动力学中被放大；是否指数增长取决于局部动力学。绘制 $t=1$ 到 $t=T$ 的 MSE 或 PSNR 曲线可以诊断长程误差，但还应同时检查感知指标、状态一致性与闭环任务表现。

### 奖励预测误差 (Reward Prediction Accuracy)

在强化学习的上下文中，除了高维的视觉观测之外，环境在每一个时间步还会返回一个标量奖励 $r_t$。世界模型的奖励预测器（Reward Predictor）需要根据转移后的隐状态预测该奖励：$\hat{r}_t = R_\phi(\hat{s}_t)$。

奖励常标记任务中的关键事件，例如越过障碍或到达终点。图像边缘略模糊的模型仍可能给规划提供有用信号，但前提是奖励事件的幅值和时间位置足够准确。评测时可以报告奖励误差、事件分类的精确率与召回率，以及预测峰值相对真值的时间偏移。

## 10.2.5 具身智能与系统级榜单概览

目前并不存在覆盖所有世界模型形态的统一榜单。实践中常借用强化学习、开放世界和机器人控制基准，分别检查数据效率、长时程技能与任务成功率。

1. **Atari 100k**：限制每个游戏 100k 个环境步，用回报衡量低数据预算下的策略学习。它能检验数据效率，但不能单独定位视觉模型、动力学模型或规划器哪一部分出了问题。
2. **Crafter / MineDojo**：提供开放式、长时程的采集、制作和探索任务。二者的任务定义与报告协议不同，不能直接混成一个排行榜；使用时应明确任务集、训练数据和是否允许预训练。
3. **RoboDesk / Meta-World**：提供机械臂连续控制任务，可报告逐任务成功率、跨任务平均值和分布外变化下的性能。成功率是关键系统指标，但仍应配合安全违规、动作平滑度和失败类型分析。

一份可复核的报告至少应覆盖与任务相符的观测指标、动作条件多步展开、奖励或事件预测、闭环任务结果，以及不同随机种子和分布外切分。并非每个模型都需要 PSNR、FID 和 FVD；没有像素解码器的模型就应改用状态、表征或任务层指标。

## 10.2.6 核心评测指标的代码实现

接下来，我们以深度学习框架的标准代码规范，严谨地实现图像级别的两大基础评测指标：源自 MSE 的 PSNR，以及综合了亮度与方差的 SSIM。同时，我们也将展示如何利用预训练模型提取特征向量以备计算 FID 统计量。

```python
import torch
import torch.nn.functional as F
import math

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    计算给定两批次图像的峰值信噪比(PSNR)。
    假设输入张量的取值范围已被归一化至 [0, max_val]。
    """
    # [计算两个图像张量之间所有对应元素的均方误差 (MSE)]
    mse = torch.mean((img1 - img2) ** 2)

    # 极值情况处理：如果两张图像完全一致，MSE为0，PSNR理论上趋于无穷大
    if mse == 0:
        return float('inf')

    # [严格依据对数能量衰减公式计算 PSNR 值]
    psnr = 20 * math.log10(max_val / math.sqrt(mse.item()))
    return psnr

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> torch.Tensor:
    """
    计算基于滑动窗口的结构相似性指数 (SSIM)。
    """
    channels = img1.size(1)

    # [利用平均池化计算局部滑动窗口的均值 μ_x 和 μ_y (近似代表局部亮度)]
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # [计算局部方差 σ_x^2, σ_y^2 以及协方差 σ_xy (代表对比度与结构相关性)]
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

    # 设定极小常数以确保除法运算的数值稳定性
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # [严格依照原始公式，在分子和分母中组合亮度、对比度与结构三项因子]
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 返回整幅图像平面的平均 SSIM 标量评分
    return ssim_map.mean()
```

下面的代码只演示“提取特征并计算均值、协方差”这一步，不用于复现论文或榜单的标准 FID。正式比较必须固定特征权重、输入归一化、样本数和矩阵平方根实现；最好直接采用经过交叉验证的同一评测库。

```python
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights

def get_inception_features(images: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    """
    使用在 ImageNet 上预训练的 Inception-v3 模型提取高阶特征。
    """
    # [加载预训练权重，并务必设置为评估模式，防止 Batch Norm 或 Dropout 引入随机性扰动]
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False)
    # 移除最后的分类映射全连接层，以获取更底层的连续语义特征
    model.fc = torch.nn.Identity()
    model = model.to(images.device).eval()

    features_list = []
    with torch.no_grad():
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i+batch_size]
            # [通过双线性插值强制将图像缩放至299x299，以吻合Inception-v3初始感受野和步长的空间设计假设]
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            mean = batch.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = batch.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            batch = (batch - mean) / std
            features = model(batch)
            features_list.append(features)

    return torch.cat(features_list, dim=0)

def compute_statistics(features: torch.Tensor):
    """
    计算高维特征分布的均值向量与协方差矩阵。
    """
    # 将计算图脱离并将张量转移至 CPU，转化为 numpy 数组以便调用高阶代数库进行大规模矩阵运算
    features_np = features.cpu().numpy()

    # [沿样本批量维度展开，计算各维特征的算术均值向量 μ]
    mu = np.mean(features_np, axis=0)

    # [严格计算多维特征之间的协方差矩阵 Σ，行维度代表变量(特征)，列维度代表具体的观察值(样本)]
    sigma = np.cov(features_np, rowvar=False)

    return mu, sigma
```

## 10.2.7 小结

本节建立了四层评测：PSNR/SSIM 检查对齐图像，FID/FVD 比较特征分布，动作条件展开检查动力学响应，闭环成功率检查系统效用。它们必须连同数据切分、样本量、展开长度和随机种子一起报告；指标之间不一致时，差异本身就是下一步故障分析的入口。
