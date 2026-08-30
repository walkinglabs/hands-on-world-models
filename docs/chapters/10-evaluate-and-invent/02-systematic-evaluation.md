# 10.2 系统级评测基准与榜单
:label:sec_systematic_evaluation

在这个由数据和算力驱动的深度学习时代，如何科学、客观地衡量一个模型的表现，与其架构设计同等重要。对于传统的监督学习（如图像分类任务），评测基准通常是简单明了的准确率（Accuracy）或交叉熵损失（Cross-Entropy Loss）。然而，当我们步入世界模型（World Models）领域时，模型的职责发生了根本性的转变：它不再仅仅是预测离散的标签，而是需要深刻理解环境的物理动态规律、在多步时间跨度上生成未来状态的观测，甚至在潜在的想象空间中规划出能够最大化奖励的动作轨迹。

这种融合了多模态（视觉观测、离散或连续动作、标量奖励）与长时空耦合的复杂输出系统，要求我们必须放弃单一维度的考察标准，转而建立一套高度系统化、多维度的综合评测基准与榜单。

## 10.2.1 历史演进与学术追溯
:label:subsec_evaluation_history

在深入具体的数学推导之前，我们有必要追溯这些评测指标诞生的历史脉络。评测标准的发展，本质上是深度学习模型能力边界不断扩张的倒影。

在早期计算机视觉领域，研究者们主要关注模型在静态图像上的识别与分类能力，ImageNet 图像识别挑战赛的 Top-1 和 Top-5 准确率成为了衡量模型性能的黄金准则 `[Deng et al., 2009]`。随着生成对抗网络（GANs）`[Goodfellow et al., 2014]` 与变分自编码器（VAEs）`[Kingma & Welling, 2013]` 的崛起，模型开始具备了“从无到有”生成高维图像的能力。由于生成任务不存在绝对唯一的正确答案（Ground Truth），研究者们提出了诸如初始分数（Inception Score, IS）`[Salimans et al., 2016]` 和弗雷歇初始距离（Fréchet Inception Distance, FID）`[Heusel et al., 2017]`，通过引入预训练的神经网络提取高维特征，在特征分布层面上对生成图像的真实感和多样性进行数学度量。

进入视频生成与预测时代后，静态特征的分布距离被进一步延展到时空维度。通过利用 3D 卷积神经网络（如 I3D），弗雷歇视频距离（Fréchet Video Distance, FVD）被提出以衡量视频序列的时空一致性 `[Unterthiner et al., 2018]`。

直至世界模型理论被系统性地提出 `[Ha & Schmidhuber, 2018]`，人们意识到，世界模型的核心并非仅仅是“逼真的视频生成器”，它更是智能体（Agent）理解世界因果关系的大脑。正如在 Dreamer 架构 `[Hafner et al., 2019]` 中所展现的那样，评测基准必须从“观测保真度”向“动作条件下的动力学一致性”和“下游任务规划成功率”发生范式转移。本节将循序渐进地拆解这套由浅入深、层层递进的系统级评测体系。

## 10.2.2 像素与结构的低维映射：基础视觉评测
:label:subsec_pixel_fidelity

当我们希望评估世界模型的观测模型（Observation Model）或者解码器（Decoder）重建当前环境的能力时，最直观的第一步，是衡量生成的预测图像与真实的观测图像之间的相似度。我们首先从高中阶段最基础的几何距离切入。

### 均方误差与峰值信噪比 (PSNR)

假设我们要比较两张分辨率完全相同的图像，真实的图像为 $I$，模型生成的图像为 $K$。在数字图像处理中，每张图像可以被展开为一个长度为 $m \times n$ 的一维向量（此处暂且忽略通道数）。衡量两个向量之间最朴素的方法，便是计算它们在多维欧几里得空间中的欧式距离的平方。

我们定义这两张图像之间的均方误差（Mean Squared Error, MSE）为每个对应像素点差值的平方的平均数：

$$ \text{MSE} = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i, j) - K(i, j)]^2 $$
:eqlabel:eq_evaluation_mse

然而，均方误差存在一个明显的缺陷：它的绝对数值缺乏一致的物理尺度。为了更符合人类对“信号质量”的直观感受，学术界引入了物理学和信号处理领域中的信噪比概念，提出了峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）。

我们可以将真实的图像 $I$ 视为纯净的“信号”，而模型重建产生的偏差 $I - K$ 视为注入的“噪声”。PSNR 通过对数尺度来衡量最大可能信号能量与噪声能量之间的比值：

$$ \text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right) = 20 \cdot \log_{10} \left( \frac{\text{MAX}_I}{\sqrt{\text{MSE}}} \right) $$
:eqlabel:eq_evaluation_psnr

其中，$\text{MAX}_I$ 是图像像素可能的最大取值（例如对于 8-bit 的图像，$\text{MAX}_I = 255$）。PSNR 的单位为分贝（dB）。由于对数函数的单调递增特性，PSNR 的值越大，表示重建图像由于误差带来的“噪声”越小，模型的保真度也就越高。

### 结构相似性指数 (SSIM)

尽管 PSNR 计算简单且数学性质优良，但它仅仅孤立地比较了像素点与对应像素点之间的绝对数值差异，完全忽略了空间上相邻像素之间的强相关性。而人类的视觉系统对图像的结构信息（如边缘、轮廓）高度敏感，对全局亮度的绝对平移却相对迟钝。

为了弥补这一缺陷，结构相似性指数（Structural Similarity Index, SSIM）被提出。SSIM 巧妙地将图像块的比较解耦为三个独立的统计物理量：亮度（Luminance）、对比度（Contrast）和结构（Structure）。

给定两个局部图像窗口 $x$ 和 $y$：
1. **亮度比较**：我们用均值 $\mu_x = \frac{1}{N} \sum_{i=1}^N x_i$ 来近似局部区域的亮度。
2. **对比度比较**：我们用方差（或标准差） $\sigma_x = \left( \frac{1}{N-1} \sum_{i=1}^N (x_i - \mu_x)^2 \right)^{1/2}$ 来近似局部区域的对比度。
3. **结构比较**：在排除了亮度和对比度的影响后，我们利用协方差 $\sigma_{xy} = \frac{1}{N-1} \sum_{i=1}^N (x_i - \mu_x)(y_i - \mu_y)$ 来严格衡量二者结构形状的线性相关性。

通过引入极小的常数 $c_1, c_2$ 以防止分母为零，这三个维度被组合为最终的局部 SSIM 计算公式：

$$ \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)} $$
:eqlabel:eq_evaluation_ssim

SSIM 的取值范围被严格界定在 $[-1, 1]$ 之间。当两张局部图像完全一致时，$\mu_x = \mu_y$ 且 $\sigma_x = \sigma_y$，此时算式简化得出 $\text{SSIM} = 1$。通过滑动窗口计算整张图像的所有局部 SSIM 并求算术平均，我们能够获得一个更加贴合人类感知的重建质量量化指标。

## 10.2.3 从单样本到概率分布：特征空间的几何度量
:label:subsec_distribution_distance

虽然 PSNR 和 SSIM 能够衡量确定的、一一对齐的重建误差，但世界模型对未来的预测往往面临着环境的内禀随机性（Aleatoric Uncertainty）。在高度动态的环境中，模型生成的未来图像在绝对像素排列上可能与某次特定的真实观测结果在空间上并不完全重合，但在物理规律和宏观语义分布上却是完全合理的。

此时，我们不能再局限于“衡量单一样本之间的欧氏距离”，而是必须跨越到更高的概率抽象层级：“衡量两个整体概率分布之间的距离”。

### 降维视角的分布距离推导：1D 高斯模型

在开始复杂的矩阵推导前，让我们先考虑最简单的情况：假设真实的特征分布 $P$ 和模型生成的特征分布 $Q$ 都是一维实数空间上的正态（高斯）分布。即 $P \sim \mathcal{N}(\mu_1, \sigma_1^2)$，且 $Q \sim \mathcal{N}(\mu_2, \sigma_2^2)$。

我们希望用一个指标来量化这两个分布有多“不同”。严格的直觉告诉我们，这种概率分布的差异来源于两部分：
其一，是它们期望（即概率密度中心位置）的平移偏移，反映为均值之差的平方 $(\mu_1 - \mu_2)^2$；
其二，是它们散布程度（概率密度的胖瘦程度）的差异，反映为标准差之差的平方 $(\sigma_1 - \sigma_2)^2$。

如果我们严密地求解将分布 $P$ 转换为分布 $Q$ 的最优传输代价（即 Wasserstein-2 距离的平方），其解析解恰好完美契合了我们的基础直觉：

$$ W_2^2(P, Q) = (\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2 = (\mu_1 - \mu_2)^2 + (\sigma_1^2 + \sigma_2^2 - 2\sigma_1\sigma_2) $$
:eqlabel:eq_evaluation_w2_1d

> 想像你是一个城市规划者，你需要将一堆沙子（代表模型生成的特征分布）搬运去填补一个特定形状的坑（代表真实的特征分布）。不仅沙子的总量必须对等，你还需要考虑搬运沙子所耗费的距离与精力。如果我们将这种“最小化总体搬运成本”的几何直觉用数学语言严密地表达出来，并假设这些沙堆都服从多维正态分布，那么我们计算出的最优传输成本，正是弗雷歇距离（Fréchet Distance）。

### 弗雷歇初始距离 (FID) 与 时空扩展 (FVD)

现在，我们将一维空间的方差延展推演至高维特征空间。此时，标量均值 $\mu$ 成为了一个多维列向量，而标量方差 $\sigma^2$ 则升级为了协方差矩阵 $\Sigma$。对于多维高斯分布，标准差的乘积项 $\sigma_1\sigma_2$ 无法直接通过简单的矩阵乘法建立对应关系。严密的黎曼流形数学推导表明，这一交叉项被矩阵乘积平方根的迹（Trace，即矩阵主对角线元素之和）所替代。

由此，我们得到了深度学习生成模型领域中最为核心的评测公式，即两组多维高斯分布间的弗雷歇距离：

$$ d^2(P, Q) = ||\mu_p - \mu_q||_2^2 + \text{Tr}\left(\Sigma_p + \Sigma_q - 2(\Sigma_p \Sigma_q)^{1/2}\right) $$
:eqlabel:eq_evaluation_frechet

在实际工程操作中，直接在原始像素空间计算协方差矩阵不仅计算复杂度极其高昂，而且像素空间的分布既不符合高斯假设，也不符合人类的语义认知。因此，**[我们引入一个在 ImageNet 上预训练的 Inception-v3 网络]** 作为特征空间映射器。我们将大量真实图像与生成图像均前向传播至该网络的深层（通常截取分类全连接层之前的 2048 维全局平均池化层），以此特征向量集合来计算统计量 $\mu$ 和 $\Sigma$。这就是广泛采用的弗雷歇初始距离（FID）。

当我们将时间轴维度纳入考量，要求世界模型不仅生成单帧静态图像，而是生成一段具有连续时空因果的视频时，我们只需将 2D 的 Inception 网络替换为 3D 的 I3D 网络（在大规模视频动作识别数据集 Kinetics 上预训练）。利用 3D 网络提取时空强耦合的深层特征，再计算多维弗雷歇距离，便自然延伸出了用于衡量视频动态一致性的弗雷歇视频距离（FVD）。

## 10.2.4 世界模型的专属核心：动作条件下的动力学评测
:label:subsec_action_conditional_dynamics

前文所述的 FID 和 FVD 均属于“开环、无条件的视觉质量评测”。然而，世界模型的终极使命是成为智能体的“内部运行模拟器”。一个合格的世界模型必须深刻理解物理环境的因果性——即它必须能够准确预测环境状态如何根据智能体执行的具体动作指令发生转变。

假设智能体在时间步 $t$ 处于内部潜在隐状态 $s_t$，执行了动作 $a_t$。世界模型的动力学模型（Dynamics Model）进行一步状态转移 $s_{t+1} = f_\theta(s_t, a_t)$。这要求我们在评测时，必须将历史动作序列作为前置条件进行强干预验证。

### 动作条件下的长程累积误差 (Action-Conditioned Long-term Rollout Error)

世界模型的长期想象（Long-horizon Imagination）能力至关重要，因为强化学习中的策略规划往往需要看透未来几十乃至上百步。我们在评测中不仅要衡量单步前向预测的准确率，更要衡量连续多步自回归生成时的误差累积速度。

给定一个来自于真实环境收集的初始状态观测 $x_0$ （被编码为隐状态 $s_0$）和一段长度为 $T$ 的真实历史动作序列 $\mathcal{A} = \{a_0, a_1, \dots, a_{T-1}\}$。世界模型在此完全断开与外界真实观测的连接闭环，仅凭借初始状态 $s_0$ 和给定动作序列，在潜在向量空间中进行连续 $T$ 步的纯自回归展开（Rollout），生成一系列预测的未来状态 $\hat{s}_1, \dots, \hat{s}_T$，以及随之解码出的预测观测 $\hat{x}_1, \dots, \hat{x}_T$。

我们将这串纯靠想象预测出的序列与真实记录的观测序列 $x_1, \dots, x_T$ 在各个时间步上逐一进行对比。通常，随着时间步 $t$ 的增大，微小的单步预测偏差会在动力学模型中被指数级放大。记录并在直角坐标系中绘制从 $t=1$ 到 $t=T$ 的 MSE 或 PSNR 衰减曲线，是评测世界模型长程动力学稳定性的最核心标准。

### 奖励预测误差 (Reward Prediction Accuracy)

在强化学习的上下文中，除了高维的视觉观测之外，环境在每一个时间步还会返回一个标量奖励 $r_t$。世界模型的奖励预测器（Reward Predictor）需要根据转移后的隐状态预测该奖励：$\hat{r}_t = R_\phi(\hat{s}_t)$。

相比于高维图像像素的细微差别，奖励往往直接指示了任务的关键转折点（例如是否成功跨越了悬崖，是否触碰了终点旗帜）。因此，即使生成的预测图像在边缘细节上变得模糊，只要动力学模型能够精准地在特定的长程时间步预测出正确的奖励峰谷，它依然可以支撑强大的策略寻优算法。在系统级评测中，我们通常计算在给定动作序列展开下，真实标量奖励序列与预测奖励序列之间的均方误差，并尤为关注其在时间轴上的相位对齐程度。

## 10.2.5 具身智能与系统级榜单概览
:label:subsec_benchmarks

为了标准化上述所有评测维度的计算，学术界与工业界已逐步构建起一系列涵盖不同环境复杂度的世界模型专属基准榜单。

1. **Atari 100k 基准**：这是早期验证基于模型强化学习（Model-Based RL）极端数据效率的经典标尺。评测核心在于：严格限制模型与环境进行最多 100k 步（大约相当于人类实际游戏时间 2 小时）的交互，随后对比不同世界模型在潜空间中生成想象数据来训练出的策略最终得分。
2. **Crafter / MineDojo**：这是在《我的世界》（Minecraft）体素物理风格下构建的开放式、长视野生存环境。它们拥有极高的行动自由度和庞杂的技能树。评测不仅看模型能否存活，还会细化地考察模型在采集木材、合成工具、对抗怪物等数十种子任务上的零样本（Zero-shot）或小样本泛化表现。
3. **RoboDesk / MetaWorld**：这是专门面向三维机械臂与具身智能（Embodied AI）连续控制任务的测试基准。此类榜单不再仅仅衡量模拟器视觉维度的重建误差，而是将最终依靠模型规划出的连续动作轨迹能否在真实物理引擎中达成推、拉、抓取等任务的物理成功率（Success Rate）作为唯一真理的终极评价指标。

一个经得起考验的世界模型，其评测报告必须是一份完整的综合体检单：包含了微观重建的 PSNR/SSIM、宏观分布特征空间的 FID/FVD，以及基于该模型的动力学引擎所规划出的动作序列在实际交互中的任务成功率。

## 10.2.6 核心评测指标的代码实现
:label:subsec_eval_code

接下来，我们以深度学习框架的标准代码规范，严谨地实现图像级别的两大基础评测指标：源自 MSE 的 PSNR，以及综合了亮度与方差的 SSIM。同时，我们也将展示如何利用预训练模型提取特征向量以备计算 FID 统计量。

```{.python .input}
#@tab pytorch
import torch
import torch.nn.functional as F
import math

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    计算给定两批次图像的峰值信噪比(PSNR)。
    假设输入张量的取值范围已被归一化至 [0, max_val]。
    """
    # [**计算两个图像张量之间所有对应元素的均方误差 (MSE)**]
    mse = torch.mean((img1 - img2) ** 2)
    
    # 极值情况处理：如果两张图像完全一致，MSE为0，PSNR理论上趋于无穷大
    if mse == 0:
        return float('inf')
        
    # [**严格依据对数能量衰减公式计算 PSNR 值**]
    psnr = 20 * math.log10(max_val / math.sqrt(mse.item()))
    return psnr

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> torch.Tensor:
    """
    计算基于滑动窗口的结构相似性指数 (SSIM)。
    """
    channels = img1.size(1)
    
    # [**利用平均池化计算局部滑动窗口的均值 μ_x 和 μ_y (近似代表局部亮度)**]
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # [**计算局部方差 σ_x^2, σ_y^2 以及协方差 σ_xy (代表对比度与结构相关性)**]
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    # 设定极小常数以确保除法运算的数值稳定性
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # [**严格依照原始公式，在分子和分母中组合亮度、对比度与结构三项因子**]
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
               
    # 返回整幅图像平面的平均 SSIM 标量评分
    return ssim_map.mean()
```

对于 FID 的计算，其核心工程难点在于跨越像素空间，在深层特征流形上建立概率分布模型。以下代码展示了如何获取这些高维特征并严谨计算多维高斯分布统计量：

```{.python .input}
#@tab pytorch
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights

def get_inception_features(images: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    """
    使用在 ImageNet 上预训练的 Inception-v3 模型提取高阶特征。
    """
    # [**加载预训练权重，并务必设置为评估模式，防止 Batch Norm 或 Dropout 引入随机性扰动**]
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    # 移除最后的分类映射全连接层，以获取更底层的连续语义特征
    model.fc = torch.nn.Identity()
    model.eval()
    
    features_list = []
    with torch.no_grad():
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i+batch_size]
            # [**通过双线性插值强制将图像缩放至299x299，以吻合Inception-v3初始感受野和步长的空间设计假设**]
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            features = model(batch)
            features_list.append(features)
            
    return torch.cat(features_list, dim=0)

def compute_statistics(features: torch.Tensor):
    """
    计算高维特征分布的均值向量与协方差矩阵。
    """
    # 将计算图脱离并将张量转移至 CPU，转化为 numpy 数组以便调用高阶代数库进行大规模矩阵运算
    features_np = features.cpu().numpy()
    
    # [**沿样本批量维度展开，计算各维特征的算术均值向量 μ**]
    mu = np.mean(features_np, axis=0)
    
    # [**严格计算多维特征之间的协方差矩阵 Σ，行维度代表变量(特征)，列维度代表具体的观察值(样本)**]
    sigma = np.cov(features_np, rowvar=False)
    
    return mu, sigma
```

## 10.2.7 小结

在本节中，我们沿着从局部确定性到全局概率性、从单帧切片图像到连续时空轨迹、从纯视觉信号解码到核心动作因果干预的学术脉络，立体地拆解了世界模型的系统级评测体系。我们利用高中代数的基础直觉严密推导了 PSNR 和 SSIM 的计算原理，引入了最优传输（Optimal Transport）的几何思想透彻解析了 FID 背后衡量分布距离的数学逻辑，并深入探讨了长程动作累积误差和奖励预测这类世界模型独有的检验机制。评测标准绝非仅仅是一把冷冰冰的标尺，它更是指引我们向着更强大的具身智能系统架构迭代前行的灯塔。

## 练习

1. 在计算图像重建的 PSNR 时，如果我们将预测图像中所有像素的亮度统一增加一个常数正向偏移 $c$，MSE 的物理值是否会发生改变？在此绝对偏移情况下，SSIM 公式计算出的局部均值亮度分量 $\mu_x, \mu_y$ 将受到何种影响，其对应的结构分量协方差是否会因绝对亮度的平移而发生形变？请尝试从数学公式层面推导。
   - **提示**：回忆高中数学阶段对于方差和平移变换的不变性定理，常数平移是否会改变差值或局部点对之间的线性相关性？
2. 请思考，为什么在计算 FID 分数时，业界标准要求必须使用预训练的 Inception 网络深层特征进行映射，而不是直接在原始的高维 RGB 像素张量空间中假设多维正态分布并计算其均值向量和协方差矩阵的弗雷歇距离？
   - **提示**：回顾我们在本节特征空间维度的深层探讨。原始像素级的纯数值欧氏距离是否能够跨越语义鸿沟（Semantic Gap）捕捉到“这是一只猫”与“那是一棵树”的高阶认知区别？像素本身的亮度分布是否真的天然满足纯粹的高斯钟形曲线假设？
3. 当测试一个采用纯自回归生成架构（Autoregressive Generation）构建的动作条件世界模型时，如果在展开（Rollout）的第 $t=5$ 步发生了细微的角度预测数值误差，请分析这一误差在传递至 $t=20$ 步时，会对依赖该模型展开数据进行强化学习策略规划（Policy Planning）的优化器产生什么量级和性质的影响？
   - **提示**：结合数学上的指数累积放大效应以及马尔可夫决策过程（MDP）的序列依赖性质进行严密的逻辑推演。

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
