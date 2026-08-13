"""互动视频路线使用的最小 PyTorch 组件。"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def foreground_weighted_mse(reconstruction, target, foreground_weight=12.0):
    """避免小物体被大面积黑色背景淹没的重建损失。"""
    foreground = target.amax(dim=1, keepdim=True) > 0.2
    weights = 1.0 + foreground.float() * foreground_weight
    weights = weights / weights.mean()
    return ((reconstruction - target).square() * weights).mean()


def red_centers(images):
    """从 [B,3,H,W] 图像估计红色物体中心；没有红色时返回 NaN。"""
    red_score = (images[:, 0] - torch.maximum(images[:, 1], images[:, 2])).clamp_min(0)
    batch, height, width = red_score.shape
    rows = torch.arange(height, device=images.device, dtype=images.dtype)
    cols = torch.arange(width, device=images.device, dtype=images.dtype)
    mass = red_score.sum(dim=(1, 2))
    safe_mass = mass.clamp_min(1e-6)
    row = (red_score * rows[None, :, None]).sum(dim=(1, 2)) / safe_mass
    col = (red_score * cols[None, None, :]).sum(dim=(1, 2)) / safe_mass
    centers = torch.stack((row, col), dim=-1)
    missing = mass < 1e-4
    centers[missing] = torch.nan
    return centers


def motion_direction_accuracy(current, predicted, target, minimum_motion=0.25):
    """比较预测与真实的主位移方向，忽略真实 stay 样本。"""
    current_center = red_centers(current)
    predicted_delta = red_centers(predicted) - current_center
    target_delta = red_centers(target) - current_center
    moving = target_delta.abs().amax(dim=-1) >= minimum_motion
    finite = torch.isfinite(predicted_delta).all(dim=-1)
    keep = moving & finite
    if not keep.any():
        return torch.tensor(float("nan"), device=current.device)
    predicted_axis = predicted_delta[keep].abs().argmax(dim=-1)
    target_axis = target_delta[keep].abs().argmax(dim=-1)
    predicted_sign = torch.sign(
        predicted_delta[keep].gather(1, predicted_axis[:, None]).squeeze(1)
    )
    target_sign = torch.sign(
        target_delta[keep].gather(1, target_axis[:, None]).squeeze(1)
    )
    return ((predicted_axis == target_axis) & (predicted_sign == target_sign)).float().mean()


class VectorQuantizer(nn.Module):
    """把连续特征换成码本编号，并用 STE 传回梯度。"""

    def __init__(self, codebook_size=32, embedding_size=16, commitment=0.25):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, embedding_size)
        self.commitment = commitment
        nn.init.uniform_(
            self.codebook.weight,
            -1.0 / codebook_size,
            1.0 / codebook_size,
        )

    def forward(self, features):
        # features: [B,D,H,W]
        channels_last = features.permute(0, 2, 3, 1).contiguous()
        flat = channels_last.reshape(-1, channels_last.shape[-1])
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.weight.T
            + self.codebook.weight.square().sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).reshape_as(channels_last)
        codebook_loss = F.mse_loss(quantized, channels_last.detach())
        commitment_loss = F.mse_loss(channels_last, quantized.detach())
        loss = codebook_loss + self.commitment * commitment_loss
        # 前向使用离散码本，反向把梯度近似送回 encoder。
        straight_through = channels_last + (quantized - channels_last).detach()
        return (
            straight_through.permute(0, 3, 1, 2).contiguous(),
            indices.reshape(features.shape[0], features.shape[2], features.shape[3]),
            loss,
        )


class TinyVQVAE(nn.Module):
    def __init__(self, codebook_size=32, embedding_size=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, embedding_size, 4, stride=2, padding=1),
        )
        self.quantizer = VectorQuantizer(codebook_size, embedding_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_size, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    @property
    def codebook_size(self):
        return self.quantizer.codebook.num_embeddings

    @torch.no_grad()
    def initialize_codebook(self, images):
        """从真实 encoder 特征抽取初始码字，降低教学小数据上的码本坍缩。"""
        features = self.encoder(images).permute(0, 2, 3, 1).reshape(-1, self.quantizer.codebook.embedding_dim)
        if len(features) < self.codebook_size:
            repeats = math.ceil(self.codebook_size / len(features))
            features = features.repeat(repeats, 1)
        # 远点采样：每次选一个离已有码字最远的 encoder 特征。
        chosen = [features[0]]
        distance = (features - chosen[0]).square().sum(dim=1)
        for _ in range(1, self.codebook_size):
            index = distance.argmax()
            chosen.append(features[index])
            new_distance = (features - features[index]).square().sum(dim=1)
            distance = torch.minimum(distance, new_distance)
        self.quantizer.codebook.weight.copy_(torch.stack(chosen))

    def continuous_loss(self, images):
        """VQ 前的普通 autoencoder 预热；只用于小数据教学。"""
        reconstruction = self.decoder(self.encoder(images))
        return foreground_weighted_mse(reconstruction, images), reconstruction

    def forward(self, images):
        features = self.encoder(images)
        quantized, indices, quantization_loss = self.quantizer(features)
        reconstruction = self.decoder(quantized)
        reconstruction_loss = foreground_weighted_mse(reconstruction, images)
        return {
            "reconstruction": reconstruction,
            "tokens": indices,
            "loss": reconstruction_loss + quantization_loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "quantization_loss": quantization_loss.detach(),
        }

    def encode_tokens(self, images):
        features = self.encoder(images)
        _, indices, _ = self.quantizer(features)
        return indices

    def decode_tokens(self, tokens):
        quantized = self.quantizer.codebook(tokens.long()).permute(0, 3, 1, 2)
        return self.decoder(quantized)


class ActionTokenTransformer(nn.Module):
    """根据当前帧 token 和动作预测下一帧 token。"""

    def __init__(
        self,
        codebook_size=32,
        action_size=5,
        tokens_per_frame=16,
        model_size=48,
        num_layers=1,
        action_injection="additive",
    ):
        super().__init__()
        if action_injection not in {"none", "additive", "film"}:
            raise ValueError("action_injection must be none, additive, or film")
        self.tokens_per_frame = tokens_per_frame
        self.action_injection = action_injection
        self.token_embedding = nn.Embedding(codebook_size, model_size)
        self.action_embedding = nn.Embedding(action_size, model_size)
        self.action_film = nn.Linear(model_size, 2 * model_size)
        self.position = nn.Parameter(torch.randn(tokens_per_frame, model_size) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_size,
            nhead=4,
            dim_feedforward=2 * model_size,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(model_size, codebook_size)

    def forward(self, current_tokens, actions):
        # 一个 frame 内的 token 已经同时可见；时间因果性由训练对齐保证。
        embedded = self.token_embedding(current_tokens.long())
        embedded = embedded + self.position[None]
        action = self.action_embedding(actions.long())
        if self.action_injection == "additive":
            embedded = embedded + action[:, None]
        elif self.action_injection == "film":
            scale, shift = self.action_film(action).chunk(2, dim=-1)
            embedded = embedded * (1 + scale[:, None]) + shift[:, None]
        hidden = self.transformer(embedded)
        return self.output(hidden)

    def loss(self, current_tokens, actions, next_tokens):
        logits = self(current_tokens, actions)
        return F.cross_entropy(logits.flatten(0, 1), next_tokens.flatten())


class TinyConditionalDenoiser(nn.Module):
    """一次去噪的动作条件基线，不冒充完整视频 Diffusion。"""

    def __init__(self, action_size=5, hidden_size=24):
        super().__init__()
        self.action_embedding = nn.Embedding(action_size, 8)
        self.network = nn.Sequential(
            nn.Conv2d(3 + 3 + 8 + 1, hidden_size, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, 3, 3, padding=1),
        )

    def forward(self, noisy_next, current, actions, noise_level):
        batch, _, height, width = noisy_next.shape
        action = self.action_embedding(actions.long())[:, :, None, None]
        action = action.expand(batch, -1, height, width)
        level = noise_level.reshape(batch, 1, 1, 1).expand(batch, 1, height, width)
        inputs = torch.cat((noisy_next, current, action, level), dim=1)
        return self.network(inputs)


def add_independent_frame_noise(video, noise_levels, noise=None):
    """按帧加入不同强度的噪声，构造 Diffusion Forcing 教学样本。

    video: [B,T,C,H,W]
    noise_levels: [B,T]，0 表示干净，1 表示完全由噪声主导。
    """
    if video.ndim != 5 or noise_levels.shape != video.shape[:2]:
        raise ValueError("expected video [B,T,C,H,W] and noise_levels [B,T]")
    if noise is None:
        noise = torch.randn_like(video)
    level = noise_levels[..., None, None, None].to(video)
    noisy = (1 - level) * video + level * noise
    return noisy, noise


def diffusion_prediction_target(clean, noise, noise_levels, target="epsilon"):
    """返回 tiny diffusion 实验的 x / epsilon / v 三种监督目标。"""
    level = noise_levels
    while level.ndim < clean.ndim:
        level = level.unsqueeze(-1)
    level = level.to(clean)
    if target == "x":
        return clean
    if target == "epsilon":
        return noise
    if target == "v":
        return (1 - level) * noise - level * clean
    raise ValueError("target must be x, epsilon, or v")


@torch.no_grad()
def rollout_token_model(model, tokenizer, start_tokens, actions, token_shape):
    """让 token 模型连续读取自己的输出，并解码整段 rollout。"""
    tokens = [start_tokens]
    for action in actions:
        if not torch.is_tensor(action):
            action = torch.tensor([action], device=start_tokens.device)
        action = action.reshape(-1).to(start_tokens.device)
        tokens.append(model(tokens[-1], action).argmax(dim=-1))
    packed = torch.cat(tokens, dim=0).reshape(-1, *token_shape)
    return tokenizer.decode_tokens(packed)


def psnr(prediction, target, data_range=1.0):
    """只衡量像素接近程度；不能替代动作一致性或长时评价。"""
    mse = F.mse_loss(prediction, target).clamp_min(1e-12)
    return 10 * torch.log10(torch.as_tensor(data_range**2, device=mse.device) / mse)


def video_batch_from_episodes(episodes):
    """把等长 episode 整理成 frame transition batch。"""
    observations = []
    next_observations = []
    actions = []
    for episode in episodes:
        observations.append(episode.observations[:-1])
        next_observations.append(episode.observations[1:])
        actions.append(episode.actions)
    current = torch.from_numpy(__import__("numpy").concatenate(observations))
    following = torch.from_numpy(__import__("numpy").concatenate(next_observations))
    action = torch.from_numpy(__import__("numpy").concatenate(actions))
    current = current.permute(0, 3, 1, 2).float() / 255.0
    following = following.permute(0, 3, 1, 2).float() / 255.0
    return current, action.long(), following


def token_accuracy(logits, targets):
    return (logits.argmax(dim=-1) == targets).float().mean()
