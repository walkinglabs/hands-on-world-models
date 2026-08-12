"""JEPA 路线使用的最小视频特征预测组件。"""

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


def patchify_video(video, patch_size=4):
    """把 [B,T,C,H,W] 变成 [B,T,N,patch_dim]。"""
    batch, time, channels, height, width = video.shape
    if height % patch_size or width % patch_size:
        raise ValueError("图片尺寸必须能被 patch_size 整除")
    patches = video.unfold(3, patch_size, patch_size).unfold(
        4, patch_size, patch_size
    )
    patches = patches.permute(0, 1, 3, 4, 2, 5, 6).contiguous()
    return patches.reshape(batch, time, -1, channels * patch_size * patch_size)


class PatchEncoder(nn.Module):
    def __init__(self, patch_dim=48, feature_size=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(patch_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, feature_size),
        )

    def forward(self, patches):
        return self.network(patches)


class TinyVideoJEPA(nn.Module):
    """用历史 patch 特征预测下一帧 target encoder 特征。"""

    def __init__(self, feature_size=32, action_size=5, patch_size=4, num_patches=16):
        super().__init__()
        self.patch_size = patch_size
        self.online_encoder = PatchEncoder(3 * patch_size * patch_size, feature_size)
        self.target_encoder = deepcopy(self.online_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.action_embedding = nn.Embedding(action_size, 8)
        self.position = nn.Parameter(torch.randn(num_patches, 8) * 0.02)
        self.predictor = nn.Sequential(
            nn.Linear(feature_size + 8 + 8, 64),
            nn.GELU(),
            nn.Linear(64, feature_size),
        )

    def forward(self, history, actions=None):
        """history 含最后一个 target frame；action 作用于倒数第二帧到最后帧。"""
        patches = patchify_video(history, self.patch_size)
        context_patches = patches[:, :-1]
        target_patches = patches[:, -1]
        context_features = self.online_encoder(context_patches)
        context = context_features.mean(dim=(1, 2))
        if actions is None:
            action_feature = torch.zeros(
                history.shape[0], 8, device=history.device, dtype=history.dtype
            )
        else:
            action_feature = self.action_embedding(actions.long())
        repeated_context = context[:, None].expand(-1, target_patches.shape[1], -1)
        repeated_action = action_feature[:, None].expand(-1, target_patches.shape[1], -1)
        predictor_input = torch.cat(
            (
                repeated_context,
                repeated_action,
                self.position[None, : target_patches.shape[1]].expand(history.shape[0], -1, -1),
            ),
            dim=-1,
        )
        prediction = self.predictor(predictor_input)
        with torch.no_grad():
            target = self.target_encoder(target_patches)
        return prediction, target, context_features

    def loss(self, history, actions=None, mask=None):
        prediction, target, features = self(history, actions)
        error = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
        if mask is not None:
            error = (error * mask.float()).sum() / mask.float().sum().clamp_min(1)
        else:
            error = error.mean()
        return error, prediction, target, features

    @torch.no_grad()
    def update_target(self, momentum=0.99):
        for online, target in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(momentum).add_(online.data, alpha=1 - momentum)


def feature_spread(features):
    """特征跨样本/时空的平均标准差；接近零可能表示坍缩。"""
    flat = features.reshape(-1, features.shape[-1])
    return flat.std(dim=0).mean()


def jepa_batch_from_episodes(episodes, history_length=3):
    """产生历史 clip、最后动作和目标方块中心。"""
    import numpy as np

    clips = []
    actions = []
    positions = []
    for episode in episodes:
        for start in range(len(episode.observations) - history_length + 1):
            stop = start + history_length
            clip = episode.observations[start:stop]
            clips.append(clip)
            actions.append(episode.actions[stop - 2])
            red = clip[-1, :, :, 0]
            rows, cols = np.where(red == red.max())
            positions.append((rows.mean() / 15.0, cols.mean() / 15.0))
    video = torch.from_numpy(np.stack(clips)).permute(0, 1, 4, 2, 3).float() / 255.0
    action = torch.tensor(actions, dtype=torch.long)
    position = torch.tensor(positions, dtype=torch.float32)
    return video, action, position


def fit_linear_probe(features, targets, ridge=1e-3):
    """闭式解线性探针，避免再引入一段训练循环。"""
    weights = fit_linear_probe_weights(features, targets, ridge)
    return apply_linear_probe(features, weights)


def fit_linear_probe_weights(features, targets, ridge=1e-3):
    """只在训练 split 拟合权重，便于在 held-out split 上评价。"""
    ones = torch.ones(features.shape[0], 1, device=features.device)
    design = torch.cat((features, ones), dim=1)
    identity = torch.eye(design.shape[1], device=features.device) * ridge
    return torch.linalg.solve(design.T @ design + identity, design.T @ targets)


def apply_linear_probe(features, weights):
    ones = torch.ones(features.shape[0], 1, device=features.device)
    design = torch.cat((features, ones), dim=1)
    return design @ weights
