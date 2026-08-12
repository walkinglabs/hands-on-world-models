"""F1/F2 使用的最小组件实现。"""

import numpy as np


def rgb_to_gray(image):
    image = np.asarray(image, dtype=np.float32)
    return image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def conv2d_valid(image, kernel):
    """从零实现单通道、无 padding 的二维卷积。"""
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    out_height = image.shape[0] - kernel.shape[0] + 1
    out_width = image.shape[1] - kernel.shape[1] + 1
    if out_height < 1 or out_width < 1:
        raise ValueError("kernel 不能比 image 更大")

    output = np.empty((out_height, out_width), dtype=np.float32)
    for row in range(out_height):
        for col in range(out_width):
            window = image[
                row : row + kernel.shape[0],
                col : col + kernel.shape[1],
            ]
            output[row, col] = np.sum(window * kernel)
    return output


def patchify(image, patch_size):
    """把 [H,W,C] 图片切成 ViT 风格 patch token。"""
    image = np.asarray(image)
    height, width, channels = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError("图片长宽必须能被 patch_size 整除")
    patches = image.reshape(
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channels,
    )
    patches = patches.transpose(0, 2, 1, 3, 4)
    return patches.reshape(-1, patch_size * patch_size * channels)


def unpatchify(tokens, image_shape, patch_size):
    height, width, channels = image_shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = np.asarray(tokens).reshape(
        grid_h,
        grid_w,
        patch_size,
        patch_size,
        channels,
    )
    return patches.transpose(0, 2, 1, 3, 4).reshape(image_shape)


def position_encoding(num_tokens):
    """为每个 patch 提供一个容易观察的二维位置。"""
    side = int(round(num_tokens**0.5))
    if side * side != num_tokens:
        raise ValueError("教学版本要求 patch 构成正方形网格")
    rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return np.stack((rows.ravel(), cols.ravel()), axis=1)


def center_of_red(image):
    red = np.asarray(image)[..., 0]
    points = np.argwhere(red > 200)
    if not len(points):
        return np.array([np.nan, np.nan])
    return points.mean(axis=0)


def remember_velocity(frames):
    """用相邻中心差构造最小的时序记忆。"""
    centers = np.stack([center_of_red(frame) for frame in frames])
    velocities = np.zeros_like(centers)
    velocities[1:] = centers[1:] - centers[:-1]
    return np.concatenate((centers, velocities), axis=1)


def block_average_encode(image, block_size=4):
    image = np.asarray(image, dtype=np.float32)
    height, width, channels = image.shape
    if height % block_size or width % block_size:
        raise ValueError("图片长宽必须能被 block_size 整除")
    blocks = image.reshape(
        height // block_size,
        block_size,
        width // block_size,
        block_size,
        channels,
    )
    return blocks.mean(axis=(1, 3))


def block_average_decode(latent, block_size=4):
    latent = np.asarray(latent)
    return latent.repeat(block_size, axis=0).repeat(block_size, axis=1)


def reconstruction_mse(image, reconstruction):
    image = np.asarray(image, dtype=np.float32)
    reconstruction = np.asarray(reconstruction, dtype=np.float32)
    return float(np.mean((image - reconstruction) ** 2))


def depth_to_points(depth, fx, fy, cx, cy):
    """把针孔相机深度图反投影到相机坐标系。"""
    depth = np.asarray(depth, dtype=np.float32)
    rows, cols = np.indices(depth.shape)
    z = depth
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)


def make_camera_transform(tx=0.0, ty=0.0, tz=0.0, yaw=0.0):
    """构造相机坐标到世界坐标的齐次变换。"""
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cosine, 0, sine, tx],
            [0, 1, 0, ty],
            [-sine, 0, cosine, tz],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def transform_points(points, transform):
    """用 4×4 齐次矩阵变换一组三维点。"""
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32)
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float32)),
        axis=1,
    )
    return (transform @ homogeneous.T).T[:, :3]


def points_to_occupancy(points, x_range, z_range, resolution):
    """把三维点落到俯视 Occupancy 网格。"""
    points = np.asarray(points)
    width = int(np.ceil((x_range[1] - x_range[0]) / resolution))
    depth = int(np.ceil((z_range[1] - z_range[0]) / resolution))
    grid = np.zeros((depth, width), dtype=np.uint8)
    x_index = ((points[:, 0] - x_range[0]) / resolution).astype(int)
    z_index = ((points[:, 2] - z_range[0]) / resolution).astype(int)
    valid = (
        (x_index >= 0)
        & (x_index < width)
        & (z_index >= 0)
        & (z_index < depth)
    )
    grid[z_index[valid], x_index[valid]] = 1
    return grid


def symlog(value):
    value = np.asarray(value, dtype=np.float32)
    return np.sign(value) * np.log1p(np.abs(value))


def symexp(value):
    value = np.asarray(value, dtype=np.float32)
    return np.sign(value) * np.expm1(np.abs(value))


def clip_by_norm(gradient, max_norm):
    gradient = np.asarray(gradient, dtype=np.float32)
    norm = np.linalg.norm(gradient)
    if norm <= max_norm or norm == 0:
        return gradient
    return gradient * (max_norm / norm)


def cem_plan_1d(start, target, horizon=5, population=400, elite=40, rounds=5, seed=0):
    """在 x_{t+1}=x_t+a_t 中用 CEM 搜索连续动作。"""
    rng = np.random.default_rng(seed)
    mean = np.zeros(horizon, dtype=np.float32)
    std = np.ones(horizon, dtype=np.float32)
    history = []

    for _ in range(rounds):
        actions = rng.normal(mean, std, size=(population, horizon))
        actions = np.clip(actions, -1.0, 1.0)
        final_positions = start + actions.sum(axis=1)
        scores = -(final_positions - target) ** 2 - 0.01 * (actions**2).sum(axis=1)
        elite_indices = np.argsort(scores)[-elite:]
        elite_actions = actions[elite_indices]
        mean = elite_actions.mean(axis=0)
        std = elite_actions.std(axis=0) + 1e-4
        history.append(float(scores[elite_indices[-1]]))

    return mean, history
