#!/usr/bin/env python3
"""Foundations 可视化：用 F1/F2/F3 的真实设定生成讲义配图。"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hwm.data import ACTION_NAMES, MovingSquareWorld
from hwm.foundations import (
    block_average_decode,
    block_average_encode,
    cem_plan_1d,
    conv2d_valid,
    depth_to_points,
    patchify,
    points_to_occupancy,
    reconstruction_mse,
    remember_velocity,
    rgb_to_gray,
    symlog,
)
from hwm.gridworld import EmpiricalDynamics, LineWorld, mpc_episode


def try_load_font(size=16):
    for path in (
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def upscale(array, scale):
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    return np.repeat(np.repeat(array, scale, axis=0), scale, axis=1)


def to_rgb(image):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        lo, hi = float(image.min()), float(image.max())
        if hi - lo < 1e-6:
            image = np.zeros_like(image, dtype=np.uint8)
        else:
            image = ((image - lo) / (hi - lo) * 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image


def paste_image(canvas, array, xy, scale=8, border="black"):
    rgb = to_rgb(array)
    pil = Image.fromarray(upscale(rgb, scale), mode="RGB")
    canvas.paste(pil, xy)
    if border:
        draw = ImageDraw.Draw(canvas)
        x, y = xy
        draw.rectangle([x - 1, y - 1, x + pil.width, y + pil.height], outline=border)


def draw_grid(draw, origin, cells, cell, color=(210, 210, 210)):
    x0, y0 = origin
    rows, cols = cells
    for r in range(rows + 1):
        y = y0 + r * cell
        draw.line([(x0, y), (x0 + cols * cell, y)], fill=color, width=1)
    for c in range(cols + 1):
        x = x0 + c * cell
        draw.line([(x, y0), (x, y0 + rows * cell)], fill=color, width=1)


def visualize_worlds(output_path):
    world = MovingSquareWorld()
    episode, positions = world.generate([2, 2, 4, 4], start=(2, 2))
    frames = [0, 2, 4]
    scale = 8
    frame_px = 16 * scale
    pad = 28
    title_h = 46
    caption_h = 70
    left_w = pad + 3 * (frame_px + 18) + pad
    right_w = 420
    img_w = left_w + right_w
    img_h = title_h + frame_px + caption_h + pad

    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(22)
    font = try_load_font(14)
    small = try_load_font(12)

    draw.text((pad, 10), "两个要学会的世界：像素里的方块，和一条会打滑的线", fill="black", font=title_font)

    labels = ["第 0 帧 (2,2)", "第 2 帧 (2,4)", "第 4 帧 (4,4)"]
    for i, idx in enumerate(frames):
        x = pad + i * (frame_px + 18)
        y = title_h
        paste_image(canvas, episode.observations[idx], (x, y), scale=scale)
        draw.text((x, y + frame_px + 8), labels[i], fill="black", font=font)

    names = [ACTION_NAMES[int(a)] for a in episode.actions]
    draw.text(
        (pad, title_h + frame_px + 30),
        "动作: " + " → ".join(names) + "    红=自己  绿=目标",
        fill=(80, 80, 80),
        font=small,
    )

    # LineWorld
    line_x = left_w + 10
    line_y = title_h + 18
    cell_w, cell_h = 46, 46
    cells = ["×", "·", "·", "A", "·", "·", "G"]
    colors = {
        "×": ((220, 70, 70), "white"),
        "G": ((46, 140, 72), "white"),
        "A": ((40, 90, 200), "white"),
        "·": ((245, 245, 245), (80, 80, 80)),
    }
    draw.text((line_x, title_h - 4), "LineWorld：7 格，20% 打滑", fill="black", font=font)
    for i, symbol in enumerate(cells):
        x1 = line_x + i * cell_w
        y1 = line_y + 18
        fill, ink = colors[symbol]
        draw.rounded_rectangle([x1, y1, x1 + cell_w - 6, y1 + cell_h], radius=6, fill=fill, outline=(40, 40, 40))
        draw.text((x1 + 14, y1 + 12), symbol, fill=ink, font=try_load_font(18))
        draw.text((x1 + 12, y1 + cell_h + 6), str(i), fill=(120, 120, 120), font=small)

    draw.text((line_x, line_y + cell_h + 44), "0 是陷阱，6 是终点，从 3 出发", fill=(80, 80, 80), font=small)
    draw.text((line_x, line_y + cell_h + 64), "left / right，停在原地也算一次转移", fill=(80, 80, 80), font=small)

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_convolution(output_path):
    world = MovingSquareWorld()
    episode, _ = world.generate([2, 2, 4, 4], start=(2, 2))
    image = episode.observations[0]
    gray = rgb_to_gray(image)
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    edge = conv2d_valid(gray, kernel)
    peak = np.unravel_index(np.argmax(np.abs(edge)), edge.shape)

    scale = 8
    pad = 28
    title_h = 44
    cell = 16 * scale
    img_w = pad * 4 + cell * 3
    img_h = title_h + cell + 78
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(14)
    small = try_load_font(12)

    draw.text((pad, 8), "同一张 16×16 图：原图、灰度、竖直边缘响应", fill="black", font=title_font)

    paste_image(canvas, image, (pad, title_h), scale=scale)
    draw.text((pad, title_h + cell + 8), "原图 (16,16,3)", fill="black", font=font)
    draw.text((pad, title_h + cell + 28), "红方块在 (2,2)", fill=(90, 90, 90), font=small)

    gray_u8 = (np.clip(gray / max(gray.max(), 1e-6), 0, 1) * 255).astype(np.uint8)
    paste_image(canvas, gray_u8, (pad * 2 + cell, title_h), scale=scale)
    draw.text((pad * 2 + cell, title_h + cell + 8), "灰度 (16,16)", fill="black", font=font)
    draw.text((pad * 2 + cell, title_h + cell + 28), "红→灰约 76.2", fill=(90, 90, 90), font=small)

    # valid conv is 14x14; pad visually to 16x16 so grids line up
    vis = np.zeros((16, 16), dtype=np.float32)
    vis[1:15, 1:15] = edge
    lo, hi = float(vis.min()), float(vis.max())
    norm = ((vis - lo) / (hi - lo) * 255).astype(np.uint8)
    heat = np.stack([norm, np.full_like(norm, 40), 255 - norm], axis=-1)
    x2 = pad * 3 + cell * 2
    paste_image(canvas, heat, (x2, title_h), scale=scale)
    # mark peak in valid coordinates, offset by 1 because of visual pad
    pr, pc = int(peak[0]) + 1, int(peak[1]) + 1
    cx = x2 + pc * scale + scale // 2
    cy = title_h + pr * scale + scale // 2
    draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(255, 220, 0), width=2)
    draw.text((x2, title_h + cell + 8), "卷积输出 (14,14)", fill="black", font=font)
    draw.text(
        (x2, title_h + cell + 28),
        f"最强响应 ({int(peak[0])},{int(peak[1])}) = {float(edge[peak]):.1f}",
        fill=(90, 90, 90),
        font=small,
    )

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_patchify(output_path):
    world = MovingSquareWorld()
    episode, _ = world.generate([2, 2, 4, 4], start=(2, 2))
    image = episode.observations[0]
    tokens = patchify(image, 4)

    scale = 8
    pad = 28
    title_h = 44
    cell = 16 * scale
    img_w = pad * 4 + cell * 2 + 220
    img_h = title_h + cell + 70
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(14)
    small = try_load_font(12)

    draw.text((pad, 8), "切成 patch 只改形状，不减少数字", fill="black", font=title_font)
    paste_image(canvas, image, (pad, title_h), scale=scale)
    draw.text((pad, title_h + cell + 8), "原图 16×16×3 = 768", fill="black", font=font)

    x1 = pad * 2 + cell
    paste_image(canvas, image, (x1, title_h), scale=scale)
    for i in range(5):
        p = i * 4 * scale
        draw.line([(x1 + p, title_h), (x1 + p, title_h + cell)], fill=(250, 200, 0), width=2)
        draw.line([(x1, title_h + p), (x1 + cell, title_h + p)], fill=(250, 200, 0), width=2)
    draw.text((x1, title_h + cell + 8), "4×4 网格，16 个 patch", fill="black", font=font)

    x2 = pad * 3 + cell * 2
    draw.text((x2, title_h + 8), f"tokens {tuple(tokens.shape)}", fill="black", font=font)
    draw.text((x2, title_h + 34), "16 × 48 = 768", fill=(40, 90, 200), font=font)
    draw.text((x2, title_h + 60), "压缩比 1.0×", fill=(40, 90, 200), font=font)
    draw.text((x2, title_h + 90), "unpatchify 后与原图", fill=(80, 80, 80), font=small)
    draw.text((x2, title_h + 108), "逐像素相同", fill=(80, 80, 80), font=small)
    draw.text((x2, title_h + 136), "位置编码另给 (row,col)", fill=(80, 80, 80), font=small)

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_history(output_path):
    world = MovingSquareWorld()
    from_left, left_pos = world.generate([2, 2], start=(5, 3), episode_id="left")
    from_right, right_pos = world.generate([1, 1], start=(5, 7), episode_id="right")
    left_state = remember_velocity(from_left.observations)
    right_state = remember_velocity(from_right.observations)

    scale = 7
    frame_px = 16 * scale
    pad = 24
    title_h = 42
    cols = 3
    img_w = pad + 2 * (cols * (frame_px + 12) + 36) + pad
    img_h = title_h + frame_px + 86
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)

    draw.text((pad, 8), "末帧相同，速度相反：单看最后一张图分不清从哪来", fill="black", font=title_font)

    def draw_seq(episodes, positions, states, x0, title, color):
        draw.text((x0, title_h - 2), title, fill=color, font=font)
        for i, frame in enumerate(episodes.observations):
            x = x0 + i * (frame_px + 12)
            paste_image(canvas, frame, (x, title_h + 18), scale=scale)
            draw.text((x, title_h + 18 + frame_px + 6), str(positions[i]), fill=(80, 80, 80), font=small)
        last = states[-1]
        draw.text(
            (x0, title_h + 18 + frame_px + 28),
            f"最后记忆 [row,col,vrow,vcol] = [{last[0]:.0f}, {last[1]:.0f}, {last[2]:.0f}, {last[3]:.0f}]",
            fill=color,
            font=small,
        )

    draw_seq(from_left, left_pos, left_state, pad, "从左边来：right, right", (40, 90, 200))
    draw_seq(
        from_right,
        right_pos,
        right_state,
        pad + cols * (frame_px + 12) + 36,
        "从右边来：left, left",
        (180, 70, 40),
    )

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_compress(output_path):
    world = MovingSquareWorld()
    episode, _ = world.generate([2, 2, 4, 4], start=(2, 2))
    image = episode.observations[0]

    scale = 8
    pad = 28
    title_h = 44
    cell = 16 * scale
    img_w = pad * 5 + cell * 4
    img_h = title_h + cell + 78
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((pad, 8), "块平均会抹掉边缘：压得越狠，红方块越糊", fill="black", font=title_font)

    paste_image(canvas, image, (pad, title_h), scale=scale)
    draw.text((pad, title_h + cell + 8), "原图 768 个数", fill="black", font=font)
    draw.text((pad, title_h + cell + 28), "中心 (3.0, 3.0)", fill=(90, 90, 90), font=small)

    for i, bs in enumerate((2, 4, 8), start=1):
        latent = block_average_encode(image, block_size=bs)
        rec = block_average_decode(latent, block_size=bs)
        mse = reconstruction_mse(image, rec)
        x = pad + i * (cell + pad)
        paste_image(canvas, rec.astype(np.uint8), (x, title_h), scale=scale)
        draw.text((x, title_h + cell + 8), f"block={bs}  {image.size // latent.size}×", fill="black", font=font)
        draw.text((x, title_h + cell + 28), f"MSE {mse:.1f}  latent {latent.size}", fill=(90, 90, 90), font=small)

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_depth_occupancy(output_path):
    depth = np.array(
        [
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [3.0, 2.0, 2.0, 2.0, 3.0],
            [3.0, 2.0, 1.5, 2.0, 3.0],
            [3.0, 2.0, 2.0, 2.0, 3.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    points = depth_to_points(depth, fx=4.0, fy=4.0, cx=2.0, cy=2.0)
    occupancy = points_to_occupancy(points, x_range=(-2, 2), z_range=(0, 4), resolution=0.5)

    pad = 28
    title_h = 44
    panel = 180
    img_w = pad * 4 + panel * 3
    img_h = title_h + panel + 78
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((pad, 8), "5×5 深度图变成 25 个三维点，再落到 8×8 Occupancy", fill="black", font=title_font)

    # depth
    x0, y0 = pad, title_h
    cell = panel // 5
    dmin, dmax = float(depth.min()), float(depth.max())
    for r in range(5):
        for c in range(5):
            t = (depth[r, c] - dmin) / (dmax - dmin)
            color = (int(40 + 40 * t), int(70 + 40 * t), int(255 - 80 * t))
            draw.rectangle(
                [x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1],
                fill=color,
            )
            draw.text((x0 + c * cell + 8, y0 + r * cell + 10), f"{depth[r, c]:.1f}", fill="white", font=small)
    draw.rectangle([x0, y0, x0 + 5 * cell, y0 + 5 * cell], outline="black")
    draw.text((x0, y0 + panel + 8), "深度 (5,5)", fill="black", font=font)
    draw.text((x0, y0 + panel + 28), "中心 1.5 → 点 (0,0,1.5)", fill=(90, 90, 90), font=small)

    # points top-down x-z
    x1 = pad * 2 + panel
    draw.rectangle([x1, y0, x1 + panel, y0 + panel], outline="black", fill=(250, 250, 250))
    for pt in points:
        px = x1 + int((pt[0] + 2) / 4 * panel)
        py = y0 + int((4 - pt[2]) / 4 * panel)
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(40, 90, 200))
    draw.text((x1, y0 + panel + 8), "点云俯视 (x, z)", fill="black", font=font)
    draw.text((x1, y0 + panel + 28), "同列同深会叠成一点", fill=(90, 90, 90), font=small)

    # occupancy
    x2 = pad * 3 + panel * 2
    gh, gw = occupancy.shape
    cell2 = panel // gw
    for r in range(gh):
        for c in range(gw):
            box = [x2 + c * cell2, y0 + r * cell2, x2 + (c + 1) * cell2 - 1, y0 + (r + 1) * cell2 - 1]
            fill = (220, 70, 70) if occupancy[r, c] else (245, 245, 245)
            draw.rectangle(box, fill=fill, outline=(210, 210, 210))
    draw.rectangle([x2, y0, x2 + gw * cell2, y0 + gh * cell2], outline="black")
    draw.text((x2, y0 + panel + 8), f"Occupancy {tuple(occupancy.shape)}", fill="black", font=font)
    draw.text((x2, y0 + panel + 28), f"{int(occupancy.sum())} 格被点落入", fill=(90, 90, 90), font=small)

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_cem_search(output_path):
    start, target = 0.0, 3.0
    actions, history = cem_plan_1d(start, target, horizon=5, population=400, elite=40, rounds=5, seed=0)

    pad = 24
    title_h = 42
    rounds = 5
    cell_w, cell_h = 130, 110
    img_w = pad + rounds * (cell_w + 12) + pad
    img_h = title_h + cell_h + 92
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((pad, 8), "CEM：随机动作逐步收成一条能走到 3 的序列", fill="black", font=title_font)

    rng = np.random.default_rng(0)
    mean = np.zeros(5, dtype=np.float32)
    std = np.ones(5, dtype=np.float32)
    for r in range(rounds):
        samples = np.clip(rng.normal(mean, std, size=(400, 5)), -1.0, 1.0)
        finals = start + samples.sum(axis=1)
        scores = -(finals - target) ** 2 - 0.01 * (samples**2).sum(axis=1)
        elite = samples[np.argsort(scores)[-40:]]
        x = pad + r * (cell_w + 12)
        y = title_h
        draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(180, 180, 180), fill=(250, 250, 250))
        def to_x(val):
            return x + 8 + int((val + 1.0) / 8.0 * (cell_w - 16))

        draw.line([(to_x(target), y + 8), (to_x(target), y + cell_h - 8)], fill=(46, 140, 72), width=2)
        for f in finals[::8]:
            fx = to_x(f)
            draw.line([(fx, y + cell_h // 2 - 16), (fx, y + cell_h // 2 + 16)], fill=(90, 140, 220), width=1)
        mean_final = float(start + elite.mean(axis=0).sum())
        draw.line([(to_x(mean_final), y + 8), (to_x(mean_final), y + cell_h - 8)], fill=(200, 50, 50), width=3)
        draw.text((x + 6, y + cell_h + 6), f"第 {r + 1} 轮", fill="black", font=font)
        draw.text((x + 6, y + cell_h + 26), f"最好 {history[r]:.3f}", fill=(90, 90, 90), font=small)
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-4

    draw.text(
        (pad, img_h - 22),
        f"绿=目标 3    红=精英均值    蓝=样本    最终动作和 = {float(actions.sum()):.3f}",
        fill=(80, 80, 80),
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_symlog(output_path):
    values = np.array([-100.0, -1.0, 0.0, 1.0, 100000.0], dtype=np.float32)
    encoded = symlog(values)
    pad = 28
    title_h = 42
    img_w = 760
    img_h = 220
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((pad, 8), "Symlog 压住数量级，正负号还在", fill="black", font=title_font)

    y1, y2 = 80, 150
    draw.line([(pad, y1), (img_w - pad, y1)], fill=(180, 180, 180), width=2)
    draw.line([(pad, y2), (img_w - pad, y2)], fill=(180, 180, 180), width=2)
    draw.text((pad, y1 - 28), "原值", fill="black", font=font)
    draw.text((pad, y2 - 28), "symlog", fill="black", font=font)

    def place(vals, y, color, labels, stagger):
        lo, hi = float(vals.min()), float(vals.max())
        for i, (val, label) in enumerate(zip(vals, labels)):
            x = pad + 80 + int((float(val) - lo) / (hi - lo) * (img_w - 2 * pad - 120))
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
            dy = 10 if i % 2 == 0 else stagger
            draw.text((x - 18, y + dy), label, fill=color, font=small)

    place(values, y1, (200, 60, 50), ["-100", "-1", "0", "1", "100000"], 26)
    place(encoded, y2, (40, 90, 200), ["-4.62", "-0.69", "0.00", "0.69", "11.51"], 26)
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_lineworld(output_path):
    world = LineWorld(slip_probability=0.2)
    random_policy = lambda state, rng: rng.choice(world.actions)
    transitions, episode_ids = world.collect(random_policy, episodes=200, max_steps=20, seed=4)
    train = [t for t, i in zip(transitions, episode_ids) if i < 140]
    model = EmpiricalDynamics().fit(train)

    pad = 28
    title_h = 42
    img_w = 820
    img_h = 320
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((pad, 8), "140 个训练 episode 数出来的 P(next | state=3, action)", fill="black", font=title_font)

    def bar_panel(x, action, color):
        dist = model.distribution(3, action)
        draw.text((x, title_h + 6), f"从 3 {action}", fill="black", font=font)
        max_h = 140
        for i, s in enumerate(range(7)):
            p = dist.get(s, 0.0)
            bh = int(p * max_h)
            bx = x + i * 46
            by = title_h + 30 + max_h - bh
            draw.rectangle([bx, title_h + 30, bx + 34, title_h + 30 + max_h], fill=(245, 245, 245))
            if bh:
                draw.rectangle([bx, by, bx + 34, title_h + 30 + max_h], fill=color)
            draw.text((bx + 8, title_h + 34 + max_h), str(s), fill=(80, 80, 80), font=small)
            if p > 0:
                draw.text((bx + 2, by - 16), f"{p:.2f}", fill=color, font=small)

    bar_panel(pad, "left", (200, 80, 50))
    bar_panel(pad + 390, "right", (40, 110, 190))
    draw.text((pad, img_h - 28), "真实设定是 0.20 停、0.80 移动；有限计数不会恰好等于这两个数", fill=(90, 90, 90), font=small)
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_mpc(output_path):
    world = LineWorld(slip_probability=0.2)
    random_policy = lambda state, rng: rng.choice(world.actions)
    transitions, episode_ids = world.collect(random_policy, episodes=200, max_steps=20, seed=4)
    train = [t for t, i in zip(transitions, episode_ids) if i < 140]
    model = EmpiricalDynamics().fit(train)
    steps, plans = mpc_episode(world, model, depth=4, max_steps=20, seed=7, action_order=world.actions)

    pad = 24
    title_h = 42
    n = len(steps)
    cell = 36
    panel_w = 7 * cell + 20
    img_w = pad + n * (panel_w + 16) + pad
    img_h = title_h + 150
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((pad, 8), "MPC：每步都重新规划，只执行第一步，打滑了也能到 6", fill="black", font=title_font)

    fills = {0: (220, 70, 70), 6: (46, 140, 72)}
    for i, (step, plan) in enumerate(zip(steps, plans)):
        x = pad + i * (panel_w + 16)
        y = title_h + 8
        for s in range(7):
            bx = x + s * cell
            fill = fills.get(s, (240, 240, 240))
            if s == step.state:
                fill = (40, 90, 200)
            draw.rounded_rectangle([bx, y, bx + cell - 4, y + cell], radius=5, fill=fill, outline=(50, 50, 50))
            label = "×" if s == 0 else "G" if s == 6 else str(s)
            if s == step.state:
                label = "A"
            ink = "white" if s in (0, 6) or s == step.state else (70, 70, 70)
            draw.text((bx + 10, y + 8), label, fill=ink, font=font)
        draw.text((x, y + cell + 10), f"{i + 1}. {step.state} --{plan.action}--> {step.next_state}", fill="black", font=font)
        draw.text((x, y + cell + 30), f"reward {step.reward:g}", fill=(90, 90, 90), font=small)
        if step.state == step.next_state:
            draw.text((x, y + cell + 48), "打滑，停在原地", fill=(180, 80, 40), font=small)

    canvas.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "docs" / "public" / "carracing"
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_worlds(output_dir / "f1-worlds.png")
    visualize_worlds(output_dir / "f1-foundations.png")
    visualize_convolution(output_dir / "f1-convolution.png")
    visualize_patchify(output_dir / "f1-patchify.png")
    visualize_history(output_dir / "f1-history.png")
    visualize_compress(output_dir / "f1-compress.png")
    visualize_depth_occupancy(output_dir / "f2-depth-occupancy.png")
    visualize_cem_search(output_dir / "f2-cem-search.png")
    visualize_symlog(output_dir / "f2-symlog.png")
    visualize_lineworld(output_dir / "f3-counts.png")
    visualize_mpc(output_dir / "f3-mpc.png")

    print("\nAll foundations visualizations generated!")
