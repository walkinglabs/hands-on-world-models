#!/usr/bin/env python3
"""
路线 B/C 讲义配图：按 B1/B2/C1/C2 的默认配置跑真实代码。
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F

from hwm.data import ACTION_NAMES, MovingSquareWorld, make_pixelworld_dataset
from hwm.jepa import (
    TinyVideoJEPA,
    apply_linear_probe,
    feature_spread,
    fit_linear_probe_weights,
    jepa_batch_from_episodes,
    patchify_video,
)
from hwm.video import (
    ActionTokenTransformer,
    TinyVQVAE,
    motion_direction_accuracy,
    red_centers,
    rollout_token_model,
    token_accuracy,
    video_batch_from_episodes,
)


def try_load_font(size=16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", size)
    except Exception:
        return ImageFont.load_default()


def _upsample(array, scale=8):
    image = np.asarray(array)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image).resize(
        (image.shape[1] * scale, image.shape[0] * scale), Image.NEAREST
    )


def _tensor_to_hwc(frame):
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = frame.detach().cpu().permute(1, 2, 0).numpy()
    else:
        frame = frame.detach().cpu().numpy()
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def _draw_line_series(draw, values, origin, plot_w, plot_h, color, width=2):
    vmax = max(values)
    vmin = min(values)
    span = max(vmax - vmin, 1e-6)
    points = []
    for index, value in enumerate(values):
        x = origin[0] + int(index / max(len(values) - 1, 1) * plot_w)
        y = origin[1] + plot_h - int((value - vmin) / span * plot_h)
        points.append((x, y))
    if len(points) == 1:
        points = [points[0], (points[0][0] + 1, points[0][1])]
    draw.line(points, fill=color, width=width)
    return points[-1]


def visualize_pixelworld(output_path):
    world = MovingSquareWorld()
    start = (2, 2)
    frames = [world.render(start)]
    position = start
    demo_actions = [2, 2, 4, 2, 4]
    for action in demo_actions:
        position = world.next_position(position, action)
        frames.append(world.render(position))

    cell = 128
    padding = 18
    title_h = 42
    caption_h = 36
    cols = len(frames)
    img_w = cols * cell + (cols + 1) * padding
    img_h = title_h + cell + caption_h + padding * 2 + 28
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)

    draw.text((padding, 8), "PixelWorld：16×16 的动作条件小世界", fill="black", font=title_font)
    labels = ["t=0 起点"] + [f"a={ACTION_NAMES[action]}" for action in demo_actions]
    for index, (frame, label) in enumerate(zip(frames, labels)):
        x = padding + index * (cell + padding)
        y = title_h + padding
        canvas.paste(_upsample(frame, scale=8), (x, y))
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="#888888")
        draw.text((x + 4, y + cell + 6), label, fill="black", font=font)

    draw.text(
        (padding, img_h - 24),
        "红=方块，绿=目标 (12,12)。动作：0 stay / 1 left / 2 right / 3 up / 4 down。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_vq_pipeline(output_path):
    cell_w = 168
    cell_h = 108
    padding = 18
    title_h = 40
    components = [
        ("16×16 帧", "[B,3,16,16]", "#1d4ed8"),
        ("Encoder\n两层 stride-2", "[B,8,4,4]", "#047857"),
        ("码本最近邻\n16 个码字", "token [B,4,4]", "#c2410c"),
        ("Decoder\n画回像素", "[B,3,16,16]", "#6d28d9"),
        ("Transformer\n+ 动作", "下一组 token", "#0f766e"),
    ]
    cols = len(components)
    img_w = cols * cell_w + (cols + 1) * padding
    img_h = cell_h + title_h + padding * 2 + 52
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "路线 B：先压成 token，再按动作猜下一组编号", fill="black", font=title_font)

    y = title_h + padding
    for i, (name, shape, color) in enumerate(components):
        x = i * (cell_w + padding) + padding
        draw.rounded_rectangle([x, y, x + cell_w, y + cell_h], radius=8, outline=color, width=3)
        for j, line in enumerate(name.split("\n")):
            draw.text((x + 10, y + 12 + j * 20), line, fill=color, font=font)
        draw.text((x + 10, y + 72), shape, fill="gray", font=small)
        if i < cols - 1:
            ax = x + cell_w + 4
            ay = y + cell_h // 2
            draw.line([(ax, ay), (ax + 10, ay)], fill="black", width=2)
            draw.polygon([(ax + 10, ay - 4), (ax + 14, ay), (ax + 10, ay + 4)], fill="black")

    draw.text(
        (padding, img_h - 28),
        "一张图变成 16 个编号。B1 默认 additive 注入动作；帧内 token 同时可见。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def _train_b1():
    torch.manual_seed(0)
    episodes = make_pixelworld_dataset(num_episodes=8, length=8, seed=0)
    current, actions, following = video_batch_from_episodes(episodes)
    images = torch.cat((current, following))
    tokenizer = TinyVQVAE(codebook_size=16, embedding_size=8)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
    for _ in range(30):
        optimizer.zero_grad()
        loss, _ = tokenizer.continuous_loss(images)
        loss.backward()
        optimizer.step()
    warmup_loss = float(loss.detach())
    tokenizer.initialize_codebook(images)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-4)
    vq_losses = []
    for _ in range(20):
        optimizer.zero_grad()
        output = tokenizer(images)
        output["loss"].backward()
        optimizer.step()
        vq_losses.append(float(output["loss"].detach()))
    used_codes = int(torch.unique(output["tokens"]).numel())
    usage = torch.bincount(output["tokens"].reshape(-1), minlength=16).tolist()
    with torch.no_grad():
        warmup_recon = tokenizer.decoder(tokenizer.encoder(images[:8]))
        current_tokens = tokenizer.encode_tokens(current).flatten(1)
        next_tokens = tokenizer.encode_tokens(following).flatten(1)
    dynamics = ActionTokenTransformer(codebook_size=16, model_size=32)
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=3e-3)
    token_losses = []
    for _ in range(35):
        optimizer.zero_grad()
        loss = dynamics.loss(current_tokens, actions, next_tokens)
        loss.backward()
        optimizer.step()
        token_losses.append(float(loss.detach()))
    with torch.no_grad():
        logits = dynamics(current_tokens, actions)
        predicted_frames = tokenizer.decode_tokens(logits.argmax(dim=-1).reshape(-1, 4, 4))
    return {
        "current": current,
        "following": following,
        "actions": actions,
        "images": images,
        "tokenizer": tokenizer,
        "output": output,
        "warmup_recon": warmup_recon,
        "warmup_loss": warmup_loss,
        "vq_losses": vq_losses,
        "used_codes": used_codes,
        "usage": usage,
        "token_losses": token_losses,
        "accuracy": float(token_accuracy(logits, next_tokens)),
        "direction": float(motion_direction_accuracy(current, predicted_frames, following)),
        "predicted_frames": predicted_frames,
        "copy_mse": float(F.mse_loss(current, following)),
        "copy_direction": float(motion_direction_accuracy(current, current, following)),
    }


def visualize_vq_recon(output_path, b1):
    cell = 96
    padding = 16
    label_w = 96
    title_h = 40
    rows = 3
    cols = 4
    img_w = label_w + cols * cell + (cols + 1) * padding
    img_h = title_h + rows * (cell + 28) + padding + 36
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "B1：原图 / 复制上一帧 / VQ 重建", fill="black", font=title_font)

    recon = b1["output"]["reconstruction"].detach()
    recon_mse = float(b1["output"]["reconstruction_loss"])
    row_specs = [
        ("原图", b1["images"][:cols], None),
        ("复制上一帧", b1["current"][:cols], f"MSE {b1['copy_mse']:.5f}"),
        ("VQ 重建", recon[:cols], f"加权 {recon_mse:.4f}"),
    ]
    for row, (name, frames, note) in enumerate(row_specs):
        y = title_h + row * (cell + 28)
        draw.text((padding, y + cell // 2 - 8), name, fill="black", font=font)
        if note:
            draw.text((padding, y + cell // 2 + 10), note, fill="gray", font=small)
        for col in range(cols):
            x = label_w + padding + col * (cell + padding // 2)
            canvas.paste(_upsample(_tensor_to_hwc(frames[col]), scale=6), (x, y))
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="#888888")
    draw.text(
        (padding, img_h - 28),
        f"码本用了 {b1['used_codes']}/16。重建损失是前景加权 MSE，数字不能直接和复制帧 MSE 比。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_codebook(output_path, b1):
    usage = b1["usage"]
    width, height = 720, 280
    padding = 48
    title_h = 36
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(12)
    draw.text((padding, 8), "B1：16 个码字都用上了，但很不均匀", fill="black", font=title_font)

    plot_w = width - padding * 2
    plot_h = height - title_h - padding - 8
    origin = (padding, title_h + 8)
    vmax = max(usage)
    gap = 2
    bar_w = max(int(plot_w / len(usage) - gap), 8)
    for index, count in enumerate(usage):
        x0 = origin[0] + index * (bar_w + gap)
        bar_h = int(count / vmax * plot_h)
        y0 = origin[1] + plot_h - bar_h
        draw.rectangle([x0, y0, x0 + bar_w, origin[1] + plot_h], fill="#1d4ed8")
        if count >= 200:
            draw.text((x0, y0 - 14), str(count), fill="#1d4ed8", font=font)
    draw.text(
        (padding, height - 28),
        f"VQ loss {b1['vq_losses'][0]:.4f} → {b1['vq_losses'][-1]:.4f}。0 / 9 / 12 / 15 号码字扛了大部分像素。",
        fill="gray",
        font=font,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_token_pred(output_path, b1):
    cell = 96
    padding = 16
    label_w = 110
    title_h = 40
    rows = 3
    cols = 4
    img_w = label_w + cols * cell + (cols + 1) * padding
    img_h = title_h + rows * (cell + 28) + padding + 36
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "B1：下一步真值 / 复制当前帧 / token 解码", fill="black", font=title_font)

    row_specs = [
        ("真值 t+1", b1["following"][:cols], None),
        ("复制当前帧", b1["current"][:cols], f"方向 {b1['copy_direction']:.2f}"),
        ("预测解码", b1["predicted_frames"][:cols], f"token {b1['accuracy']:.3f}"),
    ]
    for row, (name, frames, note) in enumerate(row_specs):
        y = title_h + row * (cell + 28)
        draw.text((padding, y + cell // 2 - 8), name, fill="black", font=font)
        if note:
            draw.text((padding, y + cell // 2 + 10), note, fill="gray", font=small)
        for col in range(cols):
            x = label_w + padding + col * (cell + padding // 2)
            canvas.paste(_upsample(_tensor_to_hwc(frames[col]), scale=6), (x, y))
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="#888888")
    draw.text(
        (padding, img_h - 28),
        f"token accuracy {b1['accuracy']:.3f}，解码后方向准确率只有 {b1['direction']:.3f}。背景猜对了，方块还没学会走。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def _train_b2():
    torch.manual_seed(1)
    episodes = make_pixelworld_dataset(12, 8, seed=2)
    current, actions, following = video_batch_from_episodes(episodes)
    images = torch.cat((current, following))
    tokenizer = TinyVQVAE(codebook_size=16, embedding_size=8)
    opt = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
    for _ in range(30):
        opt.zero_grad()
        loss, _ = tokenizer.continuous_loss(images)
        loss.backward()
        opt.step()
    tokenizer.initialize_codebook(images)
    with torch.no_grad():
        current_tokens = tokenizer.encode_tokens(current).flatten(1)
        next_tokens = tokenizer.encode_tokens(following).flatten(1)
    same_start = current_tokens[:1].expand(5, -1)
    all_actions = torch.arange(5)
    rows = []
    models = {}
    for injection in ("none", "additive", "film"):
        torch.manual_seed(7)
        model = ActionTokenTransformer(
            codebook_size=16, model_size=32, action_injection=injection
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(25):
            optimizer.zero_grad()
            train_loss = model.loss(current_tokens, actions, next_tokens)
            train_loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = model(same_start, all_actions)
            frames = tokenizer.decode_tokens(logits.argmax(-1).reshape(-1, 4, 4))
            full_logits = model(current_tokens, actions)
            acc = float(token_accuracy(full_logits, next_tokens))
            direction = float(
                motion_direction_accuracy(
                    current,
                    tokenizer.decode_tokens(full_logits.argmax(-1).reshape(-1, 4, 4)),
                    following,
                )
            )
        sensitivity = float((logits - logits[:1]).abs().mean())
        rows.append(
            {
                "name": injection,
                "loss": float(train_loss.detach()),
                "sensitivity": sensitivity,
                "acc": acc,
                "direction": direction,
                "centers": red_centers(frames).tolist(),
            }
        )
        models[injection] = model
    dynamics = models["additive"]
    with torch.no_grad():
        teacher_tokens = dynamics(current_tokens, actions).argmax(-1)
        teacher_acc = float((teacher_tokens == next_tokens).float().mean())
        teacher_frames = tokenizer.decode_tokens(teacher_tokens.reshape(-1, 4, 4))
    teacher_direction = float(motion_direction_accuracy(current, teacher_frames, following))
    right_actions = [torch.tensor([2]) for _ in range(8)]
    rollout_frames = rollout_token_model(
        dynamics, tokenizer, current_tokens[:1], right_actions, (4, 4)
    )
    return {
        "rows": rows,
        "teacher_acc": teacher_acc,
        "teacher_direction": teacher_direction,
        "rollout_frames": rollout_frames,
        "centers": red_centers(rollout_frames).tolist(),
        "start_frame": current[0],
    }


def visualize_action_ablation(output_path, b2):
    width, height = 720, 280
    padding = 48
    title_h = 36
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "B2：换动作以后，logits 会不会动？", fill="black", font=title_font)

    plot_w = width - padding * 2
    plot_h = height - title_h - padding - 20
    origin = (padding, title_h + 8)
    values = [row["sensitivity"] for row in b2["rows"]]
    names = [row["name"] for row in b2["rows"]]
    vmax = max(values + [0.05])
    gap = 28
    bar_w = 90
    colors = ["#6b7280", "#1d4ed8", "#047857"]
    for index, (name, value, color) in enumerate(zip(names, values, colors)):
        x0 = origin[0] + 40 + index * (bar_w + gap)
        bar_h = int(value / vmax * plot_h)
        y0 = origin[1] + plot_h - bar_h
        draw.rectangle([x0, y0, x0 + bar_w, origin[1] + plot_h], fill=color)
        draw.text((x0, origin[1] + plot_h + 6), name, fill="black", font=font)
        draw.text((x0, y0 - 18), f"{value:.4f}", fill=color, font=small)
    draw.text(
        (padding, height - 24),
        "none 的 sensitivity 必须是 0。additive / FiLM 只说明动作进了网络，不说明方向已经对。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_free_rollout(output_path, b2):
    frames = b2["rollout_frames"]
    picks = [0, 1, 2, 4, 8]
    cell = 96
    padding = 16
    title_h = 40
    img_w = len(picks) * cell + (len(picks) + 1) * padding
    img_h = title_h + cell + 78
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "B2：连续向右 8 步，红块很快停住", fill="black", font=title_font)
    centers = b2["centers"]
    for index, step in enumerate(picks):
        x = padding + index * (cell + padding)
        y = title_h + 6
        canvas.paste(_upsample(_tensor_to_hwc(frames[step]), scale=6), (x, y))
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="#888888")
        row, col = centers[step]
        label = f"t={step}"
        if row == row:
            label += f"  ({row:.2f},{col:.2f})"
        draw.text((x, y + cell + 6), label, fill="black", font=small)
    draw.text(
        (padding, img_h - 28),
        f"teacher-forced token acc {b2['teacher_acc']:.3f}，方向 {b2['teacher_direction']:.3f}。自由生成从第 2 步起中心不再动。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_jepa_pipeline(output_path):
    cell_w = 176
    cell_h = 112
    padding = 18
    title_h = 40
    components = [
        ("历史 2 帧\npatch 4×4", "[B,2,16,48]", "#1d4ed8"),
        ("Online\nencoder", "context 均值", "#047857"),
        ("Predictor\n+ 动作 + 位置", "pred [B,16,D]", "#c2410c"),
        ("Target\nencoder", "无梯度", "#6d28d9"),
        ("Smooth L1\npred vs target", "不画回像素", "#0f766e"),
    ]
    cols = len(components)
    img_w = cols * cell_w + (cols + 1) * padding
    img_h = cell_h + title_h + padding * 2 + 52
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "路线 C：只预测下一帧特征，屏幕上什么都不画", fill="black", font=title_font)
    y = title_h + padding
    for i, (name, shape, color) in enumerate(components):
        x = i * (cell_w + padding) + padding
        draw.rounded_rectangle([x, y, x + cell_w, y + cell_h], radius=8, outline=color, width=3)
        for j, line in enumerate(name.split("\n")):
            draw.text((x + 10, y + 12 + j * 20), line, fill=color, font=font)
        draw.text((x + 10, y + 78), shape, fill="gray", font=small)
        if i < cols - 1:
            ax = x + cell_w + 4
            ay = y + cell_h // 2
            draw.line([(ax, ay), (ax + 10, ay)], fill="black", width=2)
            draw.polygon([(ax + 10, ay - 4), (ax + 14, ay), (ax + 10, ay + 4)], fill="black")
    draw.text(
        (padding, img_h - 28),
        "Target encoder 不接收梯度，只靠 EMA 跟着 online 走。C1 默认不做动作条件。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def _train_c1():
    torch.manual_seed(0)
    episodes = make_pixelworld_dataset(6, 6, seed=0)
    video, actions, positions = jepa_batch_from_episodes(episodes, history_length=3)
    patches = patchify_video(video[:2], patch_size=4)
    model = TinyVideoJEPA(feature_size=16)
    loss, prediction, target, features = model.loss(video, actions=None)
    init = {
        "video": tuple(video.shape),
        "patches": tuple(patches.shape),
        "pred": tuple(prediction.shape),
        "target_grad": bool(target.requires_grad),
        "init_loss": float(loss.detach()),
        "init_spread": float(feature_spread(features).detach()),
    }
    parameters = list(model.online_encoder.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=3e-3)
    losses, spreads = [], []
    for _ in range(35):
        optimizer.zero_grad()
        loss, prediction, target, features = model.loss(video, actions=None)
        loss.backward()
        optimizer.step()
        model.update_target(momentum=0.99)
        losses.append(float(loss.detach()))
        spreads.append(float(feature_spread(features).detach()))
    mask = torch.tensor(([1, 0] * 8), dtype=torch.bool)[None].expand(len(video), -1)
    masked_loss, _, _, _ = model.loss(video, mask=mask)
    init.update(
        {
            "losses": losses,
            "spreads": spreads,
            "masked": int(mask.sum()),
            "masked_loss": float(masked_loss.detach()),
            "n_clips": len(video),
        }
    )
    return init


def visualize_jepa_curves(output_path, c1):
    width, height = 720, 280
    padding = 48
    title_h = 36
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(12)
    draw.text((padding, 8), "C1：loss 下降，feature spread 没有塌", fill="black", font=title_font)
    plot_w = width - padding * 2
    plot_h = height - title_h - padding - 8
    origin = (padding, title_h + 8)
    last_loss = _draw_line_series(draw, c1["losses"], origin, plot_w, plot_h, "#1d4ed8")
    last_spread = _draw_line_series(draw, c1["spreads"], origin, plot_w, plot_h, "#c2410c")
    draw.text(
        (last_loss[0] - 110, last_loss[1] - 16),
        f"loss {c1['losses'][-1]:.3f}",
        fill="#1d4ed8",
        font=font,
    )
    draw.text(
        (min(last_spread[0], width - 160), last_spread[1] + 4),
        f"spread {c1['spreads'][-1]:.3f}",
        fill="#c2410c",
        font=font,
    )
    draw.text(
        (padding, height - 28),
        f"loss {c1['losses'][0]:.3f} → {c1['losses'][-1]:.3f}    spread {c1['spreads'][0]:.3f} → {c1['spreads'][-1]:.3f}",
        fill="gray",
        font=font,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def _train_c2():
    torch.manual_seed(2)
    train_episodes = make_pixelworld_dataset(10, 7, seed=2)
    test_episodes = make_pixelworld_dataset(4, 7, seed=31)
    video, actions, positions = jepa_batch_from_episodes(train_episodes, history_length=3)
    test_video, test_actions, test_positions = jepa_batch_from_episodes(
        test_episodes, history_length=3
    )
    model = TinyVideoJEPA(feature_size=16)
    parameters = (
        list(model.online_encoder.parameters())
        + list(model.predictor.parameters())
        + list(model.action_embedding.parameters())
    )
    optimizer = torch.optim.Adam(parameters, lr=3e-3)
    losses, spreads = [], []
    for _ in range(40):
        optimizer.zero_grad()
        loss, prediction, target, features = model.loss(video, actions)
        loss.backward()
        optimizer.step()
        model.update_target(0.99)
        losses.append(float(loss.detach()))
        spreads.append(float(feature_spread(features).detach()))
    with torch.no_grad():
        _, train_target, _ = model(video, actions)
        _, test_target, _ = model(test_video, test_actions)
    probe_weights = fit_linear_probe_weights(train_target.flatten(1), positions)
    test_prediction = apply_linear_probe(test_target.flatten(1), probe_weights)
    probe_mse = float(F.mse_loss(test_prediction, test_positions))
    constant = positions.mean(0).expand_as(test_positions)
    constant_mse = float(F.mse_loss(constant, test_positions))
    same_history = video[:1].expand(5, -1, -1, -1, -1)
    all_actions = torch.arange(5)
    with torch.no_grad():
        predictions, _, _ = model(same_history, all_actions)
    differences = [
        float((predictions[0] - predictions[i]).square().mean()) for i in range(1, 5)
    ]
    predicted_positions = apply_linear_probe(predictions.flatten(1), probe_weights)
    goal = torch.tensor([12 / 15, 12 / 15])
    distances = torch.linalg.vector_norm(predicted_positions - goal, dim=-1)
    return {
        "losses": losses,
        "spreads": spreads,
        "probe_mse": probe_mse,
        "constant_mse": constant_mse,
        "pred": test_prediction.detach(),
        "true": test_positions.detach(),
        "differences": differences,
        "predicted_positions": predicted_positions.detach(),
        "distances": [float(x) for x in distances],
        "chosen": int(distances.argmin()),
        "n_train": len(video),
        "n_test": len(test_video),
    }


def visualize_probe(output_path, c2):
    width, height = 520, 360
    padding = 48
    title_h = 36
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(12)
    draw.text((padding, 8), "C2：线性探针读出的方块位置", fill="black", font=title_font)
    plot = height - title_h - padding - 20
    origin = (padding, title_h + 8)
    box = plot
    draw.rectangle(
        [origin[0], origin[1], origin[0] + box, origin[1] + box],
        outline="#d1d5db",
        fill="#f8fafc",
    )
    true = c2["true"].numpy()
    pred = c2["pred"].numpy()
    for row, col in true:
        x = origin[0] + int(col * box)
        y = origin[1] + int(row * box)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline="#047857", width=2)
    for row, col in pred:
        x = origin[0] + int(col * box)
        y = origin[1] + int(row * box)
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill="#c2410c")
    draw.text(
        (padding, height - 28),
        f"held-out probe MSE {c2['probe_mse']:.4f} < 常数基线 {c2['constant_mse']:.4f}。绿圈真值，红点探针。",
        fill="gray",
        font=font,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_action_swap(output_path, c2):
    width, height = 720, 280
    padding = 48
    title_h = 36
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "C2：同一段历史，只换动作", fill="black", font=title_font)
    names = ["left", "right", "up", "down"]
    values = c2["differences"]
    plot_w = width - padding * 2
    plot_h = height - title_h - padding - 20
    origin = (padding, title_h + 8)
    vmax = max(values)
    gap = 24
    bar_w = 80
    for index, (name, value) in enumerate(zip(names, values)):
        x0 = origin[0] + 50 + index * (bar_w + gap)
        bar_h = int(value / vmax * plot_h)
        y0 = origin[1] + plot_h - bar_h
        draw.rectangle([x0, y0, x0 + bar_w, origin[1] + plot_h], fill="#1d4ed8")
        draw.text((x0, origin[1] + plot_h + 6), name, fill="black", font=font)
        draw.text((x0, y0 - 16), f"{value:.5f}", fill="#1d4ed8", font=small)
    draw.text(
        (padding, height - 24),
        f"相对 stay 的 feature MSE 都大于 0。探针选了动作 {c2['chosen']}（{ACTION_NAMES[c2['chosen']]}）。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "docs" / "public" / "carracing"
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_pixelworld(output_dir / "bc-pixelworld.png")
    visualize_vq_pipeline(output_dir / "bc-vq-pipeline.png")
    visualize_vq_pipeline(output_dir / "bc-vq-transformer.png")
    b1 = _train_b1()
    visualize_vq_recon(output_dir / "bc-vq-recon.png", b1)
    visualize_codebook(output_dir / "bc-codebook.png", b1)
    visualize_token_pred(output_dir / "bc-token-pred.png", b1)
    b2 = _train_b2()
    visualize_action_ablation(output_dir / "bc-action-ablation.png", b2)
    visualize_free_rollout(output_dir / "bc-free-rollout.png", b2)
    visualize_jepa_pipeline(output_dir / "bc-jepa-pipeline.png")
    visualize_jepa_pipeline(output_dir / "bc-jepa.png")
    c1 = _train_c1()
    visualize_jepa_curves(output_dir / "bc-jepa-curves.png", c1)
    c2 = _train_c2()
    visualize_probe(output_dir / "bc-probe.png", c2)
    visualize_action_swap(output_dir / "bc-action-swap.png", c2)

    print("\nB1 copy_mse", round(b1["copy_mse"], 5), "token_acc", round(b1["accuracy"], 3))
    print("B2 sensitivities", [round(r["sensitivity"], 4) for r in b2["rows"]])
    print("C1 loss", round(c1["losses"][0], 3), "->", round(c1["losses"][-1], 3))
    print("C2 probe", round(c2["probe_mse"], 4), "base", round(c2["constant_mse"], 4))
    print("\nRoute B/C lecture figures generated!")
