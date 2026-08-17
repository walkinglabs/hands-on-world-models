"""可视化脚本：捕获 World Models 训练的关键画面。"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add project root to path so we can import from scripts
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.run_carracing import ConvVAE, MDNRNN, LinearController, make_env, resize_frame


def load_models(output_dir):
    """加载训练好的模型。"""
    vae = ConvVAE(z_size=32)
    vae.load_state_dict(torch.load(output_dir / "vae.pt", map_location="cpu"))
    vae.eval()

    mdn = MDNRNN()
    mdn.load_state_dict(torch.load(output_dir / "mdn.pt", map_location="cpu"))
    mdn.eval()

    controller = LinearController(32 + 256)
    controller.parameters = np.load(output_dir / "controller.npy")

    return vae, mdn, controller


def visualize_random_rollout(env, seed=0, max_steps=200):
    """可视化随机策略收集的数据：帧 + 动作 + 奖励，让读者看到『喂给模型的是什么』。"""
    obs, _ = env.reset(seed=seed)
    frames, actions, rewards = [], [], []
    target_speed = np.random.uniform(0.1, 0.5)

    for step in range(max_steps):
        if step == 0 or np.random.rand() < 0.05:
            target_speed = np.random.uniform(0.1, 0.5)
        action = np.array([
            np.random.uniform(-1, 1),
            float(target_speed),
            float(np.random.rand() < 0.1),
        ])
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(obs.copy())
        actions.append(action.copy())
        rewards.append(float(reward))
        if terminated or truncated:
            break

    # 选取关键帧：均匀分布 5 帧
    n = len(frames)
    if n >= 5:
        indices = np.linspace(0, n - 1, 5, dtype=int)
    else:
        indices = list(range(n))

    display_size = 192
    key_frames = [np.array(Image.fromarray(frames[i]).resize((display_size, display_size), Image.BILINEAR)) for i in indices]

    # 画动作和奖励条
    bar_height = 40
    strip_width = display_size * len(key_frames)
    bar_strip = np.ones((bar_height * 2, strip_width, 3), dtype=np.uint8) * 240

    for col, idx in enumerate(indices):
        x0 = col * display_size
        steer = actions[idx][0]   # [-1, 1] → 左红右绿
        throttle = actions[idx][1]  # [0, 1] → 绿色高度
        brake = actions[idx][2]     # 0 or 1
        rew = rewards[idx]

        # 上排：方向盘（红=左，绿=右）+ 油门（蓝条高度）
        steer_norm = (steer + 1) / 2  # [0, 1]
        steer_color = (int(200 * (1 - steer_norm)), int(200 * steer_norm), 50)
        bar_strip[5:18, x0 + 10:x0 + 50] = steer_color
        # 油门
        throttle_h = int(throttle * 13)
        bar_strip[18 - throttle_h:18, x0 + 55:x0 + 80] = (50, 150, 50)
        # 刹车
        if brake > 0.5:
            bar_strip[5:18, x0 + 85:x0 + 110] = (200, 50, 50)

        # 下排：奖励（绿=正，红=负）
        rew_color = (50, 180, 50) if rew > 0 else (200, 50, 50)
        rew_h = min(int(abs(rew) * 3), 13)
        if rew > 0:
            bar_strip[25:25 + rew_h, x0 + 10:x0 + display_size - 10] = rew_color
        else:
            bar_strip[40 - rew_h:40, x0 + 10:x0 + display_size - 10] = rew_color

    # 拼接：帧在上，条在下
    frame_strip = np.concatenate(key_frames, axis=1)
    comparison = np.concatenate([frame_strip, bar_strip], axis=0)
    return comparison, n


def capture_environment_frame(env, seed=0, warmup_steps=30):
    """捕获环境帧：先跑 warmup_steps 步，让车进入赛道弯道，画面更有代表性。"""
    obs, _ = env.reset(seed=seed)
    for _ in range(warmup_steps):
        # 轻微左转 + 油门，让车离开起点直道进入弯道
        action = np.array([0.3, 0.5, 0.0])
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return obs


def visualize_vae_reconstruction(vae, frame):
    """可视化 VAE 重建：原图 vs 重建。frame 为 uint8 [0,255] 96x96。"""
    frame_64 = resize_frame(frame)  # → float [0,255], 64x64
    with torch.no_grad():
        # VAE.encode 内部 /255 归一化到 [0,1]
        frame_tensor = torch.from_numpy(frame_64).unsqueeze(0).float()
        mean, logvar = vae.encode(frame_tensor)
        z = vae.reparameterize(mean, logvar)
        reconstruction = vae.decode(z).squeeze(0).permute(1, 2, 0).numpy()

    # Upscale to 256x256 for display (bilinear for smooth rendering)
    display_size = 256
    # 原图用原始 96x96 uint8 帧放大，保证清晰
    frame_display = np.array(Image.fromarray(frame).resize((display_size, display_size), Image.BILINEAR))
    # 重建图从 VAE 输出 (float [0,1]) 转 uint8 再放大
    recon_display = np.array(Image.fromarray((reconstruction * 255).astype(np.uint8)).resize((display_size, display_size), Image.BILINEAR))

    # Concatenate original and reconstruction side by side
    comparison = np.concatenate([frame_display, recon_display], axis=1)
    return comparison


def visualize_dream_generation(vae, mdn, controller, initial_frame, steps=200):
    """可视化 M 在想象中生成的世界：C 出动作 → M 想象下一帧，全程不碰真实环境。"""
    frame_64 = resize_frame(initial_frame)
    with torch.no_grad():
        frame_tensor = torch.from_numpy(frame_64).unsqueeze(0).float()
        mean, logvar = vae.encode(frame_tensor)
        z = vae.reparameterize(mean, logvar)
        hidden = torch.zeros(1, mdn.hidden_size)

        frames = [initial_frame]  # 第一帧用真实观测
        for _ in range(steps):
            # C 根据当前 latent + 记忆出动作
            features = torch.cat((z, hidden), dim=-1).numpy()
            action_np = controller.act(features, controller.parameters, noise=0.0)
            action_tensor = torch.from_numpy(action_np).float()

            # M 想象下一状态
            z, _, hidden = mdn.sample(z, action_tensor, hidden, temperature=1.0)
            reconstruction = vae.decode(z).squeeze(0).permute(1, 2, 0).numpy()
            frames.append((reconstruction * 255).astype(np.uint8))

    display_size = 256
    upscaled = [np.array(Image.fromarray(f).resize((display_size, display_size), Image.BILINEAR)) for f in frames]

    # 选取关键帧：均匀分布
    if len(upscaled) >= 5:
        indices = np.linspace(0, len(upscaled) - 1, 5, dtype=int)
        key_frames = [upscaled[i] for i in indices]
    else:
        key_frames = upscaled

    comparison = np.concatenate(key_frames, axis=1)
    return comparison, len(frames)


def visualize_mdn_free_running(vae, mdn, initial_frame, steps=100):
    """可视化 M 的 free-running rollout：看复合误差如何累积。initial_frame 为 uint8 [0,255] 96x96。"""
    frame_64 = resize_frame(initial_frame)  # → float [0,255], 64x64
    with torch.no_grad():
        # VAE.encode 内部 /255 归一化到 [0,1]
        frame_tensor = torch.from_numpy(frame_64).unsqueeze(0).float()
        mean, logvar = vae.encode(frame_tensor)
        z = vae.reparameterize(mean, logvar)
        hidden = torch.zeros(1, mdn.hidden_size)

        # 第一帧用原始观测（96x96 uint8），后续用 VAE 解码的 64x64 重建
        frames = [initial_frame]
        for _ in range(steps):
            action = torch.zeros(1, 3)  # 零动作，只看预测稳定性
            z, _, hidden = mdn.sample(z, action, hidden, temperature=1.0)
            reconstruction = vae.decode(z).squeeze(0).permute(1, 2, 0).numpy()
            frames.append((reconstruction * 255).astype(np.uint8))

    # Upscale each frame to 256x256 for display (bilinear for smooth rendering)
    display_size = 256
    upscaled = [np.array(Image.fromarray(f).resize((display_size, display_size), Image.BILINEAR)) for f in frames]

    # 选取关键帧：第 0、10、30、60、99 步
    key_indices = [0, 10, 30, 60, 99]
    key_frames = [upscaled[i] for i in key_indices]

    # 拼接成一行
    comparison = np.concatenate(key_frames, axis=1)
    return comparison, key_indices


def visualize_real_evaluation(vae, mdn, controller, env, seed=0, max_steps=200):
    """可视化真实环境评估：捕获关键帧。"""
    obs, _ = env.reset(seed=seed)
    hidden = torch.zeros(1, mdn.hidden_size)
    frames = []

    with torch.no_grad():
        for step in range(max_steps):
            # resize_frame → float [0,255] 64x64, VAE.encode 内部 /255 归一化
            frame_64 = resize_frame(obs)
            frame_tensor = torch.from_numpy(frame_64).unsqueeze(0).float()
            mean, logvar = vae.encode(frame_tensor)
            z = vae.reparameterize(mean, logvar)

            features = torch.cat((z, hidden), dim=-1).numpy()
            action = controller.act(features, controller.parameters, noise=0.0)
            action_tensor = torch.from_numpy(action).float()

            frames.append(obs.copy())

            obs, reward, terminated, truncated, _ = env.step(action.reshape(-1))
            z, _, hidden = mdn.sample(z, action_tensor, hidden, temperature=1.0)

            if terminated or truncated:
                break

    # Upscale frames to 256x256 for display (bilinear for smooth rendering)
    display_size = 256
    upscaled = [np.array(Image.fromarray(f).resize((display_size, display_size), Image.BILINEAR)) for f in frames]

    # 选取关键帧：均匀分布
    if len(upscaled) >= 5:
        indices = np.linspace(0, len(upscaled) - 1, 5, dtype=int)
        key_frames = [upscaled[i] for i in indices]
    else:
        key_frames = upscaled

    comparison = np.concatenate(key_frames, axis=1)
    return comparison, len(frames)


def add_caption(image, text, height=30):
    """给图片加标题。"""
    new_image = Image.new("RGB", (image.shape[1], image.shape[0] + height), (255, 255, 255))
    new_image.paste(Image.fromarray(image), (0, 0))

    draw = ImageDraw.Draw(new_image)
    font = None
    for font_path in [
        "/System/Library/Fonts/Supplemental/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, 16)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    x = (new_image.width - text_width) // 2
    draw.text((x, 5), text, fill=(0, 0, 0), font=font)

    return np.array(new_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/carracing-world-model"))
    parser.add_argument("--image-dir", type=Path, default=Path("docs/public/carracing"))
    args = parser.parse_args()

    args.image_dir.mkdir(parents=True, exist_ok=True)

    env = make_env()

    # 0. 随机策略数据：让读者看到『喂给模型的是什么』
    print("可视化随机策略 rollout 数据...")
    random_data, total_frames = visualize_random_rollout(env, seed=42, max_steps=200)
    random_data = add_caption(random_data, f"随机策略数据：{total_frames} 帧（方向盘 / 油门 / 刹车 / 奖励）")
    Image.fromarray(random_data).save(args.image_dir / "random-rollout.png")

    print("加载模型...")
    vae, mdn, controller = load_models(args.output_dir)

    # 1. 环境初始帧（upscale to 256x256 for display, bilinear for smooth rendering）
    print("捕获环境初始帧...")
    initial_frame = capture_environment_frame(env, seed=0)
    initial_display = np.array(Image.fromarray(initial_frame).resize((256, 256), Image.BILINEAR))
    Image.fromarray(initial_display).save(args.image_dir / "carracing-initial.png")

    # 2. VAE 重建对比
    print("可视化 VAE 重建...")
    reconstruction = visualize_vae_reconstruction(vae, initial_frame)
    reconstruction = add_caption(reconstruction, "左：原图  右：VAE 重建")
    Image.fromarray(reconstruction).save(args.image_dir / "vae-reconstruction.png")

    # 3. M free-running rollout
    print("可视化 M 的 free-running rollout...")
    free_running, key_indices = visualize_mdn_free_running(vae, mdn, initial_frame, steps=100)
    free_running = add_caption(free_running, f"M 预测退化：步 {key_indices[0]} → {key_indices[-1]}（复合误差）")
    Image.fromarray(free_running).save(args.image_dir / "mdn-free-running.png")

    # 4. 真实评估帧
    print("可视化真实评估...")
    real_eval, total_steps = visualize_real_evaluation(vae, mdn, controller, env, seed=0, max_steps=200)
    real_eval = add_caption(real_eval, f"真实评估：共 {total_steps} 步")
    Image.fromarray(real_eval).save(args.image_dir / "real-evaluation.png")

    # 5. 梦境生成：C 在 M 的想象中开车，全程不碰真实环境
    print("可视化梦境生成（C 在 M 的想象中开车）...")
    dream_gen, dream_steps = visualize_dream_generation(vae, mdn, controller, initial_frame, steps=200)
    dream_gen = add_caption(dream_gen, f"梦境世界：C 在 M 想象中开了 {dream_steps} 步（全程未接触真实环境）")
    Image.fromarray(dream_gen).save(args.image_dir / "dream-generation.png")

    print(f"完成！图片保存在 {args.image_dir}/")


if __name__ == "__main__":
    main()
