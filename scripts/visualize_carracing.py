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


def capture_environment_frame(env, seed=0):
    """捕获环境初始帧。"""
    obs, _ = env.reset(seed=seed)
    return obs


def visualize_vae_reconstruction(vae, frame):
    """可视化 VAE 重建：原图 vs 重建。"""
    # Resize frame to 64x64 as expected by VAE
    frame_64 = resize_frame(frame)
    with torch.no_grad():
        frame_tensor = torch.from_numpy(frame_64).unsqueeze(0).float()
        mean, logvar = vae.encode(frame_tensor)
        z = vae.reparameterize(mean, logvar)
        reconstruction = vae.decode(z).squeeze(0).permute(1, 2, 0).numpy()

    # Resize original frame to 64x64 for display
    frame_display = frame_64
    # Concatenate original and reconstruction side by side
    comparison = np.concatenate([frame_display, reconstruction], axis=1)
    return (comparison * 255).astype(np.uint8)


def visualize_mdn_free_running(vae, mdn, initial_frame, steps=100):
    """可视化 M 的 free-running rollout：看复合误差如何累积。"""
    # Resize initial frame to 64x64
    frame_64 = resize_frame(initial_frame)
    frames = [(frame_64 * 255).astype(np.uint8)]

    with torch.no_grad():
        frame_tensor = torch.from_numpy(frame_64).unsqueeze(0).float()
        mean, logvar = vae.encode(frame_tensor)
        z = vae.reparameterize(mean, logvar)
        hidden = torch.zeros(1, mdn.hidden_size)

        for _ in range(steps):
            action = torch.zeros(1, 3)  # 零动作，只看预测稳定性
            z, _, hidden = mdn.sample(z, action, hidden, temperature=1.0)
            # 从 z 解码回像素空间
            reconstruction = vae.decode(z).squeeze(0).permute(1, 2, 0).numpy()
            frames.append((reconstruction * 255).astype(np.uint8))

    # 选取关键帧：第 0、10、30、60、99 步
    key_indices = [0, 10, 30, 60, 99]
    key_frames = [frames[i] for i in key_indices]

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
            frame = torch.from_numpy(resize_frame(obs)).unsqueeze(0)
            mean, logvar = vae.encode(frame)
            z = vae.reparameterize(mean, logvar)

            features = torch.cat((z, hidden), dim=-1).numpy()
            action = controller.act(features, controller.parameters, noise=0.0)
            action_tensor = torch.from_numpy(action).float()

            frames.append(obs.copy())

            obs, reward, terminated, truncated, _ = env.step(action.reshape(-1))
            z, _, hidden = mdn.sample(z, action_tensor, hidden, temperature=1.0)

            if terminated or truncated:
                break

    # 选取关键帧：均匀分布
    if len(frames) >= 5:
        indices = np.linspace(0, len(frames) - 1, 5, dtype=int)
        key_frames = [frames[i] for i in indices]
    else:
        key_frames = frames

    comparison = np.concatenate(key_frames, axis=1)
    return comparison, len(frames)


def add_caption(image, text, height=30):
    """给图片加标题。"""
    new_image = Image.new("RGB", (image.shape[1], image.shape[0] + height), (255, 255, 255))
    new_image.paste(Image.fromarray(image), (0, 0))

    draw = ImageDraw.Draw(new_image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 16)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    x = (new_image.width - text_width) // 2
    draw.text((x, 5), text, fill=(0, 0, 0), font=font)

    return np.array(new_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/carracing-world-model"))
    parser.add_argument("--image-dir", type=Path, default=Path("docs/chapters/03-decision-and-planning/images"))
    args = parser.parse_args()

    args.image_dir.mkdir(parents=True, exist_ok=True)

    print("加载模型...")
    vae, mdn, controller = load_models(args.output_dir)
    env = make_env()

    # 1. 环境初始帧
    print("捕获环境初始帧...")
    initial_frame = capture_environment_frame(env, seed=0)
    Image.fromarray(initial_frame).save(args.image_dir / "carracing-initial.png")

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

    print(f"完成！图片保存在 {args.image_dir}/")


if __name__ == "__main__":
    main()
