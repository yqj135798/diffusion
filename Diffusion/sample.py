import torch
from torchvision import transforms
from model.unet import DDPMUNet
from model.ddpm import DDPM
from tqdm.auto import tqdm
from utils.img_utils import show_images, save_images
from PIL import Image
import os


@torch.no_grad()
def sample_ddpm_full(num_samples: int, ddpm: DDPM, device: torch.device):
    """原始 DDPM 完整步数采样（质量最高）"""
    ddpm.eval()
    x_t = torch.randn(num_samples, 1, 28, 28, device=device)

    # 从 99 走到 0，一步一步走，不跳步！
    for t in tqdm(range(ddpm.timesteps - 1, -1, -1), desc="Full DDPM Sampling"):
        if t > 0:
            z = torch.randn_like(x_t)
        else:
            z = torch.zeros_like(x_t)
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        x_t = ddpm.p_sample(x_t, t_tensor, z)

    # 映射到 [0, 1]
    x_t = (x_t + 1) / 2
    x_t = x_t.clamp(0, 1)

    transform = transforms.ToPILImage()
    img_list = [transform(x_t[i].cpu()) for i in range(num_samples)]
    return img_list


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 必须与训练时的配置完全一致
    model_channels = 64
    channel_mults = [1, 2, 2, 2]
    timesteps = 400

    ddpm_unet = DDPMUNet(
        img_channels=1,
        model_channels=model_channels,
        channel_mults=channel_mults
    )
    ddpm = DDPM(timesteps, ddpm_unet)
    ddpm.to(device)

    # 加载训练好的模型（如果继续训练了，就加载最新的）
    model_path = "model_ckpts/ddpm.pth"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        ddpm.load_state_dict(state_dict)
        print(f"模型加载成功：{model_path}")
    else:
        raise FileNotFoundError("模型文件不存在！")

    # 使用完整步数采样（质量最佳）
    img_list = sample_ddpm_full(9, ddpm, device)

    show_images(img_list)
    save_images(img_list, "generated_samples_full.png")
    print("图片已保存为 generated_samples_full.png")