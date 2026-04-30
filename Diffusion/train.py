import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

from model.unet import DDPMUNet
from model.ddpm import DDPM
from datasets.dataset import get_dataset
from utils.img_utils import save_images
from PIL import Image
from torchvision import transforms


@torch.no_grad()
def test_generation(ddpm, device, epoch):
    ddpm.eval()
    x_t = torch.randn(4, 1, 28, 28, device=device)
    for t in tqdm(range(ddpm.timesteps - 1, -1, -1), desc=f"Testing Epoch {epoch}", leave=False):
        if t > 0:
            z = torch.randn_like(x_t)
        else:
            z = torch.zeros_like(x_t)
        t_tensor = torch.full((4,), t, device=device, dtype=torch.long)
        x_t = ddpm.p_sample(x_t, t_tensor, z)

    x_t = (x_t + 1) / 2
    x_t = x_t.clamp(0, 1)
    transform = transforms.ToPILImage()
    img_list = [transform(x_t[i].cpu()) for i in range(4)]
    save_images(img_list, f"check_epoch_{epoch}.png")
    ddpm.train()


def train(epochs: int, dataloader: DataLoader, ddpm: DDPM, device: torch.device):
    ddpm.train()
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0

        for x_0, _ in pbar:
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]

            t = torch.randint(0, ddpm.timesteps, (batch_size,), device=device)
            eps = torch.randn_like(x_0)

            x_t = ddpm.q_sample(x_0, t, eps)
            eps_theta = ddpm(x_t, t)
            loss = loss_fn(eps_theta, eps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} finished, avg_loss: {avg_loss:.6f}")

        if (epoch + 1) % 5 == 0:
            os.makedirs("model_ckpts", exist_ok=True)
            torch.save(ddpm.state_dict(), f"model_ckpts/ddpm_epoch{epoch + 1}.pth")
            test_generation(ddpm, device, epoch + 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---------- 新配置（画质更好） ----------
    model_channels = 64
    channel_mults = [1, 2, 2, 2]
    timesteps = 400
    batch_size = 64  # 保守一点，CPU 上 64 较稳妥
    epochs = 50
    # --------------------------------------

    dataset = get_dataset("mnist")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if device.type == 'cpu' else 2
    )

    ddpm_unet = DDPMUNet(
        img_channels=1,
        model_channels=model_channels,
        channel_mults=channel_mults
    )
    ddpm = DDPM(timesteps, ddpm_unet)
    ddpm.to(device)

    print(f"模型参数量: {sum(p.numel() for p in ddpm.parameters()):,}")
    print(f"时间步数: {timesteps}")

    # ---------- 重要：不加载旧模型，从头训练 ----------
    print("从头开始训练新配置模型（不加载旧权重）...")
    # -------------------------------------------------

    train(epochs, dataloader, ddpm, device)

    os.makedirs("model_ckpts", exist_ok=True)
    torch.save(ddpm.state_dict(), "model_ckpts/ddpm.pth")
    print("✅ 训练完成！")