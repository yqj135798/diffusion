import math
import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, time_channels: int):
        super().__init__()
        self.time_channels = time_channels

        self.linear_1 = nn.Linear(time_channels // 4, time_channels)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_channels, time_channels)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.time_channels // 8
        if half_dim == 0:
            half_dim = 1
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = self.linear_1(emb)
        emb = self.act(emb)
        emb = self.linear_2(emb)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()

        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_channels, out_channels)

        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        h = h + self.time_proj(t)[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, time_channels)
        self.res2 = ResBlock(out_channels, out_channels, time_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.res2(x, t)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        concat_channels = out_channels + in_channels  # ← 修正点：拼接后为 out + in
        self.res1 = ResBlock(concat_channels, out_channels, time_channels)
        self.res2 = ResBlock(out_channels, out_channels, time_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor):
        x = self.upsample(x)
        # 尺寸对齐
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t)
        x = self.res2(x, t)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResBlock(channels, channels, time_channels)
        self.res2 = ResBlock(channels, channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.res2(x, t)
        return x


class DDPMUNet(nn.Module):
    def __init__(
            self,
            img_channels: int = 1,
            model_channels: int = 64,
            channel_mults: list = None,
    ):
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 2, 2, 2]

        self.time_channels = model_channels * 4
        self.time_embedding = TimeEmbedding(self.time_channels)

        self.init_conv = nn.Conv2d(img_channels, model_channels, 3, padding=1)

        # 计算各层通道数
        channels = [model_channels]
        for mult in channel_mults:
            channels.append(model_channels * mult)

        # 下采样
        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(DownBlock(channels[i], channels[i + 1], self.time_channels))

        # 中间层
        self.middle = MiddleBlock(channels[-1], self.time_channels)

        # 上采样
        self.ups = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.ups.append(UpBlock(channels[i], channels[i - 1], self.time_channels))

        # 输出层
        self.out_norm = nn.GroupNorm(min(8, model_channels), model_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(model_channels, img_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = self.time_embedding(t)

        x = self.init_conv(x)

        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)

        x = self.middle(x, t_emb)

        skips = skips[::-1]  # 反转以匹配从深到浅的顺序

        for i, up in enumerate(self.ups):
            x = up(x, skips[i], t_emb)

        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)

        return x