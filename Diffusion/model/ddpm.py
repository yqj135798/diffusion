import torch
from torch import nn


class DDPM(nn.Module):
    def __init__(
            self,
            timesteps: int,
            eps_model: nn.Module,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.eps_model = eps_model

        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.eps_model(x_t, t)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        eps_theta = self.eps_model(x_t, t)
        alpha_t = self.alpha[t].reshape(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        beta_t = self.beta[t].reshape(-1, 1, 1, 1)

        mean = 1.0 / torch.sqrt(alpha_t) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta)
        return mean + torch.sqrt(beta_t) * z

    def ddim_sample(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            t_next: torch.Tensor,
            eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM 单步采样（从 t 到 t_next）

        Args:
            x_t: 当前时刻的样本 [batch, channels, h, w]
            t: 当前时间步
            t_next: 下一个时间步（小于 t）
            eta: 随机性控制，0 为确定性 DDIM，1 为原始 DDPM

        Returns:
            x_next: t_next 时刻的样本
        """
        # 预测噪声
        eps_theta = self.eps_model(x_t, t)

        # 获取对应时刻的累积乘积
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        alpha_bar_t_next = self.alpha_bar[t_next].reshape(-1, 1, 1, 1)

        # 计算 x_0 的估计
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)
        x0_pred = x0_pred.clamp(-1, 1)  # 可选：限制范围以稳定生成

        # 指向 x_{t-1} 的方向
        dir_xt = torch.sqrt(1 - alpha_bar_t_next - eta ** 2 * (1 - alpha_bar_t_next) / (1 - alpha_bar_t)) * eps_theta

        # 如果 eta > 0，添加随机噪声
        if eta > 0:
            noise = torch.randn_like(x_t)
            dir_xt = dir_xt + eta * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t)) * noise

        # DDIM 更新公式
        x_next = torch.sqrt(alpha_bar_t_next) * x0_pred + dir_xt

        return x_next

    def ddim_sample_loop(
            self,
            shape: tuple,
            device: torch.device,
            num_steps: int = 50,
            eta: float = 0.0,
            progress: bool = True,
    ) -> torch.Tensor:
        """
        完整的 DDIM 采样循环

        Args:
            shape: 生成图像的形状 (batch, channels, h, w)
            device: 计算设备
            num_steps: 采样步数（越少越快，50-100 效果不错）
            eta: 随机性控制
            progress: 是否显示进度条

        Returns:
            生成的图像张量（值域 [-1, 1]）
        """
        batch_size = shape[0]

        # 从纯噪声开始
        x_t = torch.randn(shape, device=device)

        # 生成等间隔的时间步（从 T-1 到 0）
        times = torch.linspace(
            self.timesteps - 1, 0, num_steps + 1,
            dtype=torch.long, device=device
        )
        time_pairs = list(zip(times[:-1], times[1:]))

        # 如果需要进度条
        if progress:
            from tqdm.auto import tqdm
            time_pairs = tqdm(time_pairs, desc="DDIM Sampling")

        for t, t_next in time_pairs:
            t_batch = torch.full((batch_size,), t, device=device)
            t_next_batch = torch.full((batch_size,), t_next, device=device)
            x_t = self.ddim_sample(x_t, t_batch, t_next_batch, eta)

        return x_t