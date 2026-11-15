import torch
import torch.nn as nn


class DiffusionProcess:
    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cpu",
    ):
        self.timesteps = timesteps
        self.device = torch.device(device)

        self.betas = torch.linspace(
            beta_start,
            beta_end,
            timesteps,
            device=self.device,
            dtype=torch.float32,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(
            1.0 - self.alpha_bars
        )

    def add_noise(self, clean_images, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(clean_images)

        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(
            -1, 1, 1, 1
        )
        sqrt_one_minus_alpha_bar = (
            self.sqrt_one_minus_alpha_bars[timesteps].view(
                -1, 1, 1, 1
            )
        )

        noisy_images = (
            sqrt_alpha_bar * clean_images
            + sqrt_one_minus_alpha_bar * noise
        )
        return noisy_images, noise

    def training_loss(
        self,
        model,
        clean_images,
        degraded_images,
        timesteps=None,
    ):
        batch_size = clean_images.shape[0]

        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.timesteps,
                (batch_size,),
                device=clean_images.device,
            )

        noisy_images, noise = self.add_noise(
            clean_images,
            timesteps,
        )

        predicted_noise = model(
            noisy_images,
            degraded_images,
            timesteps,
        )

        return nn.functional.mse_loss(predicted_noise, noise)
