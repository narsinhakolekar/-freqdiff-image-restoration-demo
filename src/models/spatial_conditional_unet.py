
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dimension):
    half = dimension // 2

    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(
            half,
            device=timesteps.device,
            dtype=torch.float32,
        )
        / max(half - 1, 1)
    )

    angles = timesteps.float()[:, None] * frequencies[None, :]

    embedding = torch.cat(
        [torch.sin(angles), torch.cos(angles)],
        dim=1,
    )

    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))

    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        self.time_projection = nn.Linear(time_dim, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, time_embedding):
        residual = self.skip(x)

        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(
            F.silu(time_embedding)
        )[:, :, None, None]

        h = self.conv2(F.silu(self.norm2(h)))

        return h + residual


class SpatialConditionalUNet(nn.Module):
    def __init__(self, base_channels=32, time_dim=128):
        super().__init__()

        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.input_conv = nn.Conv2d(
            6, base_channels, kernel_size=3, padding=1
        )

        self.down1 = ResidualBlock(
            base_channels, base_channels, time_dim
        )
        self.downsample1 = nn.Conv2d(
            base_channels, base_channels * 2,
            4, stride=2, padding=1
        )

        self.down2 = ResidualBlock(
            base_channels * 2, base_channels * 2, time_dim
        )
        self.downsample2 = nn.Conv2d(
            base_channels * 2, base_channels * 4,
            4, stride=2, padding=1
        )

        self.middle1 = ResidualBlock(
            base_channels * 4, base_channels * 4, time_dim
        )
        self.middle2 = ResidualBlock(
            base_channels * 4, base_channels * 4, time_dim
        )

        self.upsample2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2,
            4, stride=2, padding=1
        )
        self.up2 = ResidualBlock(
            base_channels * 4, base_channels * 2, time_dim
        )

        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels,
            4, stride=2, padding=1
        )
        self.up1 = ResidualBlock(
            base_channels * 2, base_channels, time_dim
        )

        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv2d(
            base_channels, 3, kernel_size=3, padding=1
        )

    def forward(self, noisy_image, degraded_image, timesteps):
        if noisy_image.shape != degraded_image.shape:
            raise ValueError(
                "Noisy and degraded images must have the same shape."
            )

        time_embedding = timestep_embedding(
            timesteps, self.time_dim
        )
        time_embedding = self.time_mlp(time_embedding)

        x = torch.cat([noisy_image, degraded_image], dim=1)
        x = self.input_conv(x)

        skip1 = self.down1(x, time_embedding)
        x = self.downsample1(skip1)

        skip2 = self.down2(x, time_embedding)
        x = self.downsample2(skip2)

        x = self.middle1(x, time_embedding)
        x = self.middle2(x, time_embedding)

        x = self.upsample2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x, time_embedding)

        x = self.upsample1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up1(x, time_embedding)

        return self.output_conv(F.silu(self.output_norm(x)))