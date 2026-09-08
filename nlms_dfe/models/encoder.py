"""时延卷积提取局部多径结构，时间模块建模观测窗口内的变化。"""
import math

import torch
from torch import nn


class ResidualDelayBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(4, width),
                                  nn.GELU(), nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(4, width))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.body(x))


class DelayCNN(nn.Module):
    def __init__(self, input_channels, d_model, delay_bins):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, 5, padding=2, stride=2), nn.GroupNorm(4, 32), nn.GELU(),
            ResidualDelayBlock(32),
            nn.Conv1d(32, 64, 5, padding=2, stride=2), nn.GroupNorm(4, 64), nn.GELU(),
            ResidualDelayBlock(64),
            nn.Conv1d(64, 96, 3, padding=1, stride=2), nn.GroupNorm(4, 96), nn.GELU(),
            ResidualDelayBlock(96), nn.AdaptiveAvgPool1d(delay_bins))
        # 保留有序时延分箱，避免全局均值抹掉主径前后能量位置。
        self.project = nn.Linear(96 * delay_bins, d_model)

    def forward(self, x):
        return self.project(self.conv(x).flatten(1))


class ChannelEncoder(nn.Module):
    def __init__(self, input_channels, config):
        super().__init__()
        width = config["d_model"]
        if width % 2 or width % config["heads"]:
            raise ValueError("d_model 必须是偶数且能被 heads 整除。")
        self.delay = DelayCNN(input_channels, width, config["delay_bins"])
        # 分别构造各层，避免复制同一层初始化值。
        self.temporal = nn.ModuleList([
            nn.TransformerEncoderLayer(width, config["heads"], config["ffn_dim"],
                                       config["dropout"], activation="gelu", batch_first=True, norm_first=True)
            for _ in range(config["layers"])])
        self.norm = nn.LayerNorm(width)
        self.attention = nn.Linear(width, 1)

    def forward(self, x, padding_mask):
        batch, time, channels, delays = x.shape
        z = self.delay(x.reshape(batch * time, channels, delays)).reshape(batch, time, -1)
        width = z.shape[-1]
        positions = torch.arange(time, device=z.device, dtype=torch.float32)[:, None]
        rates = torch.exp(torch.arange(0, width, 2, device=z.device) * (-math.log(10000.0) / width))
        pe = torch.zeros(time, width, device=z.device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * rates), torch.cos(positions * rates)
        z = z + pe.to(z.dtype)
        for layer in self.temporal:
            z = layer(z, src_key_padding_mask=padding_mask)
        z = self.norm(z)
        weights = self.attention(z).squeeze(-1).masked_fill(padding_mask, -torch.inf).softmax(dim=1)
        return (z * weights.unsqueeze(-1)).sum(1)
