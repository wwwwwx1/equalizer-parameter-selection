"""输入只有信道及 SNR；BER 永远只出现在训练目标和离线评估中。"""
import torch
from torch import nn

from .encoder import ChannelEncoder
from ..data.transforms import FEATURE_COUNTS


class ParameterSelector(nn.Module):
    def __init__(self, config, candidates, scalar_mean=None, scalar_std=None):
        super().__init__()
        model = config["model"]
        width, dim = model["d_model"], model["candidate_dim"]
        candidates = torch.as_tensor(candidates, dtype=torch.float32).clone()
        encoded = candidates.clone()
        encoded[:, 0] = encoded[:, 0].log10()
        encoded = (encoded - encoded.mean(0)) / encoded.std(0, unbiased=False).clamp_min(1e-6)
        self.register_buffer("candidate_values", candidates)
        self.register_buffer("candidate_features", encoded)
        self.register_buffer("scalar_mean", torch.zeros(2) if scalar_mean is None else torch.as_tensor(scalar_mean).float())
        self.register_buffer("scalar_std", torch.ones(2) if scalar_std is None else torch.as_tensor(scalar_std).float().clamp_min(1e-6))
        self.channel = ChannelEncoder(FEATURE_COUNTS[config["data"]["features"]], model)
        self.scalars = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 64), nn.GELU())
        self.conditioning = model["snr_conditioning"]
        if self.conditioning not in {"film", "concat"}:
            raise ValueError("snr_conditioning 必须为 film 或 concat。")
        self.film = nn.Linear(64, width * 2)
        self.kind = model["kind"]
        if self.kind == "scorer":
            self.candidate_encoder = nn.Sequential(nn.Linear(3, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, dim))
            self.scorer = nn.Sequential(nn.Linear(width + 64 + dim, 128), nn.GELU(),
                                        nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        elif self.kind == "classifier":
            self.classifier = nn.Linear(width + 64, len(candidates))
        else:
            raise ValueError("model.kind 必须为 scorer 或 classifier。")

    def forward(self, x, scalars, padding_mask):
        z = self.channel(x, padding_mask)
        condition = self.scalars((scalars - self.scalar_mean) / self.scalar_std)
        if self.conditioning == "film":
            gamma, beta = self.film(condition).chunk(2, dim=-1)
            z = z * (1 + gamma) + beta
        context = torch.cat([z, condition], dim=-1)
        if self.kind == "classifier":
            return -self.classifier(context)  # 统一为分数越小越好。
        candidates = self.candidate_encoder(self.candidate_features)
        batch, count = len(x), len(candidates)
        features = torch.cat([context[:, None].expand(-1, count, -1),
                              candidates[None].expand(batch, -1, -1)], dim=-1)
        return self.scorer(features).squeeze(-1)
