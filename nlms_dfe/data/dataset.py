"""逐样本延迟加载，不把十万个信道同时读入内存。"""
import torch
from torch.utils.data import Dataset

from .io import load_observation
from .transforms import make_features


class ChannelDataset(Dataset):
    def __init__(self, rows, config, candidate_ids):
        self.rows = rows
        self.config = config
        self.candidate_ids = candidate_ids
        self.cache = {} if config.get("cache_features", False) else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.cache is not None and index in self.cache:
            return self.cache[index]
        row = self.rows[index]
        try:
            h, snr, ber = load_observation(row, self.config, True, self.candidate_ids)
            x, log_power = make_features(h, self.config)
        except Exception as error:
            raise ValueError(f"样本 {row['sample_id']}：{error}") from error
        item = {"x": x, "scalars": torch.tensor([snr, log_power]), "ber": torch.from_numpy(ber),
                "sample_id": row["sample_id"], "group": row["channel_source_id"]}
        if self.cache is not None:
            self.cache[index] = item
        return item


def collate_channels(items):
    lengths = [len(item["x"]) for item in items]
    channels, delays = items[0]["x"].shape[1:]
    x = torch.zeros(len(items), max(lengths), channels, delays)
    mask = torch.ones(len(items), max(lengths), dtype=torch.bool)
    for i, item in enumerate(items):
        if item["x"].shape[1:] != (channels, delays):
            raise ValueError("批次中的时延长度不同，请先统一物理裁剪窗口。")
        x[i, :lengths[i]] = item["x"]
        mask[i, :lengths[i]] = False
    result = {"x": x, "padding_mask": mask,
              "scalars": torch.stack([item["scalars"] for item in items])}
    if "ber" in items[0]:
        result["ber"] = torch.stack([item["ber"] for item in items])
    for key in ("sample_id", "group"):
        if key in items[0]:
            result[key] = [item[key] for item in items]
    return result
