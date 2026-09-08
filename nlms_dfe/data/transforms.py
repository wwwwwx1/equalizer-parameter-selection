"""明确区分 MATLAB 索引、原始信道与已裁剪信道；不做图像缩放。"""
import numpy as np
import torch


FEATURE_COUNTS = {"ri": 2, "abs": 1, "ri_abs": 3, "ri_abs_phase": 4}


def crop_channel(h, config):
    h = np.asarray(h)
    if h.ndim != 2:
        raise ValueError(f"信道必须是二维矩阵，实际 shape={h.shape}。")
    if config["axis_order"] == "delay_time":
        h = h.T
    elif config["axis_order"] != "time_delay":
        raise ValueError("axis_order 仅支持 time_delay 或 delay_time。")
    crop = config["crop"]
    width = crop["delay_before"] + crop["delay_after"] + 1
    if crop["already_cropped"]:
        if h.shape[1] != width:
            raise ValueError(f"已裁剪信道应有 {width} 个时延点，实际为 {h.shape[1]}。")
    else:
        t0, t1 = crop["time_start_matlab"] - 1, crop["time_stop_matlab"]
        if crop.get('allow_short', False):
            t1 = min(t1, h.shape[0])
        center = crop["main_path_matlab"] - 1
        d0, d1 = center - crop["delay_before"], center + crop["delay_after"] + 1
        if not (0 <= t0 < t1 <= h.shape[0] and 0 <= d0 < d1 <= h.shape[1]):
            raise ValueError(f"裁剪越界：shape={h.shape}, time=[{t0}:{t1}], delay=[{d0}:{d1}]。")
        h = h[t0:t1, d0:d1]
    if h.size == 0 or not np.isfinite(h).all():
        raise ValueError("信道为空或包含 NaN/Inf。")
    return h.astype(np.complex64)


def make_features(h, config):
    h = crop_channel(h, config)
    power = float(np.mean(np.abs(h.astype(np.complex128)) ** 2))
    if power <= 0:
        raise ValueError("全零信道不能用作有效观测。")
    h = h / np.sqrt(power)
    mode = config["features"]
    if mode not in FEATURE_COUNTS:
        raise ValueError(f"未知特征模式：{mode}")
    features = [np.abs(h)] if mode == "abs" else [h.real, h.imag]
    if mode in {"ri_abs", "ri_abs_phase"}:
        features.append(np.abs(h))
    if mode == "ri_abs_phase":
        phase = np.zeros_like(h.real)
        product = h[1:] * h[:-1].conj()
        reliable = (np.abs(h[1:]) ** 2 > config["phase_power_floor"]) & (np.abs(h[:-1]) ** 2 > config["phase_power_floor"])
        phase[1:] = np.where(reliable, np.angle(product), 0.0)
        features.append(phase)
    x = np.stack(features, axis=1)  # [T,C,L]
    block = int(config["temporal_pool"])
    if block < 1:
        raise ValueError("temporal_pool 必须 >= 1。")
    # 对已提取的幅度/相位特征分块，避免先平均复数导致相位抵消。
    # 时间压缩是显式可调近似；pool=1 保留原始时间分辨率。
    pooled = []
    for start in range(0, len(x), block):
        part = x[start:start + block]
        token = part.mean(axis=0)
        if mode == "ri_abs_phase":
            token[-1] = np.arctan2(np.sin(part[:, -1]).mean(0), np.cos(part[:, -1]).mean(0)) / np.pi
        pooled.append(token)
    return torch.from_numpy(np.stack(pooled).astype(np.float32)), float(np.log10(power))
