"""读取用户数据，不猜测 SNR、参数表、BER 或矩阵轴的含义。"""
import csv
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


def read_field(path, key, matlab_order=True):
    """支持 NPZ、传统 MAT、数值型 v7.3 MAT/HDF5；结构体用点路径。"""
    path = Path(path)
    if not key:
        raise ValueError("字段映射不能为空，请修改 configs/default.json。")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data[key])
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as data:
            node = data[key.replace(".", "/")]
            if not isinstance(node, h5py.Dataset):
                raise ValueError(f"{key} 不是数值 Dataset，请使用自定义适配器。")
            value = node[()]
            if h5py.check_dtype(ref=node.dtype) is not None:
                raise ValueError("暂不自动解析 v7.3 cell/对象引用，请在适配器中显式映射。")
            if value.dtype.names and {"real", "imag"} <= set(value.dtype.names):
                value = value["real"] + 1j * value["imag"]
            # MATLAB 的 v7.3 存储维序与逻辑维序相反；普通 HDF5 可关闭。
            return value.transpose() if matlab_order and value.ndim > 1 else value
    value = loadmat(path, simplify_cells=True, variable_names=[key.split(".")[0]])
    for part in key.split("."):
        value = value[part]
    return np.asarray(value)


def load_candidates(path, expected=294):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != expected:
        raise ValueError(f"需要真实的 {expected} 组候选参数，当前只有 {len(rows)} 行；不自动生成网格。")
    ids = np.array([int(row["candidate_id"]) for row in rows], dtype=np.int64)
    values = np.array([[float(row[k]) for k in ("mu", "N1", "N2")] for row in rows])
    if len(set(ids)) != expected or len(set(map(tuple, values))) != expected:
        raise ValueError("候选编号或参数组合重复。")
    if not np.isfinite(values).all() or np.any(values[:, 0] <= 0):
        raise ValueError("mu 必须是有限正数。")
    if np.any(values[:, 1:] != np.floor(values[:, 1:])) or np.any(values[:, 1] < 1) or np.any(values[:, 2] < 0):
        raise ValueError("N1 必须为正整数，N2 必须为非负整数。")
    return ids, values.astype(np.float32)


def load_manifest(path, require_split=True):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("数据索引为空；请先接入真实信道、SNR 和 BER 数据。")
    seen, paths, groups = set(), {}, {}
    for row in rows:
        for key in ("sample_id", "channel_source_id", "channel_path"):
            if not row.get(key, "").strip():
                raise ValueError(f"索引缺少 {key}，不能推测数据独立性。")
        if row["sample_id"] in seen:
            raise ValueError("sample_id 重复。")
        seen.add(row["sample_id"])
        for key in ("channel_path", "result_path"):
            if row.get(key):
                p = Path(row[key])
                row[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
        if require_split:
            split = row.get("split")
            if split not in {"train", "val", "test"}:
                raise ValueError("split 必须为 train / val / test。")
            group = row["channel_source_id"]
            if group in groups and groups[group] != split:
                raise ValueError(f"同源信道跨集合泄漏：{group}")
            groups[group] = split
            channel_path = str(Path(row["channel_path"])).casefold()
            if channel_path in paths and paths[channel_path] != split:
                raise ValueError("同一信道文件跨集合泄漏。")
            paths[channel_path] = split
    return rows


def load_observation(row, config, need_ber=False, candidate_ids=None):
    """后续特殊文件结构只需替换此函数，模型和训练代码无需改动。"""
    fields = config["fields"]
    order = config.get("hdf5_matlab_order", True)
    h = read_field(row["channel_path"], fields["channel"], order)
    result = row.get("result_path") or row["channel_path"]
    raw_snr = row.get("snr_db")
    if raw_snr is None or str(raw_snr).strip() == "":
        try:
            raw_snr = read_field(result, fields["snr"], order)
        except KeyError as error:
            raise ValueError("缺少真实 SNR：请填索引 snr_db 或映射结果文件中的字段。") from error
    raw_snr = np.asarray(raw_snr)
    if raw_snr.size != 1:
        raise ValueError("每个样本必须有一个 SNR 标量。")
    snr = float(raw_snr.item())
    if not np.isfinite(snr):
        raise ValueError("SNR 不能是 NaN 或 Inf。")
    if not need_ber:
        return h, snr, None
    ber = np.asarray(read_field(result, fields["ber"], order)).reshape(-1)
    ids_raw = np.asarray(read_field(result, fields["candidate_ids"], order)).reshape(-1)
    if np.iscomplexobj(ber) or np.iscomplexobj(ids_raw):
        raise ValueError("BER 和候选编号必须为实数。")
    if not np.isfinite(ids_raw).all() or np.any(ids_raw != np.floor(ids_raw)):
        raise ValueError("结果中的 candidate_ids 必须为整数。")
    ids = ids_raw.astype(np.int64)
    if len(ids) != len(ber) or len(set(ids)) != len(ids) or set(ids) != set(candidate_ids):
        raise ValueError("BER 的候选编号与参数表不一致；必须显式保存对应顺序。")
    if not np.isfinite(ber).all() or np.any(ber < 0) or np.any(ber > 1):
        raise ValueError("BER 必须是 [0,1] 的完整有限向量，不能填补缺失值。")
    lookup = {int(k): i for i, k in enumerate(ids)}
    ber = np.array([ber[lookup[int(k)]] for k in candidate_ids], dtype=np.float32)
    return h, snr, ber
