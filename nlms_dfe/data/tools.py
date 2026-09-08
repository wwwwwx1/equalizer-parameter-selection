"""接入工具：检查文件、创建待填写索引、按来源划分及完整校验。"""
import csv
from pathlib import Path

import h5py
import numpy as np
from scipy.io import whosmat

from ..common import project_path
from .io import load_candidates, load_manifest
from .dataset import ChannelDataset


def inspect_file(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return [{"field": key, "shape": list(data[key].shape), "dtype": str(data[key].dtype)} for key in data.files]
    if h5py.is_hdf5(path):
        fields = []
        with h5py.File(path, "r") as file:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    fields.append({"field": name, "stored_shape": list(obj.shape), "dtype": str(obj.dtype)})
            file.visititems(visit)
        return fields
    return [{"field": key, "shape": list(shape), "dtype": dtype} for key, shape, dtype in whosmat(path)]


def build_index(directory, destination):
    """只扫描路径；来源编号及标签映射需要真实元数据，不能按文件独立性猜测。"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in {".mat", ".npz", ".h5", ".hdf5"})
    if not paths:
        raise ValueError("目录中没有可识别数据文件。")
    with destination.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "channel_source_id", "channel_path", "result_path", "snr_db", "split"])
        for i, path in enumerate(paths):
            writer.writerow([f"sample_{i:06d}", "", str(path.resolve()), "", "", ""])
    return {"files": len(paths), "status": "请补充真实 source_id 和文件映射，再执行 split。"}


def split_index(source, destination, seed=42):
    rows = load_manifest(source, require_split=False)
    if any(r.get("split", "").strip() for r in rows):
        raise ValueError("索引已有划分，拒绝静默重划。请保留原划分或另建待划分索引。")
    groups = sorted({r["channel_source_id"] for r in rows})
    if len(groups) < 3:
        raise ValueError("至少需要三个独立来源组以建立训练/验证/测试集。")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * 0.15))
    n_test = max(1, int(len(groups) * 0.15))
    mapping = {g: "train" for g in groups}
    mapping.update({g: "val" for g in groups[:n_val]})
    mapping.update({g: "test" for g in groups[n_val:n_val + n_test]})
    for row in rows:
        row["split"] = mapping[row["channel_source_id"]]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 输出使用绝对路径，移动索引文件不会改变其指向。
    with destination.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    load_manifest(destination)  # 复核同文件异组等元数据问题。
    return {split: sum(r["split"] == split for r in rows) for split in ("train", "val", "test")}


def audit(config):
    ids, _ = load_candidates(project_path(config["data"]["candidates"]), config["data"]["num_candidates"])
    rows = load_manifest(project_path(config["data"]["manifest"]))
    dataset = ChannelDataset(rows, config["data"], ids)
    shapes, ties, zeros = set(), 0, 0
    for i in range(len(dataset)):
        item = dataset[i]
        shapes.add(tuple(item["x"].shape))
        ber = item["ber"].numpy()
        ties += int(np.sum(ber == ber.min()) > 1)
        zeros += int(np.any(ber == 0))
        if (i + 1) % 1000 == 0:
            print(f"已检查 {i + 1}/{len(dataset)} 个样本", flush=True)
    return {"samples": len(rows), "feature_shapes": [list(s) for s in sorted(shapes)],
            "groups": len({r["channel_source_id"] for r in rows}),
            "split_samples": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
            "samples_with_tied_best": ties, "samples_with_zero_ber": zeros,
            "scope": "仅结构、数值和显式来源编号审计；无法从文件内容自动证明不同来源独立。"}
