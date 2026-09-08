"""配置和结果文件的小工具；所有相对路径以项目目录为基准。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_config(path):
    with project_path(path).open(encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def active_model():
    """界面和命令行共用同一模型入口；历史模型不会被覆盖。"""
    path = ROOT/'configs/inference.json'
    if path.exists():
        return read_config(path)
    return {'checkpoint':'outputs/new_training/model.pt','target_metric':'MeanCodedBER','label':'请先训练新模型'}
