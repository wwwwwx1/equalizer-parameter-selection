"""训练、验证、保存和独立测试入口。固定基线和归一化只看训练集。"""
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common import project_path, write_json
from ..data.io import load_candidates, load_manifest
from ..data.dataset import ChannelDataset, collate_channels
from ..models.selector import ParameterSelector
from .losses import selection_loss
from .metrics import evaluate_scores
from ..progress import Progress, event


def choose_device(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def make_loader(rows, config, ids, shuffle=False):
    settings = config["training"]
    return DataLoader(ChannelDataset(rows, config["data"], ids), batch_size=settings["batch_size"],
                      shuffle=shuffle, num_workers=settings["workers"], collate_fn=collate_channels,
                      generator=torch.Generator().manual_seed(settings["seed"]))


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def run_model(model, batch):
    return model(batch["x"], batch["scalars"], batch["padding_mask"])


@torch.inference_mode()
def collect_predictions(model, loader, device):
    model.eval()
    scores, costs, ids, groups = [], [], [], []
    progress=Progress('模型评估',len(loader.dataset))
    progress.update(0,'开始读取评估批次',force=True)
    for batch in loader:
        prediction = run_model(model, to_device(batch, device)).float().cpu().numpy()
        if not np.isfinite(prediction).all():
            raise ValueError("模型输出非有限值，停止评估。")
        scores.append(prediction)
        costs.append(batch["ber"].numpy())
        ids.extend(batch["sample_id"])
        groups.extend(batch["group"])
        progress.update(len(ids),batch['sample_id'][-1])
    return np.concatenate(scores), np.concatenate(costs), ids, groups


def fit_training_statistics(loader, count):
    total, sum_x, sum_x2 = 0, np.zeros(2), np.zeros(2)
    ber_sum = np.zeros(count, dtype=np.float64)
    progress=Progress('扫描训练数据集',len(loader.dataset))
    first=loader.dataset.rows[0]['channel_path'] if len(loader.dataset) else ''
    progress.update(0,f'开始读取：{first}',force=True)
    for batch in loader:
        scalars = batch["scalars"].numpy().astype(np.float64)
        total += len(scalars)
        sum_x += scalars.sum(0)
        sum_x2 += (scalars ** 2).sum(0)
        ber_sum += batch["ber"].numpy().astype(np.float64).sum(0)
        progress.update(total,f"已读至 {batch['sample_id'][-1]}")
    mean = sum_x / total
    std = np.sqrt(np.maximum(sum_x2 / total - mean ** 2, 0)).clip(1e-6)
    return mean, std, int(ber_sum.argmin()), ber_sum / total


def train(config):
    settings = config["training"]
    if settings["epochs"] < 1:
        raise ValueError("epochs 必须 >= 1。")
    torch.set_num_threads(settings.get("cpu_threads", 4))
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    ids, candidates = load_candidates(project_path(config["data"]["candidates"]), config["data"]["num_candidates"])
    rows = load_manifest(project_path(config["data"]["manifest"]))
    subsets = {split: [r for r in rows if r["split"] == split] for split in ("train", "val", "test")}
    all_data = settings.get('mode', 'holdout') == 'all_data'
    if all_data:
        if any(r['split'] != 'train' for r in rows):
            raise ValueError('全量训练请使用全部标为train的独立索引，保留旧划分。')
    elif any(not subset for subset in subsets.values()):
        raise ValueError("train、val、test 都必须非空，且按源信道划分。")
    output = project_path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录非空，请设置新的 training.output：{output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "split_snapshot.json", rows)
    device = choose_device(settings["device"])
    print("正在扫描训练集，拟合 SNR/能量统计并确定固定参数……", flush=True)
    mean, std, fixed, fixed_costs = fit_training_statistics(make_loader(subsets["train"], config, ids), len(ids))
    write_json(output / "fixed_baseline.json", {"candidate_id": int(ids[fixed]), "parameters": candidates[fixed].tolist(),
                                              "training_mean_ber": fixed_costs.tolist(), "source": "train_only"})
    model = ParameterSelector(config, candidates, mean, std).to(device)
    write_json(output / "environment.json", {"torch": str(torch.__version__), "numpy": str(np.__version__),
                                            "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                                            "seed": seed})
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    amp = settings["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_loader = make_loader(subsets["train"], config, ids, True)
    val_loader = make_loader(subsets["train"] if all_data else subsets["val"], config, ids)
    best, best_tie_loss, stale = float("inf"), float("inf"), 0
    with (output / "training_history.csv").open("w", encoding="utf-8", newline="") as file:
        prefix = 'fit' if all_data else 'val'
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", f"{prefix}_selected_ber", f"{prefix}_regret", f"{prefix}_objective",
                                                f"{prefix}_top1", f"{prefix}_top3", "fixed_ber", "grid_best_ber", "learning_rate", "seconds"])
        writer.writeheader()
        for epoch in range(settings["epochs"]):
            model.train()
            start, total_loss, total = time.perf_counter(), 0.0, 0
            progress=Progress('训练批次',len(train_loader.dataset),epoch=epoch+1,epochs=settings['epochs'])
            progress.update(0,f'第{epoch+1}/{settings["epochs"]}轮',force=True)
            for batch in train_loader:
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    scores = run_model(model, batch)
                # 分位数和代价损失使用 float32，避免混合精度影响小 BER 差距。
                loss = selection_loss(scores.float(), batch["ber"].float(), config["loss"], epoch, config["model"]["kind"])
                if not torch.isfinite(loss):
                    raise ValueError("训练损失非有限值，停止保存模型。")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # AMP 初始缩放过大时允许 GradScaler 跳过该步并降低 scale；
                # 不能把这种正常的动态缩放过程误判成不可恢复失败。
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=not amp)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * len(batch["x"])
                total += len(batch["x"])
                progress.update(total,f'第{epoch+1}轮，loss={loss.item():.5f}')
            scheduler.step()
            scores, ber, _, groups = collect_predictions(model, val_loader, device)
            metrics, _ = evaluate_scores(scores, ber, fixed, groups, config["evaluation"]["tie_tolerance"], 0)
            value = metrics["selected_ber_macro"]
            # 小验证集常有多轮相同（甚至全零）BER。用固定定义的验证目标
            # 作第二排序键，避免无条件保留尚未收敛的第1轮；不看测试集。
            val_objective = float(selection_loss(torch.from_numpy(scores), torch.from_numpy(ber), config["loss"],
                                                 config["loss"]["warmup_epochs"], config["model"]["kind"]))
            writer.writerow({"epoch": epoch + 1, "train_loss": total_loss / total, f"{prefix}_selected_ber": value,
                             f"{prefix}_regret": metrics["mean_regret"], f"{prefix}_objective": val_objective,
                             f"{prefix}_top1": metrics['top1_tie_aware_accuracy'], f"{prefix}_top3": metrics['top3_tie_aware_accuracy'],
                             "fixed_ber":metrics['fixed_ber_macro'], "grid_best_ber":metrics['grid_best_ber_macro'],
                             "learning_rate":optimizer.param_groups[0]['lr'],
                             "seconds": time.perf_counter() - start})
            file.flush()
            print(f"epoch {epoch + 1}: loss={total_loss / total:.5f}, {prefix} BER={value:.6g}, regret={metrics['mean_regret']:.6g}", flush=True)
            event('epoch',epoch=epoch+1,epochs=settings['epochs'],loss=total_loss/total,ber=value,scope=prefix)
            if all_data:
                # 不把训练集当验证集选模。按预先设定轮数训练，导出最终轮。
                if epoch + 1 == settings['epochs']:
                    metrics['scope'] = f'{len(rows)}样本全量训练的拟合指标；没有验证/测试集，不能代表新信道性能。'
                    metrics['target_metric'] = settings.get('target_metric', 'MeanDecodedBER')
                    torch.save({'state_dict':model.state_dict(), 'config':config, 'candidates':torch.tensor(candidates),
                                'candidate_ids':torch.tensor(ids), 'fixed_index':fixed, 'epoch':epoch+1,
                                'manifest_snapshot':rows, 'training_mode':'all_data'}, output/'model.pt')
                    write_json(output/'training_fit_metrics.json', metrics)
                    with (output/'training_predictions.csv').open('w',encoding='utf-8-sig',newline='') as predictions:
                        table=csv.writer(predictions)
                        table.writerow(['sample_id','channel_source_id','candidate_id','mu','N1','N2','selected_ber','fixed_ber','grid_best_ber'])
                        selected=scores.argmin(1)
                        for i,k in enumerate(selected):
                            mu,n1,n2=candidates[k]
                            table.writerow([subsets['train'][i]['sample_id'], groups[i], int(ids[k]), float(mu),int(n1),int(n2),
                                            float(ber[i,k]),float(ber[i,fixed]),float(ber[i].min())])
                continue
            tolerance = config["evaluation"]["tie_tolerance"]
            improved = value < best - tolerance or (abs(value-best) <= tolerance and val_objective < best_tie_loss)
            if improved:
                best, best_tie_loss, stale = value, val_objective, 0
                metrics['validation_objective'] = val_objective
                torch.save({"state_dict": model.state_dict(), "config": config, "candidates": torch.tensor(candidates),
                            "candidate_ids": torch.tensor(ids), "fixed_index": fixed, "epoch": epoch + 1,
                            "manifest_snapshot": rows}, output / "best.pt")
                write_json(output / "best_validation_metrics.json", metrics)
            else:
                stale += 1
            if stale >= settings["patience"]:
                break
    filename = 'model.pt' if all_data else 'best.pt'
    print(f"训练完成：{output / filename}；" + ('全量拟合，等待新信道验证。' if all_data else '测试集尚未用于选模型。'), flush=True)
    return output / filename


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ParameterSelector(checkpoint["config"], checkpoint["candidates"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def evaluate(checkpoint_path, output_path):
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("评估输出目录非空，请使用新目录保留历史结果。")
    model, checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get('training_mode') == 'all_data':
        raise ValueError('此模型使用了全部110信道，无保留测试集。请用predict预测新信道，再进行独立均衡仿真。')
    config = checkpoint["config"]
    rows = load_manifest(project_path(config["data"]["manifest"]))
    if rows != checkpoint["manifest_snapshot"]:
        raise ValueError("索引与训练快照不同；不能悄悄更换测试集。新数据评估需单独制定协议。")
    rows = [r for r in rows if r["split"] == "test"]
    device = choose_device(config["training"]["device"])
    model.to(device)
    ids = checkpoint["candidate_ids"].numpy()
    scores, ber, sample_ids, groups = collect_predictions(model, make_loader(rows, config, ids), device)
    metrics, selected = evaluate_scores(scores, ber, checkpoint["fixed_index"], groups,
                                       config["evaluation"]["tie_tolerance"], config["evaluation"]["bootstrap_repeats"],
                                       config["training"]["seed"])
    output_path.mkdir(parents=True, exist_ok=True)
    write_json(output_path / "test_metrics.json", metrics)
    np.savez_compressed(output_path / "test_scores.npz", scores=scores, ber=ber, candidate_ids=ids)
    with (output_path / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "channel_source_id", "candidate_id", "mu", "N1", "N2", "selected_ber", "fixed_ber", "grid_best_ber"])
        for i, index in enumerate(selected):
            parameters = checkpoint["candidates"][index].tolist()
            writer.writerow([sample_ids[i], groups[i], int(ids[index]), parameters[0], int(parameters[1]), int(parameters[2]),
                             float(ber[i, index]), float(ber[i, checkpoint["fixed_index"]]), float(ber[i].min())])
    return metrics
