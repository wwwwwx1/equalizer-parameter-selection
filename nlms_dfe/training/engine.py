"""训练、验证、保存和独立测试入口。固定基线和归一化只看训练集。"""
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common import project_path, write_json
from ..data.io import load_candidates, load_manifest, load_observation
from ..data.dataset import ChannelDataset, collate_channels
from ..data.transforms import crop_channel
from ..models.selector import ParameterSelector
from .control import check_control
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
        check_control()
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
        check_control()
        scalars = batch["scalars"].numpy().astype(np.float64)
        total += len(scalars)
        sum_x += scalars.sum(0)
        sum_x2 += (scalars ** 2).sum(0)
        ber_sum += batch["ber"].numpy().astype(np.float64).sum(0)
        progress.update(total,f"已读至 {batch['sample_id'][-1]}")
    mean = sum_x / total
    std = np.sqrt(np.maximum(sum_x2 / total - mean ** 2, 0)).clip(1e-6)
    return mean, std, int(ber_sum.argmin()), ber_sum / total


def _source_fingerprint(value):
    """用文件大小和修改时间识别原始数据是否被替换，避免读取 MAT 内容求哈希。"""
    path = project_path(value)
    try:
        stat = path.stat()
        return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        # 后续真正读取样本时仍会给出具体缺失文件错误；这里不把缓存键构造变成新错误。
        return {"path": str(path), "missing": True}


def training_statistics_cache_key(rows, candidates, ids, data_config):
    """只要训练来源、候选表或信道预处理改变，就生成新缓存键。"""
    sources = []
    for row in rows:
        sources.append({
            "sample_id": row["sample_id"],
            "channel_source_id": row["channel_source_id"],
            "snr_db": row["snr_db"],
            "channel": _source_fingerprint(row["channel_path"]),
            "result": _source_fingerprint(row["result_path"]),
        })
    payload = {
        "version": 1,
        "train_rows": sources,
        "candidate_ids": np.asarray(ids).tolist(),
        "candidates": np.asarray(candidates).tolist(),
        # 标量统计使用裁剪后的信道功率；完整保留 data 配置使缓存规则更容易理解和审计。
        "data": data_config,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cached_training_statistics(rows, config, ids, candidates):
    """复用跨运行的归一化/固定基线统计；未命中时只扫描一次并写入磁盘。"""
    data_config = config["data"]
    settings = config["training"]
    if not data_config.get("reuse_training_statistics", True):
        return None
    key = training_statistics_cache_key(rows, candidates, ids, data_config)
    folder = project_path(data_config.get("training_statistics_cache_dir", "cache/training_statistics"))
    path = folder / f"{key}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == key and len(cached["fixed_costs"]) == len(ids):
                print(f"复用训练统计缓存：{path.name[:12]}…；跳过训练集预扫描。", flush=True)
                return (np.asarray(cached["mean"], dtype=np.float64),
                        np.asarray(cached["std"], dtype=np.float64),
                        int(cached["fixed_index"]),
                        np.asarray(cached["fixed_costs"], dtype=np.float64))
        except (OSError, ValueError, KeyError, TypeError):
            print("训练统计缓存无法读取，将重新扫描训练集。", flush=True)

    print("正在首次扫描训练集，拟合 SNR/能量统计并确定固定参数……", flush=True)
    # 统计阶段只需裁剪后的能量、SNR 和 BER；不生成四通道特征，减少首次扫描的 CPU 开销。
    total, sum_x, sum_x2 = 0, np.zeros(2), np.zeros(2)
    ber_sum = np.zeros(len(ids), dtype=np.float64)
    progress = Progress("扫描训练数据集", len(rows))
    progress.update(0, f"开始读取：{rows[0]['channel_path']}", force=True)
    for number, row in enumerate(rows, 1):
        h, snr, ber = load_observation(row, data_config, True, ids)
        cropped = crop_channel(h, data_config)
        power = float(np.mean(np.abs(cropped.astype(np.complex128)) ** 2))
        if power <= 0:
            raise ValueError(f"样本 {row['sample_id']}：全零信道不能用作有效观测。")
        scalars = np.asarray([snr, np.log10(power)], dtype=np.float64)
        total += 1
        sum_x += scalars
        sum_x2 += scalars ** 2
        ber_sum += np.asarray(ber, dtype=np.float64)
        progress.update(number, f"已读至 {row['sample_id']}")
    mean = sum_x / total
    std = np.sqrt(np.maximum(sum_x2 / total - mean ** 2, 0)).clip(1e-6)
    fixed_costs = ber_sum / total
    fixed = int(fixed_costs.argmin())
    folder.mkdir(parents=True, exist_ok=True)
    value = {"cache_key": key, "mean": mean.tolist(), "std": std.tolist(), "fixed_index": fixed,
             "fixed_costs": fixed_costs.tolist(), "train_samples": len(rows)}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    print(f"已写入训练统计缓存：{path.name[:12]}…；以后相同训练集可跳过扫描。", flush=True)
    return mean, std, fixed, fixed_costs


def _train(config):
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
    if settings.get('resume'):
        saved=torch.load(project_path(settings['resume']),map_location='cpu',weights_only=True)
        if saved.get('resume_version')!=1:raise ValueError('请选择新版last.pt；该文件没有完整续训状态。')
        mean=saved['state_dict']['scalar_mean'].numpy();std=saved['state_dict']['scalar_std'].numpy();fixed=saved['fixed_index']
        if not np.array_equal(saved['candidate_ids'].numpy(),ids) or not np.array_equal(saved['candidates'].numpy(),candidates):raise ValueError('续训候选参数表已改变。')
        print(f"已读取断点：完成第{saved['epoch']}轮，将从第{saved['epoch']+1}/{settings['epochs']}轮继续；无需重复扫描归一化统计。",flush=True)
        import shutil
        baseline=project_path(settings['resume']).parent/'fixed_baseline.json'
        if baseline.exists():shutil.copy2(baseline,output/'fixed_baseline.json')
    else:
        mean, std, fixed, fixed_costs = cached_training_statistics(subsets["train"], config, ids, candidates)
        if mean is None:
            print("正在扫描训练集，拟合 SNR/能量统计并确定固定参数……", flush=True)
            mean, std, fixed, fixed_costs = fit_training_statistics(make_loader(subsets["train"], config, ids), len(ids))
        write_json(output / "fixed_baseline.json", {"candidate_id": int(ids[fixed]), "parameters": candidates[fixed].tolist(),
                                                  "training_mean_ber": fixed_costs.tolist(), "source": "train_only",
                                                  "statistics_cache": bool(config["data"].get("reuse_training_statistics", True))})
    model = ParameterSelector(config, candidates, mean, std).to(device)
    write_json(output / "environment.json", {"torch": str(torch.__version__), "numpy": str(np.__version__),
                                            "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                                            "seed": seed})
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    # Old checkpoints were trained with cosine decay.  Preserve that choice
    # when they are resumed; new configurations explicitly select plateau.
    schedule = settings.get("lr_schedule", "cosine" if settings.get("resume") else "plateau")
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    elif schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=settings.get("lr_factor", .5),
            patience=settings.get("lr_patience", 3), min_lr=settings.get("min_learning_rate", 1e-6))
    elif schedule == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    else:
        raise ValueError("lr_schedule 必须是 plateau / cosine / constant。")
    amp = settings["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_loader = make_loader(subsets["train"], config, ids, True)
    val_loader = make_loader(subsets["train"] if all_data else subsets["val"], config, ids)
    from .diagnostics import measures, record as record_diagnostics
    monitor_limit=settings.get('monitor_samples',128)
    monitor_rows=sorted(subsets['train'],key=lambda r:r['sample_id'])
    if monitor_limit>0 and len(monitor_rows)>monitor_limit:
        indices=np.random.default_rng(seed).choice(len(monitor_rows),monitor_limit,replace=False)
        monitor_rows=[monitor_rows[i] for i in sorted(indices)]
    monitor_loader=make_loader(monitor_rows,config,ids)
    write_json(output/'monitor_samples.json',{'sample_ids':[r['sample_id'] for r in monitor_rows],
               'scope':'固定训练监测子集；eval模式，与验证相同计算定义；不代表全训练集。'})
    best, best_tie_loss, stale = float("inf"), float("inf"), 0
    start_epoch=0
    resume=settings.get('resume')
    if resume:
        saved=torch.load(project_path(resume),map_location='cpu',weights_only=True)
        if saved.get('resume_version')!=1:raise ValueError('旧模型没有完整续训状态，请选择新版last.pt。')
        if saved['manifest_snapshot']!=rows:raise ValueError('续训索引与原训练不一致。')
        old=saved['config']
        for section in ('model','data','loss'):
            if old[section]!=config[section]:raise ValueError('续训必须保持模型、数据和损失配置一致。')
        for key in ('epochs','batch_size','seed','learning_rate','weight_decay','mode','amp'):
            if old['training'].get(key)!=settings.get(key):raise ValueError('续训不能改变'+key)
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['epoch'];best=saved['best'];best_tie_loss=saved['best_tie_loss'];stale=saved['stale']
        random.setstate(saved['python_rng'])
        ns=saved['numpy_rng'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
        torch.set_rng_state(saved['torch_rng'])
        if torch.cuda.is_available() and saved['cuda_rng']:torch.cuda.set_rng_state_all(saved['cuda_rng'])
        train_loader.generator.set_state(saved['loader_rng'])
        import shutil
        for name in ('best.pt','best_validation_metrics.json','model.pt','training_fit_metrics.json'):
            source=Path(resume).parent/name
            if source.exists():shutil.copy2(source,output/name)
        source=Path(resume).parent/'diagnostic_history.csv'
        if source.exists():
            with source.open(encoding='utf-8') as f:diagnostics=list(csv.DictReader(f))
            if diagnostics:
                with (output/'diagnostic_history.csv').open('w',encoding='utf-8',newline='') as f:
                    dw=csv.DictWriter(f,fieldnames=list(diagnostics[0]));dw.writeheader()
                    dw.writerows(r for r in diagnostics if int(r['epoch'])<=start_epoch)
    def save_last(epoch):
        ns=np.random.get_state()
        state=dict(resume_version=1,state_dict=model.state_dict(),config=config,candidates=torch.tensor(candidates),candidate_ids=torch.tensor(ids),fixed_index=fixed,
                   epoch=epoch,manifest_snapshot=rows,training_mode=settings.get('mode','holdout'),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                   best=best,best_tie_loss=best_tie_loss,stale=stale,python_rng=random.getstate(),numpy_rng=(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],loader_rng=train_loader.generator.get_state())
        torch.save(state,output/'last.tmp');(output/'last.tmp').replace(output/'last.pt')
    save_last(start_epoch)
    batch_path=output/'batch_history.csv'
    batch_path.write_text('epoch,batch,loss,learning_rate,stage,regression,soft,regret,rank,classification,auxiliary_weight\n',encoding='utf-8')
    with (output / "training_history.csv").open("w", encoding="utf-8", newline="") as file:
        prefix = 'fit' if all_data else 'val'
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "monitor_fixed_objective", f"{prefix}_selected_ber", f"{prefix}_regret", f"{prefix}_objective",
                                                f"{prefix}_top1", f"{prefix}_top3", "fixed_ber", "grid_best_ber", "learning_rate", "seconds"])
        writer.writeheader()
        if resume:
            source=Path(resume).parent/'training_history.csv'
            if source.exists():
                with source.open(encoding='utf-8-sig') as previous:
                    for record in csv.DictReader(previous):
                        if int(record['epoch'])<=start_epoch:writer.writerow(record)
        for epoch in range(start_epoch,settings["epochs"]):
            model.train()
            start, total_loss, total = time.perf_counter(), 0.0, 0
            gradients=[];clipped=0;skipped=0;component_sums={};lr_used=optimizer.param_groups[0]['lr']
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            progress=Progress('训练批次',len(train_loader.dataset),epoch=epoch+1,epochs=settings['epochs'])
            progress.update(0,f'第{epoch+1}/{settings["epochs"]}轮',force=True)
            for batch_number,batch in enumerate(train_loader,1):
                check_control()
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    scores = run_model(model, batch)
                # 分位数和代价损失使用 float32，避免混合精度影响小 BER 差距。
                loss,parts = selection_loss(scores.float(), batch["ber"].float(), config["loss"], epoch, config["model"]["kind"], return_components=True)
                if not torch.isfinite(loss):
                    raise ValueError("训练损失非有限值，停止保存模型。")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # AMP 初始缩放过大时允许 GradScaler 跳过该步并降低 scale；
                # 不能把这种正常的动态缩放过程误判成不可恢复失败。
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=not amp)
                norm_value=float(norm);gradients.append(norm_value);clipped+=int(norm_value>settings['gradient_clip'])
                old_scale=scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped+=int(scaler.get_scale()<old_scale)
                for key,v in parts.items():component_sums[key]=component_sums.get(key,0.)+v*len(batch['x'])
                with batch_path.open('a',encoding='utf-8',newline='') as batch_file:
                    csv.writer(batch_file).writerow([epoch+1,batch_number,loss.item(),optimizer.param_groups[0]['lr'],'warmup' if epoch<config['loss']['warmup_epochs'] else 'combined']+[parts.get(key,0.) for key in ('regression','soft','regret','rank','classification','auxiliary_weight')])
                total_loss += loss.item() * len(batch["x"])
                total += len(batch["x"])
                progress.update(total,f'第{epoch+1}轮，loss={loss.item():.5f}')
            scores, ber, _, groups = collect_predictions(model, val_loader, device)
            metrics, _ = evaluate_scores(scores, ber, fixed, groups, config["evaluation"]["tie_tolerance"], 0)
            value = metrics["selected_ber_macro"]
            # 小验证集常有多轮相同（甚至全零）BER。用固定定义的验证目标
            # 作第二排序键，避免无条件保留尚未收敛的第1轮；不看测试集。
            # 训练过程的总 loss 会在 warmup/ramp 期间改变定义。此处固定用“全部辅助项已启用”的
            # 定义评估，供曲线横向比较；它不参与反向传播，也不看测试集。
            full_loss_epoch = config["loss"]["warmup_epochs"] + max(1, int(config["loss"].get("loss_ramp_epochs", 1))) - 1
            val_objective = float(selection_loss(torch.from_numpy(scores), torch.from_numpy(ber), config["loss"],
                                                 full_loss_epoch, config["model"]["kind"]))
            # 额外诊断隔离全局随机数，不影响dropout、候选抽样或恢复一致性。
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device.type=='cuda' else []):
                train_scores,train_ber,_,_=collect_predictions(model,monitor_loader,device)
            monitor_objective = float(selection_loss(torch.from_numpy(train_scores), torch.from_numpy(train_ber), config["loss"],
                                                     full_loss_epoch, config["model"]["kind"]))
            train_measures=measures(train_scores,train_ber,fixed,config['loss'])
            val_measures={} if all_data else measures(scores,ber,fixed,config['loss'])
            record_diagnostics(output,epoch+1,train_measures,val_measures,
                   {'lr_used':lr_used,'grad_norm_mean':float(np.mean([v for v in gradients if np.isfinite(v)])) if any(np.isfinite(gradients)) else 0.,
                    'clip_fraction':clipped/len(gradients),'amp_skipped_batches':skipped,'amp_scale':scaler.get_scale(),
                    'seconds':time.perf_counter()-start,'peak_gpu_mb':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0.,
                    **{'weighted_'+k:component_sums.get(k,0.)/total for k in ('regression','soft','regret','rank','classification','auxiliary_weight')}})
            np.savez_compressed(output/'latest_validation_scores.npz',scores=scores,ber=ber,epoch=epoch+1,scope='fit' if all_data else 'validation')
            # In holdout mode the validation BER is the quantity that matters
            # to parameter selection.  Plateau scheduling therefore responds
            # to BER, not to the differently-scaled compound training loss.
            if schedule == "plateau":
                scheduler.step(value)
            else:
                scheduler.step()
            writer.writerow({"epoch": epoch + 1, "train_loss": total_loss / total, "monitor_fixed_objective": monitor_objective, f"{prefix}_selected_ber": value,
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
                save_last(epoch+1)
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
            save_last(epoch+1)
            if stale >= settings["patience"]:
                break
    filename = 'model.pt' if all_data else 'best.pt'
    print(f"训练完成：{output / filename}；" + ('全量拟合，等待新信道验证。' if all_data else '测试集尚未用于选模型。'), flush=True)
    return output / filename


def train(config):
    try:
        return _train(config)
    except KeyboardInterrupt:
        output=project_path(config['training']['output'])
        print(f'训练已停止。最近完整轮次：{output / "last.pt"}；未完成轮次恢复时重跑。',flush=True)
        write_json(output/'stopped.json',{'status':'stopped','resume':str(output/'last.pt'),'exists':(output/'last.pt').exists()})
        return None


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
