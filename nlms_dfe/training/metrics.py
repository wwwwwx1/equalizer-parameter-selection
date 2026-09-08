"""评估的是选中配置的真实表内 BER，而不是把网络评分当成 BER。"""
import numpy as np


def evaluate_scores(scores, ber, fixed_index, groups, tolerance=1e-8, repeats=500, seed=42):
    scores, ber = np.asarray(scores), np.asarray(ber)
    selected_index = scores.argmin(axis=1)
    rows = np.arange(len(ber))
    selected = ber[rows, selected_index]
    fixed = ber[:, fixed_index]
    best = ber.min(axis=1)
    regret = selected - best
    difference = selected - fixed
    gap = float(np.mean(fixed - best))
    metrics = {
        "samples": len(ber), "groups": len(set(groups)),
        "selected_ber_macro": float(selected.mean()), "fixed_ber_macro": float(fixed.mean()),
        "grid_best_ber_macro": float(best.mean()),
        "mean_regret": float(regret.mean()), "median_regret": float(np.median(regret)),
        "p90_regret": float(np.quantile(regret, 0.9)), "p95_regret": float(np.quantile(regret, 0.95)),
        "better_than_fixed_fraction": float(np.mean(difference < -tolerance)),
        "worse_than_fixed_fraction": float(np.mean(difference > tolerance)),
        "equal_to_fixed_fraction": float(np.mean(np.abs(difference) <= tolerance)),
        "selected_minus_fixed_mean": float(difference.mean()),
        # 总体比率不裁剪负收益；没有可恢复空间时返回 null。
        "aggregate_grid_gap_recovery": float(-difference.mean() / gap) if gap > tolerance else None,
        "scope": "已有 BER 表的离线查询；不是独立重仿真；样本等权均值，不是合并比特 BER。"
    }
    order = np.argsort(scores, axis=1, kind="stable")
    for k in (1, 3, 5):
        top_ber = np.take_along_axis(ber, order[:, :min(k, ber.shape[1])], axis=1)
        metrics[f"top{k}_tie_aware_accuracy"] = float(np.mean(np.any(top_ber <= best[:, None] + tolerance, axis=1)))
    # 按源信道重采样，保留组内相关性；统计量仍为样本等权平均差。
    unique = sorted(set(groups))
    if len(unique) >= 2 and repeats > 0:
        groups = np.asarray(groups)
        sums = np.array([difference[groups == g].sum() for g in unique])
        counts = np.array([(groups == g).sum() for g in unique])
        rng = np.random.default_rng(seed)
        estimates = []
        for _ in range(repeats):
            chosen = rng.integers(len(unique), size=len(unique))
            estimates.append(sums[chosen].sum() / counts[chosen].sum())
        metrics["selected_minus_fixed_group_bootstrap_95ci"] = np.quantile(estimates, [0.025, 0.975]).tolist()
    else:
        metrics["selected_minus_fixed_group_bootstrap_95ci"] = None
    return metrics, selected_index
