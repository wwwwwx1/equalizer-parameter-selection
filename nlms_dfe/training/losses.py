"""在样本内部构造代价目标；不利用测试集的均值或统计量。"""
import torch
from torch.nn import functional as F


def selection_loss(scores, ber, config, epoch, kind="scorer"):
    if kind == "classifier":
        # 并列最小值均分标签质量，不任意挑第一个类别。
        ties = ber == ber.min(dim=1, keepdim=True).values
        q = ties.float() / ties.sum(dim=1, keepdim=True)
        return -(q * F.log_softmax(-scores, dim=1)).sum(1).mean()
    scale = (torch.quantile(ber, 0.9, dim=1) - torch.quantile(ber, 0.1, dim=1)).clamp_min(config["scale_floor"])
    target = (ber - ber.min(1, keepdim=True).values) / scale[:, None]
    loss = config["score_weight"] * F.smooth_l1_loss(scores, target)
    if epoch < config["warmup_epochs"]:
        return loss
    temp = config["temperature"]
    if temp <= 0:
        raise ValueError("temperature 必须大于零。")
    log_p = F.log_softmax(-scores / temp, dim=1)
    q = F.softmax(-target / temp, dim=1)
    loss = loss + config["soft_weight"] * (-(q * log_p).sum(1).mean())
    loss = loss + config["regret_weight"] * ((log_p.exp() * target).sum(1).mean())
    if config["rank_weight"]:
        pairs = config["rank_pairs"]
        a = torch.randint(scores.shape[1], (pairs,), device=scores.device)
        b = torch.randint(scores.shape[1], (pairs,), device=scores.device)
        gap = target[:, b] - target[:, a]
        weights = gap.abs().clamp(max=1)
        rank = F.softplus(-gap.sign() * (scores[:, b] - scores[:, a]))
        loss = loss + config["rank_weight"] * (weights * rank).sum() / weights.sum().clamp_min(1)
    return loss
