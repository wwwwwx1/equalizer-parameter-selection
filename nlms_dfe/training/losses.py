"""在样本内部构造代价目标；不利用测试集的均值或统计量。"""
import torch
from torch.nn import functional as F


def selection_loss(scores, ber, config, epoch, kind="scorer", return_components=False):
    if kind == "classifier":
        # 并列最小值均分标签质量，不任意挑第一个类别。
        ties = ber == ber.min(dim=1, keepdim=True).values
        q = ties.float() / ties.sum(dim=1, keepdim=True)
        loss=-(q * F.log_softmax(-scores, dim=1)).sum(1).mean()
        return (loss,{'classification':float(loss.detach())}) if return_components else loss
    scale = (torch.quantile(ber, 0.9, dim=1) - torch.quantile(ber, 0.1, dim=1)).clamp_min(config["scale_floor"])
    target = (ber - ber.min(1, keepdim=True).values) / scale[:, None]
    loss = config["score_weight"] * F.smooth_l1_loss(scores, target)
    parts={'regression':float(loss.detach()),'soft':0.,'regret':0.,'rank':0.}
    if epoch < config['warmup_epochs']:
        return (loss,parts) if return_components else loss
    # Do not suddenly replace one objective with another after warmup.  The
    # auxiliary selection losses ramp in smoothly, so adjacent epochs remain
    # comparable and the optimiser is not shocked by a large new gradient.
    ramp_epochs = max(1, int(config.get("loss_ramp_epochs", 1)))
    ramp = min(1.0, (epoch - config["warmup_epochs"] + 1) / ramp_epochs)
    parts['auxiliary_weight'] = ramp
    temp = config["temperature"]
    if temp <= 0:
        raise ValueError("temperature 必须大于零。")
    log_p = F.log_softmax(-scores / temp, dim=1)
    q = F.softmax(-target / temp, dim=1)
    term=ramp * config['soft_weight'] * (-(q * log_p).sum(1).mean())
    parts['soft']=float(term.detach());loss=loss+term
    term=ramp * config['regret_weight'] * ((log_p.exp() * target).sum(1).mean())
    parts['regret']=float(term.detach());loss=loss+term
    if config["rank_weight"]:
        pairs = config["rank_pairs"]
        a = torch.randint(scores.shape[1], (pairs,), device=scores.device)
        b = torch.randint(scores.shape[1], (pairs,), device=scores.device)
        gap = target[:, b] - target[:, a]
        weights = gap.abs().clamp(max=1)
        rank = F.softplus(-gap.sign() * (scores[:, b] - scores[:, a]))
        term=ramp * config['rank_weight'] * (weights * rank).sum() / weights.sum().clamp_min(1)
        parts['rank']=float(term.detach());loss=loss+term
    return (loss,parts) if return_components else loss
