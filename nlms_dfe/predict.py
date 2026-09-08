"""给应用程序调用的预测接口，无需 SNR/BER 标签文件。"""
import copy
import torch

from .data.transforms import make_features
from .data.dataset import collate_channels
from .training.engine import load_checkpoint


class Predictor:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        self.model, self.checkpoint = load_checkpoint(checkpoint_path, self.device)

    @torch.inference_mode()
    def predict(self, channel, snr_db, top_k=5, *, already_cropped=None, main_path_matlab=None, allow_short=False):
        if not torch.isfinite(torch.tensor(float(snr_db))):
            raise ValueError("预测必须提供有限 SNR 标量。")
        data = copy.deepcopy(self.checkpoint['config']['data'])
        if already_cropped is not None:
            data['crop']['already_cropped'] = already_cropped
        if main_path_matlab is not None:
            if data['crop']['already_cropped']:
                raise ValueError('已裁剪输入不再指定原始主径，避免混淆两个坐标系。')
            data['crop']['main_path_matlab'] = main_path_matlab
        if allow_short and not data['crop']['already_cropped']:
            time_axis = 0 if data['axis_order'] == 'time_delay' else 1
            data['crop']['time_stop_matlab'] = min(data['crop']['time_stop_matlab'], channel.shape[time_axis])
        x, power = make_features(channel, data)
        batch = collate_channels([{"x": x, "scalars": torch.tensor([float(snr_db), power])}])
        scores = self.model(batch["x"].to(self.device), batch["scalars"].to(self.device), batch["padding_mask"].to(self.device))[0].cpu()
        if not torch.isfinite(scores).all():
            raise ValueError("预测评分非有限值。")
        if not 1 <= top_k <= len(scores):
            raise ValueError("top_k 超出候选数量。")
        order = torch.argsort(scores, stable=True)[:top_k]
        results = []
        for index in order.tolist():
            mu, n1, n2 = self.checkpoint["candidates"][index].tolist()
            results.append({"candidate_id": int(self.checkpoint["candidate_ids"][index]),
                            "mu": mu, "N1": int(n1), "N2": int(n2), "score": float(scores[index])})
        return {"selected": results[0], "top_k": results,
                "target_metric": self.checkpoint['config']['training'].get('target_metric','MeanDecodedBER'),
                "score_meaning": "排序分数，越小越优；不是校准 BER 或置信概率。",
                "mode": self.checkpoint["config"]["simulation"]["mode"]}
