"""共享消融定义；界面读取时不加载PyTorch。"""
VARIANTS = {
    "full_current": {
        "label": "完整当前结构", "changes": {},
        "tests": "当前编码器、SNR、功率、FiLM、候选评分和组合损失的共同基准。"
    },
    "no_power": {
        "label": "去除绝对功率", "changes": {"model.use_channel_power": False},
        "tests": "功率支路是否改善来源组外的MeanCodedBER；先前幅度诊断表明它可能造成不合理敏感性。"
    },
    "no_snr_current": {
        "label": "去除SNR（保留功率）", "changes": {"model.use_snr": False},
        "tests": "在完整输入条件下，单独检验SNR支路的贡献。"
    },
    "concat_current": {
        "label": "SNR拼接（保留功率）", "changes": {"model.snr_conditioning": "concat"},
        "tests": "在完整输入条件下，单独检验FiLM相对SNR拼接的作用。"
    },
    "no_temporal_current": {
        "label": "无时间Transformer（保留功率）", "changes": {"model.layers": 0},
        "tests": "在完整输入条件下，单独检验时间关系建模的贡献。"
    },
    "no_delay_cnn_current": {
        "label": "无时延CNN（保留功率）", "changes": {"model.delay_encoder": "pool"},
        "tests": "在完整输入条件下，单独检验局部时延CNN相对直接池化的贡献。"
    },
    "no_residual_current": {
        "label": "无时延残差块（保留功率）", "changes": {"model.delay_residual": False},
        "tests": "在完整输入条件下，单独检验残差块的贡献。"
    },
    "score_only_current": {
        "label": "仅性能差距回归（保留功率）", "changes": {"loss.soft_weight": 0., "loss.regret_weight": 0., "loss.rank_weight": 0., "loss.warmup_epochs": 999},
        "tests": "在完整输入条件下，单独检验组合选择损失相对仅回归的贡献。"
    },
    "classifier_current": {
        "label": "直接294类分类（保留功率）", "changes": {"model.kind": "classifier"},
        "tests": "在完整输入条件下，单独检验候选条件评分相对直接分类的贡献。"
    },
    "no_snr_after_no_power": {
        "label": "去除SNR（无功率）", "changes": {"model.use_channel_power": False, "model.use_snr": False},
        "tests": "在无绝对功率混杂时，SNR是否提供选参信息。"
    },
    "concat_after_no_power": {
        "label": "SNR拼接（无FiLM）", "changes": {"model.use_channel_power": False, "model.snr_conditioning": "concat"},
        "tests": "FiLM相对简单条件拼接的作用。"
    },
    "no_temporal_after_no_power": {
        "label": "无时间Transformer", "changes": {"model.use_channel_power": False, "model.layers": 0},
        "tests": "长时间关系是否超过逐时间片编码与注意力汇总。"
    },
    "no_delay_cnn_after_no_power": {
        "label": "无时延CNN", "changes": {"model.use_channel_power": False, "model.delay_encoder": "pool"},
        "tests": "学习局部时延结构相对于直接时延池化的作用。"
    },
    "no_residual_after_no_power": {
        "label": "无时延残差块", "changes": {"model.use_channel_power": False, "model.delay_residual": False},
        "tests": "残差块是否有实际贡献，避免把CNN与残差优化混为同一贡献。"
    },
    "score_only_after_no_power": {
        "label": "仅性能差距回归", "changes": {"model.use_channel_power": False, "loss.soft_weight": 0., "loss.regret_weight": 0., "loss.rank_weight": 0., "loss.warmup_epochs": 999},
        "tests": "软标签、期望regret和排序项是否比单纯回归更符合选参目标。"
    },
    "classifier_after_no_power": {
        "label": "直接294类分类", "changes": {"model.use_channel_power": False, "model.kind": "classifier"},
        "tests": "候选参数条件评分相对于将294组合视为互不相关类别的作用。"
    },
}

