"""预留真实 NLMS-DFE 重仿真接口，后续接 MATLAB 或现有 Python 接收机。"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ReplayRequest:
    channel: np.ndarray
    snr_db: float
    mu: float
    N1: int
    N2: int
    seed: int
    simulation_id: str  # 关联编码、调制、训练长度等完整协议。


@dataclass(frozen=True)
class ReplayResult:
    bit_errors: int
    evaluated_bits: int
    elapsed_seconds: float

    @property
    def ber(self):
        if self.evaluated_bits <= 0 or not 0 <= self.bit_errors <= self.evaluated_bits:
            raise ValueError("真实重放结果中的误比特数/总比特数无效。")
        return self.bit_errors / self.evaluated_bits


class EqualizerSimulator(Protocol):
    def run(self, request: ReplayRequest) -> ReplayResult:
        """使用指定协议运行一次真实均衡与译码，禁止以评分替代仿真。"""
        ...


class PendingSimulator:
    def run(self, request: ReplayRequest) -> ReplayResult:
        raise NotImplementedError("尚未接入您的 NLMS-DFE 仿真程序；当前只能查询已保存的 BER 表。")
