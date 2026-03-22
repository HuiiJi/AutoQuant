"""
MinMaxObserver - 基于最小最大值的Observer

Author: jihui
Date: 2026-03-13
"""
import torch
from typing import Tuple
from .base import ObserverBase
from autoquant.core import QuantDtype, QScheme


class MinMaxObserver(ObserverBase):
    """
    MinMaxObserver：统计输入的最小值和最大值
    """

    def __init__(
        self,
        dtype: QuantDtype = QuantDtype.QUINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
        quant_min: int = None,
        quant_max: int = None,
        ch_axis: int = 0,
    ):
        super().__init__(dtype, qscheme, quant_min, quant_max, ch_axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，统计最小值和最大值
        """
        if not self.enabled:
            return x

        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            # 按通道统计
            dims = list(range(x.dim()))
            dims.pop(self.ch_axis)
            min_val = torch.amin(x, dim=dims, keepdim=False)
            max_val = torch.amax(x, dim=dims, keepdim=False)
        else:
            # 按张量统计
            min_val = torch.amin(x)
            max_val = torch.amax(x)

        # 更新统计量
        if self._min_val is None:
            self._min_val = min_val.detach()
            self._max_val = max_val.detach()
        else:
            self._min_val = torch.min(self._min_val, min_val.detach())
            self._max_val = torch.max(self._max_val, max_val.detach())

        return x

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算量化参数scale和zero_point
        """
        if self._min_val is None or self._max_val is None:
            # 没有统计数据，直接返回 (None, None)
            return None, None

        min_val = self._min_val
        max_val = self._max_val

        # 对称量化调整
        if self.qscheme in [QScheme.PER_TENSOR_SYMMETRIC, QScheme.PER_CHANNEL_SYMMETRIC]:
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            min_val = -max_abs
            max_val = max_abs
            if self.dtype == QuantDtype.QINT8:
                # 对称量化时zero_point固定为0
                pass

        # 计算scale和zero_point
        scale, zero_point = self._compute_qparams(min_val, max_val)
        self._scale = scale
        self._zero_point = zero_point
        return scale, zero_point

    def _compute_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算scale和zero_point的核心逻辑
        """
        # 确保min_val <= max_val
        min_val = torch.min(min_val, max_val)

        qmin = self.quant_min
        qmax = self.quant_max

        # 防止除零
        max_val = torch.max(max_val, min_val + 1e-8)

        # 计算scale
        scale = (max_val - min_val) / float(qmax - qmin)

        # 计算zero_point
        initial_zero_point = qmin - min_val / scale
        zero_point = torch.round(initial_zero_point)
        zero_point = torch.clamp(zero_point, qmin, qmax)

        # 对称量化特殊处理
        if self.qscheme in [QScheme.PER_TENSOR_SYMMETRIC, QScheme.PER_CHANNEL_SYMMETRIC]:
            zero_point = torch.zeros_like(zero_point)

        # 确保类型正确
        if self.dtype == QuantDtype.QUINT8:
            zero_point = zero_point.to(torch.uint8)
        elif self.dtype == QuantDtype.QINT8:
            zero_point = zero_point.to(torch.int8)

        return scale, zero_point

    def reset(self):
        """重置统计量"""
        self._min_val = None
        self._max_val = None
        self._scale = None
        self._zero_point = None
