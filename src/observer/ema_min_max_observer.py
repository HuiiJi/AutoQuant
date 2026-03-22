"""
MovingAverageMinMaxObserver - 滑动平均的MinMaxObserver
适用于在线学习或QAT

Author: jihui
Date: 2026-03-13
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import ObserverBase
from autoquant.core import QuantDtype, QScheme


class MovingAverageMinMaxObserver(ObserverBase):
    """
    MovingAverageMinMaxObserver：滑动平均的MinMaxObserver
    适用于在线学习或QAT
    """

    def __init__(
        self,
        dtype: QuantDtype = QuantDtype.QUINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        ch_axis: int = 0,
        momentum: float = 0.05,
    ):
        super().__init__(dtype, qscheme, quant_min, quant_max, ch_axis)
        self.momentum = momentum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，使用滑动平均更新min和max
        """
        if not self.enabled:
            return x

        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            dims = list(range(x.dim()))
            dims.pop(self.ch_axis)
            current_min = torch.amin(x, dim=dims, keepdim=False)
            current_max = torch.amax(x, dim=dims, keepdim=False)
        else:
            current_min = torch.amin(x)
            current_max = torch.amax(x)

        if self._min_val is None:
            self._min_val = current_min.detach()
            self._max_val = current_max.detach()
        else:
            # 滑动平均更新
            self._min_val = (1 - self.momentum) * self._min_val + self.momentum * current_min.detach()
            self._max_val = (1 - self.momentum) * self._max_val + self.momentum * current_max.detach()

        return x

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._min_val is not None and self._max_val is not None

        min_val = self._min_val
        max_val = self._max_val

        if self.qscheme in [QScheme.PER_TENSOR_SYMMETRIC, QScheme.PER_CHANNEL_SYMMETRIC]:
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            min_val = -max_abs
            max_val = max_abs

        scale, zero_point = self._compute_qparams(min_val, max_val)
        self._scale = scale
        self._zero_point = zero_point
        return scale, zero_point

    def _compute_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qmin = self.quant_min
        qmax = self.quant_max

        max_val = torch.max(max_val, min_val + 1e-8)

        scale = (max_val - min_val) / float(qmax - qmin)
        initial_zero_point = qmin - min_val / scale
        zero_point = torch.round(initial_zero_point)
        zero_point = torch.clamp(zero_point, qmin, qmax)

        if self.qscheme in [QScheme.PER_TENSOR_SYMMETRIC, QScheme.PER_CHANNEL_SYMMETRIC]:
            zero_point = torch.zeros_like(zero_point)

        if self.dtype == QuantDtype.QUINT8:
            zero_point = zero_point.to(torch.uint8)
        elif self.dtype == QuantDtype.QINT8:
            zero_point = zero_point.to(torch.int8)

        return scale, zero_point

    def reset(self):
        self._min_val = None
        self._max_val = None
        self._scale = None
        self._zero_point = None
