"""
PercentileObserver - 基于百分位数的Observer
可以去除极端值的影响，获得更稳定的量化参数

Author: jihui
Date: 2026-03-13
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import ObserverBase
from autoquant.core import QuantDtype, QScheme


class PercentileObserver(ObserverBase):
    """
    PercentileObserver：基于百分位数的Observer
    可以去除极端值的影响，获得更稳定的量化参数
    """

    def __init__(
        self,
        dtype: QuantDtype = QuantDtype.QUINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        ch_axis: int = 0,
        min_percentile: float = 0.0001,
        max_percentile: float = 0.9999,
    ):
        super().__init__(dtype, qscheme, quant_min, quant_max, ch_axis)
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.all_values = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，收集所有数据
        """
        if not self.enabled:
            return x

        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            self._forward_per_channel(x)
        else:
            self._forward_per_tensor(x)
        return x

    def _forward_per_tensor(self, x: torch.Tensor):
        self.all_values.append(x.flatten().detach().cpu())

    def _forward_per_channel(self, x: torch.Tensor):
        dims = list(range(x.dim()))
        dims.pop(self.ch_axis)
        x_transposed = x.permute([self.ch_axis] + dims).contiguous()
        num_channels = x.shape[self.ch_axis]

        if not self.all_values:
            self.all_values = [[] for _ in range(num_channels)]

        for i in range(num_channels):
            self.all_values[i].append(x_transposed[i].flatten().detach().cpu())

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(self.all_values) > 0, "需要先调用forward统计数据"

        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            return self._calculate_qparams_per_channel()
        else:
            return self._calculate_qparams_per_tensor()

    def _calculate_qparams_per_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        all_data = torch.cat(self.all_values)
        min_val = torch.quantile(all_data, self.min_percentile)
        max_val = torch.quantile(all_data, self.max_percentile)

        if self.qscheme in [QScheme.PER_TENSOR_SYMMETRIC, QScheme.PER_CHANNEL_SYMMETRIC]:
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            min_val = -max_abs
            max_val = max_abs

        scale, zero_point = self._compute_qparams(
            min_val.to(
                self.all_values[0].device), max_val.to(
                self.all_values[0].device))
        self._scale = scale
        self._zero_point = zero_point
        return scale, zero_point

    def _calculate_qparams_per_channel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        scales = []
        zero_points = []

        for channel_values in self.all_values:
            all_data = torch.cat(channel_values)
            min_val = torch.quantile(all_data, self.min_percentile)
            max_val = torch.quantile(all_data, self.max_percentile)

            if self.qscheme in [QScheme.PER_TENSOR_SYMMETRIC, QScheme.PER_CHANNEL_SYMMETRIC]:
                max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
                min_val = -max_abs
                max_val = max_abs

            scale, zero_point = self._compute_qparams(
                min_val.to(
                    channel_values[0].device), max_val.to(
                    channel_values[0].device))
            scales.append(scale)
            zero_points.append(zero_point)

        self._scale = torch.stack(scales)
        self._zero_point = torch.stack(zero_points)
        return self._scale, self._zero_point

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
        self.all_values = []
        self._scale = None
        self._zero_point = None
