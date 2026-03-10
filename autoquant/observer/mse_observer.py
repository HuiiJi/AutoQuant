"""
MSEObserver - 基于最小化量化误差的Observer
通过搜索找到最小化MSE的量化参数
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import ObserverBase
from autoquant.core import QuantDtype, QScheme


class MSEObserver(ObserverBase):
    """
    MSEObserver：基于最小化量化误差的Observer
    通过搜索找到最小化MSE的量化参数
    """

    def __init__(
        self,
        dtype: QuantDtype = QuantDtype.QUINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        ch_axis: int = 0,
        num_candidates: int = 100,
    ):
        super().__init__(dtype, qscheme, quant_min, quant_max, ch_axis)
        self.num_candidates = num_candidates
        self.all_values = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        min_val = torch.min(all_data)
        max_val = torch.max(all_data)
        
        best_scale, best_zero_point = self._search_best_qparams(
            all_data, min_val, max_val
        )
        
        self.scale = best_scale.to(self.all_values[0].device)
        self.zero_point = best_zero_point.to(self.all_values[0].device)
        return self.scale, self.zero_point

    def _calculate_qparams_per_channel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        scales = []
        zero_points = []
        
        for channel_values in self.all_values:
            all_data = torch.cat(channel_values)
            min_val = torch.min(all_data)
            max_val = torch.max(all_data)
            
            best_scale, best_zero_point = self._search_best_qparams(
                all_data, min_val, max_val
            )
            scales.append(best_scale.to(channel_values[0].device))
            zero_points.append(best_zero_point.to(channel_values[0].device))
        
        self.scale = torch.stack(scales)
        self.zero_point = torch.stack(zero_points)
        return self.scale, self.zero_point

    def _search_best_qparams(self, data: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qmin = self.quant_min
        qmax = self.quant_max
        
        best_mse = float('inf')
        best_scale = None
        best_zero_point = None
        
        # 搜索多个候选值
        for i in range(self.num_candidates):
            # 逐步缩小搜索范围
            shrink_factor = 0.5 ** (i // 20)
            candidate_min = min_val * (1 - shrink_factor * (i % 2))
            candidate_max = max_val * (1 + shrink_factor * ((i + 1) % 2))
            
            if self.qscheme in [QScheme.PER_TENSOR_SYMMETRIC, QScheme.PER_CHANNEL_SYMMETRIC]:
                max_abs = torch.max(torch.abs(candidate_min), torch.abs(candidate_max))
                candidate_min = -max_abs
                candidate_max = max_abs
            
            scale, zero_point = self._compute_qparams(candidate_min, candidate_max)
            mse = self._compute_mse(data, scale, zero_point)
            
            if mse < best_mse:
                best_mse = mse
                best_scale = scale
                best_zero_point = zero_point
        
        return best_scale, best_zero_point

    def _compute_mse(self, data: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> float:
        qmin = self.quant_min
        qmax = self.quant_max
        
        # 量化-反量化
        x_int = torch.round(data / scale + zero_point)
        x_int = torch.clamp(x_int, qmin, qmax)
        x_dq = (x_int - zero_point) * scale
        
        # 计算MSE
        mse = torch.mean((data - x_dq) ** 2)
        return mse.item()

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
        self.scale = None
        self.zero_point = None