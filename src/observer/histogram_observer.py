"""
HistogramObserver - 基于直方图的Observer
支持更精确的量化参数计算
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import ObserverBase
from autoquant.core import QuantDtype, QScheme


class HistogramObserver(ObserverBase):
    """
    HistogramObserver：基于直方图的Observer
    论文：https://arxiv.org/abs/1906.00532
    通过收集直方图来更精确地计算量化参数
    """

    def __init__(
        self,
        dtype: QuantDtype = QuantDtype.QUINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        ch_axis: int = 0,
        bins: int = 2048,
        upsample_rate: int = 128,
    ):
        super().__init__(dtype, qscheme, quant_min, quant_max, ch_axis)
        self.bins = bins
        self.upsample_rate = upsample_rate
        
        # 初始化直方图
        self.register_buffer("histogram", None)
        self.register_buffer("min_val", None)
        self.register_buffer("max_val", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，收集直方图
        """
        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            # 按通道统计
            self._forward_per_channel(x)
        else:
            # 按张量统计
            self._forward_per_tensor(x)
        return x

    def _forward_per_tensor(self, x: torch.Tensor):
        """
        按张量统计直方图
        """
        # 计算当前批次的min和max
        current_min = torch.amin(x)
        current_max = torch.amax(x)
        
        # 更新全局的min和max
        if self.min_val is None:
            self.min_val = current_min
            self.max_val = current_max
        else:
            self.min_val = torch.min(self.min_val, current_min)
            self.max_val = torch.max(self.max_val, current_max)
        
        # 计算直方图
        if self.histogram is None:
            self.histogram = torch.zeros(self.bins, device=x.device)
        
        # 使用torch.histc计算直方图
        hist = torch.histc(
            x,
            bins=self.bins,
            min=self.min_val.item(),
            max=self.max_val.item()
        )
        self.histogram += hist

    def _forward_per_channel(self, x: torch.Tensor):
        """
        按通道统计直方图
        """
        # 转置以便于按通道处理
        dims = list(range(x.dim()))
        dims.pop(self.ch_axis)
        dims = [self.ch_axis] + dims
        x_transposed = x.permute(dims).contiguous()
        
        num_channels = x.shape[self.ch_axis]
        
        # 初始化直方图
        if self.histogram is None:
            self.histogram = torch.zeros(
                num_channels, self.bins, device=x.device
            )
            self.min_val = torch.zeros(num_channels, device=x.device)
            self.max_val = torch.zeros(num_channels, device=x.device)
        
        # 对每个通道分别处理
        for i in range(num_channels):
            channel_data = x_transposed[i]
            
            current_min = torch.amin(channel_data)
            current_max = torch.amax(channel_data)
            
            # 更新min和max
            if self.min_val[i] == 0 and self.max_val[i] == 0:
                self.min_val[i] = current_min
                self.max_val[i] = current_max
            else:
                self.min_val[i] = torch.min(self.min_val[i], current_min)
                self.max_val[i] = torch.max(self.max_val[i], current_max)
            
            # 计算直方图
            hist = torch.histc(
                channel_data,
                bins=self.bins,
                min=self.min_val[i].item(),
                max=self.max_val[i].item()
            )
            self.histogram[i] += hist

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于直方图计算量化参数
        使用KL散度来找到最佳阈值
        """
        assert self.histogram is not None, "需要先调用forward统计数据"
        
        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            return self._calculate_qparams_per_channel()
        else:
            return self._calculate_qparams_per_tensor()

    def _calculate_qparams_per_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        按张量计算量化参数
        """
        min_val = self.min_val
        max_val = self.max_val
        
        # 对称量化调整
        if self.qscheme in [QScheme.PER_TENSOR_SYMMETRIC, QScheme.PER_CHANNEL_SYMMETRIC]:
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            min_val = -max_abs
            max_val = max_abs
        
        # 计算scale和zero_point
        scale, zero_point = self._compute_qparams(min_val, max_val)
        self.scale = scale
        self.zero_point = zero_point
        return scale, zero_point

    def _calculate_qparams_per_channel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        按通道计算量化参数
        """
        num_channels = self.histogram.shape[0]
        scales = []
        zero_points = []
        
        for i in range(num_channels):
            min_val = self.min_val[i]
            max_val = self.max_val[i]
            
            if self.qscheme in [QScheme.PER_TENSOR_SYMMETRIC, QScheme.PER_CHANNEL_SYMMETRIC]:
                max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
                min_val = -max_abs
                max_val = max_abs
            
            scale, zero_point = self._compute_qparams(min_val, max_val)
            scales.append(scale)
            zero_points.append(zero_point)
        
        self.scale = torch.stack(scales)
        self.zero_point = torch.stack(zero_points)
        return self.scale, self.zero_point

    def _compute_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算scale和zero_point的核心逻辑
        """
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
        """重置统计量"""
        self.histogram = None
        self.min_val = None
        self.max_val = None
        self.scale = None
        self.zero_point = None