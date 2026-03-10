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
        self.register_buffer("min_val", None)
        self.register_buffer("max_val", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，使用滑动平均更新min和max
        """
        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            dims = list(range(x.dim()))
            dims.pop(self.ch_axis)
            current_min = torch.amin(x, dim=dims, keepdim=False)
            current_max = torch.amax(x, dim=dims, keepdim=False)
        else:
            current_min = torch.amin(x)
            current_max = torch.amax(x)
        
        if self.min_val is None:
            self.min_val = current_min.detach()
            self.max_val = current_max.detach()
        else:
            # 滑动平均更新
            self.min_val = (1 - self.momentum) * self.min_val + self.momentum * current_min.detach()
            self.max_val = (1 - self.momentum) * self.max_val + self.momentum * current_max.detach()
        
        return x

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.min_val is not None and self.max_val is not None
        
        min_val = self.min_val
        max_val = self.max_val
        
        if self.qscheme in [QScheme.PER_TENSOR_SYMMETRIC, QScheme.PER_CHANNEL_SYMMETRIC]:
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            min_val = -max_abs
            max_val = max_abs
        
        scale, zero_point = self._compute_qparams(min_val, max_val)
        self.scale = scale
        self.zero_point = zero_point
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
        self.min_val = None
        self.max_val = None
        self.scale = None
        self.zero_point = None


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
        
        scale, zero_point = self._compute_qparams(min_val.to(self.all_values[0].device), max_val.to(self.all_values[0].device))
        self.scale = scale
        self.zero_point = zero_point
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
            
            scale, zero_point = self._compute_qparams(min_val.to(channel_values[0].device), max_val.to(channel_values[0].device))
            scales.append(scale)
            zero_points.append(zero_point)
        
        self.scale = torch.stack(scales)
        self.zero_point = torch.stack(zero_points)
        return self.scale, self.zero_point

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
