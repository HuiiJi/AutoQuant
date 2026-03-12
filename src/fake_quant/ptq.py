"""
PTQ FakeQuantize - 后训练量化专用
使用 torch.fake_quantize API，确保 ONNX 导出为 QDQ 节点

Author: jihui
Date: 2026-03-13
"""
import torch
import torch.nn as nn
from typing import Optional
from .base import FakeQuantizeBase
from autoquant.core import QuantDtype, QScheme
from autoquant.observer import ObserverBase


class PTQFakeQuantize(FakeQuantizeBase):
    """
    PTQFakeQuantize：后训练量化专用
    
    正确逻辑：
    1. 校准阶段：
       - observer 统计数据分布
       - 每次 forward 都计算 qparams 并进行量化（带噪声）
       - 这样噪声才能一层层传递，校准更准确
    2. 推理阶段：
       - 禁用 observer
       - 直接用计算好的 qparams 量化
    
    特点：
    - 使用 torch.fake_quantize_per_*_affine，确保 ONNX 导出为 QDQ
    """

    def __init__(
        self,
        observer: ObserverBase = None,
        dtype: QuantDtype = QuantDtype.QUINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        ch_axis: int = 0,
        enabled: bool = True,
    ):
        if observer is not None:
            dtype = observer.dtype
            qscheme = observer.qscheme
            quant_min = observer.quant_min
            quant_max = observer.quant_max
            ch_axis = observer.ch_axis

        super().__init__(
            observer=observer,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            ch_axis=ch_axis,
            enabled=enabled,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if not self.enabled:
            return x

        # 阶段1: 如果 observer 启用，先统计数据
        if self.observer and self.observer.enabled:
            self.observer(x)

        # 阶段2: 只要有数据，就计算 qparams 并量化
        # 这样校准阶段也能带量化噪声，噪声一层层传递
        if self.observer and self.observer.min_val is not None:
            if self.scale is None or self.zero_point is None:
                self.calculate_qparams()

        # 如果还没有 qparams（刚开始校准），直接返回
        if self.scale is None or self.zero_point is None:
            return x

        qmin = self.observer.quant_min if self.observer else self.quant_min
        qmax = self.observer.quant_max if self.observer else self.quant_max
        scale = self.scale
        zero_point = self.zero_point

        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            if zero_point.dtype not in [torch.int32, torch.float32]:
                zero_point = zero_point.to(torch.int32)
            return torch.fake_quantize_per_channel_affine(
                x, scale=scale, zero_point=zero_point, axis=self.ch_axis, quant_min=qmin, quant_max=qmax
            )
        else:
            return torch.fake_quantize_per_tensor_affine(
                x, scale=scale, zero_point=zero_point, quant_min=qmin, quant_max=qmax
            )
