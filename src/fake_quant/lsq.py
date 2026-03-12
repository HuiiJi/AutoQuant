"""
LSQ FakeQuantize - QAT 专用
论文：https://arxiv.org/abs/1902.08153
Learned Step Size Quantization，scale 作为可学习参数

Author: jihui
Date: 2026-03-13
"""
import torch
import torch.nn as nn
from typing import Optional
from .base import FakeQuantizeBase
from autoquant.core import QuantDtype, QScheme
from autoquant.observer import ObserverBase


class LSQFakeQuantize(FakeQuantizeBase):
    """
    LSQ (Learned Step Size Quantization)
    论文：https://arxiv.org/abs/1902.08153
    scale 作为可学习参数，用于 QAT 训练
    
    特点：
    - scale 是 nn.Parameter，通过反向传播学习
    - 使用自定义 autograd function 实现 STE（Straight-Through Estimator）
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
        super().__init__(
            observer=observer,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            ch_axis=ch_axis,
            enabled=enabled,
        )
        self.scale = None
        self.zero_point = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，执行 LSQ 量化"""
        if not self.enabled:
            return x

        self.observer(x)

        if self.scale is None:
            self.calculate_qparams()
            self.scale = nn.Parameter(self.scale) if not isinstance(self.scale, nn.Parameter) else self.scale

        if self.zero_point is None:
            if self.observer.zero_point is None:
                self.calculate_qparams()
            self.zero_point = self.observer.zero_point

        qmin = self.observer.quant_min
        qmax = self.observer.quant_max

        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            scale = self.scale.view([1] * x.dim())
            scale[self.ch_axis] = -1
            zero_point = self.zero_point.view([1] * x.dim())
            zero_point[self.ch_axis] = -1
        else:
            scale = self.scale
            zero_point = self.zero_point

        from autoquant.core import lsq_quantize
        return lsq_quantize(x, scale, zero_point, qmin, qmax)
