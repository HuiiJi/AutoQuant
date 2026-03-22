"""
PACT FakeQuantize - QAT 专用
论文：https://arxiv.org/abs/1805.06085
Parameterized Clipping Activation，用于激活值的可学习裁剪

Author: jihui
Date: 2026-03-13
"""
import torch
import torch.nn as nn
from typing import Optional
from .base import FakeQuantizeBase
from autoquant.core import QuantDtype, QScheme
from autoquant.observer import ObserverBase


class PACTFakeQuantize(FakeQuantizeBase):
    """
    PACT (Parameterized Clipping Activation)
    论文：https://arxiv.org/abs/1805.06085
    用于激活值的可学习裁剪范围，主要用于 QAT 训练

    特点：
    - alpha 是 nn.Parameter，通过反向传播学习裁剪上界
    - 配合标准的 FakeQuant 一起使用
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
        init_alpha: float = 10.0,
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
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，执行 PACT 量化"""
        if not self.enabled:
            return x

        self.observer(x)

        if self.scale is None or self.zero_point is None:
            self.calculate_qparams()

        qmin = self.observer.quant_min
        qmax = self.observer.quant_max

        from autoquant.core import pact_quantize
        return pact_quantize(x, self.alpha, self.scale, self.zero_point, qmin, qmax)
