"""
Observer基类定义
Observer用于在PTQ阶段统计数据分布，计算量化参数(scale, zero_point)
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from autoquant.core import QuantDtype, QScheme


class ObserverBase(nn.Module, ABC):
    """Observer基类"""

    def __init__(
        self,
        dtype: QuantDtype = QuantDtype.QINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        ch_axis: int = 0,
    ):
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.ch_axis = ch_axis
        self.torch_dtype = dtype.to_torch_dtype()
        self.torch_qscheme = qscheme.to_torch_qscheme()

        # 设置量化范围
        if quant_min is None:
            if dtype == QuantDtype.QUINT8:
                quant_min = 0
            elif dtype == QuantDtype.QINT8:
                quant_min = -128
        if quant_max is None:
            if dtype == QuantDtype.QUINT8:
                quant_max = 255
            elif dtype == QuantDtype.QINT8:
                quant_max = 127

        self.quant_min = quant_min
        self.quant_max = quant_max

        # 初始化统计量
        self._min_val = None
        self._max_val = None
        self._scale = None
        self._zero_point = None
        self.enabled = True

    @property
    def min_val(self):
        return self._min_val

    @min_val.setter
    def min_val(self, value):
        self._min_val = value

    @property
    def max_val(self):
        return self._max_val

    @max_val.setter
    def max_val(self, value):
        self._max_val = value

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    @property
    def zero_point(self):
        return self._zero_point

    @zero_point.setter
    def zero_point(self, value):
        self._zero_point = value

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，用于统计数据
        Args:
            x: 输入张量
        Returns:
            原始输入张量（不改变输入）
        """
        pass

    @abstractmethod
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算量化参数scale和zero_point
        Returns:
            (scale, zero_point)
        """
        pass

    def reset(self):
        """重置统计量"""
        self._min_val = None
        self._max_val = None
        self._scale = None
        self._zero_point = None

    def enable(self):
        """启用统计"""
        self.enabled = True

    def disable(self):
        """禁用统计"""
        self.enabled = False

    def extra_repr(self) -> str:
        return (
            f"dtype={self.dtype}, qscheme={self.qscheme}, "
            f"quant_min={self.quant_min}, quant_max={self.quant_max}, "
            f"ch_axis={self.ch_axis}, enabled={self.enabled}"
        )
