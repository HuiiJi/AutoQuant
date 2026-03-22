"""
FakeQuant基类定义
FakeQuant用于QAT阶段模拟量化误差，保留梯度传递

Author: jihui
Date: 2026-03-13
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional
from autoquant.core import QuantDtype, QScheme
from autoquant.observer import ObserverBase, MinMaxObserver


class FakeQuantizeBase(nn.Module, ABC):
    """FakeQuantize基类"""

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
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.ch_axis = ch_axis
        self.enabled = enabled
        self.observer = observer
        self._scale = None
        self._zero_point = None

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if value is not None and isinstance(value, torch.Tensor):
            # 如果有值，注册为 buffer
            if not hasattr(self, '_scale_buffer'):
                self.register_buffer('_scale_buffer', value)
            else:
                self._scale_buffer = value
        self._scale = value

    @property
    def zero_point(self):
        return self._zero_point

    @zero_point.setter
    def zero_point(self, value):
        if value is not None and isinstance(value, torch.Tensor):
            # 如果有值，注册为 buffer
            if not hasattr(self, '_zero_point_buffer'):
                self.register_buffer('_zero_point_buffer', value)
            else:
                self._zero_point_buffer = value
        self._zero_point = value

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，执行模拟量化
        Args:
            x: 输入张量
        Returns:
            模拟量化后的张量
        """
        pass

    def calculate_qparams(self):
        """从observer获取量化参数"""
        self.observer.calculate_qparams()

        # 如果 observer 没有统计数据，直接返回
        if self.observer.scale is None or self.observer.zero_point is None:
            return

        # 处理scale
        if isinstance(self.observer.scale, torch.Tensor):
            scale_value = self.observer.scale.clone().detach()
        else:
            scale_value = torch.tensor(self.observer.scale).detach()

        # 处理zero_point
        if isinstance(self.observer.zero_point, torch.Tensor):
            zero_point_value = self.observer.zero_point.clone().detach()
        else:
            zero_point_value = torch.tensor(self.observer.zero_point).detach()

        self.scale = scale_value
        self.zero_point = zero_point_value

    def disable_observer(self):
        """禁用observer统计"""
        if hasattr(self.observer, 'disable'):
            self.observer.disable()

    def enable_observer(self):
        """启用observer统计"""
        if hasattr(self.observer, 'enable'):
            self.observer.enable()

    def disable_fake_quant(self):
        """禁用fake quant"""
        self.enabled = False

    def enable_fake_quant(self):
        """启用fake quant"""
        self.enabled = True

    def extra_repr(self) -> str:
        return (
            f"dtype={self.dtype}, qscheme={self.qscheme}, "
            f"ch_axis={self.ch_axis}, enabled={self.enabled}"
        )
