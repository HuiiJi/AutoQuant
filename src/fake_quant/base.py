"""
FakeQuant基类定义
FakeQuant用于QAT阶段模拟量化误差，保留梯度传递
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
        
        # 如果没有提供observer，使用默认的MinMaxObserver
        if observer is None:
            observer = MinMaxObserver(
                dtype=dtype,
                qscheme=qscheme,
                quant_min=quant_min,
                quant_max=quant_max,
                ch_axis=ch_axis,
            )
        self.observer = observer
        
        # 量化参数
        self.register_buffer("scale", None)
        self.register_buffer("zero_point", None)

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
        self.scale = self.observer.scale
        self.zero_point = self.observer.zero_point

    def disable_observer(self):
        """禁用observer统计"""
        self.observer.eval()

    def enable_observer(self):
        """启用observer统计"""
        self.observer.train()

    def disable_fake_quant(self):
        """禁用fake quant"""
        self.enabled = False

    def enable_fake_quant(self):
        """启用fake quant"""
        self.enabled = True

    def disable(self):
        """禁用fake quant（别名）"""
        self.disable_fake_quant()

    def enable(self):
        """启用fake quant（别名）"""
        self.enable_fake_quant()

    def extra_repr(self) -> str:
        return (
            f"dtype={self.dtype}, qscheme={self.qscheme}, "
            f"ch_axis={self.ch_axis}, enabled={self.enabled}"
        )
