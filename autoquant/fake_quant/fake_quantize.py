"""
FakeQuantize实现 - 包含多种高级量化方法
"""
import torch
import torch.nn as nn
from typing import Optional
from .base import FakeQuantizeBase
from autoquant.core import (
    QuantDtype, 
    QScheme, 
    fake_quantize_ste, 
    lsq_quantize, 
    pact_quantize
)
from autoquant.observer import ObserverBase, MinMaxObserver


class FakeQuantize(FakeQuantizeBase):
    """
    FakeQuantize：实现模拟量化操作，使用自定义Autograd Function
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
        # 如果提供了observer，使用observer的参数
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
        """
        前向传播，执行模拟量化
        """
        if not self.enabled:
            return x

        # 更新observer统计（在训练模式或PTQ校准阶段）
        # 即使在评估模式下，PTQ校准也需要更新observer
        self.observer(x)

        # 如果scale和zero_point未计算，从observer获取
        if self.scale is None or self.zero_point is None:
            # 只有在训练模式或PTQ阶段才计算参数
            # 避免在推理时重复计算
            self.calculate_qparams()

        # 执行模拟量化
        x_q = self._fake_quantize(x)
        return x_q

    def _fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行模拟量化的核心逻辑，使用自定义Autograd Function
        """
        scale = self.scale
        zero_point = self.zero_point
        qmin = self.observer.quant_min
        qmax = self.observer.quant_max

        # 需要确保scale和zero_point的形状与输入匹配
        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            shape = [1] * x.dim()
            shape[self.ch_axis] = -1
            scale = scale.view(shape)
            zero_point = zero_point.view(shape)

        # 使用自定义Autograd Function
        return fake_quantize_ste(x, scale, zero_point, qmin, qmax)


class LSQFakeQuantize(FakeQuantizeBase):
    """
    LSQ (Learned Step Size Quantization)
    论文：https://arxiv.org/abs/1902.08153
    scale作为可学习参数
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
        init_scale: Optional[torch.Tensor] = None,
    ):
        # 如果提供了observer，使用observer的参数
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
        
        # LSQ的scale是可学习参数
        if init_scale is not None:
            self.scale = nn.Parameter(init_scale)
        else:
            self.scale = None
        
        self.zero_point = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，执行LSQ量化
        """
        if not self.enabled:
            return x

        # 更新observer统计（在训练模式或PTQ校准阶段）
        self.observer(x)

        # 初始化scale（如果是第一次）
        if self.scale is None:
            self.calculate_qparams()
            # 将scale转换为可学习参数
            self.scale = nn.Parameter(self.scale)

        # 如果还没有zero_point，先计算
        if self.zero_point is None:
            if self.observer.zero_point is None:
                self.calculate_qparams()
            self.zero_point = self.observer.zero_point

        qmin = self.observer.quant_min
        qmax = self.observer.quant_max

        # 调整形状
        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            shape = [1] * x.dim()
            shape[self.ch_axis] = -1
            scale = self.scale.view(shape)
            zero_point = self.zero_point.view(shape)
        else:
            scale = self.scale
            zero_point = self.zero_point

        # 使用LSQ的Autograd Function
        return lsq_quantize(x, scale, zero_point, qmin, qmax)


class PACTFakeQuantize(FakeQuantizeBase):
    """
    PACT (Parameterized Clipping Activation)
    论文：https://arxiv.org/abs/1805.06085
    用于激活值的可学习裁剪范围
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
        # 如果提供了observer，使用observer的参数
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
        
        # PACT的alpha是可学习参数（裁剪上限）
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，执行PACT量化
        """
        if not self.enabled:
            return x

        # 更新observer统计（在训练模式或PTQ校准阶段）
        self.observer(x)

        # 计算scale和zero_point
        if self.scale is None or self.zero_point is None:
            self.calculate_qparams()

        qmin = self.observer.quant_min
        qmax = self.observer.quant_max

        # 使用PACT的Autograd Function
        return pact_quantize(x, self.alpha, self.scale, self.zero_point, qmin, qmax)


class FixedFakeQuantize(FakeQuantize):
    """
    FixedFakeQuantize：固定scale和zero_point的FakeQuantize
    用于PTQ转换后或QAT训练后
    """

    def __init__(
        self,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        dtype: QuantDtype = QuantDtype.QUINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        ch_axis: int = 0,
        enabled: bool = True,
    ):
        super().__init__(
            observer=None,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            ch_axis=ch_axis,
            enabled=enabled,
        )
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，执行固定参数的模拟量化
        """
        if not self.enabled:
            return x

        # 直接执行模拟量化，不更新observer
        x_q = self._fake_quantize(x)
        return x_q
