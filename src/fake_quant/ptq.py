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
        """
        前向传播
        """
        # 获取 qmin/qmax
        qmin = self.observer.quant_min if self.observer else self.quant_min
        qmax = self.observer.quant_max if self.observer else self.quant_max

        # ==========================================
        # 核心逻辑修改：支持校准态临时量化
        # ==========================================

        # 1. 如果已经固化了参数 (Convert 之后)，直接用固化参数
        if self.scale is not None and self.zero_point is not None:
            # if self.observer and self.observer.enabled:
            #     self.observer(x)
            scale = self.scale
            zero_point = self.zero_point

        # 2. 如果还没固化参数 (Prepare/Calibration 阶段)，但 Observer 开着
        elif self.observer and self.observer.enabled:
            # A. 先让 Observer 统计当前数据 (更新 min/max 或直方图)
            self.observer(x)

            # B. 【关键】从 Observer 获取“临时”的 qparams
            # 注意：这里需要你的 Observer 支持随时计算当前状态的 qparams
            # 即使还没有调用最终的 calculate_qparams，Observer 内部应该有当前的 min_val/max_val
            temp_scale, temp_zero_point = self.observer.calculate_qparams()

            # 如果 Observer 还没统计出有效值 (比如第一步)，则无法量化，直接返回
            if temp_scale is None:
                return x

            scale = temp_scale
            zero_point = temp_zero_point

        # ==========================================
        # 确保 scale 和 zero_point 没有梯度 (detach)
        # ==========================================
        if isinstance(scale, torch.Tensor):
            scale = scale.detach()
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.detach()

        # ==========================================
        # 执行 Fake Quantize (统一出口)
        # ==========================================
        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            # 确保 zero_point 类型正确，ONNX 导出通常需要 int32
            if zero_point.dtype not in [torch.int32, torch.float32]:
                zero_point = zero_point.to(torch.int32)

            # 注意：per_channel 的 scale 和 zero_point 需要是 1D Tensor
            return torch.fake_quantize_per_channel_affine(
                x, scale=scale, zero_point=zero_point, axis=self.ch_axis, quant_min=qmin, quant_max=qmax
            )
        else:
            # per_tensor 的 scale 和 zero_point 通常是标量 (0D Tensor)
            return torch.fake_quantize_per_tensor_affine(
                x, scale=scale, zero_point=zero_point, quant_min=qmin, quant_max=qmax
            )
