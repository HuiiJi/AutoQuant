"""
QConfig配置系统 - 定义量化配置
直接绑定 TensorRT 和 ONNX Runtime 的最佳方案

Author: jihui
Date: 2026-03-13
Desc:
    - get_trt_qconfig(): TensorRT 最佳精度方案
    - get_ort_qconfig(): ONNX Runtime 最佳精度方案
    - get_default_qconfig(): 默认使用 TRT 方案
"""
from dataclasses import dataclass
from typing import Type, Callable, Any
from functools import partial
from autoquant.observer import (
    ObserverBase,
    MinMaxObserver,
    HistogramObserver,
    MovingAverageMinMaxObserver,
)
from autoquant.fake_quant import (
    FakeQuantizeBase,
    PTQFakeQuantize,
    LSQFakeQuantize,
)
from autoquant.core import QuantDtype, QScheme


@dataclass
class QConfig:
    """
    量化配置类
    包含 activation 和 weight 的量化配置
    """
    activation: Callable[[], FakeQuantizeBase]
    weight: Callable[[], FakeQuantizeBase]

    def __repr__(self) -> str:
        return f"QConfig(activation={getattr(self.activation, '__name__', self.activation)}, " \
            f"weight={getattr(self.weight, '__name__', self.weight)})"


def get_trt_qconfig() -> QConfig:
    """
    获取 TensorRT 最佳精度的量化配置

    TensorRT 最佳实践：
    - Activation: PER_TENSOR_SYMMETRIC + MinMaxObserver
    - Weight: PER_CHANNEL_SYMMETRIC + MinMaxObserver

    Returns:
        QConfig对象
    """
    # Activation: PER_TENSOR_SYMMETRIC + MinMaxObserver
    def activation_fq():
        observer = MinMaxObserver(
            dtype=QuantDtype.QUINT8,
            qscheme=QScheme.PER_TENSOR_SYMMETRIC,
        )
        return PTQFakeQuantize(observer=observer)

    # Weight: PER_CHANNEL_SYMMETRIC + MinMaxObserver
    def weight_fq():
        observer = MinMaxObserver(
            dtype=QuantDtype.QINT8,
            qscheme=QScheme.PER_CHANNEL_SYMMETRIC,
            ch_axis=0,
        )
        return PTQFakeQuantize(observer=observer)

    return QConfig(activation=activation_fq, weight=weight_fq)


def get_ort_qconfig() -> QConfig:
    """
    获取 ONNX Runtime 最佳精度的量化配置

    ONNX Runtime 最佳实践：
    - Activation: PER_TENSOR_SYMMETRIC + MovingAverageMinMaxObserver
    - Weight: PER_CHANNEL_SYMMETRIC + HistogramObserver (ONNX Runtime QLinearConv 要求对称)

    Returns:
        QConfig对象
    """
    # Activation: PER_TENSOR_SYMMETRIC + MovingAverageMinMaxObserver
    def activation_fq():
        observer = MovingAverageMinMaxObserver(
            dtype=QuantDtype.QUINT8,
            qscheme=QScheme.PER_TENSOR_SYMMETRIC,
        )
        return PTQFakeQuantize(observer=observer)

    # Weight: PER_CHANNEL_SYMMETRIC + HistogramObserver
    def weight_fq():
        observer = HistogramObserver(
            dtype=QuantDtype.QINT8,
            qscheme=QScheme.PER_CHANNEL_SYMMETRIC,
            ch_axis=0,
        )
        return PTQFakeQuantize(observer=observer)

    return QConfig(activation=activation_fq, weight=weight_fq)


def get_default_qconfig(
    activation_dtype: QuantDtype = QuantDtype.QUINT8,
    weight_dtype: QuantDtype = QuantDtype.QINT8,
    activation_qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
    weight_qscheme: QScheme = QScheme.PER_CHANNEL_AFFINE,
    activation_observer_type: str = "histogram",
    weight_observer_type: str = "minmax",
) -> QConfig:
    """
    获取默认的量化配置（参数化配置）

    Args:
        activation_dtype: 激活值量化数据类型
        weight_dtype: 权重量化数据类型
        activation_qscheme: 激活值量化方案
        weight_qscheme: 权重量化方案
        activation_observer_type: 激活值observer类型 - "minmax", "histogram", 或 "moving_avg"
        weight_observer_type: 权重observer类型

    Returns:
        QConfig对象
    """
    def get_observer_class(observer_type: str):
        return {
            "minmax": MinMaxObserver,
            "histogram": HistogramObserver,
            "moving_average": MovingAverageMinMaxObserver,
        }.get(observer_type.lower(), HistogramObserver)

    ActivationObserverClass = get_observer_class(activation_observer_type)
    WeightObserverClass = get_observer_class(weight_observer_type)

    def activation_fq():
        observer = ActivationObserverClass(
            dtype=activation_dtype,
            qscheme=activation_qscheme,
        )
        return PTQFakeQuantize(observer=observer)

    def weight_fq():
        observer = WeightObserverClass(
            dtype=weight_dtype,
            qscheme=weight_qscheme,
            ch_axis=0,
        )
        return PTQFakeQuantize(observer=observer)

    return QConfig(activation=activation_fq, weight=weight_fq)


def get_lsq_qconfig(
    activation_dtype: QuantDtype = QuantDtype.QUINT8,
    weight_dtype: QuantDtype = QuantDtype.QINT8,
    activation_qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
    weight_qscheme: QScheme = QScheme.PER_CHANNEL_AFFINE,
) -> QConfig:
    """
    获取 LSQ（Learned Step Size Quantization）配置（QAT 专用）

    Args:
        activation_dtype: 激活值的量化数据类型
        weight_dtype: 权重的量化数据类型
        activation_qscheme: 激活值的量化方案
        weight_qscheme: 权重的量化方案

    Returns:
        QConfig对象
    """
    def activation_fq():
        observer = MinMaxObserver(
            dtype=activation_dtype,
            qscheme=activation_qscheme,
        )
        return LSQFakeQuantize(observer=observer)

    def weight_fq():
        observer = MinMaxObserver(
            dtype=weight_dtype,
            qscheme=weight_qscheme,
            ch_axis=0,
        )
        return LSQFakeQuantize(observer=observer)

    return QConfig(activation=activation_fq, weight=weight_fq)
