"""
QConfig配置系统 - 定义量化配置
"""
from dataclasses import dataclass
from typing import Type
from autoquant.observer import (
    ObserverBase, 
    MinMaxObserver, 
    HistogramObserver, 
    MovingAverageMinMaxObserver,
    PercentileObserver,
    MSEObserver,
)
from autoquant.fake_quant import (
    FakeQuantizeBase, 
    FakeQuantize, 
    LSQFakeQuantize, 
    PACTFakeQuantize
)
from autoquant.core import QuantDtype, QScheme


@dataclass
class QConfig:
    """
    量化配置类
    包含activation和weight的量化配置
    """
    activation: Type[FakeQuantizeBase]
    weight: Type[FakeQuantizeBase]

    def __repr__(self) -> str:
        return f"QConfig(activation={self.activation.__name__}, weight={self.weight.__name__})"


def get_default_qconfig(
    activation_dtype: QuantDtype = QuantDtype.QUINT8,
    weight_dtype: QuantDtype = QuantDtype.QINT8,
    activation_qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
    weight_qscheme: QScheme = QScheme.PER_CHANNEL_AFFINE,
    observer_type: str = "minmax",
) -> QConfig:
    """
    获取默认的量化配置
    
    Args:
        activation_dtype: 激活值的量化数据类型
        weight_dtype: 权重的量化数据类型
        activation_qscheme: 激活值的量化方案
        weight_qscheme: 权重的量化方案
        observer_type: Observer类型，可选 'minmax', 'histogram', 'moving_avg', 'percentile', 'mse'
    
    Returns:
        QConfig对象
    """
    # 选择Observer类
    if observer_type == "minmax":
        observer_cls = MinMaxObserver
    elif observer_type == "histogram":
        observer_cls = HistogramObserver
    elif observer_type == "moving_avg":
        observer_cls = MovingAverageMinMaxObserver
    elif observer_type == "percentile":
        observer_cls = PercentileObserver
    elif observer_type == "mse":
        observer_cls = MSEObserver
    else:
        raise ValueError(f"不支持的observer类型: {observer_type}")

    # 定义activation的fake quantize
    def activation_fq():
        observer = observer_cls(
            dtype=activation_dtype,
            qscheme=activation_qscheme,
        )
        return FakeQuantize(observer=observer)

    # 定义weight的fake quantize
    def weight_fq():
        observer = observer_cls(
            dtype=weight_dtype,
            qscheme=weight_qscheme,
            ch_axis=0,
        )
        return FakeQuantize(observer=observer)

    return QConfig(activation=activation_fq, weight=weight_fq)


def get_per_channel_qconfig(
    is_symmetric: bool = False,
    observer_type: str = "minmax",
) -> QConfig:
    """
    获取按通道量化的配置
    
    Args:
        is_symmetric: 是否使用对称量化
        observer_type: Observer类型，可选 'minmax', 'histogram', 'moving_avg'
    
    Returns:
        QConfig对象
    """
    if is_symmetric:
        activation_qscheme = QScheme.PER_CHANNEL_SYMMETRIC
        weight_qscheme = QScheme.PER_CHANNEL_SYMMETRIC
    else:
        activation_qscheme = QScheme.PER_CHANNEL_AFFINE
        weight_qscheme = QScheme.PER_CHANNEL_AFFINE

    return get_default_qconfig(
        activation_qscheme=activation_qscheme,
        weight_qscheme=weight_qscheme,
        observer_type=observer_type,
    )


def get_per_tensor_qconfig(
    is_symmetric: bool = False,
    observer_type: str = "minmax",
) -> QConfig:
    """
    获取按张量量化的配置
    
    Args:
        is_symmetric: 是否使用对称量化
        observer_type: Observer类型，可选 'minmax', 'histogram', 'moving_avg'
    
    Returns:
        QConfig对象
    """
    if is_symmetric:
        activation_qscheme = QScheme.PER_TENSOR_SYMMETRIC
        weight_qscheme = QScheme.PER_TENSOR_SYMMETRIC
    else:
        activation_qscheme = QScheme.PER_TENSOR_AFFINE
        weight_qscheme = QScheme.PER_TENSOR_AFFINE

    return get_default_qconfig(
        activation_qscheme=activation_qscheme,
        weight_qscheme=weight_qscheme,
        observer_type=observer_type,
    )


def get_lsq_qconfig(
    activation_dtype: QuantDtype = QuantDtype.QUINT8,
    weight_dtype: QuantDtype = QuantDtype.QINT8,
    activation_qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
    weight_qscheme: QScheme = QScheme.PER_CHANNEL_AFFINE,
) -> QConfig:
    """
    获取LSQ（Learned Step Size Quantization）配置
    
    Args:
        activation_dtype: 激活值的量化数据类型
        weight_dtype: 权重的量化数据类型
        activation_qscheme: 激活值的量化方案
        weight_qscheme: 权重的量化方案
    
    Returns:
        QConfig对象
    """
    # 定义activation的LSQ fake quantize
    def activation_fq():
        observer = MinMaxObserver(
            dtype=activation_dtype,
            qscheme=activation_qscheme,
        )
        return LSQFakeQuantize(observer=observer)

    # 定义weight的LSQ fake quantize
    def weight_fq():
        observer = MinMaxObserver(
            dtype=weight_dtype,
            qscheme=weight_qscheme,
            ch_axis=0,
        )
        return LSQFakeQuantize(observer=observer)

    return QConfig(activation=activation_fq, weight=weight_fq)


def get_pact_qconfig(
    activation_dtype: QuantDtype = QuantDtype.QUINT8,
    weight_dtype: QuantDtype = QuantDtype.QINT8,
    activation_qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
    weight_qscheme: QScheme = QScheme.PER_CHANNEL_AFFINE,
    init_alpha: float = 10.0,
) -> QConfig:
    """
    获取PACT（Parameterized Clipping Activation）配置
    
    Args:
        activation_dtype: 激活值的量化数据类型
        weight_dtype: 权重的量化数据类型
        activation_qscheme: 激活值的量化方案
        weight_qscheme: 权重的量化方案
        init_alpha: PACT的alpha初始值
    
    Returns:
        QConfig对象
    """
    # 定义activation的PACT fake quantize
    def activation_fq():
        observer = MinMaxObserver(
            dtype=activation_dtype,
            qscheme=activation_qscheme,
        )
        return PACTFakeQuantize(observer=observer, init_alpha=init_alpha)

    # weight使用普通的FakeQuantize
    def weight_fq():
        observer = MinMaxObserver(
            dtype=weight_dtype,
            qscheme=weight_qscheme,
            ch_axis=0,
        )
        return FakeQuantize(observer=observer)

    return QConfig(activation=activation_fq, weight=weight_fq)


def get_histogram_qconfig(
    is_symmetric: bool = False,
) -> QConfig:
    """
    获取基于HistogramObserver的配置
    
    Args:
        is_symmetric: 是否使用对称量化
    
    Returns:
        QConfig对象
    """
    return get_per_channel_qconfig(
        is_symmetric=is_symmetric,
        observer_type="histogram",
    )

