"""
推理引擎适配模块
为不同的推理引擎（TensorRT、ONNX Runtime、OpenVINO、MNN等）
提供最佳的QDQ ONNX配置
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
from autoquant.core import QuantDtype, QScheme
from autoquant.utils import QConfig


class InferenceEngine(Enum):
    """支持的推理引擎"""
    TENSORRT = "tensorrt"
    ONNXRUNTIME = "onnxruntime"
    OPENVINO = "openvino"
    MNN = "mnn"
    TFLITE = "tflite"
    COREML = "coreml"


@dataclass
class EngineConfig:
    """引擎配置"""
    engine: InferenceEngine
    activation_dtype: QuantDtype
    weight_dtype: QuantDtype
    activation_qscheme: QScheme
    weight_qscheme: QScheme
    activation_observer: str
    weight_observer: str
    supports_per_channel_activation: bool
    supports_per_channel_weight: bool
    supports_asymmetric_activation: bool
    supports_asymmetric_weight: bool
    recommended_qat_method: Optional[str] = None


# 各引擎的最佳配置
ENGINE_CONFIGS: Dict[InferenceEngine, EngineConfig] = {
    InferenceEngine.TENSORRT: EngineConfig(
        engine=InferenceEngine.TENSORRT,
        activation_dtype=QuantDtype.QUINT8,
        weight_dtype=QuantDtype.QINT8,
        activation_qscheme=QScheme.PER_TENSOR_AFFINE,
        weight_qscheme=QScheme.PER_CHANNEL_SYMMETRIC,
        activation_observer="histogram",
        weight_observer="minmax",
        supports_per_channel_activation=False,
        supports_per_channel_weight=True,
        supports_asymmetric_activation=True,
        supports_asymmetric_weight=False,
        recommended_qat_method="lsq",
    ),
    InferenceEngine.ONNXRUNTIME: EngineConfig(
        engine=InferenceEngine.ONNXRUNTIME,
        activation_dtype=QuantDtype.QUINT8,
        weight_dtype=QuantDtype.QINT8,
        activation_qscheme=QScheme.PER_TENSOR_AFFINE,
        weight_qscheme=QScheme.PER_CHANNEL_AFFINE,
        activation_observer="histogram",
        weight_observer="minmax",
        supports_per_channel_activation=False,
        supports_per_channel_weight=True,
        supports_asymmetric_activation=True,
        supports_asymmetric_weight=True,
        recommended_qat_method="lsq",
    ),
    InferenceEngine.OPENVINO: EngineConfig(
        engine=InferenceEngine.OPENVINO,
        activation_dtype=QuantDtype.QUINT8,
        weight_dtype=QuantDtype.QINT8,
        activation_qscheme=QScheme.PER_TENSOR_AFFINE,
        weight_qscheme=QScheme.PER_CHANNEL_AFFINE,
        activation_observer="minmax",
        weight_observer="minmax",
        supports_per_channel_activation=False,
        supports_per_channel_weight=True,
        supports_asymmetric_activation=True,
        supports_asymmetric_weight=True,
        recommended_qat_method=None,
    ),
    InferenceEngine.MNN: EngineConfig(
        engine=InferenceEngine.MNN,
        activation_dtype=QuantDtype.QUINT8,
        weight_dtype=QuantDtype.QINT8,
        activation_qscheme=QScheme.PER_TENSOR_SYMMETRIC,
        weight_qscheme=QScheme.PER_CHANNEL_SYMMETRIC,
        activation_observer="moving_avg",
        weight_observer="minmax",
        supports_per_channel_activation=False,
        supports_per_channel_weight=True,
        supports_asymmetric_activation=False,
        supports_asymmetric_weight=False,
        recommended_qat_method="lsq",
    ),
    InferenceEngine.TFLITE: EngineConfig(
        engine=InferenceEngine.TFLITE,
        activation_dtype=QuantDtype.QUINT8,
        weight_dtype=QuantDtype.QINT8,
        activation_qscheme=QScheme.PER_TENSOR_AFFINE,
        weight_qscheme=QScheme.PER_CHANNEL_AFFINE,
        activation_observer="minmax",
        weight_observer="minmax",
        supports_per_channel_activation=False,
        supports_per_channel_weight=True,
        supports_asymmetric_activation=True,
        supports_asymmetric_weight=True,
        recommended_qat_method=None,
    ),
    InferenceEngine.COREML: EngineConfig(
        engine=InferenceEngine.COREML,
        activation_dtype=QuantDtype.QUINT8,
        weight_dtype=QuantDtype.QINT8,
        activation_qscheme=QScheme.PER_TENSOR_AFFINE,
        weight_qscheme=QScheme.PER_CHANNEL_AFFINE,
        activation_observer="minmax",
        weight_observer="minmax",
        supports_per_channel_activation=False,
        supports_per_channel_weight=True,
        supports_asymmetric_activation=True,
        supports_asymmetric_weight=True,
        recommended_qat_method=None,
    ),
}


def get_engine_config(engine: str) -> EngineConfig:
    """
    获取指定推理引擎的配置
    
    Args:
        engine: 引擎名称，支持 'tensorrt', 'onnxruntime', 'openvino', 'mnn'
    
    Returns:
        EngineConfig对象
    """
    try:
        engine_enum = InferenceEngine(engine.lower())
        return ENGINE_CONFIGS[engine_enum]
    except ValueError:
        raise ValueError(f"不支持的推理引擎: {engine}. 支持的引擎: {[e.value for e in InferenceEngine]}")


def get_qconfig_for_engine(engine: str) -> QConfig:
    """
    获取指定推理引擎的最佳QConfig
    
    Args:
        engine: 引擎名称
    
    Returns:
        QConfig对象
    """
    from autoquant.utils import (
        get_default_qconfig,
        get_lsq_qconfig,
        get_histogram_qconfig,
    )
    
    config = get_engine_config(engine)
    
    # 根据推荐选择QAT方法
    if config.recommended_qat_method == "lsq":
        return get_lsq_qconfig(
            activation_dtype=config.activation_dtype,
            weight_dtype=config.weight_dtype,
            activation_qscheme=config.activation_qscheme,
            weight_qscheme=config.weight_qscheme,
        )
    elif config.activation_observer == "histogram":
        return get_histogram_qconfig(
            is_symmetric=not config.supports_asymmetric_activation,
        )
    else:
        return get_default_qconfig(
            activation_dtype=config.activation_dtype,
            weight_dtype=config.weight_dtype,
            activation_qscheme=config.activation_qscheme,
            weight_qscheme=config.weight_qscheme,
            observer_type=config.activation_observer,
        )


def get_supported_engines() -> list:
    """获取所有支持的推理引擎列表"""
    return [e.value for e in InferenceEngine]


def print_engine_info(engine: Optional[str] = None):
    """
    打印推理引擎信息
    
    Args:
        engine: 可选，指定引擎名称，不指定则打印所有引擎信息
    """
    if engine:
        config = get_engine_config(engine)
        _print_single_engine_info(config)
    else:
        for config in ENGINE_CONFIGS.values():
            _print_single_engine_info(config)
            print("-" * 60)


def _print_single_engine_info(config: EngineConfig):
    """打印单个引擎的信息"""
    # 将引擎名称转换为首字母大写的形式，如tensorrt -> TensorRT
    engine_name = config.engine.value.title().replace('_', '')
    print(f"推理引擎: {engine_name}")
    print(f"  激活值类型: {config.activation_dtype.name}")
    print(f"  权重类型: {config.weight_dtype.name}")
    print(f"  激活值量化方案: {config.activation_qscheme.name}")
    print(f"  权重量化方案: {config.weight_qscheme.name}")
    print(f"  支持激活值per-channel: {'是' if config.supports_per_channel_activation else '否'}")
    print(f"  支持权重per-channel: {'是' if config.supports_per_channel_weight else '否'}")
    print(f"  支持激活值非对称量化: {'是' if config.supports_asymmetric_activation else '否'}")
    print(f"  支持权重非对称量化: {'是' if config.supports_asymmetric_weight else '否'}")
    if config.recommended_qat_method:
        print(f"  推荐QAT方法: {config.recommended_qat_method.upper()}")
