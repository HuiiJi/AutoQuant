"""
高级 API 函数 - 提供便捷的量化接口

Author: jihui
Date: 2026-03-13
"""
import torch
import torch.nn as nn
from typing import Optional, Set
from autoquant.utils import QConfig
from autoquant.quantization.model_quantizer import ModelQuantizer


def prepare(
    model: nn.Module,
    qconfig: QConfig,
    inplace: bool = False,
    skip_layers: Optional[Set[str]] = None,
) -> nn.Module:
    """
    准备模型用于 PTQ
    
    Args:
        model: 待量化的模型
        qconfig: 量化配置
        inplace: 是否原地修改模型
        skip_layers: 跳过量化的层名称集合，如 {'layer1.0.conv1', 'fc'}
    
    Returns:
        准备好的量化模型
    """
    quantizer = ModelQuantizer(model, qconfig)
    return quantizer.prepare(inplace=inplace, skip_layers=skip_layers)


def prepare_qat(
    model: nn.Module,
    qconfig: QConfig,
    inplace: bool = False,
    skip_layers: Optional[Set[str]] = None,
) -> nn.Module:
    """
    准备模型用于 QAT
    
    Args:
        model: 待量化的模型
        qconfig: 量化配置
        inplace: 是否原地修改模型
        skip_layers: 跳过量化的层名称集合，如 {'layer1.0.conv1', 'fc'}
    
    Returns:
        准备好的 QAT 模型
    """
    quantizer = ModelQuantizer(model, qconfig)
    model_prepared = quantizer.prepare(inplace=inplace, skip_layers=skip_layers)
    model_prepared.train()
    return model_prepared


def calibrate(
    model: nn.Module,
    calib_data,
    device: Optional[torch.device] = None,
):
    """
    校准模型（用于 PTQ）
    
    Args:
        model: 准备好的量化模型
        calib_data: 校准数据（DataLoader、list[Tensor] 或单个 Tensor）
        device: 设备
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        if isinstance(calib_data, torch.utils.data.DataLoader):
            for batch in calib_data:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                inputs = inputs.to(device)
                model(inputs)
        elif isinstance(calib_data, list):
            for inputs in calib_data:
                inputs = inputs.to(device)
                model(inputs)
        elif isinstance(calib_data, torch.Tensor):
            inputs = calib_data.to(device)
            model(inputs)
        else:
            raise ValueError(f"不支持的校准数据类型: {type(calib_data)}")


def convert(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """
    将准备好的模型转换为量化模型
    
    注意：这个函数需要配合 ModelQuantizer 使用
    
    Args:
        model: 准备好的模型（必须来自 ModelQuantizer.prepare()）
        inplace: 是否原地修改
    
    Returns:
        转换后的量化模型
    """
    # 这里我们需要找到模型中的 QuantizableModule 并转换它们
    # 简化实现：直接在模型上调用 convert 方法（如果有的话）
    # 实际上，应该使用 ModelQuantizer.convert()
    
    # 尝试递归转换所有模块
    def _convert_recursive(m):
        if hasattr(m, 'convert') and callable(m.convert):
            return m.convert()
        for name, child in m.named_children():
            setattr(m, name, _convert_recursive(child))
        return m
    
    if not inplace:
        model = copy.deepcopy(model)
    
    return _convert_recursive(model)
