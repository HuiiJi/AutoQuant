"""
高级API函数 - 提供便捷的prepare和convert接口
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
    leaf_modules: Optional[Set[str]] = None,
) -> nn.Module:
    """
    准备模型用于PTQ
    
    Args:
        model: 待量化的模型
        qconfig: 量化配置
        inplace: 是否原地修改模型
        leaf_modules: 需要量化的叶子节点名称集合（可选）
    
    Returns:
        准备好的量化模型
    """
    quantizer = ModelQuantizer(model, qconfig)
    return quantizer.prepare(inplace=inplace, leaf_modules=leaf_modules)


def prepare_qat(
    model: nn.Module,
    qconfig: QConfig,
    inplace: bool = False,
    leaf_modules: Optional[Set[str]] = None,
) -> nn.Module:
    """
    准备模型用于QAT
    
    Args:
        model: 待量化的模型
        qconfig: 量化配置
        inplace: 是否原地修改模型
        leaf_modules: 需要量化的叶子节点名称集合（可选）
    
    Returns:
        准备好的QAT模型
    """
    # QAT的prepare和PTQ类似，都需要插入fake quant
    quantizer = ModelQuantizer(model, qconfig)
    model_prepared = quantizer.prepare(inplace=inplace, leaf_modules=leaf_modules)
    # 设置为训练模式
    model_prepared.train()
    return model_prepared


def convert(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """
    将准备好的模型转换为量化模型
    
    Args:
        model: 准备好的模型
        inplace: 是否原地修改模型
    
    Returns:
        转换后的量化模型
    """
    # 这里我们需要创建一个临时的quantizer来调用convert
    # 实际项目中，可能需要更好的设计
    dummy_qconfig = None  # 这里只是占位
    quantizer = ModelQuantizer(model, dummy_qconfig)
    return quantizer.convert(model, inplace=inplace)


def calibrate(
    model: nn.Module,
    calib_data_loader,
    device: Optional[torch.device] = None,
):
    """
    校准模型（用于PTQ）
    
    Args:
        model: 准备好的量化模型
        calib_data_loader: 校准数据加载器
        device: 设备
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        for batch in calib_data_loader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device)
            model(inputs)
