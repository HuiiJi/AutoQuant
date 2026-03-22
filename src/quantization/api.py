"""
高级 API 函数 - 提供便捷的量化接口

Author: jihui
Date: 2026-03-13
"""
import torch
import torch.nn as nn
import copy
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
    verbose: bool = True,
):
    """
    校准模型（用于 PTQ）

    Args:
        model: 准备好的量化模型
        calib_data: 校准数据（DataLoader、list[Tensor] 或单个 Tensor）
        device: 设备
        verbose: 是否打印日志
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        if isinstance(calib_data, torch.utils.data.DataLoader):
            for i, batch in enumerate(calib_data):
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                inputs = inputs.to(device)
                model(inputs)
                if verbose and (i + 1) % 10 == 0:
                    print(f"  已处理 {i + 1} 批")
        elif isinstance(calib_data, list):
            for i, inputs in enumerate(calib_data):
                inputs = inputs.to(device)
                model(inputs)
                if verbose and (i + 1) % 10 == 0:
                    print(f"  已处理 {i + 1} 个样本")
        elif isinstance(calib_data, torch.Tensor):
            inputs = calib_data.to(device)
            model(inputs)
            if verbose:
                print("  已处理单样本")
        else:
            raise ValueError(f"不支持的校准数据类型: {type(calib_data)}")


def convert(
    model: nn.Module,
    inplace: bool = False,
    permanently_quantize_weight: bool = False,
) -> nn.Module:
    """
    将准备好的模型转换为量化模型

    Args:
        model: 准备好的模型
        inplace: 是否原地修改
        permanently_quantize_weight: 是否永久量化 weight（默认False）

    Returns:
        转换后的量化模型
    """
    # 尝试递归转换所有模块
    def _convert_recursive(m):
        if hasattr(m, 'convert') and callable(m.convert):
            return m.convert(permanently_quantize_weight=permanently_quantize_weight)
        for name, child in m.named_children():
            setattr(m, name, _convert_recursive(child))
        return m

    if not inplace:
        model = copy.deepcopy(model)

    return _convert_recursive(model)


def ptq(
    model: nn.Module,
    qconfig: QConfig,
    calib_data,
    device: Optional[torch.device] = None,
    inplace: bool = False,
    skip_layers: Optional[Set[str]] = None,
    permanently_quantize_weight: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    完整的 PTQ 流程：prepare → calibrate → convert

    Args:
        model: 待量化的模型
        qconfig: 量化配置
        calib_data: 校准数据
        device: 设备
        inplace: 是否原地修改
        skip_layers: 跳过量化的层
        permanently_quantize_weight: 是否永久量化 weight
        verbose: 是否打印日志

    Returns:
        量化后的模型
    """
    if verbose:
        print("=" * 60)
        print("🚀 开始 PTQ 量化流程")
        print("=" * 60)

    # 1. Prepare
    if verbose:
        print("\n[1/3] 准备模型...")
    quantizer = ModelQuantizer(model, qconfig)
    prepared_model = quantizer.prepare(inplace=inplace, skip_layers=skip_layers)

    # 2. Calibrate
    if verbose:
        print("\n[2/3] 校准模型...")
    quantizer.calibrate(calib_data, device=device, verbose=verbose)

    # 3. Convert
    if verbose:
        print("\n[3/3] 转换为推理模式...")
    quantized_model = quantizer.convert(
        inplace=True,
        permanently_quantize_weight=permanently_quantize_weight
    )

    if verbose:
        print("\n" + "=" * 60)
        print("✅ PTQ 量化完成！")
        print("=" * 60)

    return quantized_model
