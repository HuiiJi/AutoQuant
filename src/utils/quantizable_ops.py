"""
可量化操作工具函数

提供统一的接口来判断和管理哪些 PyTorch 模块可以被量化
支持用户自定义扩展可量化操作类型

Author: jihui
Date: 2026-03-23
"""
import torch.nn as nn
from typing import Set, Type, List, Optional


# ============================================================================
# 默认可量化操作类型
# ============================================================================

DEFAULT_QUANTIZABLE_OPS: Set[Type] = {
    # Conv 系列
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    
    # Linear 系列
    nn.Linear,
    nn.Bilinear,
    
    # Embedding 系列
    nn.Embedding,
    nn.EmbeddingBag,
}

# 对应的类型名称集合（用于字符串匹配）
DEFAULT_QUANTIZABLE_OP_NAMES: Set[str] = {
    'Conv1d', 'Conv2d', 'Conv3d',
    'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'Linear', 'Bilinear',
    'Embedding', 'EmbeddingBag',
}


# ============================================================================
# 核心工具函数
# ============================================================================

def get_quantizable_ops(
    include_extra: Optional[Set[str]] = None,
    exclude_ops: Optional[Set[str]] = None,
) -> Set[Type]:
    """
    获取可量化操作类型集合
    
    Args:
        include_extra: 额外包含的操作类型名称集合（例如：{'MaxPool2d', 'ReLU'}）
        exclude_ops: 要排除的操作类型名称集合（例如：{'Embedding'}）
    
    Returns:
        可量化操作类型集合
    
    Example:
        >>> # 获取默认的可量化操作
        >>> ops = get_quantizable_ops()
        
        >>> # 添加池化层支持
        >>> ops = get_quantizable_ops(include_extra={'MaxPool2d', 'AvgPool2d'})
        
        >>> # 排除 Embedding 层
        >>> ops = get_quantizable_ops(exclude_ops={'Embedding', 'EmbeddingBag'})
        
        >>> # 组合使用
        >>> ops = get_quantizable_ops(
        ...     include_extra={'MaxPool2d'},
        ...     exclude_ops={'Embedding'}
        ... )
    """
    # 复制默认集合
    ops = DEFAULT_QUANTIZABLE_OPS.copy()
    
    # 添加额外的操作类型
    if include_extra:
        for op_name in include_extra:
            # 从 nn 模块中获取对应的类型
            if hasattr(nn, op_name):
                op_type = getattr(nn, op_name)
                ops.add(op_type)
    
    # 排除指定的操作类型
    if exclude_ops:
        for op_name in exclude_ops:
            if hasattr(nn, op_name):
                op_type = getattr(nn, op_name)
                ops.discard(op_type)
    
    return ops


def get_quantizable_op_names(
    include_extra: Optional[Set[str]] = None,
    exclude_ops: Optional[Set[str]] = None,
) -> Set[str]:
    """
    获取可量化操作类型名称集合（字符串形式）
    
    Args:
        include_extra: 额外包含的操作类型名称集合
        exclude_ops: 要排除的操作类型名称集合
    
    Returns:
        可量化操作类型名称集合
    
    Example:
        >>> # 获取默认的可量化操作名称
        >>> names = get_quantizable_op_names()
        
        >>> # 添加池化层支持
        >>> names = get_quantizable_op_names(include_extra={'MaxPool2d', 'AvgPool2d'})
    """
    # 复制默认集合
    names = DEFAULT_QUANTIZABLE_OP_NAMES.copy()
    
    # 添加额外的操作类型
    if include_extra:
        names.update(include_extra)
    
    # 排除指定的操作类型
    if exclude_ops:
        names -= exclude_ops
    
    return names


def is_module_quantizable(
    module: nn.Module,
    include_extra: Optional[Set[str]] = None,
    exclude_ops: Optional[Set[str]] = None,
) -> bool:
    """
    判断模块是否可量化
    
    Args:
        module: 待判断的 PyTorch 模块
        include_extra: 额外包含的操作类型名称集合
        exclude_ops: 要排除的操作类型名称集合
    
    Returns:
        True 如果模块可量化，否则 False
    
    Example:
        >>> import torch.nn as nn
        >>> conv = nn.Conv2d(3, 16, 3)
        >>> linear = nn.Linear(10, 5)
        >>> relu = nn.ReLU()
        
        >>> is_module_quantizable(conv)  # True
        >>> is_module_quantizable(linear)  # True
        >>> is_module_quantizable(relu)  # False
        
        >>> # 添加 ReLU 支持
        >>> is_module_quantizable(relu, include_extra={'ReLU'})  # True
    """
    quantizable_ops = get_quantizable_ops(include_extra, exclude_ops)
    return isinstance(module, tuple(quantizable_ops))


def get_quantizable_layers(
    model: nn.Module,
    include_extra: Optional[Set[str]] = None,
    exclude_ops: Optional[Set[str]] = None,
) -> List[str]:
    """
    获取模型中所有可量化层的名称列表
    
    Args:
        model: PyTorch 模型
        include_extra: 额外包含的操作类型名称集合
        exclude_ops: 要排除的操作类型名称集合
    
    Returns:
        可量化层名称列表
    
    Example:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        
        >>> # 获取所有可量化层
        >>> layers = get_quantizable_layers(model)
        
        >>> # 添加池化层支持
        >>> layers = get_quantizable_layers(model, include_extra={'MaxPool2d'})
        
        >>> # 排除 Embedding 层
        >>> layers = get_quantizable_layers(model, exclude_ops={'Embedding'})
    """
    quantizable_ops = get_quantizable_ops(include_extra, exclude_ops)
    layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, tuple(quantizable_ops)) and name:
            layers.append(name)
    
    return layers


# ============================================================================
# 扩展工具函数
# ============================================================================

def add_quantizable_op(op_name: str):
    """
    动态添加可量化操作类型到默认集合
    
    ⚠️ 注意：这会修改全局默认集合，影响所有后续调用
    
    Args:
        op_name: 操作类型名称（必须在 torch.nn 模块中）
    
    Example:
        >>> # 添加 ReLU 支持
        >>> add_quantizable_op('ReLU')
        
        >>> # 添加池化层支持
        >>> add_quantizable_op('MaxPool2d')
        >>> add_quantizable_op('AvgPool2d')
    """
    if hasattr(nn, op_name):
        op_type = getattr(nn, op_name)
        DEFAULT_QUANTIZABLE_OPS.add(op_type)
        DEFAULT_QUANTIZABLE_OP_NAMES.add(op_name)
    else:
        raise ValueError(f"torch.nn 模块中没有找到操作类型：{op_name}")


def remove_quantizable_op(op_name: str):
    """
    从默认集合中移除可量化操作类型
    
    ⚠️ 注意：这会修改全局默认集合，影响所有后续调用
    
    Args:
        op_name: 操作类型名称
    
    Example:
        >>> # 移除 Embedding 支持
        >>> remove_quantizable_op('Embedding')
    """
    if hasattr(nn, op_name):
        op_type = getattr(nn, op_name)
        DEFAULT_QUANTIZABLE_OPS.discard(op_type)
        DEFAULT_QUANTIZABLE_OP_NAMES.discard(op_name)


def list_quantizable_ops() -> List[str]:
    """
    列出当前所有支持的可量化操作类型名称
    
    Returns:
        操作类型名称列表（按字母顺序排序）
    
    Example:
        >>> ops = list_quantizable_ops()
        >>> print(ops)
        ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', ...]
    """
    return sorted(DEFAULT_QUANTIZABLE_OP_NAMES)


# ============================================================================
# 预定义的常用扩展配置
# ============================================================================

def get_conv_only_ops() -> Set[Type]:
    """只返回 Conv 系列操作（不包括 Linear 和 Embedding）"""
    return {
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    }


def get_linear_only_ops() -> Set[Type]:
    """只返回 Linear 系列操作"""
    return {nn.Linear, nn.Bilinear}


def get_all_common_ops() -> Set[Type]:
    """
    返回所有常见的可量化操作（包括池化层和激活函数）
    
    这是一个更激进的配置，适用于需要更大量化的场景
    """
    ops = DEFAULT_QUANTIZABLE_OPS.copy()
    
    # 添加池化层
    pool_ops = {
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
    }
    ops.update(pool_ops)
    
    # 添加激活函数
    activation_ops = {
        nn.ReLU, nn.ReLU6,
        nn.LeakyReLU, nn.PReLU,
        nn.ELU, nn.SELU, nn.CELU,
        nn.GELU, nn.SiLU, nn.Hardswish,
    }
    ops.update(activation_ops)
    
    # 添加归一化层
    norm_ops = {
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    }
    ops.update(norm_ops)
    
    return ops
