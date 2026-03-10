"""
数据类型定义
"""
import torch
from enum import Enum


class QuantDtype(Enum):
    """量化数据类型"""
    QUINT8 = "quint8"
    QINT8 = "qint8"
    FLOAT16 = "float16"
    FLOAT32 = "float32"

    def to_torch_dtype(self):
        """转换为torch数据类型"""
        if self == QuantDtype.QUINT8:
            return torch.quint8
        elif self == QuantDtype.QINT8:
            return torch.qint8
        elif self == QuantDtype.FLOAT16:
            return torch.float16
        elif self == QuantDtype.FLOAT32:
            return torch.float32
        else:
            raise ValueError(f"不支持的数据类型: {self}")


class QScheme(Enum):
    """量化方案"""
    PER_TENSOR_AFFINE = "per_tensor_affine"
    PER_TENSOR_SYMMETRIC = "per_tensor_symmetric"
    PER_CHANNEL_AFFINE = "per_channel_affine"
    PER_CHANNEL_SYMMETRIC = "per_channel_symmetric"

    def to_torch_qscheme(self):
        """转换为torch量化方案"""
        if self == QScheme.PER_TENSOR_AFFINE:
            return torch.per_tensor_affine
        elif self == QScheme.PER_TENSOR_SYMMETRIC:
            return torch.per_tensor_symmetric
        elif self == QScheme.PER_CHANNEL_AFFINE:
            return torch.per_channel_affine
        elif self == QScheme.PER_CHANNEL_SYMMETRIC:
            return torch.per_channel_symmetric
        else:
            raise ValueError(f"不支持的量化方案: {self}")
