"""
FakeQuant模块 - 负责模拟量化误差，支持QAT
"""
from .base import FakeQuantizeBase
from .fake_quantize import FakeQuantize, FixedFakeQuantize, LSQFakeQuantize, PACTFakeQuantize

__all__ = [
    "FakeQuantizeBase",
    "FakeQuantize",
    "FixedFakeQuantize",
    "LSQFakeQuantize",
    "PACTFakeQuantize",
]
