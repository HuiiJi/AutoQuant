"""
FakeQuant模块 - 负责模拟量化误差
PTQ：后训练量化（不训练scale/zero_point）
QAT：量化感知训练（训练scale/zero_point）

Author: jihui
Date: 2026-03-13
"""
from .base import FakeQuantizeBase
from .ptq import PTQFakeQuantize
from .lsq import LSQFakeQuantize
from .pact import PACTFakeQuantize

__all__ = [
    "FakeQuantizeBase",
    "PTQFakeQuantize",
    "LSQFakeQuantize",
    "PACTFakeQuantize",
]
