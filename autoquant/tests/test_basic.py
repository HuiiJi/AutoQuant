"""
基本功能测试
"""
import torch
import torch.nn as nn
import unittest
from autoquant.observer import MinMaxObserver
from autoquant.fake_quant import FakeQuantize
from autoquant.core import QuantDtype, QScheme
from autoquant.utils import get_default_qconfig


class TestObserver(unittest.TestCase):
    """测试Observer"""
    
    def test_min_max_observer(self):
        """测试MinMaxObserver"""
        observer = MinMaxObserver(
            dtype=QuantDtype.QUINT8,
            qscheme=QScheme.PER_TENSOR_AFFINE,
        )
        
        # 前向传播
        x = torch.randn(1, 3, 10, 10)
        observer(x)
        
        # 验证统计量
        self.assertIsNotNone(observer.min_val)
        self.assertIsNotNone(observer.max_val)
        
        # 计算量化参数
        scale, zero_point = observer.calculate_qparams()
        self.assertIsNotNone(scale)
        self.assertIsNotNone(zero_point)


class TestFakeQuant(unittest.TestCase):
    """测试FakeQuantize"""
    
    def test_fake_quantize(self):
        """测试FakeQuantize"""
        fq = FakeQuantize(
            dtype=QuantDtype.QUINT8,
            qscheme=QScheme.PER_TENSOR_AFFINE,
        )
        
        # 前向传播
        x = torch.randn(1, 3, 10, 10, requires_grad=True)
        x_q = fq(x)
        
        # 验证输出形状相同
        self.assertEqual(x.shape, x_q.shape)
        
        # 验证可以反向传播
        loss = x_q.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestQConfig(unittest.TestCase):
    """测试QConfig"""
    
    def test_get_default_qconfig(self):
        """测试获取默认配置"""
        qconfig = get_default_qconfig()
        self.assertIsNotNone(qconfig)
        self.assertIsNotNone(qconfig.activation)
        self.assertIsNotNone(qconfig.weight)


class TestSimpleModel(unittest.TestCase):
    """测试简单模型"""
    
    def test_simple_model(self):
        """测试简单的CNN模型"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                return x
        
        model = SimpleModel()
        x = torch.randn(1, 3, 10, 10)
        output = model(x)
        self.assertEqual(output.shape, (1, 16, 8, 8))


if __name__ == "__main__":
    unittest.main()
