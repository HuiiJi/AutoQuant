"""
测试FakeQuantize和Autograd函数
"""
import pytest
import torch
import sys
sys.path.insert(0, '.')

from autoquant import (
    round_ste,
    clamp_grad,
    fake_quantize_ste,
    lsq_quantize,
    pact_quantize,
    FakeQuantize,
    LSQFakeQuantize,
    PACTFakeQuantize,
    MinMaxObserver,
    QuantDtype,
    QScheme,
)


class TestAutogradFunctions:
    """测试Autograd函数"""
    
    def test_round_ste(self):
        """测试round_ste"""
        x = torch.tensor([1.3, 1.6, 2.1], requires_grad=True)
        y = round_ste(x)
        
        # 前向测试
        assert torch.all(y == torch.tensor([1.0, 2.0, 2.0]))
        
        # 反向测试
        y.sum().backward()
        assert x.grad is not None
        assert torch.all(x.grad == 1.0)
    
    def test_clamp_grad(self):
        """测试clamp_grad"""
        x = torch.tensor([-2.0, 0.5, 2.0], requires_grad=True)
        y = clamp_grad(x, -1.0, 1.0)
        
        # 前向测试
        expected = torch.tensor([-1.0, 0.5, 1.0])
        assert torch.allclose(y, expected)
        
        # 反向测试
        y.sum().backward()
        assert x.grad is not None
        expected_grad = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(x.grad, expected_grad)
    
    def test_fake_quantize_ste(self):
        """测试fake_quantize_ste"""
        x = torch.tensor([-1.5, 0.0, 1.5], requires_grad=True)
        scale = torch.tensor([0.5])
        zero_point = torch.tensor([2])
        
        y = fake_quantize_ste(x, scale, zero_point, 0, 4)
        
        assert y is not None
        y.sum().backward()
        assert x.grad is not None
    
    def test_lsq_quantize(self):
        """测试lsq_quantize"""
        x = torch.tensor([-1.5, 0.0, 1.5], requires_grad=True)
        scale = torch.tensor([0.5], requires_grad=True)
        zero_point = torch.tensor([2])
        
        y = lsq_quantize(x, scale, zero_point, 0, 4)
        
        assert y is not None
        y.sum().backward()
        assert x.grad is not None
        assert scale.grad is not None
    
    def test_pact_quantize(self):
        """测试pact_quantize"""
        x = torch.tensor([-1.0, 0.0, 1.0, 5.0], requires_grad=True)
        alpha = torch.tensor([3.0], requires_grad=True)
        scale = torch.tensor([0.5])
        zero_point = torch.tensor([0])
        
        y = pact_quantize(x, alpha, scale, zero_point, 0, 6)
        
        assert y is not None
        y.sum().backward()
        assert x.grad is not None
        assert alpha.grad is not None


class TestFakeQuantize:
    """测试FakeQuantize类"""
    
    @pytest.fixture
    def test_data(self):
        return torch.randn(2, 3, 16, 16)
    
    def test_basic_fake_quantize(self, test_data):
        """测试基础FakeQuantize"""
        observer = MinMaxObserver()
        fq = FakeQuantize(observer=observer)
        
        # 训练模式
        fq.train()
        output_train = fq(test_data)
        assert output_train is not None
        
        # 评估模式
        fq.eval()
        output_eval = fq(test_data)
        assert output_eval is not None
    
    def test_lsq_fake_quantize(self, test_data):
        """测试LSQFakeQuantize"""
        observer = MinMaxObserver()
        fq = LSQFakeQuantize(observer=observer)
        
        fq.train()
        output = fq(test_data)
        assert output is not None
    
    def test_pact_fake_quantize(self, test_data):
        """测试PACTFakeQuantize"""
        observer = MinMaxObserver()
        fq = PACTFakeQuantize(observer=observer, init_alpha=5.0)
        
        fq.train()
        output = fq(test_data)
        assert output is not None
    
    def test_per_channel_fake_quantize(self, test_data):
        """测试per-channel FakeQuantize"""
        observer = MinMaxObserver(
            qscheme=QScheme.PER_CHANNEL_AFFINE,
            ch_axis=1
        )
        fq = FakeQuantize(observer=observer)
        
        fq.train()
        output = fq(test_data)
        assert output is not None
        assert output.shape == test_data.shape
    
    def test_disable_fake_quantize(self, test_data):
        """测试禁用FakeQuantize"""
        observer = MinMaxObserver()
        fq = FakeQuantize(observer=observer)
        
        fq.disable()
        output_disabled = fq(test_data)
        
        # 禁用时应该直接返回输入
        assert torch.allclose(output_disabled, test_data)


def test_quantization_pipeline(test_data):
    """测试完整的量化流水线"""
    # 1. 创建Observer
    observer = MinMaxObserver()
    
    # 2. 统计数据
    observer(test_data)
    
    # 3. 计算量化参数
    scale, zero_point = observer.calculate_qparams()
    
    # 4. 使用fake_quantize
    quantized = fake_quantize_ste(
        test_data,
        scale,
        zero_point,
        observer.quant_min,
        observer.quant_max
    )
    
    assert quantized is not None
    assert quantized.shape == test_data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
