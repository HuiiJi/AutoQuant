"""
测试所有Observer
"""
import pytest
import torch
import sys
sys.path.insert(0, '.')
print(sys.path)
from autoquant import (
    QuantDtype,
    QScheme,
    MinMaxObserver,
    HistogramObserver,
    MovingAverageMinMaxObserver,
    PercentileObserver,
    MSEObserver,
)


@pytest.fixture
def test_data():
    """生成测试数据"""
    return torch.randn(10, 3, 32, 32)


@pytest.fixture
def per_channel_data():
    """生成per-channel测试数据"""
    return torch.randn(10, 16, 32, 32)  # (batch, channels, h, w)


class TestMinMaxObserver:
    """测试MinMaxObserver"""
    
    def test_basic(self, test_data):
        """测试基础功能"""
        observer = MinMaxObserver()
        observer(test_data)
        scale, zero_point = observer.calculate_qparams()
        
        assert scale is not None
        assert zero_point is not None
        assert scale > 0
    
    def test_per_channel(self, per_channel_data):
        """测试per-channel"""
        observer = MinMaxObserver(
            qscheme=QScheme.PER_CHANNEL_AFFINE,
            ch_axis=1
        )
        observer(per_channel_data)
        scale, zero_point = observer.calculate_qparams()
        
        assert scale.shape == (16,)  # 16个通道
        assert zero_point.shape == (16,)
    
    def test_symmetric(self, test_data):
        """测试对称量化"""
        observer = MinMaxObserver(
            qscheme=QScheme.PER_TENSOR_SYMMETRIC
        )
        observer(test_data)
        scale, zero_point = observer.calculate_qparams()
        
        assert torch.all(zero_point == 0)
    
    def test_reset(self, test_data):
        """测试reset"""
        observer = MinMaxObserver()
        observer(test_data)
        observer.calculate_qparams()
        assert observer.scale is not None
        
        observer.reset()
        assert observer.scale is None


class TestHistogramObserver:
    """测试HistogramObserver"""
    
    def test_basic(self, test_data):
        """测试基础功能"""
        observer = HistogramObserver()
        observer(test_data)
        scale, zero_point = observer.calculate_qparams()
        
        assert scale is not None
        assert zero_point is not None
    
    def test_per_channel(self, per_channel_data):
        """测试per-channel"""
        observer = HistogramObserver(
            qscheme=QScheme.PER_CHANNEL_AFFINE,
            ch_axis=1
        )
        observer(per_channel_data)
        scale, zero_point = observer.calculate_qparams()
        
        assert scale.shape == (16,)


class TestMovingAverageMinMaxObserver:
    """测试MovingAverageMinMaxObserver"""
    
    def test_basic(self, test_data):
        """测试基础功能"""
        observer = MovingAverageMinMaxObserver(momentum=0.1)
        observer(test_data)
        
        # 第二次前向，测试滑动平均
        observer(test_data * 1.5)
        
        scale, zero_point = observer.calculate_qparams()
        assert scale is not None


class TestPercentileObserver:
    """测试PercentileObserver"""
    
    def test_basic(self, test_data):
        """测试基础功能"""
        observer = PercentileObserver(
            min_percentile=0.01,
            max_percentile=0.99
        )
        observer(test_data)
        scale, zero_point = observer.calculate_qparams()
        
        assert scale is not None


class TestMSEObserver:
    """测试MSEObserver"""
    
    def test_basic(self, test_data):
        """测试基础功能"""
        observer = MSEObserver(num_candidates=10)
        observer(test_data)
        scale, zero_point = observer.calculate_qparams()
        
        assert scale is not None


def test_all_observers_with_dtypes(test_data):
    """测试所有Observer与不同dtype的组合"""
    observers = [
        MinMaxObserver,
        HistogramObserver,
        MovingAverageMinMaxObserver,
    ]
    
    dtypes = [QuantDtype.QUINT8, QuantDtype.QINT8]
    
    for observer_cls in observers:
        for dtype in dtypes:
            observer = observer_cls(dtype=dtype)
            observer(test_data)
            scale, zero_point = observer.calculate_qparams()
            assert scale is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
