"""
测试QConfig和相关工具
"""
from autoquant import (
    get_default_qconfig,
    get_lsq_qconfig,
    get_pact_qconfig,
    get_histogram_qconfig,
    get_per_channel_qconfig,
    get_per_tensor_qconfig,
    get_qconfig_for_engine,
    get_supported_engines,
    print_engine_info,
    get_engine_config,
)
import pytest
import torch
import sys
sys.path.insert(0, '.')


class TestQConfig:
    """测试QConfig配置"""

    def test_default_qconfig(self):
        """测试默认配置"""
        qconfig = get_default_qconfig()
        assert qconfig is not None
        assert hasattr(qconfig, 'activation')
        assert hasattr(qconfig, 'weight')

    def test_lsq_qconfig(self):
        """测试LSQ配置"""
        qconfig = get_lsq_qconfig()
        assert qconfig is not None

    def test_pact_qconfig(self):
        """测试PACT配置"""
        qconfig = get_pact_qconfig()
        assert qconfig is not None

        # 测试自定义alpha
        qconfig = get_pact_qconfig(init_alpha=2.0)
        assert qconfig is not None

    def test_histogram_qconfig(self):
        """测试Histogram配置"""
        qconfig = get_histogram_qconfig()
        assert qconfig is not None

    def test_per_channel_qconfig(self):
        """测试per-channel配置"""
        qconfig = get_per_channel_qconfig(is_symmetric=True)
        assert qconfig is not None

        qconfig = get_per_channel_qconfig(is_symmetric=False)
        assert qconfig is not None

    def test_per_tensor_qconfig(self):
        """测试per-tensor配置"""
        qconfig = get_per_tensor_qconfig(is_symmetric=True)
        assert qconfig is not None

        qconfig = get_per_tensor_qconfig(is_symmetric=False)
        assert qconfig is not None

    def test_observer_types(self):
        """测试不同Observer类型"""
        observer_types = ['minmax', 'histogram', 'moving_avg', 'percentile', 'mse']

        for obs_type in observer_types:
            qconfig = get_default_qconfig(observer_type=obs_type)
            assert qconfig is not None


class TestEngineAdapter:
    """测试推理引擎适配"""

    def test_supported_engines(self):
        """测试支持的引擎列表"""
        engines = get_supported_engines()
        assert isinstance(engines, list)
        assert len(engines) > 0
        assert 'tensorrt' in engines
        assert 'onnxruntime' in engines

    def test_get_engine_config(self):
        """测试获取引擎配置"""
        from autoquant import get_engine_config

        config = get_engine_config('tensorrt')
        assert config is not None
        assert config.engine.value == 'tensorrt'

    def test_get_qconfig_for_engine(self):
        """测试为引擎获取QConfig"""
        engines = get_supported_engines()

        for engine in engines:
            qconfig = get_qconfig_for_engine(engine)
            assert qconfig is not None

    def test_print_engine_info(self):
        """测试打印引擎信息"""
        import io
        import contextlib

        # 捕获stdout
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_engine_info('tensorrt')

        output = f.getvalue()
        assert len(output) > 0


def test_all_engine_configs():
    """测试所有引擎配置"""
    engines = get_supported_engines()

    for engine in engines:
        # 测试获取配置
        config = get_engine_config(engine)
        assert config is not None

        # 测试获取qconfig
        qconfig = get_qconfig_for_engine(engine)
        assert qconfig is not None

        # 验证基本属性
        assert hasattr(config, 'activation_dtype')
        assert hasattr(config, 'weight_dtype')
        assert hasattr(config, 'activation_qscheme')
        assert hasattr(config, 'weight_qscheme')


def test_qconfig_with_different_dtypes():
    """测试不同dtype的QConfig"""
    from autoquant import QuantDtype, QScheme

    # 测试不同dtype组合
    activation_dtypes = [QuantDtype.QUINT8, QuantDtype.QINT8]
    weight_dtypes = [QuantDtype.QUINT8, QuantDtype.QINT8]

    for act_dtype in activation_dtypes:
        for w_dtype in weight_dtypes:
            qconfig = get_default_qconfig(
                activation_dtype=act_dtype,
                weight_dtype=w_dtype
            )
            assert qconfig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
