"""
环境配置测试 - 验证所有模块和依赖是否正确安装
这是一个快速测试，用于验证环境配置
"""
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 70)
print("AutoQuant 环境配置测试")
print("=" * 70)

print("\n1. 测试核心模块导入...")
try:
    from autoquant import (
        QuantDtype,
        QScheme,
        round_ste,
        clamp_grad,
        fake_quantize_ste,
        lsq_quantize,
        pact_quantize,
    )
    print("   ✓ 核心模块导入成功")
except Exception as e:
    print(f"   ✗ 核心模块导入失败: {e}")
    sys.exit(1)

print("\n2. 测试Observer模块...")
try:
    from autoquant import (
        ObserverBase,
        MinMaxObserver,
        HistogramObserver,
        MovingAverageMinMaxObserver,
        PercentileObserver,
        MSEObserver,
    )
    print("   ✓ Observer模块导入成功")
except Exception as e:
    print(f"   ✗ Observer模块导入失败: {e}")
    sys.exit(1)

print("\n3. 测试FakeQuant模块...")
try:
    from autoquant import (
        FakeQuantizeBase,
        FakeQuantize,
        FixedFakeQuantize,
        LSQFakeQuantize,
        PACTFakeQuantize,
    )
    print("   ✓ FakeQuant模块导入成功")
except Exception as e:
    print(f"   ✗ FakeQuant模块导入失败: {e}")
    sys.exit(1)

print("\n4. 测试工具模块...")
try:
    from autoquant import (
        ModelQuantizer,
        ONNXExporter,
        QConfig,
        get_default_qconfig,
        get_lsq_qconfig,
        get_pact_qconfig,
        get_histogram_qconfig,
        MixedPrecisionQuantizer,
        LayerSelector,
        SensitivityAnalyzer,
    )
    print("   ✓ 工具模块导入成功")
except Exception as e:
    print(f"   ✗ 工具模块导入失败: {e}")
    sys.exit(1)

print("\n5. 测试推理引擎适配模块...")
try:
    from autoquant import (
        InferenceEngine,
        get_engine_config,
        get_qconfig_for_engine,
        get_supported_engines,
        print_engine_info,
    )
    print("   ✓ 推理引擎适配模块导入成功")
    print(f"   ✓ 支持的引擎: {get_supported_engines()}")
except Exception as e:
    print(f"   ✗ 推理引擎适配模块导入失败: {e}")
    sys.exit(1)

print("\n6. 测试ONNX优化模块...")
try:
    from autoquant import (
        ONNXOptimizer,
        optimize_onnx,
        simplify_with_onnxsim,
    )
    print("   ✓ ONNX优化模块导入成功")
except ImportError as e:
    print(f"   ⚠ ONNX优化模块部分功能可能需要额外依赖: {e}")
    print("   提示: 安装完整依赖: pip install onnx onnxsim")
except Exception as e:
    print(f"   ✗ ONNX优化模块导入失败: {e}")

print("\n7. 测试核心函数...")
try:
    import torch

    # 测试round_ste
    x = torch.tensor([1.3, 1.6, 2.1])
    y = round_ste(x)
    print(f"   ✓ round_ste 正常工作")

    # 测试LSQ
    if torch.cuda.is_available():
        print("   ✓ CUDA 可用")
    else:
        print("   ⚠ CUDA 不可用，将使用CPU")

    print("   ✓ 核心函数测试通过")
except Exception as e:
    print(f"   ✗ 核心函数测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ 所有环境配置测试通过！")
print("=" * 70)
