"""
测试 NAFNet 模型的 ONNX 导出和优化
"""
import sys
import os

sys.path.insert(0, '.')

import torch
import torch.nn as nn
from autoquant import (
    ModelQuantizer,
    get_histogram_qconfig,
    ONNXExporter,
    optimize_onnx,
    simplify_with_onnxsim,
)


def test_nafnet_onnx_export():
    """测试 NAFNet 模型的 ONNX 导出"""
    print("=" * 70)
    print("测试 1: NAFNet 模型 ONNX 导出")
    print("=" * 70)
    
    # 创建模型
    model = create_nafnet_simple(dim=64, num_blocks=6)
    print(f"✓ 模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    
    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ 前向传播成功")
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    
    # 导出 ONNX
    output_path = "nafnet_model.onnx"
    print("\n🔧 开始导出 ONNX 模型...")
    
    exporter = ONNXExporter()
    exporter.export(
        model,
        dummy_input,
        output_path,
        opset_version=13,
        use_symbolic_trace=True,
        optimize=True,
        verbose=True
    )
    
    print(f"✓ ONNX 模型导出成功: {output_path}")
    
    return model, device, output_path


def test_nafnet_quantized_onnx_export():
    """测试量化后的 NAFNet 模型的 ONNX 导出"""
    print("\n" + "=" * 70)
    print("测试 2: 量化后的 NAFNet 模型 ONNX 导出")
    print("=" * 70)
    
    # 创建模型
    model = create_nafnet_simple(dim=64, num_blocks=6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 获取 qconfig
    qconfig = get_histogram_qconfig(is_symmetric=False)
    print(f"✓ 获取 qconfig 成功")
    
    # 创建 quantizer
    quantizer = ModelQuantizer(model, qconfig)
    print(f"✓ 创建 quantizer 成功")
    
    # 准备模型
    prepared_model = quantizer.prepare(inplace=False)
    prepared_model.to(device)
    prepared_model.eval()
    print(f"✓ 模型准备成功")
    
    # 创建校准数据
    calib_data = []
    for _ in range(10):
        calib_data.append(torch.randn(1, 3, 256, 256).to(device))
    
    # 校准
    quantizer.calibrate(calib_data, device)
    print(f"✓ 校准完成")
    
    # 转换
    quantized_model = quantizer.convert(prepared_model, inplace=False)
    quantized_model.to(device)
    quantized_model.eval()
    print(f"✓ 转换完成")
    
    # 测试前向传播
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = quantized_model(dummy_input)
    
    print(f"✓ 量化模型前向传播成功")
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    
    # 导出 ONNX
    output_path = "nafnet_model_quantized.onnx"
    print("\n🔧 开始导出量化 ONNX 模型...")
    
    exporter = ONNXExporter()
    exporter.export(
        quantized_model,
        dummy_input,
        output_path,
        opset_version=13,
        use_symbolic_trace=True,
        optimize=True,
        verbose=True
    )
    
    print(f"✓ 量化 ONNX 模型导出成功: {output_path}")
    
    return quantized_model, device, output_path


def test_onnx_optimization(input_path):
    """测试 ONNX 模型优化"""
    print("\n" + "=" * 70)
    print("测试 3: ONNX 模型优化")
    print("=" * 70)
    
    output_path = "nafnet_model_optimized.onnx"
    print(f"🔧 开始优化 ONNX 模型: {input_path}")
    
    # 使用 ONNXOptimizer 优化
    optimized_model = optimize_onnx(input_path, output_path, verbose=True)
    print(f"✓ ONNX 模型优化成功: {output_path}")
    
    # 使用 onnxsim 简化
    output_path_sim = "nafnet_model_simplified.onnx"
    print(f"\n🔧 开始使用 onnxsim 简化模型...")
    simplified_model = simplify_with_onnxsim(input_path, output_path_sim, check_n=3)
    print(f"✓ ONNX 模型简化成功: {output_path_sim}")
    
    return output_path, output_path_sim


def main():
    print("=" * 70)
    print("NAFNet ONNX 导出和优化测试")
    print("=" * 70)
    
    try:
        # 测试 1: 原始模型 ONNX 导出
        model, device, output_path = test_nafnet_onnx_export()
        
        # 测试 2: 量化模型 ONNX 导出
        quantized_model, device, quantized_output_path = test_nafnet_quantized_onnx_export()
        
        # 测试 3: ONNX 模型优化
        optimized_path, simplified_path = test_onnx_optimization(output_path)
        
        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)
        print(f"原始模型 ONNX: {output_path}")
        print(f"量化模型 ONNX: {quantized_output_path}")
        print(f"优化后模型: {optimized_path}")
        print(f"简化后模型: {simplified_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
