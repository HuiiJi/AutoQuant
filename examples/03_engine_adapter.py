"""
示例 03: 推理引擎适配
展示如何为 TensorRT 和 ONNX Runtime 选择最佳配置

Author: jihui
Date: 2026-03-13
"""
import torch
import torch.nn as nn
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from autoquant import (
    ModelQuantizer,
    get_trt_qconfig,
    get_ort_qconfig,
    get_default_qconfig,
    ONNXExporter,
)


class SimpleCNN(nn.Module):
    """简单的CNN模型用于演示"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(32 * 8 * 8, 10)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def export_to_onnx(model, dummy_input, filename):
    """导出模型为 ONNX 格式"""
    print(f"\n    导出 ONNX: {filename}")
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"    ✓ ONNX 导出成功")


def main():
    print("=" * 80)
    print("示例 03: 推理引擎适配 (TRT / ORT)")
    print("=" * 80)
    
    # 1. 创建模型
    print("\n" + "=" * 80)
    print("[1] 创建模型")
    print("=" * 80)
    
    dummy_input = torch.randn(1, 3, 64, 64)
    print(f"    输入形状: {dummy_input.shape}")
    
    # 2. TensorRT 最佳配置
    print("\n" + "=" * 80)
    print("[2] TensorRT 最佳配置")
    print("=" * 80)
    print("    TRT 最佳实践:")
    print("      - Activation: PER_TENSOR_SYMMETRIC + MinMaxObserver")
    print("      - Weight: PER_CHANNEL_SYMMETRIC + MinMaxObserver")
    
    try:
        model_trt = SimpleCNN()
        model_trt.eval()
        qconfig_trt = get_trt_qconfig()
        quantizer_trt = ModelQuantizer(model_trt, qconfig_trt)
        quantizer_trt.prepare()
        quantizer_trt.calibrate([dummy_input])
        quantized_model_trt = quantizer_trt.convert()
        print("    ✓ TensorRT 量化完成")
        
        # 验证
        with torch.no_grad():
            output_trt = quantized_model_trt(dummy_input)
            orig_output = SimpleCNN()(dummy_input)
        mse_trt = torch.nn.functional.mse_loss(orig_output, output_trt).item()
        print(f"    输出 MSE: {mse_trt:.6f}")
        
        # 导出 ONNX
        trt_onnx_path = os.path.join(project_root, "quantized_trt.onnx")
        export_to_onnx(quantized_model_trt, dummy_input, trt_onnx_path)
        
    except Exception as e:
        print(f"    ✗ TensorRT 量化失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. ONNX Runtime 最佳配置
    print("\n" + "=" * 80)
    print("[3] ONNX Runtime 最佳配置")
    print("=" * 80)
    print("    ORT 最佳实践:")
    print("      - Activation: PER_TENSOR_AFFINE + HistogramObserver")
    print("      - Weight: PER_CHANNEL_AFFINE + MinMaxObserver")
    
    try:
        model_ort = SimpleCNN()
        model_ort.eval()
        qconfig_ort = get_ort_qconfig()
        quantizer_ort = ModelQuantizer(model_ort, qconfig_ort)
        quantizer_ort.prepare()
        quantizer_ort.calibrate([dummy_input])
        quantized_model_ort = quantizer_ort.convert()
        print("    ✓ ONNX Runtime 量化完成")
        
        # 验证
        with torch.no_grad():
            output_ort = quantized_model_ort(dummy_input)
            orig_output = SimpleCNN()(dummy_input)
        mse_ort = torch.nn.functional.mse_loss(orig_output, output_ort).item()
        print(f"    输出 MSE: {mse_ort:.6f}")
        
        # 导出 ONNX
        ort_onnx_path = os.path.join(project_root, "quantized_ort.onnx")
        export_to_onnx(quantized_model_ort, dummy_input, ort_onnx_path)
        
    except Exception as e:
        print(f"    ✗ ONNX Runtime 量化失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 默认配置（TRT）
    print("\n" + "=" * 80)
    print("[4] 默认配置 (使用 TensorRT 方案)")
    print("=" * 80)
    
    try:
        model_default = SimpleCNN()
        model_default.eval()
        qconfig_default = get_default_qconfig()
        quantizer_default = ModelQuantizer(model_default, qconfig_default)
        quantizer_default.prepare()
        quantizer_default.calibrate([dummy_input])
        quantized_model_default = quantizer_default.convert()
        print("    ✓ 默认配置量化完成")
        
    except Exception as e:
        print(f"    ✗ 默认配置量化失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("  - get_trt_qconfig(): TensorRT 最佳精度方案")
    print("  - get_ort_qconfig(): ONNX Runtime 最佳精度方案")
    print("  - get_default_qconfig(): 默认使用 TRT 方案")
    print("=" * 80)
    print("示例 03 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
