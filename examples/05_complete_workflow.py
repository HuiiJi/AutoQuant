"""
示例 05: 完整工作流
展示从敏感度分析 -> 引擎适配 -> ONNX导出 -> ONNX优化的完整流程
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from autoquant import (
    ModelQuantizer,
    SensitivityAnalyzer,
    get_qconfig_for_engine,
    ONNXExporter,
    ONNXOptimizer,
)


class RestorationModel(nn.Module):
    """模拟一个简单的图像修复模型（类似NAFNet）"""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        feat = self.relu(self.conv_in(x))
        feat = self.relu(self.conv1(feat))
        feat = self.relu(self.conv2(feat))
        out = self.conv_out(feat) + x
        return out


def main():
    print("=" * 70)
    print("示例 05: 完整工作流")
    print("=" * 70)
    
    # 1. 创建模型
    print("\n[1] 创建模型 (类似NAFNet的修复模型)")
    model = RestorationModel()
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128)
    print(f"    输入形状: {dummy_input.shape}")
    
    # 2. 敏感度分析
    print("\n[2] 敏感度分析")
    try:
        from autoquant import get_default_qconfig
        analyzer = SensitivityAnalyzer(model, get_default_qconfig())
        analyzer.analyze(dummy_input)
        print(analyzer.generate_report())
    except Exception as e:
        print(f"    敏感度分析跳过: {e}")
    
    # 3. 为TensorRT引擎适配
    print("\n[3] 为 TensorRT 引擎优化量化")
    try:
        qconfig = get_qconfig_for_engine("tensorrt")
        quantizer = ModelQuantizer(model, qconfig)
        quantizer.prepare()
        quantizer.calibrate([dummy_input])
        quantized_model = quantizer.convert()
        print("    ✓ 量化完成")
    except Exception as e:
        print(f"    量化跳过: {e}")
    
    # 4. 导出ONNX
    print("\n[4] 导出 ONNX 模型")
    try:
        exporter = ONNXExporter(model)
        onnx_path = "restoration_model.onnx"
        exporter.export(
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
        )
        print(f"    ✓ ONNX 已导出: {onnx_path}")
    except Exception as e:
        print(f"    ONNX 导出跳过: {e}")
    
    # 5. 优化ONNX
    print("\n[5] 优化 ONNX 模型")
    try:
        optimizer = ONNXOptimizer(model_path="restoration_model.onnx")
        optimized_model = optimizer.optimize(verbose=True)
        optimized_path = "restoration_model_optimized.onnx"
        optimizer.save(optimized_path)
        print(f"    ✓ 优化后 ONNX 已保存: {optimized_path}")
    except Exception as e:
        print(f"    ONNX 优化跳过: {e}")
    
    print("\n" + "=" * 70)
    print("示例 05 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
