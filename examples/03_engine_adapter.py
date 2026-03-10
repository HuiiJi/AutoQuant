"""
示例 03: 推理引擎适配
展示如何为不同推理引擎（TensorRT/ONNX Runtime/OpenVINO/MNN）选择最佳配置
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from autoquant import (
    ModelQuantizer,
    get_qconfig_for_engine,
    print_engine_info,
    get_supported_engines,
    ONNXExporter,
    ONNXOptimizer,
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


def main():
    print("=" * 60)
    print("示例 03: 推理引擎适配")
    print("=" * 60)
    
    # 1. 查看支持的引擎
    print("\n[1] 支持的推理引擎")
    engines = get_supported_engines()
    print(f"    {engines}")
    
    # 2. 查看TensorRT配置详情
    print("\n[2] TensorRT 引擎配置详情")
    print_engine_info("tensorrt")
    
    # 3. 为TensorRT准备量化
    print("\n[3] 为 TensorRT 优化量化")
    model = SimpleCNN()
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64)
    
    try:
        # 获取TensorRT优化的QConfig
        qconfig_trt = get_qconfig_for_engine("tensorrt")
        print(f"    使用 TensorRT 优化配置")
        
        quantizer = ModelQuantizer(model, qconfig_trt)
        quantizer.prepare()
        quantizer.calibrate([dummy_input])
        quantized_model = quantizer.convert()
        print("    ✓ TensorRT 量化完成")
        
        # 验证
        with torch.no_grad():
            output = quantized_model(dummy_input)
        print(f"    输出形状: {output.shape}")
        
    except Exception as e:
        print(f"    ✗ 量化失败: {e}")
    
    # 4. 为ONNX Runtime准备量化
    print("\n[4] 为 ONNX Runtime 优化量化")
    try:
        qconfig_ort = get_qconfig_for_engine("onnxruntime")
        print(f"    使用 ONNX Runtime 优化配置")
        
        quantizer_ort = ModelQuantizer(model, qconfig_ort)
        quantizer_ort.prepare()
        quantizer_ort.calibrate([dummy_input])
        quantized_model_ort = quantizer_ort.convert()
        print("    ✓ ONNX Runtime 量化完成")
        
    except Exception as e:
        print(f"    ✗ 量化失败: {e}")
    
    # 5. 各引擎对比
    print("\n[5] 各引擎配置对比")
    print("    - TensorRT:    推荐LSQ QAT, per-channel对称权重")
    print("    - ONNX Runtime: 支持per-channel非对称")
    print("    - OpenVINO:    Intel硬件优化")
    print("    - MNN:         移动端优化")
    
    print("\n" + "=" * 60)
    print("示例 03 完成！")
    print("=" * 60)
    print("\n提示: 使用 get_qconfig_for_engine('engine_name') 一键获取最佳配置")


if __name__ == "__main__":
    main()
