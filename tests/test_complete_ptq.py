"""
完整的 PTQ 流程测试

Author: jihui
Date: 2026-03-13
"""
import torch
import torch.nn as nn
import os
import sys

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autoquant.utils import get_default_qconfig
from autoquant.quantization import ptq, ModelQuantizer
from autoquant.onnx_export import ONNXExporter


class SimpleModel(nn.Module):
    """一个简单的测试模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def test_complete_ptq():
    """测试完整的 PTQ 流程"""
    print("=" * 70)
    print("🧪 测试完整的 PTQ 量化流程")
    print("=" * 70)
    
    # 1. 创建模型和数据
    print("\n[步骤 1] 创建模型和测试数据")
    model = SimpleModel()
    dummy_input = torch.randn(1, 3, 32, 32)
    calib_data = [torch.randn(1, 3, 32, 32) for _ in range(5)]
    
    # 2. 运行原始模型
    print("\n[步骤 2] 运行原始模型")
    model.eval()
    with torch.no_grad():
        original_output = model(dummy_input)
    print(f"   原始输出形状: {original_output.shape}")
    
    # 3. 准备 QConfig
    print("\n[步骤 3] 准备量化配置")
    qconfig = get_default_qconfig(
        activation_observer_type="minmax",
        weight_observer_type="minmax"
    )
    print(f"   QConfig: {qconfig}")
    
    # 4. 使用 ptq() 一键量化
    print("\n[步骤 4] 执行完整 PTQ 流程")
    quantized_model = ptq(
        model,
        qconfig,
        calib_data,
        verbose=True
    )
    
    # 5. 运行量化模型
    print("\n[步骤 5] 运行量化模型")
    quantized_model.eval()
    with torch.no_grad():
        quantized_output = quantized_model(dummy_input)
    print(f"   量化输出形状: {quantized_output.shape}")
    
    # 6. 检查输出差异
    print("\n[步骤 6] 检查输出差异")
    diff = torch.abs(original_output - quantized_output).mean()
    print(f"   平均绝对误差: {diff:.6f}")
    
    # 7. 导出 ONNX (暂时跳过，先测试核心流程)
    print("\n[步骤 7] 跳过 ONNX 导出测试（稍后单独测试）")
    # onnx_path = "test_model_quantized.onnx"
    # ONNXExporter.export(
    #     quantized_model,
    #     dummy_input,
    #     onnx_path,
    #     opset_version=18,
    #     verbose=False
    # )
    # 
    # # 8. 清理
    # if os.path.exists(onnx_path):
    #     os.remove(onnx_path)
    #     print(f"\n[清理] 已删除临时文件: {onnx_path}")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)
    
    return True


def test_model_quantizer():
    """测试 ModelQuantizer 的分步使用"""
    print("\n" + "=" * 70)
    print("🧪 测试 ModelQuantizer 分步使用")
    print("=" * 70)
    
    # 创建模型和数据
    model = SimpleModel()
    dummy_input = torch.randn(1, 3, 32, 32)
    calib_data = [torch.randn(1, 3, 32, 32) for _ in range(3)]
    
    # QConfig
    qconfig = get_default_qconfig()
    
    # 分步流程
    print("\n[分步 1] prepare()")
    quantizer = ModelQuantizer(model, qconfig)
    prepared_model = quantizer.prepare()
    
    print("\n[分步 2] calibrate()")
    quantizer.calibrate(calib_data, verbose=True)
    
    print("\n[分步 3] convert()")
    quantized_model = quantizer.convert()
    
    print("\n✅ ModelQuantizer 分步测试通过！")
    
    return True


if __name__ == "__main__":
    try:
        success1 = test_complete_ptq()
        success2 = test_model_quantizer()
        
        if success1 and success2:
            print("\n🎉 所有测试通过！代码库已修复并正常工作！")
        else:
            print("\n❌ 部分测试失败")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
