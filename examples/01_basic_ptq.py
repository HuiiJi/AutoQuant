"""
示例 01: 基础 PTQ 量化
展示如何使用 AutoQuant 进行基础的后训练量化
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from autoquant import ModelQuantizer, get_default_qconfig


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
    print("示例 01: 基础 PTQ 量化")
    print("=" * 60)
    
    # 1. 创建模型和数据
    print("\n[1] 创建模型")
    model = SimpleCNN()
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64)
    print(f"    输入形状: {dummy_input.shape}")
    
    # 2. 获取量化配置
    print("\n[2] 获取量化配置")
    qconfig = get_default_qconfig()
    print(f"    使用配置: {qconfig}")
    
    # 3. 准备量化模型
    print("\n[3] 准备量化模型")
    quantizer = ModelQuantizer(model, qconfig)
    quantizer.prepare()
    print("    ✓ 模型准备完成")
    
    # 4. 校准（收集统计信息）
    print("\n[4] 校准模型")
    calib_data = [torch.randn(1, 3, 64, 64) for _ in range(10)]
    quantizer.calibrate(calib_data)
    print("    ✓ 校准完成")
    
    # 5. 转换为量化模型
    print("\n[5] 转换为量化模型")
    quantized_model = quantizer.convert()
    print("    ✓ 转换完成")
    
    # 6. 验证
    print("\n[6] 验证量化模型")
    with torch.no_grad():
        original_output = model(dummy_input)
        quantized_output = quantized_model(dummy_input)
    
    print(f"    原始输出形状: {original_output.shape}")
    print(f"    量化输出形状: {quantized_output.shape}")
    
    # 计算MSE
    mse = torch.nn.functional.mse_loss(original_output, quantized_output).item()
    print(f"    输出 MSE: {mse:.6f}")
    
    print("\n" + "=" * 60)
    print("示例 01 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
