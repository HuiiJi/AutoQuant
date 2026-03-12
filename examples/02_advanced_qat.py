"""
示例 02: QAT 量化感知训练
展示如何使用 LSQ 进行 QAT

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
    get_lsq_qconfig,
    NAFNet_dgf,
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
    print("=" * 80)
    print("示例 02: QAT 量化感知训练 (LSQ)")
    print("=" * 80)
    
    # 1. 创建模型
    print("\n" + "=" * 80)
    print("[1] 创建模型")
    print("=" * 80)
    
    model = SimpleCNN()
    model.train()
    dummy_input = torch.randn(1, 3, 64, 64)
    print(f"    输入形状: {dummy_input.shape}")
    
    # 2. QAT 准备
    print("\n" + "=" * 80)
    print("[2] QAT 准备 (LSQ)")
    print("=" * 80)
    
    qconfig = get_lsq_qconfig()
    quantizer = ModelQuantizer(model, qconfig)
    prepared_model = quantizer.prepare()
    print("    ✓ QAT 模型准备完成")
    
    # 3. 模拟 QAT 训练（这里只做演示，实际需要真实的训练循环）
    print("\n" + "=" * 80)
    print("[3] 模拟 QAT 训练")
    print("=" * 80)
    print("    提示: 在实际使用中，你需要:")
    print("      1. 准备真实的训练数据")
    print("      2. 设置 optimizer 和 scheduler")
    print("      3. 进行正常的训练循环")
    print("      4. 训练完成后调用 convert()")
    
    # 4. 转换为量化模型
    print("\n" + "=" * 80)
    print("[4] 转换为量化模型")
    print("=" * 80)
    
    quantized_model = quantizer.convert()
    print("    ✓ 转换完成")
    
    # 5. 验证
    print("\n" + "=" * 80)
    print("[5] 验证量化模型")
    print("=" * 80)
    
    quantized_model.eval()
    with torch.no_grad():
        original_output = model(dummy_input)
        quantized_output = quantized_model(dummy_input)
    
    mse = torch.nn.functional.mse_loss(original_output, quantized_output).item()
    print(f"    输出 MSE: {mse:.6f}")
    
    print("\n" + "=" * 80)
    print("示例 02 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
