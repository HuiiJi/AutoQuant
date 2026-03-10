"""
示例 02: 高级 QAT 量化
展示 LSQ、PACT 等高级 QAT 方法
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from autoquant import (
    ModelQuantizer,
    get_lsq_qconfig,
    get_pact_qconfig,
    get_histogram_qconfig,
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
    print("示例 02: 高级 QAT 量化")
    print("=" * 60)
    
    # 1. 创建模型和数据
    print("\n[1] 创建模型")
    model = SimpleCNN()
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64)
    print(f"    输入形状: {dummy_input.shape}")
    
    # 2. 测试 LSQ 配置
    print("\n[2] 测试 LSQ (Learned Step Size Quantization)")
    print("    LSQ: 可学习的量化步长，适合QAT")
    try:
        lsq_qconfig = get_lsq_qconfig()
        quantizer_lsq = ModelQuantizer(model, lsq_qconfig)
        quantizer_lsq.prepare()
        quantizer_lsq.calibrate([dummy_input])
        quantized_model_lsq = quantizer_lsq.convert()
        print("    ✓ LSQ 量化完成")
    except Exception as e:
        print(f"    ✗ LSQ 量化失败: {e}")
    
    # 3. 测试 PACT 配置
    print("\n[3] 测试 PACT (Parameterized Clipping Activation)")
    print("    PACT: 参数化裁剪激活，适合激活值量化")
    try:
        pact_qconfig = get_pact_qconfig(init_alpha=5.0)
        quantizer_pact = ModelQuantizer(model, pact_qconfig)
        quantizer_pact.prepare()
        quantizer_pact.calibrate([dummy_input])
        quantized_model_pact = quantizer_pact.convert()
        print("    ✓ PACT 量化完成")
    except Exception as e:
        print(f"    ✗ PACT 量化失败: {e}")
    
    # 4. 测试 Histogram 配置
    print("\n[4] 测试 HistogramObserver")
    print("    HistogramObserver: 基于直方图的精确量化")
    try:
        hist_qconfig = get_histogram_qconfig()
        quantizer_hist = ModelQuantizer(model, hist_qconfig)
        quantizer_hist.prepare()
        quantizer_hist.calibrate([dummy_input])
        quantized_model_hist = quantizer_hist.convert()
        print("    ✓ Histogram 量化完成")
    except Exception as e:
        print(f"    ✗ Histogram 量化失败: {e}")
    
    # 5. 对比验证
    print("\n[5] 对比不同方法输出验证")
    with torch.no_grad():
        original_output = model(dummy_input)
        
        try:
            output_lsq = quantized_model_lsq(dummy_input)
            mse_lsq = torch.nn.functional.mse_loss(original_output, output_lsq).item()
            print(f"    LSQ MSE: {mse_lsq:.6f}")
        except:
            pass
        
        try:
            output_pact = quantized_model_pact(dummy_input)
            mse_pact = torch.nn.functional.mse_loss(original_output, output_pact).item()
            print(f"    PACT MSE: {mse_pact:.6f}")
        except:
            pass
        
        try:
            output_hist = quantized_model_hist(dummy_input)
            mse_hist = torch.nn.functional.mse_loss(original_output, output_hist).item()
            print(f"    Histogram MSE: {mse_hist:.6f}")
        except:
            pass
    
    print("\n" + "=" * 60)
    print("示例 02 完成！")
    print("=" * 60)
    print("\n提示: 在实际QAT训练中，你需要:")
    print("  1. prepare() 后将模型切换到 train() 模式")
    print("  2. 进行正常的训练循环")
    print("  3. 训练完成后 convert() 得到量化模型")


if __name__ == "__main__":
    main()
