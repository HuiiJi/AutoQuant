"""
示例 04: Op 敏感度分析
展示如何分析每个操作对量化的敏感度，并决定哪些操作应该量化
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from autoquant import SensitivityAnalyzer, get_default_qconfig


class ComplexCNN(nn.Module):
    """更复杂的CNN模型用于敏感度分析"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    print("=" * 60)
    print("示例 04: Op 敏感度分析")
    print("=" * 60)
    
    # 1. 创建模型
    print("\n[1] 创建模型")
    model = ComplexCNN()
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    print(f"    输入形状: {dummy_input.shape}")
    
    # 2. 打印模型结构
    print("\n[2] 模型结构:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            print(f"    {name}: {module.__class__.__name__}")
    
    # 3. 执行敏感度分析
    print("\n[3] 执行敏感度分析...")
    try:
        qconfig = get_default_qconfig()
        analyzer = SensitivityAnalyzer(model, qconfig)
        
        # 分析所有可量化层的敏感度
        sensitivity_scores = analyzer.analyze(
            dummy_input,
            # skip_layers=['fc2'],  # 可以跳过某些层
            # only_layers=['conv1', 'conv2'],  # 也可以只分析特定层
        )
        
        print("    ✓ 敏感度分析完成")
        
        # 4. 生成详细报告
        print("\n[4] 敏感度分析报告:")
        report = analyzer.generate_report(
            sort_by='score',
            ascending=False  # 按敏感度从高到低排序
        )
        print(report)
        
        # 5. 获取推荐的量化策略
        print("\n[5] 推荐的量化策略:")
        threshold = 0.001  # 可以调整这个阈值
        quantizable, not_quantizable = analyzer.get_recommended_layers(threshold=threshold)
        
        print(f"    阈值: {threshold}")
        print(f"    推荐量化的层 ({len(quantizable)}):")
        for layer in quantizable:
            print(f"      - {layer}")
        
        print(f"    推荐保持浮点的层 ({len(not_quantizable)}):")
        for layer in not_quantizable:
            print(f"      - {layer}")
        
        # 6. 可视化（可选）
        print("\n[6] 可视化敏感度分析 (需要matplotlib):")
        try:
            analyzer.plot_sensitivity(
                figsize=(10, 6),
                save_path="sensitivity_analysis.png"
            )
            print("    ✓ 图已保存到 sensitivity_analysis.png")
        except ImportError:
            print("    ⚠ 需要安装matplotlib: pip install matplotlib")
        except Exception as e:
            print(f"    ⚠ 可视化跳过: {e}")
        
    except Exception as e:
        print(f"    ✗ 敏感度分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("示例 04 完成！")
    print("=" * 60)
    print("\n提示:")
    print("  - 敏感度高的层（分数高）建议保持浮点精度")
    print("  - 敏感度低的层（分数低）可以安全量化")
    print("  - 可以根据任务需求调整threshold参数")


if __name__ == "__main__":
    main()
