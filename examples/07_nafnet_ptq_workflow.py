"""
示例 07: NAFNet PTQ 完整工作流
专门针对图像修复/去噪等low-level视觉任务的PTQ量化流程

功能：
- 支持NAFNet等修复模型
- 混合精度量化（可选跳过某些层）
- PSNR/SSIM精度评估
- TensorRT/ORT等引擎适配
- ONNX导出
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Optional, List, Tuple, Callable, Any
from pathlib import Path

sys.path.insert(0, '..')

from autoquant import (
    ModelQuantizer,
    SensitivityAnalyzer,
    MixedPrecisionQuantizer,
    LayerSelector,
    QuantizationEvaluator,
    get_histogram_qconfig,
    get_qconfig_for_engine,
    ONNXExporter,
    ONNXOptimizer,
    compute_psnr,
    compute_ssim,
)


# ==========================================
# 1. 简单的NAFNet模型实现（用于演示）
# ==========================================

class SimpleNAFBlock(nn.Module):
    """简化的NAFBlock"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2)
        self.conv3 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        identity = x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.silu(x)
        x = self.conv2(x)
        x = self.silu(x)
        x = self.conv3(x)
        return x + identity


class SimpleNAFNet(nn.Module):
    """简化的NAFNet用于演示"""
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 64,
        num_blocks: int = 6
    ):
        super().__init__()
        
        self.in_conv = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        
        self.blocks = nn.Sequential(*[
            SimpleNAFBlock(dim) for _ in range(num_blocks)
        ])
        
        self.out_conv = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        identity = x
        x = self.in_conv(x)
        x = self.blocks(x)
        x = self.out_conv(x)
        return x + identity


# ==========================================
# 2. 数据工具
# ==========================================

class DummyFaceDataset(torch.utils.data.Dataset):
    """
    模拟的人脸瑕疵数据集（用于演示）
    实际使用时替换为你的真实数据集
    """
    def __init__(
        self,
        num_samples: int = 100,
        img_size: Tuple[int, int] = (256, 256),
        add_noise: bool = True
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.add_noise = add_noise
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成模拟的干净图像
        clean_img = torch.randn(3, *self.img_size) * 0.1 + 0.5
        clean_img = torch.clamp(clean_img, 0.0, 1.0)
        
        # 添加模拟的瑕疵/噪声
        if self.add_noise:
            noisy_img = clean_img + torch.randn_like(clean_img) * 0.05
            noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
            return noisy_img, clean_img
        else:
            return clean_img


def create_calibration_data(
    num_samples: int = 32,
    img_size: Tuple[int, int] = (256, 256),
    batch_size: int = 8
) -> torch.utils.data.DataLoader:
    """
    创建校准数据
    
    Args:
        num_samples: 校准样本数
        img_size: 图像尺寸
        batch_size: batch大小
        
    Returns:
        DataLoader
    """
    dataset = DummyFaceDataset(num_samples=num_samples, img_size=img_size, add_noise=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    return loader


def create_test_data(
    num_samples: int = 10,
    img_size: Tuple[int, int] = (256, 256)
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    创建测试数据
    
    Returns:
        [(noisy_img, clean_img), ...]
    """
    dataset = DummyFaceDataset(num_samples=num_samples, img_size=img_size, add_noise=True)
    return [dataset[i] for i in range(len(dataset))]


# ==========================================
# 3. NAFNet PTQ 工作流
# ==========================================

class NAFNetPTQWorkflow:
    """
    NAFNet PTQ完整工作流
    
    流程：
    1. 敏感度分析（可选）
    2. 准备模型
    3. 校准
    4. 转换
    5. 评估
    6. 导出ONNX
    """
    
    def __init__(
        self,
        model: nn.Module,
        img_size: Tuple[int, int] = (256, 256),
        engine: str = 'tensorrt'
    ):
        self.model = model
        self.img_size = img_size
        self.engine = engine
        
        self.quantizer: Optional[ModelQuantizer] = None
        self.prepared_model: Optional[nn.Module] = None
        self.quantized_model: Optional[nn.Module] = None
        self.skip_layers: List[str] = []
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def analyze_sensitivity(
        self,
        num_samples: int = 5,
        threshold: float = 0.001
    ) -> SensitivityAnalyzer:
        """
        分析敏感度，决定哪些层应该量化
        
        Args:
            num_samples: 分析用的样本数
            threshold: 敏感度阈值
            
        Returns:
            SensitivityAnalyzer
        """
        print("\n" + "=" * 70)
        print("步骤 1: 敏感度分析")
        print("=" * 70)
        
        # 创建dummy输入
        dummy_input = torch.randn(1, 3, *self.img_size).to(self.device)
        
        # 获取qconfig
        qconfig = get_histogram_qconfig(is_symmetric=False)
        
        # 运行敏感度分析
        analyzer = SensitivityAnalyzer(self.model, qconfig)
        analyzer.analyze(dummy_input)
        
        # 打印报告
        print(analyzer.generate_report())
        
        # 获取推荐
        quantizable, not_quantizable = analyzer.get_recommended_layers(threshold=threshold)
        
        print(f"\n建议:")
        print(f"  - 安全量化的层: {len(quantizable)}")
        print(f"  - 建议保持浮点的层: {len(not_quantizable)}")
        
        # 自动设置skip_layers
        self.skip_layers = list(not_quantizable)
        
        # 额外：输出层通常保持浮点
        output_layers = LayerSelector.get_output_layers(self.model)
        self.skip_layers.extend(list(output_layers))
        self.skip_layers = list(set(self.skip_layers))
        
        print(f"  - 将跳过的层: {len(self.skip_layers)}")
        
        return analyzer
    
    def set_skip_layers(self, layer_names: List[str]):
        """手动设置跳过的层"""
        self.skip_layers = layer_names
        print(f"已设置跳过层: {len(self.skip_layers)} 个")
    
    def prepare(
        self,
        use_engine_config: bool = True,
        observer_type: str = 'histogram'
    ):
        """
        准备量化模型
        
        Args:
            use_engine_config: 是否使用推理引擎专用配置
            observer_type: Observer类型 ('minmax', 'histogram', 'percentile', 'mse')
        """
        print("\n" + "=" * 70)
        print("步骤 2: 准备量化模型")
        print("=" * 70)
        
        # 获取qconfig
        if use_engine_config:
            print(f"使用 {self.engine} 专用配置")
            qconfig = get_qconfig_for_engine(self.engine)
        else:
            print(f"使用通用配置，Observer: {observer_type}")
            if observer_type == 'histogram':
                qconfig = get_histogram_qconfig(is_symmetric=False)
            elif observer_type == 'percentile':
                from autoquant.utils import get_default_qconfig
                qconfig = get_default_qconfig(observer_type='percentile')
            elif observer_type == 'mse':
                from autoquant.utils import get_default_qconfig
                qconfig = get_default_qconfig(observer_type='mse')
            else:
                from autoquant.utils import get_default_qconfig
                qconfig = get_default_qconfig()
        
        # 创建quantizer
        self.quantizer = ModelQuantizer(self.model, qconfig)
        
        # 准备模型
        print(f"跳过 {len(self.skip_layers)} 个层的量化")
        self.prepared_model = self.quantizer.prepare(
            inplace=False,
            skip_layers=set(self.skip_layers) if self.skip_layers else None
        )
        
        self.prepared_model.to(self.device)
        print("✅ 模型准备完成!")
    
    def calibrate(self, calib_data):
        """
        校准模型
        
        Args:
            calib_data: 校准数据 (DataLoader, List[Tensor], 或 Tensor)
        """
        print("\n" + "=" * 70)
        print("步骤 3: 校准模型")
        print("=" * 70)
        
        if self.quantizer is None or self.prepared_model is None:
            raise ValueError("请先调用 prepare()")
        
        self.quantizer.calibrate(calib_data, self.device)
    
    def convert(self) -> nn.Module:
        """
        转换为量化模型
        
        Returns:
            量化后的模型
        """
        print("\n" + "=" * 70)
        print("步骤 4: 转换为量化模型")
        print("=" * 70)
        
        if self.quantizer is None:
            raise ValueError("请先调用 prepare()")
        
        self.quantized_model = self.quantizer.convert(self.prepared_model, inplace=False)
        self.quantized_model.to(self.device)
        self.quantized_model.eval()
        
        print("✅ 模型转换完成!")
        return self.quantized_model
    
    def evaluate(
        self,
        test_data: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        评估量化前后的精度
        
        Args:
            test_data: 测试数据 [(noisy, clean), ...]
            
        Returns:
            评估结果字典
        """
        print("\n" + "=" * 70)
        print("步骤 5: 精度评估")
        print("=" * 70)
        
        if self.quantized_model is None:
            raise ValueError("请先调用 convert()")
        
        # 创建评估器
        evaluator = QuantizationEvaluator(self.model, self.quantized_model)
        
        # 评估模型大小和速度
        dummy_input = torch.randn(1, 3, *self.img_size).to(self.device)
        evaluator.evaluate_model_size()
        evaluator.evaluate_speed(dummy_input, iterations=50)
        
        # 评估PSNR/SSIM
        print("\n评估PSNR/SSIM...")
        orig_psnrs = []
        orig_ssims = []
        quant_psnrs = []
        quant_ssims = []
        
        with torch.no_grad():
            for noisy_img, clean_img in test_data:
                noisy_img = noisy_img.unsqueeze(0).to(self.device)
                clean_img = clean_img.unsqueeze(0).to(self.device)
                
                # 原始模型
                orig_restored = self.model(noisy_img)
                orig_psnr = compute_psnr(orig_restored, clean_img).item()
                orig_ssim = compute_ssim(orig_restored, clean_img).item()
                
                # 量化模型
                quant_restored = self.quantized_model(noisy_img)
                quant_psnr = compute_psnr(quant_restored, clean_img).item()
                quant_ssim = compute_ssim(quant_restored, clean_img).item()
                
                orig_psnrs.append(orig_psnr)
                orig_ssims.append(orig_ssim)
                quant_psnrs.append(quant_psnr)
                quant_ssims.append(quant_ssim)
        
        # 计算平均
        import numpy as np
        orig_psnr_mean = np.mean(orig_psnrs)
        orig_ssim_mean = np.mean(orig_ssims)
        quant_psnr_mean = np.mean(quant_psnrs)
        quant_ssim_mean = np.mean(quant_ssims)
        
        psnr_drop = orig_psnr_mean - quant_psnr_mean
        ssim_drop = orig_ssim_mean - quant_ssim_mean
        
        print(f"\n原始模型:")
        print(f"  PSNR: {orig_psnr_mean:.2f} dB")
        print(f"  SSIM: {orig_ssim_mean:.4f}")
        
        print(f"\n量化模型:")
        print(f"  PSNR: {quant_psnr_mean:.2f} dB")
        print(f"  SSIM: {quant_ssim_mean:.4f}")
        
        print(f"\n精度下降:")
        print(f"  PSNR: {psnr_drop:.4f} dB")
        print(f"  SSIM: {ssim_drop:.6f}")
        
        # 检查是否满足要求（PSNR下降 < 1 dB, SSIM下降 < 0.01）
        print(f"\n精度检查:")
        psnr_ok = psnr_drop < 1.0
        ssim_ok = ssim_drop < 0.01
        print(f"  PSNR下降 < 1 dB: {'✅' if psnr_ok else '❌'}")
        print(f"  SSIM下降 < 0.01: {'✅' if ssim_ok else '❌'}")
        
        return {
            'orig_psnr': orig_psnr_mean,
            'orig_ssim': orig_ssim_mean,
            'quant_psnr': quant_psnr_mean,
            'quant_ssim': quant_ssim_mean,
            'psnr_drop': psnr_drop,
            'ssim_drop': ssim_drop,
        }
    
    def export_onnx(
        self,
        output_path: str = 'nafnet_quantized.onnx',
        optimize: bool = True
    ):
        """
        导出ONNX模型
        
        Args:
            output_path: 输出路径
            optimize: 是否优化ONNX
        """
        print("\n" + "=" * 70)
        print("步骤 6: 导出ONNX")
        print("=" * 70)
        
        if self.quantized_model is None:
            raise ValueError("请先调用 convert()")
        
        # 导出
        dummy_input = torch.randn(1, 3, *self.img_size).to(self.device)
        exporter = ONNXExporter(self.quantized_model)
        
        exporter.export(
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=13,
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print(f"✅ ONNX已导出: {output_path}")
        
        # 优化
        if optimize:
            print("\n优化ONNX...")
            optimizer = ONNXOptimizer(model_path=output_path)
            optimized_model = optimizer.optimize(verbose=True)
            
            opt_path = output_path.replace('.onnx', '_optimized.onnx')
            optimizer.save(opt_path)
            print(f"✅ 优化后ONNX已保存: {opt_path}")


# ==========================================
# 4. 主函数
# ==========================================

def main():
    print("=" * 70)
    print("AutoQuant - NAFNet PTQ 完整工作流")
    print("=" * 70)
    
    # 1. 创建模型
    print("\n创建模型...")
    model = SimpleNAFNet(dim=64, num_blocks=6)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    
    # 2. 创建工作流
    workflow = NAFNetPTQWorkflow(
        model=model,
        img_size=(256, 256),
        engine='tensorrt'
    )
    
    # 3. 敏感度分析（可选，建议先运行看结果）
    try:
        workflow.analyze_sensitivity(num_samples=3, threshold=0.001)
    except Exception as e:
        print(f"敏感度分析跳过: {e}")
        # 如果没有敏感度分析，手动设置跳过输出层
        output_layers = LayerSelector.get_output_layers(model)
        workflow.set_skip_layers(list(output_layers))
    
    # 4. 准备和校准
    workflow.prepare(use_engine_config=True, observer_type='histogram')
    
    calib_loader = create_calibration_data(num_samples=32, batch_size=4)
    workflow.calibrate(calib_loader)
    
    # 5. 转换
    quantized_model = workflow.convert()
    
    # 6. 评估
    test_data = create_test_data(num_samples=10)
    results = workflow.evaluate(test_data)
    
    # 7. 导出ONNX
    try:
        workflow.export_onnx('nafnet_quantized.onnx', optimize=True)
    except Exception as e:
        print(f"ONNX导出跳过: {e}")
    
    print("\n" + "=" * 70)
    print("工作流完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
