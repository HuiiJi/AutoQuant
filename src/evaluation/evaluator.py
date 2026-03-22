"""
量化评估器
用于全面评估量化前后的模型性能、精度、速度和大小
"""
import torch
import torch.nn as nn
import time
from typing import Optional, Callable, List, Dict, Any, Union, Tuple
from tabulate import tabulate
import sys
import os

from .metrics import (
    compute_accuracy,
    compute_psnr,
    compute_ssim,
    compute_l1_error,
    compute_l2_error,
    compute_cosine_similarity,
)


class QuantizationEvaluator:
    """
    量化评估器

    功能：
    - 精度评估：Top-1/Top-5, PSNR, SSIM, L1/L2误差
    - 速度评估：推理时间、吞吐量
    - 大小评估：模型大小对比
    - 生成详细报告
    """

    def __init__(
        self,
        original_model: nn.Module,
        quantized_model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        """
        初始化评估器

        Args:
            original_model: 原始浮点模型
            quantized_model: 量化后的模型（可选，可稍后设置）
            device: 计算设备
        """
        self.original_model = original_model
        self.quantized_model = quantized_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.original_model.eval()
        self.original_model.to(self.device)

        if self.quantized_model is not None:
            self.quantized_model.eval()
            self.quantized_model.to(self.device)

        self.results: Dict[str, Any] = {}

    def set_quantized_model(self, quantized_model: nn.Module):
        """设置量化模型"""
        self.quantized_model = quantized_model
        self.quantized_model.eval()
        self.quantized_model.to(self.device)

    def evaluate_speed(
        self,
        dummy_input: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]],
        iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        评估推理速度

        Args:
            dummy_input: 输入张量或字典
            iterations: 测试迭代次数
            warmup_iterations: 预热迭代次数

        Returns:
            包含速度指标的字典
        """
        print("\n🚀 评估推理速度...")

        def _measure_speed(model, inp):
            # 预热
            with torch.no_grad():
                for _ in range(warmup_iterations):
                    if isinstance(inp, dict):
                        model(**inp)
                    elif isinstance(inp, tuple):
                        model(*inp)
                    else:
                        model(inp)

            # 正式测试
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                for _ in range(iterations):
                    if isinstance(inp, dict):
                        model(**inp)
                    elif isinstance(inp, tuple):
                        model(*inp)
                    else:
                        model(inp)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_time = time.time() - start_time

            avg_time = total_time / iterations
            throughput = 1.0 / avg_time  # samples per second

            return {
                'avg_time_ms': avg_time * 1000,
                'throughput': throughput,
                'total_time_s': total_time
            }

        # 移动输入到设备
        if isinstance(dummy_input, dict):
            dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
        elif isinstance(dummy_input, tuple):
            dummy_input = tuple(v.to(self.device) for v in dummy_input)
        else:
            dummy_input = dummy_input.to(self.device)

        # 测试原始模型
        orig_speed = _measure_speed(self.original_model, dummy_input)
        print(f"  原始模型: {orig_speed['avg_time_ms']:.2f} ms/iter, {orig_speed['throughput']:.2f} iters/s")

        # 测试量化模型
        quant_speed = None
        if self.quantized_model is not None:
            quant_speed = _measure_speed(self.quantized_model, dummy_input)
            print(f"  量化模型: {quant_speed['avg_time_ms']:.2f} ms/iter, {quant_speed['throughput']:.2f} iters/s")

            # 计算加速比
            speedup = orig_speed['avg_time_ms'] / quant_speed['avg_time_ms']
            print(f"  加速比: {speedup:.2f}x")

        result = {
            'original': orig_speed,
            'quantized': quant_speed,
            'speedup': orig_speed['avg_time_ms'] / quant_speed['avg_time_ms'] if quant_speed else None
        }

        self.results['speed'] = result
        return result

    def evaluate_model_size(self) -> Dict[str, float]:
        """
        评估模型大小

        Returns:
            包含模型大小的字典 (MB)
        """
        print("\n📦 评估模型大小...")

        def _get_model_size(model):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()

            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            total_size_mb = (param_size + buffer_size) / 1024 / 1024
            return total_size_mb

        orig_size = _get_model_size(self.original_model)
        print(f"  原始模型: {orig_size:.2f} MB")

        quant_size = None
        if self.quantized_model is not None:
            quant_size = _get_model_size(self.quantized_model)
            print(f"  量化模型: {quant_size:.2f} MB")

            compression_ratio = orig_size / quant_size
            print(f"  压缩比: {compression_ratio:.2f}x")

        result = {
            'original_mb': orig_size,
            'quantized_mb': quant_size,
            'compression_ratio': orig_size / quant_size if quant_size else None
        }

        self.results['size'] = result
        return result

    def evaluate_accuracy_classification(
        self,
        data_loader: torch.utils.data.DataLoader,
        topk: Tuple[int, ...] = (1, 5)
    ) -> Dict[str, Any]:
        """
        评估分类任务精度

        Args:
            data_loader: 测试数据加载器
            topk: 需要计算的top-k值

        Returns:
            包含精度指标的字典
        """
        print("\n🎯 评估分类精度...")

        def _evaluate_model(model):
            all_outputs = []
            all_targets = []

            with torch.no_grad():
                for batch in data_loader:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = model(inputs)
                    all_outputs.append(outputs.cpu())
                    all_targets.append(targets.cpu())

            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            return compute_accuracy(all_outputs, all_targets, topk=topk)

        orig_acc = _evaluate_model(self.original_model)
        print(f"  原始模型: {', '.join([f'{k}: {v * 100:.2f}%' for k, v in orig_acc.items()])}")

        quant_acc = None
        if self.quantized_model is not None:
            quant_acc = _evaluate_model(self.quantized_model)
            print(f"  量化模型: {', '.join([f'{k}: {v * 100:.2f}%' for k, v in quant_acc.items()])}")

            # 计算精度下降
            for k in topk:
                drop = (orig_acc[f'top{k}'] - quant_acc[f'top{k}']) * 100
                print(f"  Top-{k} 精度下降: {drop:.4f}%")

        result = {
            'original': orig_acc,
            'quantized': quant_acc,
        }

        self.results['accuracy_classification'] = result
        return result

    def evaluate_image_metrics(
        self,
        data_loader: torch.utils.data.DataLoader,
        data_range: float = 1.0
    ) -> Dict[str, Any]:
        """
        评估图像任务指标（PSNR, SSIM等）

        Args:
            data_loader: 测试数据加载器，返回 (input, target)
            data_range: 数据范围

        Returns:
            包含图像指标的字典
        """
        print("\n🖼️  评估图像指标...")

        def _evaluate_model(model):
            all_psnr = []
            all_ssim = []
            all_l1 = []
            all_l2 = []

            with torch.no_grad():
                for batch in data_loader:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = model(inputs)

                    all_psnr.append(compute_psnr(outputs, targets, data_range=data_range).cpu())
                    all_ssim.append(compute_ssim(outputs, targets, data_range=data_range).cpu())
                    all_l1.append(compute_l1_error(outputs, targets).cpu())
                    all_l2.append(compute_l2_error(outputs, targets).cpu())

            return {
                'psnr': torch.stack(all_psnr).mean().item(),
                'ssim': torch.stack(all_ssim).mean().item(),
                'l1': torch.stack(all_l1).mean().item(),
                'l2': torch.stack(all_l2).mean().item(),
            }

        orig_metrics = _evaluate_model(self.original_model)
        print(f"  原始模型 - PSNR: {orig_metrics['psnr']:.2f} dB, SSIM: {orig_metrics['ssim']:.4f}")

        quant_metrics = None
        if self.quantized_model is not None:
            quant_metrics = _evaluate_model(self.quantized_model)
            print(f"  量化模型 - PSNR: {quant_metrics['psnr']:.2f} dB, SSIM: {quant_metrics['ssim']:.4f}")

            psnr_drop = orig_metrics['psnr'] - quant_metrics['psnr']
            ssim_drop = orig_metrics['ssim'] - quant_metrics['ssim']
            print(f"  PSNR下降: {psnr_drop:.4f} dB, SSIM下降: {ssim_drop:.6f}")

        result = {
            'original': orig_metrics,
            'quantized': quant_metrics,
        }

        self.results['image_metrics'] = result
        return result

    def evaluate_output_similarity(
        self,
        dummy_input: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]
    ) -> Dict[str, float]:
        """
        评估量化前后输出的相似度

        Args:
            dummy_input: 输入张量

        Returns:
            包含相似度指标的字典
        """
        if self.quantized_model is None:
            raise ValueError("请先设置量化模型")

        print("\n📊 评估输出相似度...")

        # 移动输入到设备
        if isinstance(dummy_input, dict):
            dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
        elif isinstance(dummy_input, tuple):
            dummy_input = tuple(v.to(self.device) for v in dummy_input)
        else:
            dummy_input = dummy_input.to(self.device)

        with torch.no_grad():
            if isinstance(dummy_input, dict):
                orig_output = self.original_model(**dummy_input)
                quant_output = self.quantized_model(**dummy_input)
            elif isinstance(dummy_input, tuple):
                orig_output = self.original_model(*dummy_input)
                quant_output = self.quantized_model(*dummy_input)
            else:
                orig_output = self.original_model(dummy_input)
                quant_output = self.quantized_model(dummy_input)

        # 计算各种相似度指标
        l1_error = compute_l1_error(quant_output, orig_output).item()
        l2_error = compute_l2_error(quant_output, orig_output).item()
        cos_sim = compute_cosine_similarity(
            quant_output.flatten(1),
            orig_output.flatten(1)
        ).item()

        print(f"  L1误差: {l1_error:.6f}")
        print(f"  L2误差: {l2_error:.6f}")
        print(f"  余弦相似度: {cos_sim:.6f}")

        result = {
            'l1_error': l1_error,
            'l2_error': l2_error,
            'cosine_similarity': cos_sim,
        }

        self.results['output_similarity'] = result
        return result

    def generate_report(self, format: str = 'table') -> str:
        """
        生成评估报告

        Args:
            format: 'table' 或 'markdown'

        Returns:
            报告字符串
        """
        report = []
        report.append("=" * 80)
        report.append("AutoQuant 量化评估报告")
        report.append("=" * 80)

        # 模型大小
        if 'size' in self.results:
            size_data = self.results['size']
            report.append("\n【模型大小】")
            size_table = [
                ["原始模型", f"{size_data['original_mb']:.2f} MB", "-"],
            ]
            if size_data['quantized_mb'] is not None:
                size_table.append([
                    "量化模型",
                    f"{size_data['quantized_mb']:.2f} MB",
                    f"{size_data['compression_ratio']:.2f}x"
                ])
            report.append(tabulate(size_table, headers=["", "大小", "压缩比"], tablefmt="grid"))

        # 推理速度
        if 'speed' in self.results:
            speed_data = self.results['speed']
            report.append("\n【推理速度】")
            speed_table = [
                ["原始模型",
                 f"{speed_data['original']['avg_time_ms']:.2f} ms",
                 f"{speed_data['original']['throughput']:.2f} iters/s", "-"],
            ]
            if speed_data['quantized'] is not None:
                speed_table.append([
                    "量化模型",
                    f"{speed_data['quantized']['avg_time_ms']:.2f} ms",
                    f"{speed_data['quantized']['throughput']:.2f} iters/s",
                    f"{speed_data['speedup']:.2f}x"
                ])
            report.append(tabulate(speed_table, headers=["", "单次耗时", "吞吐量", "加速比"], tablefmt="grid"))

        # 输出相似度
        if 'output_similarity' in self.results:
            sim_data = self.results['output_similarity']
            report.append("\n【输出相似度】")
            sim_table = [
                ["L1误差", f"{sim_data['l1_error']:.6f}"],
                ["L2误差", f"{sim_data['l2_error']:.6f}"],
                ["余弦相似度", f"{sim_data['cosine_similarity']:.6f}"],
            ]
            report.append(tabulate(sim_table, headers=["指标", "数值"], tablefmt="grid"))

        # 分类精度
        if 'accuracy_classification' in self.results:
            acc_data = self.results['accuracy_classification']
            report.append("\n【分类精度】")

            # 获取topk键
            topk_keys = sorted(acc_data['original'].keys())
            acc_table = []
            orig_row = ["原始模型"] + [f"{acc_data['original'][k] * 100:.2f}%" for k in topk_keys]
            acc_table.append(orig_row)

            if acc_data['quantized'] is not None:
                quant_row = ["量化模型"] + [f"{acc_data['quantized'][k] * 100:.2f}%" for k in topk_keys]
                acc_table.append(quant_row)

                drop_row = ["精度下降"] + [
                    f"{(acc_data['original'][k] - acc_data['quantized'][k]) * 100:.4f}%"
                    for k in topk_keys
                ]
                acc_table.append(drop_row)

            report.append(tabulate(acc_table, headers=[""] + topk_keys, tablefmt="grid"))

        # 图像指标
        if 'image_metrics' in self.results:
            img_data = self.results['image_metrics']
            report.append("\n【图像指标】")
            img_table = [
                ["原始模型",
                 f"{img_data['original']['psnr']:.2f} dB",
                 f"{img_data['original']['ssim']:.4f}",
                 f"{img_data['original']['l1']:.6f}",
                 f"{img_data['original']['l2']:.6f}"],
            ]
            if img_data['quantized'] is not None:
                img_table.append([
                    "量化模型",
                    f"{img_data['quantized']['psnr']:.2f} dB",
                    f"{img_data['quantized']['ssim']:.4f}",
                    f"{img_data['quantized']['l1']:.6f}",
                    f"{img_data['quantized']['l2']:.6f}",
                ])
                img_table.append([
                    "变化",
                    f"{img_data['original']['psnr'] - img_data['quantized']['psnr']:.4f} dB",
                    f"{img_data['original']['ssim'] - img_data['quantized']['ssim']:.6f}",
                    f"{img_data['quantized']['l1'] - img_data['original']['l1']:.6f}",
                    f"{img_data['quantized']['l2'] - img_data['original']['l2']:.6f}",
                ])

            report.append(tabulate(img_table, headers=["", "PSNR", "SSIM", "L1", "L2"], tablefmt="grid"))

        report.append("\n" + "=" * 80)
        return "\n".join(report)

    def print_report(self):
        """打印报告"""
        print(self.generate_report())
