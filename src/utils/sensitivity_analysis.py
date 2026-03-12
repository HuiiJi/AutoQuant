"""
Op敏感度分析 - 分析每个操作对量化的敏感度
帮助决定哪些操作应该量化，哪些操作应该保持浮点精度

Author: jihui
Date: 2026-03-13
Desc: 正确的敏感度分析流程：
      1. 基准1：原始模型（全部浮点）- 最佳情况
      2. 基准2：全部量化 - 最差情况
      3. 对每个层：只跳过这一层，其他都量化
      4. 敏感度分数 = (跳过该层后的改善) / (全部量化的总误差)
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
import copy
from collections import defaultdict
from tabulate import tabulate
import os
from datetime import datetime


class SensitivityAnalyzer:
    """
    操作敏感度分析器 - 正确的 PTQ 敏感度分析流程
    """
    
    def __init__(
        self,
        model: nn.Module,
        qconfig,
        metric_fn: Optional[Callable] = None,
    ):
        """
        Args:
            model: 待分析的模型
            qconfig: 量化配置
            metric_fn: 评估指标函数，如果为None，默认使用输出差异
        """
        self.model = model
        self.qconfig = qconfig
        self.metric_fn = metric_fn or self._default_metric
        self.sensitivity_scores: Dict[str, float] = {}
        
        # 基准分数
        self.original_score: float = 0.0  # 原始模型（全浮点）的分数（应该是0）
        self.full_quant_score: float = 0.0  # 全部量化的分数
        
        # 缓存
        self.original_output = None
        self.calib_data = None
        self.dummy_input = None
        
    @staticmethod
    def _default_metric(original_output, quantized_output) -> float:
        """
        默认评估指标：计算输出的MSE
        支持 tuple 输出（取第一个元素）
        """
        def get_first_tensor(x):
            if isinstance(x, tuple):
                return x[0] if len(x) > 0 else None
            return x
        
        orig = get_first_tensor(original_output)
        quant = get_first_tensor(quantized_output)
        
        if orig is None or quant is None:
            return float('inf')
        
        return torch.nn.functional.mse_loss(orig, quant).item()
    
    def analyze(
        self,
        dummy_input: torch.Tensor,
        calib_data: Optional[List] = None,
        skip_layers: Optional[List[str]] = None,
        only_layers: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        执行敏感度分析 - 正确的逻辑！
        
        正确逻辑：
        1. 基准1: 原始模型（全浮点）- 最佳情况
        2. 基准2: 全部量化 - 最差情况  
        3. 对每个层：只跳过这一层，其他都量化
        4. 敏感度分数 = (full_quant_score - skip_this_layer_score) / full_quant_score
           - 分数越高，表示跳过这一层带来的改善越大 → 越敏感，越应该跳过
        
        Args:
            dummy_input: 用于分析的输入
            calib_data: 校准数据（可选，如果不提供就用 dummy_input）
            skip_layers: 跳过分析的层列表
            only_layers: 只分析的层列表
        
        Returns:
            敏感度分数字典
        """
        from autoquant import ModelQuantizer
        
        skip_layers = skip_layers or []
        self.calib_data = calib_data if calib_data is not None else [dummy_input]
        self.dummy_input = dummy_input
        self.model.eval()
        
        # ====================================================================
        # 步骤 1: 获取原始模型的输出作为基准（全浮点）
        # ====================================================================
        print("\n" + "=" * 80)
        print("步骤 1/5: 获取原始模型基准（全浮点）")
        print("=" * 80)
        
        with torch.no_grad():
            self.original_output = self.model(dummy_input)
        
        # 如果是 tuple，取第一个元素
        if isinstance(self.original_output, tuple):
            self.original_output = self.original_output[0]
        
        self.original_score = 0.0  # 和自己比当然是0
        print(f"    ✓ 原始输出形状: {self.original_output.shape}")
        print(f"    ✓ 原始模型分数 (基准): {self.original_score:.10f}")
        
        # ====================================================================
        # 步骤 2: 获取全部量化的模型分数（最差情况）
        # ====================================================================
        print("\n" + "=" * 80)
        print("步骤 2/5: 获取全部量化模型基准（最差情况）")
        print("=" * 80)
        
        self.full_quant_score = self._analyze_full_quant(dummy_input, self.calib_data)
        print(f"    ✓ 全部量化分数: {self.full_quant_score:.10f}")
        
        if self.full_quant_score == float('inf') or self.full_quant_score <= 0:
            print("    ⚠️  全部量化失败或分数异常，无法继续分析")
            return {}
        
        # ====================================================================
        # 步骤 3: 收集所有可量化的层
        # ====================================================================
        print("\n" + "=" * 80)
        print("步骤 3/5: 收集可量化层")
        print("=" * 80)
        
        quantizable_layers = self._get_quantizable_layers()
        
        if only_layers:
            quantizable_layers = [name for name in quantizable_layers if name in only_layers]
        
        quantizable_layers = [name for name in quantizable_layers if name not in skip_layers]
        
        print(f"    ✓ 找到 {len(quantizable_layers)} 个可量化层")
        
        # ====================================================================
        # 步骤 4: 逐个分析每个层 - 只跳过这一层，其他都量化
        # ====================================================================
        print("\n" + "=" * 80)
        print("步骤 4/5: 分析每个层的敏感度（只跳过当前层）")
        print("=" * 80)
        
        for i, layer_name in enumerate(quantizable_layers):
            print(f"\n    [{i+1}/{len(quantizable_layers)}] 分析层: {layer_name}")
            score = self._analyze_skip_single_layer(
                layer_name, dummy_input, self.calib_data
            )
            
            # 计算敏感度分数：(全部量化误差 - 跳过该层的误差) / 全部量化误差
            # 表示跳过该层能挽回多少比例的误差
            if score != float('inf') and self.full_quant_score > 0:
                improvement = self.full_quant_score - score
                sensitivity_score = improvement / self.full_quant_score
                # 确保分数在合理范围
                sensitivity_score = max(0.0, min(1.0, sensitivity_score))
            else:
                sensitivity_score = 0.0
            
            self.sensitivity_scores[layer_name] = sensitivity_score
            print(f"        跳过该层后的分数: {score:.10f}")
            print(f"        敏感度分数 (挽回比例): {sensitivity_score:.4f} ({sensitivity_score*100:.1f}%)")
        
        return self.sensitivity_scores
    
    def _get_quantizable_layers(self) -> List[str]:
        """
        获取所有可量化的层名称
        支持所有 ModelQuantizer 支持的模块
        """
        layers = []
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            quantizable_types = {
                'Conv1d', 'Conv2d', 'Conv3d',
                'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
                'Linear', 'Bilinear',
                'Embedding', 'EmbeddingBag',
                'ReLU', 'ReLU6', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'CELU', 'GELU', 'SiLU', 'Hardswish',
                'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
                'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
                'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
                'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d',
                'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
                'LayerNorm', 'GroupNorm',
                'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
            }
            if module_type in quantizable_types and name:
                layers.append(name)
        return layers
    
    def _analyze_full_quant(
        self,
        dummy_input: torch.Tensor,
        calib_data: List,
    ) -> float:
        """
        分析全部量化的模型（跳过0层）
        """
        from autoquant import ModelQuantizer
        
        try:
            model_copy = copy.deepcopy(self.model)
            quantizer = ModelQuantizer(model_copy, self.qconfig)
            
            prepared_model = quantizer.prepare(skip_layers=set())
            quantizer.calibrate(calib_data)
            quantized_model = quantizer.convert()
            
            with torch.no_grad():
                quantized_output = quantized_model(dummy_input)
            
            score = self.metric_fn(self.original_output, quantized_output)
            return score
            
        except Exception as e:
            print(f"        ⚠️  分析全部量化时出错: {e}")
            return float('inf')
    
    def _analyze_skip_single_layer(
        self,
        layer_name: str,
        dummy_input: torch.Tensor,
        calib_data: List,
    ) -> float:
        """
        分析只跳过当前层的情况（其他层都量化）
        
        这才是正确的敏感度分析！
        - 只跳过这一层不量化
        - 其他所有层都正常量化
        - 这样能准确衡量：如果不量化这一层，能挽回多少误差
        """
        from autoquant import ModelQuantizer
        
        try:
            model_copy = copy.deepcopy(self.model)
            quantizer = ModelQuantizer(model_copy, self.qconfig)
            
            # 只跳过当前层！
            prepared_model = quantizer.prepare(skip_layers={layer_name})
            quantizer.calibrate(calib_data)
            quantized_model = quantizer.convert()
            
            with torch.no_grad():
                quantized_output = quantized_model(dummy_input)
            
            score = self.metric_fn(self.original_output, quantized_output)
            return score
            
        except Exception as e:
            print(f"        ⚠️  分析层 {layer_name} 时出错: {e}")
            return float('inf')
    
    def generate_report(
        self, 
        sort_by: str = 'score', 
        ascending: bool = False,
        top_n_percent: float = 10.0,
    ) -> str:
        """
        生成敏感度分析报告
        
        Args:
            sort_by: 排序字段，'score' 或 'name'
            ascending: 是否升序（默认降序，敏感度高的在前）
            top_n_percent: 前 N% 视为高敏感度层
        
        Returns:
            格式化的报告字符串
        """
        if not self.sensitivity_scores:
            return "请先执行敏感度分析"
        
        # 准备数据
        data = []
        total_layers = len(self.sensitivity_scores)
        top_n_count = max(1, int(total_layers * top_n_percent / 100))
        
        # 排序
        sorted_items = sorted(
            self.sensitivity_scores.items(), 
            key=lambda x: x[1], 
            reverse=not ascending
        )
        
        # 标记前 N% 为高敏感度
        high_sensitivity_layers = set()
        for i, (name, score) in enumerate(sorted_items):
            if i < top_n_count and score != float('inf'):
                high_sensitivity_layers.add(name)
        
        for name, score in sorted_items:
            is_high_sensitivity = name in high_sensitivity_layers
            data.append({
                '层名称': name,
                '敏感度分数': score,
                '挽回比例': f'{score*100:.1f}%',
                '高敏感度': '✓' if is_high_sensitivity else '',
                '建议': '保持浮点' if is_high_sensitivity else '可以量化'
            })
        
        # 排序
        if sort_by == 'score':
            data.sort(key=lambda x: x['敏感度分数'], reverse=not ascending)
        else:
            data.sort(key=lambda x: x['层名称'])
        
        # 生成表格
        table = tabulate(
            data,
            headers='keys',
            tablefmt='grid',
            floatfmt='.6f'
        )
        
        # 添加总结
        quantizable_layers = sum(1 for d in data if d['建议'] == '可以量化')
        high_sens_count = sum(1 for d in data if d['高敏感度'] == '✓')
        
        summary = f"\n\n"
        summary += "=" * 80 + "\n"
        summary += "敏感度分析总结\n"
        summary += "=" * 80 + "\n"
        summary += f"  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"\n"
        summary += f"  基准分数:\n"
        summary += f"    - 原始模型 (全浮点): {self.original_score:.10f}\n"
        summary += f"    - 全部量化: {self.full_quant_score:.10f}\n"
        summary += f"\n"
        summary += f"  层统计:\n"
        summary += f"    - 总层数: {total_layers}\n"
        summary += f"    - 高敏感度层 (前 {top_n_percent}%): {high_sens_count} 个\n"
        summary += f"    - 建议量化的层: {quantizable_layers} 个\n"
        summary += f"    - 建议保持浮点的层: {total_layers - quantizable_layers} 个\n"
        summary += f"\n"
        summary += f"  高敏感度层列表 (建议保持浮点):\n"
        for d in data:
            if d['高敏感度'] == '✓':
                summary += f"    - {d['层名称']}: {d['敏感度分数']:.6f} ({d['挽回比例']})\n"
        
        return table + summary
    
    def get_recommended_layers(
        self, 
        threshold: Optional[float] = None,
        top_n_percent: float = 10.0,
    ) -> Tuple[List[str], List[str]]:
        """
        获取推荐的量化层和保持浮点的层
        
        Args:
            threshold: 敏感度阈值（可选，如果不提供，使用 top_n_percent）
            top_n_percent: 前 N% 视为高敏感度层
        
        Returns:
            (推荐量化的层列表, 推荐保持浮点的层列表)
        """
        if not self.sensitivity_scores:
            return [], []
        
        total_layers = len(self.sensitivity_scores)
        top_n_count = max(1, int(total_layers * top_n_percent / 100))
        
        sorted_items = sorted(
            self.sensitivity_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        high_sensitivity_layers = set()
        if threshold is not None:
            for name, score in sorted_items:
                if score > threshold and score != float('inf'):
                    high_sensitivity_layers.add(name)
        else:
            for i, (name, score) in enumerate(sorted_items):
                if i < top_n_count and score != float('inf'):
                    high_sensitivity_layers.add(name)
        
        quantizable = []
        not_quantizable = []
        
        for name, score in self.sensitivity_scores.items():
            if name in high_sensitivity_layers:
                not_quantizable.append(name)
            else:
                quantizable.append(name)
        
        return quantizable, not_quantizable
    
    def plot_sensitivity(
        self,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
        top_n: Optional[int] = None,
        top_n_percent: float = 10.0,
    ):
        """
        绘制敏感度分析图（完整版 - 4张子图）
        
        Args:
            figsize: 图大小
            save_path: 保存路径（可选）
            top_n: 只显示前 N 个层（可选）
            top_n_percent: 前 N% 视为高敏感度层
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("需要安装 matplotlib 和 numpy: pip install matplotlib numpy")
            return
        
        if not self.sensitivity_scores:
            print("请先执行敏感度分析")
            return
        
        # 准备数据
        valid_data = [(name, score) for name, score in self.sensitivity_scores.items() 
                      if score != float('inf')]
        valid_data.sort(key=lambda x: x[1], reverse=True)
        
        if top_n:
            valid_data = valid_data[:top_n]
        
        layers = [d[0] for d in valid_data]
        scores = [d[1] for d in valid_data]
        
        # 确定高敏感度层
        total_layers = len(valid_data)
        top_n_count = max(1, int(total_layers * top_n_percent / 100))
        high_sens_mask = [i < top_n_count for i in range(len(layers))]
        
        # 创建图表 - 2x2 布局
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])
        
        # 图 1: 敏感度柱状图（主图）
        ax1 = fig.add_subplot(gs[0, 0])
        y_pos = np.arange(len(layers))
        colors = ['#ff6b6b' if m else '#4ecdc4' for m in high_sens_mask]
        bars = ax1.barh(y_pos, scores, color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(layers, fontsize=7)
        ax1.set_xlabel('敏感度分数 (挽回比例)', fontsize=11)
        ax1.set_ylabel('层名称', fontsize=11)
        ax1.set_title('各层敏感度分析 (红色=建议保持浮点)', fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=6)
        
        # 图 2: 敏感度分布直方图
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(scores, bins=30, color='#4ecdc4', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('敏感度分数', fontsize=11)
        ax2.set_ylabel('层数', fontsize=11)
        ax2.set_title('敏感度分数分布', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 图 3: 累积敏感度曲线
        ax3 = fig.add_subplot(gs[1, 0])
        sorted_scores = sorted(scores, reverse=True)
        cumulative = np.cumsum(sorted_scores)
        cumulative_normalized = cumulative / cumulative[-1]
        ax3.plot(range(1, len(cumulative_normalized)+1), cumulative_normalized, 
                'o-', color='#9b59b6', linewidth=2, markersize=4)
        ax3.axvline(x=top_n_count, color='#ff6b6b', linestyle='--', 
                   label=f'前 {top_n_percent}% 层 ({top_n_count} 个)')
        ax3.set_xlabel('层数 (按敏感度排序)', fontsize=11)
        ax3.set_ylabel('累积敏感度比例', fontsize=11)
        ax3.set_title('累积敏感度曲线', fontsize=13, fontweight='bold')
        ax3.grid(alpha=0.3)
        ax3.legend(fontsize=10)
        
        # 图 4: 饼图 - 量化 vs 保持浮点
        ax4 = fig.add_subplot(gs[1, 1])
        quant_count = len(layers) - top_n_count
        skip_count = top_n_count
        labels = [f'建议量化 ({quant_count})', f'建议保持浮点 ({skip_count})']
        sizes = [quant_count, skip_count]
        colors = ['#4ecdc4', '#ff6b6b']
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax4.set_title('量化建议分布', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"    ✓ 图已保存到: {save_path}")
        else:
            plt.show()
    
    def save_report(
        self,
        save_path: str,
        top_n_percent: float = 10.0,
    ):
        """
        保存报告到文件
        
        Args:
            save_path: 保存路径
            top_n_percent: 前 N% 视为高敏感度层
        """
        report = self.generate_report(top_n_percent=top_n_percent)
        
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"    ✓ 报告已保存到: {save_path}")
    
    def save_results(
        self,
        output_dir: str,
        top_n_percent: float = 10.0,
    ):
        """
        保存所有结果到指定目录（报告 + 图表）
        
        Args:
            output_dir: 输出目录
            top_n_percent: 前 N% 视为高敏感度层
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report_path = os.path.join(output_dir, f'sensitivity_report_{timestamp}.txt')
        self.save_report(report_path, top_n_percent=top_n_percent)
        
        plot_path = os.path.join(output_dir, f'sensitivity_plot_{timestamp}.png')
        try:
            self.plot_sensitivity(save_path=plot_path, top_n_percent=top_n_percent)
        except Exception as e:
            print(f"    ⚠️  图表生成失败: {e}")
        
        print(f"\n    ✓ 所有结果已保存到: {output_dir}")
