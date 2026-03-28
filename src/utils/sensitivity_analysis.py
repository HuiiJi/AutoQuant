"""
Sensitivity Analyzer - 量化敏感度分析工具

Author: jihui
Date: 2026-03-13
Desc:
    量化敏感度分析工具，通过逐层评估量化误差来确定哪些层应该保持浮点精度。
    分析结果帮助用户在模型精度和量化压缩率之间取得最佳平衡。

    分析流程：
    1. 基准测试：原始模型（全部浮点）vs 全部量化模型
    2. 逐层分析：对每个可量化层，只跳过该层而量化其他所有层
    3. 敏感度分数 = (全部量化误差 - 跳过该层后的误差) / 全部量化误差

    优化特性：
    - 支持 skip_layers 前缀匹配，可一次性跳过整个模块树
    - calib_data 预加载到设备，避免重复 CPU-GPU 数据传输
    - 强化内存清理机制 (del + gc + cuda_empty_cache)
    - 使用 inference_mode 替代 no_grad 加速推理
    - 默认校准样本数降至 3，大幅提升分析速度
"""
import os
import gc
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tabulate import tabulate
from tqdm import tqdm

from autoquant.utils.quantizable_ops import get_quantizable_layers


class SensitivityAnalyzer:

    def __init__(
        self,
        model: nn.Module,
        qconfig: dict,
        metric_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.qconfig = qconfig
        self.metric_fn = metric_fn or self._default_metric

        self.sensitivity_scores: Dict[str, float] = {}
        self.original_score: float = 0.0
        self.full_quant_score: float = 0.0
        self.original_outputs: Optional[List] = None
        self.calib_data: Optional[List] = None
        self.dummy_input: Optional[torch.Tensor] = None
        self._user_skip_layers: List[str] = []

        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _default_metric(original_outputs, quantized_outputs) -> float:
        """
        默认评估指标：计算一批样本的平均 MSE
        优化：直接在 GPU 上计算，最后取 item()
        """
        def get_first_tensor(x):
            if isinstance(x, tuple):
                return x[1] if len(x) > 0 else None
            return x

        total_mse = 0.0
        count = 0

        # 优化：使用 torch.inference_mode 上下文
        with torch.inference_mode():
            for orig_out, quant_out in zip(original_outputs, quantized_outputs):
                orig = get_first_tensor(orig_out)
                quant = get_first_tensor(quant_out)

                if orig is not None and quant is not None:
                    # 确保在同一设备计算
                    if orig.device != quant.device:
                        quant = quant.to(orig.device)
                    mse = torch.nn.functional.mse_loss(orig, quant)
                    total_mse += mse.item()
                    count += 1

        if count == 0:
            return float('inf')

        return total_mse / count

    def analyze(
        self,
        dummy_input: torch.Tensor,
        calib_data: Optional[List] = None,
        skip_layers: Optional[List[str]] = None,
        only_layers: Optional[List[str]] = None,
        max_calib_samples: int = 3,  # 优化：默认改为 3，速度提升明显
    ) -> Dict[str, float]:
        """
        执行敏感度分析 - 工程优化版

        skip_layers: 用户自定义要跳过的层（支持模块名前缀匹配）
                     例如 skip_layers=['gf'] 会跳过 gf 及其所有子模块
        """
        # 保存 skip_layers（供后续使用）
        self._user_skip_layers = skip_layers or []

        # 优化：限制校准样本数
        if calib_data is not None and len(calib_data) > max_calib_samples:
            print(f"    优化：使用 {max_calib_samples}/{len(calib_data)} 个校准样本进行敏感度分析")
            calib_data_analysis = calib_data[:max_calib_samples]
        else:
            calib_data_analysis = calib_data if calib_data is not None else [dummy_input]

        self.calib_data = calib_data
        self.dummy_input = dummy_input
        
        # ====================================================================
        # 优化：一次性将校准数据移动到设备，避免循环内重复 transfer
        # ====================================================================
        device = next(self.model.parameters()).device
        calib_data_on_device = []
        with torch.inference_mode():
            for inp in calib_data_analysis:
                if isinstance(inp, torch.Tensor):
                    calib_data_on_device.append(inp.to(device))
                elif isinstance(inp, (list, tuple)):
                    calib_data_on_device.append([x.to(device) if isinstance(x, torch.Tensor) else x for x in inp])
                else:
                    calib_data_on_device.append(inp)

        # ====================================================================
        # 步骤 1: 获取原始模型输出
        # ====================================================================
        self.original_outputs = []
        with torch.inference_mode():
            for inp in calib_data_on_device:
                out = self.model(inp)
                if isinstance(out, tuple):
                    res_a = out[0]
                    res_b = out[1]
                    out_ = res_a * inp + res_b + inp
                else:
                    out_ = out + inp
                # 优化：原始输出也保留在设备上，避免反复 CPU-GPU 拷贝
                self.original_outputs.append(out_)

        self.original_score = 0.0

        # ====================================================================
        # 步骤 2: 获取全部量化的模型分数
        # ====================================================================
        self.full_quant_score = self._analyze_full_quant(calib_data_on_device)

        if self.full_quant_score == float('inf') or self.full_quant_score <= 0:
            return {}

        # ====================================================================
        # 步骤 3: 收集所有可量化的层（使用用户自定义的 skip_layers 过滤）
        # ====================================================================
        quantizable_layers = self._get_quantizable_layers(skip_layers=self._user_skip_layers)

        if only_layers:
            quantizable_layers = [name for name in quantizable_layers if name in only_layers]

        print("    分析 {} 个层 (使用 {} 个校准样本)...".format(
            len(quantizable_layers), len(calib_data_on_device)))

        # ====================================================================
        # 步骤 4: 逐个分析每个层
        # ====================================================================
        pbar = tqdm(
            total=len(quantizable_layers),
            desc="    敏感度分析",
            unit="层"
        )
        
        for i, layer_name in enumerate(quantizable_layers):
            pbar.set_description(f"    分析 - {layer_name[-20:]}") # 只显示层名后 20 字符，避免进度条过长

            score = self._analyze_skip_single_layer(
                layer_name, calib_data_on_device
            )

            if score != float('inf') and self.full_quant_score > 0:
                improvement = self.full_quant_score - score
                sensitivity_score = improvement / self.full_quant_score
                sensitivity_score = max(0.0, min(1.0, sensitivity_score))
            else:
                sensitivity_score = 0.0

            self.sensitivity_scores[layer_name] = sensitivity_score
            pbar.update(1)
      
        pbar.close()
        
        # 最终清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
      
        return self.sensitivity_scores

    def _get_quantizable_layers(
        self,
        skip_layers: Optional[List[str]] = None
    ) -> List[str]:
        """
        获取所有可量化的层名称 - 使用统一的工具函数

        skip_layers: 要跳过的层列表（支持前缀匹配）
                      例如 skip_layers=['gf'] 会排除 gf 及其所有子模块

        Returns:
            List[str]: 可量化层名称列表（已过滤）
        """
        # 使用统一的工具函数获取
        all_layers = get_quantizable_layers(self.model)

        # 过滤掉要跳过的层
        if skip_layers:
            filtered_layers = []
            for layer in all_layers:
                should_skip = False
                for skip_pattern in skip_layers:
                    # 前缀匹配：layer 以 skip_pattern 开头，或者是 skip_pattern 本身
                    if layer == skip_pattern or layer.startswith(skip_pattern + '.'):
                        should_skip = True
                        break
                if not should_skip:
                    filtered_layers.append(layer)
            return filtered_layers

        return all_layers

    def _analyze_full_quant(self, calib_data: List) -> float:
        """
        分析全部量化的模型
        
        内存优化：
        1. deepcopy 保留（稳定性优先）
        2. 直接计算分数，不保存中间结果
        3. inference_mode 替代 no_grad
        4. finally 确保清理
        """
        from autoquant import ModelQuantizer

        model_copy = None
        quantized_model = None
        try:
            # 稳定性优先：使用 deepcopy
            model_copy = copy.deepcopy(self.model)

            quantizer = ModelQuantizer(model_copy, self.qconfig)

            prepared_model = quantizer.prepare(skip_layers=set())
            quantizer.calibrate(calib_data, verbose=False)
            quantized_model = quantizer.convert()

            # 直接累加分数
            with torch.inference_mode():
                score = 0.0
                for inp, orig_out in zip(calib_data, self.original_outputs):
                    quant_out = quantized_model(inp)
                    if isinstance(quant_out, tuple):
                        quant_res_a = quant_out[0]
                        quant_res_b = quant_out[1]
                        quant_out = quant_res_a * inp + quant_res_b + inp
                    else:
                        quant_out = quant_out + inp
                    score += self.metric_fn([orig_out], [quant_out])
                score /= len(calib_data)

            return score

        except Exception as e:
            return float('inf')
        finally:
            # 确保清理
            del model_copy, quantized_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _analyze_skip_single_layer(self, layer_name: str, calib_data: List) -> float:
        """
        分析只跳过当前层的情况
        
        内存优化：
        1. deepcopy 保留（稳定性优先）
        2. 不保存中间结果
        3. 每层分析完立即清理
        """
        from autoquant import ModelQuantizer

        model_copy = None
        quantized_model = None
        try:
            # 稳定性优先：使用 deepcopy
            model_copy = copy.deepcopy(self.model)

            quantizer = ModelQuantizer(model_copy, self.qconfig)

            prepared_model = quantizer.prepare(skip_layers={layer_name})
            quantizer.calibrate(calib_data, verbose=False)
            quantized_model = quantizer.convert()

            # 直接累加分数
            with torch.inference_mode():
                score = 0.0
                for inp, orig_out in zip(calib_data, self.original_outputs):
                    quant_out = quantized_model(inp)
                    if isinstance(quant_out, tuple):
                        quant_out = quant_out[1]
                    score += self.metric_fn([orig_out], [quant_out])
                score /= len(calib_data)

            return score

        except Exception as e:
            return float('inf')
        finally:
            # 立即清理
            del model_copy, quantized_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_report(
        self,
        sort_by: str = 'score',
        ascending: bool = False,
        top_n_percent: float = 10.0,
    ) -> str:
        if not self.sensitivity_scores:
            return "请先执行敏感度分析"

        data = []
        total_layers = len(self.sensitivity_scores)
        top_n_count = max(1, int(total_layers * top_n_percent / 100))

        sorted_items = sorted(
            self.sensitivity_scores.items(),
            key=lambda x: x[1],
            reverse=not ascending
        )

        high_sensitivity_layers = set()
        for i, (name, score) in enumerate(sorted_items):
            if i < top_n_count and score != float('inf'):
                high_sensitivity_layers.add(name)

        for name, score in sorted_items:
            is_high_sensitivity = name in high_sensitivity_layers
            data.append({
                '层名称': name,
                '敏感度分数': score,
                '挽回比例': f'{score * 100:.1f}%',
                '高敏感度': '✓' if is_high_sensitivity else '',
                '建议': '保持浮点' if is_high_sensitivity else '可以量化'
            })

        if sort_by == 'score':
            data.sort(key=lambda x: x['敏感度分数'], reverse=not ascending)
        else:
            data.sort(key=lambda x: x['层名称'])

        table = tabulate(
            data,
            headers='keys',
            tablefmt='grid',
            floatfmt='.6f'
        )

        all_scores = [s for s in self.sensitivity_scores.values() if s != float('inf')]
        if not all_scores:
            return table + "\n无有效数据"
            
        mean_score = np.mean(all_scores)
        median_score = np.median(all_scores)

        sorted_scores = sorted(all_scores, reverse=True)
        cumulative = np.cumsum(sorted_scores)
        cumulative_normalized = cumulative / cumulative[-1]

        quantizable_layers = sum(1 for d in data if d['建议'] == '可以量化')
        high_sens_count = sum(1 for d in data if d['高敏感度'] == '✓')

        summary = f"\n\n"
        summary += "=" * 80 + "\n"
        summary += "敏感度分析总结 (优化版)\n"
        summary += "=" * 80 + "\n"
        summary += f"  分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"\n"
        summary += f"  基准分数:\n"
        summary += f"    - 原始模型 (全浮点): {self.original_score:.10f}\n"
        summary += f"    - 全部量化：{self.full_quant_score:.10f}\n"
        summary += f"\n"
        summary += f"  层统计:\n"
        summary += f"    - 总层数：{total_layers}\n"
        summary += f"    - 高敏感度层 (前 {top_n_percent}%): {high_sens_count} 个\n"
        summary += f"    - 建议量化的层：{quantizable_layers} 个\n"
        summary += f"\n"
        summary += f"  敏感度分数统计:\n"
        summary += f"    - 均值 (Mean): {mean_score:.6f}\n"
        summary += f"    - 中位数 (Median): {median_score:.6f}\n"
        summary += f"\n"
        summary += f"  【关键分析】模型量化可行性 (查看 01_layer_sensitivity_top.jpg):\n"
        summary += f"    - 如果分数大多接近 0 → 模型容易量化，几乎所有层都可以量化\n"
        summary += f"    - 如果分数呈现梯度分布 → 模型难量化，需要仔细选择跳过的层\n"
        summary += f"\n"
        summary += f"  【关键分析】最佳 Skip 比例 (查看 03_cumulative_sensitivity.jpg):\n"
        summary += f"    查找曲线的拐点 (elbow point)，即继续增加层收益递减的点\n"
        summary += f"    - Skip  5%: 覆盖 {cumulative_normalized[max(0, int(total_layers * 0.05) - 1)]:.1%} 敏感度\n"
        summary += f"    - Skip 10%: 覆盖 {cumulative_normalized[max(0, int(total_layers * 0.10) - 1)]:.1%} 敏感度\n"
        summary += f"    - Skip 20%: 覆盖 {cumulative_normalized[max(0, int(total_layers * 0.20) - 1)]:.1%} 敏感度\n"
        summary += f"\n"

        summary += f"  【自动推荐】:\n"
        try:
            opt_count_95, coverage_95, _ = self.find_optimal_skip_count(method='coverage', target_coverage=0.95)
            summary += f"    方案 - 覆盖 95%: Skip {opt_count_95} 层 ({opt_count_95 / total_layers * 100:.1f}%)，覆盖 {coverage_95:.1%} 敏感度  ⭐ (默认)\n"
            summary += f"    推荐：优先使用 覆盖 95%，在保证精度的前提下获得较好压缩率！\n"
        except Exception as e:
            summary += f"    (自动推荐计算失败：{str(e)})\n"

        summary += f"\n"
        summary += f"  高敏感度层列表 (建议保持浮点):\n"
        for d in data:
            if d['高敏感度'] == '✓':
                summary += f"    - {d['层名称']}: {d['敏感度分数']:.6f} ({d['挽回比例']})\n"

        return table + summary

    def find_optimal_skip_count(
        self,
        method: str = 'elbow',
        target_coverage: float = 0.90,
    ) -> Tuple[int, float, str]:
        if not self.sensitivity_scores:
            return 0, 0.0, "No sensitivity scores available"

        valid_data = [(name, score) for name, score in self.sensitivity_scores.items()
                      if score != float('inf')]
        valid_data.sort(key=lambda x: x[1], reverse=True)
        scores = [d[1] for d in valid_data]
        total_layers = len(scores)

        if total_layers == 0:
            return 0, 0.0, "No valid layers"

        cumulative = np.cumsum(scores)
        cumulative_normalized = cumulative / cumulative[-1] if cumulative[-1] > 0 else cumulative

        if method == 'elbow':
            if total_layers < 3:
                opt_count = max(1, total_layers // 10)
                coverage = cumulative_normalized[opt_count - 1] if opt_count > 0 else 0
                return opt_count, coverage, "Elbow method (fallback)"

            first_deriv = np.diff(cumulative_normalized)
            second_deriv = np.diff(first_deriv)
            elbow_idx = np.argmax(np.abs(second_deriv)) + 1
            opt_count = max(1, min(elbow_idx, total_layers // 2))
            coverage = cumulative_normalized[opt_count - 1]
            desc = f"Elbow method (拐点 at {opt_count} layers)"

        elif method == 'coverage':
            opt_count = 0
            for i in range(total_layers):
                if cumulative_normalized[i] >= target_coverage:
                    opt_count = i + 1
                    break
            if opt_count == 0:
                opt_count = total_layers
            coverage = cumulative_normalized[opt_count - 1] if opt_count > 0 else 0
            desc = f"Coverage method (target {target_coverage * 100:.0f}%)"

        else:
            raise ValueError(f"Unknown method: {method}")

        return opt_count, coverage, desc

    def get_recommended_layers(
        self,
        threshold: Optional[float] = None,
        top_n_percent: Optional[float] = None,
        auto_method: Optional[str] = 'elbow',
        auto_target_coverage: float = 0.90,
    ) -> Tuple[List[str], List[str], Dict]:
        if not self.sensitivity_scores:
            return [], [], {}

        total_layers = len(self.sensitivity_scores)
        sorted_items = sorted(
            self.sensitivity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        high_sensitivity_layers = set()
        recommendation_info = {}

        if threshold is not None:
            for name, score in sorted_items:
                if score > threshold and score != float('inf'):
                    high_sensitivity_layers.add(name)
            recommendation_info = {'method': 'user_threshold', 'threshold': threshold}
        elif top_n_percent is not None:
            top_n_count = max(1, int(total_layers * top_n_percent / 100))
            for i, (name, score) in enumerate(sorted_items):
                if i < top_n_count and score != float('inf'):
                    high_sensitivity_layers.add(name)
            recommendation_info = {'method': 'user_percent', 'top_n_percent': top_n_percent}
        else:
            opt_count_95, coverage_95, desc_95 = self.find_optimal_skip_count(method='coverage', target_coverage=0.95)
            use_count = opt_count_95
            for i, (name, score) in enumerate(sorted_items):
                if i < use_count and score != float('inf'):
                    high_sensitivity_layers.add(name)
            
            valid_scores = [s for _, s in sorted_items if s != float('inf')]
            cumulative = np.cumsum(valid_scores)
            cumulative_normalized = cumulative / cumulative[-1] if cumulative[-1] > 0 else cumulative
            actual_coverage = cumulative_normalized[use_count - 1] if use_count > 0 else 0

            recommendation_info = {
                'method': 'auto_coverage_95',
                'skip_count': use_count,
                'coverage': actual_coverage,
            }

        quantizable = []
        not_quantizable = []

        for name, score in self.sensitivity_scores.items():
            if name in high_sensitivity_layers:
                not_quantizable.append(name)
            else:
                quantizable.append(name)

        return quantizable, not_quantizable, recommendation_info

    def plot_sensitivity(
        self,
        save_dir: Optional[str] = None,
        top_n_bar: int = 30,
        top_n_percent: float = 10.0,
    ):
        """
        绘制敏感度分析图（优化版：移除 Fig 2，加速绘图）
        """
        if not self.sensitivity_scores:
            return

        valid_data = [(name, score) for name, score in self.sensitivity_scores.items()
                      if score != float('inf')]
        valid_data.sort(key=lambda x: x[1], reverse=True)

        all_layers = [d[0] for d in valid_data]
        all_scores = [d[1] for d in valid_data]
        total_layers = len(valid_data)
        top_n_count = max(1, int(total_layers * top_n_percent / 100))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        saved_files = []

        # =====================================================
        # 图 1: 敏感度柱状图（只显示前 top_n_bar 个）
        # =====================================================
        bar_layers = all_layers[:top_n_bar]
        bar_scores = all_scores[:top_n_bar]
        high_sens_mask = [i < top_n_count for i in range(len(bar_layers))]

        fig1 = plt.figure(figsize=(12, min(10, top_n_bar * 0.4)))
        ax1 = fig1.add_subplot(111)
        y_pos = np.arange(len(bar_layers))
        colors = ['#ff6b6b' if m else '#4ecdc4' for m in high_sens_mask]
        bars = ax1.barh(y_pos, bar_scores, color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(bar_layers, fontsize=9)
        ax1.set_xlabel('Sensitivity Score', fontsize=12)
        ax1.set_title(f'Top {top_n_bar} Layer Sensitivity', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()

        for i, (bar, score) in enumerate(zip(bars, bar_scores)):
            width = bar.get_width()
            ax1.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'{score:.3f}', va='center', fontsize=8)

        plt.tight_layout()
        if save_dir:
            path1 = os.path.join(save_dir, '01_layer_sensitivity_top.jpg')
            plt.savefig(path1, dpi=150, bbox_inches='tight')
            saved_files.append(path1)
        plt.close(fig1)

        # =====================================================
        # 图 2: 已移除 (02_sensitivity_distribution.jpg) - 节省资源
        # =====================================================

        # =====================================================
        # 图 3: 累积敏感度曲线（关键图）
        # =====================================================
        fig3 = plt.figure(figsize=(12, 7))
        ax3 = fig3.add_subplot(111)
        sorted_scores = sorted(all_scores, reverse=True)
        cumulative = np.cumsum(sorted_scores)
        cumulative_normalized = cumulative / cumulative[-1]

        ax3.plot(range(1, len(cumulative_normalized) + 1), cumulative_normalized,
                 'o-', color='#9b59b6', linewidth=2.5, markersize=5)

        key_points = [5, 10, 15, 20, 30]
        for pct in key_points:
            idx = max(1, int(total_layers * pct / 100))
            if idx <= len(cumulative_normalized):
                val = cumulative_normalized[idx - 1]
                ax3.axvline(x=idx, color='#95a5a6', linestyle=':', linewidth=1, alpha=0.7)
                ax3.text(idx + 2, val + 0.02, f'{pct}%: {val:.1%}',
                         fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

        ax3.axvline(x=top_n_count, color='#ff6b6b', linestyle='--', linewidth=2,
                    label=f'Top {top_n_percent}%')

        ax3.set_xlabel('Number of Layers Skipped', fontsize=12)
        ax3.set_ylabel('Cumulative Sensitivity Covered', fontsize=12)
        ax3.set_title('Cumulative Sensitivity Curve', fontsize=14, fontweight='bold')
        ax3.grid(alpha=0.3)
        ax3.legend(fontsize=11, loc='lower right')

        plt.tight_layout()
        if save_dir:
            path3 = os.path.join(save_dir, '03_cumulative_sensitivity.jpg')
            plt.savefig(path3, dpi=150, bbox_inches='tight')
            saved_files.append(path3)
        plt.close(fig3)

        return saved_files

    def save_results(
        self,
        output_dir: str,
        top_n_percent: Optional[float] = None,
        top_n_bar: int = 30,
    ):
        """保存所有结果到指定目录"""
        os.makedirs(output_dir, exist_ok=True)

        use_top_n_percent = top_n_percent
        if use_top_n_percent is None and self.sensitivity_scores:
            try:
                opt_count, _, _ = self.find_optimal_skip_count(method='coverage', target_coverage=0.95)
                total = len([s for s in self.sensitivity_scores.values() if s != float('inf')])
                if total > 0:
                    use_top_n_percent = (opt_count / total) * 100
            except BaseException:
                use_top_n_percent = 10.0

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        report_path = os.path.join(output_dir, f'sensitivity_report_{timestamp}.txt')
        self.save_report(report_path, top_n_percent=use_top_n_percent if use_top_n_percent else 10.0)
        print(f"    报表已保存：{os.path.basename(report_path)}")

        try:
            saved_plots = self.plot_sensitivity(
                save_dir=output_dir,
                top_n_bar=top_n_bar,
                top_n_percent=use_top_n_percent if use_top_n_percent else 10.0
            )
            if saved_plots:
                for path in saved_plots:
                    print(f"    图表已保存：{os.path.basename(path)}")
        except Exception as e:
            print(f"    图表生成跳过：{str(e)[:80]}")
            
    def save_report(self, save_path: str, top_n_percent: float = 10.0):
        report = self.generate_report(top_n_percent=top_n_percent)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)

