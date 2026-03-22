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
      优化：全OP分析 + tqdm进度条 + 静默打印 + 单独JPG图表 + 关键分析
"""
import os
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，确保在无GUI环境也能保存
except BaseException:
    pass

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import copy
from collections import defaultdict
from tabulate import tabulate
from datetime import datetime
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


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
        self.original_outputs = None  # 原始模型在所有calib_data上的输出列表
        self.calib_data = None
        self.dummy_input = None

    @staticmethod
    def _default_metric(original_outputs, quantized_outputs) -> float:
        """
        默认评估指标：计算一批样本的平均MSE
        支持 tuple 输出（取第一个元素）

        Args:
            original_outputs: 原始模型的输出列表 [output1, output2, ...]
            quantized_outputs: 量化模型的输出列表 [output1, output2, ...]

        Returns:
            平均 MSE
        """
        def get_first_tensor(x):
            if isinstance(x, tuple):
                return x[1] if len(x) > 0 else None
            return x

        total_mse = 0.0
        count = 0

        for orig_out, quant_out in zip(original_outputs, quantized_outputs):
            orig = get_first_tensor(orig_out)
            quant = get_first_tensor(quant_out)

            if orig is not None and quant is not None:
                mse = torch.nn.functional.mse_loss(orig, quant).item()
                total_mse += mse
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
        max_calib_samples: int = 10,
    ) -> Dict[str, float]:
        """
        执行敏感度分析 - 正确的逻辑！

        正确逻辑：
        1. 基准1: 原始模型（全浮点）- 在所有calib_data样本上推理
        2. 基准2: 全部量化 - 最差情况
        3. 对每个层：只跳过这一层，其他都量化
        4. 敏感度分数 = (full_quant_score - skip_this_layer_score) / full_quant_score
           - 分数越高，表示跳过这一层带来的改善越大 → 越敏感，越应该跳过

        Args:
            dummy_input: 用于分析的输入（仅用于形状检查）
            calib_data: 校准数据（必须！用于计算平均MSE）
            skip_layers: 跳过分析的层列表
            only_layers: 只分析的层列表
            max_calib_samples: 敏感度分析使用的最大校准样本数（默认5，加速分析）

        Returns:
            敏感度分数字典
        """
        from autoquant import ModelQuantizer

        skip_layers = skip_layers or []

        # 优化：限制用于敏感度分析的校准样本数，加速分析
        if calib_data is not None and len(calib_data) > max_calib_samples:
            print(f"    优化：使用 {max_calib_samples}/{len(calib_data)} 个校准样本进行敏感度分析")
            calib_data_analysis = calib_data[:max_calib_samples]
        else:
            calib_data_analysis = calib_data if calib_data is not None else [dummy_input]

        self.calib_data = calib_data  # 保存完整的校准数据
        self.dummy_input = dummy_input
        self.model.eval()

        # ====================================================================
        # 步骤 1: 获取原始模型在所有calib_data样本上的输出
        # ====================================================================
        self.original_outputs = []
        with torch.no_grad():
            for inp in calib_data_analysis:
                out = self.model(inp)
                self.original_outputs.append(out)

        self.original_score = 0.0  # 和自己比当然是0

        # ====================================================================
        # 步骤 2: 获取全部量化的模型分数（最差情况）
        # ====================================================================
        self.full_quant_score = self._analyze_full_quant(calib_data_analysis)

        if self.full_quant_score == float('inf') or self.full_quant_score <= 0:
            return {}

        # ====================================================================
        # 步骤 3: 收集所有可量化的层
        # ====================================================================
        quantizable_layers = self._get_quantizable_layers()

        if only_layers:
            quantizable_layers = [name for name in quantizable_layers if name in only_layers]

        quantizable_layers = [name for name in quantizable_layers if name not in skip_layers]

        print(f"    分析 {len(quantizable_layers)} 个层 (使用 {len(calib_data_analysis)} 个校准样本)...")

        # ====================================================================
        # 步骤 4: 逐个分析每个层 - 只跳过这一层，其他都量化
        # ====================================================================
        if TQDM_AVAILABLE:
            pbar = tqdm(
                total=len(quantizable_layers),
                desc="    敏感度分析",
                unit="层"
            )
            for i, layer_name in enumerate(quantizable_layers):
                current = i + 1
                # 更新进度条描述
                pbar.set_description(f"    敏感度分析 - {layer_name}")

                score = self._analyze_skip_single_layer(
                    layer_name, calib_data_analysis
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
        else:
            # 降级到原始进度条
            for i, layer_name in enumerate(quantizable_layers):
                current = i + 1
                total = len(quantizable_layers)
                percentage = (current / total) * 100
                bar_length = 50
                filled_length = int(bar_length * current / total)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)

                print(f"\r    [{bar}] {current}/{total} ({percentage:.1f}%) - {layer_name}", end='', flush=True)

                score = self._analyze_skip_single_layer(
                    layer_name, calib_data_analysis
                )

                if score != float('inf') and self.full_quant_score > 0:
                    improvement = self.full_quant_score - score
                    sensitivity_score = improvement / self.full_quant_score
                    sensitivity_score = max(0.0, min(1.0, sensitivity_score))
                else:
                    sensitivity_score = 0.0

                self.sensitivity_scores[layer_name] = sensitivity_score

            print(f"\r    [{'█' * 50}] {len(quantizable_layers)}/{len(quantizable_layers)} (100.0%) - 完成")

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
        calib_data: List,
    ) -> float:
        """
        分析全部量化的模型（跳过0层）- 在所有calib_data样本上推理
        """
        from autoquant import ModelQuantizer

        try:
            # 优化：使用更小的 dummy input 进行 prepare
            dummy_input_small = torch.randn(1, 3, 64, 64).to(next(self.model.parameters()).device)

            # 优化：模型 deepcopy 还是需要的，但我们用小 dummy input
            model_copy = copy.deepcopy(self.model)
            quantizer = ModelQuantizer(model_copy, self.qconfig)

            prepared_model = quantizer.prepare(skip_layers=set())
            quantizer.calibrate(calib_data, verbose=False)
            quantized_model = quantizer.convert()

            quantized_outputs = []
            with torch.no_grad():
                for inp in calib_data:
                    out = quantized_model(inp)
                    quantized_outputs.append(out)

            score = self.metric_fn(self.original_outputs, quantized_outputs)
            return score

        except Exception as e:
            return float('inf')

    def _analyze_skip_single_layer(
        self,
        layer_name: str,
        calib_data: List,
    ) -> float:
        """
        分析只跳过当前层的情况（其他层都量化）- 在所有calib_data样本上推理

        这才是正确的敏感度分析！
        - 只跳过这一层不量化
        - 其他所有层都正常量化
        - 这样能准确衡量：如果不量化这一层，能挽回多少误差
        """
        from autoquant import ModelQuantizer

        try:
            # 优化：模型 deepcopy 还是需要的，但我们用小 dummy input
            model_copy = copy.deepcopy(self.model)
            quantizer = ModelQuantizer(model_copy, self.qconfig)

            prepared_model = quantizer.prepare(skip_layers={layer_name})
            quantizer.calibrate(calib_data, verbose=False)
            quantized_model = quantizer.convert()

            quantized_outputs = []
            with torch.no_grad():
                for inp in calib_data:
                    out = quantized_model(inp)
                    quantized_outputs.append(out)

            score = self.metric_fn(self.original_outputs, quantized_outputs)
            return score

        except Exception as e:
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
                '挽回比例': f'{score * 100:.1f}%',
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

        # 计算统计量
        all_scores = [s for s in self.sensitivity_scores.values() if s != float('inf')]
        mean_score = np.mean(all_scores)
        median_score = np.median(all_scores)

        # 计算累积敏感度覆盖情况
        sorted_scores = sorted(all_scores, reverse=True)
        cumulative = np.cumsum(sorted_scores)
        cumulative_normalized = cumulative / cumulative[-1]

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
        summary += f"  敏感度分数统计:\n"
        summary += f"    - 均值 (Mean): {mean_score:.6f}\n"
        summary += f"    - 中位数 (Median): {median_score:.6f}\n"
        summary += f"\n"
        summary += f"  【关键分析 1】模型量化可行性 (查看 02_sensitivity_distribution.jpg):\n"
        summary += f"    - 如果分数大多接近 0 → 模型容易量化，几乎所有层都可以量化\n"
        summary += f"    - 如果分数呈现梯度分布 → 模型难量化，需要仔细选择跳过的层\n"
        summary += f"    - 当前: 均值={mean_score:.4f}, 中位数={median_score:.4f}\n"
        summary += f"\n"
        summary += f"  【关键分析 2】最佳 Skip 比例 (查看 03_cumulative_sensitivity.jpg):\n"
        summary += f"    查找曲线的拐点 (elbow point)，即继续增加层收益递减的点\n"
        summary += f"    - Skip  5%: 覆盖 {cumulative_normalized[max(0, int(total_layers * 0.05) - 1)]:.1%} 敏感度\n"
        summary += f"    - Skip 10%: 覆盖 {cumulative_normalized[max(0, int(total_layers * 0.10) - 1)]:.1%} 敏感度\n"
        summary += f"    - Skip 15%: 覆盖 {cumulative_normalized[max(0, int(total_layers * 0.15) - 1)]:.1%} 敏感度\n"
        summary += f"    - Skip 20%: 覆盖 {cumulative_normalized[max(0, int(total_layers * 0.20) - 1)]:.1%} 敏感度\n"
        summary += f"    - Skip 30%: 覆盖 {cumulative_normalized[max(0, int(total_layers * 0.30) - 1)]:.1%} 敏感度\n"
        summary += f"\n"

        # 自动推荐
        summary += f"  【自动推荐】(无需手动设定 Top N%！):\n"
        try:
            opt_count_elbow, coverage_elbow, _ = self.find_optimal_skip_count(method='elbow')
            opt_count_90, coverage_90, _ = self.find_optimal_skip_count(method='coverage', target_coverage=0.90)
            opt_count_95, coverage_95, _ = self.find_optimal_skip_count(method='coverage', target_coverage=0.95)

            summary += f"    方案1 - 拐点法 (Elbow): Skip {opt_count_elbow} 层 ({
                opt_count_elbow /
                total_layers *
                100:.1f}%)，覆盖 {
                coverage_elbow:.1%} 敏感度\n"
            summary += f"    方案2 - 覆盖90%: Skip {opt_count_90} 层 ({
                opt_count_90 /
                total_layers *
                100:.1f}%)，覆盖 {
                coverage_90:.1%} 敏感度\n"
            summary += f"    方案3 - 覆盖95%: Skip {opt_count_95} 层 ({
                opt_count_95 /
                total_layers *
                100:.1f}%)，覆盖 {
                coverage_95:.1%} 敏感度  ⭐ (默认)\n"
            summary += f"\n"
            summary += f"    推荐：优先使用 覆盖95%，在保证精度的前提下获得较好压缩率！\n"
        except Exception as e:
            summary += f"    (自动推荐计算失败: {str(e)})\n"

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
        """
        自动寻找最佳的 skip 层数

        Args:
            method: 方法选择 - 'elbow' (拐点) 或 'coverage' (覆盖目标敏感度)
            target_coverage: 目标覆盖比例 (仅 coverage 方法使用，默认 0.90 = 90%)

        Returns:
            (最佳 skip 层数, 覆盖的敏感度比例, 方法描述)
        """
        if not self.sensitivity_scores:
            return 0, 0.0, "No sensitivity scores available"

        valid_data = [(name, score) for name, score in self.sensitivity_scores.items()
                      if score != float('inf')]
        valid_data.sort(key=lambda x: x[1], reverse=True)
        scores = [d[1] for d in valid_data]
        total_layers = len(scores)

        if total_layers == 0:
            return 0, 0.0, "No valid layers"

        # 计算累积敏感度
        cumulative = np.cumsum(scores)
        cumulative_normalized = cumulative / cumulative[-1] if cumulative[-1] > 0 else cumulative

        if method == 'elbow':
            # 方案1: 找拐点 (elbow point) - 二阶导数最大的点
            # 计算一阶导数
            if total_layers < 3:
                # 层数太少，直接返回一个合理值
                opt_count = max(1, total_layers // 10)
                coverage = cumulative_normalized[opt_count - 1] if opt_count > 0 else 0
                return opt_count, coverage, "Elbow method (fallback, too few layers)"

            first_deriv = np.diff(cumulative_normalized)
            # 计算二阶导数
            second_deriv = np.diff(first_deriv)
            # 找二阶导数最大的点（最拐点）
            elbow_idx = np.argmax(np.abs(second_deriv)) + 1  # +1 因为二阶导数少两个点
            # 确保在合理范围内
            opt_count = max(1, min(elbow_idx, total_layers // 2))
            coverage = cumulative_normalized[opt_count - 1]
            desc = f"Elbow method (拐点 at {opt_count} layers)"

        elif method == 'coverage':
            # 方案2: 覆盖目标敏感度比例
            # 找第一个 >= target_coverage 的点
            opt_count = 0
            for i in range(total_layers):
                if cumulative_normalized[i] >= target_coverage:
                    opt_count = i + 1
                    break
            if opt_count == 0:
                opt_count = total_layers  # 没找到，全部跳过
            coverage = cumulative_normalized[opt_count - 1] if opt_count > 0 else 0
            desc = f"Coverage method (target {target_coverage * 100:.0f}%)"

        else:
            raise ValueError(f"Unknown method: {method}, use 'elbow' or 'coverage'")

        return opt_count, coverage, desc

    def get_recommended_layers(
        self,
        threshold: Optional[float] = None,
        top_n_percent: Optional[float] = None,
        auto_method: Optional[str] = 'elbow',
        auto_target_coverage: float = 0.90,
    ) -> Tuple[List[str], List[str], Dict]:
        """
        获取推荐的量化层和保持浮点的层

        优先级：threshold > top_n_percent > auto_method

        Args:
            threshold: 敏感度阈值（可选）
            top_n_percent: 前 N% 视为高敏感度层（可选）
            auto_method: 自动方法 - 'elbow' (拐点) 或 'coverage' (覆盖目标)
            auto_target_coverage: coverage 方法的目标覆盖比例 (默认 0.90)

        Returns:
            (推荐量化的层列表, 推荐保持浮点的层列表, 推荐信息字典)
        """
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
            # 模式1：用户指定阈值
            for name, score in sorted_items:
                if score > threshold and score != float('inf'):
                    high_sensitivity_layers.add(name)
            recommendation_info = {
                'method': 'user_threshold',
                'threshold': threshold,
                'skip_count': len(high_sensitivity_layers),
                'skip_percent': len(high_sensitivity_layers) / total_layers * 100,
            }
        elif top_n_percent is not None:
            # 模式2：用户指定百分比
            top_n_count = max(1, int(total_layers * top_n_percent / 100))
            for i, (name, score) in enumerate(sorted_items):
                if i < top_n_count and score != float('inf'):
                    high_sensitivity_layers.add(name)
            recommendation_info = {
                'method': 'user_percent',
                'top_n_percent': top_n_percent,
                'skip_count': len(high_sensitivity_layers),
            }
        else:
            # 模式3：自动推荐（默认）
            opt_count_elbow, coverage_elbow, desc_elbow = self.find_optimal_skip_count(method='elbow')
            opt_count_90, coverage_90, desc_90 = self.find_optimal_skip_count(method='coverage', target_coverage=0.90)
            opt_count_95, coverage_95, desc_95 = self.find_optimal_skip_count(method='coverage', target_coverage=0.95)

            # 默认使用 coverage 95% 方法
            use_count = opt_count_95
            use_desc = desc_95

            for i, (name, score) in enumerate(sorted_items):
                if i < use_count and score != float('inf'):
                    high_sensitivity_layers.add(name)

            # 计算实际覆盖的敏感度
            valid_scores = [s for _, s in sorted_items if s != float('inf')]
            cumulative = np.cumsum(valid_scores)
            cumulative_normalized = cumulative / cumulative[-1] if cumulative[-1] > 0 else cumulative
            actual_coverage = cumulative_normalized[use_count - 1] if use_count > 0 else 0

            recommendation_info = {
                'method': 'auto_elbow',
                'description': use_desc,
                'skip_count': use_count,
                'skip_percent': use_count / total_layers * 100,
                'coverage': actual_coverage,
                'alternatives': {
                    'elbow': {'count': opt_count_elbow, 'coverage': coverage_elbow, 'desc': desc_elbow},
                    'coverage_90': {'count': opt_count_90, 'coverage': coverage_90, 'desc': desc_90},
                    'coverage_95': {'count': opt_count_95, 'coverage': coverage_95, 'desc': desc_95},
                }
            }

        quantizable = []
        not_quantizable = []

        for name, score in self.sensitivity_scores.items():
            if name in high_sensitivity_layers:
                not_quantizable.append(name)
            else:
                quantizable.append(name)

        return quantizable, not_quantizable, recommendation_info

    def _setup_chinese_font(self):
        """配置 matplotlib 中文字体，解决乱码问题 - 增强版（静默警告）"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            import platform
            import os
            import warnings

            # 静默字体警告
            warnings.filterwarnings("ignore", category=UserWarning)

            system_name = platform.system()

            # 构建字体搜索列表，按优先级排序
            font_candidates = []

            if system_name == "Windows":
                # Windows 系统常见中文字体路径
                win_font_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
                font_candidates.extend([
                    ("SimHei", os.path.join(win_font_dir, "simhei.ttf")),
                    ("Microsoft YaHei", os.path.join(win_font_dir, "msyh.ttc")),
                    ("Microsoft YaHei", os.path.join(win_font_dir, "msyh.ttf")),
                    ("SimSun", os.path.join(win_font_dir, "simsun.ttc")),
                    ("Arial Unicode MS", os.path.join(win_font_dir, "arialuni.ttf")),
                ])
                font_candidates.extend([
                    ("SimHei", None),
                    ("Microsoft YaHei", None),
                    ("SimSun", None),
                    ("Arial Unicode MS", None),
                ])
            elif system_name == "Darwin":  # macOS
                mac_font_dirs = [
                    "/System/Library/Fonts",
                    "/Library/Fonts",
                    os.path.expanduser("~/Library/Fonts")
                ]
                for font_dir in mac_font_dirs:
                    font_candidates.extend([
                        ("PingFang SC", os.path.join(font_dir, "PingFang.ttc")),
                        ("Heiti TC", os.path.join(font_dir, "STHeiti Light.ttc")),
                        ("Arial Unicode MS", os.path.join(font_dir, "Arial Unicode.ttf")),
                    ])
                font_candidates.extend([
                    ("PingFang SC", None),
                    ("Heiti TC", None),
                    ("Arial Unicode MS", None),
                ])
            else:  # Linux
                linux_font_dirs = [
                    "/usr/share/fonts",
                    "/usr/local/share/fonts",
                    os.path.expanduser("~/.fonts"),
                ]
                for font_dir in linux_font_dirs:
                    font_candidates.extend([
                        ("WenQuanYi Micro Hei", os.path.join(font_dir, "wqy-microhei/WenQuanYiMicroHei.ttf")),
                        ("Noto Sans CJK SC", os.path.join(font_dir, "truetype/noto/NotoSansCJK-Regular.ttc")),
                    ])
                font_candidates.extend([
                    ("WenQuanYi Micro Hei", None),
                    ("Noto Sans CJK SC", None),
                    ("Arial Unicode MS", None),
                ])

            # 添加通用英文字体作为后备
            font_candidates.extend([
                ("DejaVu Sans", None),
                ("Arial", None),
                ("sans-serif", None),
            ])

            font_found = False

            # 逐个尝试字体
            for font_name, font_path in font_candidates:
                try:
                    if font_path and os.path.exists(font_path):
                        # 直接指定字体文件
                        plt.rcParams["font.family"] = "sans-serif"
                        plt.rcParams["font.sans-serif"] = [font_name]
                        plt.rcParams["axes.unicode_minus"] = False
                        font_found = True
                        break
                    elif font_name:
                        # 按名称查找
                        plt.rcParams["font.family"] = "sans-serif"
                        plt.rcParams["font.sans-serif"] = [font_name]
                        plt.rcParams["axes.unicode_minus"] = False
                        # 测试一下是否能正常显示
                        test_fonts = [f.name for f in fm.fontManager.ttflist]
                        if font_name in test_fonts or font_name.lower() in [f.lower() for f in test_fonts]:
                            font_found = True
                            break
                except Exception:
                    continue

            # 如果都找不到，使用默认字体
            if not font_found:
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["axes.unicode_minus"] = False

        except Exception as e:
            try:
                import matplotlib.pyplot as plt
                plt.rcParams["axes.unicode_minus"] = False
            except BaseException:
                pass

    def plot_sensitivity(
        self,
        save_dir: Optional[str] = None,
        top_n_bar: int = 30,
        top_n_percent: float = 10.0,
    ):
        """
        绘制敏感度分析图（每个子图单独保存为 JPG）

        Args:
            save_dir: 保存目录
            top_n_bar: 柱状图显示前 N 个层（避免太多层字体太小）
            top_n_percent: 前 N% 视为高敏感度层
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("需要安装 matplotlib 和 numpy: pip install matplotlib numpy")
            return

        self._setup_chinese_font()

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
        # 图 1: 敏感度柱状图（只显示前 top_n_bar 个，避免拥挤）
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
        ax1.set_ylabel('Layer Name', fontsize=12)
        ax1.set_title(f'Top {top_n_bar} Layer Sensitivity (Red=High Sensitivity)',
                      fontsize=14, fontweight='bold')
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
        else:
            plt.show()
        plt.close(fig1)

        # =====================================================
        # 图 2: 敏感度分数分布直方图
        # =====================================================
        fig2 = plt.figure(figsize=(10, 7))
        ax2 = fig2.add_subplot(111)
        n, bins, patches = ax2.hist(all_scores, bins=30, color='#4ecdc4',
                                    alpha=0.7, edgecolor='black', linewidth=1.2)
        ax2.set_xlabel('Sensitivity Score', fontsize=12)
        ax2.set_ylabel('Number of Layers', fontsize=12)
        ax2.set_title('Sensitivity Score Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        mean_score = np.mean(all_scores)
        median_score = np.median(all_scores)
        ax2.axvline(mean_score, color='#e74c3c', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_score:.3f}')
        ax2.axvline(median_score, color='#2ecc71', linestyle=':', linewidth=2,
                    label=f'Median: {median_score:.3f}')
        ax2.legend(fontsize=11)

        note_text = "Note: If scores are spread evenly (gradient), model is hard to quantize.\n"
        note_text += "If most scores are near 0, model is easy to quantize."
        fig2.text(0.5, 0.01, note_text, ha='center', fontsize=10, style='italic',
                  bbox=dict(facecolor='yellow', alpha=0.3, pad=5))

        plt.tight_layout()
        if save_dir:
            path2 = os.path.join(save_dir, '02_sensitivity_distribution.jpg')
            plt.savefig(path2, dpi=150, bbox_inches='tight')
            saved_files.append(path2)
        else:
            plt.show()
        plt.close(fig2)

        # =====================================================
        # 图 3: 累积敏感度曲线（关键图，帮助决定 skip 比例）
        # =====================================================
        fig3 = plt.figure(figsize=(12, 7))
        ax3 = fig3.add_subplot(111)
        sorted_scores = sorted(all_scores, reverse=True)
        cumulative = np.cumsum(sorted_scores)
        cumulative_normalized = cumulative / cumulative[-1]

        ax3.plot(range(1, len(cumulative_normalized) + 1), cumulative_normalized,
                 'o-', color='#9b59b6', linewidth=2.5, markersize=5)

        # 标注几个关键点
        key_points = [5, 10, 15, 20, 30, 50]
        for pct in key_points:
            idx = max(1, int(total_layers * pct / 100))
            if idx <= len(cumulative_normalized):
                val = cumulative_normalized[idx - 1]
                ax3.axvline(x=idx, color='#95a5a6', linestyle=':', linewidth=1, alpha=0.7)
                ax3.text(idx + 2, val + 0.02, f'{pct}%: {val:.1%}',
                         fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

        ax3.axvline(x=top_n_count, color='#ff6b6b', linestyle='--', linewidth=2,
                    label=f'Current: Top {top_n_percent}% (covers {cumulative_normalized[top_n_count - 1]:.1%} sensitivity)')

        ax3.set_xlabel('Number of Layers Skipped (Sorted by Sensitivity)', fontsize=12)
        ax3.set_ylabel('Cumulative Sensitivity Covered', fontsize=12)
        ax3.set_title('Cumulative Sensitivity Curve (Determine Skip Ratio)',
                      fontsize=14, fontweight='bold')
        ax3.grid(alpha=0.3)
        ax3.legend(fontsize=11, loc='lower right')

        note_text = "Key: This curve shows how much total sensitivity you cover by skipping top N layers.\n"
        note_text += "Find the elbow point where adding more layers gives diminishing returns."
        fig3.text(0.5, 0.01, note_text, ha='center', fontsize=10, style='italic',
                  bbox=dict(facecolor='lightgreen', alpha=0.3, pad=5))

        plt.tight_layout()
        if save_dir:
            path3 = os.path.join(save_dir, '03_cumulative_sensitivity.jpg')
            plt.savefig(path3, dpi=150, bbox_inches='tight')
            saved_files.append(path3)
        else:
            plt.show()
        plt.close(fig3)

        return saved_files

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

    def save_results(
        self,
        output_dir: str,
        top_n_percent: Optional[float] = None,
        top_n_bar: int = 30,
    ):
        """
        保存所有结果到指定目录（报告 + 单独的 JPG 图表）

        Args:
            output_dir: 输出目录
            top_n_percent: 前 N% 视为高敏感度层（None=自动推荐）
            top_n_bar: 柱状图显示前 N 个层
        """
        os.makedirs(output_dir, exist_ok=True)

        # 如果没指定，先算自动推荐的 skip 百分比 (默认 coverage 95%)
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
        print(f"    报表已保存: {os.path.basename(report_path)}")

        try:
            saved_plots = self.plot_sensitivity(
                save_dir=output_dir,
                top_n_bar=top_n_bar,
                top_n_percent=use_top_n_percent if use_top_n_percent else 10.0
            )
            if saved_plots:
                for path in saved_plots:
                    print(f"    图表已保存: {os.path.basename(path)}")
        except Exception as e:
            print(f"    图表生成跳过: {str(e)[:80]}")
