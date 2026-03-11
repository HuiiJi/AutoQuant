"""
Op敏感度分析 - 分析每个操作对量化的敏感度
帮助决定哪些操作应该量化，哪些操作应该保持浮点精度
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
import copy
from collections import defaultdict
from tabulate import tabulate


class SensitivityAnalyzer:
    """
    操作敏感度分析器
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
        self.original_outputs: Dict[str, torch.Tensor] = {}
        self.quantized_outputs: Dict[str, torch.Tensor] = {}
        
    @staticmethod
    def _default_metric(original_output: torch.Tensor, quantized_output: torch.Tensor) -> float:
        """
        默认评估指标：计算输出的MSE
        """
        return torch.nn.functional.mse_loss(original_output, quantized_output).item()
    
    def analyze(
        self,
        dummy_input: torch.Tensor,
        skip_layers: Optional[List[str]] = None,
        only_layers: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        执行敏感度分析
        
        Args:
            dummy_input: 用于分析的输入
            skip_layers: 跳过分析的层列表
            only_layers: 只分析的层列表
        
        Returns:
            敏感度分数字典
        """
        skip_layers = skip_layers or []
        self.model.eval()
        
        # 首先获取原始模型的输出
        with torch.no_grad():
            original_output = self.model(dummy_input)
        
        # 收集所有可量化的层
        quantizable_layers = self._get_quantizable_layers()
        
        if only_layers:
            quantizable_layers = [name for name in quantizable_layers if name in only_layers]
        
        quantizable_layers = [name for name in quantizable_layers if name not in skip_layers]
        
        # 逐个分析每个层
        for layer_name in quantizable_layers:
            print(f"分析层: {layer_name}")
            score = self._analyze_single_layer(layer_name, dummy_input, original_output)
            self.sensitivity_scores[layer_name] = score
        
        return self.sensitivity_scores
    
    def _get_quantizable_layers(self) -> List[str]:
        """
        获取所有可量化的层名称
        """
        quantizable_types = (
            nn.Conv2d,
            nn.Linear,
            nn.Conv1d,
            nn.Conv3d,
        )
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, quantizable_types):
                layers.append(name)
        return layers
    
    def _analyze_single_layer(
        self,
        layer_name: str,
        dummy_input: torch.Tensor,
        original_output: torch.Tensor,
    ) -> float:
        """
        分析单个层的敏感度
        """
        # 复制模型，只量化当前层
        model_copy = copy.deepcopy(self.model)
        
        # 找到目标层
        target_module = None
        parent_module = None
        for name, module in model_copy.named_modules():
            if name == layer_name:
                target_module = module
                break
        
        if target_module is None:
            return float('inf')
        
        # 临时替换为量化版本（这里简化处理，实际应该使用prepare/convert）
        # 对于敏感度分析，我们可以使用一个简单的方法：
        # 计算量化该层前后的输出差异
        with torch.no_grad():
            # 获取原始层的输出
            class OutputHook:
                def __init__(self):
                    self.output = None
                def __call__(self, module, input, output):
                    self.output = output
            
            hook = OutputHook()
            handle = target_module.register_forward_hook(hook)
            model_copy(dummy_input)
            handle.remove()
            original_layer_output = hook.output
            
            # 量化权重
            quantized_weight = self._quantize_tensor(target_module.weight)
            
            # 临时替换权重
            original_weight = target_module.weight.data.clone()
            target_module.weight.data = quantized_weight
            
            # 推理
            quantized_output = model_copy(dummy_input)
            
            # 恢复原始权重
            target_module.weight.data = original_weight
        
        # 计算敏感度分数
        score = self.metric_fn(original_output, quantized_output)
        return score
    
    def _quantize_tensor(
        self,
        tensor: torch.Tensor,
        quant_min: int = -128,
        quant_max: int = 127,
    ) -> torch.Tensor:
        """
        简单的张量量化，用于敏感度分析
        """
        min_val = tensor.min()
        max_val = tensor.max()
        
        scale = (max_val - min_val) / (quant_max - quant_min)
        zero_point = quant_min - torch.round(min_val / scale)
        
        # 量化
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, quant_min, quant_max)
        
        # 反量化
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def generate_report(self, sort_by: str = 'score', ascending: bool = True) -> str:
        """
        生成敏感度分析报告
        
        Args:
            sort_by: 排序字段，'score' 或 'name'
            ascending: 是否升序
        
        Returns:
            格式化的报告字符串
        """
        if not self.sensitivity_scores:
            return "请先执行敏感度分析"
        
        # 准备数据
        data = []
        for name, score in self.sensitivity_scores.items():
            data.append({
                '层名称': name,
                '敏感度分数': score,
                '建议': '保持浮点' if score > 0.1 else '可以量化'
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
        total_layers = len(data)
        quantizable_layers = sum(1 for d in data if d['建议'] == '可以量化')
        
        summary = f"\n\n总结:\n"
        summary += f"总层数: {total_layers}\n"
        summary += f"建议量化的层数: {quantizable_layers}\n"
        summary += f"建议保持浮点的层数: {total_layers - quantizable_layers}\n"
        
        return table + summary
    
    def get_recommended_layers(self, threshold: float = 0.1) -> Tuple[List[str], List[str]]:
        """
        获取推荐的量化层和保持浮点的层
        
        Args:
            threshold: 敏感度阈值
        
        Returns:
            (推荐量化的层列表, 推荐保持浮点的层列表)
        """
        quantizable = []
        not_quantizable = []
        
        for name, score in self.sensitivity_scores.items():
            if score <= threshold:
                quantizable.append(name)
            else:
                not_quantizable.append(name)
        
        return quantizable, not_quantizable
    
    def plot_sensitivity(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
    ):
        """
        绘制敏感度分析图
        
        Args:
            figsize: 图大小
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("需要安装matplotlib: pip install matplotlib")
            return
        
        if not self.sensitivity_scores:
            print("请先执行敏感度分析")
            return
        
        # 准备数据
        layers = list(self.sensitivity_scores.keys())
        scores = list(self.sensitivity_scores.values())
        
        # 排序
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
        layers = [layers[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # 绘制
        plt.figure(figsize=figsize)
        plt.barh(range(len(layers)), scores)
        plt.yticks(range(len(layers)), layers)
        plt.xlabel('敏感度分数 (MSE)')
        plt.ylabel('层名称')
        plt.title('各层敏感度分析')
        plt.axvline(x=0.1, color='r', linestyle='--', label='建议阈值 (0.1)')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图已保存到: {save_path}")
        else:
            plt.show()
