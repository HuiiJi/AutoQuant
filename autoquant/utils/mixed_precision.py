"""
混合精度量化 - 支持不同层使用不同的量化精度
增强版：支持自动bit-width搜索、基于敏感度的配置、Transformer专用策略
"""
import torch
import torch.nn as nn
import copy
from typing import Dict, Optional, Set, Callable, List, Tuple
from autoquant.utils import QConfig, get_default_qconfig, get_lsq_qconfig
from autoquant.core import QuantDtype, QScheme
from autoquant.quantization import prepare, convert
from autoquant.utils.sensitivity_analysis import SensitivityAnalyzer


class MixedPrecisionQuantizer:
    """
    增强版混合精度量化器
    
    功能：
    - 为不同层设置不同的量化配置
    - 基于敏感度自动配置
    - 支持bit-width搜索
    - 支持Transformer专用策略
    """

    def __init__(
        self,
        model: nn.Module,
        default_qconfig: Optional[QConfig] = None
    ):
        self.model = model
        self.default_qconfig = default_qconfig or get_default_qconfig()
        self.layer_qconfigs: Dict[str, Optional[QConfig]] = {}
        self.layer_precisions: Dict[str, QuantDtype] = {}
        self.sensitivity_scores: Optional[Dict[str, float]] = None
        
    def set_layer_config(
        self,
        layer_name: str,
        qconfig: Optional[QConfig] = None,
        keep_fp: bool = False,
    ):
        """
        为特定层设置量化配置
        
        Args:
            layer_name: 层的名称
            qconfig: 量化配置（None表示不量化）
            keep_fp: 保持浮点精度（等价于qconfig=None）
        """
        if keep_fp:
            self.layer_qconfigs[layer_name] = None
        else:
            self.layer_qconfigs[layer_name] = qconfig
    
    def set_fp_layers(self, layer_names: List[str]):
        """
        批量设置保持浮点的层
        
        Args:
            layer_names: 层名称列表
        """
        for name in layer_names:
            self.set_layer_config(name, keep_fp=True)
    
    def auto_config_by_sensitivity(
        self,
        dummy_input: torch.Tensor,
        threshold: float = 0.001,
        qconfig_high_sensitivity: Optional[QConfig] = None,
        qconfig_low_sensitivity: Optional[QConfig] = None,
    ):
        """
        基于敏感度自动配置混合精度
        
        Args:
            dummy_input: 输入张量
            threshold: 敏感度阈值
            qconfig_high_sensitivity: 高敏感层的配置（默认LSQ QAT）
            qconfig_low_sensitivity: 低敏感层的配置（默认PTQ）
        """
        print("🔍 运行敏感度分析...")
        analyzer = SensitivityAnalyzer(self.model, self.default_qconfig)
        self.sensitivity_scores = analyzer.analyze(dummy_input)
        
        # 默认配置
        qconfig_high = qconfig_high_sensitivity or get_lsq_qconfig()
        qconfig_low = qconfig_low_sensitivity or self.default_qconfig
        
        num_high_sensitivity = 0
        num_low_sensitivity = 0
        
        for layer_name, score in self.sensitivity_scores.items():
            if score > threshold:
                # 高敏感度：使用更高级的量化方法或保持浮点
                self.set_layer_config(layer_name, qconfig=qconfig_high)
                num_high_sensitivity += 1
            else:
                # 低敏感度：使用标准量化
                self.set_layer_config(layer_name, qconfig=qconfig_low)
                num_low_sensitivity += 1
        
        print(f"✓ 自动配置完成:")
        print(f"  - 高敏感度层: {num_high_sensitivity}")
        print(f"  - 低敏感度层: {num_low_sensitivity}")
        print(f"  - 阈值: {threshold}")
        
        return analyzer
    
    def search_bit_width(
        self,
        dummy_input: torch.Tensor,
        bit_widths: List[int] = [8, 16],
        objective: str = 'balance',  # 'accuracy', 'speed', 'balance'
    ) -> Dict[str, int]:
        """
        自动搜索最优bit-width分配
        
        Args:
            dummy_input: 输入张量
            bit_widths: 候选bit-width列表
            objective: 优化目标
            
        Returns:
            每层的最优bit-width
        """
        print(f"🔍 搜索最优bit-width分配，目标: {objective}")
        
        if self.sensitivity_scores is None:
            self.auto_config_by_sensitivity(dummy_input)
        
        bit_allocation = {}
        
        for layer_name, score in self.sensitivity_scores.items():
            if objective == 'accuracy':
                # 优先精度：高敏感层用高bit
                bit = 16 if score > 0.01 else 8
            elif objective == 'speed':
                # 优先速度：尽量用低bit
                bit = 8
            else:  # balance
                # 平衡：根据敏感度分配
                if score > 0.01:
                    bit = 16
                elif score > 0.001:
                    bit = 8 if 8 in bit_widths else 16
                else:
                    bit = min(bit_widths)
            
            bit_allocation[layer_name] = bit
        
        print(f"✓ Bit-width搜索完成")
        return bit_allocation
    
    def prepare(
        self,
        inplace: bool = False,
    ) -> nn.Module:
        """
        准备混合精度量化模型
        
        Args:
            inplace: 是否原地修改
        
        Returns:
            准备好的模型
        """
        if not inplace:
            model = copy.deepcopy(self.model)
        else:
            model = self.model
        
        # 为每个层应用配置
        # 注意：这里需要ModelQuantizer支持层级配置
        # 当前使用简化实现：先全部用默认配置，然后标记特殊层
        from autoquant.quantization import prepare as base_prepare
        
        # 首先用默认配置准备
        prepared_model = base_prepare(model, self.default_qconfig, inplace=True)
        
        # 然后处理需要保持浮点的层
        for layer_name, qconfig in self.layer_qconfigs.items():
            if qconfig is None:
                # 移除这个层的量化
                try:
                    module = prepared_model
                    for part in layer_name.split('.'):
                        module = getattr(module, part)
                    if hasattr(module, 'disable_fake_quant'):
                        module.disable_fake_quant()
                except Exception as e:
                    print(f"⚠ 无法设置层 {layer_name} 为浮点: {e}")
        
        return prepared_model
    
    def get_config_summary(self) -> str:
        """
        获取配置摘要
        
        Returns:
            配置摘要字符串
        """
        summary = []
        summary.append("=" * 60)
        summary.append("混合精度配置摘要")
        summary.append("=" * 60)
        
        if not self.layer_qconfigs:
            summary.append("  (未配置特殊层，全部使用默认配置)")
        else:
            fp_layers = [name for name, qc in self.layer_qconfigs.items() if qc is None]
            quant_layers = [name for name, qc in self.layer_qconfigs.items() if qc is not None]
            
            if fp_layers:
                summary.append(f"\n保持浮点的层 ({len(fp_layers)}):")
                for name in sorted(fp_layers)[:10]:
                    summary.append(f"  - {name}")
                if len(fp_layers) > 10:
                    summary.append(f"  ... 还有 {len(fp_layers) - 10} 层")
            
            if quant_layers:
                summary.append(f"\n自定义配置的层 ({len(quant_layers)}):")
                for name in sorted(quant_layers)[:10]:
                    summary.append(f"  - {name}")
                if len(quant_layers) > 10:
                    summary.append(f"  ... 还有 {len(quant_layers) - 10} 层")
        
        summary.append("\n" + "=" * 60)
        return "\n".join(summary)


class LayerSelector:
    """
    增强版层选择器 - 用于选择需要量化的层
    
    功能：
    - 按类型选择
    - 按名称模式选择
    - 按大小/参数量选择
    - Transformer专用层选择
    """

    @staticmethod
    def get_leaf_modules(model: nn.Module, prefix: str = "") -> Set[str]:
        """获取模型的所有叶子模块名称"""
        leaf_modules = set()
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if len(list(module.children())) == 0:
                leaf_modules.add(full_name)
            else:
                leaf_modules.update(LayerSelector.get_leaf_modules(module, full_name))
        return leaf_modules

    @staticmethod
    def get_modules_by_type(
        model: nn.Module,
        module_types: tuple,
        prefix: str = "",
    ) -> Set[str]:
        """获取指定类型的模块名称"""
        modules = set()
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(module, module_types):
                modules.add(full_name)
            modules.update(LayerSelector.get_modules_by_type(module, module_types, full_name))
        return modules
    
    @staticmethod
    def get_conv_layers(model: nn.Module) -> Set[str]:
        """获取所有卷积层"""
        return LayerSelector.get_modules_by_type(model, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
    
    @staticmethod
    def get_linear_layers(model: nn.Module) -> Set[str]:
        """获取所有线性层"""
        return LayerSelector.get_modules_by_type(model, (nn.Linear,))
    
    @staticmethod
    def get_transformer_layers(model: nn.Module) -> Set[str]:
        """
        获取Transformer相关层（Attention, FFN等）
        基于名称模式匹配
        """
        transformer_layers = set()
        patterns = ['attention', 'attn', 'q_proj', 'k_proj', 'v_proj', 'out_proj', 'ffn', 'feed_forward']
        
        for name, _ in model.named_modules():
            if any(pattern in name.lower() for pattern in patterns):
                transformer_layers.add(name)
        
        return transformer_layers
    
    @staticmethod
    def get_large_layers(
        model: nn.Module,
        min_params: int = 1000000,  # 1M参数
    ) -> Set[str]:
        """获取参数量大的层"""
        large_layers = set()
        
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue  # 跳过非叶子模块
            
            num_params = sum(p.numel() for p in module.parameters())
            if num_params >= min_params:
                large_layers.add(name)
        
        return large_layers
    
    @staticmethod
    def get_output_layers(model: nn.Module) -> Set[str]:
        """
        获取输出层（通常是最后几层）
        基于名称模式匹配
        """
        output_layers = set()
        patterns = ['fc', 'head', 'out', 'final', 'logits', 'classifier']
        
        for name, _ in model.named_modules():
            if any(pattern in name.lower() for pattern in patterns):
                output_layers.add(name)
        
        return output_layers

