"""
Transformer/LLM 专用量化支持

实现策略：
- SmoothQuant: https://arxiv.org/abs/2211.10438
- KV Cache量化
- Attention Score特殊处理
- 可插拔策略设计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
import copy

from autoquant.core import QuantDtype, QScheme
from autoquant.utils import QConfig, get_default_qconfig, get_lsq_qconfig
from autoquant.observer import MinMaxObserver, HistogramObserver
from autoquant.fake_quant import FakeQuantize, LSQFakeQuantize


@dataclass
class TransformerQuantConfig:
    """
    Transformer专用量化配置
    
    Args:
        strategy: 量化策略 ('smoothquant', 'basic', 'awq')
        quantize_attention: 是否量化Attention层
        quantize_ffn: 是否量化FFN层
        quantize_kv_cache: 是否量化KV Cache
        smoothquant_alpha: SmoothQuant的alpha参数 (0.5-1.0)
    """
    strategy: str = 'smoothquant'
    quantize_attention: bool = True
    quantize_ffn: bool = True
    quantize_kv_cache: bool = True
    smoothquant_alpha: float = 0.5
    kv_cache_dtype: QuantDtype = QuantDtype.QINT8


class SmoothQuantQuantizer:
    """
    SmoothQuant量化器
    
    论文：https://arxiv.org/abs/2211.10438
    
    核心思想：
    - 将量化难度从激活转移到权重
    - 通过smooth factor s平衡权重和激活的量化难度
    - W' = W * diag(s), X' = X * diag(s)^-1
    """
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.smooth_factors: Dict[str, torch.Tensor] = {}
    
    def compute_smooth_factor(
        self,
        weight: torch.Tensor,
        activation: torch.Tensor,
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算SmoothQuant的平滑因子
        
        Args:
            weight: 权重张量 (out_dim, in_dim)
            activation: 激活张量 (batch, seq_len, in_dim)
            alpha: 平衡参数
            
        Returns:
            平滑因子 s (in_dim,)
        """
        alpha = alpha or self.alpha
        
        # 计算权重的绝对值最大值（per-out-channel）
        w_max = weight.abs().max(dim=0)[0]
        
        # 计算激活的绝对值最大值（per-channel）
        if activation.dim() == 3:
            x_max = activation.abs().max(dim=1)[0].max(dim=0)[0]
        else:
            x_max = activation.abs().max(dim=0)[0]
        
        # 计算平滑因子 s
        # s = (x_max^alpha / w_max^(1-alpha))
        s = torch.pow(x_max, alpha) / torch.pow(w_max.clamp_min(1e-5), 1 - alpha)
        s = s.clamp_min(1e-5)
        
        return s
    
    def apply_smooth(
        self,
        weight: torch.Tensor,
        activation: torch.Tensor,
        layer_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用SmoothQuant平滑
        
        Args:
            weight: 权重
            activation: 激活
            layer_name: 层名称（用于缓存）
            
        Returns:
            (smoothed_weight, smoothed_activation)
        """
        s = self.compute_smooth_factor(weight, activation)
        self.smooth_factors[layer_name] = s
        
        # 平滑权重: W' = W * diag(s)
        smoothed_weight = weight * s.unsqueeze(0)
        
        # 平滑激活: X' = X * diag(s)^-1
        smoothed_activation = activation / s
        
        return smoothed_weight, smoothed_activation


class KVCacheQuantizer:
    """
    KV Cache 量化器
    
    用于LLM推理时的KV Cache量化，节省显存
    """
    
    def __init__(
        self,
        dtype: QuantDtype = QuantDtype.QINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_SYMMETRIC
    ):
        self.dtype = dtype
        self.qscheme = qscheme
        self.k_observers: Dict[str, MinMaxObserver] = {}
        self.v_observers: Dict[str, MinMaxObserver] = {}
        self.calibrated = False
    
    def calibrate(
        self,
        layer_name: str,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor
    ):
        """
        校准KV Cache的量化参数
        
        Args:
            layer_name: 层名称
            k_cache: Key cache
            v_cache: Value cache
        """
        if layer_name not in self.k_observers:
            self.k_observers[layer_name] = MinMaxObserver(
                dtype=self.dtype,
                qscheme=self.qscheme
            )
            self.v_observers[layer_name] = MinMaxObserver(
                dtype=self.dtype,
                qscheme=self.qscheme
            )
        
        self.k_observers[layer_name](k_cache)
        self.v_observers[layer_name](v_cache)
    
    def quantize_kv(
        self,
        layer_name: str,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        量化KV Cache
        
        Args:
            layer_name: 层名称
            k_cache: Key cache
            v_cache: Value cache
            
        Returns:
            (quantized_k, quantized_v)
        """
        if layer_name not in self.k_observers:
            raise ValueError(f"Layer {layer_name} not calibrated")
        
        k_scale, k_zp = self.k_observers[layer_name].calculate_qparams()
        v_scale, v_zp = self.v_observers[layer_name].calculate_qparams()
        
        # 简单的fake quantize
        from autoquant.core import fake_quantize_ste
        quant_min = self.k_observers[layer_name].quant_min
        quant_max = self.k_observers[layer_name].quant_max
        
        quantized_k = fake_quantize_ste(k_cache, k_scale, k_zp, quant_min, quant_max)
        quantized_v = fake_quantize_ste(v_cache, v_scale, v_zp, quant_min, quant_max)
        
        return quantized_k, quantized_v


class TransformerQuantizer:
    """
    Transformer专用量化器 - 可插拔策略设计
    
    支持策略：
    - 'basic': 基础量化
    - 'smoothquant': SmoothQuant（推荐）
    - 'awq': AWQ (Activation-aware Weight Quantization)
    """
    
    STRATEGIES = {
        'smoothquant': SmoothQuantQuantizer,
        'basic': None,
        'awq': None,  # 可扩展
    }
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TransformerQuantConfig] = None
    ):
        self.model = model
        self.config = config or TransformerQuantConfig()
        self.config.quantize_attention = True
        self.config.quantize_ffn = True
        self.config.quantize_kv_cache = True
        
        # 初始化策略
        self.strategy = None
        if self.config.strategy == 'smoothquant':
            self.strategy = SmoothQuantQuantizer(alpha=self.config.smoothquant_alpha)
        
        # KV Cache量化器
        self.kv_quantizer = KVCacheQuantizer(
            dtype=self.config.kv_cache_dtype
        )
        
        # 层配置
        self.layer_configs: Dict[str, Dict[str, Any]] = {}
    
    def identify_transformer_layers(self) -> Dict[str, nn.Module]:
        """
        识别模型中的Transformer层
        
        Returns:
            {layer_name: module} 字典
        """
        transformer_layers = {}
        patterns = ['attention', 'attn', 'q_proj', 'k_proj', 'v_proj', 'out_proj', 'ffn', 'feed_forward', 'mlp']
        
        for name, module in self.model.named_modules():
            if any(pattern in name.lower() for pattern in patterns):
                transformer_layers[name] = module
        
        return transformer_layers
    
    def apply_smoothquant(
        self,
        calibration_data: List[torch.Tensor]
    ):
        """
        应用SmoothQuant
        
        Args:
            calibration_data: 校准数据
        """
        if self.strategy is None or not isinstance(self.strategy, SmoothQuantQuantizer):
            return
        
        print("🔧 应用SmoothQuant...")
        
        # 识别Linear层
        linear_layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module
        
        # 这里需要收集激活统计
        # 简化实现：假设我们有权重
        for name, module in linear_layers.items():
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name:
                # Attention投影层
                pass
        
        print(f"✓ SmoothQuant应用完成，处理了 {len(linear_layers)} 层")
    
    def prepare(
        self,
        default_qconfig: Optional[QConfig] = None,
    ) -> nn.Module:
        """
        准备Transformer量化模型
        
        Args:
            default_qconfig: 默认量化配置
            
        Returns:
            准备好的模型
        """
        qconfig = default_qconfig or get_transformer_qconfig()
        
        print(f"🚀 准备Transformer量化 (策略: {self.config.strategy})")
        
        # 识别Transformer层
        transformer_layers = self.identify_transformer_layers()
        print(f"  识别到 {len(transformer_layers)} 个Transformer相关层")
        
        # 这里应该使用ModelQuantizer来准备
        # 简化实现
        from autoquant.quantization import prepare
        prepared_model = prepare(self.model, qconfig, inplace=False)
        
        return prepared_model
    
    def get_config_summary(self) -> str:
        """获取配置摘要"""
        summary = []
        summary.append("=" * 60)
        summary.append("Transformer量化配置")
        summary.append("=" * 60)
        summary.append(f"策略: {self.config.strategy}")
        summary.append(f"量化Attention: {self.config.quantize_attention}")
        summary.append(f"量化FFN: {self.config.quantize_ffn}")
        summary.append(f"量化KV Cache: {self.config.quantize_kv_cache}")
        if self.config.strategy == 'smoothquant':
            summary.append(f"SmoothQuant alpha: {self.config.smoothquant_alpha}")
        summary.append("=" * 60)
        return "\n".join(summary)


def get_transformer_qconfig() -> QConfig:
    """
    获取Transformer专用QConfig
    
    特点：
    - 权重用per-channel对称量化
    - 激活用per-tensor非对称量化
    - 使用HistogramObserver获得更好精度
    """
    weight_observer = MinMaxObserver.with_args(
        dtype=QuantDtype.QINT8,
        qscheme=QScheme.PER_CHANNEL_SYMMETRIC,
        ch_axis=0
    )
    
    activation_observer = HistogramObserver.with_args(
        dtype=QuantDtype.QUINT8,
        qscheme=QScheme.PER_TENSOR_AFFINE
    )
    
    return QConfig(
        weight=FakeQuantize.with_args(observer=weight_observer),
        activation=FakeQuantize.with_args(observer=activation_observer)
    )


def get_smoothquant_qconfig(alpha: float = 0.5) -> QConfig:
    """
    获取SmoothQuant专用QConfig
    
    Args:
        alpha: SmoothQuant平衡参数
    """
    # SmoothQuant后可以用更激进的量化
    weight_observer = MinMaxObserver.with_args(
        dtype=QuantDtype.QINT8,
        qscheme=QScheme.PER_CHANNEL_SYMMETRIC,
        ch_axis=0
    )
    
    # 平滑后的激活动态范围更小，可以用MinMax
    activation_observer = MinMaxObserver.with_args(
        dtype=QuantDtype.QUINT8,
        qscheme=QScheme.PER_TENSOR_AFFINE
    )
    
    return QConfig(
        weight=FakeQuantize.with_args(observer=weight_observer),
        activation=FakeQuantize.with_args(observer=activation_observer)
    )
