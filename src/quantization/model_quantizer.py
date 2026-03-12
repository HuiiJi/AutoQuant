"""
Model Quantizer - 核心的模型量化类

对 ResNet 等模型的 Conv 和 Linear 进行量化，支持 per-tensor 和 per-channel 量化

Author: jihui
Date: 2026-03-13
"""
import torch
import torch.nn as nn
import copy
from typing import Dict, Optional, Set
from autoquant.utils import QConfig
from autoquant.fake_quant import PTQFakeQuantize
from autoquant.core import QScheme


class QuantStub(nn.Module):
    """
    输入量化节点 - 在模型输入处插入
    作用：对输入进行量化
    """
    def __init__(self, qconfig: QConfig):
        super().__init__()
        self.quant = qconfig.activation()

    def forward(self, x):
        return self.quant(x)


class DeQuantStub(nn.Module):
    """
    输出反量化节点 - 在模型输出处插入
    作用：占位符，ONNX 导出时会正确处理
    """
    def forward(self, x):
        return x


class QuantizableModule(nn.Module):
    """
    可量化模块包装器
    将 Conv/Linear 包装起来，添加 weight 和 activation 的量化
    
    简化逻辑：
    - weight 和 activation 都通过 fake quant
    - 校准阶段：统计 + 量化（带噪声）
    - 推理阶段：直接量化
    """
    def __init__(self, module: nn.Module, qconfig: QConfig):
        super().__init__()
        self.module = module
        self.weight_fake_quant = qconfig.weight()
        self.activation_fake_quant = qconfig.activation()

    def forward(self, x):
        # 只有某些特定模块才量化 weight
        # 不量化 weight 的模块：BatchNorm、LayerNorm、GroupNorm、InstanceNorm、PReLU 等
        should_quantize_weight = False
        module_type = type(self.module).__name__
        
        # 只对这些模块量化 weight
        weight_quant_modules = {
            'Conv1d', 'Conv2d', 'Conv3d',
            'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
            'Linear', 'Bilinear',
            'Embedding', 'EmbeddingBag'
        }
        
        if hasattr(self.module, 'weight') and self.module.weight is not None:
            if module_type in weight_quant_modules:
                should_quantize_weight = True
        
        if should_quantize_weight:
            quant_weight = self.weight_fake_quant(self.module.weight)
            original_weight = self.module.weight
            self.module.weight = nn.Parameter(quant_weight)
            output = self.module(x)
            self.module.weight = original_weight
        else:
            output = self.module(x)
        
        # 所有模块都量化 activation
        output = self.activation_fake_quant(output)
        return output
    
    def convert(self):
        """
        转换为推理模式：
        1. 计算 qparams（只对有统计数据的）
        2. 永久量化权重（只对特定模块）
        3. 禁用 observer
        """
        # 判断是否需要量化 weight
        should_quantize_weight = False
        module_type = type(self.module).__name__
        weight_quant_modules = {
            'Conv1d', 'Conv2d', 'Conv3d',
            'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
            'Linear', 'Bilinear',
            'Embedding', 'EmbeddingBag'
        }
        
        if hasattr(self.module, 'weight') and self.module.weight is not None:
            if module_type in weight_quant_modules:
                should_quantize_weight = True
        
        # 只对需要量化 weight 的模块计算 weight qparams
        if should_quantize_weight and hasattr(self.weight_fake_quant, 'calculate_qparams'):
            self.weight_fake_quant.calculate_qparams()
        
        # 总是计算 activation qparams
        if hasattr(self.activation_fake_quant, 'calculate_qparams'):
            self.activation_fake_quant.calculate_qparams()
        
        # 只对特定模块永久量化 weight
        if should_quantize_weight:
            quant_weight = self.weight_fake_quant(self.module.weight)
            self.module.weight = nn.Parameter(quant_weight)
        
        # 禁用所有 observer
        if hasattr(self.activation_fake_quant, 'disable_observer'):
            self.activation_fake_quant.disable_observer()
        if hasattr(self.weight_fake_quant, 'disable_observer'):
            self.weight_fake_quant.disable_observer()
        
        return self


class QuantizableModelWrapper(nn.Module):
    """
    模型包装器 - 在整个模型的输入和输出处插入量化节点
    """
    def __init__(self, model: nn.Module, qconfig: QConfig):
        super().__init__()
        self.model = model
        self.quant = QuantStub(qconfig)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


class ModelQuantizer:
    """
    模型量化器
    
    正确 PTQ 流程：
    ┌─────────────────────────────────────────────────────────────┐
    │  PREPARE  →  CALIBRATE (统计+量化噪声)  →  CONVERT  →  ONNX │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, model: nn.Module, qconfig: QConfig):
        self.model = model
        self.qconfig = qconfig
        self.original_modules: Dict[str, nn.Module] = {}
        self.quantized_modules: Dict[str, nn.Module] = {}
        self.prepared_model: Optional[nn.Module] = None

    def prepare(self, inplace: bool = False, skip_layers: Optional[Set[str]] = None) -> nn.Module:
        if not inplace:
            model = copy.deepcopy(self.model)
        else:
            model = self.model

        self._save_original_modules(model)
        self._replace_quantizable_modules(model, skip_layers)
        model = self._insert_quant_dequant_stubs(model)
        
        self._enable_all_observers(model)
        self.prepared_model = model
        return model
    
    def _enable_all_observers(self, model: nn.Module):
        for name, module in model.named_children():
            if hasattr(module, 'enable_observer'):
                module.enable_observer()
            self._enable_all_observers(module)
    
    def calibrate(self, calib_data, device: Optional[torch.device] = None):
        if self.prepared_model is None:
            raise ValueError("请先调用 prepare() 准备模型")
        
        if device is None:
            device = next(self.prepared_model.parameters()).device
        
        self.prepared_model.eval()
        self.prepared_model.to(device)
        
        print("🔧 开始校准...")
        
        with torch.no_grad():
            if isinstance(calib_data, torch.utils.data.DataLoader):
                for i, batch in enumerate(calib_data):
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    inputs = inputs.to(device)
                    self.prepared_model(inputs)
                    if (i + 1) % 10 == 0:
                        print(f"  已处理 {i + 1} 批")
            
            elif isinstance(calib_data, list):
                for i, inputs in enumerate(calib_data):
                    inputs = inputs.to(device)
                    self.prepared_model(inputs)
                    if (i + 1) % 10 == 0:
                        print(f"  已处理 {i + 1} 个样本")
            
            elif isinstance(calib_data, torch.Tensor):
                inputs = calib_data.to(device)
                self.prepared_model(inputs)
                print("  已处理单样本")
            
            else:
                raise ValueError(f"不支持的校准数据类型: {type(calib_data)}")
        
        print("✅ 校准完成！")
    
    def convert(self, inplace: bool = False) -> nn.Module:
        if self.prepared_model is None:
            raise ValueError("请先调用 prepare() 和 calibrate()")
        
        if not inplace:
            model = copy.deepcopy(self.prepared_model)
        else:
            model = self.prepared_model
        
        self._convert_modules(model)
        return model
    
    def _convert_modules(self, model: nn.Module, prefix: str = ""):
        for name, module in model.named_children():
            if isinstance(module, QuantizableModule):
                fixed_module = module.convert()
                setattr(model, name, fixed_module)
            elif isinstance(module, QuantizableModelWrapper):
                self._convert_modules(module.model, prefix)
            elif isinstance(module, QuantStub):
                if hasattr(module.quant, 'disable_observer'):
                    module.quant.disable_observer()
            else:
                full_name = f"{prefix}.{name}" if prefix else name
                self._convert_modules(module, full_name)

    def _save_original_modules(self, model: nn.Module, prefix: str = ""):
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            self.original_modules[full_name] = module
            self._save_original_modules(module, full_name)

    def _replace_quantizable_modules(self, model: nn.Module, skip_layers: Optional[Set[str]] = None, prefix: str = ""):
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            should_quantize = True
            if skip_layers is not None and full_name in skip_layers:
                should_quantize = False

            if should_quantize and self._is_quantizable(module):
                quant_module = self._create_quantized_module(module)
                setattr(model, name, quant_module)
                self.quantized_modules[full_name] = quant_module
            else:
                self._replace_quantizable_modules(module, skip_layers, full_name)

    def _is_quantizable(self, module: nn.Module) -> bool:
        """
        判断模块是否可量化 - 支持所有 PyTorch 官方支持的量化模块
        
        支持的模块：
        - Conv 系列：Conv1d, Conv2d, Conv3d
        - Linear
        - 激活函数：ReLU, ReLU6, LeakyReLU, PReLU, ELU, SELU, CELU
        - 池化层：MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptiveAvgPool1d/2d/3d
        - 归一化层：BatchNorm1d/2d/3d, LayerNorm, GroupNorm, InstanceNorm1d/2d/3d
        - 其他：ConvTranspose1d/2d/3d, Embedding
        """
        quantizable_types = (
            # Conv 系列
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            
            # Linear
            nn.Linear,
            nn.Bilinear,
            
            # 嵌入层
            nn.Embedding,
            nn.EmbeddingBag,
            
            # 激活函数
            nn.ReLU,
            nn.ReLU6,
            nn.LeakyReLU,
            nn.PReLU,
            nn.ELU,
            nn.SELU,
            nn.CELU,
            nn.GELU,
            nn.SiLU,
            nn.Hardswish,
            
            # 池化层
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool3d,
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
            
            # 归一化层
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        )
        return isinstance(module, quantizable_types)

    def _create_quantized_module(self, module: nn.Module) -> nn.Module:
        """
        创建量化版本的模块 - 支持所有可量化模块
        
        对于有 weight 的模块（Conv/Linear/Embedding 等）：量化 weight + activation
        对于无 weight 的模块（激活/池化/归一化）：只量化 activation
        """
        # 所有支持的模块都用 QuantizableModule 包装
        # QuantizableModule 会自动判断模块是否有 weight
        quant_module = QuantizableModule(module, self.qconfig)
        return quant_module

    def _insert_quant_dequant_stubs(self, model: nn.Module) -> nn.Module:
        return QuantizableModelWrapper(model, self.qconfig)
