"""
Model Quantizer - 核心的模型量化类
"""
import torch
import torch.nn as nn
import copy
from typing import Dict, List, Optional, Any, Callable, Set
from autoquant.utils import QConfig
from autoquant.fake_quant import FakeQuantize, FixedFakeQuantize
from autoquant.observer import ObserverBase


class QuantStub(nn.Module):
    """
    Quantize Stub - 在输入处插入的量化节点
    """
    def __init__(self, qconfig: QConfig):
        super().__init__()
        self.quant = qconfig.activation()

    def forward(self, x):
        return self.quant(x)


class DeQuantStub(nn.Module):
    """
    DeQuantize Stub - 在输出处插入的反量化节点
    """
    def forward(self, x):
        return x


class ModelQuantizer:
    """
    模型量化器 - 负责模型的prepare、calibrate和convert
    
    完整的PTQ流程：
    1. prepare - 插入Observer和FakeQuant节点
    2. calibrate - 在校准数据上运行，统计分布
    3. convert - 转换为固定量化参数的模型
    """

    def __init__(self, model: nn.Module, qconfig: QConfig):
        self.model = model
        self.qconfig = qconfig
        self.original_modules: Dict[str, nn.Module] = {}
        self.quantized_modules: Dict[str, nn.Module] = {}
        self.quant_stubs: Dict[str, QuantStub] = {}
        self.dequant_stubs: Dict[str, DeQuantStub] = {}
        self.prepared_model: Optional[nn.Module] = None

    def prepare(self, inplace: bool = False, leaf_modules: Optional[Set[str]] = None, skip_layers: Optional[Set[str]] = None) -> nn.Module:
        """
        准备模型用于PTQ/QAT
        
        Args:
            inplace: 是否原地修改
            leaf_modules: 需要量化的叶子节点名称集合（可选）
            skip_layers: 跳过量化的层名称集合（保持浮点）
        
        Returns:
            准备好的量化模型
        """
        if not inplace:
            model = copy.deepcopy(self.model)
        else:
            model = self.model

        # 保存原始模块
        self._save_original_modules(model)

        # 替换可量化的模块
        self._replace_quantizable_modules(model, leaf_modules, skip_layers)

        # 在输入和输出处插入stub
        model = self._insert_quant_dequant_stubs(model)
        
        self.prepared_model = model
        return model
    
    def calibrate(self, calib_data, device: Optional[torch.device] = None):
        """
        校准模型（PTQ阶段）
        在校准数据上运行，统计激活和权重的分布
        
        Args:
            calib_data: 校准数据，可以是：
                       - DataLoader
                       - List[Tensor]
                       - Tensor
            device: 计算设备
        """
        if self.prepared_model is None:
            raise ValueError("请先调用 prepare() 准备模型")
        
        if device is None:
            device = next(self.prepared_model.parameters()).device
        
        self.prepared_model.eval()
        self.prepared_model.to(device)
        
        print("🔧 开始校准...")
        
        with torch.no_grad():
            if isinstance(calib_data, torch.utils.data.DataLoader):
                # DataLoader
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
                # List of tensors
                for i, inputs in enumerate(calib_data):
                    inputs = inputs.to(device)
                    self.prepared_model(inputs)
                    if (i + 1) % 10 == 0:
                        print(f"  已处理 {i + 1} 个样本")
            
            elif isinstance(calib_data, torch.Tensor):
                # Single tensor
                inputs = calib_data.to(device)
                self.prepared_model(inputs)
                print("  已处理单样本")
            
            else:
                raise ValueError(f"不支持的校准数据类型: {type(calib_data)}")
        
        print("✅ 校准完成！")

    def convert(self, model: nn.Module, inplace: bool = False) -> nn.Module:
        """
        将准备好的模型转换为量化模型
        
        Args:
            model: 准备好的模型
            inplace: 是否原地修改
        
        Returns:
            转换后的量化模型
        """
        if not inplace:
            model = copy.deepcopy(model)
        else:
            pass

        # 转换模块
        self._convert_modules(model)

        return model

    def _save_original_modules(self, model: nn.Module, prefix: str = ""):
        """保存原始模块"""
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            self.original_modules[full_name] = module
            self._save_original_modules(module, full_name)

    def _replace_quantizable_modules(self, model: nn.Module, leaf_modules: Optional[Set[str]] = None, skip_layers: Optional[Set[str]] = None, prefix: str = ""):
        """替换可量化的模块"""
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # 检查是否需要跳过
            should_quantize = True
            if skip_layers is not None and full_name in skip_layers:
                should_quantize = False
            
            # 检查是否是指定的叶子模块
            if leaf_modules is not None and full_name not in leaf_modules:
                should_quantize = False

            if should_quantize and self._is_quantizable(module):
                # 替换为量化版本
                quant_module = self._create_quantized_module(module)
                setattr(model, name, quant_module)
                self.quantized_modules[full_name] = quant_module
            else:
                # 递归处理子模块
                self._replace_quantizable_modules(module, leaf_modules, skip_layers, full_name)

    def _is_quantizable(self, module: nn.Module) -> bool:
        """判断模块是否可量化"""
        quantizable_types = (
            nn.Conv2d,
            nn.Linear,
            nn.Conv1d,
            nn.Conv3d,
            nn.BatchNorm2d,
            nn.ReLU,
            nn.ReLU6,
            nn.MaxPool2d,
            nn.AvgPool2d,
        )
        return isinstance(module, quantizable_types)

    def _create_quantized_module(self, module: nn.Module) -> nn.Module:
        """创建量化版本的模块"""
        # 对于权重需要量化的模块，添加weight的fake quant
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d, nn.Conv3d)):
            quant_module = QuantizableModule(module, self.qconfig)
            return quant_module
        else:
            # 对于激活值需要量化的模块，直接返回原模块
            return module

    def _insert_quant_dequant_stubs(self, model: nn.Module) -> nn.Module:
        """在模型的输入和输出处插入quant/dequant stubs"""
        return QuantizableModelWrapper(model, self.qconfig)

    def _convert_modules(self, model: nn.Module, prefix: str = ""):
        """转换模块为量化版本"""
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(module, QuantizableModule):
                # 转换为固定量化的模块
                fixed_module = module.convert()
                setattr(model, name, fixed_module)
            elif isinstance(module, QuantizableModelWrapper):
                # 转换wrapper内部的模块
                self._convert_modules(module.model, prefix)
            else:
                # 递归处理子模块
                self._convert_modules(module, full_name)


class QuantizableModule(nn.Module):
    """
    可量化的模块包装器
    包含weight的fake quant
    """
    def __init__(self, module: nn.Module, qconfig: QConfig):
        super().__init__()
        self.module = module
        self.weight_fake_quant = qconfig.weight()
        self.activation_fake_quant = qconfig.activation()

    def forward(self, x):
        # 量化权重
        if hasattr(self.module, 'weight') and self.module.weight is not None:
            quant_weight = self.weight_fake_quant(self.module.weight)
            # 临时替换权重
            original_weight = self.module.weight
            self.module.weight = nn.Parameter(quant_weight)
            output = self.module(x)
            # 恢复原权重
            self.module.weight = original_weight
        else:
            output = self.module(x)
        
        # 量化激活值
        output = self.activation_fake_quant(output)
        return output

    def convert(self) -> nn.Module:
        """转换为固定量化的模块"""
        # 计算weight的量化参数
        if hasattr(self.module, 'weight') and self.module.weight is not None:
            self.weight_fake_quant.calculate_qparams()
            # 量化权重
            quant_weight = self.weight_fake_quant._fake_quantize(self.module.weight)
            self.module.weight = nn.Parameter(quant_weight)
        
        # 计算activation的量化参数
        self.activation_fake_quant.calculate_qparams()
        
        # 创建固定的fake quant
        fixed_act_fq = FixedFakeQuantize(
            scale=self.activation_fake_quant.scale,
            zero_point=self.activation_fake_quant.zero_point,
            dtype=self.activation_fake_quant.dtype,
            qscheme=self.activation_fake_quant.qscheme,
            quant_min=self.activation_fake_quant.observer.quant_min,
            quant_max=self.activation_fake_quant.observer.quant_max,
            ch_axis=self.activation_fake_quant.ch_axis,
        )
        
        return FixedQuantizableModule(self.module, fixed_act_fq)


class FixedQuantizableModule(nn.Module):
    """
    固定量化的模块
    """
    def __init__(self, module: nn.Module, activation_fake_quant: FixedFakeQuantize):
        super().__init__()
        self.module = module
        self.activation_fake_quant = activation_fake_quant

    def forward(self, x):
        output = self.module(x)
        output = self.activation_fake_quant(output)
        return output


class QuantizableModelWrapper(nn.Module):
    """
    可量化模型的包装器
    在输入和输出处插入quant/dequant stubs
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
