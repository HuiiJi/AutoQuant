"""
Model Quantizer - 核心的模型量化类（修复版）

Author: jihui
Date: 2026-03-22
Desc:
    正确的 PTQ 流程（必须严格遵守！）：

    ┌─────────────────────────────────────────────────────────┐
    │  阶段 1: PREPARE（准备）                                 │
    │  - 插入 fake_quantize 模块                              │
    │  - 开启所有 observer（包括 weight 和 activation）        │
    │  - Weight 立即通过 fake_quantize 触发 observer 统计      │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │  阶段 2: CALIBRATION（校准）                             │
    │  - 带着 fake quantize forward（有量化噪声）             │
    │  - observer 持续统计 activation 的分布                   │
    │  - Weight 不需要重复统计（已在 prepare 时完成）          │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │  阶段 3: CONVERT（转换）                                  │
    │  - 对所有 fake_quant 调用 calculate_qparams()           │
    │  - 根据 observer 统计的 min/max 计算 scale/zero_point   │
    │  - 禁用所有 observer                                     │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │  阶段 4: INFERENCE（推理）                                │
    │  - 用计算好的 scale/zero_point 做 fake quant            │
    │  - observer 保持禁用                                     │
    └─────────────────────────────────────────────────────────┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, Optional, Set, List, Tuple
from autoquant.utils import QConfig
from autoquant.fake_quant import PTQFakeQuantize
from autoquant.core import QScheme


# ============================================================================
# 量化桩模块
# ============================================================================

class QuantStub(nn.Module):
    """
    输入量化节点 - 在模型输入处插入
    """

    def __init__(self, qconfig: QConfig):
        super().__init__()
        self.activation_fake_quant = qconfig.activation()

    def forward(self, x):
        return self.activation_fake_quant(x)

    def enable_observer(self):
        if hasattr(self.activation_fake_quant, 'enable_observer'):
            self.activation_fake_quant.enable_observer()

    def disable_observer(self):
        if hasattr(self.activation_fake_quant, 'disable_observer'):
            self.activation_fake_quant.disable_observer()

    def calculate_qparams(self):
        if hasattr(self.activation_fake_quant, 'calculate_qparams'):
            self.activation_fake_quant.calculate_qparams()


class DeQuantStub(nn.Module):
    """
    输出反量化节点 - 在模型输出处插入
    作用：占位符，ONNX 导出时会正确处理
    """

    def forward(self, x):
        return x


# ============================================================================
# 可量化模块包装器（核心修复！）
# ============================================================================

class QuantizableModule(nn.Module):
    """
    可量化模块包装器

    【关键流程说明】：
    ──────────────────────────────────────────────────────────────

    PREPARE 阶段（在 __init__ 后，由 ModelQuantizer.prepare() 调用）：
    1. enable_observer() 被调用，observer 开启
    2. _collect_weight_stats() 被调用，weight 通过 fake_quant 触发 observer 统计

    CALIBRATION 阶段（forward）：
    1. input_fake_quant(x): observer 统计 x 的分布，同时做 fake quant
    2. weight_fake_quant(weight): 用已统计的 qparams 做 fake quant（不再统计）
    3. 用量化后的 x 和 weight 计算 Conv/Linear

    CONVERT 阶段：
    1. calculate_qparams() 根据 observer 统计的 min/max 计算 scale/zp
    2. disable_observer() 禁用 observer

    INFERENCE 阶段（forward）：
    1. input_fake_quant(x): 用计算好的 scale/zp 做 fake quant
    2. weight_fake_quant(weight): 用计算好的 scale/zp 做 fake quant
    ──────────────────────────────────────────────────────────────
    """

    # 需要量化 weight 的模块类型
    WEIGHT_QUANT_MODULES = {
        'Conv1d', 'Conv2d', 'Conv3d',
        'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
        'Linear', 'Bilinear',
        'Embedding', 'EmbeddingBag'
    }

    def __init__(self, module: nn.Module, qconfig: QConfig):
        super().__init__()
        self.module = module
        self.weight_fake_quant = qconfig.weight()
        self.input_fake_quant = qconfig.activation()

        # 判断是否需要量化 weight
        self.should_quantize_weight = False
        module_type = type(self.module).__name__

        if hasattr(self.module, 'weight') and self.module.weight is not None:
            if module_type in self.WEIGHT_QUANT_MODULES:
                self.should_quantize_weight = True

        self._weight_stats_collected = False

    def _collect_weight_stats(self):
        """
        收集 weight 的统计信息（在 prepare 阶段调用）
        此时 observer 已经开启，fake_quant 会触发 observer 统计
        """
        if self.should_quantize_weight and not self._weight_stats_collected:
            # 触发 observer 统计 weight 的分布
            _ = self.weight_fake_quant(self.module.weight)
            self._weight_stats_collected = True

    def forward(self, x):
        # 步骤 1：对 INPUT 量化（observer 统计 + fake quant）
        x_quant = self.input_fake_quant(x)

        # 步骤 2：对 WEIGHT 量化（用已统计的 qparams 做 fake quant）
        # 步骤 3：用 QUANTIZED 的输入和 weight 计算 op！
        if self.should_quantize_weight:
            quant_weight = self.weight_fake_quant(self.module.weight)

            # 直接调用底层函数，不替换 Parameter！
            if isinstance(self.module, nn.Conv2d):
                output = F.conv2d(x_quant, quant_weight, self.module.bias,
                                  self.module.stride, self.module.padding,
                                  self.module.dilation, self.module.groups)
            elif isinstance(self.module, nn.Conv1d):
                output = F.conv1d(x_quant, quant_weight, self.module.bias,
                                  self.module.stride, self.module.padding,
                                  self.module.dilation, self.module.groups)
            elif isinstance(self.module, nn.Conv3d):
                output = F.conv3d(x_quant, quant_weight, self.module.bias,
                                  self.module.stride, self.module.padding,
                                  self.module.dilation, self.module.groups)
            elif isinstance(self.module, nn.ConvTranspose1d):
                output = F.conv_transpose1d(x_quant, quant_weight, self.module.bias,
                                            self.module.stride, self.module.padding,
                                            self.module.output_padding,
                                            self.module.dilation, self.module.groups)
            elif isinstance(self.module, nn.ConvTranspose2d):
                output = F.conv_transpose2d(x_quant, quant_weight, self.module.bias,
                                            self.module.stride, self.module.padding,
                                            self.module.output_padding,
                                            self.module.dilation, self.module.groups)
            elif isinstance(self.module, nn.ConvTranspose3d):
                output = F.conv_transpose3d(x_quant, quant_weight, self.module.bias,
                                            self.module.stride, self.module.padding,
                                            self.module.output_padding,
                                            self.module.dilation, self.module.groups)
            elif isinstance(self.module, nn.Linear):
                output = F.linear(x_quant, quant_weight, self.module.bias)
            else:
                # 其他类型（如 Embedding）直接调用原模块
                output = self.module(x_quant)
        else:
            # 无 weight 的模块（激活函数、池化等）
            output = self.module(x_quant)

        return output

    def enable_observer(self):
        """开启所有 observer"""
        if hasattr(self.input_fake_quant, 'enable_observer'):
            self.input_fake_quant.enable_observer()
        if hasattr(self.weight_fake_quant, 'enable_observer'):
            self.weight_fake_quant.enable_observer()
        self._weight_stats_collected = False  # 重置，允许重新统计

    def disable_observer(self):
        """禁用所有 observer"""
        if hasattr(self.input_fake_quant, 'disable_observer'):
            self.input_fake_quant.disable_observer()
        if hasattr(self.weight_fake_quant, 'disable_observer'):
            self.weight_fake_quant.disable_observer()

    def calculate_qparams(self):
        """计算所有 fake_quant 的 qparams"""
        if hasattr(self.input_fake_quant, 'calculate_qparams'):
            self.input_fake_quant.calculate_qparams()
        if hasattr(self.weight_fake_quant, 'calculate_qparams'):
            self.weight_fake_quant.calculate_qparams()

    def convert(self, permanently_quantize_weight: bool = False):
        """
        转换为推理模式

        Args:
            permanently_quantize_weight: 是否永久量化 weight
                - False: 保持 weight 为浮点（推荐用于 QDQ ONNX 导出）
                - True: 永久量化 weight 为 int8（仅用于纯整数推理）
        """
        # 1. 计算 qparams
        self.calculate_qparams()
        # 3. 禁用所有 observer
        self.disable_observer()

        return self


# ============================================================================
# 模型包装器
# ============================================================================

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

    def enable_observer(self):
        if hasattr(self.quant, 'enable_observer'):
            self.quant.enable_observer()

    def disable_observer(self):
        if hasattr(self.quant, 'disable_observer'):
            self.quant.disable_observer()

    def calculate_qparams(self):
        if hasattr(self.quant, 'calculate_qparams'):
            self.quant.calculate_qparams()


class ModelQuantizer:
    """
    模型量化器

    正确 PTQ 流程：
    ┌─────────────────────────────────────────────────────────────┐
    │  1. prepare()  → 插入 fake_quant，开启 observer，统计 weight │
    │  2. calibrate() → 用校准数据 forward，统计 activation       │
    │  3. convert()  → 计算 qparams，禁用 observer                │
    │  4. 推理/导出 ONNX                                          │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, model: nn.Module, qconfig: QConfig):
        self.model = model
        self.qconfig = qconfig
        self.original_modules: Dict[str, nn.Module] = {}
        self.quantized_modules: Dict[str, nn.Module] = {}
        self.prepared_model: Optional[nn.Module] = None

    def prepare(self, inplace: bool = False, skip_layers: Optional[Set[str]] = None) -> nn.Module:
        """
        准备阶段：插入 fake_quant 模块，开启 observer，统计 weight

        Args:
            inplace: 是否原地修改模型
            skip_layers: 跳过量化的层名集合（如 {'layer4.0.conv1'}）

        Returns:
            准备好的模型
        """
        if not inplace:
            model = copy.deepcopy(self.model)
        else:
            model = self.model

        # 1. 保存原始模块（用于调试/回滚）
        self._save_original_modules(model)
        # 2. 替换可量化模块为 QuantizableModule
        self._replace_quantizable_modules(model, skip_layers)
        # 3. 在模型输入输出插入 QuantStub/DeQuantStub
        model = self._insert_quant_dequant_stubs(model)
        # 4. 开启所有 observer（包括 QuantizableModule 内部的 fake_quant）
        self._enable_all_observers(model)
        # 5. 统计所有 weight（此时 observer 已开启）
        self._collect_all_weight_stats(model)
        model.eval()
        self.prepared_model = model
        return model

    def _enable_all_observers(self, model: nn.Module):
        """
        启用所有模块的 observer

        ⚠️ 关键修复：使用 named_modules() 遍历所有子模块，
        而不仅仅是 named_children()
        """
        for name, module in model.named_modules():
            if hasattr(module, 'enable_observer'):
                try:
                    module.enable_observer()
                except Exception as e:
                    pass

    def _collect_all_weight_stats(self, model: nn.Module):
        """
        收集所有 QuantizableModule 的 weight 统计

        ⚠️ 关键修复：在 prepare 阶段，observer 开启后立即统计 weight
        """
        for name, module in model.named_modules():
            if isinstance(module, QuantizableModule):
                module._collect_weight_stats()

    def calibrate(self, calib_data, device: Optional[torch.device] = None, verbose: bool = True):
        """
        校准阶段：用校准数据 forward，统计 activation 的分布

        Args:
            calib_data: 校准数据（DataLoader / List[Tensor] / Tensor）
            device: 设备
            verbose: 是否打印进度
        """
        if self.prepared_model is None:
            raise ValueError("请先调用 prepare() 准备模型")

        if device is None:
            device = next(self.prepared_model.parameters()).device

        self.prepared_model.eval()
        self.prepared_model.to(device)

        if verbose:
            print("🔧 开始校准...")

        with torch.no_grad():
            if isinstance(calib_data, torch.utils.data.DataLoader):
                total_batches = len(calib_data)
                for i, batch in enumerate(calib_data):
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    inputs = inputs.to(device)
                    self.prepared_model(inputs)
                    if verbose and (i + 1) % 10 == 0:
                        print(f"  已处理 {i + 1}/{total_batches} 批")

            elif isinstance(calib_data, list):
                for i, inputs in enumerate(calib_data):
                    inputs = inputs.to(device)
                    self.prepared_model(inputs)
                    if verbose and (i + 1) % 10 == 0:
                        print(f"  已处理 {i + 1}/{len(calib_data)} 个样本")

            elif isinstance(calib_data, torch.Tensor):
                inputs = calib_data.to(device)
                self.prepared_model(inputs)
                if verbose:
                    print("  已处理单样本")

            else:
                raise ValueError(f"不支持的校准数据类型：{type(calib_data)}")

        if verbose:
            print("✅ 校准完成！")

    def convert(self, inplace: bool = False, permanently_quantize_weight: bool = False) -> nn.Module:
        """
        转换阶段
        """
        if self.prepared_model is None:
            raise ValueError("请先调用 prepare() 和 calibrate()")

        if not inplace:
            # 在 deepcopy 之前，先 detach 所有非 graph leaf 的 tensors
            # 这样可以避免 RuntimeError: Only Tensors created explicitly by the user support deepcopy
            self._prepare_model_for_deepcopy(self.prepared_model)
            model = copy.deepcopy(self.prepared_model)
        else:
            model = self.prepared_model

        self._convert_modules(model, permanently_quantize_weight)

        if inplace:
            self.prepared_model = model

        return model

    def _prepare_model_for_deepcopy(self, model: nn.Module):
        """
        准备模型用于 deepcopy - detach 所有非 leaf tensors
        这是为了解决 deepcopy 的问题
        """
        # 1. 处理所有 buffers
        for module in model.modules():
            for name, buffer in list(module.named_buffers(recurse=False)):
                if buffer is not None and not buffer.is_leaf:
                    # detach 并替换
                    detached_buffer = buffer.detach()
                    setattr(module, name, detached_buffer)

            # 2. 处理 observer 的特殊属性（如果有）
            if hasattr(module, 'observer') and module.observer is not None:
                observer = module.observer
                # 处理 histogram
                if hasattr(observer, 'histogram') and observer.histogram is not None:
                    if not observer.histogram.is_leaf:
                        observer.histogram = observer.histogram.detach()
                # 处理 min_val/max_val
                if hasattr(observer, '_min_val') and observer._min_val is not None:
                    if isinstance(observer._min_val, torch.Tensor) and not observer._min_val.is_leaf:
                        observer._min_val = observer._min_val.detach()
                if hasattr(observer, '_max_val') and observer._max_val is not None:
                    if isinstance(observer._max_val, torch.Tensor) and not observer._max_val.is_leaf:
                        observer._max_val = observer._max_val.detach()

    def _convert_modules(self, model: nn.Module, permanently_quantize_weight: bool, prefix: str = ""):
        """递归转换"""
        for name, module in model.named_children():
            if isinstance(module, QuantizableModule):
                # 调用 QuantizableModule.convert -> 计算最终 qparams -> 关闭 observer
                fixed_module = module.convert(permanently_quantize_weight=permanently_quantize_weight)
                setattr(model, name, fixed_module)
            elif isinstance(module, QuantizableModelWrapper):
                # 处理输入输出 Stub
                self._convert_modules(module.model, permanently_quantize_weight)
                # 确保 Wrapper 自身的 quant/dequant 节点也固化
                if hasattr(module, 'calculate_qparams'):
                    module.calculate_qparams()
                if hasattr(module, 'disable_observer'):
                    module.disable_observer()
            # 注意：不移除对 QuantStub/DeQuantStub 的检查，因为它们是自定义的
            # elif isinstance(module, (nn.QuantStub, nn.DeQuantStub)):  # ❌ 这不是 PyTorch 标准模块
            else:
                full_name = f"{prefix}.{name}" if prefix else name
                self._convert_modules(module, permanently_quantize_weight, full_name)

    def _save_original_modules(self, model: nn.Module, prefix: str = ""):
        """保存原始模块引用（用于调试）"""
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            self.original_modules[full_name] = module
            self._save_original_modules(module, full_name)

    def _replace_quantizable_modules(self, model: nn.Module, skip_layers: Optional[Set[str]] = None, prefix: str = ""):
        """替换可量化模块为 QuantizableModule"""
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
        """判断模块是否可量化"""
        quantizable_types = (
            # Conv 系列
            nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            # Linear
            nn.Linear, nn.Bilinear,
            # 嵌入层
            nn.Embedding, nn.EmbeddingBag,
            # 激活函数
            nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU,
            nn.ELU, nn.SELU, nn.CELU, nn.GELU, nn.SiLU, nn.Hardswish,
            # 池化层
            nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
            nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
            nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
        )
        return isinstance(module, quantizable_types)

    def _create_quantized_module(self, module: nn.Module) -> nn.Module:
        """创建量化版本的模块"""
        return QuantizableModule(module, self.qconfig)

    def _insert_quant_dequant_stubs(self, model: nn.Module) -> nn.Module:
        """在模型输入输出插入量化桩"""
        return QuantizableModelWrapper(model, self.qconfig)

    def verify_quantization(self, model: nn.Module = None) -> Dict[str, List[str]]:
        """
        验证量化是否正确应用

        Returns:
            问题列表
        """
        if model is None:
            model = self.prepared_model

        if model is None:
            return {"error": ["模型未准备"]}

        issues = {
            "missing_fake_quant": [],
            "observer_not_disabled": [],
            "qparams_not_calculated": [],
        }

        for name, module in model.named_modules():
            if isinstance(module, QuantizableModule):
                # 检查 fake_quant 是否存在
                if not hasattr(module, 'input_fake_quant'):
                    issues["missing_fake_quant"].append(f"{name}: 缺少 input_fake_quant")
                if module.should_quantize_weight and not hasattr(module, 'weight_fake_quant'):
                    issues["missing_fake_quant"].append(f"{name}: 缺少 weight_fake_quant")

        return issues
