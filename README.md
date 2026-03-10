# AutoQuant - 专业的AI模型量化工具链条

AutoQuant 是一套参考 MQBench 设计的专业模型量化工具链条，支持从 PyTorch 模型到 ONNX 模型（包含 QDQ 操作）的完整量化流程。

## 核心特性

- **自定义 Observer**：完全自定义的 Observer 实现，支持多种数据统计策略
- **自定义 FakeQuant**：完整的 FakeQuantize 实现，支持 STE（直通估计器）、LSQ、PACT 等高级量化方法
- **PTQ/QAT 支持**：同时支持训练后量化（PTQ）和量化感知训练（QAT）
- **混合精度量化**：支持不同层使用不同的量化精度
- **符号化追踪**：基于 torch.fx 的符号化追踪，减少胶水节点
- **ONNX 导出**：支持导出包含 QDQ 节点的 ONNX 模型，兼容 TensorRT/OpenVINO
- **叶子节点追踪**：支持针对特定叶子节点进行量化（如 NAFNet 的特定层）
- **敏感度分析**：自动分析各层的量化敏感度，辅助混合精度配置
- **Transformer 支持**：专门针对 Transformer 模型的量化策略，包括 SmoothQuant

## 项目结构

```
autoquant/
├── autoquant/
│   ├── __init__.py
│   ├── core/                   # 核心定义
│   │   ├── __init__.py
│   │   ├── dtype.py            # 数据类型和量化方案
│   │   └── autograd_functions.py # 自动梯度函数（STE、LSQ、PACT）
│   ├── observer/               # Observer 模块
│   │   ├── __init__.py
│   │   ├── base.py             # Observer 基类
│   │   ├── min_max_observer.py # MinMaxObserver
│   │   ├── histogram_observer.py # HistogramObserver
│   │   ├── moving_average_min_max_observer.py # 滑动平均 MinMaxObserver
│   │   ├── percentile_observer.py # 百分位数 Observer
│   │   └── mse_observer.py     # MSE 优化 Observer
│   ├── fake_quant/             # FakeQuant 模块
│   │   ├── __init__.py
│   │   ├── base.py             # FakeQuant 基类
│   │   └── fake_quantize.py    # FakeQuant 实现（支持 LSQ、PACT）
│   ├── quantization/           # 量化核心逻辑
│   │   ├── __init__.py
│   │   ├── model_quantizer.py  # 模型量化器
│   │   └── api.py              # 高级 API
│   ├── utils/                  # 工具模块
│   │   ├── __init__.py
│   │   ├── qconfig.py          # QConfig 配置
│   │   ├── mixed_precision.py  # 混合精度量化
│   │   └── sensitivity_analysis.py # 敏感度分析
│   ├── onnx_export/            # ONNX 导出
│   │   ├── __init__.py
│   │   ├── exporter.py         # 导出器
│   │   ├── engine_adapter.py   # 推理引擎适配器
│   │   └── onnx_optimizer.py   # ONNX 优化器
│   ├── evaluation/             # 量化评估模块
│   │   ├── __init__.py
│   │   ├── evaluator.py        # 量化评估器
│   │   └── metrics.py          # 评估指标（PSNR、SSIM 等）
│   ├── special_models/         # 特殊模型支持
│   │   ├── __init__.py
│   │   └── transformer.py      # Transformer 模型量化策略
│   ├── cli.py                  # 命令行工具
│   └── tests/                  # 测试
├── examples/                   # 示例
│   ├── 01_basic_ptq.py         # 基础 PTQ 示例
│   ├── 02_advanced_qat.py      # 高级 QAT 示例
│   ├── 03_engine_adapter.py    # 推理引擎适配示例
│   ├── 04_sensitivity_analysis.py # 敏感度分析示例
│   ├── 05_complete_workflow.py # 完整工作流示例
│   ├── 06_advanced_mixed_precision_transformer.py # 混合精度 Transformer 示例
│   └── 07_nafnet_ptq_workflow.py # NAFNet 量化示例
├── docs/                       # 文档
│   └── CLI_GUIDE.md            # 命令行工具指南
├── setup.py
├── README.md
├── PROJECT_ANALYSIS.md         # 项目分析文档
└── GITHUB_SETUP_GUIDE.md       # GitHub 上传指南
```

## 快速开始

### 安装

```bash
pip install -e .
```

### PTQ 使用示例

```python
import torch
import torchvision.models as models
from autoquant.utils import get_per_channel_qconfig
from autoquant.quantization.api import prepare, convert, calibrate
from autoquant.onnx_export import ONNXExporter

# 1. 加载模型
model = models.resnet18(pretrained=True)
model.eval()

# 2. 获取量化配置
qconfig = get_per_channel_qconfig(is_symmetric=False)

# 3. 准备 PTQ
model_prepared = prepare(model, qconfig)

# 4. 校准
calibrate(model_prepared, calib_data_loader)

# 5. 转换为量化模型
model_quantized = convert(model_prepared)

# 6. 导出 ONNX
dummy_input = torch.randn(1, 3, 224, 224)
exporter = ONNXExporter()
exporter.export(model_quantized, dummy_input, "model_ptq.onnx")
```

### QAT 使用示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from autoquant.utils import get_lsq_qconfig
from autoquant.quantization.api import prepare_qat, convert
from autoquant.onnx_export import ONNXExporter

# 1. 加载模型
model = models.resnet18(pretrained=True)

# 2. 获取量化配置（使用 LSQ 量化）
qconfig = get_lsq_qconfig()

# 3. 准备 QAT
model_prepared = prepare_qat(model, qconfig)

# 4. QAT 训练
optimizer = optim.SGD(model_prepared.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model_prepared.train()
    optimizer.zero_grad()
    output = model_prepared(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

# 5. 转换为量化模型
model_quantized = convert(model_prepared)

# 6. 导出 ONNX
dummy_input = torch.randn(1, 3, 224, 224)
exporter = ONNXExporter()
exporter.export(model_quantized, dummy_input, "model_qat.onnx")
```

### 敏感度分析和混合精度量化示例

```python
import torch
import torchvision.models as models
from autoquant.utils import get_default_qconfig, SensitivityAnalyzer, MixedPrecisionQuantizer

# 加载模型
model = models.resnet18(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224)

# 1. 敏感度分析
analyzer = SensitivityAnalyzer(model, get_default_qconfig())
sensitivity_scores = analyzer.analyze(dummy_input)

# 2. 混合精度量化
mp_quantizer = MixedPrecisionQuantizer(model)

# 自动配置混合精度
mp_quantizer.auto_config_by_sensitivity(dummy_input, threshold=0.001)

# 准备模型
model_prepared = mp_quantizer.prepare()

# 查看配置摘要
print(mp_quantizer.get_config_summary())
```

### Transformer 模型量化示例

```python
from autoquant.special_models import TransformerQuantizer, get_transformer_qconfig

# 加载 Transformer 模型
model = YourTransformerModel()

# 获取 Transformer 专用量化配置
qconfig = get_transformer_qconfig()

# 使用 Transformer 量化器
quantizer = TransformerQuantizer(model, qconfig)
model_prepared = quantizer.prepare()

# 校准和转换...
```

## 核心组件说明

### Observer

Observer 用于在 PTQ 阶段统计数据分布，计算量化参数。

- `MinMaxObserver`：基于最小最大值的统计，简单高效
- `HistogramObserver`：基于直方图的统计，更准确捕捉数据分布
- `MovingAverageMinMaxObserver`：滑动平均的 MinMax 统计，适用于 QAT
- `PercentileObserver`：基于百分位数的统计，去除极端值影响
- `MSEObserver`：基于最小化 MSE 的统计，精度最高

### FakeQuantize

FakeQuantize 用于在 QAT 阶段模拟量化误差，保留梯度传递。

- `FakeQuantize`：基础的 FakeQuantize 实现，支持 STE
- `LSQFakeQuantize`：基于 LSQ（Learnable Step Size Quantization）的实现
- `PACTFakeQuantize`：基于 PACT（Parameterized Clipping Activation）的实现
- 支持 per-tensor 和 per-channel 量化
- 支持对称和非对称量化

### QConfig

量化配置类，定义 activation 和 weight 的量化策略。

- `get_default_qconfig()`：获取默认配置
- `get_per_channel_qconfig()`：获取按通道量化配置
- `get_per_tensor_qconfig()`：获取按张量量化配置
- `get_lsq_qconfig()`：获取 LSQ 量化配置
- `get_pact_qconfig()`：获取 PACT 量化配置
- `get_histogram_qconfig()`：获取使用 HistogramObserver 的配置
- `get_transformer_qconfig()`：获取 Transformer 专用配置

### 推理引擎适配

为不同的推理引擎提供最优的量化配置：

- `InferenceEngine.TENSORRT`：TensorRT 引擎
- `InferenceEngine.ONNXRUNTIME`：ONNX Runtime 引擎
- `InferenceEngine.OPENVINO`：OpenVINO 引擎
- `InferenceEngine.MNN`：MNN 引擎
- `InferenceEngine.TFLITE`：TFLite 引擎
- `InferenceEngine.COREML`：CoreML 引擎

```python
from autoquant.onnx_export import get_qconfig_for_engine, InferenceEngine

# 获取适用于 TensorRT 的量化配置
qconfig = get_qconfig_for_engine(InferenceEngine.TENSORRT)
```

## 命令行工具

AutoQuant 提供了命令行工具，方便快速进行量化操作：

```bash
# 基本量化
autoquant quantize --model path/to/model.pth --output path/to/quantized.onnx --calib-data path/to/calib_data

# 敏感度分析
autoquant analyze --model path/to/model.pth --output path/to/analysis.json

# ONNX 优化
autoquant optimize --input path/to/model.onnx --output path/to/optimized.onnx

# 引擎信息
autoquant engine-info
```

## 量化评估

使用内置的评估模块评估量化模型的性能：

```python
from autoquant.evaluation import QuantizationEvaluator, compute_psnr, compute_ssim

# 评估 PSNR 和 SSIM
evaluator = QuantizationEvaluator(model, quantized_model)
metrics = evaluator.evaluate(data_loader)
print(f"PSNR: {metrics['psnr']:.4f}")
print(f"SSIM: {metrics['ssim']:.4f}")
```

## 扩展性设计

### 添加自定义 Observer

```python
from autoquant.observer import ObserverBase

class MyObserver(ObserverBase):
    def forward(self, x):
        # 自定义统计逻辑
        pass
    
    def calculate_qparams(self):
        # 自定义量化参数计算
        pass
```

### 添加自定义 FakeQuantize

```python
from autoquant.fake_quant import FakeQuantizeBase

class MyFakeQuantize(FakeQuantizeBase):
    def forward(self, x):
        # 自定义模拟量化逻辑
        pass
```

## 推理引擎兼容性

导出的 ONNX 模型包含 QDQ（QuantizeLinear/DequantizeLinear）节点，兼容：

- TensorRT
- OpenVINO
- ONNX Runtime
- MNN
- TFLite
- CoreML
- 其他支持 QDQ 格式的推理引擎

## 提交记录

### 最近更新

- **2026-03-11**：修复 PTQ 流程中的形状不匹配错误
  - 修复 `FakeQuantize` 类：确保从 observer 中获取正确的参数（dtype、qscheme、ch_axis 等）
  - 修复 `LSQFakeQuantize` 类：添加相同的参数同步逻辑
  - 修复 `PACTFakeQuantize` 类：添加相同的参数同步逻辑
  - 修复 `metrics.py`：修复 SSIM 计算中的 torch.exp 调用错误
  - 验证 NAFNet 模型的 PTQ 流程完全通过

- **2026-03-11**：拆分 observer 模块，将不同类型的 observer 分离到单独文件中
  - 添加 `histogram_observer.py`：实现 HistogramObserver
  - 添加 `moving_average_min_max_observer.py`：实现 MovingAverageMinMaxObserver
  - 添加 `percentile_observer.py`：实现 PercentileObserver
  - 添加 `mse_observer.py`：实现 MSEObserver
  - 更新 `__init__.py`：调整导入路径

- **2026-03-10**：实现 Transformer 模型专用量化策略
  - 添加 `special_models/transformer.py`：实现 TransformerQuantizer
  - 支持 SmoothQuant 量化方法
  - 支持 KV Cache 量化

- **2026-03-09**：添加命令行工具
  - 添加 `cli.py`：实现完整的命令行接口
  - 支持量化、分析、优化等操作

- **2026-03-08**：添加量化评估模块
  - 添加 `evaluation/` 目录：实现量化评估功能
  - 支持 PSNR、SSIM、准确率等评估指标

- **2026-03-07**：完善 ONNX 导出和优化
  - 添加 `onnx_optimizer.py`：实现 ONNX 模型优化
  - 支持与 onnxsim 集成

- **2026-03-06**：实现敏感度分析和混合精度量化
  - 添加 `sensitivity_analysis.py`：实现敏感度分析
  - 增强 `mixed_precision.py`：支持自动混合精度配置

- **2026-03-05**：实现 LSQ 和 PACT 量化方法
  - 添加 `autograd_functions.py`：实现 LSQ 和 PACT 自动梯度函数
  - 更新 `fake_quantize.py`：支持 LSQFakeQuantize 和 PACTFakeQuantize

- **2026-03-04**：初始版本
  - 实现基础的 PTQ 和 QAT 功能
  - 支持 ONNX 导出
  - 支持基本的 Observer 和 FakeQuantize

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！