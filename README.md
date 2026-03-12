# AutoQuant - 专业的AI模型量化工具链条

AutoQuant 是一套专业模型量化工具链条，支持从 PyTorch 模型到 ONNX 模型（包含 QDQ 操作）的完整量化流程。

## 核心特性

- **自定义 Observer**：完全自定义的 Observer 实现，支持多种数据统计策略
- **自定义 FakeQuant**：完整的 FakeQuantize 实现，支持 STE（直通估计器）、LSQ 等高级量化方法
- **PTQ/QAT 支持**：同时支持训练后量化（PTQ）和量化感知训练（QAT）
- **混合精度量化**：结合敏感度分析的混合精度量化
- **ONNX 导出**：支持导出包含 QDQ 节点的 ONNX 模型，兼容 TensorRT/ONNX Runtime
- **敏感度分析**：自动分析各层的量化敏感度，辅助混合精度配置
- **引擎最佳配置**：直接绑定 TensorRT 和 ONNX Runtime 的最佳精度方案

## 项目结构

```
autoquant/
├── src/                       # 核心源码
│   ├── __init__.py
│   ├── core/                   # 核心定义
│   │   ├── __init__.py
│   │   ├── dtype.py            # 数据类型和量化方案
│   │   └── autograd_functions.py # 自动梯度函数（STE、LSQ）
│   ├── observer/               # Observer 模块
│   │   ├── __init__.py
│   │   ├── base.py             # Observer 基类
│   │   ├── min_max_observer.py # MinMaxObserver
│   │   └── histogram_observer.py # HistogramObserver
│   ├── fake_quant/             # FakeQuant 模块
│   │   ├── __init__.py
│   │   ├── base.py             # FakeQuant 基类
│   │   ├── ptq.py              # PTQFakeQuantize
│   │   └── lsq.py              # LSQFakeQuantize
│   ├── quantization/           # 量化核心逻辑
│   │   ├── __init__.py
│   │   ├── model_quantizer.py  # 模型量化器
│   │   └── api.py              # 高级 API
│   ├── utils/                  # 工具模块
│   │   ├── __init__.py
│   │   ├── qconfig.py          # QConfig 配置（TRT/ORT 最佳方案）
│   │   └── sensitivity_analysis.py # 敏感度分析
│   ├── onnx_export/            # ONNX 导出
│   │   ├── __init__.py
│   │   ├── exporter.py         # 导出器
│   │   └── onnx_optimizer.py   # ONNX 优化器
│   ├── evaluation/             # 量化评估模块
│   │   ├── __init__.py
│   │   ├── evaluator.py        # 量化评估器
│   │   └── metrics.py          # 评估指标（PSNR、SSIM 等）
│   └── special_models/         # 特殊模型支持
│       ├── __init__.py
│       └── nafnet.py           # NAFNet 模型支持
├── tests/                      # 测试
│   └── test_environment.py     # 环境测试
├── examples/                   # 示例
│   ├── 01_basic_ptq.py         # 基础 PTQ + 敏感度分析
│   ├── 02_advanced_qat.py      # QAT 量化感知训练（LSQ）
│   ├── 03_engine_adapter.py    # 引擎适配（TRT/ORT）
│   └── 04_mixed_precision.py   # 混合精度量化
├── setup.py                    # 包配置
└── README.md                   # 项目说明
```

## 快速开始

### 安装

```bash
pip install -e .
```

### PTQ 使用示例（TensorRT 最佳配置）

```python
import torch
from autoquant import (
    ModelQuantizer,
    get_trt_qconfig,
    get_default_qconfig,
    SensitivityAnalyzer,
)

# 1. 加载模型
model = YourModel()
model.eval()
dummy_input = torch.randn(1, 3, 64, 64)

# 2. 获取量化配置（默认使用 TensorRT 最佳方案）
qconfig = get_default_qconfig()  # 或 get_trt_qconfig() / get_ort_qconfig()

# 3. 敏感度分析（可选，用于混合精度）
analyzer = SensitivityAnalyzer(model, qconfig)
sensitivity_scores = analyzer.analyze(dummy_input, calib_data=[dummy_input])
quantizable_layers, skip_layers = analyzer.get_recommended_layers(top_n_percent=10.0)

# 4. 准备 PTQ
quantizer = ModelQuantizer(model, qconfig)
prepared_model = quantizer.prepare(skip_layers=set(skip_layers))

# 5. 校准
quantizer.calibrate([dummy_input])

# 6. 转换为量化模型
quantized_model = quantizer.convert()

# 7. 导出 ONNX
torch.onnx.export(
    quantized_model,
    dummy_input,
    "model_ptq.onnx",
    opset_version=13,
    dynamo=False
)
```

### QAT 使用示例（LSQ）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from autoquant import ModelQuantizer, get_lsq_qconfig

# 1. 加载模型
model = YourModel()

# 2. 获取量化配置（使用 LSQ 量化）
qconfig = get_lsq_qconfig()

# 3. 准备 QAT
quantizer = ModelQuantizer(model, qconfig)
prepared_model = quantizer.prepare()

# 4. QAT 训练
optimizer = optim.SGD(prepared_model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    prepared_model.train()
    optimizer.zero_grad()
    output = prepared_model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

# 5. 转换为量化模型
quantized_model = quantizer.convert()
```

## 核心组件说明

### QConfig - 引擎最佳配置

直接绑定 TensorRT 和 ONNX Runtime 的最佳精度方案：

| 函数 | 说明 |
|------|------|
| `get_trt_qconfig()` | TensorRT 最佳精度方案 |
| `get_ort_qconfig()` | ONNX Runtime 最佳精度方案 |
| `get_default_qconfig()` | 默认使用 TRT 方案 |
| `get_lsq_qconfig()` | QAT 专用（LSQ） |

**TensorRT 最佳实践**：
- Activation: `PER_TENSOR_SYMMETRIC` + `MinMaxObserver`
- Weight: `PER_CHANNEL_SYMMETRIC` + `MinMaxObserver`

**ONNX Runtime 最佳实践**：
- Activation: `PER_TENSOR_AFFINE` + `HistogramObserver`
- Weight: `PER_CHANNEL_AFFINE` + `MinMaxObserver`

### Observer

Observer 用于在 PTQ 阶段统计数据分布，计算量化参数。

- `MinMaxObserver`：基于最小最大值的统计，简单高效
- `HistogramObserver`：基于直方图的统计，更准确捕捉数据分布

### FakeQuantize

FakeQuantize 用于在 QAT 阶段模拟量化误差，保留梯度传递。

- `PTQFakeQuantize`：PTQ 专用的 FakeQuantize 实现
- `LSQFakeQuantize`：基于 LSQ（Learnable Step Size Quantization）的实现

## 推理引擎兼容性

导出的 ONNX 模型包含 QDQ（QuantizeLinear/DequantizeLinear）节点，兼容：

- TensorRT
- ONNX Runtime
- 其他支持 QDQ 格式的推理引擎

## 更新记录

### 2026-03-13
- 重新设计 QConfig，直接绑定 TensorRT 和 ONNX Runtime 最佳方案
- 优化敏感度分析脚本，使用真正的 PTQ 流程
- 简化 examples，删除 Transformer 相关
- 清理项目结构，删除不必要的文件

### 2026-03-12
- 重构项目目录结构，使用 src 目录
- 完善 PTQ 流程（prepare → calibrate → convert）

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
