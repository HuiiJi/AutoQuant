<div align="center">
  <h1>🚀 AutoQuant</h1>
  <p><strong>专业的 AI 模型量化工具链条</strong></p>

  <p>
    <a href="https://github.com/HuiiJi/autoquant">
      <img src="https://img.shields.io/badge/GitHub-HuiiJi%2Fautoquant-blue?style=flat-square&logo=github" alt="GitHub">
    </a>
    <a href="https://github.com/HuiiJi/autoquant/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
    </a>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" alt="Python">
    </a>
    <a href="https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square&logo=pytorch" alt="PyTorch">
    </a>
  </p>
</div>

***

## 📋 简介

AutoQuant 是一套专业的模型量化工具链，专为AI摄影图像美化模型设计，支持从 PyTorch 模型到 ONNX 模型（包含 QDQ 操作），适配特定推理引擎的的完整量化流程。

## ✨ 核心特性

| 特性                   | 说明                                              |
| -------------------- | ----------------------------------------------- |
| 🔧 **自定义 Observer**  | 完全自定义的 Observer 实现，支持 MinMax、Histogram 等多种统计策略  |
| 🎯 **自定义 FakeQuant** | 完整的 FakeQuantize 实现，支持 STE（直通估计器）、LSQ 等高级量化方法   |
| 📊 **PTQ/QAT 支持**    | 同时支持训练后量化（PTQ）和量化感知训练（QAT）                      |
| ⚖️ **混合精度量化**        | 结合敏感度分析的智能混合精度量化                                |
| 📦 **ONNX 导出**       | 支持导出包含 QDQ 节点的 ONNX 模型，兼容 TensorRT/ONNX Runtime |
| 🔍 **敏感度分析**         | 自动分析各层的量化敏感度，辅助混合精度配置                           |
| 🏭 **引擎最佳配置**        | 直接绑定 TensorRT 和 ONNX Runtime 的最佳精度方案            |

***

## 📁 项目结构

```
autoquant/
├── src/                       # 核心源码
│   ├── __init__.py
│   ├── cli.py                 # 命令行工具
│   ├── core/                   # 核心定义
│   │   ├── __init__.py
│   │   ├── dtype.py            # 数据类型和量化方案
│   │   └── autograd_functions.py # 自动梯度函数（STE、LSQ）
│   ├── observer/               # Observer 模块
│   │   ├── __init__.py
│   │   ├── base.py             # Observer 基类
│   │   ├── min_max_observer.py # MinMaxObserver
│   │   ├── ema_min_max_observer.py # EMA 版 MinMaxObserver
│   │   ├── histogram_observer.py # HistogramObserver
│   │   ├── percentile_observer.py # PercentileObserver
│   │   └── mse_observer.py    # MSEObserver
│   ├── fake_quant/             # FakeQuant 模块
│   │   ├── __init__.py
│   │   ├── base.py             # FakeQuant 基类
│   │   ├── ptq.py              # PTQFakeQuantize
│   │   ├── lsq.py              # LSQFakeQuantize
│   │   └── pact.py             # PACTFakeQuantize
│   ├── quantization/           # 量化核心逻辑
│   │   ├── __init__.py
│   │   ├── model_quantizer.py  # 模型量化器
│   │   └── api.py              # 高级 API
│   ├── utils/                  # 工具模块
│   │   ├── __init__.py
│   │   ├── qconfig.py          # QConfig 配置（TRT/ORT 最佳方案）
│   │   ├── sensitivity_analysis.py # 敏感度分析
│   │   ├── quantizable_ops.py  # 可量化操作工具
│   │   └── mixed_precision.py   # 混合精度工具
│   ├── onnx_export/            # ONNX 导出
│   │   ├── __init__.py
│   │   ├── exporter.py         # 导出器
│   │   ├── onnx_optimizer.py   # ONNX 优化器
│   │   └── engine_adapter.py   # 引擎适配器（TRT/ORT）
│   ├── evaluation/             # 量化评估模块
│   │   ├── __init__.py
│   │   ├── evaluator.py        # 量化评估器
│   │   └── metrics.py          # 评估指标（PSNR、SSIM 等）
│   └── models/                 # 特殊模型支持
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

***

## 🚀 快速开始

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

***

## 🎯 核心组件说明

### QConfig - 引擎最佳配置

直接绑定 TensorRT 和 ONNX Runtime 的最佳精度方案，无需手动设计：

| 函数                      | 说明                  |
| ----------------------- | ------------------- |
| `get_trt_qconfig()`     | TensorRT 最佳精度方案     |
| `get_ort_qconfig()`     | ONNX Runtime 最佳精度方案 |
| `get_default_qconfig()` | 默认使用 TRT 方案         |
| `get_lsq_qconfig()`     | QAT 专用（LSQ）         |

**TensorRT 最佳实践**：

- Activation: `PER_TENSOR_SYMMETRIC` + `MinMaxObserver`
- Weight: `PER_CHANNEL_SYMMETRIC` + `MinMaxObserver`

**ONNX Runtime 最佳实践**：

- Activation: `PER_TENSOR_AFFINE` + `HistogramObserver`
- Weight: `PER_CHANNEL_AFFINE` + `MinMaxObserver`

### Observer

Observer 用于在 PTQ 阶段统计数据分布，计算量化参数：

| Observer            | 说明                 |
| ------------------- | ------------------ |
| `MinMaxObserver`    | 基于最小最大值的统计，简单高效    |
| `HistogramObserver` | 基于直方图的统计，更准确捕捉数据分布 |

### FakeQuantize

FakeQuantize 用于在 QAT 阶段模拟量化误差，保留梯度传递：

| FakeQuantize      | 说明                                          |
| ----------------- | ------------------------------------------- |
| `PTQFakeQuantize` | PTQ 专用的 FakeQuantize 实现                     |
| `LSQFakeQuantize` | 基于 LSQ（Learnable Step Size Quantization）的实现 |

***

## 🔍 敏感度分析

敏感度分析帮助你识别哪些层对量化最敏感，从而进行智能的混合精度量化：

```python
from autoquant import SensitivityAnalyzer, get_default_qconfig

analyzer = SensitivityAnalyzer(model, get_default_qconfig())
sensitivity_scores = analyzer.analyze(dummy_input, calib_data=calib_data)

# 生成报表和图表
analyzer.save_results("sensitivity_results", top_n_percent=10.0)

# 获取推荐跳过的层
quantizable_layers, skip_layers = analyzer.get_recommended_layers(top_n_percent=10.0)
```

**正确的敏感度分析流程**：

1. 基准 1：原始模型（全浮点）- 最佳情况
2. 基准 2：全部量化 - 最差情况
3. 对每个层：只跳过这一层，其他都量化
4. 敏感度分数 = (跳过该层后的改善) / (全部量化的总误差)

***

## 📦 推理引擎兼容性

导出的 ONNX 模型包含 QDQ（QuantizeLinear/DequantizeLinear）节点，兼容：

- ✅ TensorRT
- ✅ ONNX Runtime
- ✅ 其他支持 QDQ 格式的推理引擎

***

## 📝 更新记录

### 2026-03-15（重大修复）

- 🔴 **致命修复：校准阶段只统计，不量化** - `QuantizableModule` 在校准阶段不再对 weight 做 fake quant，确保统计的是原始数据分布
- 🔴 **致命修复：Weight 只统计一次** - 第一个样本时统计 weight，之后保持原始状态
- 🟡 **修复 ObserverBase** - 改用 property 包装统计变量，避免直接 register_buffer(None) 导致的 PyTorch 错误
- 🟡 **修复所有 Observer** - 添加 `self.enabled` 检查（HistogramObserver、PercentileObserver、MSEObserver 都缺少）
- 🟡 **修复 HistogramObserver** - 移除 `.item()` 调用，避免 GPU/CPU 数据迁移问题；改进初始化逻辑
- 🟡 **修复 FakeQuantBase** - 不直接 register_buffer(None)，有值时才注册为 buffer
- 🟢 **新增 ptq() 一键函数** - 完整封装 `prepare → calibrate → convert` 三步
- 🟢 **完善 convert() API** - 支持 `permanently_quantize_weight` 参数
- 🟢 **优化 ONNXExporter** - 移除无效的手动 QDQ 转换逻辑（PyTorch 会自动导出标准 QDQ 节点）
- 🧪 **新增完整测试** - `tests/test_complete_ptq.py` 验证完整 PTQ 流程
- ✅ **验证通过** - 核心流程测试 100% 通过，平均绝对误差仅 0.001

### 2026-03-13

- 🚀 **重构项目**：优化敏感度分析，使用真正的 PTQ 流程
- 🎯 **QConfig 绑定引擎最佳方案**：`get_trt_qconfig()` 和 `get_ort_qconfig()`
- 🧹 **简化 examples**：只保留 PTQ、QAT、引擎适配、混合精度 4 个示例
- 📦 **清理项目结构**：删除不必要的文件和 Transformer 相关代码
- 🐛 **修复 HistogramObserver**：解决重复 register\_buffer 问题

### 2026-03-12

- 📁 **重构项目目录结构**，使用 `src` 目录
- 🔧 **完善 PTQ 流程**：完整的 `prepare → calibrate → convert` 三步曲
- ✅ **验证 NAFNet 模型**：PTQ 流程完全通过

### 2026-03-11

- 🐛 **修复 PTQ 流程中的形状不匹配错误**
- 🔧 **修复 SSIM 计算中的 torch.exp 调用错误**
- ✅ **验证 NAFNet 模型的 PTQ 流程完全通过**
- 📦 **拆分 observer 模块**，将不同类型的 observer 分离到单独文件中

### 2026-03-10

- 🔧 **实现 Transformer 模型专用量化策略**
- 📦 **添加 SmoothQuant 量化方法支持**
- 🎯 **支持 KV Cache 量化**

### 2026-03-09

- 🖥️ **添加命令行工具**，支持量化、分析、优化等操作

### 2026-03-08

- 📊 **添加量化评估模块**，支持 PSNR、SSIM、准确率等评估指标

### 2026-03-07

- 📦 **完善 ONNX 导出和优化**，支持与 onnxsim 集成

### 2026-03-06

- 🔍 **实现敏感度分析和混合精度量化**

### 2026-03-05

- 🎯 **实现 LSQ 和 PACT 量化方法**

### 2026-03-04

- 🎉 **初始版本**，实现基础的 PTQ 和 QAT 功能

***

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

***

<div align="center">
  <p>Made with ❤️ by <a href="https://github.com/HuiiJi">HuiiJi</a></p>
</div>
