# AutoQuant - 专业的AI模型量化工具链条

AutoQuant 是一套参考 MQBench 设计的专业模型量化工具链条，支持从 PyTorch 模型到 ONNX 模型（包含 QDQ 操作）的完整量化流程。

## 核心特性

- **自定义 Observer**：完全自定义的 Observer 实现，支持多种数据统计策略
- **自定义 FakeQuant**：完整的 FakeQuantize 实现，支持 STE（直通估计器）
- **PTQ/QAT 支持**：同时支持训练后量化（PTQ）和量化感知训练（QAT）
- **混合精度量化**：支持不同层使用不同的量化精度
- **符号化追踪**：基于 torch.fx 的符号化追踪，减少胶水节点
- **ONNX 导出**：支持导出包含 QDQ 节点的 ONNX 模型，兼容 TensorRT/OpenVINO
- **叶子节点追踪**：支持针对特定叶子节点进行量化（如 NAFNet 的特定层）

## 项目结构

```
autoquant/
├── autoquant/
│   ├── __init__.py
│   ├── core/                   # 核心定义
│   │   ├── __init__.py
│   │   └── dtype.py            # 数据类型和量化方案
│   ├── observer/               # Observer 模块
│   │   ├── __init__.py
│   │   ├── base.py             # Observer 基类
│   │   └── min_max_observer.py # MinMaxObserver
│   ├── fake_quant/             # FakeQuant 模块
│   │   ├── __init__.py
│   │   ├── base.py             # FakeQuant 基类
│   │   └── fake_quantize.py    # FakeQuant 实现
│   ├── quantization/           # 量化核心逻辑
│   │   ├── __init__.py
│   │   ├── model_quantizer.py  # 模型量化器
│   │   └── api.py              # 高级 API
│   ├── utils/                  # 工具模块
│   │   ├── __init__.py
│   │   ├── qconfig.py          # QConfig 配置
│   │   └── mixed_precision.py  # 混合精度量化
│   ├── onnx_export/            # ONNX 导出
│   │   ├── __init__.py
│   │   └── exporter.py         # 导出器
│   ├── examples/               # 示例
│   │   ├── ptq_example.py      # PTQ 示例
│   │   ├── qat_example.py      # QAT 示例
│   │   └── nafnet_example.py   # NAFNet 风格示例
│   └── tests/                  # 测试
├── setup.py
└── README.md
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
from autoquant.quantization import prepare, convert, calibrate
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
from autoquant.utils import get_per_channel_qconfig
from autoquant.quantization import prepare_qat, convert
from autoquant.onnx_export import ONNXExporter

# 1. 加载模型
model = models.resnet18(pretrained=True)

# 2. 获取量化配置
qconfig = get_per_channel_qconfig(is_symmetric=False)

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

### 特定叶子节点量化示例

```python
from autoquant.utils import LayerSelector

# 获取所有叶子模块
leaf_modules = LayerSelector.get_leaf_modules(model)

# 获取特定类型的模块
conv_modules = LayerSelector.get_modules_by_type(model, (nn.Conv2d,))

# 只量化选中的叶子模块
model_prepared = prepare(model, qconfig, leaf_modules=conv_modules)
```

## 核心组件说明

### Observer

Observer 用于在 PTQ 阶段统计数据分布，计算量化参数。

- `MinMaxObserver`：基于最小最大值的统计
- 可扩展：继承 `ObserverBase` 实现自定义 Observer（如 HistogramObserver）

### FakeQuantize

FakeQuantize 用于在 QAT 阶段模拟量化误差，保留梯度传递。

- 支持 STE（直通估计器）
- 支持 per-tensor 和 per-channel 量化
- 支持对称和非对称量化

### QConfig

量化配置类，定义 activation 和 weight 的量化策略。

- `get_default_qconfig()`：获取默认配置
- `get_per_channel_qconfig()`：获取按通道量化配置
- `get_per_tensor_qconfig()`：获取按张量量化配置

### 符号化追踪

使用 `torch.fx` 进行符号化追踪，优化计算图，减少胶水节点：

```python
from autoquant.onnx_export import SymbolicTracer

tracer = SymbolicTracer()
traced_model = tracer.trace(model, dummy_input)
traced_model = tracer.optimize_graph(traced_model)
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
- 其他支持 QDQ 格式的推理引擎

## 后续扩展建议

1. 添加更多 Observer：
   - HistogramObserver（基于直方图的统计）
   - EntropyObserver（基于熵的统计）
   - MovingAverageMinMaxObserver（滑动平均）

2. 添加更多量化策略：
   - 动态量化
   - 权重量化（只量化权重）
   - 激活值量化（只量化激活值）

3. 优化功能：
   - 敏感度分析
   - 自动混合精度选择
   - 量化性能调优工具

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
