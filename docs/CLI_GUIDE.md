# AutoQuant 命令行工具 (CLI) 指南

## 为什么这样写命令行工具？

### 核心设计理念

```python
# autoquant/cli.py 的设计模式
├─ 使用 argparse (Python标准库)
├─ 子命令设计 (类似 git/docker)
├─ 模块化函数组织
└─ 类型注解 + 详细文档
```

---

## 1. 为什么用 `argparse`？

### 优势对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **argparse** (我们用的) | Python标准库、功能强大、自动生成帮助 | 代码稍多 |
| click | 装饰器语法、简洁 | 需要额外安装 |
| typer | 类型注解、现代 | 需要额外安装 |

### 为什么选argparse？
1. **零依赖** - 用户安装即用，不需要 `pip install click`
2. **面试加分** - 展示你掌握Python标准库
3. **功能完整** - 支持子命令、类型检查、自动帮助

---

## 2. 子命令设计详解

### 代码结构

```python
# 1. 创建主解析器
parser = argparse.ArgumentParser(prog="autoquant")

# 2. 创建子解析器容器
subparsers = parser.add_subparsers(dest="command")

# 3. 为每个功能添加子命令
add_quantize_command(subparsers)    # autoquant quantize ...
add_analyze_command(subparsers)      # autoquant analyze ...
add_optimize_command(subparsers)     # autoquant optimize ...
```

### 为什么这样设计？

**类比git命令：**
```bash
git commit ...    # 子命令 commit
git push ...      # 子命令 push
git pull ...      # 子命令 pull

# 我们的：
autoquant quantize ...    # 子命令 quantize
autoquant analyze ...     # 子命令 analyze
autoquant optimize ...    # 子命令 optimize
```

**好处：**
1. **清晰** - 每个功能独立，不会混淆
2. **易扩展** - 添加新功能只需加一个子命令
3. **专业** - 类似业界主流工具的设计

---

## 3. 关键代码解释

### 3.1 `if __name__ == "__main__":`

```python
if __name__ == "__main__":
    main()
```

**为什么必须要有这个？**

```python
# 场景1: 作为模块导入
import autoquant.cli
# 不会执行 main()

# 场景2: 作为脚本运行
python autoquant/cli.py
# __name__ 变成 "__main__"，执行 main()
```

**这是Python的标准模式！**

---

### 3.2 `entry_points` (setup.py中)

```python
entry_points={
    "console_scripts": [
        "autoquant=autoquant.cli:main",
    ],
}
```

**这是魔法！它的作用：**

```bash
# 安装前
python autoquant/cli.py --help

# 安装后 (pip install -e .)
autoquant --help    # ← 直接用！
```

**工作原理：**
1. pip安装时，在Python的Scripts目录创建一个 `autoquant` 可执行文件
2. 这个文件调用 `autoquant.cli:main()`
3. 用户可以在任何地方直接用 `autoquant` 命令

---

### 3.3 参数类型检查

```python
parser_quantize.add_argument(
    "--engine", "-e",
    required=True,
    type=str,
    choices=["tensorrt", "onnxruntime", "openvino", "mnn"],  # ← 关键！
    help="目标推理引擎"
)
```

**`choices` 的作用：**
- 自动验证输入，不在列表里会报错
- 自动在帮助里显示可选值
- 防止用户输入错误

---

## 4. 完整使用示例

### 安装（开发模式）

```bash
cd autoquant/
pip install -e .
# -e 表示 editable mode，修改代码不用重新安装
```

### 查看帮助

```bash
# 总帮助
autoquant --help

# 查看某个子命令的帮助
autoquant quantize --help
autoquant analyze --help
```

### 常用命令

```bash
# 1. 查看引擎信息
autoquant engine-info --engine tensorrt

# 2. 量化模型
autoquant quantize \
  --model model.pth \
  --engine tensorrt \
  --output quantized.onnx \
  --input-shape 1,3,224,224 \
  --observer histogram

# 3. 敏感度分析
autoquant analyze \
  --model model.pth \
  --output sensitivity.csv \
  --threshold 0.001

# 4. 优化ONNX
autoquant optimize \
  --input model.onnx \
  --output optimized.onnx
```

---

## 5. 扩展：添加新的子命令

假设我们要添加一个 `export` 子命令：

```python
# 步骤1: 在cli.py中添加函数
def add_export_command(subparsers):
    parser_export = subparsers.add_parser(
        "export",
        help="导出模型"
    )
    parser_export.add_argument("--model", required=True)
    parser_export.add_argument("--format", choices=["onnx", "torchscript"])

# 步骤2: 在main()中注册
subparsers = parser.add_subparsers(...)
add_quantize_command(subparsers)
add_analyze_command(subparsers)
add_export_command(subparsers)  # ← 新增

# 步骤3: 在execute_command中添加
def execute_command(args):
    if args.command == "quantize":
        execute_quantize(args)
    elif args.command == "export":      # ← 新增
        execute_export(args)
```

就是这么简单！

---

## 6. 总结

### CLI设计的关键点

1. ✅ **用argparse** - 标准库，零依赖
2. ✅ **子命令设计** - 清晰、专业、易扩展
3. ✅ **entry_points** - 让用户直接用 `autoquant` 命令
4. ✅ **`if __name__ == "__main__":`** - Python标准模式
5. ✅ **choices参数** - 自动验证输入

### 面试时怎么说？

> "我用Python标准库argparse实现了命令行工具，采用了子命令设计模式，
> 类似git的使用方式。通过setup.py的entry_points配置，用户安装后
> 可以直接用autoquant命令。这种设计模块化好，易于扩展新功能。"
