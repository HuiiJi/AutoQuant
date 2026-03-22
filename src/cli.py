"""
AutoQuant 命令行工具
提供便捷的命令行接口来执行量化、分析、导出等操作

使用方法：
    autoquant --help
    autoquant quantize --model model.pth --engine tensorrt --output model.onnx
    autoquant analyze --model model.pth --output sensitivity.csv
"""
import argparse
import sys
import os
from typing import Optional


def main():
    """
    主入口函数

    为什么这样设计？
    1. 使用argparse：Python标准库，无需额外依赖，功能强大
    2. 子命令设计：类似git、docker等工具，每个功能一个子命令
    3. 模块化：每个子命令对应独立的函数，易于维护和扩展
    4. 类型提示：使用类型注解提高代码可读性
    """
    # 创建主解析器
    parser = argparse.ArgumentParser(
        prog="autoquant",
        description="AutoQuant - 专业的AI模型量化工具链",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础PTQ量化
  autoquant quantize --model model.pth --engine tensorrt --output quantized.onnx

  # 敏感度分析
  autoquant analyze --model model.pth --output sensitivity.csv

  # 优化ONNX
  autoquant optimize --input model.onnx --output optimized.onnx

  # 查看引擎信息
  autoquant engine-info --engine tensorrt
        """
    )

    # 添加版本选项
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s 1.0.0",
        help="显示版本信息"
    )

    # 创建子解析器 - 这是关键！支持多个子命令
    subparsers = parser.add_subparsers(
        title="子命令",
        description="可用的子命令",
        help="使用 'autoquant <子命令> --help' 查看详细帮助",
        dest="command"  # 存储选择的子命令名称
    )

    # 注册各个子命令
    add_quantize_command(subparsers)
    add_analyze_command(subparsers)
    add_optimize_command(subparsers)
    add_engine_info_command(subparsers)

    # 解析命令行参数
    args = parser.parse_args()

    # 如果没有指定子命令，显示帮助
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # 执行对应的子命令
    execute_command(args)


def add_quantize_command(subparsers):
    """
    添加 'quantize' 子命令

    为什么单独一个函数？
    - 每个子命令的配置独立，代码更清晰
    - 易于测试和维护
    - 可以随时添加新的子命令
    """
    parser_quantize = subparsers.add_parser(
        "quantize",
        help="量化模型",
        description="对PyTorch模型进行PTQ/QAT量化"
    )

    # 必填参数
    parser_quantize.add_argument(
        "--model", "-m",
        required=True,
        type=str,
        help="PyTorch模型路径 (.pth/.pt)"
    )

    parser_quantize.add_argument(
        "--engine", "-e",
        required=True,
        type=str,
        choices=["tensorrt", "onnxruntime", "openvino", "mnn"],
        help="目标推理引擎"
    )

    parser_quantize.add_argument(
        "--output", "-o",
        required=True,
        type=str,
        help="输出ONNX模型路径"
    )

    # 可选参数
    parser_quantize.add_argument(
        "--input-shape",
        type=str,
        default="1,3,224,224",
        help="输入形状，逗号分隔 (默认: 1,3,224,224)"
    )

    parser_quantize.add_argument(
        "--qat",
        action="store_true",
        help="使用QAT而不是PTQ"
    )

    parser_quantize.add_argument(
        "--observer",
        type=str,
        default="minmax",
        choices=["minmax", "histogram", "percentile", "mse", "moving_avg"],
        help="Observer类型 (默认: minmax)"
    )


def add_analyze_command(subparsers):
    """添加 'analyze' 子命令 - 敏感度分析"""
    parser_analyze = subparsers.add_parser(
        "analyze",
        help="分析模型敏感度",
        description="分析各层对量化的敏感度"
    )

    parser_analyze.add_argument(
        "--model", "-m",
        required=True,
        type=str,
        help="PyTorch模型路径"
    )

    parser_analyze.add_argument(
        "--output", "-o",
        type=str,
        help="输出报告路径 (CSV或TXT)"
    )

    parser_analyze.add_argument(
        "--input-shape",
        type=str,
        default="1,3,224,224",
        help="输入形状"
    )

    parser_analyze.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="敏感度阈值"
    )


def add_optimize_command(subparsers):
    """添加 'optimize' 子命令 - ONNX优化"""
    parser_optimize = subparsers.add_parser(
        "optimize",
        help="优化ONNX模型",
        description="优化和清理ONNX模型"
    )

    parser_optimize.add_argument(
        "--input", "-i",
        required=True,
        type=str,
        help="输入ONNX模型路径"
    )

    parser_optimize.add_argument(
        "--output", "-o",
        required=True,
        type=str,
        help="输出ONNX模型路径"
    )

    parser_optimize.add_argument(
        "--passes",
        type=str,
        default="all",
        help="优化passes，逗号分隔 (默认: all)"
    )


def add_engine_info_command(subparsers):
    """添加 'engine-info' 子命令 - 查看引擎信息"""
    parser_engine = subparsers.add_parser(
        "engine-info",
        help="查看推理引擎信息",
        description="显示各推理引擎的最佳配置"
    )

    parser_engine.add_argument(
        "--engine", "-e",
        type=str,
        choices=["tensorrt", "onnxruntime", "openvino", "mnn", "all"],
        default="all",
        help="指定引擎 (默认: all)"
    )


def execute_command(args):
    """
    执行选择的子命令

    这是命令行工具的核心：根据args.command调用对应的功能
    """
    print(f"🚀 AutoQuant 命令行工具")
    print(f"执行命令: {args.command}")

    try:
        if args.command == "quantize":
            execute_quantize(args)
        elif args.command == "analyze":
            execute_analyze(args)
        elif args.command == "optimize":
            execute_optimize(args)
        elif args.command == "engine-info":
            execute_engine_info(args)
        else:
            print(f"错误: 未知命令 '{args.command}'")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def execute_quantize(args):
    """执行量化命令"""
    print("\n📦 开始量化...")
    print(f"  模型: {args.model}")
    print(f"  引擎: {args.engine}")
    print(f"  输出: {args.output}")
    print(f"  模式: {'QAT' if args.qat else 'PTQ'}")
    print(f"  Observer: {args.observer}")

    # 解析输入形状
    input_shape = tuple(map(int, args.input_shape.split(",")))
    print(f"  输入形状: {input_shape}")

    # TODO: 实际的量化逻辑
    print("\n✅ 量化命令框架完成！")
    print("   (实际量化逻辑需要加载模型并调用ModelQuantizer)")


def execute_analyze(args):
    """执行敏感度分析命令"""
    print("\n🔍 开始敏感度分析...")
    print(f"  模型: {args.model}")
    if args.output:
        print(f"  输出: {args.output}")
    print(f"  阈值: {args.threshold}")

    # TODO: 实际的敏感度分析逻辑
    print("\n✅ 敏感度分析命令框架完成！")


def execute_optimize(args):
    """执行ONNX优化命令"""
    print("\n⚡ 开始优化ONNX...")
    print(f"  输入: {args.input}")
    print(f"  输出: {args.output}")
    print(f"  优化Passes: {args.passes}")

    # TODO: 实际的ONNX优化逻辑
    print("\n✅ ONNX优化命令框架完成！")


def execute_engine_info(args):
    """执行引擎信息命令"""
    print("\n📋 推理引擎信息:")

    from autoquant import get_supported_engines, print_engine_info

    if args.engine == "all":
        print_engine_info()
    else:
        print_engine_info(args.engine)


# 这是关键！当脚本被直接运行时调用main()
# 为什么这样写？
# - 当模块被import时不会执行
# - 当作为脚本运行时才会执行
# - 这是Python的标准写法
if __name__ == "__main__":
    main()
