"""
ONNX导出模块 - 支持符号化追踪和QDQ节点导出
"""
import torch
import torch.nn as nn
import onnx
from typing import Optional, Dict, Any, List
from torch.fx import symbolic_trace, GraphModule


class SymbolicTracer:
    """
    符号化追踪器 - 用于获取干净的计算图
    """

    @staticmethod
    def trace(
        model: nn.Module,
        dummy_input: torch.Tensor,
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> GraphModule:
        """
        使用torch.fx进行符号化追踪
        
        Args:
            model: 待追踪的模型
            dummy_input: 用于追踪的dummy输入
            concrete_args: 具体参数（可选）
        
        Returns:
            追踪后的GraphModule
        """
        # 设置为eval模式
        model.eval()
        
        # 符号化追踪
        if concrete_args is None:
            traced_model = symbolic_trace(model)
        else:
            traced_model = symbolic_trace(model, concrete_args=concrete_args)
        
        return traced_model

    @staticmethod
    def optimize_graph(traced_model: GraphModule) -> GraphModule:
        """
        优化计算图，减少胶水节点
        
        Args:
            traced_model: 追踪后的模型
        
        Returns:
            优化后的模型
        """
        # 1. 常量折叠
        traced_model = SymbolicTracer._fold_constants(traced_model)
        
        # 2. 死代码消除
        traced_model = SymbolicTracer._eliminate_dead_code(traced_model)
        
        # 3. 操作融合
        traced_model = SymbolicTracer._fuse_operations(traced_model)
        
        return traced_model
    
    @staticmethod
    def _fold_constants(model: GraphModule) -> GraphModule:
        """
        常量折叠优化
        """
        # 这里可以实现基于 torch.fx 的常量折叠
        # 简化实现，实际可以使用更复杂的算法
        return model
    
    @staticmethod
    def _eliminate_dead_code(model: GraphModule) -> GraphModule:
        """
        死代码消除
        """
        # 这里可以实现基于 torch.fx 的死代码消除
        # 简化实现，实际可以使用更复杂的算法
        return model
    
    @staticmethod
    def _fuse_operations(model: GraphModule) -> GraphModule:
        """
        操作融合
        比如将 conv + relu 融合为一个操作
        """
        # 这里可以实现基于 torch.fx 的操作融合
        # 简化实现，实际可以使用更复杂的算法
        return model


class ONNXExporter:
    """
    ONNX导出器 - 导出包含QDQ节点的ONNX模型
    """

    @staticmethod
    def export(
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
        opset_version: int = 13,
        use_symbolic_trace: bool = True,
        do_constant_folding: bool = True,
        verbose: bool = False,
        optimize: bool = True,
        optimization_passes: Optional[List[str]] = None,
    ):
        """
        导出模型为ONNX格式
        
        Args:
            model: 待导出的模型
            dummy_input: dummy输入
            output_path: 输出路径
            opset_version: ONNX opset版本
            use_symbolic_trace: 是否使用符号化追踪
            do_constant_folding: 是否进行常量折叠
            verbose: 是否打印详细信息
            optimize: 是否在导出后优化模型
            optimization_passes: 优化pass列表
        """
        model.eval()
        
        # 检查模型是否为量化模型（包含 FakeQuantize 或 QuantizableModule）
        is_quantized_model = False
        for name, module in model.named_modules():
            if 'FakeQuantize' in type(module).__name__ or 'Quantizable' in type(module).__name__:
                is_quantized_model = True
                break
        
        # 对于量化模型，不使用符号化追踪，因为可能会与 Proxy 对象不兼容
        if use_symbolic_trace and not is_quantized_model:
            tracer = SymbolicTracer()
            model = tracer.trace(model, dummy_input)
            model = tracer.optimize_graph(model)
        elif is_quantized_model:
            print("检测到量化模型，跳过符号化追踪...")
        
        # 导出ONNX
        # 对于量化模型，使用旧版的捕获策略，避免torch.export的问题
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            verbose=verbose,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            # 对于量化模型，禁用dynamo捕获策略
            dynamo=False if is_quantized_model else None,
        )
        
        print(f"模型已成功导出到: {output_path}")
        
        # 验证ONNX模型
        ONNXExporter.validate_onnx(output_path)
        
        # 优化模型
        if optimize:
            from autoquant.onnx_export.onnx_optimizer import optimize_onnx
            print("\n开始优化ONNX模型...")
            optimized_model = optimize_onnx(output_path, output_path, passes=optimization_passes, verbose=verbose)
            print("ONNX模型优化完成！")

    @staticmethod
    def validate_onnx(onnx_path: str):
        """
        验证ONNX模型的正确性
        
        Args:
            onnx_path: ONNX模型路径
        """
        # 加载ONNX模型
        model_onnx = onnx.load(onnx_path)
        
        # 检查模型
        onnx.checker.check_model(model_onnx)
        
        # 打印模型信息
        print("ONNX模型验证通过！")
        print(f"ONNX版本: {model_onnx.opset_import[0].version}")
        print(f"图输入: {[input.name for input in model_onnx.graph.input]}")
        print(f"图输出: {[output.name for output in model_onnx.graph.output]}")
        print(f"节点数量: {len(model_onnx.graph.node)}")

    @staticmethod
    def has_qdq_nodes(onnx_path: str) -> bool:
        """
        检查ONNX模型是否包含QDQ节点
        
        Args:
            onnx_path: ONNX模型路径
        
        Returns:
            是否包含QDQ节点
        """
        model_onnx = onnx.load(onnx_path)
        
        qdq_ops = {'QuantizeLinear', 'DequantizeLinear'}
        for node in model_onnx.graph.node:
            if node.op_type in qdq_ops:
                return True
        
        return False
