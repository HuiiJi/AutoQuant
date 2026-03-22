"""
ONNX导出模块 - 支持符号化追踪和QDQ节点导出

Author: jihui
Date: 2026-03-13
Desc:
    - 使用 torch.fake_quantize_per_tensor_affine/channel
    - PyTorch 会自动导出标准 QuantizeLinear/DequantizeLinear 节点
    - 不需要手动转换！
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
        model.eval()

        if concrete_args is None:
            traced_model = symbolic_trace(model)
        else:
            traced_model = symbolic_trace(model, concrete_args=concrete_args)

        return traced_model

    @staticmethod
    def optimize_graph(traced_model: GraphModule) -> GraphModule:
        traced_model = SymbolicTracer._fold_constants(traced_model)
        traced_model = SymbolicTracer._eliminate_dead_code(traced_model)
        traced_model = SymbolicTracer._fuse_operations(traced_model)
        return traced_model

    @staticmethod
    def _fold_constants(model: GraphModule) -> GraphModule:
        return model

    @staticmethod
    def _eliminate_dead_code(model: GraphModule) -> GraphModule:
        return model

    @staticmethod
    def _fuse_operations(model: GraphModule) -> GraphModule:
        return model


class ONNXExporter:
    """
    ONNX导出器 - 导出包含QDQ节点的ONNX模型

    关键设计：
    - 使用 torch.fake_quantize_per_tensor_affine/channel 会自动导出 QDQ 节点
    - PyTorch ONNX 导出器自动处理，不需要手动转换
    """

    @staticmethod
    def export(
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
        opset_version: int = 18,
        use_symbolic_trace: bool = True,
        do_constant_folding: bool = True,
        verbose: bool = False,
        optimize: bool = True,
        optimization_passes: Optional[List[str]] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ):
        model.eval()

        # 检查是否是量化模型
        is_quantized_model = False
        for name, module in model.named_modules():
            if 'FakeQuantize' in type(module).__name__ or 'Quantizable' in type(module).__name__:
                is_quantized_model = True
                break

        if not is_quantized_model and use_symbolic_trace:
            tracer = SymbolicTracer()
            model = tracer.trace(model, dummy_input)
            model = tracer.optimize_graph(model)

        # 默认输入输出名称
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']

        # 默认 dynamic axes
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }

        # 导出 ONNX - 使用旧版导出器以支持 fake_quantize
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            verbose=verbose,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_params=True,
            keep_initializers_as_inputs=False,
            # 使用旧版导出器以避免 torch.export 的问题
            dynamo=False,
            fallback=True,
        )

        print(f"✅ 模型已成功导出到: {output_path}")

        # 验证 ONNX
        ONNXExporter.validate_onnx(output_path)

        # 优化 ONNX
        if optimize:
            try:
                from autoquant.onnx_export.onnx_optimizer import optimize_onnx
                print("\n🔧 开始优化 ONNX 模型...")
                optimized_model = optimize_onnx(output_path, output_path, passes=optimization_passes, verbose=verbose)
                print("✅ ONNX 模型优化完成！")
            except Exception as e:
                print(f"⚠️  ONNX 优化跳过: {e}")

        # 检查是否有 QDQ 节点
        if is_quantized_model:
            has_qdq = ONNXExporter.has_qdq_nodes(output_path)
            if has_qdq:
                print("\n✅ 检测到标准 QDQ 节点！")
            else:
                print("\n⚠️  未检测到 QDQ 节点！请确保使用 PTQFakeQuantize！")

    @staticmethod
    def validate_onnx(onnx_path: str):
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        print("✅ ONNX 模型验证通过！")
        print(f"   ONNX 版本: {model_onnx.opset_import[0].version}")
        print(f"   图输入: {[input.name for input in model_onnx.graph.input]}")
        print(f"   图输出: {[output.name for output in model_onnx.graph.output]}")
        print(f"   节点数量: {len(model_onnx.graph.node)}")

    @staticmethod
    def has_qdq_nodes(onnx_path: str) -> bool:
        model_onnx = onnx.load(onnx_path)
        qdq_ops = {'QuantizeLinear', 'DequantizeLinear'}
        qdq_count = 0
        for node in model_onnx.graph.node:
            if node.op_type in qdq_ops:
                qdq_count += 1
        if qdq_count > 0:
            print(f"   QDQ 节点数量: {qdq_count}")
            return True
        return False
