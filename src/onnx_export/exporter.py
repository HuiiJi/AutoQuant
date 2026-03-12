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
    ):
        model.eval()
        
        is_quantized_model = False
        for name, module in model.named_modules():
            if 'FakeQuantize' in type(module).__name__ or 'Quantizable' in type(module).__name__:
                is_quantized_model = True
                break
        
        if not is_quantized_model and use_symbolic_trace:
            tracer = SymbolicTracer()
            model = tracer.trace(model, dummy_input)
            model = tracer.optimize_graph(model)
        
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
            }
        )
        
        print(f"模型已成功导出到: {output_path}")
        
        ONNXExporter.validate_onnx(output_path)
        
        if optimize:
            from autoquant.onnx_export.onnx_optimizer import optimize_onnx
            print("\n开始优化ONNX模型...")
            optimized_model = optimize_onnx(output_path, output_path, passes=optimization_passes, verbose=verbose)
            print("ONNX模型优化完成！")
        
        if is_quantized_model:
            print("\n开始将模型转换为QDQ节点...")
            ONNXExporter.convert_to_qdq(output_path)
            print("QDQ节点转换完成！")

    @staticmethod
    def validate_onnx(onnx_path: str):
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        print("ONNX模型验证通过！")
        print(f"ONNX版本: {model_onnx.opset_import[0].version}")
        print(f"图输入: {[input.name for input in model_onnx.graph.input]}")
        print(f"图输出: {[output.name for output in model_onnx.graph.output]}")
        print(f"节点数量: {len(model_onnx.graph.node)}")

    @staticmethod
    def has_qdq_nodes(onnx_path: str) -> bool:
        model_onnx = onnx.load(onnx_path)
        qdq_ops = {'QuantizeLinear', 'DequantizeLinear'}
        for node in model_onnx.graph.node:
            if node.op_type in qdq_ops:
                return True
        return False
    
    @staticmethod
    def convert_to_qdq(onnx_path: str, output_path: str = None):
        """
        将ONNX模型中的div、round、clip操作链转换为QDQ节点
        """
        if output_path is None:
            output_path = onnx_path
        
        model_onnx = onnx.load(onnx_path)
        
        new_graph = onnx.helper.make_graph(
            [],
            model_onnx.graph.name,
            model_onnx.graph.input,
            model_onnx.graph.output,
            list(model_onnx.graph.initializer)
        )
        
        zero_point_tensor = onnx.helper.make_tensor(
            name="zero_point_constant",
            data_type=onnx.TensorProto.INT8,
            dims=[],
            vals=[0]
        )
        new_graph.initializer.append(zero_point_tensor)
        
        i = 0
        qdq_pairs_created = 0
        nodes_skipped = 0
        
        # 收集所有初始化器名称
        init_names = {init.name for init in model_onnx.graph.initializer}
        
        while i < len(model_onnx.graph.node):
            node = model_onnx.graph.node[i]
            
            if node.op_type == 'Div' and len(node.input) == 2:
                scale_input = node.input[1]
                is_scale_constant = scale_input in init_names
                
                if is_scale_constant:
                    found_chain = False
                    chain_nodes = [node]
                    next_output = node.output[0]
                    
                    # 查找操作链：Div -> Add -> Round -> Clip -> Sub -> Mul
                    chain_ops = []
                    current_idx = i
                    
                    for op_type in ['Add', 'Round', 'Clip', 'Sub', 'Mul']:
                        found = False
                        for j in range(current_idx + 1, min(current_idx + 5, len(model_onnx.graph.node))):
                            current_node = model_onnx.graph.node[j]
                            if next_output in current_node.input and current_node.op_type == op_type:
                                chain_nodes.append(current_node)
                                chain_ops.append(op_type)
                                next_output = current_node.output[0]
                                current_idx = j
                                found = True
                                break
                        if not found:
                            break
                    
                    # 如果找到完整的操作链
                    if len(chain_ops) == 5 and chain_ops == ['Add', 'Round', 'Clip', 'Sub', 'Mul']:
                        mul_node = chain_nodes[-1]
                        
                        # 创建QDQ节点
                        quant_node = onnx.helper.make_node(
                            'QuantizeLinear',
                            inputs=[node.input[0], scale_input, "zero_point_constant"],
                            outputs=[f"quantized_{qdq_pairs_created}"],
                            name=f"QuantizeLinear_{qdq_pairs_created}"
                        )
                        
                        dequant_node = onnx.helper.make_node(
                            'DequantizeLinear',
                            inputs=[f"quantized_{qdq_pairs_created}", scale_input, "zero_point_constant"],
                            outputs=mul_node.output,
                            name=f"DequantizeLinear_{qdq_pairs_created}"
                        )
                        
                        new_graph.node.append(quant_node)
                        new_graph.node.append(dequant_node)
                        
                        qdq_pairs_created += 1
                        nodes_skipped += len(chain_nodes)
                        i = current_idx + 1
                        continue
            
            new_graph.node.append(node)
            i += 1
        
        new_model = onnx.helper.make_model(new_graph, producer_name="AutoQuant")
        
        for opset in model_onnx.opset_import:
            new_model.opset_import.append(opset)
        
        onnx.save(new_model, output_path)
        print(f"已将模型转换为包含QDQ节点的版本，保存到: {output_path}")
        print(f"  创建了 {qdq_pairs_created} 对QDQ节点")
        print(f"  替换了 {nodes_skipped} 个原始节点")
