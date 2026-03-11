"""
ONNX优化工具
类似onnxsim和onnxsurge的功能，优化和清理量化ONNX模型
"""
import os
from typing import Optional, List, Union
import onnx
from onnx import helper, numpy_helper
import numpy as np


class ONNXOptimizer:
    """
    ONNX模型优化器
    提供多种优化pass来清理和优化量化模型
    """

    def __init__(self, model_path: Optional[str] = None, model: Optional[onnx.ModelProto] = None):
        """
        Args:
            model_path: ONNX模型路径
            model: ONNX模型对象
        """
        if model_path:
            self.model = onnx.load(model_path)
        elif model:
            self.model = model
        else:
            raise ValueError("必须提供model_path或model")

    def optimize(
        self,
        passes: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> onnx.ModelProto:
        """
        执行优化
        
        Args:
            passes: 要执行的优化pass列表，默认执行所有pass
            verbose: 是否打印优化信息
        
        Returns:
            优化后的ONNX模型
        """
        if passes is None:
            passes = self._get_default_passes()

        optimized_model = self.model
        
        if verbose:
            print(f"初始模型大小: {len(optimized_model.SerializeToString()) / 1024:.2f} KB")

        for pass_name in passes:
            if pass_name == "eliminate_identity":
                optimized_model = self._eliminate_identity_nodes(optimized_model)
            elif pass_name == "fold_constants":
                optimized_model = self._fold_constants(optimized_model)
            elif pass_name == "merge_qdq":
                optimized_model = self._merge_qdq_pairs(optimized_model)
            elif pass_name == "cleanup":
                optimized_model = self._cleanup_unused_nodes(optimized_model)
            elif pass_name == "simplify_qdq":
                optimized_model = self._simplify_qdq_patterns(optimized_model)
            elif pass_name == "optimize_layernorm":
                optimized_model = self._optimize_layernorm(optimized_model)
            else:
                if verbose:
                    print(f"警告: 未知的优化pass: {pass_name}")

        if verbose:
            print(f"优化后模型大小: {len(optimized_model.SerializeToString()) / 1024:.2f} KB")

        return optimized_model

    def _get_default_passes(self) -> List[str]:
        """获取默认的优化pass列表"""
        return [
            "fold_constants",
            "eliminate_identity",
            "simplify_qdq",
            "merge_qdq",
            "optimize_layernorm",
            "cleanup",
        ]

    def _eliminate_identity_nodes(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """消除Identity节点"""
        graph = model.graph
        new_nodes = []
        name_map = {}

        for node in graph.node:
            if node.op_type == "Identity":
                if len(node.input) == 1 and len(node.output) == 1:
                    name_map[node.output[0]] = node.input[0]
                    continue
            new_nodes.append(node)

        # 更新所有节点的输入
        for node in new_nodes:
            for i in range(len(node.input)):
                if node.input[i] in name_map:
                    node.input[i] = name_map[node.input[i]]

        # 更新graph的输出
        for output in graph.output:
            if output.name in name_map:
                output.name = name_map[output.name]

        del graph.node[:]
        graph.node.extend(new_nodes)
        return model

    def _fold_constants(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """尝试折叠常量"""
        try:
            from onnxsim import simplify
            model, check = simplify(model)
            if check:
                return model
        except ImportError:
            print("提示: 安装onnxsim可获得更好的常量折叠效果: pip install onnxsim")
        except Exception as e:
            print(f"常量折叠时出错: {e}")
        return model

    def _merge_qdq_pairs(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """合并连续的Q-D-Q-D模式"""
        graph = model.graph
        nodes_to_remove = []
        input_replacements = {}

        for i, node in enumerate(graph.node):
            if node.op_type == "DequantizeLinear":
                # 查找前一个QuantizeLinear
                for j, prev_node in enumerate(graph.node):
                    if i != j and prev_node.op_type == "QuantizeLinear":
                        if len(prev_node.output) > 0 and len(node.input) > 0:
                            if prev_node.output[0] == node.input[0]:
                                # 找到Q-D对，检查是否有后续的Q-D
                                for k, next_node in enumerate(graph.node):
                                    if k != i and next_node.op_type == "QuantizeLinear":
                                        if len(node.output) > 0 and len(next_node.input) > 0:
                                            if node.output[0] == next_node.input[0]:
                                                # 找到了Q-D-Q-D模式
                                                # 检查是否可以合并
                                                if (len(prev_node.input) >= 3 and 
                                                    len(next_node.input) >= 3):
                                                    # 比较scale和zero_point
                                                    scale1 = self._get_initializer_value(graph, prev_node.input[1])
                                                    zp1 = self._get_initializer_value(graph, prev_node.input[2])
                                                    scale2 = self._get_initializer_value(graph, next_node.input[1])
                                                    zp2 = self._get_initializer_value(graph, next_node.input[2])
                                                    
                                                    if (scale1 is not None and scale2 is not None and
                                                        zp1 is not None and zp2 is not None):
                                                        if np.allclose(scale1, scale2) and np.allclose(zp1, zp2):
                                                            # scale和zp相同，可以合并
                                                            nodes_to_remove.extend([j, i, k])
                                                            input_replacements[next_node.output[0]] = prev_node.input[0]

        # 执行替换
        new_nodes = []
        for i, node in enumerate(graph.node):
            if i not in nodes_to_remove:
                for j in range(len(node.input)):
                    if node.input[j] in input_replacements:
                        node.input[j] = input_replacements[node.input[j]]
                new_nodes.append(node)

        del graph.node[:]
        graph.node.extend(new_nodes)
        return model

    def _simplify_qdq_patterns(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """简化QDQ模式"""
        graph = model.graph
        
        # 检查并修复一些常见的QDQ模式问题
        for node in graph.node:
            if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                # 确保axis属性正确
                has_axis = any(attr.name == "axis" for attr in node.attribute)
                if not has_axis:
                    # 智能判断axis值
                    # 对于权重：通常是axis=0（输出通道）
                    # 对于激活：通常是axis=1（通道维度）
                    # 但需要根据输入形状来判断
                    axis = 1  # 默认值
                    
                    # 尝试根据输入形状判断
                    for input_name in node.input:
                        for value_info in graph.input:
                            if value_info.name == input_name:
                                if len(value_info.type.tensor_type.shape.dim) == 4:
                                    # 4D张量，可能是激活或权重
                                    # 对于Conv2d权重：[out_channels, in_channels, kernel_h, kernel_w]
                                    # 对于激活：[batch, channels, height, width]
                                    axis = 0  # 权重
                                break
                        for initializer in graph.initializer:
                            if initializer.name == input_name:
                                if len(initializer.dims) == 4:
                                    # 4D初始化器，通常是权重
                                    axis = 0
                                break
                    
                    axis_attr = helper.make_attribute("axis", axis)
                    node.attribute.append(axis_attr)
        
        return model

    def _optimize_layernorm(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """优化LayerNorm操作，特别是针对NAFNet中的LayerNorm2d"""
        graph = model.graph
        new_nodes = []
        nodes_to_remove = []
        input_replacements = {}
        
        # 查找 permute -> layer_norm -> permute 模式
        i = 0
        while i < len(graph.node):
            node = graph.node[i]
            
            # 检查是否是第一个permute
            if node.op_type == "Transpose" and i + 2 < len(graph.node):
                next_node = graph.node[i + 1]
                next_next_node = graph.node[i + 2]
                
                # 检查是否是 layer_norm
                if next_node.op_type == "LayerNormalization" and next_next_node.op_type == "Transpose":
                    # 检查permute的顺序是否匹配
                    # 对于LayerNorm2d: permute(0, 2, 3, 1) -> layer_norm -> permute(0, 3, 1, 2)
                    permute1_order = None
                    permute2_order = None
                    
                    for attr in node.attribute:
                        if attr.name == "perm":
                            # 检查 attr.ints 的类型
                            if hasattr(attr.ints, '__iter__'):
                                # 尝试获取每个元素的 i 属性，如果失败则直接使用元素
                                try:
                                    permute1_order = [dim.i for dim in attr.ints]
                                except AttributeError:
                                    permute1_order = list(attr.ints)
                    
                    for attr in next_next_node.attribute:
                        if attr.name == "perm":
                            # 检查 attr.ints 的类型
                            if hasattr(attr.ints, '__iter__'):
                                # 尝试获取每个元素的 i 属性，如果失败则直接使用元素
                                try:
                                    permute2_order = [dim.i for dim in attr.ints]
                                except AttributeError:
                                    permute2_order = list(attr.ints)
                    
                    if permute1_order == [0, 2, 3, 1] and permute2_order == [0, 3, 1, 2]:
                        # 找到匹配的模式，创建一个新的LayerNormalization节点，直接在正确的维度上操作
                        new_layernorm = helper.make_node(
                            "LayerNormalization",
                            inputs=[node.input[0]] + next_node.input[1:],
                            outputs=[next_next_node.output[0]],
                            name=f"optimized_layernorm_{i}"
                        )
                        
                        # 复制LayerNormalization的属性
                        for attr in next_node.attribute:
                            if attr.name != "axis":
                                new_layernorm.attribute.append(attr)
                        
                        # 设置axis为1（通道维度）
                        axis_attr = helper.make_attribute("axis", 1)
                        new_layernorm.attribute.append(axis_attr)
                        
                        # 添加新节点
                        new_nodes.append(new_layernorm)
                        
                        # 标记要删除的节点
                        nodes_to_remove.extend([i, i+1, i+2])
                        
                        # 跳过已处理的节点
                        i += 3
                        continue
            
            # 如果不是匹配的模式，添加到新节点列表
            if i not in nodes_to_remove:
                new_nodes.append(node)
            i += 1
        
        # 替换节点
        if nodes_to_remove:
            del graph.node[:]
            graph.node.extend(new_nodes)
        
        return model
    
    def _cleanup_unused_nodes(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """清理未使用的节点和initializer"""
        try:
            from onnxsim import simplify
            model, check = simplify(model)
            if check:
                return model
        except ImportError:
            pass
        except Exception as e:
            print(f"清理时出错: {e}")
        return model

    def _get_initializer_value(self, graph: onnx.GraphProto, name: str):
        """获取initializer的值"""
        for init in graph.initializer:
            if init.name == name:
                return numpy_helper.to_array(init)
        return None

    def save(self, output_path: str):
        """保存优化后的模型"""
        onnx.save(self.model, output_path)
        print(f"模型已保存到: {output_path}")

    @staticmethod
    def from_path(model_path: str) -> "ONNXOptimizer":
        """从文件路径创建优化器"""
        return ONNXOptimizer(model_path=model_path)

    @staticmethod
    def from_model(model: onnx.ModelProto) -> "ONNXOptimizer":
        """从模型对象创建优化器"""
        return ONNXOptimizer(model=model)


def optimize_onnx(
    input_path: str,
    output_path: Optional[str] = None,
    passes: Optional[List[str]] = None,
    verbose: bool = True,
) -> onnx.ModelProto:
    """
    便捷函数：优化ONNX模型
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径（可选）
        passes: 优化pass列表
        verbose: 是否打印信息
    
    Returns:
        优化后的ONNX模型
    """
    optimizer = ONNXOptimizer(model_path=input_path)
    optimized_model = optimizer.optimize(passes=passes, verbose=verbose)
    
    if output_path:
        onnx.save(optimized_model, output_path)
        if verbose:
            print(f"优化后的模型已保存到: {output_path}")
    
    return optimized_model


def simplify_with_onnxsim(
    input_path: str,
    output_path: Optional[str] = None,
    **kwargs,
) -> onnx.ModelProto:
    """
    使用onnxsim进行完整简化
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        **kwargs: 传递给onnxsim.simplify的参数
    
    Returns:
        简化后的ONNX模型
    """
    try:
        from onnxsim import simplify
    except ImportError:
        raise ImportError("请先安装onnxsim: pip install onnxsim")
    
    model = onnx.load(input_path)
    simplified_model, check = simplify(model, **kwargs)
    
    if not check:
        print("警告: 模型检查失败，但仍返回简化后的模型")
    
    if output_path:
        onnx.save(simplified_model, output_path)
        print(f"简化后的模型已保存到: {output_path}")
    
    return simplified_model
