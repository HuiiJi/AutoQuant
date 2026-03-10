"""
示例 06: 高级混合精度与Transformer支持
展示如何使用增强版混合精度量化器和Transformer专用策略
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from autoquant import (
    ModelQuantizer,
    MixedPrecisionQuantizer,
    LayerSelector,
    SensitivityAnalyzer,
    TransformerQuantizer,
    get_transformer_qconfig,
    get_smoothquant_qconfig,
    get_default_qconfig,
    QuantizationEvaluator,
)


class SimpleTransformer(nn.Module):
    """一个简单的Transformer类模型用于演示"""
    def __init__(self, hidden_dim=512, num_heads=8, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 输入嵌入
        self.embedding = nn.Linear(3, hidden_dim)
        
        # Transformer层（简化版）
        self.transformer_layers = nn.ModuleList()
        for i in range(num_layers):
            self.transformer_layers.append(nn.ModuleDict({
                'q_proj': nn.Linear(hidden_dim, hidden_dim),
                'k_proj': nn.Linear(hidden_dim, hidden_dim),
                'v_proj': nn.Linear(hidden_dim, hidden_dim),
                'out_proj': nn.Linear(hidden_dim, hidden_dim),
                'ffn1': nn.Linear(hidden_dim, hidden_dim * 4),
                'ffn2': nn.Linear(hidden_dim * 4, hidden_dim),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
            }))
        
        # 输出头
        self.head = nn.Linear(hidden_dim, 3)
    
    def forward(self, x):
        # x: (batch, 3, h, w) -> (batch, h*w, 3)
        batch_size = x.shape[0]
        x = x.flatten(2).transpose(1, 2)
        
        # 嵌入
        x = self.embedding(x)
        
        # Transformer层
        for layer in self.transformer_layers:
            # 简化的Attention
            residual = x
            x = layer['norm1'](x)
            q = layer['q_proj'](x)
            k = layer['k_proj'](x)
            v = layer['v_proj'](x)
            
            # 简化的self-attention
            attn = q @ k.transpose(-2, -1) / (self.hidden_dim ** 0.5)
            attn = attn.softmax(dim=-1)
            x = attn @ v
            x = layer['out_proj'](x)
            x = x + residual
            
            # FFN
            residual = x
            x = layer['norm2'](x)
            x = layer['ffn1'](x)
            x = x.relu()
            x = layer['ffn2'](x)
            x = x + residual
        
        # 输出
        x = x.mean(dim=1)
        x = self.head(x)
        return x


def demo_mixed_precision():
    """演示混合精度量化"""
    print("\n" + "=" * 70)
    print("示例 1: 增强版混合精度量化")
    print("=" * 70)
    
    # 创建模型
    model = SimpleTransformer(hidden_dim=128)
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # 1. 使用LayerSelector选择层
    print("\n[1] 使用LayerSelector选择层:")
    conv_layers = LayerSelector.get_conv_layers(model)
    linear_layers = LayerSelector.get_linear_layers(model)
    transformer_layers = LayerSelector.get_transformer_layers(model)
    output_layers = LayerSelector.get_output_layers(model)
    
    print(f"    Linear层: {len(linear_layers)}")
    print(f"    Transformer层: {len(transformer_layers)}")
    print(f"    输出层: {len(output_layers)}")
    
    # 2. 初始化混合精度量化器
    print("\n[2] 初始化混合精度量化器...")
    mp_quantizer = MixedPrecisionQuantizer(model, get_default_qconfig())
    
    # 3. 基于敏感度自动配置
    print("\n[3] 基于敏感度自动配置...")
    analyzer = mp_quantizer.auto_config_by_sensitivity(
        dummy_input,
        threshold=0.001
    )
    
    # 4. 手动配置：保持输出层为浮点
    print("\n[4] 手动配置：保持输出层为浮点...")
    if output_layers:
        mp_quantizer.set_fp_layers(list(output_layers))
    
    # 5. Bit-width搜索
    print("\n[5] Bit-width搜索...")
    bit_allocation = mp_quantizer.search_bit_width(
        dummy_input,
        objective='balance'
    )
    print(f"    已为 {len(bit_allocation)} 层分配bit-width")
    
    # 6. 显示配置摘要
    print("\n[6] 配置摘要:")
    print(mp_quantizer.get_config_summary())
    
    # 7. 准备模型
    print("\n[7] 准备混合精度模型...")
    prepared_model = mp_quantizer.prepare()
    print("    ✓ 完成!")


def demo_transformer_quantization():
    """演示Transformer专用量化"""
    print("\n" + "=" * 70)
    print("示例 2: Transformer专用量化")
    print("=" * 70)
    
    # 创建模型
    model = SimpleTransformer(hidden_dim=256)
    
    # 1. 初始化Transformer量化器
    print("\n[1] 初始化Transformer量化器 (SmoothQuant策略)...")
    from autoquant.special_models import TransformerQuantConfig
    config = TransformerQuantConfig(
        strategy='smoothquant',
        smoothquant_alpha=0.5,
        quantize_attention=True,
        quantize_ffn=True,
        quantize_kv_cache=True,
    )
    
    transformer_quantizer = TransformerQuantizer(model, config)
    
    # 2. 显示配置
    print("\n[2] 配置摘要:")
    print(transformer_quantizer.get_config_summary())
    
    # 3. 识别Transformer层
    print("\n[3] 识别Transformer层...")
    transformer_layers = transformer_quantizer.identify_transformer_layers()
    print(f"    识别到 {len(transformer_layers)} 个Transformer相关层")
    for name in list(transformer_layers.keys())[:10]:
        print(f"      - {name}")
    
    # 4. 获取专用QConfig
    print("\n[4] 获取Transformer专用QConfig...")
    transformer_qconfig = get_transformer_qconfig()
    print("    ✓ 获取完成")
    
    smoothquant_qconfig = get_smoothquant_qconfig(alpha=0.5)
    print("    ✓ SmoothQuant QConfig获取完成")


def demo_evaluation():
    """演示量化评估"""
    print("\n" + "=" * 70)
    print("示例 3: 量化评估")
    print("=" * 70)
    
    # 创建模型
    model = SimpleTransformer(hidden_dim=128)
    dummy_input = torch.randn(2, 3, 32, 32)
    
    # 先量化一个版本
    print("\n[1] 量化模型...")
    quantizer = ModelQuantizer(model, get_default_qconfig())
    quantizer.prepare()
    quantizer.calibrate([dummy_input])
    quantized_model = quantizer.convert()
    
    # 初始化评估器
    print("\n[2] 初始化评估器...")
    evaluator = QuantizationEvaluator(model, quantized_model)
    
    # 评估模型大小
    print("\n[3] 评估模型大小...")
    evaluator.evaluate_model_size()
    
    # 评估推理速度
    print("\n[4] 评估推理速度...")
    evaluator.evaluate_speed(dummy_input, iterations=50)
    
    # 评估输出相似度
    print("\n[5] 评估输出相似度...")
    evaluator.evaluate_output_similarity(dummy_input)
    
    # 生成报告
    print("\n[6] 生成评估报告...")
    evaluator.print_report()


def main():
    print("=" * 70)
    print("AutoQuant 高级功能演示")
    print("=" * 70)
    
    try:
        demo_mixed_precision()
    except Exception as e:
        print(f"混合精度演示跳过: {e}")
    
    try:
        demo_transformer_quantization()
    except Exception as e:
        print(f"Transformer演示跳过: {e}")
    
    try:
        demo_evaluation()
    except Exception as e:
        print(f"评估演示跳过: {e}")
    
    print("\n" + "=" * 70)
    print("示例 06 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
