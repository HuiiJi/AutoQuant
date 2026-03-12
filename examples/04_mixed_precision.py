"""
示例 04: 混合精度量化
展示如何结合敏感度分析进行混合精度量化

Author: jihui
Date: 2026-03-13
"""
import torch
import torch.nn as nn
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from autoquant import (
    ModelQuantizer,
    get_default_qconfig,
    NAFNet_dgf,
    SensitivityAnalyzer,
)


def export_to_onnx(model, dummy_input, filename="quantized_model.onnx"):
    """导出模型为 ONNX 格式"""
    print(f"\n[9] 导出 ONNX 模型: {filename}")
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"    ✓ ONNX 模型导出成功！")


def main():
    print("=" * 80)
    print("示例 04: 混合精度量化 (敏感度分析 + PTQ)")
    print("=" * 80)
    
    # ========================================================================
    # 步骤 1: 创建 NAFNet 模型
    # ========================================================================
    print("\n" + "=" * 80)
    print("[1] 创建 NAFNet 模型")
    print("=" * 80)
    
    model = NAFNet_dgf(
        img_channel=3,
        width=8,
        middle_blk_num=2,
        enc_blk_nums=[1, 2, 2, 2],
        dec_blk_nums=[2, 2, 2, 1],
    )
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64)
    print(f"    输入形状: {dummy_input.shape}")
    
    # ========================================================================
    # 步骤 2: 敏感度分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("[2] 敏感度分析")
    print("=" * 80)
    
    qconfig = get_default_qconfig()
    analyzer = SensitivityAnalyzer(model, qconfig)
    calib_data = [torch.randn(1, 3, 64, 64) for _ in range(10)]
    
    # 只分析前 10 个层做演示
    all_layers = analyzer._get_quantizable_layers()
    analyze_layers = all_layers[:10]
    print(f"\n    为了演示速度，只分析前 {len(analyze_layers)} 个层")
    
    sensitivity_scores = analyzer.analyze(
        dummy_input,
        calib_data=calib_data,
        only_layers=analyze_layers
    )
    
    # ========================================================================
    # 步骤 3: 生成报表
    # ========================================================================
    print("\n" + "=" * 80)
    print("[3] 生成敏感度分析报表")
    print("=" * 80)
    
    output_dir = os.path.join(project_root, "mixed_precision_results")
    analyzer.save_results(output_dir, top_n_percent=10.0)
    print("\n" + analyzer.generate_report(top_n_percent=10.0))
    
    # ========================================================================
    # 步骤 4: 获取推荐跳过的层
    # ========================================================================
    print("\n" + "=" * 80)
    print("[4] 确定要跳过的层（前 10% 敏感度高的层）")
    print("=" * 80)
    
    quantizable_layers, skip_layers = analyzer.get_recommended_layers(top_n_percent=10.0)
    
    print(f"\n    推荐量化的层: {len(quantizable_layers)} 个")
    print(f"    推荐跳过的层: {len(skip_layers)} 个")
    
    if skip_layers:
        print(f"\n    跳过的层列表:")
        for layer in skip_layers:
            print(f"      - {layer}")
    
    # ========================================================================
    # 步骤 5: 准备量化模型 - 跳过敏感度高的层
    # ========================================================================
    print("\n" + "=" * 80)
    print("[5] 准备量化模型 (跳过敏感度高的层)")
    print("=" * 80)
    
    quantizer = ModelQuantizer(model, qconfig)
    prepared_model = quantizer.prepare(skip_layers=set(skip_layers))
    print(f"    ✓ 模型准备完成")
    print(f"    ✓ 跳过了 {len(skip_layers)} 个敏感度高的层")
    
    # ========================================================================
    # 步骤 6: 校准
    # ========================================================================
    print("\n" + "=" * 80)
    print("[6] 校准模型")
    print("=" * 80)
    
    calib_data = [torch.randn(1, 3, 64, 64) for _ in range(50)]
    quantizer.calibrate(calib_data)
    print("    ✓ 校准完成")
    
    # ========================================================================
    # 步骤 7: 转换为量化模型
    # ========================================================================
    print("\n" + "=" * 80)
    print("[7] 转换为量化模型")
    print("=" * 80)
    
    quantized_model = quantizer.convert()
    print("    ✓ 转换完成")
    
    # ========================================================================
    # 步骤 8: 验证
    # ========================================================================
    print("\n" + "=" * 80)
    print("[8] 验证量化模型")
    print("=" * 80)
    
    with torch.no_grad():
        original_output = model(dummy_input)
        quantized_output = quantized_model(dummy_input)

    if isinstance(quantized_output, tuple):
        quantized_output = quantized_output[0]
        original_output = original_output[0]
    
    mse = torch.nn.functional.mse_loss(original_output, quantized_output).item()
    print(f"    输出 MSE: {mse:.6f}")
    
    # ========================================================================
    # 步骤 9: 导出 ONNX
    # ========================================================================
    onnx_path = os.path.join(project_root, "mixed_precision_model.onnx")
    export_to_onnx(quantized_model, dummy_input, onnx_path)
    
    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"  ✓ 敏感度分析完成")
    print(f"  ✓ 跳过敏感度高的层: {len(skip_layers)} 个")
    print(f"  ✓ 量化层: {len(quantizable_layers)} 个")
    print(f"  ✓ 完整 PTQ 流程完成")
    print(f"  ✓ 输出 MSE: {mse:.6f}")
    print(f"  ✓ 结果已保存到: {output_dir}")
    print(f"  ✓ ONNX 模型已导出: {onnx_path}")
    print("=" * 80)
    print("示例 04 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
