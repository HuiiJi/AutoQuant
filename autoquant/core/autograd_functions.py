"""
自定义Autograd Function - 用于QAT的梯度传递
实现STE（Straight-Through Estimator）和高级量化梯度
"""
import torch
import torch.nn as nn
from torch.autograd import Function


class RoundSTE(Function):
    """
    带STE的Round操作
    前向：使用torch.round
    反向：梯度直接通过（直通估计器）
    """
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


round_ste = RoundSTE.apply


class ClampGrad(Function):
    """
    Clamp操作的梯度处理
    在clamp范围内梯度通过，范围外梯度为0
    """
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        ctx.save_for_backward(x, torch.tensor(min_val), torch.tensor(max_val))
        return torch.clamp(x, min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        x, min_val, max_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < min_val] = 0
        grad_input[x > max_val] = 0
        return grad_input, None, None


clamp_grad = ClampGrad.apply


class FakeQuantizeSTE(Function):
    """
    完整的FakeQuantize Autograd Function
    包含量化、反量化和STE梯度
    """
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max):
        """
        前向：量化 -> 反量化
        Args:
            x: 输入张量
            scale: 量化步长
            zero_point: 零点
            quant_min: 量化最小值
            quant_max: 量化最大值
        Returns:
            反量化后的张量
        """
        # 保存用于反向传播的信息
        ctx.save_for_backward(x, scale, zero_point)
        ctx.quant_min = quant_min
        ctx.quant_max = quant_max

        # 量化：x_int = round(x / scale + zero_point)
        x_int = torch.round(x / scale + zero_point)
        # 裁剪
        x_int = torch.clamp(x_int, quant_min, quant_max)
        # 反量化：x_dq = (x_int - zero_point) * scale
        x_dq = (x_int - zero_point) * scale

        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向：STE梯度
        Args:
            grad_output: 输出梯度
        Returns:
            输入梯度
        """
        x, scale, zero_point = ctx.saved_tensors
        quant_min = ctx.quant_min
        quant_max = ctx.quant_max

        # 计算x在量化后的整数表示（不round）
        x_int_unrounded = x / scale + zero_point

        # 只在quant_min到quant_max范围内传递梯度
        grad_input = grad_output.clone()
        mask = (x_int_unrounded >= quant_min) & (x_int_unrounded <= quant_max)
        grad_input[~mask] = 0

        # scale和zero_point的梯度（可选，用于LSQ等可学习scale的方法）
        # 这里先返回None，在LSQ中单独处理
        return grad_input, None, None, None, None


fake_quantize_ste = FakeQuantizeSTE.apply


class LSQQuantize(Function):
    """
    LSQ (Learned Step Size Quantization) Autograd Function
    支持可学习的scale，包含scale的梯度
    论文：https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max):
        """
        前向：量化 -> 反量化
        """
        # 保存用于反向传播
        ctx.save_for_backward(x, scale, zero_point)
        ctx.quant_min = quant_min
        ctx.quant_max = quant_max

        # 量化
        x_int = torch.round(x / scale + zero_point)
        x_int = torch.clamp(x_int, quant_min, quant_max)
        x_dq = (x_int - zero_point) * scale

        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向：计算x和scale的梯度
        LSQ的scale梯度有特殊计算方式
        """
        x, scale, zero_point = ctx.saved_tensors
        quant_min = ctx.quant_min
        quant_max = ctx.quant_max

        # 计算辅助变量
        x_int_unrounded = x / scale + zero_point
        x_int = torch.round(x_int_unrounded)
        x_int_clamped = torch.clamp(x_int, quant_min, quant_max)

        # x的梯度：只在范围内传递
        grad_x = grad_output.clone()
        mask = (x_int_unrounded >= quant_min - 0.5) & (x_int_unrounded <= quant_max + 0.5)
        grad_x[~mask] = 0

        # scale的梯度：LSQ的特殊计算
        # dL/ds = sum(grad_output * (-(x_int_clamped - zero_point) + x/s))
        grad_scale = grad_output * (-(x_int_clamped - zero_point) + x / scale)
        grad_scale = grad_scale.sum()

        # 梯度归一化（LSQ论文中的技巧）
        grad_scale = grad_scale / (torch.sqrt(torch.tensor(quant_max - quant_min)))

        return grad_x, grad_scale, None, None, None


lsq_quantize = LSQQuantize.apply


class PACTQuantize(Function):
    """
    PACT (Parameterized Clipping Activation) Quantize
    用于激活值的可学习裁剪范围
    论文：https://arxiv.org/abs/1805.06085
    """
    @staticmethod
    def forward(ctx, x, alpha, scale, zero_point, quant_min, quant_max):
        """
        前向：先PACT裁剪，再量化
        """
        ctx.save_for_backward(x, alpha, scale, zero_point)
        ctx.quant_min = quant_min
        ctx.quant_max = quant_max

        # PACT裁剪：min(max(x, 0), alpha)
        x_clamped = torch.clamp(x, 0, alpha)
        
        # 量化
        x_int = torch.round(x_clamped / scale + zero_point)
        x_int = torch.clamp(x_int, quant_min, quant_max)
        x_dq = (x_int - zero_point) * scale

        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向：计算x和alpha的梯度
        """
        x, alpha, scale, zero_point = ctx.saved_tensors
        quant_min = ctx.quant_min
        quant_max = ctx.quant_max

        # x的梯度
        grad_x = grad_output.clone()
        grad_x[x < 0] = 0
        grad_x[x > alpha] = 0

        # alpha的梯度
        grad_alpha = grad_output[x > alpha].sum()

        return grad_x, grad_alpha, None, None, None, None


pact_quantize = PACTQuantize.apply
