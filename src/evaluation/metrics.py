"""
量化评估指标
包含常用的精度评估指标：
- 分类任务: Top-1/Top-5 Accuracy
- 图像任务: PSNR, SSIM
- 通用指标: L1, L2, Cosine Similarity
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1,)
) -> dict:
    """
    计算分类任务的准确率

    Args:
        outputs: 模型输出 logits (batch_size, num_classes)
        targets: 真实标签 (batch_size,)
        topk: 需要计算的top-k值，例如 (1, 5)

    Returns:
        包含准确率的字典，例如 {'top1': 0.95, 'top5': 0.99}
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res[f'top{k}'] = correct_k / batch_size

    return res


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0
) -> torch.Tensor:
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    常用于图像恢复、超分辨率等任务

    Args:
        pred: 预测图像
        target: 真实图像
        data_range: 数据范围，通常为1.0或255.0

    Returns:
        PSNR值 (dB)
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(data_range ** 2 / mse)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
    data_range: float = 1.0
) -> torch.Tensor:
    """
    计算SSIM (Structural Similarity Index)
    常用于衡量图像结构相似性

    Args:
        pred: 预测图像 (N, C, H, W)
        target: 真实图像 (N, C, H, W)
        window_size: 滑动窗口大小
        size_average: 是否对batch求平均
        data_range: 数据范围

    Returns:
        SSIM值
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # 创建高斯窗口
    channel = pred.size(1)
    window = _create_window(window_size, channel).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred**2, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target**2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=(1, 2, 3))


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    """创建高斯窗口（SSIM内部使用）"""
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2)**2 / float(2 * sigma**2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def compute_l1_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    计算L1误差 (MAE - Mean Absolute Error)

    Args:
        pred: 预测值
        target: 真实值
        reduction: 'mean', 'sum', 或 'none'

    Returns:
        L1误差
    """
    return F.l1_loss(pred, target, reduction=reduction)


def compute_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    计算L2误差 (MSE - Mean Squared Error)

    Args:
        pred: 预测值
        target: 真实值
        reduction: 'mean', 'sum', 或 'none'

    Returns:
        L2误差
    """
    return F.mse_loss(pred, target, reduction=reduction)


def compute_cosine_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    计算余弦相似度

    Args:
        pred: 预测值
        target: 真实值
        dim: 计算相似度的维度
        eps: 防止除零的小值

    Returns:
        余弦相似度
    """
    return F.cosine_similarity(pred, target, dim=dim, eps=eps).mean()
