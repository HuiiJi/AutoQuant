"""
旷视科技 NAFNet 模型结构
论文: https://arxiv.org/abs/2204.04676
官方代码: https://github.com/megvii-research/NAFNet
"""

import torch.fx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List, Tuple


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0, dilation=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=dilation,
            stride=1,
            groups=dw_channel,
            bias=True,
            dilation=dilation,
        )

        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


class LayerNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.c = c

    def forward(self, x, eps=1e-6):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        y = self.weight.view(1, self.c, 1, 1) * y + self.bias.view(1, self.c, 1, 1)
        return y


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFNet_flow(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=8,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1],
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.ending = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=2,
                      kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.Tanh()
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan, dilation=2) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for idx, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan // 2, 1, bias=False),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                )
            )
            chan = chan // 2
            dec = nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            self.decoders.append(dec)

    def forward(self, inp):
        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        return x


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm2d):
        super().__init__()

        self.box_filter = nn.Conv2d(
            3, 3, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=3
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, bias=False),
        )

        # 初始化为均值滤波器
        self.box_filter.weight.data[...] = 1.0 / (2 * radius + 1)**2

    def forward(self, x_lr, y_lr):
        _, _, h_lrx, w_lrx = x_lr.size()

        N = self.box_filter(torch.ones((1, 3, h_lrx, w_lrx)).to(x_lr.device))

        # mean_x
        mean_x = self.box_filter(x_lr) / N
        # mean_y
        mean_y = self.box_filter(y_lr) / N
        # cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        # var_x
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x
        # A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        # b
        b = mean_y - A * mean_x
        return A, b


class NAFNet_dgf(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=8,
        middle_blk_num=2,
        enc_blk_nums=[1, 2, 2, 2],
        dec_blk_nums=[2, 2, 2, 1],
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.radius = 1
        self.gf = ConvGuidedFilter(self.radius, norm=AdaptiveNorm)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for idx, num in enumerate(dec_blk_nums):

            self.ups.append(
                nn.Sequential(
                    # nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                    nn.Conv2d(chan, chan // 2, 1, bias=False),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                )
            )
            chan = chan // 2
            dec = nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            self.decoders.append(dec)

    def forward(self, inp):
        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        A_lr, b_lr = self.gf(inp, x)
        return A_lr, b_lr


class NAFNet_dgf_4c(nn.Module):
    def __init__(
        self,
        img_channel=4,
        width=8,
        middle_blk_num=2,
        enc_blk_nums=[1, 2, 2, 2],
        dec_blk_nums=[2, 2, 2, 1],
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=4,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=3,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.gf = ConvGuidedFilter(1, norm=AdaptiveNorm)
        self.attentions = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan, dilation=2) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for idx, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan // 2, 1, bias=False),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                )
            )
            chan = chan // 2
            dec = nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            self.decoders.append(dec)
            self.attentions.append(WrinkleAttention(chan))

    def forward(self, inp, mask):
        com = torch.concat([inp, mask], dim=1)
        x = self.intro(com)
        encs = []

        # Encoder 流程
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip, attn in zip(self.decoders, self.ups, encs[::-1], self.attentions):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            x = attn(x, mask)

        x = self.ending(x)

        A_lr, b_lr = self.gf(inp, x)
        return A_lr, b_lr
