import cv2
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmengine.model import BaseModule
import torch
import torch.nn as nn


class CrossConv(nn.Module):
    def __init__(self, in_channels, out_channels, horizontal_range, vertical_range, stride=1):
        super(CrossConv, self).__init__()
        self.horizontal_range = horizontal_range
        self.vertical_range = vertical_range
        self.stride = stride

        self.conv_horizontal = nn.Conv2d(in_channels, out_channels, (1, horizontal_range), stride=stride, padding=(0, (horizontal_range - 1) // 2))
        self.conv_vertical = nn.Conv2d(in_channels, out_channels, (vertical_range, 1), stride=stride, padding=((vertical_range - 1) // 2, 0))

    def forward(self, x):
        output_horizontal = self.conv_horizontal(x)
        output_vertical = self.conv_vertical(x)
        output = torch.cat([output_horizontal, output_vertical], dim=1)
        return output


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化卷积核权重和偏置
        self.weights = (torch.zeros((out_channels, in_channels, kernel_size, kernel_size))).to(self.device)
        self.bias = (nn.Parameter(torch.zeros((out_channels,)))).to(self.device)

        # 设置十字形卷积核权重
        center = kernel_size // 2  # 十字形卷积核的中心位置
        nn.Parameter(self.weights[:, :, center, :]) # 设置竖直方向的权重
        nn.Parameter(self.weights[:, :, :, center])  # 设置水平方向的权重

    def forward(self, input_tensor):
        batch_size, in_channels, in_height, in_width = input_tensor.shape
        out_height = int((in_height - self.kernel_size + 2 * self.padding) / self.stride) + 1
        out_width = int((in_width - self.kernel_size + 2 * self.padding) / self.stride) + 1

        # 添加填充
        padded_input = F.pad(input_tensor, (self.padding, self.padding, self.padding, self.padding))

        # 初始化输出张量
        output_tensor = (torch.zeros((batch_size, self.out_channels, out_height, out_width))).to(self.device)

        # 执行卷积计算
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size

                        receptive_field = padded_input[b, :, h_start:h_end, w_start:w_end]
                        output_tensor[b, c_out, h_out, w_out] = torch.sum(receptive_field * self.weights[c_out]) + \
                                                                self.bias[c_out]

        return output_tensor


@MODELS.register_module()
class CrossHead(BaseDecodeHead):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, num_classes=2, in_index=2):
        super().__init__(in_channels=in_channels, num_classes=num_classes, channels=2*in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_index = in_index
        self.conv = Conv2D(self.in_channels, self.out_channels, self.kernel_size, self.stride)

    def forward(self, input):
        """Forward function."""
        output = self.conv(input[self.in_index])
        return output

