# -*- coding: utf-8 -*-
"""
@File    : U_net_model.py
@Author  : 莫胜辉
@Email   : 18230571270@163.com
@Date    : 2025/7/1 19:32
@Version : 1.0
@Desc    : TODO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import Skin_Dataset


class AttentionBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channel, in_channel // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channel, in_channel // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习缩放系数

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W

        # [B, C', H, W] → [B, C', N] → [B, N, C']
        proj_query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # Q: [B, N, C']
        proj_key  = self.key_conv(x).view(B, -1, N)                     # K: [B, C', N]
        proj_value = self.value_conv(x).view(B, -1, N).permute(0, 2, 1)  # V: [B, N, C]

        # Attention: [B, N, N]
        attention = torch.bmm(proj_query, proj_key)  # Q × K
        attention = F.softmax(attention, dim=-1)

        # Output: [B, N, C]
        out = torch.bmm(attention, proj_value)
        out = out.permute(0, 2, 1).view(B, C, H, W)

        # Residual
        out = self.gamma * out + x
        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.sequential = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(),
            nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.sequential(x)


class Conv_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.sequential = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.sequential(x)


class UNET(nn.Module):
    def __init__(self, in_channel=3, residual_block_number=3):
        super().__init__()
        self.residual_number = residual_block_number
        self.mid_channel = 32
        self.in_channel_list = [self.mid_channel * 2 ** i for i in range(self.residual_number)]
        self.ConvBlock1 = Conv_block(in_channel=in_channel, out_channel=self.mid_channel)
        self.max_pool_1 = nn.MaxPool2d(2)
        self.ConvBlock = nn.ModuleList(
            [Conv_block(in_channel=self.in_channel_list[i], out_channel=self.in_channel_list[i] * 2)
             for i in range(residual_block_number)])
        self.max_pool = nn.ModuleList([nn.MaxPool2d(2) for i in range(len(self.in_channel_list))])
        self.residual_block = nn.ModuleList([ResidualBlock(x) for x in self.in_channel_list])
        self.up_sampler = nn.ModuleList(
            [nn.ConvTranspose2d(in_channels=x * 2, out_channels=x, kernel_size=2, stride=2) for x in
             self.in_channel_list[::-1]])
        self.up_sampler_ = nn.ConvTranspose2d(in_channels=self.mid_channel, out_channels=self.mid_channel,
                                              kernel_size=2, stride=2)
        self.conv_ = nn.Conv2d(self.mid_channel, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.fuse_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.fuse_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.fuse_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.att_in_channel = [32,64,128]
        self.att_in_skip1 = AttentionBlock(self.att_in_channel[0])
        self.att_in_skip2 = AttentionBlock(self.att_in_channel[1])
        self.att_in_skip3 = AttentionBlock(self.att_in_channel[2])
        self.att_in_encoder = AttentionBlock(256)
        self.dropout_bridge = nn.Dropout2d(p=0.3)

    def forward(self, x):  # 16,3,256,256
        x = self.ConvBlock1(x)  # 16,32,256,256
        x = self.max_pool_1(x)  # 16,32,128,128
        x = self.residual_block[0](x)  # 16,32,128,128
        skip_connection1 = self.att_in_skip1(x)
        x = self.ConvBlock[0](x)  # 16,64,128,128
        x = self.max_pool[0](x)  # 16,64,64,64
        x = self.residual_block[1](x)  # 16,64,64,64
        skip_connection2 = self.att_in_skip2(x)
        x = self.ConvBlock[1](x)  # 16,128,64,64
        x = self.max_pool[1](x)  # 16,128,32,32
        x = self.residual_block[2](x)  # 16,128,32,32
        skip_connection3 = self.att_in_skip3(x)
        x = self.ConvBlock[2](x)  # 16,256,32,32
        x = self.max_pool[2](x)  # 16,256,16,16
        x = self.att_in_encoder(x)
        x = self.dropout_bridge(x)
        x = self.up_sampler[0](x)  # 16,128,32,32
        x = torch.cat([x, skip_connection3], dim=1)
        x = self.fuse_conv1(x)
        x= self.dropout_bridge(x)
        x = self.up_sampler[1](x)  # 16,64,64,64
        x = torch.cat([x, skip_connection2], dim=1)
        x = self.fuse_conv2(x)
        x = self.dropout_bridge(x)
        x = self.up_sampler[2](x)  # 16,32,128,128
        x = torch.cat([x, skip_connection1], dim=1)
        x = self.fuse_conv3(x)
        x = self.dropout_bridge(x)
        x = self.up_sampler_(x)  # 16,16,256,256
        x = self.conv_(x)  # 16,1,256,256
        x = self.sigmoid(x)  # 16,1,256,256
        return x
