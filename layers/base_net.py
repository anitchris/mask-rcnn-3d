# -*- coding:utf-8 -*-
"""
   File Name：     base_net.py
   Description :   res18+unet的基网络部分
   Author :        royce.mao
   date：          2019/05/06
"""

import torch
from torch import nn
from config import current_config as cfg


class Net(nn.Module):
    def __init__(self):
        """
        删除coord参数与结构上coord的channels做torch.cat的res18+unet基网络
        """
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True))

        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
        self.featureNum_forw = [24, 32, 64, 96, 96]  # 原[24, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 96]  # 原[128, 64, 64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_back[i + 1] + self.featureNum_forw[i + 2],
                                          self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        # self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        # self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2, stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(96, 96, kernel_size=2, stride=2),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),
                                    nn.ReLU(),
                                    # nn.Dropout3d(p = 0.3),
                                    nn.Conv3d(64, 5 * len(cfg['anchors']), kernel_size=1))

    def forward(self, x):
        """
        前向传播的具体网络结构，输入为x，暂未注释输入维度
        :param x: [Batch, Channel, D, H, W]
        :return: [anchors_all, (coord_z, coord_y, coord_x, diameters, cls_tag)]
        """
        # 收缩路径
        out = self.preBlock(x)  # 16
        out_pool, indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)  # 32
        out1_pool, indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)  # 64
        # out2 = self.drop(out2)
        out2_pool, indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)  # 96
        out3_pool, indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)  # 96
        # out4 = self.drop(out4)
        # 扩张路径
        rev3 = self.path1(out4)  # 96，ConvTranspose3d操作并未改变channels数
        comb3 = self.back3(torch.cat((rev3, out3), 1))  # 96+96 -> back后的64
        # comb3 = self.drop(comb3)
        rev2 = self.path2(comb3)  # 64，ConvTranspose3d操作并未改变channels数
        comb2 = self.back2(torch.cat((rev2, out2), 1))  # 64+64 -> back后的128
        # rpn_head的基feature_map
        feature = self.drop(comb2)
        # 输出层
        out = self.output(feature)  # 128 -> 64
        size = out.size()  # [batch, channel, d, h, w]
        # out层的维度reshape（待修改）
        out = out.view(out.size(0), out.size(1),  -1)
        # transpose(1, 2)以修改为channel-last，reshape为每个锚点上anchors数量的分类、回归输出
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(cfg['anchors']), 5)  # out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        # 最终的reshape
        out = out.view(-1, 5)

        return out, feature


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out