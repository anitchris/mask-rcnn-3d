# -*- coding:utf-8 -*-
"""
   File Name：     base_net.py
   Description :   res18+unet的基网络部分
   Author :        royce.mao
   date：          2019/05/06
"""

import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, num_blocks_forw, num_blocks_back):
        """
        
        :param num_blocks_forw: 维度（4,），具体取值[2, 2, 3, 3]
        :param num_blocks_back: 维度（2,），具体取值[3, 3]
        """
        super(Net, self).__init__()
        self.featureNum_forw = [24, 32, 64, 96, 96]  # 原[24, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 96]  # 原[128, 64, 64]
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True))
        # 修改的block layers
        self.forw1 = self._make_layer(num_blocks_forw[0], 0, True)
        self.forw2 = self._make_layer(num_blocks_forw[1], 1, True)
        self.forw3 = self._make_layer(num_blocks_forw[2], 2, True)
        self.forw4 = self._make_layer(num_blocks_forw[3], 3, True)
        self.back3 = self._make_layer(num_blocks_back[1], 1, False)
        self.back2 = self._make_layer(num_blocks_back[0], 0, False)
        # pooling3d
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        # 上采样layer
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(96, 96, kernel_size=2, stride=2),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        # dropout
        self.drop = nn.Dropout3d(p=0.5, inplace=False)

    def _make_layer(self, num_blocks, inds, is_down):
        """
        指定num_blocks数量的layers结构
        :param num_blocks: 标量（num_blocks_forw = [2, 2, 3, 3]与featureNum_back = [128, 64, 96]）
        :param inds: 标量（featureNum_forw与featureNum_back的索引）
        :param is_down: boolean值（unet下采样部分or上采样部分）
        :return:
        """
        layers = []
        for i in range(num_blocks):
            if i == 0:
                if is_down:
                    layers.append(Bottleneck(self.featureNum_forw[inds], self.featureNum_forw[inds + 1]))
                else:
                    layers.append(Bottleneck(self.featureNum_back[inds + 1] + self.featureNum_forw[inds + 2],
                                             self.featureNum_back[inds]))
            else:
                if is_down:
                    layers.append(Bottleneck(self.featureNum_forw[inds + 1], self.featureNum_forw[inds + 1]))
                else:
                    layers.append(Bottleneck(self.featureNum_back[inds], self.featureNum_back[inds]))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        具体网络结构
        :param x: [batch, channel, D, H, W]
        :return: [batch, channel, D, H, W]
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

        return feature


class Bottleneck(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        """
        
        :param n_in: 标量，输入tensor的channel数
        :param n_out: 标量，输出tensor的channel数
        :param stride: 标量
        """
        super(Bottleneck, self).__init__()
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

# if __name__ == '__main__':
#     net = Net(3)
#     from torchsummary import summary
#
#     summary(net, (1, 32, 32, 32))
#     input = torch.randn(2, 1, 32, 32, 32)
#     out = net(input)
#     print(out[0].shape, out[1].shape)
