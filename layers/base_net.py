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
    def __init__(self, num_anchors):
        """
        删除coord参数与结构上coord的channels做torch.cat的res18+unet基网络
        """
        super(Net, self).__init__()
        self.num_anchors = num_anchors
        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
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

    def _make_layer(self, num_blocks, indc, isDown):
        """
        指定num_blocks数量的layers结构
        :param num_blocks: num_blocks_forw 与 num_blocks_back
        :param indc: featureNum变量中channel的索引
        :param isDown: True or False 代表unet的特征提取部分还是上采样部分
        :return:
        """
        layers = []
        for i in range(num_blocks):
            if i == 0:
                if isDown:
                    layers.append(PostRes(self.featureNum_forw[indc], self.featureNum_forw[indc + 1]))
                else:
                    layers.append(PostRes(self.featureNum_back[indc + 1] + self.featureNum_forw[indc + 2],
                                          self.featureNum_back[indc]))
            else:
                if isDown:
                    layers.append(PostRes(self.featureNum_forw[indc + 1], self.featureNum_forw[indc + 1]))
                else:
                    layers.append(PostRes(self.featureNum_back[indc], self.featureNum_back[indc]))

        return nn.Sequential(*layers)

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

        return feature


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


# if __name__ == '__main__':
#     net = Net(3)
#     from torchsummary import summary
#
#     summary(net, (1, 32, 32, 32))
#     input = torch.randn(2, 1, 32, 32, 32)
#     out = net(input)
#     print(out[0].shape, out[1].shape)
