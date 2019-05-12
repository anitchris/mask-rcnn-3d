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


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 channels=[24, 32, 64, 64, 64],
                 blocks=[2, 2, 2, 2]):
        super(ResNet, self).__init__()
        c1, c2, c3, c4, c5 = channels  # 通道数
        b2, b3, b4, b5 = blocks  # 每个stage包含的block数
        # 第一个stage 7*7*7的卷积改为两个3*3*3的卷积
        self.stage1 = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
            nn.Conv3d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1))
        self.stage2 = _make_stage(block, c1, c2, b2, stride=2)
        self.stage3 = _make_stage(block, c2, c3, b3, stride=2)
        self.stage4 = _make_stage(block, c3, c4, b4, stride=2)
        self.stage5 = _make_stage(block, c4, c5, b5, stride=2)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = f3 = self.stage3(x)
        x = f4 = self.stage4(x)
        x = f5 = self.stage5(x)

        return x, f3, f4, f5


class BasicBlock(nn.Module):
    """
    resnet基础block;包含两层卷积(conv-relu-bn)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if stride != 1 or in_channels != out_channels:  # 需要下采样的情况
            self.down_sample = nn.Sequential(nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False), nn.BatchNorm3d(out_channels))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, 'down_sample') and self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
        resnet bottleneck block;包含三层卷积(conv-relu-bn)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.base_channels = out_channels // 4  # 输出通道数的4分之一
        self.conv1 = nn.Conv3d(in_channels, self.base_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.base_channels)
        self.conv2 = nn.Conv3d(
            self.base_channels, self.base_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(self.base_channels)
        self.conv3 = nn.Conv3d(self.base_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:  # 需要下采样的情况
            self.down_sample = nn.Sequential(nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False), nn.BatchNorm3d(out_channels))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if hasattr(self, 'down_sample') and self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


def _make_stage(block, in_channels, out_channels, num_blocks, stride=1):
    """

    :param block: BasicBlock 或 Bottleneck
    :param in_channels:
    :param out_channels:
    :param num_blocks: 本层(stage)包含的block块数
    :param stride: 步长
    :return:
    """
    layers = list([])
    # 第一层可能有下采样或通道变化
    layers.append(block(in_channels, out_channels, stride))
    # 后面每一次输入输出通道都一致
    for i in range(1, num_blocks):
        layers.append(block(out_channels, out_channels, 1))

    return nn.Sequential(*layers)


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [24, 32, 64, 64, 64], [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [24, 32, 64, 64, 64], [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [24, 32, 64, 64, 64], [3, 4, 6, 3], **kwargs)
    return model


def main():
    net = resnet50()
    from torchsummary import summary

    summary(net, (1, 32, 32, 32))
    inputs = torch.randn(2, 1, 32, 32, 32)
    out = net(inputs)
    print(out[0].shape, out[1].shape)


if __name__ == '__main__':
    main()
