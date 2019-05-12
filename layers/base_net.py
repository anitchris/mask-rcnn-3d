# -*- coding:utf-8 -*-
"""
   File Name：     base_net.py
   Description :   res18+unet的基网络部分
   Author :        royce.mao
   date：          2019/05/06
"""

import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self,
                 block,
                 channels=[24, 32, 64, 64, 64],
                 blocks=[2, 2, 2, 2],
                 decode_channel=[64, 64, 64, 64, 64],
                 **kwargs):
        """

        :param block: BasicBlock 或 Bottleneck
        :param channels: resnet网络的5个stage的输出通道数;
        :param blocks: resnet网络stage 2~5 的block块数，不同的block块数对应不同的网络层数;
        :param decode_channel: unet decoder部分输出通道数
        :param kwargs:
        """
        super(UNet, self).__init__()
        c1, c2, c3, c4, c5 = channels  # 通道数
        dc0, dc1, dc2, dc3, dc4 = decode_channel  # unet解码部分通道数

        self.bone_net = ResNet(block, channels, blocks, **kwargs)
        self.up5 = _up(c5, c5)
        self.up4 = _up(dc4, dc4)
        self.up3 = _up(dc3, dc3)
        self.up2 = _up(dc2, dc2)
        self.up1 = _up(dc1, dc1)
        # 每个stage上采样合并后,使用一个block块融合特征
        self.decode_stage4 = _make_stage(block, c5 + c4, dc4, 1)
        self.decode_stage3 = _make_stage(block, dc4 + c3, dc3, 1)
        self.decode_stage2 = _make_stage(block, dc3 + c2, dc2, 1)
        self.decode_stage1 = _make_stage(block, dc2 + c1, dc1, 1)
        self.decode_stage0 = _make_stage(block, dc1 + 1, dc0, 1)

    def forward(self, x):
        f5, f4, f3, f2, f1 = self.bone_net(x)
        p5 = f5
        up5 = self.up5(p5)
        comb4 = torch.cat((f4, up5), dim=1)  # [b,2*c5,y,x,z]
        p4 = self.decode_stage4(comb4)

        up4 = self.up4(p4)
        comb3 = torch.cat((f3, up4), dim=1)  # [b,dc4 + c4,y,x,z]
        p3 = self.decode_stage3(comb3)

        up3 = self.up3(p3)
        comb2 = torch.cat((f2, up3), dim=1)  # [b,dc3 + c3,y,x,z]
        p2 = self.decode_stage2(comb2)

        up2 = self.up2(p2)
        comb1 = torch.cat((f1, up2), dim=1)  # [b,dc2 + c2,y,x,z]
        p1 = self.decode_stage1(comb1)

        up1 = self.up2(p1)
        comb0 = torch.cat((x, up1), dim=1)  # [b,dc1 + c1,y,x,z]
        p0 = self.decode_stage0(comb0)

        return p0


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 channels=[24, 32, 64, 64, 64],
                 blocks=[2, 2, 2, 2]):
        """
        resnet 基础网络
        :param block: BasicBlock 或 Bottleneck
        :param channels: resnet网络的5个stage的输出通道数;
        :param blocks: resnet网络stage 2~5 的block块数，不同的block块数对应不同的网络层数;
        """
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
        x = f1 = self.stage1(x)
        x = f2 = self.stage2(x)
        x = f3 = self.stage3(x)
        x = f4 = self.stage4(x)
        f5 = self.stage5(x)

        return f5, f4, f3, f2, f1


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


def _up(in_channels, out_channels):
    """
    上采样过程
    :param in_channels:
    :param out_channels:
    :return:
    """
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True))


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
    net = UNet(BasicBlock)
    from torchsummary import summary

    summary(net, (1, 32, 32, 32))
    inputs = torch.randn(2, 1, 32, 32, 32)
    out = net(inputs)
    print(out[0].shape, out[1].shape)


if __name__ == '__main__':
    main()
