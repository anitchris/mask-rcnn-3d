# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/5/9 
@file: net_head.py
@description:
"""
from torch import nn


class RpnHead(nn.Module):
    def __init__(self, num_anchors):
        """
        :param num_anchors: int, number of anchor per feature map grid
        """
        super(RpnHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv_head = nn.Conv3d(128, self.num_anchors * 7, kernel_size=1)

    def forward(self, x):
        """
        :param x: 5-d feature map from base_net,[b,c,h,w,d]
        :return: rpn_target,[b,total_num_anchor,6+1]
        """
        out_head = self.conv_head(x)
        size = out_head.size()
        b, h, w, d = size[0], size[2], size[3], size[4]
        rpn_target = out_head.view(b, 7, h * w * d * self.num_anchors)
        rpn_target = rpn_target.transpose((0, 2, 1))
        return rpn_target


class MrcnnHead(nn.Module):
    def __init__(self, in_channel=128, kernel_size=3, num_classes=2, out_channel_branch1=16,
                 out_channel_branch2=16, pool_size_h=7, pool_size_w=7, pool_size_d=7):
        """
        构造函数
        :param in_channel: int, 输入通道大小
        :param kernel_size: int, 卷积核大小
        :param num_classes: int, 类别数量
        :param out_channel_branch1: int, 分支1的输出通道大小
        :param out_channel_branch2: int, 分支2的输出通道大小
        :param pool_size_h: int, roi align后的高
        :param pool_size_w: int, roi align后的宽
        :param pool_size_d: int, roi align后的深度
        """
        super(MrcnnHead, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.out_channel_branch1 = out_channel_branch1
        self.out_channel_branch2 = out_channel_branch2
        self.pool_size_h = pool_size_h
        self.pool_size_w = pool_size_w
        self.pool_size_d = pool_size_d
        self.flatten_features = out_channel_branch1 * pool_size_h * pool_size_w * pool_size_d

        # branch1, 得到class和regr box
        self.branch_1 = nn.Sequential(
            nn.Conv3d(self.in_channel, self.out_channel_branch1, kernel_size=self.kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.out_channel_branch1, self.out_channel_branch1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Linear(self.flatten_features, self.num_classes)
        self.regr = nn.Linear(self.flatten_features, self.num_classes * 6)

        # branch2, 得到mask
        self.branch_2 = nn.Sequential(
            nn.Conv3d(self.in_channel, self.out_channel_branch2, kernel_size=self.kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.out_channel_branch2, self.out_channel_branch2, kernel_size=self.kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.out_channel_branch2, self.out_channel_branch2, kernel_size=self.kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.out_channel_branch2, self.out_channel_branch2, kernel_size=self.kernel_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(self.out_channel_branch2, self.out_channel_branch2, kernel_size=2, stride=2),
            nn.Conv3d(self.out_channel_branch2, self.num_classes, kernel_size=1, stride=1)
        )

    def forward(self, rois):
        """
        :param rois: tensor, shape[roi_num, channel, height, weight, depth]
        :return:
                cls: tensor, shape[roi_num, num_classes]
                regr: tensor, shape[roi_num, (dy,dx,dz,dh,dw,dd)]
                mask: tensor, shape[roi_num, height, weight, depth, channel]
        """
        out1 = self.branch_1(rois)
        out1 = out1.view(-1, self.flatten_features)
        cls = self.cls(out1)
        regr = self.regr(out1)
        mask = self.branch_2(rois)
        mask = mask.transpose((0, 2, 3, 4, 1))
        return cls, regr, mask

# if __name__ == '__main__':
#     import torch
#     from layers.base_net import Net
#     net = Net(3)
#     input = torch.randn(2, 1, 32, 32, 32)
#     out = net(input)
#     print(out.shape)
#     rpn_target = RpnHead(3)(out)
#     print(rpn_target.size())
