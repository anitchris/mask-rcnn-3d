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
        super(RpnHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv_head = nn.Conv3d(128, self.num_anchors * 7, kernel_size=1)

    def forward(self, x):
        out_head = self.conv_head(x)
        size = out_head.size()
        b, h, w, d = size[0], size[2], size[3], size[4]
        rpn_target = out_head.view(b, 7, h * w * d * self.num_anchors)
        rpn_target = rpn_target.transpose((0,2,1))
        return rpn_target


class MrcnnHead(nn.Module):
    def __init__(self, cfg):
        """
        构造函数
        :param cfg: 配置上下文
        """
        super(MrcnnHead, self).__init__()
        self.in_channel = cfg.IN_CHANNEL
        self.kernel_size = cfg.KERNEL_SIZE,
        self.num_classes = cfg.NUM_CLASSES
        self.out_channel_branch1 = cfg.OUT_CHANNEL_BRANCH1
        self.out_channel_branch2 = cfg.OUT_CHANNEL_BRANCH2
        self.pool_size_h = cfg.POOL_SIZE_H
        self.pool_size_w = cfg.POOL_SIZE_W
        self.pool_size_t = cfg.POOL_SIZE_T
        self.flatten_features = cfg.OUT_CHANNEL_BRANCH1 * cfg.POOL_SIZE_H * cfg.POOL_SIZE_W * cfg.POOL_SIZE_T

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
