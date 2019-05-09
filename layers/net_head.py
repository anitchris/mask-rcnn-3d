# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/5/9 
@file: net_head.py
@description:
"""
import torch
from torch import nn
from layers.base_net import Net


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
        return rpn_target


if __name__ == '__main__':
    net = Net(3)
    input = torch.randn(2, 1, 32, 32, 32)
    out = net(input)
    print(out.shape)
    rpn_target = RpnHead(3)(out)
    print(rpn_target.size())
