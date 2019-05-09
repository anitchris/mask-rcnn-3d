# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/5/9 
@file: net_head.py
@description:
"""
import torch
from torch import nn
import torch.nn.functional as F
from layers.base_net import Net


class RpnHead(nn.Module):
    def __init__(self, num_anchors):
        super(RpnHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv_regr = nn.Conv3d(128, self.num_anchors * 6, kernel_size=1)
        self.conv_cls = nn.Conv3d(128, self.num_anchors * 2, kernel_size=1)

    def forward(self, x):
        out_regr = self.conv_regr(x)
        out_cls = self.conv_cls(x)
        out_cls = F.softmax(out_cls, dim=0)
        size = out_regr.size()
        b, h, w, d = size[0], size[2], size[3], size[4]
        rpn_regr_pred = out_regr.view(b, 6, h * w * d * self.num_anchors)
        rpn_cls_pred = out_cls.view(b, 2, h * w * d * self.num_anchors)
        return rpn_regr_pred, rpn_cls_pred


if __name__ == '__main__':
    net = Net(3)
    input = torch.randn(2, 1, 32, 32, 32)
    out = net(input)
    print(out.shape)
    rpn_regr_pred, rpn_cls_pred = RpnHead(3)(out)
    print(rpn_regr_pred.size(), rpn_cls_pred.size())
