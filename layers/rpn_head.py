# -*- coding:utf-8 -*-
"""
   File Name：     rpn_head.py
   Description :   rpn的网络结构部分（输入、输出待修改）
   Author :        royce.mao
   date：          2019/05/07
"""

import torch.nn.functional as func
from torch import nn

from .box_coder import BoxCoder  # bounding boxes的编码、解码（参考的facebookresearch/maskrcnn-benchmark）
from .loss import Loss
from .anchors import Anchor3D
from .target import Target3D


class RPN(nn.Module):
    def __init__(self, cfg, in_channels, stage='train'):
        """
        RPN HEAD
        :param cfg: 
        :param in_channels: MODEL.BACKBONE.OUT_CHANNELS（基网络返回的feature_map对应的channels）
        """
        super(RPN, self).__init__()
        self.cfg = cfg.clone()
        # 生成3D_Anchors函数（待完成）
        anchors = Anchor3D(cfg)
        num_anchors = anchors.num_anchors_per_location()[0]
        # RPN_HEAD的部分卷积网络层变量
        self.conv = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        # 输出层变量
        self.cls_logits = nn.Conv3d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv3d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1  # (z, y, x, diameter) or (z,y,x,z,y,x) ？
        )
        # 参数初始化
        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(l.weight, std=0.01)  # 正态分布
            nn.init.constant_(l.bias, 0)  # 常数
        # RPN target box采样相关（待完成）
        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))  # ?
        targets = Target3D(cfg, rpn_box_coder, is_train=True)
        proposals = Target3D(cfg, rpn_box_coder, is_train=False)
        # loss相关（待完成）
        loss = Loss(cfg, rpn_box_coder)
        # anchors、targets、losses变量相关
        self.stage = stage
        self.anchors = anchors
        self.targets = targets
        self.loss = loss
        self.proposals = proposals

    def forward(self, images, feature):
        # HEAD中间卷积层与输出
        t = func.relu(self.conv(feature))
        rpn_cls = self.cls_logits(t)
        rpn_reg = self.bbox_pred(t)
        # HEAD在指定feature map上生成anchors
        anchors = self.anchors(images, feature)
        # 训练阶段（根据cls_tag, box_regression_deltas，生成用于训练的分类、回归目标与loss）
        if self.stage == 'train':
            # with torch.no_grad():  # torch 1.0.1版本warning
            # 需要做anchors采样，得到进入training batch用于训练的正、负样本boxes（不是返回的cls_tag与reg_deltas目标？）
            anchors_tag, deltas = self.targets(
                anchors, rpn_cls, rpn_reg
            )
            loss_cls, loss_reg = self.loss(
                anchors_tag, deltas, rpn_cls, rpn_reg
            )
            losses = {
                "loss_cls": loss_cls,
                "loss_reg": loss_reg,
            }
            return losses
        # 测试阶段（根据score排序，生成roi）
        else:
            boxes = self.proposals(anchors, rpn_cls, rpn_reg)
            inds = [
                box.get_field("score").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]

            return boxes
