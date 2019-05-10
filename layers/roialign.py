# -*- coding: utf-8 -*-
"""
   File Name：     roialign.py
   Description :  RoiAlign
   Author :       mick.yi
   Date：          2019/5/8
"""

import torch
from torch import nn
from cuda_functions.roi_align_3D.roi_align.crop_and_resize import CropAndResizeFunction as crop_and_resize


class RoiAlign(nn.Module):
    # pool_size_h, pool_size_w, pool_size_t, image_size均为整数，不能是torch tensor
    def __init__(self, pool_size_h, pool_size_w, pool_size_d, image_size):
        """

        :param pool_size_h: 池化后的高度
        :param pool_size_w: 池化后的宽度
        :param pool_size_d: 池化后的厚度
        :param image_size: 图像尺寸，默认图像各维度尺寸一致
        """
        self.pool_size_h = pool_size_h
        self.pool_size_w = pool_size_w
        self.pool_size_d = pool_size_d
        self.image_size = image_size
        super(RoiAlign, self).__init__()

    def forward(self, features, rois, rois_indices):
        """

        :param: features [batch,C,H,W,T]
        :param: rois [roi_num,(y1,x1,z1,y2,x2,z2)]
        :param: rois_indices [roi_num]  指示roi属于batch中哪个样本
        :return: 池化后的特征  [roi_num,C,ph,pw,pt]  pt代表pool_size_h

        """
        # 坐标归一化
        boxes = rois / self.image_size
        # 转换维度
        boxes = boxes[:, [0, 1, 3, 4, 2, 5]]  # [n,(y1,x1,z1,y2,x2,z2)] => [n,(y1,x1,y2,x2,z1,z2)]
        batch_indices = rois_indices.int()
        x = crop_and_resize(self.pool_size_h, self.pool_size_w, self.pool_size_d, 0)(features, boxes, batch_indices)
        return x
