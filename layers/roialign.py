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
    def __init__(self, pool_size_h, pool_size_w, pool_size_t, image_size):
        self.pool_size_h = pool_size_h
        self.pool_size_w = pool_size_w
        self.pool_size_t = pool_size_t
        self.image_size = image_size
        super(RoiAlign, self).__init__()

    def forward(self, features, rois):
        """

        :param: features [batch,C,H,W,T]
        :param: rois [roi_num,(y1,x1,z1,y2,x2,z2,batch_index)]
        :return: 池化后的特征  [roi_num,C,ph,pw,pt]

        """
        # 坐标归一化
        boxes = rois[:, :6] / self.image_size
        batch_indices = rois[:, 6].int()
        x = crop_and_resize(self.pool_size_h, self.pool_size_w, self.pool_size_t, 0)(features, boxes, batch_indices)
        return x
