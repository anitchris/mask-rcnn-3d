# -*- coding: utf-8 -*-
"""
   File Name：     target.py
   Description :  rpn和rcnn 分类、回归、分割目标
   Author :       mick.yi
   Date：          2019/5/8
"""
import torch
from torch import nn
import numpy as np
from utils.np_utils import iou_3d, regress_target_3d


class RpnTarget(nn.Module):
    def __init__(self, train_anchors_per_image=800, positive_iou_threshold=0.5,
                 negative_iou_threshold=0.02, train_positive_anchors=2):
        self.train_anchors_per_image = train_anchors_per_image
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.train_positive_anchors = train_positive_anchors
        super(RpnTarget, self).__init__()

    def forward(self, anchors, gt_boxes, gt_labels):
        """

        :param anchors: numpy 数组[anchors_num,(y1,x1,z1,y2,x2,z2)]
        :param gt_boxes: list of numpy [n,(y1,x1,z1,y2,x2,z2)]
        :param gt_labels: list of numpy [n]
        :return: anchors_tag: torch tensor [batch,anchors_num]  1-正样本，-1-负样本，0-不参与训练
                 deltas: torch tensor  [anchors_num,(dy,dx,dz,dh,dw,dd)]
        """
        batch_anchors_tag = []
        batch_deltas = []
        # 逐个样本处理
        for boxes, labels in zip(gt_boxes, gt_labels):
            iou = iou_3d(boxes, anchors)  # [gt_num,anchors_num]
            anchors_iou_max = np.max(iou, axis=0)  # [anchors_num]
            anchors_tag = np.zeros_like(anchors[:, 0], np.int)  # [anchors_num]
            # 正负样本
            pos_indices = np.where(anchors_iou_max >= self.positive_iou_threshold)[0]
            neg_indices = np.where(anchors_iou_max <= self.negative_iou_threshold)[0]
            # 正样本采样
            pos_num = min(pos_indices.shape[0], self.train_positive_anchors)
            pos_indices = np.random.shuffle(pos_indices)[:pos_num]
            anchors_tag[pos_indices] = 1
            # 负样本采样
            neg_num = min(neg_indices.shape[0], self.train_anchors_per_image - pos_num)
            neg_indices = np.random.shuffle(neg_indices)[:neg_num]
            anchors[neg_indices] = -1

            # 计算回归目标
            anchors_iou_argmax = np.argmax(iou, axis=0)  # [anchors_num]
            pos_gt_indices = anchors_iou_argmax[pos_indices]  # 正样本对应的gt索引号
            pos_gt_boxes = boxes[pos_gt_indices]
            pos_anchors = anchors[pos_indices]
            deltas = np.zeros_like(anchors)  # [anchors_num,(dy,dx,dz,dh,dw,dd)]
            deltas[pos_indices] = regress_target_3d(pos_anchors, pos_gt_boxes)

            # 转为tensor
            batch_anchors_tag.append(torch.from_numpy(anchors_tag).cuda())
            batch_deltas.append(torch.from_numpy(deltas).cuda())

        batch_anchors_tag = torch.stack(batch_anchors_tag, dim=0)
        batch_deltas = torch.stack(batch_deltas, dim=0)
        return batch_anchors_tag, batch_deltas
