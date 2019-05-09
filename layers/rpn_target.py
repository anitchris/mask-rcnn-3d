# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/5/8 
@file: rpn_target.py
@description:
"""
from utils.torch_utils import iou_3d
import torch


def rpn_targets_graph(gt_boxes, anchors, rpn_train_anchors=None):
    """
    :param gt_boxes: [num_gt,(dy,dx,dz,dh,dw,dd)]
    :param anchors: [num_anchor,(dy,dx,dz,dh,dw,dd)]
    :param rpn_train_anchors: int,eg.256
    :return:
    """
    pos_iou_thresh = 0.7
    anchor_sign = torch.full((anchors.size()),0)

    # compute iou matrix
    iou = iou_3d(gt_boxes,anchors)
    # set positive anchor if iou>0.7
    pos_iou = torch.where(iou>=pos_iou_thresh,torch.ones_like(iou),torch.zeros_like(iou))
    anchor_sign_pos = torch.max(pos_iou,dim=0)[0]

    # set postive anchor if it has biggest iou with gt
    row_max = torch.argmax(iou,dim=1,keepdim=True)
    anchor_sign_pos[]

    #print(iou[col_max])

    # set negtive anchor if iou<0.3

    # sample and pad anchors to have similar number of positive anchor and negtive anchor
    # calculate deltas target for positive anchors
    # calculate cls target for sampled anchors
    print(iou,'\n',row_max,'\n',pos_iou,'\n',anchor_sign_pos)
    return None


gt_boxes = torch.Tensor([[1.1, 2, 3, 12, 32, 43], [1, 2, 3, 22, 42, 13]])
anchors = torch.Tensor([[6, 9, 9, 12, 32, 43], [1, 2.2, 3, 22, 42, 13], [1, 2.2, 3, 22, 42, 33]])
rpn_train_anchors = 2
rpn_targets_graph(gt_boxes, anchors, rpn_train_anchors)