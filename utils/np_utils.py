# -*- coding: utf-8 -*-
"""
   File Name：     np_utils.py
   Description :   numpy 工具类
   Author :       mick.yi
   Date：          2019/5/8
"""
import numpy as np


def iou_3d(cubes_a, cubes_b):
    """
    numpy 计算IoU
    :param cubes_a: [N,(y1,x1,z1,y2,x2,z2)]
    :param cubes_b: [M,(y1,x1,z1,y2,x2,z2)]
    :return:  IoU [N,M]
    """
    # 扩维
    cubes_a = np.expand_dims(cubes_a, axis=1)  # [N,1,4]
    cubes_b = np.expand_dims(cubes_b, axis=0)  # [1,M,4]

    # 分别计算高度和宽度的交集
    overlap = np.maximum(0.0,
                         np.minimum(cubes_a[..., 3:], cubes_b[..., 3:]) -
                         np.maximum(cubes_a[..., :3], cubes_b[..., :3]))  # [N,M,(h,w,t)]

    # 交集
    overlap = np.prod(overlap, axis=-1)  # [N,M]

    # 计算面积
    area_a = np.prod(cubes_a[..., 3:] - cubes_a[..., :3], axis=-1)
    area_b = np.prod(cubes_b[..., 3:] - cubes_b[..., :3], axis=-1)

    # 交并比
    iou = overlap / (area_a + area_b - overlap)
    return iou


def regress_target_3d(anchors, gt_boxes):
    """
    计算回归目标,输入的gt和anchor为一一对应关系
    :param anchors: [N, (y1, x1, z1, y2, x2, z2)]
    :param gt_boxes: [N, (y1, x1, z1, y2, x2, z2)]
    :return: [N,(dy,dx,dz,dh,dw,dd)]
    """
    # 高度、宽度、深度
    h = anchors[:, 3] - anchors[:, 0]
    w = anchors[:, 4] - anchors[:, 1]
    d = anchors[:, 5] - anchors[:, 2]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 0]
    gt_w = gt_boxes[:, 4] - gt_boxes[:, 1]
    gt_d = gt_boxes[:, 5] - gt_boxes[:, 2]

    # 中心点
    center_y = (anchors[:, 3] + anchors[:, 0]) * 0.5
    center_x = (anchors[:, 4] + anchors[:, 1]) * 0.5
    center_z = (anchors[:, 5] + anchors[:, 2]) * 0.5
    gt_center_y = (gt_boxes[:, 3] + gt_boxes[:, 0]) * 0.5
    gt_center_x = (gt_boxes[:, 4] + gt_boxes[:, 1]) * 0.5
    gt_center_z = (gt_boxes[:, 5] + gt_boxes[:, 2]) * 0.5

    # 回归目标
    dy = (gt_center_y - center_y) / h
    dx = (gt_center_x - center_x) / w
    dz = (gt_center_z - center_z) / d
    dh = np.log(gt_h / h)
    dw = np.log(gt_w / w)
    dd = np.log(gt_d / d)

    target = np.stack([dy, dx, dz, dh, dw, dd], axis=1)
    #target /= np.array([0.1, 0.1, 0.2, 0.2])
    return target
