# -*- coding: utf-8 -*-
"""
   File Name：     torch_utils.py
   Description :  torch 工具类
   Author :       mick.yi
   Date：          2019/5/7
"""

import torch


def iou_3d(boxes_a, boxes_b):
    """
    3d iou
    :param boxes_a: [N, (y1, x1, z1, y2, x2, z2)]
    :param boxes_b: [M, (y1, x1, z1, y2, x2, z2)]
    :return: iou: [N,M}
    """
    # 扩维
    boxes_a = torch.unsqueeze(boxes_a, dim=1)  # [N,1,6]
    boxes_b = torch.unsqueeze(boxes_b, dim=0)  # [1,M,6]
    # 计算交集
    zero = torch.zeros(1)
    if boxes_a.is_cuda:
        zero = zero.cuda()
    overlaps = torch.max(torch.min(boxes_a[..., 3:], boxes_b[..., 3:])
                         - torch.max(boxes_a[..., :3], boxes_b[..., :3]),
                         zero)  # [N,M,3]
    overlaps = torch.prod(overlaps, dim=-1)  # [N,M]

    # 计算各自体积
    volumes_a = torch.prod(boxes_a[..., 3:] - boxes_a[..., :3], dim=-1)  # [N,1]
    volumes_b = torch.prod(boxes_b[..., 3:] - boxes_b[..., :3], dim=-1)  # [1,M]

    # 计算iou
    iou = overlaps / (volumes_a + volumes_b - overlaps)
    return iou
