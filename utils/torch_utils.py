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
    overlaps = torch.max(torch.min(boxes_a[..., 3:], boxes_b[..., 3:]) -
                         torch.max(boxes_a[..., :3], boxes_b[..., :3]),
                         zero)  # [N,M,3]
    overlaps = torch.prod(overlaps, dim=-1)  # [N,M]

    # 计算各自体积
    volumes_a = torch.prod(boxes_a[..., 3:] - boxes_a[..., :3], dim=-1)  # [N,1]
    volumes_b = torch.prod(boxes_b[..., 3:] - boxes_b[..., :3], dim=-1)  # [1,M]

    # 计算iou
    iou = overlaps / (volumes_a + volumes_b - overlaps)
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
    dh = torch.log(gt_h / h)
    dw = torch.log(gt_w / w)
    dd = torch.log(gt_d / d)

    target = torch.stack((dy, dx, dz, dh, dw, dd), dim=1)
    return target


def apply_regress_3d(deltas, anchors):
    """
    应用回归目标到边框,用rpn网络预测的delta refine anchor
    :param deltas: [N,(dy,dx,dz,dh,dw,dd)]
    :param anchors: [N, (y1, x1, z1, y2, x2, z2)]
    :return: [N, (y1, x1, z1, y2, x2, z2)]
    """
    # 高度、宽度、深度
    h = anchors[:, 3] - anchors[:, 0]
    w = anchors[:, 4] - anchors[:, 1]
    d = anchors[:, 5] - anchors[:, 2]

    # 中心点坐标
    cy = (anchors[:, 3] + anchors[:, 0]) * 0.5
    cx = (anchors[:, 4] + anchors[:, 1]) * 0.5
    cz = (anchors[:, 5] + anchors[:, 2]) * 0.5

    # 回归系数
    # deltas *= tf.constant([0.1, 0.1, 0.2, 0.2])
    dy, dx, dz, dh, dw, dd = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3], deltas[:, 4], deltas[:, 5]

    # 中心坐标调整
    cy += dy * h
    cx += dx * w
    cz += dz * d
    # 高度宽度深度调整
    h *= torch.exp(dh)
    w *= torch.exp(dw)
    d *= torch.exp(dd)

    # 转为y1,x1,z1,y2,x2,z2
    y1 = cy - h * 0.5
    x1 = cx - w * 0.5
    z1 = cz - d * 0.5
    y2 = cy + h * 0.5
    x2 = cx + w * 0.5
    z2 = cz + w * 0.5

    anchors_refined = torch.stack((y1, x1, z1, y2, x2, z2), dim=1)
    return anchors_refined


def main():
    boxes_a = torch.Tensor([[1, 2, 3, 12, 32, 43], [1, 2, 3, 22, 42, 13]])
    boxes_b = torch.Tensor([[6, 9, 9, 12, 32, 43], [1, 2, 3, 22, 42, 13], [22, 22, 23, 42, 42, 63]])

    iou = iou_3d(boxes_a, boxes_b)
    print("iou:{}".format(iou.numpy()))


if __name__ == '__main__':
    main()
