# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/5/8 
@file: regr_target.py
@description:
"""
import torch


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


if __name__ == '__main__':
    def regress_target_3d_test():
        gt_boxes = torch.Tensor([[1.1, 2, 3, 12, 32, 43], [1, 2, 3, 22, 42, 13]])
        anchors = torch.Tensor([[6, 9, 9, 12, 32, 43], [1, 2.2, 3, 22, 42, 13]])
        target = regress_target_3d(anchors, gt_boxes)
        print(target)


    def apply_regress_3d_test():
        deltas = torch.Tensor([[1.1, 2, 3, 12, 32, 43], [1, 2, 3, 22, 42, 13]])
        anchors = torch.Tensor([[6, 9, 9, 12, 32, 43], [1, 2.2, 3, 22, 42, 13]])
        anchors_refined = apply_regress_3d(deltas, anchors)
        print(anchors_refined)


    regress_target_3d_test()
    apply_regress_3d_test()
