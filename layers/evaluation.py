# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/5/10 
@file: evaluation.py
@description:
"""

from utils.torch_utils import iou_3d
import torch


def eval_3d(gt_boxes, pred_boxes, iou_thread):
    """ 给定一个iou阈值，根据gt boxes和网络预测的boxes（经nms之后的）计算准确率指标
    :param gt_boxes: 2-d tensor [n,(y1,x1,z1,y2,x2,z2)]
    :param pred_boxes: 2-d tensor [m,(y1,x1,z1,y2,x2,z2)]
    :param iou_thread: eg.0.5
    :return:
    """
    iou = iou_3d(gt_boxes, pred_boxes)
    iou_sign = torch.where(iou >= iou_thread, torch.ones_like(iou), torch.zeros_like(iou))
    row_max = torch.max(iou_sign, 1)[0]
    col_sum = torch.sum(iou_sign, 0)
    row_sum = torch.sum(iou_sign, 1)
    tp = torch.sum(row_max).item()
    tn = 0
    fp = row_sum[row_sum > 1].size()[0] + col_sum[col_sum == 0].size()[0]
    fn = row_max[row_max == 0].size()[0]
    sensitivity = recall = tp / (tp + fn)
    specificity = tn / (fp + tn)  # 总为0，因为真实和预测的box里没有负样本
    precision = tp / (tp + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    metrics = sensitivity, specificity, precision, f1
    return [round(i, 2) for i in metrics]


# if __name__ == '__main__':
#     iou_thread = 0.5
#     gt_boxes = torch.Tensor([[1, 2, 3, 12, 32, 43], [1, 2, 3, 22, 42, 13], [13, 2, 3, 16, 32, 43]])
#     pred_boxes = torch.Tensor([[1, 9, 9, 12, 32, 43], [2, 9, 9, 12, 32, 43],
#                                [1, 2, 3, 22, 42, 13], [22, 22, 23, 42, 42, 63]])
#     sensitivity, specificity, precision, f1 = eval_3d(gt_boxes, pred_boxes, iou_thread)
#     print(sensitivity, specificity, precision, f1)
