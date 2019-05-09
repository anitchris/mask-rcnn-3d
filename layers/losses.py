# -*- coding: utf-8 -*-
"""
   File Name：     losses.py
   Description :   损失函数层
   Author :       mick.yi
   Date：          2019/5/9
"""
import torch.nn.functional as F


def rpn_cls_loss(anchors_tag, predict_logits):
    """
    rpn分类损失
    :param anchors_tag:  torch tensor [batch,anchors_num]  1-正样本，-1-负样本，0-不参与训练
    :param predict_logits: torch tensor [batch,anchors_num] 预测一个值
    :return:
    """
    # 获取训练的anchors的索引号
    ix = (anchors_tag != 0).nonzero()
    labels = anchors_tag[ix[:, 0], ix[:, 1]]
    predict_logits = predict_logits[ix[:, 0], ix[:, 1]]
    loss = F.binary_cross_entropy_with_logits(predict_logits, labels)  # 标量
    return loss


def rpn_regress_loss():
    pass


def mrcnn_cls_loss():
    pass


def mrcnn_regress_loss():
    pass


def mrcnn_mask_loss():
    pass
