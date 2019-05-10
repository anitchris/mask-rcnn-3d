# -*- coding: utf-8 -*-
"""
   File Name：     losses.py
   Description :   损失函数层
   Author :       mick.yi
   Date：          2019/5/9
"""
import torch.nn.functional as F
from utils import torch_utils


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


def rpn_regress_loss(deltas, predict_deltas, anchors_tag):
    """
    rpn 回归损失
    :param deltas: 真实回归目标torch tensor [batch,anchors_num,(dy,dx,dz,dh,dw,dd)]
    :param predict_deltas: 预测回归目标torch tensor [batch,anchors_num,(dy,dx,dz,dh,dw,dd)]
    :param anchors_tag: torch tensor [batch,anchors_num]  1-正样本，-1-负样本，0-不参与训练
    :return:
    """
    # 获取训练的anchors的索引号
    ix = (anchors_tag == 1).nonzero()  # 只有正样本回归
    deltas = deltas[ix[:, 0], ix[:, 1]]  # [n,(dy,dx,dz,dh,dw,dd)]
    predict_deltas = predict_deltas[ix[:, 0], ix[:, 1]]  # [n,(dy,dx,dz,dh,dw,dd)]
    loss = F.smooth_l1_loss(predict_deltas, deltas)  # 标量
    return loss


def mrcnn_cls_loss(rois_labels, predict_logits):
    """
    mrcnn 分类损失
    :param rois_labels: torch tensor [rois_num]
    :param predict_logits: torch tensor[rois_num,num_classes]
    :return:
    """
    # 转one hot编码
    num_classes = predict_logits.size(1)
    labels = torch_utils.one_hot(rois_labels, num_classes)
    loss = F.cross_entropy(predict_logits, labels)  # 标量
    return loss


def mrcnn_regress_loss(rois_deltas, predict_deltas, rois_labels):
    """
    mrcnn 回归损失
    :param rois_deltas: torch tensor [rois_num,(dy,dx,dz,dh,dw,dd)]
    :param predict_deltas: torch tensor [rois_num,(dy,dx,dz,dh,dw,dd)]
    :param rois_labels: torch tensor [rois_num]
    :return:
    """
    # 只有正样本计算损失
    ix = (rois_labels > 0).nonzero()[:, 0]  # [pos_rois_num]
    rois_deltas = rois_deltas[ix]
    predict_deltas = predict_deltas[ix]
    loss = F.smooth_l1_loss(predict_deltas, rois_deltas)  # 标量
    return loss


def mrcnn_mask_loss(mask, predict_mask_logits, rois_labels):
    """
    mrcnn mask损失
    :param mask: 真实的 mask torch tensor [rois_num,y,x,z] 0,1值
    :param predict_mask_logits: torch tensor [rois_num,y,x,z,num_classes]
    :param rois_labels:真实的类别 torch tensor[rois_num]
    :return:
    """
    predict_mask = F.softmax(predict_mask_logits, dim=-1)  # 转为得分
    # 只处正样本区域
    ix = (rois_labels > 0).nonzero()[:, 0]  # [pos_rois_num]

    rois_labels = rois_labels[ix]
    predict_mask = predict_mask[ix, rois_labels]  # 对应的样本，对应的类别预测的mask值
    mask = mask[ix]

    loss = F.binary_cross_entropy(predict_mask, mask)
    return loss
