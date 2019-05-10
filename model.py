# -*- coding: utf-8 -*-
"""
   File Name:     model.py
   Description:   网络
   Author:        steven.yi
   Date:          2019/5/10
"""
import torch
from torch import nn
from layers.base_net import Net
from layers.net_head import RpnHead, MrcnnHead
from layers.anchors import generate_anchors
from layers.proposals import Proposal
from layers.target import RpnTarget, MrcnnTarget
from layers.roialign import RoiAlign
from layers.losses import rpn_cls_loss, rpn_regress_loss, mrcnn_cls_loss, mrcnn_regress_loss, mrcnn_mask_loss


class LungNet(nn.Module):
    def __init__(self, cfg, parse="train"):
        """

        :param cfg: 配置上下文
        :param parse: "train" or "test"
        """
        super(LungNet, self).__init__()
        self.cfg = cfg
        self.phase = parse
        self.base_net = Net(cfg.NUM_ANCHORS)
        self.rpn_head = RpnHead(cfg.NUM_ANCHORS)
        self.mrcnn_head = MrcnnHead(cfg)
        self.anchors = generate_anchors(cfg.SCALES, cfg.STRIDE, cfg.HEIGHT,
                                        cfg.WEIGHT, cfg.DEPTH)
        self.proposal = Proposal(cfg.PRE_NMS_LIMIT, cfg.NMS_THRESHOLD, cfg.MAX_OUTPUT_NUM)
        self.mrcnn_traget = MrcnnTarget(cfg.TRAIN_ROIS_PER_IMAGE)
        self.rpn_target = RpnTarget()
        self.roi_align = RoiAlign(cfg.POOL_SIZE_H, self.POOL_SIZE_T, self.POOL_SIZE_T, cfg.IMAGE_SIZE)
        self.total_loss = None

    def forward(self, x, gt_boxes, gt_labels):
        """
        前向传播
        :param x: tensor, [Batch, Channel, D, H, W]
        :param gt_boxes: list of numpy [n,(y1,x1,z1,y2,x2,z2)]
        :param gt_labels: list of numpy [n]
        :return:
        """
        # 获取feature map
        feature_map = self.base_net(x)

        # 获取rpn的输出
        rpn_output = self.rpn_head(feature_map)
        predict_scores, predict_deltas = rpn_output[:, :, -1], rpn_output[:, :, :-1]
        # 获取anchors的真实类别和真实偏移量
        gt_anchors_tag, gt_anchors_deltas = self.rpn_target(self.anchors, gt_boxes, gt_labels)

        # 计算rpn阶段loss
        cls_loss_rpn = rpn_cls_loss(gt_anchors_tag, predict_scores)
        regr_loss_rpn = rpn_regress_loss(gt_anchors_deltas, predict_deltas, gt_anchors_tag)

        # 获取proposals
        batch_proposals, batch_scores, batch_indices = self.proposal(self.anchors, predict_scores, predict_deltas)
        # 获取mrcnn target
        batch_rois, gt_deltas, gt_labels, rois_indices = self.mrcnn_traget(batch_proposals, batch_indices,
                                                                                 gt_boxes, gt_labels)
        # 维度处理
        rois_indices = torch.unsqueeze(rois_indices, dim=1)
        rois = torch.cat((batch_rois, rois_indices), dim=1)
        # roi align
        rois = self.roi_align(feature_map, rois)

        # 获取mrcnn head输出
        predict_mrcnn_cls, predict_mrcnn_regr, predict_mask = self.mrcnn_head(rois)

        # 计算mrcnn阶段loss
        cls_loss_mrcnn = mrcnn_cls_loss(gt_labels, predict_mrcnn_cls)
        regr_loss_mrcnn = mrcnn_regress_loss(gt_deltas, predict_mrcnn_regr, gt_labels)
        gt_mask = None  # todo: 真实的mask将从mrcnn_target中返回，待完成
        mask_loss = mrcnn_mask_loss(gt_mask, predict_mask, gt_labels)

        # 总的Loss
        self.total_loss = cls_loss_rpn + regr_loss_rpn + cls_loss_mrcnn + regr_loss_mrcnn + mask_loss

        if self.phase == 'test':
            # todo: 预测过程
            pass

    # property装饰的方法可以直接当作属性调用
    @property
    def loss(self):
        return self.total_loss
