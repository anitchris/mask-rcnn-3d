# -*- coding:utf-8 -*-
"""
   File Name：     detector.py
   Description :   mrcnn网络输出还原映射为原图上的bbox与mask
   Author :        royce.mao
   date：          2019/05/10
"""
import torch
import torchvision
from torch import nn
import torch.utils.data
from utils.torch_utils import apply_regress_3d
from cuda_functions.nms_3D.pth_nms import nms_gpu as nms_3d


class Detector(nn.Module):
    def __init__(self, pre_nms_limit, nms_threshold, max_output_num, min_score=0.):
        self.pre_nms_limit = pre_nms_limit
        self.nms_threshold = nms_threshold
        self.max_output_num = max_output_num
        self.min_score = min_score
        self.pil = torchvision.transforms.ToPILImage()  # torch tensor转PIL对象
        self.tensor = torchvision.transforms.Compose([
                     torchvision.transforms.ToTensor()])  # PIL转torch tensor对象
        super(Detector, self).__init__()

    def forward(self, proposals, predict_scores, predict_deltas, predict_masks):
        """
        proposal坐标根据deltas还原，mask的resize加偏移还原（这里的batch是1张完整3D图像的crop数量）
        :param proposals: [batch, N, (y1, x1, z1, y2, x2, z2)]
        :param predict_scores: [batch, N]
        :param predict_deltas: [batch, N, (dy, dx, dz, dh, dw, dd)]
        :param predict_masks: [batch, N, 1, 28, 28, 28] 从RoiAlign后的7*7*7有4倍上采样
        :return: 
        batch_bboxes: [batch, N, (y1, x1, z1, y2, x2, z2)] 原图上真实bbox坐标
        batch_masks: [batch, N, 1, z2-z1, y2-y1, x2-x1] 原图上真实bbox区域对应mask像素值
        batch_indices: [batch*N, 1] 原图上真实bbox的batch索引
        """
        batch_bboxes = []
        batch_masks = []
        batch_indices = []
        # batch_scores = []
        # 逐个样本处理
        batch_size = predict_scores.shape[0]
        for bix in range(batch_size):
            # 应用proposals还原（apply回归目标）
            scores, order = torch.sort(predict_scores[bix], descending=True)
            order = order[:self.pre_nms_limit]
            scores = scores[:self.pre_nms_limit]
            deltas = predict_deltas[bix, order, :]
            cur_proposals = torch.clone(proposals)
            bboxes = apply_regress_3d(deltas, cur_proposals[bix, order, :])
            keep = nms_3d(torch.cat((bboxes, scores.unsqueeze(-1)), -1),
                          self.nms_threshold)  # [N,(y1,x1,z1,y2,x2,z2,scores)]
            keep = keep[:self.max_output_num]
            batch_bboxes.append(keep[:, :-1])
            # 生成batch_indices
            indices = torch.Tensor([bix] * keep.shape[0])
            batch_indices.append(indices)
            # RoiAlign和偏移后的resize还原目标
            dz = keep[:, 5] - keep[:, 2]
            dy = keep[:, 3] - keep[:, 0]
            dx = keep[:, 4] - keep[:, 1]
            ratios = (dz, dy, dx)
            resize = torchvision.transforms.Resize(ratios)  # resize对象
            # 应用mask还原（基网络4倍下采样，mrcnn4倍上采样，所以做RoiAlign和偏移后的resize还原，对应到原图区域即可）
            masks = predict_masks[bix, order, :]
            masks_resized = self.tensor(resize(self.pil(masks)))  # 28*28*28到应用回归后的keep的resize
            batch_masks.append(masks_resized)

        # 在batch上打平，之前的batch_size维度没有了
        batch_bboxes = torch.cat(batch_bboxes, dim=0)
        batch_masks = torch.cat(batch_masks, dim=0)
        batch_indices = torch.cat(batch_indices, dim=0)
        # batch_scores = torch.cat(batch_scores, dim=0)

        return batch_bboxes, batch_masks, batch_indices
