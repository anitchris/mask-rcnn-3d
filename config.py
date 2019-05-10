# -*- coding: utf-8 -*-
"""
   File Name：     config.py
   Description :   配置类
   Author :       mick.yi
   Date：          2019/5/7
"""
import numpy as np


class Config(object):
    DATA_DIR = '/home/dataset/medical/jida_dicom/subset/'
    NUM_CLASSES = 2

    CROP_SIZE = [128, 128, 128]
    IMAGE_SIZE = CROP_SIZE[0]
    BOUND_SIZE = 12
    STRIDE = 4
    PAD_VALUE = 0
    # anchors
    ANCHOR_SCALES = [2, 3, 4]
    # rpn网络
    TRAIN_ANCHORS_PER_IMAGE = 800
    # proposals
    PRE_NMS_LIMIT = 6000
    RPN_NMS_THRESHOLD = 0.5
    # 训练和预测阶段NMS后保留的ROIs数
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # mrcnn 网络
    POOL_SIZE_HEIGHT = 7
    POOL_SIZE_WIDTH = 7
    POOL_SIZE_DEPTH = 7
    MASK_POOL_SIZE = 28
    TRAIN_ROIS_PER_IMAGE = 64

    BATCH_SIZE = 1

    def __init__(self):
        super(Config, self).__init__()
        self.NUM_ANCHORS = len(self.ANCHOR_SCALES)
        # feature map的高度，宽度，厚度
        self.FEATURES_HEIGHT, self.FEATURES_WIDTH, self.FEATURES_DEPTH = np.array(self.CROP_SIZE) // self.STRIDE


cur_config = Config()
