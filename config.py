# -*- coding: utf-8 -*-
"""
   File Name：     config.py
   Description :   配置类
   Author :       mick.yi
   Date：          2019/5/7
"""


class Config(object):
    ANCHOR_SCALES = [2, 3, 4]
    BATCH_SIZE = 1

    CROP_SIZE = [128,128,128]
    BOUND_SIZE = 12
    STRIDE = 4
    PAD_VALUE = 0


cur_config = Config()
