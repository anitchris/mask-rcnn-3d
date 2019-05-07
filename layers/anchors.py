# -*- coding: utf-8 -*-
"""
   File Name：     anchors.py
   Description :   生成anchors
   Author :       mick.yi
   Date：          2019/5/7
"""
import numpy as np


def generate_base_anchors(scales):
    """
    根据scale尺寸生成3d anchors坐标；默认长宽高尺寸一致
    :param scales: 尺寸列表
    :return:
    """
    scales = np.array(scales, np.float)  # [n]
    # [n,(y1,x1,z1,y2,x2,z2)]
    return np.stack([-0.5 * scales, -0.5 * scales, -0.5 * scales,
                     0.5 * scales, 0.5 * scales, 0.5 * scales], axis=1)


def shift(base_anchors, stride, h, w, z):
    """
    在feature map 移动anchors的中心
    :param base_anchors: [n,(y1,x1,z1,y2,x2,z2)]
    :param stride: 步长,默认各个维度步长一致
    :param h: feature map的高度
    :param w: feature map宽度
    :param z: feature map厚度
    :return:
    """
    ctr_x = np.arange(0.5, w) * stride
    ctr_y = np.arange(0.5, h) * stride
    ctr_z = np.arange(0.5, z) * stride
    ctr_x, ctr_y, ctr_z = np.meshgrid(ctr_x, ctr_y, ctr_z)  # 每一个维度都是[h,w,z]
    # anchors中心点坐标
    ctr = np.stack([np.reshape(ctr_y, [-1]),
                    np.reshape(ctr_x, [-1]),
                    np.reshape(ctr_z, [-1]),
                    np.reshape(ctr_y, [-1]),
                    np.reshape(ctr_x, [-1]),
                    np.reshape(ctr_z, [-1])], axis=1)  # [h*w*z,(h,w,z,h,w,z)]
    # 扩维[h*w*z,1,6]
    ctr = np.expand_dims(ctr, axis=1)
    base_anchors = np.expand_dims(base_anchors, axis=0)  # [1,n,6]

    # 打平返回
    return np.reshape(ctr + base_anchors, [-1, 6])  # [n*h*w*z,6]


def generate_anchors(scales, stride, h, w, z):
    """
    生成所有的anchors
    :param scales: anchors尺寸
    :param stride: 步长,默认各个维度步长一致
    :param h: feature map的高度
    :param w: feature map宽度
    :param z: feature map厚度
    :return: 所有的anchors [n,(y1,x1,z1,y2,x2,z2)]
    """
    base_anchors = generate_base_anchors(scales)
    return shift(base_anchors, stride, h, w, z)


def main():
    base_anchors = generate_anchors([2, 3])
    print("base_anchors:{}".format(base_anchors))
    anchors = shift(base_anchors, 4, 16, 16, 16)
    print("anchors:{}".format(anchors))


if __name__ == '__main__':
    main()
