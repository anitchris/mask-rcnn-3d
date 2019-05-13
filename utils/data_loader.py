# -*- coding:utf-8 -*-
"""
   File Name：     data.py
   Description :   数据加载 train/val/test_loader（进行中）
   Author :        royce.mao
   date：          2019/05/08
"""

import os
import torch
import numpy as np
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from config import cur_config as cfg


class Data3Lung(Dataset):  # Dataset是一个包装类，用来将数据包装为Dataset类，方便传入DataLoader中（getitem与len方法一般不可少）
    def __init__(self, data_dir, crop, phase='train'):
        """
        所有患者img、mask、gt的数据初始化
        :param data_dir: subset地址
        :param crop: Crop对象
        :param phase: 
        """
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        idcs = list()
        if phase != 'test':
            idcs = os.listdir(os.path.join(data_dir, 'img'))
        # 路径
        self.img_file = [os.path.join(data_dir, 'img', '%s' % idx, '%s.npy' % idx) for idx in idcs]
        self.mask_file = [os.path.join(data_dir, 'mask', '%s' % idx, '%s.npy' % idx) for idx in idcs]
        self.gt_file = [os.path.join(data_dir, 'gt', '%s' % idx, '%s.npy' % idx) for idx in idcs]
        # 数组
        imgs, masks, gts = [], [], []
        for i in range(len(idcs)):
            img_sin = np.load(self.img_file[i])
            mask_sin = np.load(self.mask_file[i])
            gt_sin = np.load(self.gt_file[i])
            #
            imgs.append(img_sin)
            masks.append(mask_sin)
            gts.append(gt_sin)
        # torch.tensor
        # self.imgs = torch.from_numpy(np.array(imgs)).float().cuda()
        # self.masks = torch.from_numpy(np.array(masks)).float().cuda()
        # self.gts = torch.from_numpy(np.array(gts)).float().cuda()
        self.imgs = np.array(imgs)
        self.masks = np.array(masks)
        self.gts = np.array(gts)
        self.is_random_img = False  # 随机患者or指定患者
        # 先创建crop对象作为self变量
        self.crop = crop

    def __getitem__(self, idx):  # getitem方法支持从0到len(self)的索引，方便于按照索引加载数据
        """
        随机加载 or 根据指定索引加载（返回的crop样本）
        :param idx: 患者索引（在该患者的3Dimg基础上做patch的crop操作，作为进入网络的图片样本）
        :return: 
        """
        # 1个epoch样本采样情况（先imgs的长度*1个样本，把所有目标按target_box做crop，然后imgs的长度*1个样本，按照空目标做随机crop）
        if self.phase != 'test':
            if idx >= len(self.imgs):
                is_random = True
                idx = idx % len(self.imgs)
            else:
                is_random = False
        else:
            is_random = False
        # crop部分
        if self.phase != 'test':  # 训练/验证阶段
            if self.is_random_img:
                rand_id = np.random.randint(len(self.imgs))
            else:
                rand_id = idx
            img = self.imgs[rand_id]
            mask = self.masks[rand_id]
            gt = self.gts[rand_id] if not is_random else []
            # crop得到样本
            sample, sam_mask, sam_gt, sam_label = self.crop(img, mask, gt)
            # augment先不管
            return sample, sam_gt, sam_label, sam_mask

        else:  # 测试阶段（待完成）
            img = self.imgs[idx]
            mask = self.masks[idx]
            gt = self.gts[idx]
            # 测试阶段的的data_loader，不返回patch，返回完整的imgs
            imgs = 0
            return imgs

    def __len__(self):  # len方法提供了dataset的大小，可用于指定1个epoch中idx遍历的长度
        if self.phase == 'train':
            return 2 * len(self.imgs)
        elif self.phase == 'val':
            return len(self.imgs)
        else:
            return len(self.imgs)


# crop操作类
class Crop(object):
    def __init__(self, crop_size, bound_size, stride, pad_value):
        """

        :param crop_size: 裁剪尺寸
        :param bound_size:
        :param stride: 步长
        :param pad_value: padding 填充
        """
        self.crop_size = crop_size
        self.bound_size = bound_size
        self.stride = stride
        self.pad_value = pad_value
        super(Crop, self).__init__()

    def __call__(self, img, mask, gt):
        """
        注：1位患者只有1个target_box
        :param img: 3d 图像 [30, 512, 512] (z,y,x)
        :param mask: 3d mask [30, 512, 512] (z,y,x)
        :param gt: 3d cube [6,]->(y1,x1,z1,y2,x2,z2)
        :return: 
        patch_img: [batch, 1, 128, 128, 128] (batch,channel,z,y,x)
        patch_mask: [batch, 1, 128, 128, 128] (batch,channel,z,y,x)
        target_box: [batch, 6] (batch,y1,x1,z1,y2,x2,z2)
        patch_label: [batch, 1]
        """
        # gt中(y1,x1,z1,y2,x2,z2)表示转(y,x,z,diameter)
        crop_size = np.array(self.crop_size)
        bound_size = self.bound_size
        target_box = np.copy(gt)
        if target_box.any():
            target_box = np.array([np.mean([target_box[0], target_box[3]]), np.mean([target_box[1], target_box[4]]),
                                   np.mean([target_box[2], target_box[5]]),
                                   np.max([target_box[3] - target_box[0], target_box[4] - target_box[1],
                                           target_box[5] - target_box[2]])])
        else:
            target_box = np.array([np.nan, np.nan, np.nan, np.nan])
        # (y,x,z,diameter)顺序转(z,y,x,diameter)
        target_box = np.array([target_box[2], target_box[0], target_box[1], target_box[3]])
        # 根据target_box是否为空，寻找采样边界与随机采样点
        if not np.isnan(target_box).any():  # 以target为中心的采样边界
            radius = target_box[3] / 2
            start = np.floor(np.array([target_box[:3] - radius], dtype='float32'))[0] + 1 - bound_size
            end = np.ceil(np.array([target_box[:3] + radius], dtype='float32'))[0] + 1 + bound_size - crop_size
            # 转置处理之后，根据start与end大小关系，在x、y、z每个维度选择patch顶点的起始坐标值
            border = np.stack((start, end), axis=-1)
            # 根据target_box采样时，如果start <= end以target_box中心点为中心采样，否则在（end，start）区域内随机
            point = np.array([
                int(target_box[i]) - int(crop_size[i] / 2) + np.random.randint(int(-bound_size / 2),
                                                                               int(bound_size / 2))
                if border[i][0] <= border[i][1] else np.random.randint(min(border[i][0], border[i][1]),
                                                                       max(border[i][0], border[i][1])) for i in
                range(len(border))])
        else:  # 随机采样边界
            start = - np.array([bound_size, bound_size, bound_size])
            end = np.array(img.shape) + bound_size - crop_size
            # 转置处理之后，根据start与end大小关系，在x、y、z每个维度选择patch顶点的起始坐标值
            border = np.stack((start, end), axis=-1)
            # 随机采样时，没有target_box做参照，直接在（start，end）区域内随机
            point = np.array(
                [np.random.randint(min(border[i][0], border[i][1]), max(border[i][0], border[i][1])) for i in
                 range(len(border))])
        # 寻找padding区域
        left_pad = - point
        left_pad[left_pad < 0] = 0
        right_pad = np.array(point + crop_size - img.shape)
        right_pad[right_pad < 0] = 0
        padding = np.stack((left_pad, right_pad), axis=-1)
        # 新增batch维度的全零padding初始化
        padding = np.concatenate([np.array([[0, 0]]), padding], axis=0)
        img = img[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        # crop与padding得到patch_img与patch_mask
        patch_img = img[:,
                    max(point[0], 0):min(point[0] + crop_size[0], img.shape[1]),
                    max(point[1], 0):min(point[1] + crop_size[1], img.shape[2]),
                    max(point[2], 0):min(point[2] + crop_size[2], img.shape[3])]
        patch_mask = mask[:,
                     max(point[0], 0):min(point[0] + crop_size[0], mask.shape[1]),
                     max(point[1], 0):min(point[1] + crop_size[1], mask.shape[2]),
                     max(point[2], 0):min(point[2] + crop_size[2], mask.shape[3])]
        # 维度需要匹配
        patch_img = np.pad(patch_img, padding, 'constant', constant_values=self.pad_value)
        patch_mask = np.pad(patch_mask, padding, 'constant', constant_values=self.pad_value)
        # 相对坐标变换得到target_box
        target_box[:3] = target_box[:3] - point
        # (z,y,x,diameter)还原为(y1,x1,z1,y2,x2,z2)
        target_box = np.array(
            [target_box[1] - target_box[3] / 2, target_box[2] - target_box[3] / 2, target_box[0] - target_box[3] / 2,
             target_box[1] + target_box[3] / 2, target_box[2] + target_box[3] / 2,
             target_box[0] + target_box[3] / 2])
        patch_label = np.ones([1])

        return patch_img, patch_mask, target_box, patch_label


def main():
    """
    测试类
    :return: 
    """
    data_dir = 'F:\迅雷下载\dicom文件-标记文件\subset'
    crop = Crop(cfg.CROP_SIZE,
                cfg.BOUND_SIZE,
                cfg.STRIDE,
                cfg.PAD_VALUE)
    # 数据包装
    dataset = Data3Lung(
        data_dir,
        crop,
        phase='train')
    # 传入DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,  # 使用几个子进程
        # collate_fn=lambda x:x,  # 取样本的方式可自定义（稍微复杂难理解一点）
        pin_memory=False)  # 是否将tensors拷贝到CUDA中的固定内存
    # 3个epochs的输出测试
    for epoch in range(3):
        for i, data in enumerate(train_loader):
            # 将数据从train_loader中读出来，一次读取的样本数是batch_size=4个
            inputs, masks, gts, labels = data
            # 将这些数据转换成Variable类型
            inputs, masks, gts, labels = Variable(inputs), Variable(masks), Variable(gts), Variable(labels)
            # 接下来就是训练环节，这里使用print来代替
            print("epoch：", epoch, "的第", i, "个inputs", inputs.data.size(), "masks", masks.data.size(), "gts",
                  gts.data.size(), "labels", labels.data.size())


if __name__ == '__main__':
    main()
