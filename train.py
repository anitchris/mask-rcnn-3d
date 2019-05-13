# -*- coding: utf-8 -*-
"""
   File Name:     train.py
   Description:   训练
   Author:        steven.yi
   Date:          2019/5/10
"""
from utils.data_loader import Data3Lung, Crop
from torch.utils.data import DataLoader
from config import cur_config as cfg
from torch import optim
from model import LungNet


def main():
    # 加载数据
    crop = Crop(cfg.CROP_SIZE, cfg.BOUND_SIZE, cfg.STRIDE, cfg.PAD_VALUE)
    dataset = Data3Lung(cfg.DATA_DIR, crop, phase='train')
    train_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # 初始化网络
    net = LungNet(phase='train')
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 训练
    for epoch in range(cfg.EPOCHS):
        for i, data in enumerate(train_loader, 1):
            inputs, boxes, labels, masks = data
            # 清零梯度
            optimizer.zero_grad()
            # 正向传播 + 反向传播 + 参数更新
            outputs = net(inputs, boxes, labels, masks)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            # 每训练100个bathes后打印日志
            if i % 100 == 0:
                print('[INFO] Epoch: {} Batches: {} Loss: {}'.format(epoch + 1, i, loss / 100))

    print('[INFO] Finished Training')


if __name__ == '__main__':
    main()
