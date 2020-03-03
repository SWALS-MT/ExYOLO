import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import json
import glob

import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as T

from dataloader import Loader
from models import DivExYOLO
from utils import EvalAcc as ac
from utils import MakeTargetTensor as M
from utils import ImageProcessing as IP

import time
import datetime

# __VERSION__
# Python
print('Python: ', sys.version)

# PyTorch
print('PyTorch: ', torch.__version__)
# print('torchvision: ', torchvision.__version__)

# __Initialize__
# Hyper-parameters
epochs = 5
batch_size = 10
learning_rate = 0.001
# image settings
img_size = 224
output_size = 14
output_channels = 9
depth_max = 10000
# modelの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DivExYOLO.DivExYOLOVGG16()
model = model.to(device)
print(model)
print('model is_cuda:', next(model.parameters()).is_cuda)
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 学習率のスケジューラー
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
# グラフ作成用配列
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
# Datasetディレクトリ
train_data_dir = '/mnt/GPUServerWrokspace/mtakahashi/dataset/new_mydata/2019-10-14-18-56-35'
val_data_dir = '/mnt/GPUServerWrokspace/mtakahashi/dataset/new_mydata/2019-10-14-18-56-35/val'
# transform
transforms = T.Compose([M.Numpy2Tensor()])
target_transforms = T.Compose([M.Numpy2Tensor()])
# image processing
RGBD2DivRGBD = IP.RGBD2DivRGBD(img_size=img_size, div_num=output_size, depth_max=depth_max, device=device)

# 学習を開始した日付
dt_now = datetime.datetime.now()
dt_str = dt_now.strftime('%Y-%m-%d')


def train():
    ##__Initialize__##
    model.train()
    runnning_loss = 0.0
    iou_scores = 0
    data_count = 0

    dataset_train = Loader.ExYOLOMakeDatasetObject(train_data_dir+'/color',
                                                   train_data_dir+'/depth',
                                                   train_data_dir+'/AABB_3D',
                                                   img_size=img_size,
                                                   output_size=output_size,
                                                   output_channels=output_channels)
    dataloader_train = data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)

    for step, (rgbds, targets) in enumerate(dataloader_train, 1):
        optimizer.zero_grad()

        rgbds = rgbds.to(device)
        targets = targets.to(device)
        rgbds = RGBD2DivRGBD(rgbds)
        print(rgbds.size())
        outputs, loss = model(rgbds, targets)
        loss.backward()
        optimizer.step()
        runnning_loss += loss.item()
        targets = targets.detach()
        outputs = outputs.detach()

        iou_scores += ac.calculate_accuracy(outputs, targets)
        data_count = step
        if step % 100 == 0:
            print('Step: ' + str(step + 1), 'Loss: ' + str(runnning_loss / float(step + 1)))

    train_loss = runnning_loss / data_count
    train_acc = iou_scores / data_count

    return train_loss, train_acc


if __name__ == '__main__':
    for epoch in range(epochs):
        train_loss, train_acc = train()
        print('epoch %d, train_loss: %.4f train_acc:%.4f' % (epoch + 1, train_loss, train_acc))