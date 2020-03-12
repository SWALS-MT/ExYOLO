# base
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from visdom import Visdom
from tqdm import tqdm

# torch
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
epochs = 135
batch_size = 10
learning_rate = 0.01
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
train_data_dir = '/mnt/HDD1/mtakahashi/dataset/new_mydata/2019-10-14-18-56-35'
val_data_dir = '/mnt/HDD1/mtakahashi/dataset/new_mydata/2019-10-14-18-56-35/val'
# transform
transforms = T.Compose([M.Numpy2Tensor()])
target_transforms = T.Compose([M.Numpy2Tensor()])
# image processing
RGBD2DivRGBD = IP.RGBD2DivRGBD(img_size=img_size, div_num=output_size, depth_max=depth_max, device=device)

# 学習を開始した日付
dt_now = datetime.datetime.now()
dt_str = dt_now.strftime('%Y-%m-%d')

# Dataset object
dataset_full = Loader.ExYOLOMakeDatasetObject(train_data_dir + '/color',
                                              train_data_dir + '/depth',
                                              train_data_dir + '/AABB_3D',
                                              img_size=img_size,
                                              output_size=output_size,
                                              output_channels=output_channels)


def train(dataset_train):
    # __Initialize__ #
    model.train()
    running_loss = 0.0
    iou_scores = 0.0
    data_count = 0

    dataloader_train = data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    for step, (rgbds, targets) in enumerate(tqdm(dataloader_train, leave=False), 1):
        optimizer.zero_grad()

        rgbds = rgbds.to(device)
        targets = targets.to(device)
        rgbds = RGBD2DivRGBD(rgbds)

        outputs, loss = model(rgbds, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        targets = targets.detach()
        outputs = outputs.detach()

        iou_scores += ac.calculate_accuracy(outputs, targets)
        data_count = step

    train_loss = running_loss / data_count
    train_acc = iou_scores / data_count

    return train_loss, train_acc


def eval(dataset_val):
    # __Initialize__ #
    model.eval()
    running_loss = 0.0
    iou_scores = 0.0
    data_count = 0
    with torch.no_grad():
        dataloader_val = data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)
        for step, (rgbds, targets) in enumerate(tqdm(dataloader_val, leave=False), 1):
            rgbds = rgbds.to(device)
            targets = targets.to(device)
            rgbds = RGBD2DivRGBD(rgbds)
            outputs, loss = model(rgbds, targets)
            running_loss += loss.item()
            targets = targets.detach()
            outputs = outputs.detach()

            iou_scores += ac.calculate_accuracy(outputs, targets)
            data_count = step

        val_loss = running_loss / data_count
        val_acc = iou_scores / data_count

        return val_loss, val_acc


if __name__ == '__main__':
    os.makedirs('./outputs/'+dt_str, exist_ok=True)
    train_dataset_length = int(len(dataset_full) * 0.8)
    val_dataset_length = int(len(dataset_full)) - train_dataset_length
    viz = Visdom()
    for epoch in range(epochs):
        train_dataset, val_dataset \
            = torch.utils.data.dataset.random_split(dataset_full, [train_dataset_length, val_dataset_length])
        train_loss, train_acc = train(train_dataset)
        print('epoch %d, train_loss: %.4f train_acc:%.4f' % (epoch + 1, train_loss, train_acc))
        print('loss_c:', model.yolo_loss.loss_c.item())
        print('loss_x:', model.yolo_loss.loss_x.item())
        print('loss_y:', model.yolo_loss.loss_y.item())
        print('loss_z:', model.yolo_loss.loss_z.item())
        print('loss_w:', model.yolo_loss.loss_w.item())
        print('loss_h:', model.yolo_loss.loss_h.item())
        print('loss_d:', model.yolo_loss.loss_d.item())

        val_loss, val_acc = eval(val_dataset)
        print('epoch %d, val_loss: %.4f val_acc:%.4f' % (epoch + 1, val_loss, val_acc))
        print('loss_c:', model.yolo_loss.loss_c.item())
        print('loss_x:', model.yolo_loss.loss_x.item())
        print('loss_y:', model.yolo_loss.loss_y.item())
        print('loss_z:', model.yolo_loss.loss_z.item())
        print('loss_w:', model.yolo_loss.loss_w.item())
        print('loss_h:', model.yolo_loss.loss_h.item())
        print('loss_d:', model.yolo_loss.loss_d.item())

        # display

        viz.line(X=np.array([epoch]), Y=np.array([train_loss]), win='loss', name='avg_train_loss', update='append')
        viz.line(X=np.array([epoch]), Y=np.array([train_acc]), win='acc', name='avg_train_acc', update='append')
        viz.line(X=np.array([epoch]), Y=np.array([val_loss]), win='loss', name='avg_val_loss', update='append')
        viz.line(X=np.array([epoch]), Y=np.array([val_acc]), win='acc', name='avg_val_acc', update='append')
        # modelとグラフの保存
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        if epoch % 10 == 9:
            # modelとグラフの保存
            torch.save(model.state_dict(), './outputs/'+dt_str+'/DivExYOLO_'+dt_str+'_Epoch'+str(epoch+1)+'.pth')
            np.savez('./outputs/'+dt_str+'/train_loss_acc_backup_'+dt_str+'.npz', loss=np.array(train_loss_list),
                     acc=np.array(train_acc_list))
            np.savez('./outputs/'+dt_str+'/val_loss_acc_backup_'+dt_str+'.npz', loss=np.array(val_loss_list),
                     acc=np.array(val_acc_list))
        if 75 <= epoch < 105:
            learning_rate = 0.001
            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        elif 105 <= epoch:
            learning_rate = 0.0001
            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # 最終結果の保存
    torch.save(model.state_dict(), './data/' + dt_str + '/DivExYOLO_' + dt_str + '_Epoch135' + '.pth')
    np.savez('./outputs/' + dt_str + '/train_loss_acc_backup_' + dt_str + '.npz', loss=np.array(train_loss_list),
             acc=np.array(train_acc_list))
    np.savez('./outputs/' + dt_str + '/val_loss_acc_backup_' + dt_str + '.npz', loss=np.array(val_loss_list),
             acc=np.array(val_acc_list))