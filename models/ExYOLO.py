# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# original
from models.Backbones import Darknet53, Upsample, VGG16


class HeadNetDarknet(nn.Module):
    def __init__(self, first_channels=4, output_layers_num=2):
        super(HeadNetDarknet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # feature extraction
        self.Darknet = Darknet53(first_channels=first_channels, output_layers_num=output_layers_num)

        # transform architecture
        self.conv2_1 = nn.Conv2d(1024, 632, kernel_size=1, stride=1, padding=0)
        self.bn2_1 = nn.BatchNorm2d(632)
        self.upsample2 = Upsample(scale_factor=2, mode="nearest")
        self.conv2_x1 = nn.Conv2d(1144, 572, kernel_size=3, stride=1, padding=1)
        self.bn2_x1 = nn.BatchNorm2d(572)
        self.conv2_x2 = nn.Conv2d(572, 1144, kernel_size=1, stride=1, padding=0)
        self.bn2_x2 = nn.BatchNorm2d(1144)

        self.conv3d2_2 = nn.Conv3d(44, 88, kernel_size=1, stride=1, padding=0)
        self.bn3d2_2 = nn.BatchNorm3d(88)
        self.conv3d2_3 = nn.Conv3d(88, 44, kernel_size=3, stride=1, padding=1)
        self.bn3d2_3 = nn.BatchNorm3d(44)
        self.conv3d2_4 = nn.Conv3d(44, 88, kernel_size=1, stride=1, padding=0)
        self.bn3d2_4 = nn.BatchNorm3d(88)
        self.conv3d2_5 = nn.Conv3d(88, 44, kernel_size=3, stride=1, padding=1)
        self.bn3d2_5 = nn.BatchNorm3d(44)
        self.conv3d2_6 = nn.Conv3d(44, 88, kernel_size=1, stride=1, padding=0)
        self.bn3d2_6 = nn.BatchNorm3d(88)
        self.conv3d2_7 = nn.Conv3d(88, 88, kernel_size=3, stride=1, padding=1)
        self.bn3d2_7 = nn.BatchNorm3d(88)

        self.conv3d2_8 = nn.Conv3d(88, 9, kernel_size=1, stride=1, padding=0)
        self.bn3d2_8 = nn.BatchNorm3d(9)

    def forward(self, x):
        out2, out3 = self.Darknet(x)

        out3 = F.leaky_relu(self.bn2_1(self.conv2_1(out3)))
        out3 = self.upsample2(out3)
        output = torch.cat((out2, out3), dim=1)
        output = F.leaky_relu(self.bn2_x1(self.conv2_x1(output)))
        output = F.leaky_relu(self.bn2_x2(self.conv2_x2(output)))

        output = torch.reshape(output, (output.size(0), 44, 26, 26, 26))
        output = F.leaky_relu(self.bn3d2_2(self.conv3d2_2(output)))
        output = F.leaky_relu(self.bn3d2_3(self.conv3d2_3(output)))
        output = F.leaky_relu(self.bn3d2_4(self.conv3d2_4(output)))
        output = F.leaky_relu(self.bn3d2_5(self.conv3d2_5(output)))
        output = F.leaky_relu(self.bn3d2_6(self.conv3d2_6(output)))
        output = F.leaky_relu(self.bn3d2_7(self.conv3d2_7(output)))

        output = torch.sigmoid(self.bn3d2_8(self.conv3d2_8(output)))

        return output


class ExYOLODarknet53(nn.Module):
    def __init__(self, first_channels=4):
        super(ExYOLODarknet53, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.HeadNet = HeadNetDarknet(first_channels=first_channels)

        self.conf_thresh = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 10
        self.noobj_scale = 1

    def loss_calculation(self, model_output, targets):
        out_size = model_output.size()
        targets_conf = targets[:, 0, :, :, :]
        # output bb tensor の準備
        model_output_bb_c = model_output[:, 0, :, :, :]
        model_output_bb_x = model_output[:, 1, :, :, :]
        model_output_bb_y = model_output[:, 2, :, :, :]
        model_output_bb_z = model_output[:, 3, :, :, :]
        model_output_bb_w = model_output[:, 4, :, :, :]
        model_output_bb_h = model_output[:, 5, :, :, :]
        model_output_bb_d = model_output[:, 6, :, :, :]
        model_output_bb_c_obj = model_output_bb_c[targets_conf == 1]
        model_output_bb_c_noobj = model_output_bb_c[targets_conf < 1]
        model_output_bb_x = model_output_bb_x[targets_conf == 1]
        model_output_bb_y = model_output_bb_y[targets_conf == 1]
        model_output_bb_z = model_output_bb_z[targets_conf == 1]
        model_output_bb_w = model_output_bb_w[targets_conf == 1]
        model_output_bb_h = model_output_bb_h[targets_conf == 1]
        model_output_bb_d = model_output_bb_d[targets_conf == 1]
        # target bb tensor
        targets_x = targets[:, 1, :, :, :]
        targets_y = targets[:, 2, :, :, :]
        targets_z = targets[:, 3, :, :, :]
        targets_w = targets[:, 4, :, :, :]
        targets_h = targets[:, 5, :, :, :]
        targets_d = targets[:, 6, :, :, :]
        targets_c_obj = targets_conf[targets_conf == 1]
        targets_c_noobj = targets_conf[targets_conf < 1]
        targets_x = targets_x[targets_conf == 1]
        targets_y = targets_y[targets_conf == 1]
        targets_z = targets_z[targets_conf == 1]
        targets_w = targets_w[targets_conf == 1]
        targets_h = targets_h[targets_conf == 1]
        targets_d = targets_d[targets_conf == 1]

        # object loss
        loss_bb = self.bce_loss(model_output_bb_c_obj, targets_c_obj)   # confident
        loss_bb += self.mse_loss(model_output_bb_x, targets_x)          # Bounding Box
        loss_bb += self.mse_loss(model_output_bb_y, targets_y)
        loss_bb += self.mse_loss(model_output_bb_z, targets_z)
        loss_bb += self.mse_loss(model_output_bb_w, targets_w)
        loss_bb += self.mse_loss(model_output_bb_h, targets_h)
        loss_bb += self.mse_loss(model_output_bb_d, targets_d)
        for i in range(7, out_size[1]):
            model_output_class = model_output[:, i, :, :, :]
            model_output_class = model_output_class[targets_conf == 1]
            targets_class = targets[:, i, :, :, :]
            targets_class = targets_class[targets_conf == 1]
            loss_bb += self.bce_loss(model_output_class, targets_class)
        loss_bb = loss_bb * self.obj_scale
        # print('loss_bb', loss_bb.item())

        # no object loss
        loss_nobb = self.noobj_scale * self.bce_loss(model_output_bb_c_noobj, targets_c_noobj)
        # print('loss_nobb', loss_nobb.item())
        loss = loss_bb + loss_nobb

        return loss

    def forward(self, x, targets=None):
        model_output = self.HeadNet(x)
        if targets is not None:
            loss = self.loss_calculation(model_output, targets)
            return model_output, loss
        else:
            return model_output


class HeadNetVGG16(nn.Module):
    def __init__(self, first_channels=4):
        super(HeadNetVGG16, self).__init__()

        # feature extraction
        self.VGG16 = VGG16(first_channels=first_channels)

        self.conv3d2_2 = nn.Conv3d(44, 88, kernel_size=1, stride=1, padding=0)
        self.bn3d2_2 = nn.BatchNorm3d(88)
        self.conv3d2_3 = nn.Conv3d(88, 44, kernel_size=3, stride=1, padding=1)
        self.bn3d2_3 = nn.BatchNorm3d(44)
        self.conv3d2_4 = nn.Conv3d(44, 88, kernel_size=1, stride=1, padding=0)
        self.bn3d2_4 = nn.BatchNorm3d(88)
        self.conv3d2_5 = nn.Conv3d(88, 44, kernel_size=3, stride=1, padding=1)
        self.bn3d2_5 = nn.BatchNorm3d(44)
        self.conv3d2_6 = nn.Conv3d(44, 88, kernel_size=1, stride=1, padding=0)
        self.bn3d2_6 = nn.BatchNorm3d(88)
        self.conv3d2_7 = nn.Conv3d(88, 88, kernel_size=3, stride=1, padding=1)
        self.bn3d2_7 = nn.BatchNorm3d(88)

        self.conv3d2_8 = nn.Conv3d(88, 9, kernel_size=1, stride=1, padding=0)
        self.bn3d2_8 = nn.BatchNorm3d(9)

    def forward(self, x):
        x = self.VGG16(x)

        output = torch.reshape(x, (x.size(0), 44, 26, 26, 26))
        output = F.leaky_relu(self.bn3d2_2(self.conv3d2_2(output)))
        output = F.leaky_relu(self.bn3d2_3(self.conv3d2_3(output)))
        output = F.leaky_relu(self.bn3d2_4(self.conv3d2_4(output)))
        output = F.leaky_relu(self.bn3d2_5(self.conv3d2_5(output)))
        output = F.leaky_relu(self.bn3d2_6(self.conv3d2_6(output)))
        output = F.leaky_relu(self.bn3d2_7(self.conv3d2_7(output)))

        output = torch.sigmoid(self.bn3d2_8(self.conv3d2_8(output)))

        return output


class ExYOLOVGG16(nn.Module):
    def __init__(self, first_channels=4):
        super(ExYOLOVGG16, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.HeadNet = HeadNetVGG16()

        self.conf_thresh = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 10
        self.noobj_scale = 1

    def loss_calculation(self, model_output, targets):
        out_size = model_output.size()
        targets_conf = targets[:, 0, :, :, :]
        # output bb tensor の準備
        model_output_bb_c = model_output[:, 0, :, :, :]
        model_output_bb_x = model_output[:, 1, :, :, :]
        model_output_bb_y = model_output[:, 2, :, :, :]
        model_output_bb_z = model_output[:, 3, :, :, :]
        model_output_bb_w = model_output[:, 4, :, :, :]
        model_output_bb_h = model_output[:, 5, :, :, :]
        model_output_bb_d = model_output[:, 6, :, :, :]
        model_output_bb_c_obj = model_output_bb_c[targets_conf == 1]
        model_output_bb_c_noobj = model_output_bb_c[targets_conf < 1]
        model_output_bb_x = model_output_bb_x[targets_conf == 1]
        model_output_bb_y = model_output_bb_y[targets_conf == 1]
        model_output_bb_z = model_output_bb_z[targets_conf == 1]
        model_output_bb_w = model_output_bb_w[targets_conf == 1]
        model_output_bb_h = model_output_bb_h[targets_conf == 1]
        model_output_bb_d = model_output_bb_d[targets_conf == 1]
        # target bb tensor
        targets_x = targets[:, 1, :, :, :]
        targets_y = targets[:, 2, :, :, :]
        targets_z = targets[:, 3, :, :, :]
        targets_w = targets[:, 4, :, :, :]
        targets_h = targets[:, 5, :, :, :]
        targets_d = targets[:, 6, :, :, :]
        targets_c_obj = targets_conf[targets_conf == 1]
        targets_c_noobj = targets_conf[targets_conf < 1]
        targets_x = targets_x[targets_conf == 1]
        targets_y = targets_y[targets_conf == 1]
        targets_z = targets_z[targets_conf == 1]
        targets_w = targets_w[targets_conf == 1]
        targets_h = targets_h[targets_conf == 1]
        targets_d = targets_d[targets_conf == 1]

        # object loss
        loss_bb = self.bce_loss(model_output_bb_c_obj, targets_c_obj)   # confident
        loss_bb += self.mse_loss(model_output_bb_x, targets_x)          # Bounding Box
        loss_bb += self.mse_loss(model_output_bb_y, targets_y)
        loss_bb += self.mse_loss(model_output_bb_z, targets_z)
        loss_bb += self.mse_loss(model_output_bb_w, targets_w)
        loss_bb += self.mse_loss(model_output_bb_h, targets_h)
        loss_bb += self.mse_loss(model_output_bb_d, targets_d)
        for i in range(7, out_size[1]):
            model_output_class = model_output[:, i, :, :, :]
            model_output_class = model_output_class[targets_conf == 1]
            targets_class = targets[:, i, :, :, :]
            targets_class = targets_class[targets_conf == 1]
            loss_bb += self.bce_loss(model_output_class, targets_class)
        loss_bb = loss_bb * self.obj_scale

        # no object loss
        loss_nobb = self.noobj_scale * self.bce_loss(model_output_bb_c_noobj, targets_c_noobj)
        # print('loss_nobb', loss_nobb.item())
        loss = loss_bb + loss_nobb

        return loss

    def forward(self, x, targets):
        model_output = self.HeadNet(x)
        if targets is not None:
            loss = self.loss_calculation(model_output, targets)
            return model_output, loss
        else:
            return model_output


class DarknetPretrain(nn.Module):
    def __init__(self):
        super(DarknetPretrain, self).__init__()
        self.Darknet53 = Darknet53()
        # transform architecture
        # block6
        self.conv2_1 = nn.Conv2d(1024, 632, kernel_size=1, stride=1, padding=0)
        self.bn2_1 = nn.BatchNorm2d(632)
        # block5
        self.upsample2 = Upsample(scale_factor=2, mode="nearest")
        self.conv2_x1 = nn.Conv2d(1144, 572, kernel_size=3, stride=1, padding=1)
        self.bn2_x1 = nn.BatchNorm2d(572)
        self.conv2_x2 = nn.Conv2d(572, 1144, kernel_size=1, stride=1, padding=0)
        self.bn2_x2 = nn.BatchNorm2d(1144)
        # block4
        self.upsample3 = Upsample(scale_factor=2, mode="nearest")
        self.conv3_x1 = nn.Conv2d(1144, 572, kernel_size=3, stride=1, padding=1)
        self.bn3_x1 = nn.BatchNorm2d(572)
        self.conv3_x2 = nn.Conv2d(572, 1144, kernel_size=1, stride=1, padding=0)
        self.bn3_x2 = nn.BatchNorm2d(1144)
        # block3
        self.upsample4 = Upsample(scale_factor=2, mode="nearest")
        self.conv4_x1 = nn.Conv2d(1144, 572, kernel_size=3, stride=1, padding=1)
        self.bn4_x1 = nn.BatchNorm2d(572)
        self.conv4_x2 = nn.Conv2d(572, 1144, kernel_size=1, stride=1, padding=0)
        self.bn4_x2 = nn.BatchNorm2d(1144)
        # block2
        self.upsample5 = Upsample(scale_factor=2, mode="nearest")
        self.conv5_x1 = nn.Conv2d(1144, 572, kernel_size=3, stride=1, padding=1)
        self.bn5_x1 = nn.BatchNorm2d(572)
        self.conv5_x2 = nn.Conv2d(572, 1144, kernel_size=1, stride=1, padding=0)
        self.bn5_x2 = nn.BatchNorm2d(1144)
        # block1
        self.upsample6 = Upsample(scale_factor=2, mode="nearest")
        self.conv6_x1 = nn.Conv2d(1144, 572, kernel_size=3, stride=1, padding=1)
        self.bn6_x1 = nn.BatchNorm2d(572)
        self.conv6_x2 = nn.Conv2d(572, 894, kernel_size=1, stride=1, padding=0)
        self.bn6_x2 = nn.BatchNorm2d(894)

    def forward(self, x):
        out2, out3 = self.Darknet(x)
        # block6
        out3 = F.leaky_relu(self.bn2_1(self.conv2_1(out3)))
        # block5
        out3 = self.upsample2(out3)
        output = torch.cat((out2, out3), dim=1)
        output = F.leaky_relu(self.bn2_x1(self.conv2_x1(output)))
        output = F.leaky_relu(self.bn2_x2(self.conv2_x2(output)))
        # block4
        output = self.upsample3(output)
        output = F.leaky_relu(self.bn3_x1(self.conv3_x1(output)))
        output = F.leaky_relu(self.bn3_x2(self.conv3_x2(output)))
        # block3
        output = self.upsample4(output)
        output = F.leaky_relu(self.bn4_x1(self.conv4_x1(output)))
        output = F.leaky_relu(self.bn4_x2(self.conv4_x2(output)))
        # block2
        output = self.upsample5(output)
        output = F.leaky_relu(self.bn5_x1(self.conv5_x1(output)))
        output = F.leaky_relu(self.bn5_x2(self.conv5_x2(output)))
        # block1
        output = self.upsample6(output)
        output = F.leaky_relu(self.bn6_x1(self.conv6_x1(output)))
        output = F.leaky_relu(self.bn6_x2(self.conv6_x2(output)))

        return output