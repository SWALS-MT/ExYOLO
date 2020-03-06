# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# original
from models.Backbones import Darknet53, Upsample, DivVGG16
from utils.LossFunctions import YoloBBLoss


class HeadNetDivVGG16(nn.Module):
    def __init__(self, first_channels=42):
        super(HeadNetDivVGG16, self).__init__()

        # feature extraction
        self.Div_VGG16 = DivVGG16(first_channels=first_channels)

        self.conv3d2_2 = nn.Conv3d(9, 18, kernel_size=1, stride=1, padding=0)
        self.bn3d2_2 = nn.BatchNorm3d(18)
        self.conv3d2_3 = nn.Conv3d(18, 9, kernel_size=3, stride=1, padding=1)
        self.bn3d2_3 = nn.BatchNorm3d(9)
        self.conv3d2_4 = nn.Conv3d(9, 18, kernel_size=1, stride=1, padding=0)
        self.bn3d2_4 = nn.BatchNorm3d(18)
        self.conv3d2_5 = nn.Conv3d(18, 9, kernel_size=3, stride=1, padding=1)
        self.bn3d2_5 = nn.BatchNorm3d(9)
        self.conv3d2_6 = nn.Conv3d(9, 18, kernel_size=1, stride=1, padding=0)
        self.bn3d2_6 = nn.BatchNorm3d(18)
        self.conv3d2_7 = nn.Conv3d(18, 18, kernel_size=3, stride=1, padding=1)
        self.bn3d2_7 = nn.BatchNorm3d(18)

        self.conv3d2_8 = nn.Conv3d(18, 9, kernel_size=1, stride=1, padding=0)
        self.bn3d2_8 = nn.BatchNorm3d(9)

    def forward(self, x):
        x = self.Div_VGG16(x)

        output = torch.reshape(x, (x.size(0), 9, 14, 14, 14))
        output = F.leaky_relu(self.bn3d2_2(self.conv3d2_2(output)))
        output = F.leaky_relu(self.bn3d2_3(self.conv3d2_3(output)))
        output = F.leaky_relu(self.bn3d2_4(self.conv3d2_4(output)))
        output = F.leaky_relu(self.bn3d2_5(self.conv3d2_5(output)))
        output = F.leaky_relu(self.bn3d2_6(self.conv3d2_6(output)))
        output = F.leaky_relu(self.bn3d2_7(self.conv3d2_7(output)))

        output = torch.sigmoid(self.bn3d2_8(self.conv3d2_8(output)))

        return output


class DivExYOLOVGG16(nn.Module):
    def __init__(self, first_channels=4):
        super(DivExYOLOVGG16, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.HeadNet = HeadNetDivVGG16()

        self.conf_thresh = 0.5
        self.obj_scale = 10
        self.noobj_scale = 1

        self.yolo_loss = YoloBBLoss(self.obj_scale, self.noobj_scale)

    def forward(self, x, targets):
        model_output = self.HeadNet(x)
        if targets is not None:
            loss = self.yolo_loss(model_output, targets)
            return model_output, loss
        else:
            return model_output
