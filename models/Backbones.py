# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    This is the residual block.
    One block of Residual Network.
    """
    expansion = 2

    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()

        bottleneck_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity mapping
        if in_channels != out_channels:  # downsampling
            self.shortcut.add_module('conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        return F.leaky_relu(out, inplace=True)


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class Darknet53(nn.Module):
    """
    Darknet53 represents a Darknet.
    This is used in YOLOv3 as a feature extractor.
    self.output_layers_num:
        This model can be selected the number of outputs
        These outputs are from the different layers).
    """

    def __init__(self, img_size=416, first_channels=4, output_layers_num=3):
        super(Darknet53, self).__init__()
        self.output_layers_num = output_layers_num

        # block1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conv1_1 = nn.Conv2d(first_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)

        # block2
        self.residual_block1_1 = ResidualBlock(64, 64, 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # block3
        self.residual_block3_1 = ResidualBlock(128, 128, 1)
        self.residual_block3_2 = ResidualBlock(128, 128, 1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # block4
        self.residual_block4_1 = ResidualBlock(256, 256, 1)
        self.residual_block4_2 = ResidualBlock(256, 256, 1)
        self.residual_block4_3 = ResidualBlock(256, 256, 1)
        self.residual_block4_4 = ResidualBlock(256, 256, 1)
        self.residual_block4_5 = ResidualBlock(256, 256, 1)
        self.residual_block4_6 = ResidualBlock(256, 256, 1)
        self.residual_block4_7 = ResidualBlock(256, 256, 1)
        self.residual_block4_8 = ResidualBlock(256, 256, 1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # block5
        self.residual_block5_1 = ResidualBlock(512, 512, 1)
        self.residual_block5_2 = ResidualBlock(512, 512, 1)
        self.residual_block5_3 = ResidualBlock(512, 512, 1)
        self.residual_block5_4 = ResidualBlock(512, 512, 1)
        self.residual_block5_5 = ResidualBlock(512, 512, 1)
        self.residual_block5_6 = ResidualBlock(512, 512, 1)
        self.residual_block5_7 = ResidualBlock(512, 512, 1)
        self.residual_block5_8 = ResidualBlock(512, 512, 1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        # block6
        self.residual_block6_1 = ResidualBlock(1024, 1024, 1)
        self.residual_block6_2 = ResidualBlock(1024, 1024, 1)
        self.residual_block6_3 = ResidualBlock(1024, 1024, 1)
        self.residual_block6_4 = ResidualBlock(1024, 1024, 1)
        self.conv6_1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.bn6_1 = nn.BatchNorm2d(512)
        self.conv6_2 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn6_2 = nn.BatchNorm2d(1024)
        self.conv6_3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.bn6_3 = nn.BatchNorm2d(512)
        self.conv6_4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn6_4 = nn.BatchNorm2d(1024)
        self.conv6_5 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.bn6_5 = nn.BatchNorm2d(512)
        self.conv6_6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn6_6 = nn.BatchNorm2d(1024)

    def forward(self, x):
        # block1
        output = F.leaky_relu(self.bn1_1(self.conv1_1(x)))
        output = F.leaky_relu(self.bn1_2(self.conv1_2(output)))

        # block2
        output = self.residual_block1_1(output)
        output = F.leaky_relu(self.bn2(self.conv2(output)))

        # block3
        output = self.residual_block3_1(output)
        output = self.residual_block3_2(output)
        output = F.leaky_relu(self.bn3(self.conv3(output)))

        # block4
        output = self.residual_block4_1(output)
        output = self.residual_block4_2(output)
        output = self.residual_block4_3(output)
        output = self.residual_block4_4(output)
        output = self.residual_block4_5(output)
        output = self.residual_block4_6(output)
        output = self.residual_block4_7(output)
        output = self.residual_block4_8(output)
        if self.output_layers_num == 3:
            out1 = output.clone()
        else:
            out1 = None
        output = F.leaky_relu(self.bn4(self.conv4(output)))

        # block5
        output = self.residual_block5_1(output)
        output = self.residual_block5_2(output)
        output = self.residual_block5_3(output)
        output = self.residual_block5_4(output)
        output = self.residual_block5_5(output)
        output = self.residual_block5_6(output)
        output = self.residual_block5_7(output)
        output = self.residual_block5_8(output)
        out2 = output.clone()
        output = F.leaky_relu(self.bn5(self.conv5(output)))

        # block6
        output = self.residual_block6_1(output)
        output = self.residual_block6_2(output)
        output = self.residual_block6_3(output)
        output = self.residual_block6_4(output)
        output = F.leaky_relu(self.bn6_1(self.conv6_1(output)))
        output = F.leaky_relu(self.bn6_2(self.conv6_2(output)))
        output = F.leaky_relu(self.bn6_3(self.conv6_3(output)))
        output = F.leaky_relu(self.bn6_4(self.conv6_4(output)))
        output = F.leaky_relu(self.bn6_5(self.conv6_5(output)))
        output = F.leaky_relu(self.bn6_6(self.conv6_6(output)))
        out3 = output

        if self.output_layers_num == 3:
            return out1, out2, out3
        elif self.output_layers_num == 2:
            return out2, out3
        else:
            return out3


class VGG16(nn.Module):
    def __init__(self, first_channels=4):
        super(VGG16, self).__init__()

        self.block1_conv1 = nn.Conv2d(first_channels, 64, 3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.block1_BN2d = nn.BatchNorm2d(64)
        self.block1_pool = nn.MaxPool2d(2, stride=2)

        self.block2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.block2_BN2d = nn.BatchNorm2d(128)
        self.block2_pool = nn.MaxPool2d(2, stride=2)

        self.block3_conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.block3_BN2d = nn.BatchNorm2d(256)
        self.block3_pool = nn.MaxPool2d(2, stride=2)

        self.block4_conv1 = nn.Conv2d(256, 512, 3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, 3, padding=1)
        self.block4_BN2d = nn.BatchNorm2d(512)
        self.block4_pool = nn.MaxPool2d(2, stride=2)

        self.block5_conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.block5_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.block5_conv3 = nn.Conv2d(1024, 1144, 3, padding=1)
        self.block5_BN2d = nn.BatchNorm2d(1144)

    def forward(self, x):
        # block1
        x = F.relu(self.block1_conv1(x))
        x = self.block1_pool(F.relu(self.block1_BN2d(self.block1_conv2(x))))

        # block2
        x = F.relu(self.block2_conv1(x))
        x = self.block2_pool(F.relu(self.block2_BN2d(self.block2_conv2(x))))

        # block3
        x = F.relu(self.block3_conv1(x))
        x = F.relu(self.block3_conv2(x))
        x = self.block3_pool(F.relu(self.block3_BN2d(self.block3_conv2(x))))

        # block4
        x = F.relu(self.block4_conv1(x))
        x = F.relu(self.block4_conv2(x))
        x = self.block4_pool(F.relu(self.block4_BN2d(self.block4_conv3(x))))

        # block5
        x = F.relu(self.block5_conv1(x))
        x = F.relu(self.block5_conv2(x))
        x = torch.sigmoid(self.block5_BN2d(self.block5_conv3(x)))

        return x


class DivVGG16(nn.Module):
    """
    This is DivVGG16, not normal VGG16.
    It can be set the number of first and last channels.
    first_channels:
        It is the number of the first channel, and the 10 layers depends on this value.
    output_channels:
        It is the number of the last channel, and the 3 layers depends on this value.
    """
    def __init__(self, first_channels=42, output_channels=126):
        super(DivVGG16, self).__init__()

        self.block1_conv1 = nn.Conv2d(first_channels, first_channels * 2, 3, padding=1)
        self.block1_BN2d1 = nn.BatchNorm2d(first_channels * 2)
        self.block1_conv2 = nn.Conv2d(first_channels * 2, first_channels * 2, 3, padding=1)
        self.block1_BN2d2 = nn.BatchNorm2d(first_channels * 2)
        self.block1_conv_st2 = nn.Conv2d(first_channels * 2, first_channels * 2, kernel_size=3, padding=1, stride=2)
        self.block1_BN2d3 = nn.BatchNorm2d(first_channels * 2)

        self.block2_conv1 = nn.Conv2d(first_channels * 2, first_channels * 4, 3, padding=1)
        self.block2_BN2d1 = nn.BatchNorm2d(first_channels * 4)
        self.block2_conv2 = nn.Conv2d(first_channels * 4, first_channels * 4, 3, padding=1)
        self.block2_BN2d2 = nn.BatchNorm2d(first_channels * 4)
        self.block2_conv_st2 = nn.Conv2d(first_channels * 4, first_channels * 4, kernel_size=3, padding=1, stride=2)
        self.block2_BN2d3 = nn.BatchNorm2d(first_channels * 4)

        self.block3_conv1 = nn.Conv2d(first_channels * 4, first_channels * 8, 3, padding=1)
        self.block3_BN2d1 = nn.BatchNorm2d(first_channels * 8)
        self.block3_conv2 = nn.Conv2d(first_channels * 8, first_channels * 8, 3, padding=1)
        self.block3_BN2d2 = nn.BatchNorm2d(first_channels * 8)
        self.block3_conv3 = nn.Conv2d(first_channels * 8, first_channels * 8, 3, padding=1)
        self.block3_BN2d3 = nn.BatchNorm2d(first_channels * 8)
        self.block3_conv_st2 = nn.Conv2d(first_channels * 8, first_channels * 8, kernel_size=3, padding=1, stride=2)
        self.block3_BN2d4 = nn.BatchNorm2d(first_channels * 8)

        self.block4_conv1 = nn.Conv2d(first_channels * 8, first_channels * 16, 3, padding=1)
        self.block4_BN2d1 = nn.BatchNorm2d(first_channels * 16)
        self.block4_conv2 = nn.Conv2d(first_channels * 16, first_channels * 16, 3, padding=1)
        self.block4_BN2d2 = nn.BatchNorm2d(first_channels * 16)
        self.block4_conv3 = nn.Conv2d(first_channels * 16, first_channels * 16, 3, padding=1)
        self.block4_BN2d3 = nn.BatchNorm2d(first_channels * 16)
        self.block4_conv_st2 = nn.Conv2d(first_channels * 16, first_channels * 16, kernel_size=3, padding=1, stride=2)
        self.block4_BN2d4 = nn.BatchNorm2d(first_channels * 16)

        self.block5_conv1 = nn.Conv2d(first_channels * 16, output_channels, 3, padding=1)
        self.block5_BN2d1 = nn.BatchNorm2d(output_channels)
        self.block5_conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.block5_BN2d2 = nn.BatchNorm2d(output_channels)
        self.block5_conv3 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.block5_BN2d3 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        # block1
        x = F.relu(self.block1_BN2d1(self.block1_conv1(x)))
        x = F.relu(self.block1_BN2d2(self.block1_conv2(x)))
        x = F.relu(self.block1_BN2d3(self.block1_conv_st2(x)))

        # block2
        x = F.relu(self.block2_BN2d1(self.block2_conv1(x)))
        x = F.relu(self.block2_BN2d2(self.block2_conv2(x)))
        x = F.relu(self.block2_BN2d3(self.block2_conv_st2(x)))

        # block3
        x = F.relu(self.block3_BN2d1(self.block3_conv1(x)))
        x = F.relu(self.block3_BN2d2(self.block3_conv2(x)))
        x = F.relu(self.block3_BN2d3(self.block3_conv3(x)))
        x = F.relu(self.block3_BN2d3(self.block3_conv_st2(x)))

        # block4
        x = F.relu(self.block4_BN2d1(self.block4_conv1(x)))
        x = F.relu(self.block4_BN2d2(self.block4_conv2(x)))
        x = F.relu(self.block4_BN2d3(self.block4_conv3(x)))
        x = F.relu(self.block4_BN2d3(self.block4_conv_st2(x)))

        # block5
        x = F.relu(self.block5_BN2d1(self.block5_conv1(x)))
        x = F.relu(self.block5_BN2d2(self.block5_conv2(x)))
        x = F.relu(self.block5_BN2d3(self.block5_conv3(x)))

        return x
