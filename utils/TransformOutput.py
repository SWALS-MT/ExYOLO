# pytorch
import torch

import os
import numpy as np
import cv2

class YOLOOutput2BB():
    """
    grid_scale: The grid size of the model output.
    x_scale: The x scale of input.
    y_scale: The y scale of input.
    z_scale: The z scale of input.
    """
    def __init__(self, grid_scale, x_scale, y_scale, z_scale, conf_thresh, device):
        self.grid_scale = grid_scale
        self.x_scale = x_scale
        self.st_center_x = x_scale / 2
        self.y_scale = y_scale
        self.st_center_y = y_scale / 2
        self.z_scale = z_scale

        self.conf_thresh = conf_thresh
        self.device = device

    def __call__(self, model_output):
        conf_tensor = (model_output[:, 0] > self.conf_thresh).nonzero()
        onehot = torch.zeros((model_output.size()[0], self.grid_scale, self.grid_scale, self.grid_scale),
                             dtype=torch.float32).zero_().to(self.device)
        onehot[conf_tensor[:, 0], conf_tensor[:, 1], conf_tensor[:, 2], conf_tensor[:, 3]] = 1

        # xyz to grid scale
        model_output[conf_tensor[:, 0], 1, conf_tensor[:, 1], conf_tensor[:, 2], conf_tensor[:, 3]] += conf_tensor[:, 3]
        model_output[conf_tensor[:, 0], 2, conf_tensor[:, 1], conf_tensor[:, 2], conf_tensor[:, 3]] += conf_tensor[:, 2]
        model_output[conf_tensor[:, 0], 3, conf_tensor[:, 1], conf_tensor[:, 2], conf_tensor[:, 3]] += conf_tensor[:, 1]
        # xyz whd
        x = (model_output[:, 1]) * onehot
        x = ((x[x > 0] / self.grid_scale) * self.x_scale) - self.st_center_x
        y = model_output[:, 2] * onehot
        y = ((y[y > 0] / self.grid_scale) * self.y_scale) - self.st_center_y
        z = model_output[:, 3] * onehot
        z = ((z[z > 0] / self.grid_scale) * self.z_scale)
        w = model_output[:, 4] * onehot
        w = w[w > 0] * self.x_scale
        h = model_output[:, 5] * onehot
        h = h[h > 0] * self.y_scale
        d = model_output[:, 6] * onehot
        d = d[d > 0] * self.z_scale

        # xxyyzz
        x1 = (x - (w/2)).unsqueeze(0)
        x2 = (x + (w/2)).unsqueeze(0)
        y1 = (y - (h/2)).unsqueeze(0)
        y2 = (y + (h/2)).unsqueeze(0)
        z1 = (z - (d/2)).unsqueeze(0)
        z2 = (z + (d/2)).unsqueeze(0)
        # print('x1', x1)
        # print('x2', x2)
        # print('y1', y1)
        # print('y2', y2)
        # print('z1', z1)
        # print('z2', z2)
        bb = torch.cat((x1, x2, y1, y2, z1, z2), dim=0)
        bb = torch.transpose(bb, dim0=0, dim1=1)
        # print(bb)

        return bb


class SaveData():
    def __init__(self, color_save_flag, depth_save_flag, txt_save_flag, save_dir):
        """
        Currently, this function is needed to only 1 frame as input.
        :param color_save_flag: Flag of saving color image
        :param depth_save_flag: Flag of saving depth image
        :param txt_save_flag: Flag of saving the txt of bounding boxes.
        :param save_dir:
        """
        os.makedirs(save_dir + '/color', exist_ok=True)
        os.makedirs(save_dir + '/depth', exist_ok=True)
        os.makedirs(save_dir + '/bb', exist_ok=True)

        self.color_flag = color_save_flag
        self.depth_flag = depth_save_flag
        self.txt_flag = txt_save_flag
        self.save_dir = save_dir

    def __call__(self, save_name, color=None, depth=None, bb=None):
        if self.color_flag is True and color is None:
            raise Exception('SaveData Error: color_save_flag is True, but there is no color input.')
        elif self.depth_flag is True and depth is None:
            raise Exception('SaveData Error: depth_save_flag is True, but there is no depth input.')
        elif self.txt_flag is True and bb is None:
            raise Exception('SaveData Error: txt_save_flag is True, but there is no bb input.')
        else:
            if self.color_flag is True:
                self.color_save(color=color, name=save_name)
            if self.depth_flag is True:
                self.depth_save(depth=depth, name=save_name)
            if self.txt_flag is True:
                self.bb_save(bb=bb, name=save_name)

    def color_save(self, color, name):
        cv2.imwrite(filename=self.save_dir + '/color/' + name + '.png', img=color)

    def depth_save(self, depth, name):
        cv2.imwrite(filename=self.save_dir + '/depth/' + name + '.png', img=depth)

    def bb_save(self, bb, name):
        bb = bb.to("cpu").numpy()
        np.savetxt(self.save_dir + '/bb/' + name + '.txt', bb)
