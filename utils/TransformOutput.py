# pytorch
import torch
import numpy as np


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
        x1 = x - (w/2)
        x2 = x + (w/2)
        y1 = y - (h/2)
        y2 = y + (h/2)
        z1 = z - (d/2)
        z2 = z + (d/2)
        print('x1', x1)
        print('x2', x2)
        print('y1', y1)
        print('y2', y2)
        print('z1', z1)
        print('z2', z2)

        return model_output
