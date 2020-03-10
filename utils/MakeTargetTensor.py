import numpy as np
import cv2
import torch
import torch.nn.functional as F
import math
import scipy.signal


class LoadBBTensor():
    def __init__(self, tensor_size, channels):
        # gaussian sphere
        gaussian_sphere_base = np.zeros([100, 100, 100])
        ar_x = scipy.signal.gaussian(100, std=20).astype(np.float32)
        for k in range(0, 100):
            for j in range(0, 100):
                for i in range(0, 100):
                    dist = int(math.sqrt((50 - k) ** 2 + (50 - j) ** 2 + (50 - i) ** 2))
                    if dist > 50:
                        continue
                    elif dist == 0:
                        gaussian_sphere_base[i, j, k] = 1
                    else:
                        dist -= 1
                        gaussian_sphere_base[i, j, k] = ar_x[50:100][dist]
        self.GaussianSphere = torch.from_numpy(gaussian_sphere_base).detach()
        # self.GaussianSphere = gaussian_sphere_base
        self.tensor_size = tensor_size
        self.channels = channels

    def __call__(self, target_path):
        bb_tensor = self._load_bb_tensor(target_path)
        return bb_tensor

    def _load_bb_tensor(self, target_path):
        f = open(target_path)
        lines = f.readlines()
        f.close()
        target_tensor_np = np.zeros([self.channels,
                                     self.tensor_size,
                                     self.tensor_size,
                                     self.tensor_size]).astype(np.float32)  # img_num, channels, z, y, x
        for line in lines:
            # print('line:', line)
            words = line.split(' ')
            if words[0] == 'person':
                i = 0
                class_channel_num = 7
            else:  # not person
                if words[1].isalpha():  # all CATEGORIES are not same verbs
                    i = 1
                else:
                    i = 0
                class_channel_num = 8
            x = float(words[i + 1])  # point cloud center is not (0, 0, 0)
            y = float(words[i + 2])
            z = float(words[i + 3])
            xx = float(words[i + 4])
            yy = float(words[i + 5])
            zz = float(words[i + 6])

            anchor_x = ((((x + xx) / 2) + 10) / 20) * self.tensor_size  # -10~10 -> 0~20 -> 0~1 -> 0~26
            if anchor_x >= self.tensor_size:
                anchor_x = self.tensor_size - 0.01
            elif anchor_x <= 0:
                anchor_x = 0
            anchor_x_grid = math.floor(anchor_x)  # width max -> 20m
            anchor_x_value = anchor_x - anchor_x_grid
            width = (xx - x) / 20
            x_grid_start = math.floor(anchor_x - ((width * self.tensor_size) / 2))
            x_grid_stop = math.floor(anchor_x + ((width * self.tensor_size) / 2))
            if x_grid_stop >= self.tensor_size:
                x_grid_stop = self.tensor_size - 0.01
            elif x_grid_start <= 0:
                x_grid_start = 0

            anchor_y = ((((y + yy) / 2) + 2.5) / 5) * self.tensor_size  # -2.5~2.5 -> 0~5 -> 0~1 -> 0~26
            if anchor_y >= self.tensor_size:
                anchor_y = self.tensor_size - 0.01
            elif anchor_y <= 0:
                anchor_y = 0
            anchor_y_grid = math.floor(anchor_y)  # height max -> 5m
            anchor_y_value = anchor_y - anchor_y_grid
            height = (yy - y) / 5
            y_grid_start = math.floor(anchor_y - ((height * self.tensor_size) / 2))
            y_grid_stop = math.floor(anchor_y + ((height * self.tensor_size) / 2))
            if y_grid_stop >= self.tensor_size:
                y_grid_stop = self.tensor_size - 0.01
            elif y_grid_start <= 0:
                y_grid_start = 0

            anchor_z = (((z + zz) / 2) / 10) * self.tensor_size  # 0~10 -> 0~1 -> 0~26
            if anchor_z >= self.tensor_size:
                anchor_z = self.tensor_size - 0.01
            elif anchor_z <= 0:
                anchor_z = 0
            anchor_z_grid = math.floor(anchor_z)  # depth max -> 10m
            anchor_z_value = anchor_z - anchor_z_grid
            depth = (zz - z) / 10
            z_grid_start = math.floor(anchor_z - ((depth * self.tensor_size) / 2))
            z_grid_stop = math.floor(anchor_z + ((depth * self.tensor_size) / 2))
            if z_grid_stop >= self.tensor_size:
                z_grid_stop = self.tensor_size - 0.01
            elif z_grid_start <= 0:
                z_grid_start = 0
            # print(words[0])
            # print(x/20 * 26, y/5 * 26, z/10 * 26)
            # print(xx/20 * 26, yy/5 * 26, zz/10 * 26)
            # print(anchor_x_grid, anchor_y_grid, anchor_z_grid)

            # confident
            bb_conf \
                = self._make_bb_conf_array_torch(x_grid_stop - x_grid_start + 1,
                                                 y_grid_stop - y_grid_start + 1,
                                                 z_grid_stop - z_grid_start + 1)
            conf = target_tensor_np[0, :, :, :]
            target_tensor_np[0, z_grid_start:z_grid_stop + 1, y_grid_start:y_grid_stop + 1, x_grid_start:x_grid_stop + 1] \
                = np.where(bb_conf > conf[z_grid_start:z_grid_stop + 1, y_grid_start:y_grid_stop + 1, x_grid_start:x_grid_stop + 1],
                           bb_conf, conf[z_grid_start:z_grid_stop + 1, y_grid_start:y_grid_stop + 1, x_grid_start:x_grid_stop + 1])
            target_tensor_np[0, anchor_z_grid, anchor_y_grid, anchor_x_grid] = 1
            # class
            target_tensor_np[class_channel_num, anchor_z_grid, anchor_y_grid, anchor_x_grid] = 1
            # anchor
            target_tensor_np[1, anchor_z_grid, anchor_y_grid, anchor_x_grid] = anchor_x_value  # x
            target_tensor_np[2, anchor_z_grid, anchor_y_grid, anchor_x_grid] = anchor_y_value  # y
            target_tensor_np[3, anchor_z_grid, anchor_y_grid, anchor_x_grid] = anchor_z_value  # z
            target_tensor_np[4, anchor_z_grid, anchor_y_grid, anchor_x_grid] = width  # w
            target_tensor_np[5, anchor_z_grid, anchor_y_grid, anchor_x_grid] = height  # h
            target_tensor_np[6, anchor_z_grid, anchor_y_grid, anchor_x_grid] = depth  # d
            # print(anchor_x_value, anchor_y_value, anchor_z_value, width, height, depth)

        return target_tensor_np

    def _make_bb_conf_array(self, bb_width, bb_height, bb_depth):
        gaussian_sphere_c = self.GaussianSphere.copy()
        target = np.zeros((bb_depth, bb_height, bb_width))
        target_con = np.zeros((bb_depth, 100, bb_width))
        # resize 3d array
        # stable y
        for i in range(gaussian_sphere_c.shape[1]):
            img = gaussian_sphere_c[:, i, :]
            img = cv2.resize(img, (bb_width, bb_depth))  # cv2.resize -> size(x, y) = transpose x and y.
            target_con[:, i, :] = img
        # stable z
        for c in range(target_con.shape[0]):
            img = target_con[c, :, :]
            img = cv2.resize(img, (bb_width, bb_height))
            target[c, :, :] = img

        return target

    def _make_bb_conf_array_torch(self, bb_width, bb_height, bb_depth):
        gaussian_sphere_t = self.GaussianSphere.clone()
        gaussian_sphere_t = F.adaptive_avg_pool3d(gaussian_sphere_t[None, :, :, :], (bb_depth, bb_height, bb_width))
        return gaussian_sphere_t[0].numpy()


class Numpy2Tensor():
    def __call__(self, rgbd):
        rgbd = torch.from_numpy(rgbd)
        return rgbd
