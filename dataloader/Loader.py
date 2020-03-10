# standard
import sys
import math
import time
import pathlib
import numpy as np
import cv2
# pytorch
import torch
import torch.utils.data
# mine
from utils import MakeTargetTensor as M


class ExYOLOMakeDatasetObject(torch.utils.data.Dataset):
    def __init__(self,
                 color_dir,
                 depth_dir,
                 target_dir,
                 img_size,
                 output_size,
                 output_channels,
                 transform=None,
                 target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size

        p_color = pathlib.Path(color_dir)
        p_depth = pathlib.Path(depth_dir)
        p_target = pathlib.Path(target_dir)

        self.color_list = list(p_color.glob('./*.png'))
        self.depth_list = list(p_depth.glob('./*.png'))
        self.target_list = list(p_target.glob('./*.txt'))
        if len(self.color_list) == 0:
            print('ERROR: There is no file in the color directory.', file=sys.stderr)
            sys.exit(1)
        elif len(self.depth_list) == 0:
            print('ERROR: There is no file in the depth directory.', file=sys.stderr)
            sys.exit(2)
        elif len(self.target_list) == 0:
            print('ERROR: There is no file in the target directory.', file=sys.stderr)
            sys.exit(3)

        self.data_num = len(self.color_list)
        self.load_bbtensor = M.LoadBBTensor(tensor_size=output_size, channels=output_channels)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        color_dir_path = str(self.color_list[idx])
        depth_dir_path = str(self.depth_list[idx])
        target_dir_path = str(self.target_list[idx])

        rgbd = self._load_rgbd(color_dir_path, depth_dir_path, self.img_size)
        if self.transform is not None:
            rgbd = self.transform(rgbd)

        target = self.load_bbtensor(target_dir_path)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return rgbd, target

    @staticmethod
    def _load_rgbd(color_dir_path, depth_dir_path, img_size):
        color = cv2.imread(color_dir_path)
        color = cv2.resize(color, (img_size, img_size)).astype(np.float32)
        depth = cv2.imread(depth_dir_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (img_size, img_size), cv2.INTER_NEAREST)
        depth = depth[:, :, np.newaxis].astype(np.float32)
        rgbd = np.append(color, depth, axis=2)
        return rgbd
