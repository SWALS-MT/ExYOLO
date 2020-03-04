import numpy as np
import cv2
import torch


class RGBD2DivRGBD():
    def __init__(self, img_size, div_num, depth_max, device):
        self.img_size = img_size
        self.div_num = div_num
        self.depth_max = depth_max
        self.device = device

    def __call__(self, rgbd):
        rgbd_size0 = rgbd.size()[0]
        ones = torch.ones((rgbd_size0, self.img_size, self.img_size, self.div_num, 3)).to(self.device)

        depth_t = rgbd[:, :, :, 3].to(self.device)
        depth_t = (depth_t * (self.div_num/self.depth_max))
        depth_t = torch.clamp(depth_t, min=0, max=(self.div_num - 1)).long()
        depth_t = depth_t.unsqueeze(3)

        onehot = torch.zeros(rgbd_size0, self.img_size, self.img_size, self.div_num, dtype=torch.float32, device=self.device)
        onehot = onehot.scatter_(-1, depth_t, 1.0)
        onehot = ones * onehot[:, :, :, :, None]

        color_t = rgbd[:, :, :, :3].to(self.device)
        color_t = color_t[:, :, :, None, :]
        color_t = onehot * color_t
        color_t = torch.flatten(color_t, start_dim=3, end_dim=4)
        color_t = torch.transpose(color_t, dim0=1, dim1=3)
        color_t = torch.transpose(color_t, dim0=2, dim1=3)
        color_t /= 255.0

        return color_t


if __name__ == '__main__':
    color_dir_path = '/mnt/UMENAS_mtakahashi/25期(M1)/高橋/Datasets/new_mydata/2019-10-14-18-56-35/color/00000.png'
    depth_dir_path = '/mnt/UMENAS_mtakahashi/25期(M1)/高橋/Datasets/new_mydata/2019-10-14-18-56-35/depth/00000.png'
    color = cv2.imread(color_dir_path)
    color = cv2.resize(color, (224, 224)).astype(np.float32)
    depth = cv2.imread(depth_dir_path, cv2.IMREAD_UNCHANGED)
    depth = cv2.resize(depth, (224, 224), cv2.INTER_NEAREST)
    depth = depth[:, :, np.newaxis].astype(np.float32)
    rgbd = torch.from_numpy(np.append(color, depth, axis=2))
    rgbd = rgbd[None, :, :, :]
    print('rgbd', rgbd.size())
    rgbd2divrgbd = RGBD2DivRGBD(224, 14, 10000, device=torch.device("cuda"))
    c = rgbd2divrgbd(rgbd)
