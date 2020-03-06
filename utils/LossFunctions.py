# pytorch
import torch
import torch.nn as nn


class YoloBBLoss():
    def __init__(self, obj_scale=10, nopbj_scale=1):
        self.obj_scale = obj_scale
        self.noobj_scale = nopbj_scale
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

    def __call__(self, model_output, targets):
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
        self.loss_c = self.bce_loss(model_output_bb_c_obj, targets_c_obj)   # confident
        self.loss_x = self.mse_loss(model_output_bb_x, targets_x)           # Bounding Box
        self.loss_y = self.mse_loss(model_output_bb_y, targets_y)
        self.loss_z = self.mse_loss(model_output_bb_z, targets_z)
        self.loss_w = self.mse_loss(model_output_bb_w, targets_w)
        self.loss_h = self.mse_loss(model_output_bb_h, targets_h)
        self.loss_d = self.mse_loss(model_output_bb_d, targets_d)
        self.loss_bb = self.loss_c + self.loss_x + self.loss_y + self.loss_z + self.loss_w + self.loss_h + self.loss_d
        for i in range(7, out_size[1]):
            model_output_class = model_output[:, i, :, :, :]
            model_output_class = model_output_class[targets_conf == 1]
            targets_class = targets[:, i, :, :, :]
            targets_class = targets_class[targets_conf == 1]
            self.loss_bb += self.bce_loss(model_output_class, targets_class)
        loss_bb = self.loss_bb * self.obj_scale

        # no object loss
        loss_nobb = self.noobj_scale * self.bce_loss(model_output_bb_c_noobj, targets_c_noobj)
        # print('loss_nobb', loss_nobb.item())
        loss = loss_bb + loss_nobb

        return loss
