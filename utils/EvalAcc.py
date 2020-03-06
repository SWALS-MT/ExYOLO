import torch


def calculate_accuracy(output_tensor, target_tensor):
    ious = []
    for out, target in zip(output_tensor, target_tensor):
        # print('#', out.size())

        maxvalue_t = torch.max(target[0, :, :, :])
        maxindex_t = (target[0] == maxvalue_t).nonzero()[0]
        # out
        out_w = out[4, maxindex_t[0], maxindex_t[1], maxindex_t[2]]
        out_h = out[5, maxindex_t[0], maxindex_t[1], maxindex_t[2]]
        out_d = out[6, maxindex_t[0], maxindex_t[1], maxindex_t[2]]
        out_x = (out[1, maxindex_t[0], maxindex_t[1], maxindex_t[2]] + maxindex_t[2]) - out.size()[3] * out_w/2
        out_y = (out[2, maxindex_t[0], maxindex_t[1], maxindex_t[2]] + maxindex_t[1]) - out.size()[2] * out_h/2
        out_z = (out[3, maxindex_t[0], maxindex_t[1], maxindex_t[2]] + maxindex_t[0]) - out.size()[1] * out_d/2
        out_xx = (out[1, maxindex_t[0], maxindex_t[1], maxindex_t[2]] + maxindex_t[2]) + out.size()[3] * out_w/2
        out_yy = (out[2, maxindex_t[0], maxindex_t[1], maxindex_t[2]] + maxindex_t[1]) + out.size()[2] * out_h/2
        out_zz = (out[3, maxindex_t[0], maxindex_t[1], maxindex_t[2]] + maxindex_t[0]) + out.size()[1] * out_d/2
        # target
        tar_w = target[4, maxindex_t[0], maxindex_t[1], maxindex_t[2]]
        tar_h = target[5, maxindex_t[0], maxindex_t[1], maxindex_t[2]]
        tar_d = target[6, maxindex_t[0], maxindex_t[1], maxindex_t[2]]
        tar_x = (target[1, maxindex_t[0], maxindex_t[1], maxindex_t[2]] + maxindex_t[2]) - out.size()[3] * tar_w/2
        tar_y = (target[2, maxindex_t[0], maxindex_t[1], maxindex_t[2]] + maxindex_t[1]) - out.size()[2] * tar_h/2
        tar_z = (target[3, maxindex_t[0], maxindex_t[1], maxindex_t[2]] + maxindex_t[0]) - out.size()[1] * tar_d/2
        tar_xx = (target[1, maxindex_t[0], maxindex_t[1], maxindex_t[2]] + maxindex_t[2]) + out.size()[3] * tar_w/2
        tar_yy = (target[2, maxindex_t[0], maxindex_t[1], maxindex_t[2]] + maxindex_t[1]) + out.size()[2] * tar_h/2
        tar_zz = (target[3, maxindex_t[0], maxindex_t[1], maxindex_t[2]] + maxindex_t[0]) + out.size()[1] * tar_d/2

        # x
        if out_x <= tar_x <= out_xx <= tar_xx:
            width_bb = out_xx - tar_x
        elif tar_x <= out_x <= tar_xx <= out_xx:
            width_bb = tar_xx - out_x
        elif out_x <= tar_x <= tar_xx <= out_xx:
            width_bb = tar_xx - tar_x
        elif tar_x <= out_x <= out_xx <= tar_xx:
            width_bb = out_xx - out_x
        else:
            width_bb = 0
            iou = 0
            ious.append(iou)
            continue

        # y
        if out_y <= tar_y <= out_yy <= tar_yy:
            height_bb = out_yy - tar_y
        elif tar_y <= out_y <= tar_yy <= out_yy:
            height_bb = tar_yy - out_y
        elif out_y <= tar_y <= tar_yy <= out_yy:
            height_bb = tar_yy - tar_y
        elif tar_y <= out_y <= out_yy <= tar_yy:
            height_bb = out_yy - out_y
        else:
            height_bb = 0
            iou = 0
            ious.append(iou)
            continue

        # z
        if out_z <= tar_z <= out_zz <= tar_zz:
            depth_bb = out_zz - tar_z
        elif tar_z <= out_z <= tar_zz <= out_zz:
            depth_bb = tar_zz - out_z
        elif out_z <= tar_z <= tar_zz <= out_zz:
            depth_bb = tar_zz - tar_z
        elif tar_z <= out_z <= out_zz <= tar_zz:
            depth_bb = out_zz - out_z
        else:
            depth_bb = 0
            iou = 0
            ious.append(iou)
            continue

        # iou
        iou_v = width_bb * height_bb * depth_bb
        out_v = ((out.size()[3])**3) * out_w * out_h * out_d
        tar_v = ((target.size()[3])**3) * tar_w * tar_h * tar_d
        iou = iou_v / (out_v + tar_v - iou_v)
        iou = iou.to("cpu").item()
        ious.append(iou)
    iou_mean = sum(ious) / len(ious)

    return iou_mean


if __name__=='__main__':
    x = torch.randn((3, 5, 10, 10, 10))
    calculate_accuracy(x, x)