import torch
import torch.nn as nn
import numpy as np
from .skconv import SKUnit, SKConv


class BoxSE(nn.Module):
    def __init__(self, neck_opt=None, integrate_opt=None, detector_opt=None,
                groups=8, down_ratio=4, stride=1, min_dim=32):
        super(BoxSE, self).__init__()

        self.use_cls = detector_opt.use_cls
        self.detach_cls = detector_opt.detach_cls

        in_channels = integrate_opt.gcn_out_channels
        hidden_channels = detector_opt.detector_hidden_channels
        out_channels = detector_opt.detector_out_channels
        mid_channels = detector_opt.skc_mid_channels
        branches = detector_opt.skc_branches
        self.reg_neck = nn.ModuleList([])

        if isinstance(mid_channels, int):
            mid_channels = [mid_channels] * detector_opt.skc_layer_num

        for i in range(detector_opt.skc_layer_num):
            if i == 0:
                self.reg_neck.append(SKUnit(in_channels, hidden_channels, mid_channels[i], branches, groups, down_ratio, stride, min_dim))
            elif i == detector_opt.skc_layer_num-1:
                self.reg_neck.append(SKUnit(hidden_channels, out_channels, mid_channels[i], branches, groups, down_ratio, stride, min_dim))
            else:
                self.reg_neck.append(SKUnit(hidden_channels, hidden_channels, mid_channels[i], branches, groups, down_ratio, stride, min_dim))
        self.reg_head = nn.Conv2d(out_channels, 4, kernel_size=3, stride=1, padding=1)
        self.reg_scale = nn.Parameter(torch.tensor(1.))

        if self.use_cls:
            self.cls_neck = nn.ModuleList([])
            for i in range(detector_opt.skc_layer_num):
                if i == 0:
                    self.cls_neck.append(SKUnit(in_channels, hidden_channels, mid_channels[i], branches, groups, down_ratio, stride, min_dim))
                elif i == detector_opt.skc_layer_num-1:
                    self.cls_neck.append(SKUnit(hidden_channels, out_channels, mid_channels[i], branches, groups, down_ratio, stride, min_dim))
                else:
                    self.cls_neck.append(SKUnit(hidden_channels, hidden_channels, mid_channels[i], branches, groups, down_ratio, stride, min_dim))
            self.cls_head = nn.Conv2d(out_channels, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_reg = x
        for neck_layer in self.reg_neck:
            x_reg = neck_layer(x_reg)
        x_reg = self.reg_head(x_reg)
        offsets = torch.exp(x_reg*self.reg_scale)

        if self.use_cls:
            if self.detach_cls:
                x_cls = x.clone().detach()
            else:
                x_cls = x
            for neck_layer in self.cls_neck:
                x_cls = neck_layer(x_cls)
            output_cls = self.cls_head(x_cls)
            return offsets, output_cls
        else:
            return offsets, None

class Point(object):
    def __init__(self, stride, size):
        self.stride = stride
        self.size = size

        self.points = self.generate_points(self.stride, self.size)

    def generate_points(self, stride, size):
        """
        returns:
            points - (x, y)
        """
        if isinstance(size, int):
            size = [size, size]
        row_index = torch.arange(size[0]) * stride + int(stride/2)
        column_index = torch.arange(size[1]) * stride + int(stride/2)
        row_indices, column_indices = torch.meshgrid(row_index, column_index)
        points = torch.stack([column_indices, row_indices], dim=0).to(torch.float)

        return points



if __name__ == "__main__":
    point = Point(8, 18)
    print(point.points)
