# naive version, share adjustlayer(conv3), fixed pooling size
import torch
import torch.nn as nn
import torch.nn.functional as f
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import prroi_pool2d
from mmdet.ops import RoIAlign, roi_align, roi_pool

class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=8, t_stride=8):
        super(AdjustLayer, self).__init__()
        self.pool_size = pool_size
        self.t_stride = t_stride
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.adjust = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.prroi_pool = PrRoIPool2D(pool_size, pool_size, 1/t_stride)


    def forward(self, x, tmp_flag=False, roi=None):
        """
        args:
            x - feature map of the template patch or search patch
            tmp_flag - the flag for template
            roi - shape: [num_roi, 5], coordinates: [index, x1, y1, x2, y2]
        """
        if isinstance(x, list):
            x = x[0]
        x = self.downsample(x)#shape=[num_sequences, batch, height, width]
        x = self.adjust(x)
        channel = x.shape[1]
        if tmp_flag:
            assert(roi.shape[-1]==5), 'The coordinates of rois need to include the batch index!'
            # roi的坐标格式是 [index, x1, y1, x2, y2]
            template = self.prroi_pool(x, roi)

            return x, template
        else:
            return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes, t_strides):
        super(AdjustAllLayer, self).__init__()
        if not isinstance(out_channels, list):
            in_channels = [in_channels]
            out_channels = [out_channels]
            pool_sizes = [pool_sizes]
            t_strides = [t_strides]

        self.num=len(out_channels)
        if self.num == 1:
            self.adjustlayer = AdjustLayer(in_channels[0], out_channels[0],
                                           pool_sizes[0], t_strides[0])
        else:
            for i in range(self.num):
                self.add_module('adjustlayer{:d}'.format(i+2),
                                AdjustLayer(in_channels[i], out_channels[i],
                                            pool_sizes[i], t_strides[i]))

    def forward(self, features, tmp_flag=False, roi=None):
        """
        args:
            features (list[Tensor]) - len(features) = num_level
        """
        if tmp_flag:
            batch_size = features[0].size()[0]
            batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(features[0].device)
            roi_xyxy = torch.cat((roi[:, 0:2], roi[:,0:2]+roi[:,2:4]), dim=-1)
            roi = torch.cat((batch_index, roi_xyxy), dim=-1)

        outs = []
        outs_template=[]

        if self.num == 1:
            out = self.adjustlayer(features[0], tmp_flag, roi)
            if not tmp_flag:
                outs.append(out)
            else:
                outs.append(out[0])
                outs_template.append(out[1])

        else:
            for i in range(self.num):
                adj_layer = getattr(self, 'adjustlayer{:d}'.format(i+2))
                out = adj_layer(features[i], tmp_flag, roi)
                if not tmp_flag:
                    outs.append(out)
                else:
                    outs.append(out[0])
                    outs_template.append(out[1])
        if not tmp_flag:
            return outs
        else:
            return outs, outs_template
