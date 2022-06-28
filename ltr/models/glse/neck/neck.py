# naive version, share adjustlayer(conv3), fixed pooling size
import torch
import torch.nn as nn
import torch.nn.functional as f
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import prroi_pool2d
# from mmdet.ops import RoIAlign, roi_align, roi_pool

class AdjustLayer(nn.Module):
    """pool_size_src is not given until start forward propa
    """
    def __init__(self, neck_opt=None):
        super(AdjustLayer, self).__init__()
        self.t_stride = neck_opt.state_estiamtion_stride
        self.pool_size_tmp = neck_opt.pool_size_tmp
        self.downsample = nn.Sequential(
            nn.Conv2d(neck_opt.neck_in_channels, neck_opt.neck_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(neck_opt.neck_out_channels),
        )

        self.adjust = nn.Sequential(
            nn.Conv2d(neck_opt.neck_out_channels, neck_opt.neck_out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(neck_opt.neck_out_channels),
            nn.ReLU(inplace=True)
        )

        self.prroi_pool_tmp = PrRoIPool2D(self.pool_size_tmp, self.pool_size_tmp, 1./self.t_stride)


    def forward(self, x, tmp_flag, roi=None, pooled_height_src=18, pooled_width_src=18):
        """
        args:
            x - feature map of the template patch or search patch
            tmp_flag - the flag for template
            roi - shape: [num_roi, 5], coordinates: [index, x1, y1, x2, y2]
        """
        assert(roi.dim()==2), "The dimension of roi should be set as 2!"
        if roi.shape[-1] != 5:
            batch_size = x.size()[0]
            batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(x.device)
            roi_xyxy = torch.cat((roi[:, 0:2], roi[:,0:2]+roi[:,2:4]), dim=-1)
            roi = torch.cat((batch_index, roi_xyxy), dim=-1)

        x = self.downsample(x)
        x = self.adjust(x)
        channel = x.shape[1]

        assert(roi.shape[-1]==5), 'The coordinates of rois need to include the batch index!'
        # roi: [index, x1, y1, x2, y2]
        if tmp_flag:
            roi_feature = self.prroi_pool_tmp(x, roi)
        else:
            roi_feature = prroi_pool2d(x, roi, pooled_height_src, pooled_width_src, 1./self.t_stride)
        return x, roi_feature, roi



class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes_tmp, pool_sizes_src, t_strides):
        super(AdjustAllLayer, self).__init__()
        if not isinstance(out_channels, list):
            in_channels = [in_channels]
            out_channels = [out_channels]
            pool_sizes_tmp = [pool_sizes_tmp]
            pool_sizes_src = [pool_sizes_src]
            t_strides = [t_strides]

        self.num=len(out_channels)
        if self.num == 1:
            self.adjustlayer = AdjustLayer(in_channels[0], out_channels[0],
                                           pool_sizes_tmp[0], pool_sizes_src[0], t_strides[0])
        else:
            for i in range(self.num):
                self.add_module('adjustlayer{:d}'.format(i+2),
                                AdjustLayer(in_channels[i], out_channels[i],
                                            pool_sizes_tmp[i], pool_sizes_src[i], t_strides[i]))

    def forward(self, features, tmp_flag=False, roi=None):
        """
        args:
            features (list[Tensor]) - len(features) = num_level
        """

        batch_size = features[0].size()[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(features[0].device)
        roi_xyxy = torch.cat((roi[:, 0:2], roi[:,0:2]+roi[:,2:4]), dim=-1)
        roi = torch.cat((batch_index, roi_xyxy), dim=-1)

        outs = []
        outs_roi=[]
        if self.num == 1:
            out, roi_feature = self.adjustlayer(features[0], tmp_flag, roi)
            outs.append(out)
            outs_roi.append(roi_feature)

        else:
            for i in range(self.num):
                adj_layer = getattr(self, 'adjustlayer{:d}'.format(i+2))
                out, roi_feature = adj_layer(features[i], tmp_flag, roi)
                outs.append(out)
                outs_roi.append(roi_feature)

        return outs, outs_roi
