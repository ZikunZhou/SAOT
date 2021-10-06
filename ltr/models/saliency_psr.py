import torch
import torch.nn as nn
import torch.nn.functional as F
from ltr.admin.utils import Visualizer

import math, itertools
import numpy as np

a = Visualizer()

class SaliencyEvaluator_PSR(nn.Module):
    def __init__(self, mainlobe_window_radius=2, detach_saliency=False):
        super(SaliencyEvaluator_PSR, self).__init__()
        self.mainlobe_window_radius = mainlobe_window_radius
        self.detach_saliency = detach_saliency
        self.offsets = self.gen_offsets(self.mainlobe_window_radius)
        print(self.offsets.shape)

    def gen_offsets(self, mainlobe_window_radius=3):
        offset = list(range(-1*mainlobe_window_radius, mainlobe_window_radius+1))
        offsets = list(itertools.permutations(offset, 2))
        offsets.extend(zip(offset, offset))
        return torch.tensor(offsets)

    def forward(self, cost_volume, peak_coords):
        """Evaluate the normalized difference between the maximum and other points
        args:
            cost_volume (Tensor) - shape = [t_batch, t_height*t_width, s_height, s_width]
            peak_coords (Tensor) - shape = [30, 64, 2], (y, x)
        returns:
            saliency (Tensor) - shape = [t_batch, t_height*t_width] (the computed saliency may by negative)
        """
        batch, channel, height, width = cost_volume.shape
        mainlobe_window = peak_coords.unsqueeze(-2).cpu() + self.offsets.unsqueeze(0).unsqueeze(0)
        mainlobe_window = torch.stack([mainlobe_window[...,0].clamp(min=0, max=height-1),
                            mainlobe_window[...,1].clamp(min=0, max=width-1)], dim=-1)
        mainlobe_window = mainlobe_window.view(batch*channel, -1, 2)
        num_mainlobe_points = mainlobe_window.shape[1]

        cost_volume = cost_volume.view(batch*channel, height, width)
        #把所有的这些坐标转换为3个list
        batch_index = torch.arange(batch*channel).unsqueeze(-1).expand(-1, num_mainlobe_points)
        weights = torch.ones(batch*channel, height, width)
        weights[batch_index.numpy().tolist(), mainlobe_window[...,0].numpy().tolist(),\
                                mainlobe_window[...,1].numpy().tolist()] = 0

        cost_volume_flatten = cost_volume.view(batch*channel, height*width)
        weights = weights.view(batch*channel, height*width).to(cost_volume.device)

        num_sidelobe_points = weights.sum(dim=-1, keepdim=True)
        assert(torch.all(num_sidelobe_points>=2)), 'The sidelobe area is too small!'
        mean_sidelobe = (cost_volume_flatten * weights).sum(dim=-1, keepdim=True) / num_sidelobe_points
        var_sidelobe = pow((cost_volume_flatten - mean_sidelobe) * weights, 2).sum(dim=-1, keepdim=True) / (num_sidelobe_points - 1)
        peaks = cost_volume_flatten.max(dim=-1, keepdim=True)[0]

        #cost_volume[batch_index.numpy().tolist(), mainlobe_window[...,0].numpy().tolist(),\
        #                        mainlobe_window[...,1].numpy().tolist()] = -10000.0
        #cost_volume0 = cost_volume[0].view(-1)
        #index = (cost_volume0!=-10000.0).nonzero().squeeze().view(-1)
        #cost_volume0 = cost_volume0[index]
        #psr0 = (peaks[0]-cost_volume0.mean())/cost_volume0.var()

        psr = ((peaks - mean_sidelobe) / var_sidelobe).view(batch, channel)
        saliency = psr / (psr.mean(dim=-1, keepdim=True)+1e-8)

        #print('saliency', saliency[0])
        #print(saliency.mean(dim=-1).mean(), saliency.var(dim=-1).mean())
        if self.detach_saliency:
            return saliency.detach()
        else:
            return saliency
