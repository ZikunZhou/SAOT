import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math, itertools
import numpy as np

class SaliencyEvaluator_PSRW(nn.Module):
    def __init__(self, mainlobe_window_radius_priori=1.5, mainlobe_window_radius_max=4,
                 min_k=1, ml_width_pow=1, detach_saliency=False, use_std_dev=False):
        super(SaliencyEvaluator_PSRW, self).__init__()
        self.detach_saliency = detach_saliency
        self.mainlobe_window_radius_max = mainlobe_window_radius_max
        self.gen_offsets(range(1, mainlobe_window_radius_max+1))
        self.offsets_priori = self.get_offsets(self.offsets_dict[math.ceil(mainlobe_window_radius_priori)], \
                                               mainlobe_window_radius_priori)
        self.offsets_max = None
        self.ml_width_pow = ml_width_pow
        self.min_k = min_k
        self.use_std_dev = use_std_dev


    def gen_offsets(self, mainlobe_window_radius_list = [1,2,3,4]):
        self.offsets_dict = OrderedDict()
        for radius in mainlobe_window_radius_list:
            offset = list(range(-1*radius, radius+1))
            offsets = list(itertools.permutations(offset, 2))
            offsets.extend(zip(offset, offset))
            self.offsets_dict[radius] = torch.tensor(offsets).cuda()


    def get_offsets(self, offsets, mainlobe_window_radius=3):
        """
        args:
            mainlobe_window_radius (int or float or Tensor)
        """
        mask = torch.sqrt(torch.pow(offsets, 2).sum(dim=-1).to(torch.float)) <= mainlobe_window_radius
        offsets = (mask.to(torch.float).unsqueeze(-1) * offsets.to(torch.float)).to(torch.long)
        return offsets

    def forward(self, cost_volume, peak_coords, mesh):
        """Evaluate the normalized difference between the maximum and other points
        args:
            cost_volume (Tensor) - shape = [t_batch, t_height*t_width, s_height, s_width]
            peak_coords (Tensor) - shape = [30, 64, 2], (y, x)
            mesh (list[Tensor]) - shape = [30, 64, 18, 18], (y, x)
        returns:
            saliency (Tensor) - shape = [t_batch, t_height*t_width] (the computed saliency may by negative)
        """
        batch, channel, height, width = cost_volume.shape
        if self.offsets_max is None:
            self.offsets_max = self.offsets_dict[self.mainlobe_window_radius_max].unsqueeze(0).expand(batch*channel, -1, -1).to(cost_volume.device)

        # -----------------------compute approaximate mean start-----------------------------
        mainlobe_window_priori = peak_coords.unsqueeze(-2) + self.offsets_priori.unsqueeze(0).unsqueeze(0).to(peak_coords.device)
        mainlobe_window_priori = torch.stack([mainlobe_window_priori[...,0].clamp(min=0, max=height-1),
                            mainlobe_window_priori[...,1].clamp(min=0, max=width-1)], dim=-1)
        mainlobe_window_priori = mainlobe_window_priori.view(batch*channel, -1, 2).cpu()
        priori_weights = torch.ones(batch*channel, height, width, device=cost_volume.device)
        
        num_mainlobe_points_priori = mainlobe_window_priori.shape[1]
        batch_index_priori = torch.arange(batch*channel).unsqueeze(-1).expand(-1, num_mainlobe_points_priori)

        priori_weights[batch_index_priori, mainlobe_window_priori[...,0], \
                                mainlobe_window_priori[...,1]] = 0

        priori_weights = priori_weights.view(batch*channel, height*width)
        num_sidelobe_points_priori = priori_weights.sum(dim=-1, keepdim=True)

        cost_volume = cost_volume.view(batch*channel, height, width)
        cost_volume_flatten = cost_volume.view(batch*channel, height*width)
        peak_coords_flatten = peak_coords.view(batch*channel, -1)
        cost_volume_mean = (cost_volume_flatten * priori_weights).sum(dim=-1, keepdim=True) / num_sidelobe_points_priori
        # -----------------------compute approaximate mean end-----------------------------

        mesh_y = mesh[0].view(batch*channel, height*width).to(torch.long)
        mesh_x = mesh[1].view(batch*channel, height*width).to(torch.long)

        distance = torch.sqrt((torch.pow(mesh_y - peak_coords_flatten[:,:1], 2) + \
                               torch.pow(mesh_x - peak_coords_flatten[:,1:], 2)).to(torch.float))

        batch_index = torch.arange(batch*channel)
        compare_results = (cost_volume <= cost_volume_mean.unsqueeze(-1))
        compare_results[batch_index, peak_coords_flatten[:,0].cpu(), \
                        peak_coords_flatten[:,1].cpu()] = 0

        compare_results_flatten = compare_results.view(batch*channel, -1)

        drop_points_coord = (compare_results_flatten<=0).nonzero().cpu().permute(1,0)
        distance[drop_points_coord[0], drop_points_coord[1]] = 100

        mainlobe_widthes, closest_sidelobe_indices = torch.topk(distance, self.min_k, dim=-1, largest=False)
        mainlobe_widthes = mainlobe_widthes.mean(dim=-1, keepdim=True)

        mainlobe_window = peak_coords_flatten.unsqueeze(-2) + self.get_offsets(self.offsets_max, \
                                                              torch.clamp(mainlobe_widthes, 1.5, self.mainlobe_window_radius_max+0.5))
        mainlobe_window = torch.stack([mainlobe_window[...,0].clamp(min=0, max=height-1),
                            mainlobe_window[...,1].clamp(min=0, max=width-1)], dim=-1).cpu()
        num_mainlobe_points = mainlobe_window.shape[1]
        batch_index = torch.arange(batch*channel).unsqueeze(-1).expand(-1, num_mainlobe_points)

        mask_weights = torch.ones(batch*channel, height, width, device=cost_volume.device)
        mask_weights[batch_index, mainlobe_window[...,0], \
                                mainlobe_window[...,1]] = 0

        
        mask_weights = mask_weights.view(batch*channel, height*width)
        num_sidelobe_points = mask_weights.sum(dim=-1, keepdim=True)
        assert(torch.all(num_sidelobe_points>=2)), 'The sidelobe area is too small!'
        mean_sidelobe = (cost_volume_flatten * mask_weights).sum(dim=-1, keepdim=True) / num_sidelobe_points
        if self.use_std_dev:
            var_sidelobe = pow(pow((cost_volume_flatten - mean_sidelobe) * mask_weights, 2).sum(dim=-1, keepdim=True) / (num_sidelobe_points - 1) + 1e-16, 0.5)
        else:
            var_sidelobe = pow((cost_volume_flatten - mean_sidelobe) * mask_weights, 2).sum(dim=-1, keepdim=True) / (num_sidelobe_points - 1)
        peaks = cost_volume_flatten.max(dim=-1, keepdim=True)[0]

        psrw = ((peaks - mean_sidelobe) / (var_sidelobe*torch.pow(mainlobe_widthes, self.ml_width_pow)+1e-16)).view(batch, channel)

        saliency = psrw / (psrw.mean(dim=-1, keepdim=True)+1e-8)

        if self.detach_saliency:
            return saliency.detach()
        else:
            return saliency
