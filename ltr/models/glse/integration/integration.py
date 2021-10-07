import torch
import torch.nn as nn
import torch.nn.functional as F
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D

from ltr.models.glse.integration.saliency_psrw import SaliencyEvaluator_PSRW
from ltr.models.glse.integration.fusiongcn import FusionGCN_SSA
import numpy as np

import math
import numpy as np


class FGXCorr(nn.Module):
    """
    Here is a problem: If we train the network with a image sequence, the variable ‘num_images’ may be supposed to be 1.
    For now, I'm not sure about this problem
    """
    def __init__(self, template_area):
        """
        args:
            template_area - num of the template feature pixels
        """
        super(FGXCorr, self).__init__()
        self.template_area = template_area

    def forward(self, template, search, norm=False):
        """
        args:
            template (Tensor) - feature map of the target, shape = [t_batch, t_channel, t_height, t_width]
            search (Tensor) - feature map of the search window, shape = [s_batch, s_channel, s_height, s_width]
            norm (bool) - True: compute the consine similarity; False: compute the inner product
        returns:
            fg_xcorr (Tensor) - cost volume between the template and search, shape = [t_batch, t_height*t_width, s_height, s_width]
            num_images (int) - num of the images in the search set
            num_sequences (int) - num of the sequence in a mini-batch
        """
        t_batch, t_channel, t_height, t_width = template.shape
        s_batch, s_channel, s_height, s_width = search.shape
        num_images = int(s_batch/t_batch)
        template = template.unsqueeze(0).expand(num_images, -1, -1, -1, -1)
        template = template.contiguous().view(-1, t_channel, t_height, t_width)
        t_batch = t_batch * num_images

        assert(t_batch == s_batch and t_channel == s_channel), \
            'The number of channels of the template and search features should be the same!'
        assert(self.template_area == t_height*t_width), \
            'The area of the template need to equal to the pre-defined one!'
        template = template.view(t_batch, t_channel, t_height*t_width).permute(0,2,1).contiguous()
        search = search.view(s_batch, s_channel, s_height*s_width)
        fg_xcorr = torch.bmm(template, search)
        if norm:
            magnitude_template = torch.sqrt((template * template).sum(2, keepdim=True))
            magnitude_search = torch.sqrt((search * search).sum(1, keepdim=True))
            norm_item = torch.bmm(magnitude_template, magnitude_search)+1e-8

            assert(torch.all(norm_item!=0)), 'There is one or more zeros in norm_item!'
            fg_xcorr = fg_xcorr / (norm_item)

        fg_xcorr = fg_xcorr.view(t_batch, t_height*t_width, s_height, s_width)
        return fg_xcorr, num_images, int(t_batch/num_images)

class Integration(nn.Module):
    def __init__(self, neck_opt=None, integrate_opt=None):
        super(Integration, self).__init__()
        assert(isinstance(neck_opt.pool_size_tmp, int) or isinstance(neck_opt.pool_size_tmp, float))
        template_area = neck_opt.pool_size_tmp * neck_opt.pool_size_tmp
        self.mesh = None

        self.use_saliency = integrate_opt.use_saliency
        self.saliency_topk = integrate_opt.saliency_topk
        self.use_priori_mask = integrate_opt.use_priori_mask
        self.use_priori_mask_plus = integrate_opt.use_priori_mask_plus
        if self.use_priori_mask:
            self.priori_mask = self.generate_priori_gauss(neck_opt.pool_size_tmp, sigma_factor=integrate_opt.priori_mask_sigma).view(-1)
        if integrate_opt.learn_supress_sigma_factor:
            self.supress_sigma_factor = nn.Parameter(torch.tensor(integrate_opt.supress_sigma_factor))
        else:
            self.supress_sigma_factor = integrate_opt.supress_sigma_factor
        if integrate_opt.learn_saliency_scale_factor:
            self.saliency_scale_factor = nn.Parameter(torch.tensor(integrate_opt.saliency_scale_factor))
        else:
            self.saliency_scale_factor = integrate_opt.saliency_scale_factor

        self.saliencyeval = SaliencyEvaluator_PSRW(mainlobe_window_radius_priori=integrate_opt.mainlobe_window_radius_priori,
                                                   mainlobe_window_radius_max=integrate_opt.mainlobe_window_radius_max,
                                                   min_k=integrate_opt.min_k, ml_width_pow=integrate_opt.ml_width_pow,
                                                   detach_saliency=integrate_opt.detach_saliency, use_std_dev=integrate_opt.use_std_dev)

        self.xcorr = FGXCorr(template_area)

        self.fusiongcn = FusionGCN_SSA(gcn_in_channels=integrate_opt.gcn_in_channels,
                                       gcn_hidden=integrate_opt.gcn_hidden_channels,
                                       gcn_out_channels=integrate_opt.gcn_out_channels,
                                       graph_sizes=integrate_opt.possible_graph_sizes,
                                       phi_in_channels=integrate_opt.phi_in_channels,
                                       phi_hidden_channels=integrate_opt.phi_hidden_channels,
                                       use_high_order_poly = integrate_opt.use_high_order_poly,
                                       order_num = integrate_opt.order_num,
                                       use_sa_adjust_edge = integrate_opt.use_sa_adjust_edge,
                                       use_difference=integrate_opt.use_difference,
                                       saliency_topk=self.saliency_topk)

    def forward(self, template, search, graph_size):
        xcorr_map, num_images, num_sequences = self.xcorr(template, search, norm=True)
        peak_coords = self.find2Dpeak(xcorr_map)

        if self.mesh is None:
            self.mesh = self.compute_mesh(xcorr_map)
        if self.use_saliency:
            saliency = self.saliencyeval(xcorr_map, peak_coords, self.mesh)# shape=[30, 64]
        else:
            saliency = torch.ones(*xcorr_map.shape[:2], device=xcorr_map.device)

        normed_saliency = F.softmax(saliency * self.saliency_scale_factor, dim=1) * saliency.shape[1]
        if self.use_priori_mask:
            if self.use_priori_mask_plus:
                normed_saliency = (normed_saliency + (self.priori_mask/self.priori_mask.mean()).unsqueeze(0).to(normed_saliency.device)) / 2.
            else:
                normed_saliency = normed_saliency * self.priori_mask.unsqueeze(0).to(normed_saliency.device)

        processed_xcorr_map = self.process_xcorr(xcorr_map, normed_saliency, peak_coords)
        modulated_search = self.fusiongcn(search, processed_xcorr_map, normed_saliency, peak_coords, graph_size)

        return modulated_search

    def find2Dpeak(self, xcorr_map):
        batch, channel, height, width = xcorr_map.shape
        max_indices = torch.argmax(xcorr_map.view(batch, channel, -1), dim=2)
        y_coordinates, x_coordinates = max_indices/width, torch.remainder(max_indices, width)
        peak_coords = torch.stack([y_coordinates, x_coordinates], dim=-1)
        return peak_coords

    def process_xcorr(self, xcorr_map, normed_saliency, peak_coords):
        """
        args:
            xcorr_map (Tensor) - shape = [30, 64, 18, 18]
            normed_saliency (Tensor) - shape = [30, 64]
            peak_coords (Tensor) - shape = [30, 64, 2], (y, x)
        returns:
            xcorr_map (Tensor) - shape = [30, 64, 18, 18]
        """
        batch, channel, _, _ = xcorr_map.shape
        try:
            assert(torch.all(normed_saliency>=0)), 'Non-positive normed saliency!'
        except Exception as e:
            print(normed_saliency)

        peak_coords = peak_coords.to(torch.float)
        peak_y = peak_coords[:,:,0].unsqueeze(-1).unsqueeze(-1).expand_as(xcorr_map)
        peak_x = peak_coords[:,:,1].unsqueeze(-1).unsqueeze(-1).expand_as(xcorr_map)
        normed_saliency = normed_saliency.unsqueeze(-1).unsqueeze(-1).expand_as(xcorr_map)

        sigma = self.supress_sigma_factor

        if isinstance(sigma, torch.Tensor):
            gauss_maps = normed_saliency * torch.exp(-0.5*torch.pow(self.mesh[0]-peak_y, 2)/torch.pow(sigma, 2) + \
                     -0.5*torch.pow(self.mesh[1]-peak_x, 2)/torch.pow(sigma, 2))
        else:
            gauss_maps = normed_saliency * torch.exp(-0.5*torch.pow(self.mesh[0]-peak_y, 2)/pow(float(sigma), 2) + \
                     -0.5*torch.pow(self.mesh[1]-peak_x, 2)/pow(float(sigma), 2))

        return xcorr_map * gauss_maps

    def compute_mesh(self, xcorr_map):
        """
        returns: shape=[30, 64, 18, 18]

        """
        row_index = torch.arange(xcorr_map.shape[-2])
        column_index = torch.arange(xcorr_map.shape[-1])
        row_indices, column_indices = torch.meshgrid(row_index, column_index)

        row_indices = row_indices.unsqueeze(0).unsqueeze(0).expand_as(xcorr_map).to(torch.float).to(xcorr_map.device)
        column_indices = column_indices.unsqueeze(0).unsqueeze(0).expand_as(xcorr_map).to(torch.float).to(xcorr_map.device)
        return row_indices, column_indices

    def generate_priori_gauss(self, size, sigma_factor=0.75):
        row_index = torch.arange(size).to(torch.float)
        column_index = torch.arange(size).to(torch.float)
        center = (size - 1) / 2.
        sigma = size * sigma_factor
        row_indices, column_indices = torch.meshgrid(row_index, column_index)
        gauss = torch.exp(-0.5*(torch.pow((row_indices-center)/sigma, 2)+torch.pow((column_indices-center)/sigma, 2)))
        return gauss
