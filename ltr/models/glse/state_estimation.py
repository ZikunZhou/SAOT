import torch, math
import torch.nn as nn
from ltr.models.glse.neck import AdjustLayer, AdjustAllLayer
from ltr.models.glse.integration import Integration
from ltr.models.glse.detection import BoxSE
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import prroi_pool2d
from collections import Iterable

__visualize__ = False

class Estimator(nn.Module):
    def __init__(self, kp_opt=None):
        super(Estimator, self).__init__()
        self.mesh_y, self.mesh_x = None, None
        self.t_stride = kp_opt.seneck_settings.state_estiamtion_stride

        self.neck = AdjustLayer(neck_opt=kp_opt.seneck_settings)

        self.integrator = Integration(neck_opt=kp_opt.seneck_settings, integrate_opt=kp_opt.seintegrate_settings)

        self.detector = BoxSE(neck_opt=kp_opt.seneck_settings, integrate_opt=kp_opt.seintegrate_settings,
                              detector_opt=kp_opt.sedetector_settings)

    def forward(self, train_feats, test_feats, train_boxes, test_win, test_bb_inwin, online_response, graph_size=(18, 18)):
        """ Only used for off-line training
        args:
            train_feats (list[Tensor]) - len=num_level
            test_feats (list[Tensor]) - len=num_level
            test_win (Tensor) - shape=[num_images*num_sequences, 4]
            online_response (list[Tensor]) - including the response predicted by all the iter filter,
                                             the last one is the final optimized filter, len(online_response)=6,
                                             shape=[num_images*num_sequences, 19, 19]
            graph_size (list) - (height, width) of the subsearch_window_feat
        """

        assert train_boxes.dim() == 3
        num_images, num_sequences = train_boxes.shape[:2]
        if isinstance(train_feats, list):
            train_feat0 = [train_feat[0,...] if train_feat.dim()==5 else \
                       train_feat.reshape(-1, num_sequences, *train_feat.shape[-3:])[0,...] \
                       for train_feat in train_feats]
        else:
            train_feat0 = train_feats[0,...] if train_feats.dim()==5 else \
                          train_feats.reshape(-1, num_sequences, *train_feats.shape[-3:])[0,...]
        train_box0 = train_boxes[0,...]

        init_search_feats, templates, _ = self.neck(train_feat0, tmp_flag=True, roi=train_box0)

        searches, subsearch_windows, subsearch_rois = self.neck(test_feats, tmp_flag=False, roi=test_win,
                                        pooled_height_src=graph_size[0], pooled_width_src=graph_size[1])

        modulated_search = self.integrator(templates, subsearch_windows, graph_size)

        bbox_offsets, output_cls = self.detector(modulated_search)
        return bbox_offsets, output_cls

    def compute_location_map(self, response, subsearch_rois, graph_size):
        batch, channel, height, width = response.shape
        if self.mesh_y is None or self.mesh_x is None:
            self.mesh_y, self.mesh_x = self.compute_mesh(height, width, response)

        max_indices = torch.argmax(response.view(batch, channel, -1), dim=2)
        row_index, column_index = max_indices/width, torch.remainder(max_indices, width)
        row_index = row_index.unsqueeze(-1).unsqueeze(-1).to(torch.float)
        column_index = column_index.unsqueeze(-1).unsqueeze(-1).to(torch.float)

        location_map = -1/math.sqrt(height*width)*(torch.sqrt(torch.pow(self.mesh_y-row_index, 2) + \
                            torch.pow(self.mesh_x-column_index, 2)))
        min_response = location_map.view(batch, channel, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        location_map -= min_response
        max_value = location_map.view(batch, channel, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        location_map = location_map / (max_value + 1e-8)
        location_map = prroi_pool2d(location_map, subsearch_rois, graph_size[0], graph_size[1], 1./self.t_stride)
        return location_map

    def compute_mesh(self, height, width, response):
        row_index = torch.arange(height)
        column_index = torch.arange(width)
        row_indices, column_indices = torch.meshgrid(row_index, column_index)
        row_indices = row_indices.unsqueeze(0).unsqueeze(0).expand_as(response).to(response.device)
        column_indices = column_indices.unsqueeze(0).unsqueeze(0).expand_as(response).to(response.device)
        return row_indices.to(torch.float), column_indices.to(torch.float)

    def init(self, template_feat, target_box):
        """
        args:
            template_feat (Tensor) - shape = [1, channel, height, width]
            target_box (Tensor) - shape = [1, 4]
        """
        init_search_feat, template, _ = self.neck(template_feat, tmp_flag=True, roi=target_box)
        return init_search_feat.detach(), template.detach()

    def track(self, template, test_feat, test_win, pooled_height_src, pooled_width_src):

        search, subsearch_window, subsearch_roi = self.neck(test_feat, tmp_flag=False, roi=test_win,
                    pooled_height_src=pooled_height_src, pooled_width_src=pooled_width_src)

        if __visualize__:
            modulated_search, xcorr_map, processed_xcorr_map = self.integrator(template, subsearch_window, (pooled_height_src, pooled_width_src))
            bbox_offsets, output_cls = self.detector(modulated_search)
            return bbox_offsets.detach(), output_cls.detach(), xcorr_map, processed_xcorr_map

        modulated_search = self.integrator(template, subsearch_window, (pooled_height_src, pooled_width_src))

        bbox_offsets, output_cls = self.detector(modulated_search)
        if output_cls is not None:
            return bbox_offsets.detach(), output_cls.detach()
        else:
            return bbox_offsets.detach(), None
