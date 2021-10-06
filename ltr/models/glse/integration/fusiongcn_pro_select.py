# the second version of fusiongcn
import torch
import torch.nn as nn
from torch.nn import Parameter
from ltr.models.layers.blocks import LinearBlock
from collections import OrderedDict
import math, itertools
import numpy as np

class GraphConvLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class HO_POLY_GraphConvLayer(nn.Module):
    """
    GAN layer with high-order polynomial of the adjacency matrix
    """
    def __init__(self, in_features, out_features, bias=False, high_order=1):
        super(HO_POLY_GraphConvLayer, self).__init__()
        self.parallel_gcn_layers = nn.ModuleList([])
        self.high_order = high_order
        self.weights = nn.Parameter(torch.ones(high_order))

        for i in range(self.high_order):
            gcn_layer = GraphConvLayer(in_features, out_features, bias)
            gcn_layer.reset_parameters()
            self.parallel_gcn_layers.append(gcn_layer)

    def forward(self, input, adj):
        adj_list = [adj]
        for i in range(self.high_order-1):
            adj_list.append(torch.matmul(adj_list[-1], adj))
        outputs = []
        for i, gcn_layer in enumerate(self.parallel_gcn_layers):
            outputs.append(gcn_layer(input, adj_list[i]))

        outputs = torch.stack(outputs, dim=0)*(self.weights.view(-1,1,1,1))
        output = outputs.sum(dim=0)/self.weights.sum()
        return output


class FusionGCN_SSA(nn.Module):
    def __init__(self, gcn_in_channels, gcn_hidden, gcn_out_channels, graph_sizes, phi_in_channels,
                    phi_hidden_channels, use_high_order_poly=False,
                    order_num=2, use_sa_adjust_edge=False, use_difference=False, saliency_topk=32):
        """
        args:
            in_channels (int): channel num of the input graph node feature
            hidden (int): channel num of the output graph node feature of the first gcn layer
            out_channels (int): output channel num
            graph_sizes: (list(size1, size2, ...)) pre-defined graph sizes
            phi_in_channels: pre-1*1-conv-layer input channel num, which should be the same
                             as the channel num of the search feature
            phi_hidden_channels: pre-1*1-conv-layer output channel num
            gcn_process_feat_xcorr: whether gcn only process the xcorr map or both the feature and xcorr maps

        """
        super(FusionGCN_SSA, self).__init__()
        if isinstance(graph_sizes[0], int):
            self.graph_sizes = [(graph_size, graph_size) for graph_size in graph_sizes]
        else:
            self.graph_sizes = graph_sizes

        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.gcn_out_channels = gcn_out_channels
        self.use_sa_adjust_edge = use_sa_adjust_edge
        self.use_difference = use_difference
        self.saliency_topk = saliency_topk

        self.coords_pair_8neigh_dict = OrderedDict()
        for graph_size in self.graph_sizes:
            self.coords_pair_8neigh_dict[graph_size] = self.gen_8neigh_coords_pair(*graph_size)

        self.phi = nn.Sequential(
                LinearBlock(phi_in_channels, phi_hidden_channels, input_sz=1, bias=True, batch_norm=True, relu=True),
                nn.Linear(phi_hidden_channels, 1, bias=True),
                nn.Sigmoid())
        if use_high_order_poly:
            self.gc1 = HO_POLY_GraphConvLayer(gcn_in_channels, gcn_hidden, order_num)
            self.gc2 = HO_POLY_GraphConvLayer(gcn_hidden, gcn_out_channels, order_num)
        else:
            self.gc1 = GraphConvLayer(gcn_in_channels, gcn_hidden)
            self.gc2 = GraphConvLayer(gcn_hidden, gcn_out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.activate_layer = nn.ReLU()


    def forward(self, search_feature, xcorr_map, saliency, key_coords, graph_size):
        #起码应该加一个sigmoid
        """
        args:
            search_feature (Tensor) - shape=[num_images*num_sequences, channel, *graph_size]
            key_coords ([Tensor]) - shape=[num_images*num_sequences, num_keypoints, 2] coordinate_defination=[y, x] corresponding to [row, column]
        """
        #print(self.alpha)
        assert(isinstance(graph_size, tuple)), 'graph_size is used as dict key, need to be a tuple!'

        coords_pair_kpoint, edge_weights = self.gen_kpoint_coords_pair(*graph_size, key_coords, saliency)

        coords_pair = self.gen_coords_pair(coords_pair_kpoint, self.coords_pair_8neigh_dict[graph_size], search_feature.device)

        catted_feature = torch.cat([search_feature, xcorr_map], dim=1)
        adjacency_mats = self.cal_adjacency_mat(coords_pair, catted_feature, edge_weights)

        normed_adjmat = self.gen_adjs(adjacency_mats)
        node_feat = torch.cat([search_feature, xcorr_map], dim=1)

        batch, channel, height, width = node_feat.shape
        node_feat = node_feat.view(batch, channel, height*width).permute(0,2,1)
        node_feat = self.relu(self.gc1(node_feat, adjacency_mats))

        spatial_mask = self.activate_layer(self.gc2(node_feat, adjacency_mats)).permute(0,2,1)
        spatial_mask = spatial_mask.view(batch, self.gcn_out_channels, height, width)

        return spatial_mask

    def cal_adjacency_mat(self, coords_pair, features, edge_weights=None):
        """The adjacency_mats of every sample are different, compute seperately
        args:
            coords_pair (list[Tensor]) - len = num_kps, shape=[num_node_pair, 2]
            features (Tensor) - shape [num_images*num_sequences, channels, height, width]
            edge_weights (Tensor) - shape=[num_images*num_sequences, height*width]
        """
        #print(self.alpha)
        if self.use_sa_adjust_edge:
            edge_weights = edge_weights.detach()
        batch, channel, height, width = features.shape
        flatter_features = features.permute(2,3,0,1).contiguous().view(-1, batch, channel).permute(1,2,0)
        adjacency_mats = []
        for i, coord_pair in enumerate(coords_pair):
            adjacency_mat = torch.zeros(height*width, height*width, device=features.device)
            first_point_feature = flatter_features[i][:, coord_pair[:,0]].permute(1,0).contiguous()
            second_point_feature = flatter_features[i][:, coord_pair[:,1]].permute(1,0).contiguous()
            if self.use_sa_adjust_edge:
                first_point_weight = edge_weights[i][coord_pair[:,0]].unsqueeze(1)
                second_point_weight = edge_weights[i][coord_pair[:,1]].unsqueeze(1)
                edge = self.phi(torch.exp(self.alpha*first_point_weight) * first_point_feature * \
                                torch.exp(self.alpha*second_point_weight) * second_point_feature).view(-1)
            elif self.use_difference:
                edge = self.phi(torch.abs(first_point_feature-second_point_feature)).view(-1)
            else:
                edge = self.phi(first_point_feature*second_point_feature).view(-1)
            indices = coord_pair.permute(1,0).cpu().numpy().tolist()
            adjacency_mat[indices[0], indices[1]] = edge
            adjacency_mats.append(adjacency_mat)

        return torch.stack(adjacency_mats, dim=0)

    def gen_8neigh_coords_pair(self, height, width):#应该对这个邻接矩阵做一个行归一化
        #不同的图对应的这个邻接矩阵应该一致
        row_coord = torch.arange(height)
        column_coord = torch.arange(width)
        row_coords, column_coords = torch.meshgrid(row_coord, column_coord)
        coords = torch.stack([row_coords, column_coords], dim=-1).view(-1, 2)
        coords_pair_8neigh = []

        for i, i_coor in enumerate(coords):
            for j, j_coor in enumerate(coords):
                distance = torch.pow((i_coor - j_coor), 2).sum()
                if distance > 0 and distance <= 2:
                    coords_pair_8neigh.append([i, j])
        coords_pair_8neigh = torch.tensor(coords_pair_8neigh)
        print(len(coords_pair_8neigh))
        return coords_pair_8neigh

    def gen_kpoint_coords_pair(self, height, width, key_coords, normed_saliency):
        """
        args:
            key_coords (Tensor) - shape=[num_images*num_sequences, num_keypoints, 2] coordinate_defination=[y, x] corresponding to [row, column]
            normed_saliency (Tensor) -normalized saliency in integration.process_xcorr, whose mean equals 1, shape=[num_images*num_sequences, height*width]
        returns:
            edge_weights (Tensor)
            key_point_pairs (list[Tensor]) - len = num_kps, shape=[num_node_pair, 2]
        """

        batch = normed_saliency.shape[0]
        edge_weights = torch.zeros(batch, height*width).to(torch.float32)
        key_point_pairs = []
        key_indices = key_coords[:,:,0] * width + key_coords[:,:,1]
        for i, key_index in enumerate(key_indices):
            unique_key_index, inverse_indices = torch.unique(key_index, return_inverse=True, sorted=False)

            # ------------------compute the edge weight-------------------------
            unique_key_saliency = []
            for j in range(len(unique_key_index)):
                indices = torch.nonzero(inverse_indices==j).squeeze()
                unique_key_saliency.append(normed_saliency[i, indices].max().item())

            unique_key_saliency = torch.tensor(unique_key_saliency)
            if self.saliency_topk < len(unique_key_saliency):
                if torch.all(unique_key_saliency==1):
                    select_key_index = torch.from_numpy(np.random.choice(np.arange(len(unique_key_saliency)), \
                                                                         self.saliency_topk, replace=False))
                    unique_key_saliency = unique_key_saliency[select_key_index]
                else:
                    unique_key_saliency, select_key_index = torch.topk(unique_key_saliency, self.saliency_topk, dim=-1, largest=True)
                unique_key_index = unique_key_index[select_key_index]
            edge_weights[i, unique_key_index] = unique_key_saliency

            # ----------------compute key point pair index----------------------
            if len(unique_key_index) >= 2:
                combine2 = list(itertools.permutations(unique_key_index.cpu().numpy().tolist(),2))
                tensor_combine2 = torch.tensor(combine2)
                key_point_pairs.append(tensor_combine2)
            else:
                key_point_pairs.append(None)
        return key_point_pairs, edge_weights.to(key_coords.device)

    def gen_coords_pair(self, coords_pair_kpoint, coords_pair_neigh, device):
        """
        coords_pair_kpoint (list[Tensor])
        """
        coords_pair = []
        for coords_pair_kp in coords_pair_kpoint:
            if coords_pair_kp is not None:
                coords_pair.append(torch.cat([coords_pair_kp, coords_pair_neigh.clone()], dim=0).to(device).unique(sorted=False, dim=0))
            else:
                coords_pair.append(coords_pair_neigh.clone().to(device))
        return coords_pair

    def gen_adjs(self, As):
        """Validated
        args:
            As (Tensor) - shape = [batch, N, N]
        """
        A_hats = As + torch.eye(As.shape[1]).unsqueeze(0).to(As.device)
        D_hats = torch.pow(A_hats.sum(2), -0.5)
        D_hats = torch.stack([torch.diag(D_hat) for D_hat in D_hats], dim=0)
        normed_adjmat = torch.matmul(torch.matmul(A_hats, D_hats).transpose(dim0=1, dim1=2), D_hats)
        return normed_adjmat

    def gen_adj(self, A):
        # 没有对batch进行处理
        # A是二维邻接矩阵
        A_hat = A + torch.eye(A.shape[0]).to(A.device)
        D_hat = torch.pow(A_hat.sum(1).float(), -0.5)
        D_hat = torch.diag(D_hat)#生成对角化的度矩阵
        adj = torch.matmul(torch.matmul(A_hat, D_hat).t(), D_hat)
        return adj


if __name__ == "__main__":
    adjacency_4neigh, coords_dict = gen_adjmat_4neigh(3,3)
    gen_adjmat_saliency(3, 3, coords_dict, list(coords_dict.keys())[0:3])
