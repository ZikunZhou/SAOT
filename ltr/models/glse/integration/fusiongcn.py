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


class FusionGCN(nn.Module):
    def __init__(self, gcn_in_channels, gcn_hidden, gcn_out_channels, graph_sizes, phi_in_channels,
                    phi_hidden_channels, gcn_process_feat_xcorr):
        """
        args:
            in_channels (int): channel num of the input graph node feature
            hidden (int): channel num of the output graph node feature of the first gcn layer
            out_channels (int): output channel num
            graph_sizes: (list(size1, size2, ...)) pre-defined graph sizes
            phi_in_channels: pre-1*1-conv-layer input channel num, which should be the same
                             as the channel num of the search feature
            phi_hidden_channels: pre-1*1-conv-layer output channel num

        """
        super(FusionGCN, self).__init__()
        if isinstance(graph_sizes[0], int):
            self.graph_sizes = [(graph_size, graph_size) for graph_size in graph_sizes]
        else:
            self.graph_sizes = graph_sizes

        self.gcn_process_feat_xcorr = gcn_process_feat_xcorr

        self.coords_pair_8neigh_dict = OrderedDict()
        for graph_size in self.graph_sizes:
            self.coords_pair_8neigh_dict[graph_size] = self.gen_8neigh_coords_pair(*graph_size)

        self.phi = nn.Sequential(
                LinearBlock(phi_in_channels, phi_hidden_channels, input_sz=1, bias=True, batch_norm=True, relu=True),
                nn.Linear(phi_hidden_channels, 1, bias=True),
                nn.Sigmoid())

        self.gc1 = GraphConvLayer(gcn_in_channels, gcn_hidden)
        self.gc2 = GraphConvLayer(gcn_hidden, gcn_out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, search_feature, xcorr_map, saliency, key_coords, graph_size):
        #起码应该加一个sigmoid
        """
        args:
            search_feature (Tensor) - shape=[num_images*num_sequences, channel, *graph_size]
            key_coords ([Tensor]) - shape=[num_images*num_sequences, num_keypoints, 2] coordinate_defination=[y, x] corresponding to [row, column]
        """

        assert(isinstance(graph_size, tuple)), 'graph_size is used as dict key, need to be a tuple!'

        coords_pair_kpoint = self.gen_kpoint_coords_pair(*graph_size, key_coords)
        coords_pair = self.gen_coords_pair(coords_pair_kpoint, self.coords_pair_8neigh_dict[graph_size], search_feature.device)

        catted_feature = torch.cat([search_feature, xcorr_map], dim=1)
        adjacency_mats = self.cal_adjacency_mat(coords_pair, catted_feature)

        normed_adjmat = self.gen_adjs(adjacency_mats)
        if self.gcn_process_feat_xcorr:
            node_feat = torch.cat([search_feature, xcorr_map], dim=1)
        else:
            node_feat = xcorr_map
        batch, channel, height, width = node_feat.shape
        node_feat = node_feat.view(batch, channel, height*width).permute(0,2,1)
        node_feat = self.relu(self.gc1(node_feat, adjacency_mats))
        spatial_mask = self.sigmoid(self.gc2(node_feat, adjacency_mats))
        spatial_mask = spatial_mask.view(batch, height, width).unsqueeze(1)
        return spatial_mask

    def cal_adjacency_mat(self, coords_pair, features):
        """
        args:
            coords_pair (list[Tensor]) -
            features (Tensor) - shape [num_images*num_sequences, channels, height, width]
        """
        batch, channel, height, width = features.shape
        flatter_features = features.permute(2,3,0,1).contiguous().view(-1, batch, channel).permute(1,2,0)
        adjacency_mats = []
        for i, coords_pair in enumerate(coords_pair):
            #print(coords_pair.shape)
            adjacency_mat = torch.zeros(height*width, height*width, device=features.device)
            first_point_feature = flatter_features[i][:, coords_pair[:,0]].permute(1,0).contiguous()
            second_point_feature = flatter_features[i][:, coords_pair[:,1]].permute(1,0).contiguous()
            edge = self.phi(torch.abs(first_point_feature-second_point_feature)).view(-1)
            indices = coords_pair.permute(1,0).cpu().numpy().tolist()
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

    def gen_kpoint_coords_pair(self, height, width, key_coords):
        """
        key_coords (Tensor) - shape - [num_images*num_sequences, num_keypoints, 2]
        """
        coords_pair_kpoint = []
        key_coords_indices = (key_coords[:,:,0] * width + key_coords[:,:,1])
        for key_coords_indice in key_coords_indices:
            key_coords_indice = torch.unique(key_coords_indice, sorted=False)
            if len(key_coords_indice) >= 2:
                combine2 = list(itertools.permutations(key_coords_indice.cpu().numpy().tolist(),2))
                tensor_combine2 = torch.tensor(combine2)
                coords_pair_kpoint.append(tensor_combine2)
            else:
                coords_pair_kpoint.append(None)
        return coords_pair_kpoint

    def gen_coords_pair(self, coords_pair_kpoint, coords_pair_8neigh, device):
        """
        coords_pair_kpoint (list[Tensor])
        """
        coords_pair = []
        for coords_pair_kp in coords_pair_kpoint:
            if coords_pair_kp is not None:
                coords_pair.append(torch.cat([coords_pair_kp, coords_pair_8neigh.clone()], dim=0).to(device).unique(sorted=False, dim=0))
            else:
                coords_pair.append(coords_pair_8neigh.clone().to(device))
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
