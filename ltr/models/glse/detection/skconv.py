import torch
import torch.nn as nn

class SKConv(nn.Module):
    def __init__(self, in_channels, branches, groups, down_ratio, stride=1, min_dim=32, dalition_flag=True):
        """ Constructor
        Args:
            in_channels (int) - input channel dimensionality.
            self.branches - the number of branches.
            self.groups - num of convolution groups.
            self.down_ratio - the ratio for compute d, the length of z.
            stride - stride, default 1.
            min_dim - the minimum dim of the vector z in paper, default 32.
        Note:
            I am not sure that whether the fc layer need to contain bias. There is no bias in SeNet's fc layer.
        """
        super(SKConv, self).__init__()
        self.branches = branches#SKConv的分支数量
        self.groups = groups
        self.down_ratio = down_ratio#fc layers的下采样比例
        self.in_channels = in_channels
        self.d = max(int(in_channels/self.down_ratio), min_dim)#中间fc　layer的神经元数量
        self.convs = nn.ModuleList([])
        if dalition_flag:#use dilation convolution to increase RF
            for i in range(self.branches):
                self.convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, dilation=1+i, padding=1+i, groups=self.groups),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=False)
                ))
        else:#increase RF by increasing the kernel size
            for i in range(branches):
                self.convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3+2*i, stride=stride, padding=1+i, groups=self.groups),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=False)
                ))

        self.gap = nn.AdaptiveAvgPool2d((1,1))# global average pooling
        self.fc = nn.Sequential(nn.Linear(in_channels, self.d),
                                nn.BatchNorm1d(self.d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(self.branches):
            self.fcs.append(
                nn.Linear(self.d, in_channels)#no activation
            )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        for i, conv in enumerate(self.convs):
            feature = conv(x).unsqueeze(dim=1)
            if i == 0:
                features = feature
            else:
                features = torch.cat([features, feature], dim=1)
        feature_U = torch.sum(features, dim=1)#sum in dim1,then remove dim1
        feature_s = self.gap(feature_U).squeeze_()#batch*in_channels
        #the next line is neccessary for performing tracking
        if len(feature_s.shape) == 1:
            feature_s = feature_s.unsqueeze(dim=0)
        feature_z = self.fc(feature_s)#batch*self.d
        for i, fc in enumerate(self.fcs):
            vector = fc(feature_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)#add two dimention related to h and w
        feature_v = (features * attention_vectors).sum(dim=1)
        return feature_v

class SKUnit(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, branches,
                groups, down_ratio, stride=1, min_dim=32):
        """ Constructor
        Args:
            in_channels - input channel dimensionality.
            out_channels - output channel dimensionality.
            mid_channels - the channle dim of the middle conv with stride not 1, default out_features/2.
            stride - stride.
            min_dim - the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_channels is None:
            mid_channels = int(out_channels/2)
        self.skunit = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            SKConv(mid_channels, branches, groups, down_ratio, stride=stride, min_dim=min_dim),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, x):
        feature = self.skunit(x)
        return feature
