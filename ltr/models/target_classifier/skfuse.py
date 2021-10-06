import torch
from torch import nn

class SKLayer(nn.Module):
    def __init__(self, used_layers=['layer2', 'layer3', 'layer4'], in_channels=[512, 1024, 2048],
                strides=[1, 1, 1], out_channels=1024, down_ratio=4, min_dim=128):
        """ Constructor
        Args:
            used_layers (list) - layers to be fused in resnet
            in_channels (list) - input channels dimensionality of used resnet layers..
            strides (list) - stride for each pre-processing convs layer, all the strides need to be the same.
            out_channels (int) - output channels
            down_ratio - the ratio for computing fc dimentions
            min_dim - the minimum dim of the vector z in paper, default 128.
        Note:
            I am not sure that whether the fc layer need to contain bias. There is no bias in SeNet's fc layer.
        """
        super(SKLayer, self).__init__()
        self.branches = used_layers
        self.in_channels = in_channels
        self.strides = strides
        self.out_channels = out_channels
        self.down_ratio = down_ratio
        self.min_dim = max(int(self.out_channels/self.down_ratio), min_dim)#中间fc　layer的神经元数量

        self.convs = nn.ModuleList([])
        for layer, in_channels, stride in zip(self.branches, self.in_channels, self.strides):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm2d(self.out_channels),
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))# global average pooling
        self.fc = nn.Sequential(nn.Linear(self.out_channels, self.min_dim),
                                nn.BatchNorm1d(self.min_dim),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for layer in self.branches:
            self.fcs.append(
                nn.Linear(self.min_dim, self.out_channels)#no activation
            )
        self.softmax = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm2d(self.out_channels)


    def forward(self, x):
        for i, (xx, conv) in enumerate(zip(x.values(), self.convs)):
            feature = conv(xx).unsqueeze_(dim=1)
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
            vector = fc(feature_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)#add two dimention related to h and w
        feature_v = self.bn((features * attention_vectors).sum(dim=1))

        return feature_v


class SeLayer(nn.Module):
    def __init__(self, cfg, in_channels, L = 16):
        super(SeLayer, self).__init__()
        self.down_ratio = cfg.MODEL.SE.DOWN_RATIO#fc layers的下采样比例
        #self.down_ratio = 4
        self.d = max(int(in_channels/self.down_ratio), L)#中间fc　layer的神经元数量
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.d, bias = False),
            nn.ReLU(inplace = False),
            nn.Linear(self.d, in_channels, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, _, _ = x.size()
        y = self.gap(x).view(batch, channel)
        y = self.fc(y).view(batch, channel, 1, 1)
        #return x * y.expand_as(x)
        return y


if __name__=='__main__':
    x1 = torch.rand(8,512,32,32)
    x2 = torch.rand(8,1024,32,32)
    x3 = torch.rand(8,2048,32,32)
    x = [x1, x2, x3]
    conv = SKLayer()
    print(conv)
    out = conv(x)
    print(out.shape)
    for key, param in conv.named_parameters():
        print(key)
