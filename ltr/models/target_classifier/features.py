import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from ltr.models.layers.normalization import InstanceL2Norm
from ltr.models.layers.transform import InterpCat
from ltr.models.target_classifier.skfuse import SKLayer

def residual_basic_block(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                         interp_cat=False, final_relu=False, init_pool=False):
    """Construct a network block based on the BasicBlock used in ResNet 18 and 34."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    if interp_cat:
        feat_layers.append(InterpCat())
    if init_pool:
        feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(BasicBlock(feature_dim, odim))
    if final_conv:
        feat_layers.append(nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
        if final_relu:
            feat_layers.append(nn.ReLU(inplace=True))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))
    return nn.Sequential(*feat_layers)


def residual_basic_block_pool(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                              pool=True):
    """Construct a network block based on the BasicBlock used in ResNet."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(BasicBlock(feature_dim, odim))
    if final_conv:
        feat_layers.append(nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
    if pool:
        feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))

    return nn.Sequential(*feat_layers)


def residual_bottleneck(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                        interp_cat=False, final_relu=False, final_pool=False, skfuse=False,
                        fusion_used_layers=['layer2','layer3','layer4'], in_channels=[512,1024,248],
                        fusion_strides=[1,1,1], final_stride=1):
    """Construct a network block based on the Bottleneck block used in ResNet."""
    # 我得把它重新定义为class
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    if interp_cat:
        feat_layers.append(InterpCat())
    for i in range(num_blocks):
        planes = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim // 4
        feat_layers.append(Bottleneck(4*feature_dim, planes))
    if skfuse:
        feat_layers.append(SKLayer(used_layers=fusion_used_layers,
                                   in_channels=in_channels,
                                   out_channels=out_dim,
                                   strides=fusion_strides,
                                   min_dim=32,))
        #feat_layers.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=final_stride, bias=False))
        feat_layers.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=final_stride, bias=False, groups=out_dim))
    if final_conv and not skfuse:# 这里面的stride=final_stride是我加的
        feat_layers.append(nn.Conv2d(4*feature_dim, out_dim, kernel_size=3, padding=1, stride=final_stride, bias=False))
        if final_relu:
            feat_layers.append(nn.ReLU(inplace=True))
        if final_pool:
            feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))
    return nn.Sequential(*feat_layers)

class ResidualBottleNeck(nn.Module):
    """Construct a network block based on the Bottleneck block used in ResNet."""
    def __init__(self, feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                            interp_cat=False, final_relu=False, final_pool=False, skfuse=False,
                            fusion_used_layers=['layer2','layer3','layer4'], in_channels=[512,1024,248],
                            fusion_strides=[1,1,1], final_stride=1):
        super(ResidualBottleNeck, self).__init__()
        if out_dim is None:
            out_dim = feature_dim
        if interp_cat:
            self.interp_cat_layer = InterpCat()
        for i in range(num_blocks):
            planes = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim // 4
            self.bottleneck_layer = Bottleneck(4*feature_dim, planes)
        if skfuse:
            self.skfuse_layer = SKLayer(used_layers=fusion_used_layers,
                                       in_channels=in_channels,
                                       out_channels=out_dim,
                                       strides=fusion_strides,
                                       min_dim=32,)
            self.downsample_layer = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=final_stride, bias=False, groups=out_dim)
        if final_conv and not skfuse:
            feat_layers = []
            feat_layers.append(nn.Conv2d(4*feature_dim, out_dim, kernel_size=3, padding=1, stride=final_stride, bias=False))
            if final_relu:
                feat_layers.append(nn.ReLU(inplace=True))
            if final_pool:
                feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.final_conv_layer = nn.Sequential(*feat_layers)
        if l2norm:
            self.l2norm_layer = InstanceL2Norm(scale=norm_scale)

    def forward(self, x):
        interp_cat_layer = getattr(self, 'interp_cat_layer', None)
        if interp_cat_layer:
            x_interp_cat = interp_cat_layer(x)
        else:
            x_interp_cat = x

        bottleneck_layer = getattr(self, 'bottleneck_layer', None)
        if bottleneck_layer:
            x_bottleneck = bottleneck_layer(x_interp_cat)
        else:
            x_bottleneck = x_interp_cat

        skfuse_layer = getattr(self, 'skfuse_layer', None)
        if skfuse_layer:
            x_skfuse = skfuse_layer(x_bottleneck)
        else:
            x_skfuse = x_bottleneck

        downsample_layer = getattr(self, 'downsample_layer', None)
        if downsample_layer:
            x_downsample = downsample_layer(x_skfuse)
        else:
            x_downsample = x_skfuse
        #暂时先不考虑不使用skfuse的情况
        l2norm_layer = getattr(self, 'l2norm_layer', None)
        if l2norm_layer:
            output = l2norm_layer(x_downsample)
        else:
            output = x_downsample
        return output, x_skfuse
