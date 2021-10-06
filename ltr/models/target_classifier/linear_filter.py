import torch
import torch.nn as nn
import ltr.models.layers.filter as filter_layer
import math
from collections import OrderedDict


class LinearFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]
        if isinstance(train_feat, OrderedDict):
            for key in train_feat.keys():
                if train_feat[key].dim() == 5:
                    train_feat[key] = train_feat[key].view(-1, *train_feat[key].shape[-3:])#channel, height, width
                if test_feat[key].dim() == 5:
                    test_feat[key] = test_feat[key].view(-1, *test_feat[key].shape[-3:])
        else:
            if train_feat.dim() == 5:
                train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
            if test_feat.dim() == 5:
                test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat, train_skfeat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat, test_skfeat = self.extract_classification_feat(test_feat, num_sequences)
        #print('added by zikun, train_feat', train_feat.shape)
        #print('added by zikun, test_feat', test_feat.shape)

        # Train filter, filter.shape=[10, 512, 4, 4]
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb, *args, **kwargs)
        downsample_param = self.feature_extractor.downsample_layer.weight.data.clone().detach()
        downsample_param = nn.functional.adaptive_avg_pool2d(downsample_param.permute(1,0,2,3), (1,1))
        miner_value = torch.ones_like(downsample_param) * 1e-8
        downsample_param = torch.where(downsample_param!=0, downsample_param, miner_value)
        channel_importance = nn.functional.adaptive_avg_pool2d(filter.clone().detach(), (1,1)) / downsample_param
        #print(channel_importance.view(-1))
        if torch.any(torch.isnan(channel_importance)):
            print('----------channel_importance has a nan value-----------')
            #raise Exception
            channel_importance = torch.ones_like(channel_importance).to(channel_importance.device)
        # Classify samples using all return filters
        test_scores = [self.classify(f, test_feat) for f in filter_iter]

        return test_scores, train_skfeat, test_skfeat, channel_importance

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output, skfuse_feat = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:]), skfuse_feat.reshape(-1, num_sequences, *skfuse_feat.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)

        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, bb=bb, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses

    def train_classifier(self, backbone_feat, bb):
        num_sequences = bb.shape[1]

        if backbone_feat.dim() == 5:
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])

        # Extract features
        train_feat, train_skfeat = self.extract_classification_feat(backbone_feat, num_sequences)

        # Get filters from each iteration
        final_filter, _, train_losses = self.get_filter(train_feat, bb)
        return final_filter, train_losses

    def track_frame(self, filter_weights, backbone_feat):
        if backbone_feat.dim() == 5:
            num_sequences = backbone_feat.shape[1]
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])
        else:
            num_sequences = None

        test_feat, test_skfeat = self.extract_classification_feat(backbone_feat, num_sequences)

        scores = filter_layer.apply_filter(test_feat, filter_weights)

        return scores
