import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random
import numpy as np
from ltr.models.glse.detection.head import Point



class BoxRegression(nn.Module):
    """
    Using distance IoU loss
    """
    def __init__(self, stride, feat_size, loss_type='IoU'):
        super(BoxRegression, self).__init__()
        self.point = Point(stride, feat_size)#(x, y)
        self.iou_loss_func = IoULoss(loss_type)

    def forward(self, reg_predict, cls_predict, target_bb):
        cls_target, reg_target = self.generate_reg_cls_target(target_bb)
        #print(cls_target.shape)
        cls_target = cls_target.view(-1)
        select = torch.nonzero(cls_target==1).squeeze()

        reg_predict = reg_predict.permute(0,1,3,4,2).contiguous().view(-1,4)
        reg_target = reg_target.permute(0,2,3,1).contiguous().view(-1, 4)

        reg_predict = reg_predict[select]
        reg_target = reg_target[select]
        iou_loss = self.iou_loss_func(reg_predict, reg_target)

        if cls_predict is not None:
            cls_predict = cls_predict.permute(0,1,3,4,2).contiguous().view(-1,2)
            cls_predict = F.log_softmax(cls_predict, dim=1)
            cls_loss = self.cls_loss_func(cls_predict, cls_target.to(torch.long))
            return iou_loss, cls_loss
        else:
            return iou_loss, None

    def cls_loss_func(self, cls_predict, cls_target):
        pos = cls_target.data.eq(1).nonzero().squeeze().to(cls_predict.device)
        neg = cls_target.data.eq(0).nonzero().squeeze().to(cls_predict.device)

        loss_pos = self.get_cls_loss(cls_predict, cls_target, pos)
        loss_neg = self.get_cls_loss(cls_predict, cls_target, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def get_cls_loss(self, pred, label, select):
        if len(select.size()) == 0 or select.size() == torch.Size([0]):
            return 0
        # index_selec: 第一个参数是输入, 第二个参数是指定的维度, 第三个参数是选定的索引
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return F.nll_loss(pred, label)

    def generate_reg_cls_target(self, target_bb):
        """
        args:
            target_bb (Tensor) - Dims=(num_images, num_sequences, 4), [x1, y1, w, h]
        returns:
            cls (Tensor) - Dims=[num_images*num_sequences, feat_size, feat_size]
            delta (Tensor) - Dims=[num_images*num_sequences, 4, feat_size, feat_size]
        """
        num_images, num_sequences = target_bb.shape[:2]
        points = self.point.points.unsqueeze(1).to(target_bb.device)
        height, width = points.shape[-2:]

        pos = torch.ones(num_images*num_sequences, height, width, device=target_bb.device)
        neg = torch.zeros(num_images*num_sequences, height, width, device=target_bb.device)
        undefine = -1 * torch.ones(num_images*num_sequences, height, width, device=target_bb.device)

        delta = torch.ones(4, num_images*num_sequences, height, width, device=target_bb.device)
        target_bb = target_bb.view(-1, 4)

        target_bb_center = torch.cat([target_bb[:,:2]+(target_bb[:,2:]-1)/2, target_bb[:,2:]], dim=-1)
        tcx, tcy, tw, th = target_bb_center.unsqueeze(-1).unsqueeze(-1).permute(1,0,2,3)

        target_bb_corner = torch.cat([target_bb[:,:2], target_bb[:,:2]+target_bb[:,2:]-1], dim=-1)
        target_bb_corner = target_bb_corner.unsqueeze(-1).unsqueeze(-1).permute(1,0,2,3)

        delta[0,...] = points[0,...] - target_bb_corner[0,...]# l
        delta[1,...] = points[1,...] - target_bb_corner[1,...]# t
        delta[2,...] = target_bb_corner[2,...] - points[0,...]# r
        delta[3,...] = target_bb_corner[3,...] - points[1,...]# b

        cls1 = torch.where(torch.pow(tcx-points[0,...],2)/torch.pow(tw/4,2)+\
                torch.pow(tcy-points[1,...],2)/torch.pow(th/4,2)<1, pos, undefine).to(torch.int)

        cls2 = torch.where(torch.pow(tcx-points[0,...],2)/torch.pow(tw/2,2)+\
                torch.pow(tcy-points[1,...],2)/torch.pow(th/2,2)>1, neg, pos).to(torch.int)

        cls = cls1 * cls2
        num_pos = torch.sum((cls==1)).item()
        num_neg = torch.sum((cls==0)).item()
        ignored_neg_num = int(num_neg) - int(math.floor(num_pos * (3. + random.random())))

        def select(cls, ignored_neg_num, num_neg):
            neg_indices = torch.nonzero(cls==0).cpu().numpy()
            slt = np.arange(num_neg)
            np.random.shuffle(slt)
            selected_indices = neg_indices[slt[:ignored_neg_num]]
            selected_indices = list(zip(*selected_indices.tolist()))
            cls[selected_indices] = -1
            return cls

        if ignored_neg_num > 0:
            cls = select(cls, ignored_neg_num, num_neg)

        return cls, delta.permute(1,0,2,3)

    def visualize(self, images, target_bb, cls):
        a = Visualizer()
        batch = cls.shape[0]
        points = self.point.points.unsqueeze(0).expand(batch, -1, -1, -1).permute(0,2,3,1)
        cls = cls.unsqueeze(1).permute(0,2,3,1)
        print('cls', cls.shape)
        print(points.shape)

        for i, image in enumerate(images):
            a.visualize_normed_image_with_box_dot(image, target_bb[i], cls[i].view(-1), points[i].view(-1, 2),
                            output_size=(288, 288), path='/home/zikun/work/tracking/KPT/target', name='image{:0>3d}'.format(i))

class IoULoss(nn.Module):
    def __init__(self, loss_type):
        super(IoULoss, self).__init__()
        assert(loss_type in ['IoU', 'GIoU', 'DistanceIoU']), 'Unkown type for defining the IoU loss'
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        args:
            pred (Tensor) - Dims=[batch*height*width, 4]
            target (Tensor) - Dims=[batch*height*width, 4]
            weight (Tensor) - Dims=[batch*height*width]
        """

        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_w = target_left + target_right
        target_h = target_top + target_bottom
        pred_w = pred_left + pred_right
        pred_h = pred_top + pred_bottom

        target_area = target_w * target_h
        pred_area = pred_w * pred_h

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)
        w_enclose = torch.max(pred_left, target_left) + \
                  torch.max(pred_right, target_right)
        h_enclose = torch.max(pred_bottom, target_bottom) + \
                  torch.max(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        iou = area_intersect / (area_union + 1e-8)

        area_enclose = w_enclose * h_enclose

        pred_center_x = (pred_right - pred_left)/2
        pred_center_y = (pred_bottom - pred_top)/2

        target_center_x = (target_right - target_left)/2
        target_center_y = (target_bottom - target_top)/2

        euclidean_distance = torch.pow(pred_center_x - target_center_x, 2)+\
                            torch.pow(pred_center_y - target_center_y, 2)
        diagonal_length = torch.pow(w_enclose, 2) + torch.pow(h_enclose, 2)
        distance_penalty = euclidean_distance / (diagonal_length + 1e-8)

        v = 4/pow(math.pi, 2)*torch.pow((torch.atan(target_w/target_h) - \
                            torch.atan(pred_w/pred_h)), 2)
        alpha = v / (1 - iou + v + 1e-8)

        if self.loss_type == 'IoU':
            loss = 1 - iou
        elif self.loss_type == 'GIoU':
            loss = 1 - iou + (area_enclose - area_union)/area_enclose
        elif self.loss_type == 'DistanceIoU':
            loss = 1 - iou + distance_penalty + alpha * v
        return loss.mean()
