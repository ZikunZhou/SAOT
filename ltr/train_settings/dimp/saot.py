import torch.nn as nn
import torch.optim as optim
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import dimpnet
import ltr.models.loss as ltr_losses
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class OPT_KPT(object):
    def __init__(self, name=None):
        if name:
            self.name = name
        else:
            self.name = 'hyper-parameters settings for key point tracker'

def run(settings):
    settings.description = 'Default train settings for DiMP with ResNet50 as backbone.'
    settings.batch_size = 2
    settings.num_workers = 8
    settings.multi_gpu = True
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 6.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 22
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 5.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    settings.zoomin_scale = 0.5
    settings.reg_feature_size = 22

    settings.frozen_backbone_layers = ['conv1', 'bn1', 'layer1', 'layer2']

    kp_opt = OPT_KPT()
    kp_opt.state_estiamtion_layers = ['layer2', 'layer3']
    kp_opt.graph_size = (settings.reg_feature_size, settings.reg_feature_size)

    kp_opt.cls_settings = OPT_KPT(name='settings for the cls feature extractor')
    kp_opt.cls_settings.skfuse = True
    kp_opt.cls_settings.fusion_used_layers = ['layer2','layer3']
    kp_opt.cls_settings.sk_in_channels = [512, 1024]
    kp_opt.cls_settings.fusion_strides = [1,1]
    kp_opt.cls_settings.final_stride = 2

    kp_opt.seneck_settings = OPT_KPT(name='settings for the neck layer of state estimator')
    kp_opt.seneck_settings.neck_in_channels = 512
    kp_opt.seneck_settings.neck_out_channels = 256
    kp_opt.seneck_settings.pool_size_tmp = 8
    kp_opt.seneck_settings.state_estiamtion_stride = 8

    kp_opt.seintegrate_settings = OPT_KPT(name='settings for the integrator of state estimator')
    kp_opt.seintegrate_settings.detach_saliency = False
    # hyper-parameter of the saliency Evaluator
    kp_opt.seintegrate_settings.mainlobe_window_radius_priori = 2.5
    kp_opt.seintegrate_settings.mainlobe_window_radius_max = 5
    kp_opt.seintegrate_settings.min_k = 4
    kp_opt.seintegrate_settings.ml_width_pow = 0.1
    kp_opt.seintegrate_settings.use_std_dev = True
    kp_opt.seintegrate_settings.use_priori_mask = True
    kp_opt.seintegrate_settings.use_priori_mask_plus = True
    kp_opt.seintegrate_settings.priori_mask_sigma = 2.0

    kp_opt.seintegrate_settings.learn_supress_sigma_factor = True
    kp_opt.seintegrate_settings.supress_sigma_factor = 1.50
    kp_opt.seintegrate_settings.learn_saliency_scale_factor = True
    kp_opt.seintegrate_settings.saliency_scale_factor = 0.25
    kp_opt.seintegrate_settings.saliency_topk = 48
    kp_opt.seintegrate_settings.use_saliency = True

    kp_opt.seintegrate_settings.gcn_in_channels = 256 + 64
    kp_opt.seintegrate_settings.gcn_hidden_channels = 512
    kp_opt.seintegrate_settings.gcn_out_channels = 512
    kp_opt.seintegrate_settings.possible_graph_sizes = [22, 20, 18, 16, 14]
    kp_opt.seintegrate_settings.phi_in_channels = 256 + 64
    kp_opt.seintegrate_settings.phi_hidden_channels = 64
    kp_opt.seintegrate_settings.use_high_order_poly = True
    kp_opt.seintegrate_settings.order_num = 2
    kp_opt.seintegrate_settings.use_sa_adjust_edge = False
    kp_opt.seintegrate_settings.use_difference = True

    kp_opt.sedetector_settings = OPT_KPT(name='settings for the detector of state estimator')
    kp_opt.sedetector_settings.detector_hidden_channels = 256
    kp_opt.sedetector_settings.detector_out_channels = 256
    kp_opt.sedetector_settings.skc_layer_num = 2
    kp_opt.sedetector_settings.skc_mid_channels = [256, 128]
    kp_opt.sedetector_settings.skc_branches = 4
    kp_opt.sedetector_settings.use_cls = True
    kp_opt.sedetector_settings.detach_cls = True


    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    coco_train = MSCOCOSeq(settings.env.coco_dir)

    # Validation datasets
    got10k_val = Got10k(settings.env.got10k_dir, split='votval')

    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),# jitter in terms of brightness
                                    tfm.RandomHorizontalFlip(probability=0.5),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 8, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    data_processing_train = processing.DiMPProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      crop_type='inside_major',
                                                      max_scale_change=1.5,
                                                      zoomin_scale=settings.zoomin_scale,
                                                      mode='sequence',
                                                      proposal_params=proposal_params,
                                                      label_function_params=label_params,
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    data_processing_val = processing.DiMPProcessing(search_area_factor=settings.search_area_factor,
                                                    output_sz=settings.output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    crop_type='inside_major',
                                                    max_scale_change=1.5,
                                                    zoomin_scale=settings.zoomin_scale,
                                                    mode='sequence',
                                                    proposal_params=proposal_params,
                                                    label_function_params=label_params,
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)

    # Train sampler and loader
    dataset_train = sampler.DiMPSampler([lasot_train, got10k_train, trackingnet_train, coco_train], [1,1,1,1],
                                        samples_per_epoch=40000, max_gap=200, num_test_frames=3, num_train_frames=3,
                                        processing=data_processing_train)
    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    train_shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=train_shuffle, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    dataset_val = sampler.DiMPSampler([got10k_val], [1], samples_per_epoch=10000, max_gap=200,
                                      num_test_frames=3, num_train_frames=3,
                                      processing=data_processing_val)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           drop_last=True, epoch_interval=5, stack_dim=1, sampler=val_sampler)

    # Create network and actor
    net = dimpnet.dimpnet50(filter_size=settings.target_filter_sz, backbone_pretrained=True, optim_iter=5,
                            clf_feat_norm=True, clf_feat_blocks=0, final_conv=True, out_feature_dim=512,
                            optim_init_step=0.9, optim_init_reg=0.1,
                            init_gauss_sigma=output_sigma * settings.feature_sz, num_dist_bins=100,
                            bin_displacement=0.1, mask_init_factor=3.0, target_mask_act='sigmoid', score_act='relu',
                            frozen_backbone_layers=settings.frozen_backbone_layers, kp_opt=kp_opt)

    # Wrap the network for multi GPU training
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.devices = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.devices = torch.device("cuda:0")
    # if settings.multi_gpu:
    #     net = MultiGPU(net, dim=1)

    objective = {'test_clf': ltr_losses.LBHinge(threshold=settings.hinge_threshold),
                 'box': ltr_losses.BoxRegression(stride=kp_opt.seneck_settings.state_estiamtion_stride,
                                                 feat_size=settings.reg_feature_size)}

    loss_weight = {'box': 5, 'cls': 5, 'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400}

    actor = actors.DiMPActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer
    optimizer = optim.Adam([{'params': actor.net.module.classifier.filter_initializer.parameters(), 'lr': 5e-5},
                            {'params': actor.net.module.classifier.filter_optimizer.parameters(), 'lr': 5e-4},
                            {'params': actor.net.module.classifier.feature_extractor.parameters(), 'lr': 5e-5},
                            {'params': actor.net.module.state_estimator.parameters(), 'lr': 2e-4, 'weight_decay': 0.0001},
                            {'params': actor.net.module.feature_extractor.layer3.parameters(), 'lr': 2e-5}],
                           lr=2e-4*2)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(50, load_latest=True, fail_safe=True)
