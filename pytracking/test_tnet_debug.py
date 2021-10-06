# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys

import cv2
import torch
import numpy as np
import pdb

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from toolkit.utils.bbox import get_axis_aligned_bbox

from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

from pytracking.evaluation import Tracker

torch.manual_seed(4321)
torch.cuda.manual_seed_all(4321)

parser = argparse.ArgumentParser(description='siamrpn tracking')

parser.add_argument('tracker_name', type=str)
parser.add_argument('tracker_param', type=str)
parser.add_argument('--run_id', type=int, default=None)
parser.add_argument('--debug', type=int, default=0, help='Debug level.')
parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom')
parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom')
parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom')
parser.add_argument('--checkpoint_path', type=str, default='dimp_debug', help='Path for checkpoint files.')
parser.add_argument('--checkpoint_num', type=int, default=None, help='checkpoint number')
parser.add_argument('--dataset', type=str, help='datasets')
parser.add_argument('--video', default='', type=str, help='eval one special video')
parser.add_argument('--vis', action='store_true', help='whether visualzie result')
parser.add_argument('--tracker_descrip', type=str, default='default')

parser.add_argument('--start', default=1, type=int, help='the start index of the testing checkpoints')
parser.add_argument('--end', default=1, type=int, help='the end index of the testing checkpoints')
parser.add_argument('--run_times', default=1, type=int, help='the name of the tracker')

args = parser.parse_args()

torch.set_num_threads(1)


def main(checkpoint_num, run_id):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '/home/zikun/repository/data/siamrpn_test_data/testing_dataset', args.dataset)

    visdom_info = {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}

    config_tracker = Tracker(args.tracker_name, args.tracker_param, run_id=run_id, checkpoint_path=args.checkpoint_path, checkpoint_num=checkpoint_num, dataset_name=args.dataset)
    if 'VOT' in args.dataset or 'vot' in args.dataset:
        tracker = config_tracker.run_vot_pysot(args.debug, visdom_info)
    else:
        tracker = config_tracker.run_otb_pysot(args.debug, visdom_info)
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.tracker_descrip
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    # tracker.init(img, gt_bbox_)
                    # pdb.set_trace()
                    tracker.track_vot_initialize(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    bbox = np.array([cx-(w-1)/2, cy-(h-1)/2, w, h])

                    pred_bbox = tracker.track_vot_frame(img, bbox)
                    #pred_bbox = tracker.track_vot_frame(img, np.array(gt_bbox))
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if args.vis and idx >= frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
                # pdb.set_trace()
            toc /= cv2.getTickFrequency()
            # save results
            if checkpoint_num is None:
                video_path = '{}/{}/{}/{}/{:s}'.format('tracking_results', args.tracker_name, args.tracker_param, args.dataset, args.tracker_descrip)
            else:
                video_path = '{}/{}/{}/{}/{:s}_{:04d}'.format('tracking_results', args.tracker_name, args.tracker_param, args.dataset, args.tracker_descrip, checkpoint_num)
            if run_id is None:
                pass
            else:
                video_path = '{}_{:03d}'.format(video_path, run_id)

            video_path = os.path.join(video_path,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            # pdb.set_trace()
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        worse_sequences = {
                    'CyIhI7Vbzr0_0': 71,
                    'DP9z5qDrrlY_0': 156,
                    'hXTRhpolmkQ_0': 196,
                    'ThudiuJW5Kg_0': 376,# 很有必要可视化一下 移动范围比较大, 调到目标上的例子
                    'TdkeFfjNUpg_0': 165,# 看下psr
                    '0-6LB4FqxoE_0': 1,#138,
                    '8XhNvHbY4e0_0': 182,
                    'EILr0LNw0Mg_0': 722,
                    #'2FVEzOxvjj8_0': #这个都可以看看
                    '5-t2w-R1AHg_0': 81,
                    'Vgn-TZkzDV0_0': 67,# 这个可以都看一下
                    'nHpL3SlCABM_0': 270,
                    'aKg7xivrI9Y_0': 170,
                    '5RJXgYSJaVE_0': 185,#198,# 这个就属于移动范围比较大, 但是找到目标的例子
                    'iCmWVMcSnh4_1': 152,
                    'YaXKZdSEwt4_0': 696,# 这个应该是累积效应的原因
                    'OxQbu0kwObo_0': 145,
                    'gcoZEf8SRvs_0': 14,
                    'FpDI3f_tYj4_0': 55,
                    'bl-jwa1jRTE_0': 79,
                    '0GER2Qd0vFw_0': 22,
                    'IxjBypJ83pA_0': 44,
                    'Gw5FxWOjKUE_0': 27,# 这个都不行
                    'O1QjMCPJn5A_0': 89}
        better_sequences ={
                    'IUhkjSSb9a8_0': 6,
		            '4rT02vTH8qg_0': 152,
		            'HoWrvbRF5Uw_0': 97,# 98 99
		            'kmWAD0fCAUc_0': 118,
	                'm6lV1lfv7GE_0': 293,
                    'LcKCWQgxPv4_0': 253,
	                'OpuH-1YGcY8_0': 329,
                    '8VkHx1GXvmo_0': 121,
                    'Zljto-7mKTI_0': 137,
    		        'UUyk7Eojl1I_0': 197,
		            'xA2ZIXWuqxY_0': 364,
                    'XX1eVms9ZcE_0': 125,
                    'wRDRtaxcsQg_0': 195,
                    'za7pL4OB-_o_0': 243,
                    'tv2_ONbSPis_0': 152,
                    'OU72LG0O9_M_0': 300
            }
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            if video.name not in worse_sequences:
                continue
            if checkpoint_num is None:
                model_path = '{}/{}/{}/{}/{:s}'.format('tracking_results', args.tracker_name, args.tracker_param, args.dataset, args.tracker_descrip)
            else:
                model_path = '{}/{}/{}/{}/{:s}_{:04d}'.format('tracking_results', args.tracker_name, args.tracker_param, args.dataset, args.tracker_descrip, checkpoint_num)
            if run_id is None:
                pass
            else:
                model_path = '{}_{:03d}'.format(model_path, run_id)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            if os.path.exists(result_path):
                print(result_path, 'exists!')
                continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, float(w), float(h)]
                    tracker.track_pysot_initialize(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    if idx == worse_sequences[video.name]:
                        visualize_flag=True
                    else:
                        visualize_flag=False
                    outputs = tracker.track_pysot_frame(img, visualize_flag=visualize_flag, video_name=video.name)
                    pred_bbox = outputs['target_bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    if not None in gt_bbox:
                        gt_bbox = list(map(int, gt_bbox))
                        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    if args.run_times > 1:
        if args.run_id is None:
            start_run_id=0
        else:
            start_run_id=args.run_id
        for run_id in range(start_run_id, start_run_id+args.run_times+1):
            if args.checkpoint_num is not None:
                main(args.checkpoint_num, run_id)
            elif args.start is not None and args.end is not None:
                for ckpt_num in range(args.start, args.end+1):
                    main(ckpt_num, run_id)
            else:
                raise ValueError('Unknown checkpoint index.')
    else:
        if args.checkpoint_num is not None:
            main(args.checkpoint_num, args.run_id)
        elif args.start is not None and args.end is not None:
            for ckpt_num in range(args.start, args.end+1):
                main(ckpt_num, args.run_id)
        else:
            raise ValueError('Unknown checkpoint index.')
