import os
import sys
import argparse
import cv2
import numpy as np
import torch
import glob
from os.path import join, realpath, dirname

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker as KPTracker

from got10k.trackers import Tracker
from got10k.experiments import ExperimentTColor128, ExperimentGOT10k

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--checkpoint_path', type=str, default='dimp_debug', help='Path for checkpoint files.')
parser.add_argument('--start', type=int, default=None, help='start checkpoint index')
parser.add_argument('--end', type=int, default=None, help='end checkpoint index')
parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
parser.add_argument('--tracker_descrip', type=str, default='default', help='short description of the tracker')
parser.add_argument('--run_times', type=int, default=1, help='times for testing')
parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom')
parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom')
parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom')
args = parser.parse_args()


class MyTracker(Tracker):
    def __init__(self, tracker_name, tracker_param, checkpoint_path, checkpoint_num, run_id, tracker_descrip):
        super(MyTracker, self).__init__(name='{:s}_ep{:0>4d}_{:0>2d}'.format(tracker_descrip, checkpoint_num, run_id))

        visdom_info = {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}

        tracker_generator = KPTracker(tracker_name, tracker_param,
                                         run_id=run_id,
                                         checkpoint_path=checkpoint_path,
                                         checkpoint_num=checkpoint_num,
                                         tracker_descrip=tracker_descrip)
        params = tracker_generator.get_parameters()
        self.tracker = tracker_generator.tracker_class(params)

    def init(self, image, box):
        #atom tracker input bbox coordinates definiation: [x1, y1, w, h]
        gt_bbox = {'init_bbox': box}
        image = np.array(image)
        self.tracker.initialize(image, gt_bbox)

    def update(self, image):
        image = np.array(image)
        outputs = self.tracker.track(image)
        pred_bbox = outputs['target_bbox']
        return pred_bbox

if __name__ == '__main__':
    # setup tracker
    args = parser.parse_args()
    assert (args.start is not None and args.end is not None), 'Please set the start and end checkpoints!'
    for i in range(args.run_times):
        for ckpt_num in range(args.start, args.end+1):
            tracker = MyTracker(args.tracker_name,
                                args.tracker_param,
                                args.checkpoint_path,
                                ckpt_num,
                                i,
                                args.tracker_descrip)

            # run experiments on GOT-10k (validation subset)
            experiment = ExperimentGOT10k('/home/zikun/repository/data/raw_data/got10k', subset='test')
            experiment.run(tracker, visualize=False)

            # report performance
            #experiment.report([tracker.name])
