import json
import os
import numpy as np

from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video
import pathlib
import os

class TrackingNetVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(TrackingNetVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)

    # def load_tracker(self, path, tracker_names=None):
    #     """
    #     Args:
    #         path(str): path to result
    #         tracker_name(list): name of tracker
    #     """
    #     if not tracker_names:
    #         tracker_names = [x.split('/')[-1] for x in glob(path)
    #                 if os.path.isdir(x)]
    #     if isinstance(tracker_names, str):
    #         tracker_names = [tracker_names]
    #     # self.pred_trajs = {}
    #     for name in tracker_names:
    #         traj_file = os.path.join(path, name, self.name+'.txt')
    #         if os.path.exists(traj_file):
    #             with open(traj_file, 'r') as f :
    #                 self.pred_trajs[name] = [list(map(float, x.strip().split(',')))
    #                         for x in f.readlines()]
    #             if len(self.pred_trajs[name]) != len(self.gt_traj):
    #                 print(name, len(self.pred_trajs[name]), len(self.gt_traj), self.name)
    #         else:

    #     self.tracker_names = list(self.pred_trajs.keys())

class TrackingNetDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "NFS30" or "NFS240"
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(TrackingNetDataset, self).__init__(name, dataset_root)

        basepath = os.path.join(dataset_root, 'TEST')
        frames_path = os.path.join(basepath, 'frames')
        dataset = os.listdir(frames_path)
        print('number of sequence:', len(dataset))
        self.videos = {}
        for video in dataset:
            init_rect_ = np.loadtxt(os.path.join(basepath,'anno/{}.txt'.format(video)), delimiter=',')
            init_rect_ = list(init_rect_)
            init_rect = [[None]]*10000
            init_rect[0] = init_rect_
            gt_rect = init_rect
            _, img_names = self.get_fileNames(os.path.join(frames_path, video))

            self.videos[video] = TrackingNetVideo(video,
                                          frames_path,
                                          video,
                                          init_rect,
                                          img_names,
                                          gt_rect,
                                          None)
            self.attr = {}
            self.attr['ALL'] = list(self.videos.keys())

    def get_fileNames(self, rootdir):
        fs = []
        fs_all = []
        for root, dirs, files in os.walk(rootdir, topdown=True):
            files.sort()
            files.sort(key = len)
            if files is not None:
                for name in files:
                    _, ending = os.path.splitext(name)
                    if ending == ".jpg":
                        _, root_ = os.path.split(root)
                        fs.append(os.path.join(root_, name))
                        fs_all.append(os.path.join(root, name))

        return fs_all, fs
