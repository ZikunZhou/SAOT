import os
import sys
import argparse
import torch
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker

#torch.manual_seed(4321)
#torch.cuda.manual_seed_all(4321)

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                visdom_info=None, checkpoint_path=None, checkpoint_num=None, tracker_descrip='default'):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """

    visdom_info = {} if visdom_info is None else visdom_info

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, run_id, dataset_name=dataset_name,
                checkpoint_path=checkpoint_path, checkpoint_num=checkpoint_num, tracker_descrip=tracker_descrip)]

    run_dataset(dataset, trackers, debug, threads, visdom_info=visdom_info)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--run_id', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom')
    parser.add_argument('--checkpoint_path', type=str, default='dimp_debug', help='Path for checkpoint files.')
    parser.add_argument('--checkpoint_num', type=int, default=None, help='checkpoint number')
    parser.add_argument('--start', type=int, default=None, help='start checkpoint index')
    parser.add_argument('--end', type=int, default=None, help='end checkpoint index')
    parser.add_argument('--tracker_descrip', type=str, default='default', help='short description of the tracker')
    parser.add_argument('--run_times', type=int, default=1, help='times for testing')
    args = parser.parse_args()
    if args.run_times > 1:
        if args.run_id is None:
            start_run_id=0
        else:
            start_run_id=args.run_id
        for run_id in range(start_run_id, start_run_id+args.run_times+1):
            if args.checkpoint_num is not None:
                run_tracker(args.tracker_name, args.tracker_param, run_id, args.dataset, args.sequence, args.debug, args.threads,
                            {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}, args.checkpoint_path, args.checkpoint_num, args.tracker_descrip)
            elif args.start is not None and args.end is not None:
                for ckpt_num in range(args.start, args.end+1):
                    run_tracker(args.tracker_name, args.tracker_param, run_id, args.dataset, args.sequence, args.debug, args.threads,
                                {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}, args.checkpoint_path, ckpt_num, args.tracker_descrip)
            else:
                raise ValueError('Unknown checkpoint index.')

    else:
        if args.checkpoint_num is not None:
            run_tracker(args.tracker_name, args.tracker_param, args.run_id, args.dataset, args.sequence, args.debug, args.threads,
                        {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}, args.checkpoint_path, args.checkpoint_num, args.tracker_descrip)
        elif args.start is not None and args.end is not None:
            for ckpt_num in range(args.start, args.end+1):
                run_tracker(args.tracker_name, args.tracker_param, args.run_id, args.dataset, args.sequence, args.debug, args.threads,
                            {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}, args.checkpoint_path, ckpt_num, args.tracker_descrip)
        else:
            raise ValueError('Unknown checkpoint index.')


if __name__ == '__main__':
    main()
