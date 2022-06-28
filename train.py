import os
import argparse

import tensorboard

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--train_module', type=str, help='training script name')
    parser.add_argument('--train_name', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--mode', type=str, choices=["single", "multiple"], default="multiple", help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--master_port', type=int, default=12345)


    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.mode == "single":
        train_cmd = "python ltr/run_training.py --train_module %s --train_name %s" \
                    % (args.train_module, args.train_name)
    elif args.mode == "multiple":
        train_cmd = "python -m torch.distributed.launch --nproc_per_node %d --master_port %d ltr/run_training.py " \
                    "--train_module %s --train_name %s" \
                    % (args.nproc_per_node, args.master_port, args.train_module, args.train_name)
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    os.system(train_cmd)


if __name__ == "__main__":
    main()