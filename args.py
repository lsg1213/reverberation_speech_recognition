import argparse


def get_args(args: argparse.ArgumentParser = None):
    if args is None:
        args = argparse.ArgumentParser()
    args.add_argument('--name', type=str, default='')
    args.add_argument('--datapath', type=str, default='/root/bigdatasets')
    args.add_argument('--speechnum', type=int, default=2, choices=[2, 3])
    args.add_argument('--model', type=str, default='conformer', choices=['conformer'])
    args.add_argument('--task', type=str, default='clean')
    args.add_argument('--epoch', type=int, default=300)
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--max_patience', type=int, default=30)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--sr', type=int, default=16000, choices=[8000, 16000])
    args.add_argument('--win_size', type=int, default=25)
    args.add_argument('--win_stride', type=int, default=10)
    args.add_argument('--max_wave_length', type=int, default=559280)
    args.add_argument('--max_label_length', type=int, default=212)
    args.add_argument('--tensorboard_path', type=str, default='tensorboard_log')
    args.add_argument('--gpus', type=str, default='-1')
    args.add_argument('--resume', action='store_true')
    config = args.parse_args()
    return config
    