import argparse
from graphgym.utils.agg_runs import agg_batch


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Aggregate the evalaution results'
    )
    parser.add_argument(
        '--dir',
        dest='dir',
        help='Dir for batch of results',
        # required=True,
        type=str,

        default='/home/zhengxiaohan/Design-Space-for-GNN-based-CF-main/run/results/cf_grid_cf'
    )
    parser.add_argument(
        '--metric',
        dest='metric',
        help='metric to select best epoch',
        required=False,
        type=str,
        default='rmse'
    )
    return parser.parse_args()


args = parse_args()
agg_batch(args.dir, args.metric)
