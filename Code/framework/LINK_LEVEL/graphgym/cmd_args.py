import argparse
import sys


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train a classification model'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file path',
        # required=True,
        type=str,
        default='configs/example2.yaml'
    )
    parser.add_argument(
        '--repeat',
        dest='repeat',
        help='Repeat how many random seeds',
        default=1,
        type=int
    )
    parser.add_argument(
        '--mark_done',
        dest='mark_done',
        action='store_true',
        help='mark yaml as yaml_done after a job has finished',
    )
    parser.add_argument(
        'opts',
        help='See graphgym/config.py for all options',
        default=('device', 'cuda:0'),
        nargs=argparse.REMAINDER
    )
    parser.add_argument(
        '--exp_num',
        dest='exp_num',
        help='No. of current experiment',
        default=0,
        type=int
    )
    parser.add_argument(
        '--gpu_strategy',
        dest='gpu_strategy',
        help='GPU selection strategy',
        default='random',
        type=str,
        choices=['random', 'greedy']
    )


    parser.add_argument(
        '--FEATURE_ENGINEERING', 
        nargs='+', 
        default=[], 
        help='feature_engineering'
        )
    
    parser.add_argument(
        '--epoch',
        dest='epoch',
        help='epoch',
        default=200,
        type=int
    )
    
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='learning_rate',
        default=0.01,
        type=float
    )
    


    return parser.parse_args()
