import os
from datetime import datetime
import time
import argparse
import json
import pickle
import logging
import numpy as np

import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
from logging_util import init_logger
from train4tune import main

import operations
import genotypes

hyper_space ={'model': 'f2gnn',
              'hidden_size': hp.choice('hidden_size', [16, 32, 64, 128, 256, 512]),
              'learning_rate': hp.uniform("lr", 0.001, 0.01),
              'weight_decay': hp.uniform("wr", 0.0001, 0.001),
              'optimizer': hp.choice('opt', ['adagrad', 'adam']),
              'dropout': hp.choice('dropout', [0, 1, 2, 3, 4, 5, 6]),
              'activation': hp.choice('act', ['relu', 'elu'])
              }




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--arch_filename', type=str, default='', help='given the location of searched res')
    parser.add_argument('--arch', type=str, default='', help='given the specific of searched res')
    parser.add_argument('--num_layers', type=int, default=4, help='num of GNN layers in SANE')
    parser.add_argument('--hyper_epoch', type=int, default=30, help='epoch in hyperopt.')
    parser.add_argument('--epochs', type=int, default=400, help='epoch in train GNNs.')
    parser.add_argument('--cos_lr', action='store_true', default=False, help='using lr decay in training GNNs.')
    parser.add_argument('--std_times', type=int, default=5, help=' the times in calculating the std')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--layer_norm', type=bool, default=False, help='use layer norm in trainging supernet.')
    parser.add_argument('--batch_norm', type=bool, default=False, help='use batch norm in trainging supernet.')
    parser.add_argument('--with_linear', type=bool, default=False, help='add extra linear in convs.')
    parser.add_argument('--min_lr', type=float, default=0.0, help='the minimal learning rate of lr decay')
    parser.add_argument('--rm_feature', action='store_true', help='rm the features in the autograph dataset')
    

    # parser.add_argument('--NA_PRIMITIVES', nargs='+', default=['SAGEConv','GCNConv'], help='aggregation_operation')
    parser.add_argument('--NA_PRIMITIVES', nargs='+', default=["GCNConv","GATConv","GINConv","GraphConv","SAGEConv","TransformerConv","ChebConv"], help='aggregation_operation')

    
    parser.add_argument('--FEATURE_ENGINEERING', nargs='+', default=['AddSelfLoops'], help='feature_engineering')
    # parser.add_argument('--ACTIVATION', nargs='+', default=['relu', 'elu'], help='activation function')
    parser.add_argument('--FF_PRIMITIVES', nargs='+', default=['sum','mean'], help='fusion_operation')
    parser.add_argument('--ACTIVATION', nargs='+', default=['relu', 'elu'], help='activation function')


    global args1
    args1 = parser.parse_args()


    genotypes.set_na_primitives(args1.NA_PRIMITIVES)
    operations.set_na_ops(args1.NA_PRIMITIVES)
    genotypes.set_ff_primitives(args1.FF_PRIMITIVES)
    operations.set_ff_ops(args1.FF_PRIMITIVES)
    set_activation(args1.ACTIVATION)



def set_activation(activation_function_list):
    hyper_space['activation'] = hp.choice('act', activation_function_list)



class ARGS(object):

    def __init__(self):
        super(ARGS, self).__init__()

def generate_args(arg_map):
    args = args1
    for k, v in arg_map.items():
        setattr(args, k, v)

    setattr(args, 'rnd_num', 1)

    args.dropout = args.dropout / 10.0
    # args.data = args1.data

    # args.epochs = args1.epochs
    # args.arch = args1.arch
    # args.gpu = args1.gpu
    # args.num_layers = args1.num_layers
    args.seed = 2
    args.grad_clip = 5
    args.momentum = 0.9
    return args

current_hyper_epoch = 0
def objective(args):
    global current_hyper_epoch
    current_hyper_epoch += 1
    print(args)
    # print('[Hyper Epoch: {}/{}]'.format(current_hyper_epoch, args.hyper_epoch))

    print('Current Hyperparameter:', args)
    args = generate_args(args)
    vali_acc, test_acc, args = main(args)
    return {
        'loss': -vali_acc,
        'test_acc': test_acc,
        'status': STATUS_OK,
        'eval_time': round(time.time(), 2),
    }






def run_fine_tune():

    tune_str = time.strftime('%Y%m%d-%H%M%S')
    path = 'logs/tune-%s_%s' % (args1.data, tune_str)
    if not os.path.exists(path):
        os.mkdir(path)
    log_filename = os.path.join(path, 'log.txt')
    logger = init_logger('fine-tune', log_filename, logging.INFO, False)

    lines = open(args1.arch_filename, 'r').readlines()

    suffix = args1.arch_filename.split('_')[-1][:-4]

    test_res = []
    arch_set = set()

    def process_hyper_space(i):
        if 'sage' in i:
            if args1.data in ['PubMed', 'Computer', 'flickr', 'DBLP']:
                hyper_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64, 128, 256])
            if args1.data == 'flickr':
                hyper_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64, 128, 256])
                hyper_space['learning_rate'] = hp.uniform("lr", 0.001, 0.01)
        else:

            if args1.data in ['PubMed', 'Computer', 'flickr', 'DBLP']:
                hyper_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64, 128])
            if args1.data == 'flickr':
                hyper_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64, 128])
                hyper_space['learning_rate'] = hp.uniform("lr", 0.001, 0.01)

            if args1.data == 'physics':
                hyper_space['hidden_size'] = hp.choice('hidden_size', [32, 64, 128])
            if args1.num_layers > 4:
                hyper_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64, 128, 256])
            if args1.num_layers > 8:
                hyper_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64, 128])

        if args1.num_layers in [6, 8] and args1.data in ['physics', 'Computer', 'DBLP', 'PubMed']:
            hyper_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64, 128, 256])
        elif args1.num_layers==10 and args1.data in ['physics', 'Computer', 'DBLP', 'PubMed']:
            hyper_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64, 128])
        if args1.data in ['texas', 'squirrel']:
            hyper_space['learning_rate'] = hp.uniform("lr", 0.01, 0.05)


    for ind, l in enumerate(lines):
        try:
            # print('**********process {}-th/{}, logfilename={}**************'.format(ind+1, len(lines), log_filename))
            print('*************** [process {}-th/{}] ***************'.format(ind+1, len(lines)))

            logging.info('**********process {}-th/{}**************'.format(ind+1, len(lines)))
            res = {}
            #iterate each searched architecture
            parts = l.strip().split(',')
            arch = parts[1].split('=')[1]



            args1.arch = arch
            if arch in arch_set:
                logging.info('the %s-th arch %s already searched....info=%s', ind+1, arch, l.strip())
                continue
            else:
                arch_set.add(arch)
            res['searched_info'] = l.strip()
            process_hyper_space(l)

            start = time.time()
            trials = Trials()
            #tune with validation acc, and report the test accuracy with the best validation acc
            best = fmin(objective, hyper_space, algo=partial(tpe.suggest, n_startup_jobs=int(args1.hyper_epoch/5)),
                        max_evals=args1.hyper_epoch, trials=trials, show_progressbar=False)

            space = hyperopt.space_eval(hyper_space, best)
            print('Best space is ', space)
            res['best_space'] = space
            args = generate_args(space)
            print('Best args from space is ', args.__dict__)
            res['tuned_args'] = args.__dict__

            test_accs = []
            vali_accs = []
            for i in range(args1.std_times):
                vali_acc, t_acc, test_args = main(args, run=i)
                # print('cal std: times:{}, valid_Acc:{}, test_acc:{}'.format(i, vali_acc, t_acc))
                print('Calculate standard: times:{}, valid_Acc:{}, test_acc:{}'.format(i, vali_acc, t_acc))

                test_accs.append(t_acc)
                vali_accs.append(vali_acc)
            test_accs = np.array(test_accs)
            vali_accs = np.array(vali_accs)
            print('Test results {} times:{:.04f}+-{:.04f}. Valid results {} times:{:.04f}+-{:.04f}'.format(args1.std_times, np.mean(test_accs), np.std(test_accs), args1.std_times, np.mean(vali_accs), np.std(vali_accs)))
            test_res.append(res)


            # test_res.append(res)


            #save final resutls
            txt_file_path = 'tuned_logs/%s_res_%s_%s_finalresults.txt' % (args1.data, tune_str, suffix)
            with open(txt_file_path, 'a') as fw:
                best_space_str = 'Best space is {}\n'.format(space)
                fw.write(best_space_str)

                best_args_str = 'Best args from space is {}\n'.format(args.__dict__)
                fw.write(best_args_str)

                test_results_str = 'Test results {} times: {:.04f}+-{:.04f}. Valid results {} times: {:.04f}+-{:.04f}\n'.format(args1.std_times, np.mean(test_accs), np.std(test_accs), args1.std_times, np.mean(vali_accs), np.std(vali_accs))
                fw.write(test_results_str)



            txt_file_path = 'tuned_logs/%s_res_%s_%s.txt' % (args1.data, tune_str, suffix)
            with open(txt_file_path, 'w') as fw:
                fw.write(str(test_res))
            
            txt_file_path = 'tuned_logs/tune_results_path.txt'
            tune_results_path = 'Finsh tunining {} archs, saved in {}'.format(len(arch_set),' tuned_logs/%s_res_%s_%s.txt' % (args1.data, tune_str, suffix))
            with open(txt_file_path, 'w') as fw:
                fw.write(str(tune_results_path))

            output_dir = 'tuned_res'
            os.makedirs(output_dir, exist_ok=True)  # This will create the directory if it doesn't exist

            with open('tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix), 'wb+') as fw:
                pickle.dump(test_res, fw)
            logging.info('**********finish {}-th/{}***************'.format(ind+1, len(lines)))
        except Exception as e:
            logging.info('errror occured for %s-th, arch_info=%s, error=%s', ind+1, l.strip(), e)
            import traceback
            traceback.print_exc()
    print('Finsh tunining {} archs, saved in {}'.format(len(arch_set), 'tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix)))


if __name__ == '__main__':
    get_args()
    if args1.arch_filename:
        run_fine_tune()


