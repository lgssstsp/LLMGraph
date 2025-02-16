import os
from datetime import datetime
import time
import argparse
import json
import pickle
import logging
import numpy as np
import torch
import random
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
from logging_util import init_logger
from train4tune import main
from genotypes import SC_PRIMITIVES, FF_PRIMITIVES, NA_PRIMITIVES, READOUT_PRIMITIVES
import operations
import genotypes


from setproctitle import setproctitle
setproctitle("LRGNN")


hyper_space ={'model': '',
              'hidden_size': hp.choice('hidden_size', [32, 64, 128]),
              'learning_rate': hp.uniform("lr", 0.001, 0.01),
              'weight_decay': hp.uniform("wr", 0.0001, 0.001),
              'optimizer': hp.choice('opt', ['adagrad', 'adam']),
              'dropout': hp.choice('dropout', [0, 1, 2, 3, 4, 5, 6])
              }



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--arch_filename', type=str, default='', help='given the location of searched res')
    parser.add_argument('--arch', type=str, default='', help='given the specific of searched res')

    #arch params
    parser.add_argument('--num_blocks', type=int, default=8, help='num of blocks')
    # parser.add_argument('--num_blocks', type=int, default=4, help='num of blocks')

    parser.add_argument('--num_cells', type=int, default=1, help='num of cells')
    parser.add_argument('--cell_mode', type=str, default='full', choices=['repeat', 'diverse', 'full'])

    #used for RB search
    parser.add_argument('--agg', type=str, default='gcn')
    parser.add_argument('--search_agg', type=bool, default=False, help='search aggregators')
    parser.add_argument('--model_type', type=str, default='NONE', help='how to update alpha', choices=['random', 'bayes', 'rank'])
    parser.add_argument('--hidden_size',  type=int, default=64, help='default hidden_size in supernet')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='init L2')

    #ft params
    parser.add_argument('--ft_mode', type=str, default='10fold', choices=['811', '622', '10fold'], help='data split function.')
    parser.add_argument('--BN', action='store_true', default=True, help='Batch norm')
    # parser.add_argument('--BN', action='store_true', default=False, help='Batch norm')    
    parser.add_argument('--LN', action='store_true', default=False, help='Layer norm')
    parser.add_argument('--rml2', action='store_true', default=False, help='rm L2 weight decay in the optimization')
    parser.add_argument('--rmdropout', action='store_true', default=False, help='rm dropout in GNN.')
    # parser.add_argument('--withcelljk', action='store_true', default=False, help='use jk among cells.')
    # parser.add_argument('--celljkmode', type=str, default='cat')

    # parser.add_argument('--hyper_epoch', type=int, default=2, help='epoch in hyperopt.')
    parser.add_argument('--hyper_epoch', type=int, default=30, help='epoch in hyperopt.')
    # parser.add_argument('--epochs', type=int, default=3, help='epoch in train GNNs.')
    parser.add_argument('--epochs', type=int, default=100, help='epoch in train GNNs.')
    parser.add_argument('--cos_lr', action='store_true', default=False, help='using lr decay in training GNNs.')
    # parser.add_argument('--cos_lr', action='store_true', default=True, help='using lr decay in training GNNs.')
    
    parser.add_argument('--lr_min', type=float, default=0.0, help='the minimal learning rate of lr decay')

    parser.add_argument('--std_times', type=int, default=5, help=' the times in calculating the std')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--tune_id', type=int, default=-1, help='tund i-th archs in the searched arch set.')

    parser.add_argument('--data_fold', type=int, default=10, help='x_fold cross-validation')
    
    
    parser.add_argument('--NA_PRIMITIVES', nargs='+', default=['SAGEConv','GCNConv'], help='aggregation_operation')
    # parser.add_argument('--FEATURE_ENGINEERING', nargs='+', default=['AddSelfLoops'], help='feature_engineering')
    parser.add_argument('--FEATURE_ENGINEERING', nargs='+', default=[], help='feature_engineering')
    
    parser.add_argument('--FF_PRIMITIVES', nargs='+', default=['sum','mean'], help='fusion_operation')
    parser.add_argument('--READOUT_PRIMITIVES', nargs='+', default=['global_mean','global_sum',], help='readout_operation')
    parser.add_argument('--ACTIVATION', nargs='+', default=['relu', 'elu'], help='activation function')



    global args1
    args1 = parser.parse_args()
    genotypes.set_na_primitives(args1.NA_PRIMITIVES)
    operations.set_na_ops(args1.NA_PRIMITIVES)
    genotypes.set_ff_primitives(args1.FF_PRIMITIVES)
    operations.set_ff_ops(args1.FF_PRIMITIVES)
    genotypes.set_readout_primitives(args1.READOUT_PRIMITIVES)
    operations.set_readout_ops(args1.READOUT_PRIMITIVES)
    set_activation(args1.ACTIVATION)

def set_activation(activation_function_list):
    hyper_space['activation'] = hp.choice('act', activation_function_list)
    # print("hyper_space:", hyper_space)
    # print("activation_function_list:", activation_function_list)


class ARGS(object):

    def __init__(self):
        super(ARGS, self).__init__()

def generate_args(arg_map):
    args = args1
    for k, v in arg_map.items():
        setattr(args, k, v)

    setattr(args, 'rnd_num', 1)

    args.dropout = args.dropout / 10.0
    # args.seed = args1.seed
    args.grad_clip = 5
    args.momentum = 0.9
    return args

def objective(args):
    print('current_hyper:', args)
    args = generate_args(args)
    vali_acc, test_acc, test_std, args = main(args)
    return {
        'loss': -vali_acc,
        'test_acc': test_acc,
        'test_std': test_std,
        'status': STATUS_OK,
        'eval_time': round(time.time(), 2),
    }

# tmp_name = 'abcdefghijklmno'
# autograph_dataset = []
# for i in range(len(tmp_name)):
#     autograph_dataset.append(tmp_name[i])
# print(autograph_dataset)

def run_fine_tune():
    seed = np.random.randint(0, 10000)
    args1.seed = seed
    random.seed(args1.seed)
    torch.cuda.manual_seed_all(args1.seed)
    torch.manual_seed(args1.seed)
    np.random.seed(args1.seed)
    os.environ.setdefault("HYPEROPT_FMIN_SEED", str(args1.seed))

    tune_str = time.strftime('%Y%m%d-%H%M%S')
    path = 'logs/tune-%s_%s' % (args1.data, tune_str)
    if not os.path.exists(path):
        os.mkdir(path)
    log_filename = os.path.join(path, 'log.txt')
    init_logger('fine-tune', log_filename, logging.INFO, False)

    lines = open(args1.arch_filename, 'r').readlines()

    # suffix = args1.arch_filename.split('_')[-1][:-4]

    test_res = []
    arch_set = set()

    if 'ogb' in args1.data:
        hyper_space['learning_rate'] = hp.uniform("lr", 0.005, 0.01)
        hyper_space['dropout'] = hp.choice('dropout', [2, 3, 4, 5])
        hyper_space['hidden_size'] = hp.choice('hidden_size', [64, 128, 256, 300, 512])
        hyper_space['weight_decay'] = hp.uniform("wr", 0.0005, 0.005)

        # hyper_space['learning_rate'] = hp.choice('lr', [0.007,0.010])
        # hyper_space['dropout'] = hp.choice('dropout', [2, 3, 4, 5])
        # hyper_space['hidden_size'] = hp.choice('hidden_size', [300])
        # hyper_space['optimizer']: hp.choice('opt', ['adam'])
        # hyper_space['weight_decay'] = hp.choice('wr', [2, 3, 4, 5])

    if 'DD' in args1.data:
        hyper_space['hidden_size'] = hp.choice('hidden_size', [32, 64])

    if args1.data in ['COX2', ]:
        hyper_space['dropout'] = hp.choice('dropout', [0, 1, 2, 3])
        hyper_space['weight_decay'] = hp.uniform("wr", 0.0005, 0.001)

    if args1.data in ['NCI1', 'NCI109']:
        hyper_space['dropout'] = hp.choice('dropout', [3, 4, 5, 6])
        hyper_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64, 128, 256])
        hyper_space['learning_rate'] = hp.uniform("lr", 0.005, 0.015)
        if args1.num_blocks >16:
            hyper_space['learning_rate'] = hp.uniform("lr", 0.005, 0.015)
            hyper_space['hidden_size'] = hp.choice('hidden_size', [32, 64, 128, 256])



    if args1.rml2:
        hyper_space['weight_decay'] = hp.choice('wd', [0.0, ])
    if args1.rmdropout:
        hyper_space['dropout'] = hp.choice('dropout', [0, ])


    for ind, l in enumerate(lines):
        try:
            print('**********process {}-th/{}, logfilename={}**************'.format(ind+1, len(lines), log_filename))
            logging.info('**********process {}-th/{}**************'.format(ind+1, len(lines)))
            res = {}

            #iterate each searched architecture
            parts = l.strip().split(',')
            arch = parts[1].split('=')[1]
            args1.arch = arch

            #tune specific archs
            if args1.tune_id != -1 and args1.tune_id != ind:
                continue

            if arch in arch_set:
                logging.info('the %s-th arch %s already searched....info=%s', ind+1, arch, l.strip())
                continue
            else:
                arch_set.add(arch)
            res['searched_info'] = l.strip()

            start = time.time()
            trials = Trials()
            best = fmin(objective, hyper_space, algo=partial(tpe.suggest, n_startup_jobs=int(args1.hyper_epoch/5)),
                        max_evals=args1.hyper_epoch, trials=trials, show_progressbar=False)

            space = hyperopt.space_eval(hyper_space, best)
            args = generate_args(space)

            res['best_space'] = space
            print('best args from space is ', args.__dict__)
            res['tuned_args'] = args.__dict__

            test_accs = []
            test_stds = []
            valid_accs = []
            # valid_stds = []
            for i in range(args1.std_times):
                # args.epochs = 100
                valid_acc, t_acc, t_std, test_args = main(args)
                print('cal std: times:{}, valid_Acc:{}, test_acc:{:.04f}+-{:.04f}'.format(i, valid_acc, t_acc, t_std))
                test_accs.append(t_acc)
                test_stds.append(t_std)
                valid_accs.append(valid_acc)
                # valid_stds.append(valid_std)
            test_accs = np.array(test_accs)
            test_stds = np.array(test_stds)
            valid_accs = np.array(valid_accs)


            #re-evaluate the performance on the selected hypers
            for i in range(5):
                # print('Train from scratch {}/5: Test_acc:{:.04f}+-{:.04f}'.format(i, test_accs[i], test_stds[i]))
                print('Train from scratch {}/5: Test_acc:{:.04f}. Valid from scratch {}/5: Valid_acc:{:.04f}'.format(i, test_accs[i], i, valid_accs[i]))
                
            print('Test results 5 times:{:.04f}+-{:.04f}. Valid results 5 times:{:.04f}+-{:.04f}'.format(np.mean(test_accs), np.std(test_accs), np.mean(valid_accs), np.std(valid_accs)))
            # print('test_results_5_times:{:.04f}+-{:.04f}'.format(np.mean(test_accs), np.std(test_accs)))
            
            
            
            res['accs_train_from_scratch'] = test_accs
            res['stds_train_from_scratch'] = test_stds
            test_res.append(res)


            # test_res.append(res)
            suffix = args1.arch_filename.split('-')[-1][:-4]

            #save final resutls
            txt_file_path = 'tuned_logs/%s_res_%s_%s_finalresults.txt' % (args1.data, tune_str, suffix)
            with open(txt_file_path, 'a') as fw:
                for i in range(5):
                    re_evaluate_str = 'Train from scratch {}/5: Test_acc:{:.04f}. Valid from scratch {}/5: Valid_acc:{:.04f}\n'.format(i, test_accs[i], i, valid_accs[i])
                    fw.write(re_evaluate_str)
                    # print('Train from scratch {}/5: Test_acc:{:.04f}. Valid from scratch {}/5: Valid_acc:{:.04f}'.format(i, test_accs[i], i, valid_accs[i]))
                
                # print('Test results 5 times:{:.04f}+-{:.04f}. Valid results 5 times:{:.04f}+-{:.04f}'.format(np.mean(test_accs), np.std(test_accs), np.mean(valid_accs), np.std(valid_accs)))
                overall_str = 'Test results 5 times:{:.04f}+-{:.04f}. Valid results 5 times:{:.04f}+-{:.04f}\n'.format(np.mean(test_accs), np.std(test_accs), np.mean(valid_accs), np.std(valid_accs))
                fw.write(overall_str)


                pass
                # best_space_str = 'Best space is {}\n'.format(space)
                # fw.write(best_space_str)

                # best_args_str = 'Best args from space is {}\n'.format(args.__dict__)
                # fw.write(best_args_str)

                # test_results_str = 'Test results {} times: {:.04f}+-{:.04f}. Valid results {} times: {:.04f}+-{:.04f}\n'.format(args1.std_times, np.mean(test_accs), np.std(test_accs), args1.std_times, np.mean(vali_accs), np.std(vali_accs))
                # fw.write(test_results_str)
            


            txt_file_path = 'tuned_logs/%s_res_%s_%s.txt' % (args1.data, tune_str, suffix)
            with open(txt_file_path, 'w') as fw:
                fw.write(str(test_res))
            
            txt_file_path = 'tuned_logs/tune_results_path.txt'
            tune_results_path = 'Finsh tunining {} archs, saved in {}'.format(len(arch_set),' tuned_logs/%s_res_%s_%s.txt' % (args1.data, tune_str, suffix))
            with open(txt_file_path, 'w') as fw:
                fw.write(str(tune_results_path))

            with open('tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix), 'wb+') as fw:
                pickle.dump(test_res, fw)


            logging.info('**********finish {}-th/{}**************8'.format(ind+1, len(lines)))
        except Exception as e:
            logging.info('errror occured for %s-th, arch_info=%s, error=%s', ind+1, l.strip(), e)
            import traceback
            traceback.print_exc()
    # print('finsh tunining {} archs, saved in {}'.format(len(arch_set), 'tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str)))


if __name__ == '__main__':
    get_args()
    if args1.arch_filename:
        run_fine_tune()


