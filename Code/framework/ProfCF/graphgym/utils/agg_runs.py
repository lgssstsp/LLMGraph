import os
import json
import numpy as np
import shutil
import ast
import pandas as pd
import logging

from graphgym.config import cfg
from graphgym.utils.io import dict_list_to_json, dict_list_to_tb, \
     json_to_dict_list, makedirs_rm_exist, string_to_python, dict_to_json
from tensorboardX import SummaryWriter

import pdb




def is_seed(s):
    try:
        int(s)
        return True
    except:
        return False


def is_split(s):
    if s in ['train', 'val', 'test']:
        return True
    else:
        return False


def join_list(l1, l2):
    assert len(l1) == len(l2), \
        'Results with different seeds must have the save format'
    for i in range(len(l1)):
        l1[i] += l2[i]
    return l1


def agg_dict_list(dict_list):
    '''default agg: mean + std'''
    dict_agg = {'epoch': dict_list[0]['epoch']}
    for key in dict_list[0]:
        if key != 'epoch':
            value = np.array([dict[key] for dict in dict_list])
            dict_agg[key] = np.mean(value).round(cfg.round)
            dict_agg['{}_std'.format(key)] = np.std(value).round(cfg.round)
    return dict_agg

def name_to_dict(run):
    cols = run.split('-')[1:]
    keys, vals = [], []
    keys_mapping = {'d':'dataset', 'i':'init_dim', 'm':'msg', 'g':'gnn_layer', 'l':'layers_num', 's':'stage', 'in':'inter_func', 'a':'act', 'cn':'cpnt_num', 'ca':'cpnt_aggr', 'gd':'gnn_dropout', 'e': 'max_epoch', 'w':'weight_decay'}
    for col in cols:
        try:
            key, val = col.split('=')
            if key in keys_mapping.keys():
                key = keys_mapping[key]
            keys.append(key)
            # Mapping sample dim name
            if val in keys_mapping.keys():
                val = keys_mapping[val]
            vals.append(string_to_python(val))
        except:
            if vals[len(vals) - 1] == 'amazon' or vals[len(vals) - 1] == 'ml':
                vals[len(vals) - 1] = f'{vals[len(vals) - 1]}-{col}'
    return dict(zip(keys, vals))

def rm_keys(dict, keys):
    for key in keys:
        dict.pop(key, None)


# single experiments
def agg_runs(dir, metric_best='rmse'):
    results = {'train': None, 'val': None, 'test': None}
    results_best = {'train': None, 'val': None, 'test': None}
    
    for seed in os.listdir(dir):
        if is_seed(seed):
            dir_seed = os.path.join(dir, seed)

            split = 'val'
            if split in os.listdir(dir_seed):
                dir_split = os.path.join(dir_seed, split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                stats_list = json_to_dict_list(fname_stats)
                if metric_best == 'auto':
                    metric = 'auc' if 'auc' in stats_list[0] else 'accuracy'
                elif metric_best == 'hr_at_k':
                    metric = 'rmse'
                else:
                    metric = metric_best
                performance_np = np.array([stats[metric] for stats in stats_list])
                if metric_best in ['mae', 'mse', 'rmse']:
                    best_epoch = stats_list[performance_np.argmin()]['epoch']
                else:
                    best_epoch = stats_list[performance_np.argmax()]['epoch']
                print('best epoch of {} is {}'.format(split, best_epoch))

            for split in os.listdir(dir_seed):
                if is_split(split):
                    dir_split = os.path.join(dir_seed, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    stats_list = json_to_dict_list(fname_stats)
                    stats_best = [stats for stats in stats_list if stats['epoch'] == best_epoch][0]
                    print('{}:{}'.format(split, stats_best))
                    stats_list = [[stats] for stats in stats_list]
                    if results[split] is None:
                        results[split] = stats_list
                    else:
                        results[split] = join_list(results[split], stats_list)
                    if results_best[split] is None:
                        results_best[split] = [stats_best]
                    else:
                        results_best[split] += [stats_best]
    results = {k: v for k, v in results.items() if v is not None} # rm None
    results_best = {k: v for k, v in results_best.items() if v is not None} # rm None
    for key in results:
        for i in range(len(results[key])):
            results[key][i] = agg_dict_list(results[key][i])
    for key in results_best:
        results_best[key] = agg_dict_list(results_best[key])
    # save aggregated results
    for key, value in results.items():
        dir_out = os.path.join(dir, 'agg', key)
        makedirs_rm_exist(dir_out)
        fname = os.path.join(dir_out, 'stats.json')
        dict_list_to_json(value, fname)

        if cfg.tensorboard_agg:
            writer = SummaryWriter(dir_out)
            dict_list_to_tb(value, writer)
            writer.close()
    for key, value in results_best.items():
        dir_out = os.path.join(dir, 'agg', key)
        fname = os.path.join(dir_out, 'best.json')
        dict_to_json(value, fname)
    logging.info('Results aggregated across runs saved in {}'.format(
        os.path.join(dir, 'agg')))

# agg across grid search
def agg_batch(dir, metric_best='auto'):
    results = {'train': [], 'val': [], 'test': []}
    for run in os.listdir(dir):
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')

            if os.path.isdir(dir_run):
                for split in os.listdir(dir_run):
                    dir_split = os.path.join(dir_run, split) #dir_split = /./train  /./val  /./test
                    fname_stats = os.path.join(dir_split, 'best.json')
                    dict_stats = json_to_dict_list(fname_stats)[-1]  # get best val epoch
                    rm_keys(dict_stats, ['lr', 'lr_std', 'eta', 'eta_std','params_std'])
                    results[split].append({**dict_name, **dict_stats})

    dir_out = os.path.join(dir, 'agg')
    makedirs_rm_exist(dir_out)
    for key in results:
        if len(results[key])>0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(list(dict_name.keys()),
                                     ascending=[True]*len(dict_name))
            fname = os.path.join(dir_out, '{}_best.csv'.format(key))
            results[key].to_csv(fname, index=False)


    results = {'train': [], 'val': [], 'test': []}
    for run in os.listdir(dir):
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                for split in os.listdir(dir_run):
                    dir_split = os.path.join(dir_run, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    dict_stats = json_to_dict_list(fname_stats)[-1] # get last epoch
                    rm_keys(dict_stats, ['lr', 'lr_std', 'eta', 'eta_std','params_std'])
                    results[split].append({**dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key])>0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(list(dict_name.keys()),
                                     ascending=[True]*len(dict_name))
            fname = os.path.join(dir_out, '{}.csv'.format(key))
            results[key].to_csv(fname, index=False)


    results = {'train': [], 'val': [], 'test': []}
    for run in os.listdir(dir):
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                for split in os.listdir(dir_run):
                    dir_split = os.path.join(dir_run, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    dict_stats = json_to_dict_list(fname_stats) # get best epoch
                    if metric_best == 'auto':
                        metric = 'auc' if 'auc' in dict_stats[0] else 'accuracy'
                    else:
                        metric = metric_best
                    performance_np = np.array([stats[metric] for stats in dict_stats])
                    if metric_best in ['mae', 'mse', 'rmse']:
                        dict_stats = dict_stats[performance_np.argmin()]
                    else:
                        dict_stats = dict_stats[performance_np.argmax()]
                    rm_keys(dict_stats, ['lr', 'lr_std', 'eta', 'eta_std','params_std'])
                    results[split].append({**dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key])>0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(list(dict_name.keys()),
                                     ascending=[True]*len(dict_name))
            fname = os.path.join(dir_out, '{}_bestepoch.csv'.format(key))
            results[key].to_csv(fname, index=False)


    print('Results aggregated across models saved in {}'.format(dir_out))


# # ## test
# dir = '/home/wzy/lalala/AutoRec-530/run/results/condense_single_sports'
# dir = "/home/zhengxiaohan/Design-Space-for-GNN-based-CF-main/run/results/example2"
# agg_runs(dir)



