import yaml
import os
import argparse
import csv
import random
import copy
import numpy as np

from graphgym.utils.io import makedirs_rm_exist, string_to_python
from graphgym.utils.comp_budget import dict_match_baseline

random.seed(123)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        dest='config',
        help='the base configuration file used for edit',
        default=None,
        type=str
    )
    parser.add_argument(
        '--grid',
        dest='grid',
        help='configuration file for grid search',
        required=True,
        type=str
    )
    parser.add_argument(
        '--sample_alias',
        dest='sample_alias',
        help='configuration file for sample alias',
        default=None,
        required=False,
        type=str
    )
    parser.add_argument(
        '--sample_num',
        dest='sample_num',
        help='Number of random samples in the space',
        default=10,
        type=int
    )
    parser.add_argument(
        '--out_dir',
        dest='out_dir',
        help='output directory for generated config files',
        default='configs',
        type=str
    )
    parser.add_argument(
        '--config_budget',
        dest='config_budget',
        help='the base configuration file used for matching computation',
        default=None,
        type=str
    )
    return parser.parse_args()


def get_fname(string):
    if string is not None:
        return string.split('/')[-1].split('.')[0]
    else:
        return 'default'


def grid2list(grid):
    list_in = [[]]
    for grid_temp in grid:
        list_out = []
        for val in grid_temp:
            for list_temp in list_in:
                list_out.append(list_temp + [val])
        list_in = list_out
    return list_in


def lists_distance(l1, l2):
    assert len(l1) == len(l2)
    dist = 0
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            dist += 1
    return dist


def grid2list_sample(grid, sample=10):
    configs = []
    while len(configs) < sample:
        config = []
        for grid_temp in grid:
            config.append(random.choice(grid_temp))
        if config not in configs:
            configs.append(config)
    return configs


def load_config(fname):
    if fname is not None:
        with open(fname) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    else:
        return {}


def load_search_file(fname):
    '''
    return: list of list, each element in "outs" list is a list of (dim_name, dim_alias, search_space)
    '''
    with open(fname, 'r') as f:
        out_raw = csv.reader(f, delimiter=' ')
        outs = []
        out = []
        for row in out_raw:
            if '#' in row:
                continue
            elif len(row) > 0:
                assert len(row) == 3, \
                    'Exact 1 space between each grid argument file' \
                    'And no spaces within each argument is allowed'
                out.append(row)
            else:
                if len(out) > 0:
                    outs.append(out)
                out = []
        if len(out) > 0:
            outs.append(out)
    return outs


def load_alias_file(fname):
    with open(fname, 'r') as f:
        file = csv.reader(f, delimiter=' ')
        for line in file:
            break
    return line


def exclude_list_id(list, id):
    return [list[i] for i in range(len(list)) if i != id]


def gen_grid(args, config, config_budget={}):
    task_name = '{}_grid_{}'.format(get_fname(args.config),
                                    get_fname(args.grid))
    # print(task_name)

    fname_start = get_fname(args.config)
    out_dir = '{}/{}'.format(args.out_dir, task_name)

    # print(out_dir)
    makedirs_rm_exist(out_dir)
    config['out_dir'] = os.path.join(config['out_dir'], task_name)

    outs = load_search_file(args.grid)

    # print(outs)
    for i, out in enumerate(outs):
        vars_label = [row[0].split('.') for row in out]
        vars_alias = [row[1] for row in out]
        vars_value = grid2list([string_to_python(row[2]) for row in out])

        # print("vars_label",vars_label)
        # print("vars_alias",vars_label)
        # print("vars_value",vars_value)

        if i == 0:
            pass
            # print('Variable label: {}'.format(vars_label))
            # print('Variable alias: {}'.format(vars_alias))

        for vars in vars_value:
            # print("vars",vars)
            config_out = config.copy()
            fname_out = fname_start
            for id, var in enumerate(vars):
                if len(vars_label[id]) == 1:
                    config_out[vars_label[id][0]] = var
                elif len(vars_label[id]) == 2:
                    if vars_label[id][0] in config_out:  # if key1 exist
                        config_out[vars_label[id][0]][vars_label[id][1]] = var
                    else:
                        config_out[vars_label[id][0]] = {vars_label[id][1]: var}
                else:
                    raise ValueError('Only 2-level config files are supported')
                fname_out += '-{}={}'.format(vars_alias[id],
                                             str(var).strip("[]").strip("''"))
            if len(config_budget) > 0:
                config_out = dict_match_baseline(config_out, config_budget)
            with open('{}/{}.yaml'.format(out_dir, fname_out), "w") as f:
                yaml.dump(config_out, f, default_flow_style=False)
        print('{} configurations saved to: {}'.format(
            len(vars_value), out_dir))


def gen_grid_sample(args, config, config_budget={}, compare_alias_list=[]):
    task_name = '{}_grid_{}'.format(get_fname(args.config),
                                    get_fname(args.grid))
    
  

    fname_start = get_fname(args.config)
    out_dir = '{}/{}'.format(args.out_dir, task_name)
    makedirs_rm_exist(out_dir)
    config['out_dir'] = os.path.join(config['out_dir'], task_name)
    outs = load_search_file(args.grid)

    counts = []
    for out in outs:
        vars_grid = [string_to_python(row[2]) for row in out]
        count = 1
        for var in vars_grid:
            count *= len(var)
        counts.append(count)

    counts = np.array(counts)
    print('Total size of each chunk of experiment space:', counts)
    counts = counts / np.sum(counts)
    counts = np.round(counts * args.sample_num)
    counts[0] += args.sample_num - np.sum(counts)
    print('Total sample size of each chunk of experiment space:', counts)

    for i, out in enumerate(outs):
        
        vars_label = [row[0].split('.') for row in out]
        vars_alias = [row[1] for row in out]
        if i == 0:
            print('Variable label: {}'.format(vars_label))
            print('Variable alias: {}'.format(vars_alias))
        vars_grid = [string_to_python(row[2]) for row in out]


        for alias in compare_alias_list:
            alias_id = vars_alias.index(alias)

            print(alias_id)
            vars_grid_select = copy.deepcopy(vars_grid[alias_id])
            print(vars_grid_select)
            
            vars_grid[alias_id] = [vars_grid[alias_id][0]]
            print(vars_grid)
            vars_value = grid2list_sample(vars_grid, counts[i])
            print(vars_value)

            vars_value_new = []
            for vars in vars_value:
                print(vars)
                for grid in vars_grid_select:
                    
                    vars[alias_id] = grid
                    vars_value_new.append(copy.deepcopy(vars))
            vars_value = vars_value_new



            vars_grid[alias_id] = vars_grid_select
            for vars in vars_value:
                config_out = config.copy()
                fname_out = fname_start + '-sample={}'.format(
                    vars_alias[alias_id])
                for id, var in enumerate(vars):
                    if len(vars_label[id]) == 1:
                        config_out[vars_label[id][0]] = var
                    elif len(vars_label[id]) == 2:
                        if vars_label[id][0] in config_out:  # if key1 exist
                            config_out[vars_label[id][0]][
                                vars_label[id][1]] = var
                        else:
                            config_out[vars_label[id][0]] = {
                                vars_label[id][1]: var}
                    else:
                        raise ValueError(
                            'Only 2-level config files are supported')
                    # if vars_alias[id] in compare_alias_list:
                    fname_out += '-{}={}'.format(vars_alias[id],
                                                 str(var).strip("[]").strip(
                                                     "''"))
                if len(config_budget) > 0:
                    config_out = dict_match_baseline(
                        config_out, config_budget, verbose=False)
                with open('{}/{}.yaml'.format(out_dir, fname_out), "w") as f:
                    yaml.dump(config_out, f, default_flow_style=False)
            print('Chunk {}/{}: Perturbing design dimension {}, '
                  '{} configurations saved to: {}'.format(i + 1, len(outs),
                                                          alias,
                                                          len(vars_value),
                                                          out_dir))


args = parse_args()
config = load_config(args.config)


config_budget = load_config(args.config_budget)
if len(config_budget) > 0:
    # Match the dataset to match the Embedding matrix rows 
    config_budget['dataset']['name'] = config['dataset']['name']



if args.sample_alias is None:
    # print("gen_grid")
    gen_grid(args, config, config_budget)
else:
    # print("gen_grid_sample")
    alias_list = load_alias_file(args.sample_alias)
    gen_grid_sample(args, config, config_budget, alias_list)
