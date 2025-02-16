import os.path as osp
import sys
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, PPI, Reddit, Coauthor, CoraFull, gnn_benchmark_dataset, Flickr, CitationFull, Amazon, Actor, CoraFull
from torch_geometric.data import NeighborSampler
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops, is_undirected, to_undirected
import random
import time
import numpy as np
import torch
from torch_sparse import SparseTensor, coalesce

from torch_geometric.data import Data
path = './data/'

import os
from sklearn.model_selection import train_test_split


def load_fixed_splits(dataset, sub_dataset):
    name = dataset
    if sub_dataset:
        name += f'-{sub_dataset}'

    splits_lst = np.load('./splits/{}-splits.npy'.format(name), allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst




def split_622_new(data):
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    train_indices, temp_data = train_test_split(indices, test_size=0.4, random_state=42)
    val_indices, test_indices = train_test_split(temp_data, test_size=0.5, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_indices = torch.tensor(train_indices, device=device)
    val_indices = torch.tensor(val_indices, device=device)
    test_indices = torch.tensor(test_indices, device=device)

    data.train_mask = index_to_mask(train_indices, data.num_nodes)
    data.val_mask = index_to_mask(val_indices, data.num_nodes)
    data.test_mask = index_to_mask(test_indices, data.num_nodes)

    return data


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def split_622(data):
    split = gen_uniform_60_20_20_split(data)
    data.train_mask = index_to_mask(split[0], data.num_nodes)
    data.val_mask = index_to_mask(split[1], data.num_nodes)
    data.test_mask = index_to_mask(split[2], data.num_nodes)
    return data

def gen_uniform_60_20_20_split(data):
    skf = StratifiedKFold(5, shuffle=True, random_state=12345)
    idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
    return torch.cat(idx[:3], 0), torch.cat(idx[3:4], 0), torch.cat(idx[4:], 0)



def get_dataset(name, feature_engineerings ,split=True, run=0):
    

    transforms_list = []
    for fe_name in feature_engineerings:
        transform_class = getattr(T, fe_name, None)
        if transform_class:
            # print("feature init:", fe_name)
            if fe_name == 'SIGN':
                transforms_list.append(transform_class(K=2))
            elif fe_name == 'AddRandomMetaPaths':
                pass
            elif fe_name == 'GDC':
                pass
            elif fe_name == 'AddRandomWalkPE':
                transforms_list.append(transform_class(walk_length=2))
            elif fe_name == 'OneHotDegree':
                # transforms_list.append(transform_class(max_degree=2))    
                pass      
            elif fe_name == 'LocalDegreeProfile':
                pass        
            else:
                transforms_list.append(transform_class())
        else:
            print(f"Warning: Transform {fe_name} not found.")
            sys.exit()

    transform = T.Compose(transforms_list)

    
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(path, name)
        dataset.transform = transform
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]

        if split:
            data = split_622(data)
        return data, num_features, num_classes

    ##########CSTAG dataset
    elif name in ['cstag-children', 'cstag-computers', 'cstag-photo', 'cstag-history', 'cstag-fitness']:
        # print("dataset name:", name)
        class CustomDataset:
            def __init__(self, data, transform=None):
                self.data = data
                self.transform = transform
                
                self.num_features = data.x.shape[1] if hasattr(data, 'x') else None
                
                self.num_classes = len(torch.unique(data.y)) if hasattr(data, 'y') else None

            def __getitem__(self, idx):
                return self.data

            def __len__(self):
                return 1  
        
        if name == 'cstag-children':
            cs_dataset_path = '../../../TAG-datasets/dataset/Children/Children.pt'
            data = torch.load(cs_dataset_path)
            dataset = CustomDataset(data)
            dataset.transform = transform
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            if split:
                data = split_622(data)
            return data, num_features, int(num_classes)
        elif name == 'cstag-photo':
            cs_dataset_path = '../../../TAG-datasets/dataset/Photo/Photo.pt'
            data = torch.load(cs_dataset_path)
            dataset = CustomDataset(data)
            dataset.transform = transform
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            if split:
                data = split_622(data)
            return data, num_features, int(num_classes)
        elif name == 'cstag-history':
            cs_dataset_path = '../../../TAG-datasets/dataset/History/History.pt'
            data = torch.load(cs_dataset_path)
            dataset = CustomDataset(data)
            dataset.transform = transform
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            if split:
                data = split_622(data)
                
            # print(data, num_features, int(num_classes))
            return data, num_features, int(num_classes)
        elif name == 'cstag-fitness':
            cs_dataset_path = '../../../TAG-datasets/dataset/Fitness/Fitness.pt'
            data = torch.load(cs_dataset_path)
            dataset = CustomDataset(data)
            dataset.transform = transform
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            if split:
                data = split_622(data)
            return data, num_features, int(num_classes)        
        elif name == 'cstag-computers':
            cs_dataset_path = '../../../TAG-datasets/dataset/Computers/Computers.pt'
            data = torch.load(cs_dataset_path)
            
            dataset = CustomDataset(data)
            # print(f"Number of features: {dataset.num_features}")
            # print(f"Number of classes: {dataset.num_classes}")
            # print(f"Dataset size (number of graphs): {len(dataset)}")
            
            dataset.transform = transform
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            
            if split:
                data = split_622(data)
            
            return data, num_features, int(num_classes)

    
    ##########


    elif name in ['genius', 'deezer-europe', 'yelp-chi', 'snap-patents', 'pokec', 'arxiv-year']:
        from new_benchmark_dataset import load_nc_dataset
        dataset = load_nc_dataset(name)
        x = dataset.graph['node_feat']
        edge_index = dataset.graph['edge_index']


        y = dataset.label.clone().detach()
        y = y.to(dtype=torch.long) 


        if not is_undirected(edge_index, num_nodes=x.size(0)):
            print('use undirected graph for dataset ',name)
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        if name == 'genius':
            y = y.to(torch.float)

        split_idx_lst = load_fixed_splits(name, None)
        split_idx = split_idx_lst[run]
        train_mask = index_to_mask(split_idx['train'], x.size()[0])
        val_mask = index_to_mask(split_idx['valid'], x.size()[0])
        test_mask = index_to_mask(split_idx['test'], x.size()[0])
        print(x.size(), edge_index.size(), y.max().item() + 1, train_mask.sum(), val_mask.sum(), test_mask.sum(),
              y.size(), y[0])
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)


        data = transform(data)
        num_classes = y.max().item() + 1
        num_features = dataset.graph['node_feat'].shape[1]


        
        return data, num_features, int(num_classes)
    
    elif name == 'actor':
        dataset = Actor(path + 'Actor/')
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        data = transform(data)

        # data.train_mask = data.train_mask[:, run]
        # data.val_mask = data.val_mask[:, run]
        # data.test_mask = data.test_mask[:, run]
        
        
        def actor_gen_uniform_60_20_20_split(data):
            skf = StratifiedKFold(5, shuffle=True, random_state=12345)
            idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
            return torch.cat(idx[:3], 0), torch.cat(idx[3:4], 0), torch.cat(idx[4:], 0)
        def actor_split_622(data):
            split = actor_gen_uniform_60_20_20_split(data)
            data.train_mask = index_to_mask(split[0], data.num_nodes)
            data.val_mask = index_to_mask(split[1], data.num_nodes)
            data.test_mask = index_to_mask(split[2], data.num_nodes)
            return data
        
        data = actor_split_622(data)

        return data, num_features, num_classes

    elif name in ['squirrel', 'texas', 'corafull', 'chameleon', 'wisconsin', 'cornell']:

        edge_file = path + name + '/out1_graph_edges.txt'
        feature_file = path + name + '/out1_node_feature_label.txt'
        mask_file = path + name + '/' + name + '_split_0.6_0.2_'+str(run) + '.npz'

        data = open(feature_file).readlines()[1:]
        x = []
        y = []
        for i in data:
            tmp = i.rstrip().split('\t')
            y.append(int(tmp[-1]))
            tmp_x = tmp[1].split(',')
            tmp_x = [int(fi) for fi in tmp_x]
            x.append(tmp_x)

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y)

        edges = open(edge_file)
        edges = edges.readlines()
        edge_index = []
        for i in edges[1:]:
            tmp = i.rstrip()
            tmp = tmp.split('\t')
            edge_index.append([int(tmp[0]), int(tmp[1])])
            edge_index.append([int(tmp[1]), int(tmp[0])])
        # edge_index = np.array(edge_index).transpose(1, 0)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
        print('edge_index:', edge_index.size())


        # mask
        mask = np.load(mask_file)
        train_mask = torch.from_numpy(mask['train_mask.npy']).to(torch.bool)
        val_mask = torch.from_numpy(mask['val_mask.npy']).to(torch.bool)
        test_mask = torch.from_numpy(mask['test_mask.npy']).to(torch.bool)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        data = transform(data)

        return data, x.shape[1], int(y.max().item()) + 1

    elif name in ['CS', 'physics']:
        if name == 'CS':
            dataset = Coauthor(path + 'CoauthorCS/', 'CS')
        else:
            # dataset = Coauthor(path + '/' + path + 'CoauthorPhysics/', 'physics')
            dataset = Coauthor(path + 'CoauthorPhysics/', 'physics')
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        data = transform(data)

        if split:
            data = split_622(data)
        return data, num_features, num_classes

    elif name == 'DBLP':
        dataset = CitationFull(path + 'DBLP', 'dblp')


        dataset.transform = transform


        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = split_622(dataset[0])
        return data, num_features, num_classes

    elif name == 'flickr':
        dataset = Flickr(path + 'flickr')
        dataset.transform = transform

        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        if split:
            data = split_622(data)
        return data, num_features, num_classes

    elif name in ['Photo', 'Computer']:
        if name == 'Computer':
            dataset = Amazon(path + 'AmazonComputers', 'Computers')

            dataset.transform = transform


        elif name == 'Photo':
            dataset = Amazon(path + 'AmazonPhoto', 'Photo')

            dataset.transform = transform

        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        data = split_622(data)
        return data, num_features, num_classes
    
    
    if name == 'userDataset':
        edge_index = []
        num_nodes = 0
        with open(path + 'userDataset/DS_A.txt', 'r') as f:
            next(f)
            for line in f:
                source, target = map(int, line.strip().split())
                edge_index.append([source, target])
                num_nodes = max(num_nodes, source, target) + 1


        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

        x = []
        y = []
        with open(path + 'userDataset/DS_node.txt', 'r') as f:
            next(f)
            for line in f:
                node_id, label, features = line.strip().split(' ', 2)
                features = [float(i) for i in features.strip('[]').split(',') if i]
                x.append(features)
                y.append(int(label))

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data = transform(data)



        if split:
            data = split_622(data)  

        num_features = data.num_features

        num_classes = len(torch.unique(y))


        return data, num_features, num_classes
    

    elif 'ogb' in name:
        # name in ['ogbn-arxiv']:

        if name in ['ogbn-arxiv', 'ogbn-products']:

            # dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=path)
            # dataset = PygNodePropPredDataset(name=name, root=path, transform=T.ToSparseTensor())
            dataset = PygNodePropPredDataset(name=name, root=path)


            transformations = T.Compose([
                transform,
                T.ToSparseTensor(),
            ])
            dataset.transform = transformations

            data = dataset[0]
            data.adj_t = data.adj_t.to_symmetric()

            num_features = data.num_features
            num_classes = dataset.num_classes

        elif name == 'ogbn-proteins':
            # dataset = PygNodePropPredDataset(name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
            dataset = PygNodePropPredDataset(name='ogbn-proteins')

    
            transformations = T.Compose([
                transform,
                T.ToSparseTensor(attr='edge_attr'),
            ])
            dataset.transform = transformations

            
            data = dataset[0]
            data.x = data.adj_t.mean(dim=1)
            data.adj_t.set_value_(None)
            data.y = data.y.to(torch.float)

            num_features = data.num_features
            num_classes = data.y.size()[1]

        split_idx = dataset.get_idx_split()


        data.train_mask = index_to_mask(split_idx['train'], size=data.num_nodes)
        data.val_mask = index_to_mask(split_idx['valid'], size=data.num_nodes)
        data.test_mask = index_to_mask(split_idx['test'], size=data.num_nodes)


        
        data.y = torch.squeeze(data.y, dim=1)

        return data, num_features, num_classes

