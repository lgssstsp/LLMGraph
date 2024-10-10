import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, LSTM, GRU

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, JumpingKnowledge,SAGPooling
from torch_geometric.nn import ChebConv, GraphConv, SuperGATConv,LGConv, SGConv, MessagePassing, GravNetConv, TransformerConv 
from torch_geometric.nn import GINConv, SGConv, MFConv, GatedGraphConv, GCN2Conv, GINEConv, TAGConv
from torch.nn import Conv1d
from pyg_gnn_layer import GeoLayer
from torch_geometric.nn import global_add_pool,global_mean_pool,global_max_pool,global_sort_pool,GlobalAttention,Set2Set
import json
import argparse

NA_OPS = {}
def set_na_ops(na_list):
    global NA_OPS
    for PRIMITIVE in na_list:
        NA_OPS[PRIMITIVE] = (lambda in_dim, out_dim, PRIMITIVE=PRIMITIVE: NaAggregator(in_dim, out_dim, PRIMITIVE))



SC_OPS={
    'zero': lambda: Zero(),
    'identity': lambda: Identity(),
}


FF_OPS = {}
def set_ff_ops(ff_list):
    global FF_OPS
    for PRIMITIVE in ff_list:
        FF_OPS[PRIMITIVE] = (lambda hidden_size, num_layers, PRIMITIVE=PRIMITIVE: LaAggregator(PRIMITIVE, hidden_size, num_layers))




READOUT_OPS = {}
def set_readout_ops(readout_list):
    readout_dict = {"global_mean": 'mean',
    "global_sum": 'add',
    "global_max": 'max',
    'mean_max': 'mema',
    "none": 'none',
    'global_att': 'att',
    'global_sort': 'sort',
    'set2set': 'set2set'
    }

    global READOUT_OPS
    for PRIMITIVE in readout_list:
        operation = readout_dict[PRIMITIVE]
        READOUT_OPS[PRIMITIVE] = (lambda hidden_size, operation=operation: Readout_func(operation, hidden_size))
# readout_op, hidden



class NaAggregator(nn.Module):

    def __init__(self, in_dim, out_dim, aggregator):
        super(NaAggregator, self).__init__()
      
        # directly explore aggregator via its name from pyg.nn
        if aggregator == 'ChebConv':
            K = 3 
            self._op = globals()[aggregator](in_dim, out_dim, K)
        elif aggregator == 'SimpleConv':
            self._op = globals()[aggregator](in_channels=in_dim, out_channels=out_dim, combine_root='sum')
 
        else:
            self._op = globals()[aggregator](in_dim, out_dim)
        self.reset_parameters()


    def reset_parameters(self):
        self._op.reset_parameters()
        
    def forward(self, x, edge_index):
        return self._op(x, edge_index)



class LaAggregator(nn.Module):

    def __init__(self, mode, hidden_size, num_layers=3):
        super(LaAggregator, self).__init__()
        self.mode = mode
        if mode in ['lstm', 'cat', 'max']:
            self.jump = JumpingKnowledge(mode, hidden_size, num_layers=num_layers)
        elif mode == 'att':
            self.att = Linear(hidden_size, 1)

        if mode == 'cat':
            self.lin = Linear(hidden_size * num_layers, hidden_size)
        else:
            self.lin = Linear(hidden_size, hidden_size)
    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.mode in ['lstm', 'cat', 'max']:
            self.jump.reset_parameters()
        if self.mode == 'att':
            self.att.reset_parameters()

    def forward(self, xs):
        if self.mode in ['lstm', 'cat', 'max']:
            output = self.jump(xs)
        elif self.mode == 'sum':
            output = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            output = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'att':
            input = torch.stack(xs, dim=-1).transpose(1, 2)
            weight = self.att(input)
            weight = F.softmax(weight, dim=1)# cal the weightes of each layers and each node
            output = torch.mul(input, weight).transpose(1, 2).sum(dim=-1) #weighte sum

        return self.lin(F.relu(output))

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)


class Readout_func(nn.Module):
    def __init__(self, readout_op, hidden):

        super(Readout_func, self).__init__()
        self.readout_op = readout_op

        if readout_op == 'mean':
            self.readout = global_mean_pool

        elif readout_op == 'max':
            self.readout = global_max_pool

        elif readout_op == 'add':
            self.readout = global_add_pool

        elif readout_op == 'att':
            self.readout = GlobalAttention(Linear(hidden, 1))

        elif readout_op == 'set2set':
            processing_steps = 2
            self.readout = Set2Set(hidden, processing_steps=processing_steps)
            self.s2s_lin = Linear(hidden*processing_steps, hidden)


        elif readout_op == 'sort':
            self.readout = global_sort_pool
            self.k = 10
            self.sort_conv = Conv1d(hidden, hidden, 5)#kernel size 3, output size: hidden,
            self.sort_lin = Linear(hidden*(self.k-5 + 1), hidden)
        elif readout_op =='mema':
            self.readout = global_mean_pool
            self.lin = Linear(hidden*2, hidden)
        elif readout_op == 'none':
            self.readout = global_mean_pool
        # elif self.readout_op == 'mlp':

    def reset_parameters(self):
        if self.readout_op =='sort':
            self.sort_conv.reset_parameters()
            self.sort_lin.reset_parameters()
        if self.readout_op in ['set2set', 'att']:
            self.readout.reset_parameters()
        if self.readout_op =='set2set':
            self.s2s_lin.reset_parameters()
        if self.readout_op == 'mema':
            self.lin.reset_parameters()
    def forward(self, x, batch):
        #sparse data
        if self.readout_op == 'none':
            x = self.readout(x, batch)
            return x.mul(0.)
            # return None
        elif self.readout_op == 'sort':
            x = self.readout(x, batch, self.k)
            x = x.view(len(x), self.k, -1).permute(0, 2, 1)
            x = F.elu(self.sort_conv(x))
            x = x.view(len(x), -1)
            x = self.sort_lin(x)
            return x
        elif self.readout_op == 'mema':
            x1 = global_mean_pool(x, batch)
            x2 = global_max_pool(x, batch)
            x = torch.cat([x1, x2], dim=-1)
            x = self.lin(x)
            return x
        else:
            try:
                x = self.readout(x, batch)
            except:
                print(self.readout_op)
                print('size:', x.size, batch.size())
            if self.readout_op == 'set2set':
                x = self.s2s_lin(x)
            return x