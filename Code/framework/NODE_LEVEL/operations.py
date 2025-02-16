import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, LSTM, GRU

from torch_geometric.nn import *
from torch_geometric.nn import SimpleConv, GCNConv, ChebConv, SAGEConv, CuGraphSAGEConv, GraphConv, GravNetConv, GatedGraphConv, ResGatedGraphConv, GATConv, CuGraphGATConv, FusedGATConv, GATv2Conv, TransformerConv, AGNNConv, TAGConv, GINConv, GINEConv, ARMAConv, SGConv, SSGConv, APPNP, MFConv, RGCNConv, FastRGCNConv, CuGraphRGCNConv, RGATConv, SignedConv, DNAConv, PointNetConv, GMMConv, SplineConv, NNConv, CGConv, EdgeConv, DynamicEdgeConv, XConv, PPFConv, FeaStConv, PointTransformerConv, HypergraphConv, LEConv, PNAConv, ClusterGCNConv, GENConv, GCN2Conv, PANConv, WLConv, WLConvContinuous, FiLMConv, SuperGATConv, FAConv, EGConv, PDNConv, GeneralConv, HGTConv, HEATConv, HeteroConv, HANConv, LGConv, PointGNNConv, GPSConv, AntiSymmetricConv, DirGNNConv, MixHopConv

from pyg_gnn_layer import GeoLayer
from geniepath import GeniePathLayer
import json
import argparse
from blockgnn import GNNBlock




NA_OPS = {}
def set_na_ops(na_list):
    global NA_OPS
    for PRIMITIVE in na_list:
        try:
            NA_OPS[PRIMITIVE] = (lambda in_dim, out_dim, PRIMITIVE=PRIMITIVE: NaAggregator(in_dim, out_dim, PRIMITIVE))
        except KeyError as e:
            print(f"An unexpected error occurred: {e}")
            continue



SC_OPS={
    'zero': lambda: Zero(),
    'identity': lambda: Identity(),
}


FF_OPS = {}
def set_ff_ops(ff_list):
    global FF_OPS
    for PRIMITIVE in ff_list:
        FF_OPS[PRIMITIVE] = (lambda hidden_size, num_layers, PRIMITIVE=PRIMITIVE: LaAggregator(PRIMITIVE, hidden_size, num_layers))




class NaAggregator(nn.Module):

    def __init__(self, in_dim, out_dim, aggregator):
        super(NaAggregator, self).__init__()
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

