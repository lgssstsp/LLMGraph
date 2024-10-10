


import json
import argparse
import ast
# from train_search import args


NA_PRIMITIVES = []
def set_na_primitives(na_list):
    global NA_PRIMITIVES
    for PRIMITIVES in na_list:
        NA_PRIMITIVES.append(PRIMITIVES)


SC_PRIMITIVES = [
    'zero',
    'identity',
]

FF_PRIMITIVES = []
def set_ff_primitives(ff_list):
    global FF_PRIMITIVES
    for PRIMITIVES in ff_list:
        FF_PRIMITIVES.append(PRIMITIVES)




READOUT_PRIMITIVES = []
def set_readout_primitives(readout_list):
    global READOUT_PRIMITIVES
    for PRIMITIVES in readout_list:
        READOUT_PRIMITIVES.append(PRIMITIVES)

