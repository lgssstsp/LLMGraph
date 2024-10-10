import json
import argparse
import ast

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


