import torch

import pickle as cp
from .dnn import GraphClassifier
from .gcn_modules import S2VNodeClassifier, GCNModule
from . import context

def load_graph_model(model_dir, model_name, label_map):
    with open('{}/{}-args.pkl'.format(model_dir, model_name), 'rb') as f:
        base_args = cp.load(f)
    base_args.ctx = context.ctx

    classifier = GraphClassifier(label_map, **vars(base_args))
    if context.ctx == 'gpu':
        classifier = classifier.cuda()

    classifier.load_state_dict(torch.load('{}/{}.model'.format(model_dir, model_name)))

    return classifier

def load_node_model(model_dir, model_name):
    with open('{}/{}-args.pkl'.format(model_dir, model_name), 'rb') as f:
        base_args = cp.load(f)
    base_args.ctx = context.ctx
    if 'mean_field' in model_name:
        mod = S2VNodeClassifier
    elif 'gcn' in model_name:
        mod = GCNModule
    gcn = mod(**vars(base_args))
    if context.ctx == 'gpu':
        gcn = gcn.cuda()
    model = torch.load('{}/{}.model'.format(model_dir, model_name), map_location=context.ctx)
    gcn.load_state_dict(model)
    return gcn