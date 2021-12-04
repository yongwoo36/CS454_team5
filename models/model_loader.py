import torch

import pickle as cp
from .dnn import GraphClassifier
from . import context

def load_graph_model(model_dir, model_name, label_map):
    with open('{}/{}-args.pkl'.format(model_dir, model_name), 'rb') as f:
        base_args = cp.load(f)

    print(base_args)
    classifier = GraphClassifier(label_map, **vars(base_args))
    if context.ctx == 'gpu':
        classifier = classifier.cuda()

    classifier.load_state_dict(torch.load('{}/{}.model'.format(model_dir, model_name)))
   
    return classifier

def load_node_model(model_dir, model_name, classifier_name, ):
    with open('{}/{}-args.pkl'.format(model_dir, model_name), 'rb') as f:
        base_args = cp.load(f)

    if 'mean_field' == model_name:
        mod = S2VNodeClassifier
    elif 'gcn' == model_name:
        mod = GCNModule

    gcn = mod(**vars(base_args))
    if cmd_args.ctx == 'gpu':
        gcn = gcn.cuda()
    gcn.load_state_dict(torch.load('{}/{}.model'.format(model_dir, model_name)))
    gcn.eval()
    return gcn