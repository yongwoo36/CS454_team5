import torch

import pickle as cp
from .dnn import GraphClassifier
from . import context

def load_graph_model(model_dir, model_name, label_map):
    with open(f'{model_dir}/{model_name}-args.pkl', 'rb') as f:
        base_args = cp.load(f)

    print(base_args)
    classifier = GraphClassifier(label_map, **vars(base_args))
    if context.ctx == 'gpu':
        classifier = classifier.cuda()

    classifier.load_state_dict(torch.load(f'{model_dir}/{model_name}.model'))
   
    return classifier

def load_node_model(model_dir, model_name, classifier_name, ):
    with open(f'{model_dir}/{model_name}-args.pkl', 'rb') as f:
        base_args = cp.load(f)

    if 'mean_field' == model_name:
        mod = S2VNodeClassifier
    elif 'gcn' == model_name:
        mod = GCNModule

    gcn = mod(**vars(base_args))
    if cmd_args.ctx == 'gpu':
        gcn = gcn.cuda()
    gcn.load_state_dict(torch.load(f'{model_dir}/{model_name}.model'))
    gcn.eval()
    return gcn