import scipy.sparse as sp
import numpy as np
import pickle as pkl
import networkx as nx
import os
import sys
import torch


def load_pkl(fname, num_graph):
    g_list = []
    with open(fname, 'rb') as f:
        for i in range(num_graph):
            g = pkl.load(f)
            g_list.append(g)
    return g_list

def load_raw_graph(data_folder, dataset_str):
    bin_file = "{}/ind.{}.{}".format(data_folder, dataset_str, 'graph')
    if os.path.isfile(bin_file):
        with open(bin_file, 'rb') as f:
            if sys.version_info > (3, 0):
                graph = pkl.load(f, encoding='latin1')
            else:
                graph = pkl.load(f)
    else:
        txt_file = data_folder + '/adj_list.txt'
        graph = {}
        with open(txt_file, 'r') as f:
            cur_idx = 0
            for row in f:
                row = row.strip().split()
                adjs = []
                for j in range(1, len(row)):
                    adjs.append(int(row[j]))
                graph[cur_idx] = adjs
                cur_idx += 1

    return graph

class StaticGraph(object):
    graph = None

    @staticmethod
    def get_gsize():
        return torch.Size( (len(StaticGraph.graph), len(StaticGraph.graph)) )

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features, ctx):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)    
    sp_tuple = sparse_to_tuple(features)
    idxes = torch.LongTensor(sp_tuple[0]).transpose(0, 1).contiguous()
    values = torch.Tensor(sp_tuple[1].astype(np.float32))

    mat = torch.sparse.FloatTensor(idxes, values, torch.Size(sp_tuple[2]))
    if ctx == 'gpu':
        mat = mat.cuda()
    return mat