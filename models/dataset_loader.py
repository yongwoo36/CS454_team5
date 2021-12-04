# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from data_util import *
from node_utils import StaticGraph
from graph_embedding import S2VGraph

"""
데이터셋은 https://www.dropbox.com/sh/mu8odkd36x54rl3/AABg8ABiMqwcMEM5qKIY97nla?dl=0 에서 다운로드후 dropbox로 이름 변경
"""

#Graph attack용 데이터 불러오는 함수들
#(min_n, max_n, er_p): 그래프의 노드 개수 범위 (15,20,0.05) | (15,20,0.15) | (40,50,0.05) | (40,50,0.15) | (90,100,0.02) | (90,100,0.05)
#n_graphs: 5000
#(min_c, max_c): Component 개수 (1,3)
#data_folder: ../dropbox/data/components

def load_graphs(min_n, max_n, n_graphs, er_p, min_c, max_c, data_folder):
    frac_train = 0.9
    pattern = 'nrange-%d-%d-n_graph-%d-p-%.2f' % (min_n, max_n, n_graphs, er_p)

    num_train = int(frac_train * n_graphs)

    train_glist = []
    test_glist = []
    label_map = {}
    for i in range(min_c, max_c + 1):
        cur_list = load_pkl('%s/ncomp-%d-%s.pkl' % (data_folder, i, pattern), n_graphs)
        assert len(cur_list) == n_graphs
        train_glist += [S2VGraph(cur_list[j], i) for j in range(num_train)]
        test_glist += [S2VGraph(cur_list[j], i) for j in range(num_train, len(cur_list))]
        label_map[i] = i - min_c

    print('# train:', len(train_glist), ' # test:', len(test_glist))

    return label_map, train_glist, test_glist

#Node attack용 데이터 불러오는 함수들
#data_folder: ../dropbox/data + / + {dataset_str}
#dataset_str: pubmed | cora | siteseer
#ctx: cpu | gpu

def load_txt_data(data_folder, dataset_str, ctx):
    idx_train = list(np.loadtxt(data_folder + '/train_idx.txt', dtype=int))
    idx_val = list(np.loadtxt(data_folder + '/val_idx.txt', dtype=int))
    idx_test = list(np.loadtxt(data_folder + '/test_idx.txt', dtype=int))
    labels = np.loadtxt(data_folder + '/label.txt')
    
    with open(data_folder + '/meta.txt', 'r') as f:
        num_nodes, num_class, feature_dim = [int(w) for w in f.readline().strip().split()]

    graph = load_raw_graph(data_folder, dataset_str)
    assert len(graph) == num_nodes
    StaticGraph.graph = nx.from_dict_of_lists(graph) # should be StaticGraph of node_utils instead of data_util
    
    row_ptr = []
    col_idx = []
    vals = []
    with open(data_folder + '/features.txt', 'r') as f:
        nnz = 0
        for row in f:
            row = row.strip().split()
            row_ptr.append(nnz)            
            for i in range(1, len(row)):
                w = row[i].split(':')
                col_idx.append(int(w[0]))
                vals.append(float(w[1]))
            nnz += int(row[0])
        row_ptr.append(nnz)
    assert len(col_idx) == len(vals) and len(vals) == nnz and len(row_ptr) == num_nodes + 1

    features = sp.csr_matrix((vals, col_idx, row_ptr), shape=(num_nodes, feature_dim))
       
    return preprocess_features(features, ctx), labels, idx_train, idx_val, idx_test
