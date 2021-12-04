import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh

from custom_func import GraphLaplacianNorm, GraphDegreeNorm
import context

class StaticGraph(object):
    graph = None

    @staticmethod
    def get_gsize():
        return torch.Size( (len(StaticGraph.graph), len(StaticGraph.graph)) )

class GraphNormTool(object):
    def __init__(self, adj_norm, gm):
        self.adj_norm = adj_norm
        self.gm = gm
        g = StaticGraph.graph
        edges = np.array(g.edges(), dtype=np.int64)
        rev_edges = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
        self_edges = np.array([range(len(g)), range(len(g))], dtype=np.int64)

        edges = np.hstack((edges.T, rev_edges, self_edges))

        idxes = torch.LongTensor(edges)
        values = torch.ones(idxes.size()[1])

        self.raw_adj = torch.sparse.FloatTensor(idxes, values, StaticGraph.get_gsize())
        if context.ctx == 'gpu':
            self.raw_adj = self.raw_adj.cuda()
        
        self.normed_adj = self.raw_adj.clone()
        if self.adj_norm:
            if self.gm == 'gcn':
                GraphLaplacianNorm(self.normed_adj)
            else:
                GraphDegreeNorm(self.normed_adj)

    def norm_extra(self, added_adj = None):
        if added_adj is None:
            return self.normed_adj

        new_adj = self.raw_adj + added_adj
        if self.adj_norm:
            if self.gm == 'gcn':
                GraphLaplacianNorm(new_adj)
            else:
                GraphDegreeNorm(new_adj)
        return new_adj
