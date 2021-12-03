import numpy as np
import networkx as nx

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None):        
        self.num_nodes = len(g.nodes())
        self.node_tags = node_tags
        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.label = label
        
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = self.edge_pairs.flatten()

    def to_networkx(self):
        edges = np.reshape(self.edge_pairs, (self.num_edges, 2))
        g = nx.Graph()
        g.add_edges_from(edges)
        return g
    