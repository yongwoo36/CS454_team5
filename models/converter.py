import pickle
import os

import networkx as nx
from networkx.readwrite import json_graph
import json

def graphs_to_json(dataset_dir):
    for (root, dirs, files) in os.walk(dataset_dir):
        for file_name in files:
            path = os.path.join(root, file_name)
            if 'args' in file_name or not file_name.endswith('.pkl'):
                continue
            graphs = []
            with open(path, 'rb') as f:
                while 1:
                    try:
                        graphs.append( pickle.load(f))
                    except (EOFError, pickle.UnpicklingError):
                        break
            datas = [json_graph.node_link_data(graph) for graph in graphs]
            model_name, _ = os.path.splitext(file_name)
            json_path = os.path.join(root, model_name + '.json')
            print(json_path)
            with open(json_path, 'wb') as f:
                json.dump(datas, f)

def convert_graphs(dataset_dir):
    for (root, dirs, files) in os.walk(dataset_dir):
        for file_name in files:
            path = os.path.join(root, file_name)

            if 'args' in file_name or not file_name.endswith('.json'):
                continue
            with open(path, 'rb') as f:
                json_datas = json.load(f)
            graphs = [json_graph.node_link_graph(json_data) for json_data in json_datas]
            model_name, _ = os.path.splitext(file_name)
            pkl_path = os.path.join(root, model_name + '.pkl')
            print(pkl_path)
            with open(pkl_path, 'wb') as f:
                for graph in graphs:
                    pickle.dump(graph, f)

# load graphs with networkx 1 and dump json in python2
# $ python2 models/converter.py ../graph_adversarial_attack/data/components
# load json data to networkx 2 graphs and dump graphs as pickle in python3
# $ python models/converter.py ../graph_adversarial_attack/data/components
if __name__ == "__main__":
    import sys
    
    dataset_dir = sys.argv[1]
    py_ver = int(sys.version[0])
    if py_ver == 2:
        graphs_to_json(dataset_dir)
    elif py_ver == 3:
        convert_graphs(dataset_dir)
    