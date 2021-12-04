# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np

from dataset_loader import load_graphs, load_txt_data
from model_loader import load_graph_model, load_node_model

def graph_classification():
    dataset_dir = '../graph_adversarial_attack/data/components'
    label_map, train_list, test_list = load_graphs(15, 20, 5000, 0.15, 1, 3, dataset_dir)
    model_dir = '../graph_adversarial_attack/scratch/results/graph_classification/components/nodes-15-20-p-0.15-c-1-3-lv-2'
    model_name = 'epoch-best'
    classifier = load_graph_model(model_dir, model_name, label_map)
    log_ll, loss, acc = classifier(test_list[:5])
    print('log_ll :', log_ll)
    print('loss :', loss)
    print('acc :', acc)

def node_classification():
    dataset_dir = '../graph_adversarial_attack/data/pubmed'
    features, labels, idx_train, idx_val, idx_test = load_txt_data(dataset_dir, "pubmed", "cpu")
    features = Variable( features )
    labels = Variable( torch.LongTensor( np.argmax(labels, axis=1) ) )

    model_dir = '../graph_adversarial_attack/scratch/results/node_classification/pubmed'
    model_name = 'model-gcn-epoch-best-0.00'
    gcn = load_node_model(model_dir, model_name)
    gcn.eval() # 테스트시에만. train할땐 안쓰는듯
    adj = Variable(gcn.norm_tool.normed_adj) # train 모드일때랑 eval 모드일떄랑 다른듯? gcn.py 참고  
    _, loss, acc = gcn(features, adj, idx_test, labels)
    acc = acc.sum() / float(len(idx_test))
    print("Test set results:",
          "loss= {:.4f}".format(loss.data[0]),
          "accuracy= {:.4f}".format(acc))
if __name__ == "__main__":
    graph_classification()
    node_classification()