from .dataset_loader import load_graphs, load_txt_data
from .model_loader import load_graph_model


def graph_classification():
    dataset_dir = '../graph_adversarial_attack/data/components'
    label_map, train_list, test_list = load_graphs(15, 20, 5000, 0.15, 1, 3, dataset_dir)
    classifier = load_graph_model('../graph_adversarial_attack/scratch/results/graph_classification/components/nodes-15-20-p-0.15-c-1-3-lv-2', 'epoch-best', label_map)
    log_ll, loss, acc = classifier(test_list[:5])
    print(log_ll, loss, acc)

# load_graphs(15, 20, 5000, 0.05, 1, 3, "dropbox/data/components")
# print(load_txt_data("dropbox/data/pubmed", "pubmed", "cpu"))

if __name__ == "__main__":
    graph_classification()