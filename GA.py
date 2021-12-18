import networkx as nx
import random
import models.dataset_loader
import models.model_loader
import models.graph_embedding
import torch


class GA:
    def __init__(self, model):
        self.model = model
        self.rounds = 10
        self.population_size = 100
        self.selected_population_size = 50
        self.crossover_rate = 0.7
        self.mutation_rate = 0.05
        self.mutation_limit = 2
        self.label = None

    def crossover(self, p0, p1):
        if random.random > self.crossover_rate:
            return nx.Graph.copy(p0)
        else:
            child = nx.create_empty_copy(p0)
            p0_edges = set(p0.edges())
            p1_edges = set(p1.edges())

            for u, v in p0_edges & p1_edges:
                child.add_edge(u, v)

            for u, v in p0_edges - p1_edges:
                if random.random() < 0.5:
                    child.add_edge(u, v)

            for u, v in p1_edges - p0_edges:
                if random.random() < 0.5:
                    child.add_edge(u, v)

            if self.get_label(child) <= 3:
                return child
            else:
                return nx.Graph.copy(p0)

    def mutation(self, g):
        child = nx.Graph.copy(g)
        for _ in range(self.mutation_limit):
            if random.random() < self.mutation_rate:
                u, v = random.sample(child.edges(), 1)[0]
                node = random.sample(child.nodes(), 1)[0]
                if random.random() < 0.5 and (u, node) not in child.edges():
                    child.add_edge(u, node)
                    child.remove_edge(u, v)
                elif (node, v) not in child.edges():
                    child.add_edge(node, v)
                    child.remove_edge(u, v)
                else:
                    pass

        if self.get_label(child) <= 3 and self.get_label(child) == self.label:
            return child
        else:
            return nx.Graph.copy(g)

    def generate_breeding_population(self, g):
        return [self.mutation(g) for _ in range(self.selected_population_size)]

    def generate(self, graphs):
        population = []
        while len(population) < self.population_size:
            p0, p1 = random.sample(graphs, 2)
            child = self.crossover(p0, p1)
            if self.get_label(child) == self.label:
                population.append(self.crossover(p0, p1))

        population = [self.mutation(g) for g in population]
        return population

    def get_label(self, g):
        return nx.number_connected_components(g)

    # using loss function of model
    def get_loss(self, g):
        logits, loss, _ = \
            self.model([models.graph_embedding.S2VGraph(g, self.get_label(g))])
        label = torch.np.argmax(logits.data) + 1
        return label, loss.data[0]

    def select(self, graphs):
        graphs.sort(key=lambda x: x[1], reverse=True)
        return [g[0] for g in graphs[:self.selected_population_size]]

    def run(self, graph):
        self.label = self.get_label(graph)
        graphs = self.generate_breeding_population(graph)
        for i in range(self.rounds):
            graphs = [(g, self.get_loss(g)) for g in self.generate(graphs)]
            for g, (l, _) in graphs:
                if self.label == self.get_label(g) and self.label != l:
                    print('Actual label: ' + str(self.label))
                    print('Predicted label: ' + str(l))
                    return i, g
            else:
                graphs = self.select(graphs)

        return None, None


if __name__ == '__main__':
    label_map, train_glist, test_glist = models.dataset_loader.load_graphs(
        90, 100, 5000, 0.02, 1, 3, 'graph_adversarial_attack/data/components')
    model = \
        models.model_loader.load_graph_model(
            'graph_adversarial_attack/scratch/results/graph_classification/components/nodes-90-100-p-0.02-c-1-3-lv-5/',
            'epoch-best',
            label_map)

    attack_success = [0,0,0]
    for g in test_glist:
        round, adversarial = GA(model).run(g.to_networkx())
        real_label = nx.number_connected_components(g.to_networkx()) - 1
        if adversarial:
            attack_success[real_label] += 1
    
    
    attack_success_total = attack_success[0] + attack_success[1] + attack_success[2]
    print('[+] Final attack success rate : %.2f' % (attack_success_total/len(test_glist) * 100))
