import networkx as nx
import random

class GA:
    def __init__(self, m):
        self.first_population_num = 10
        self.generation_num = 100
        self.max_graphs_num = 50
        self.select_rate = 0.6
        self.crossover_rate = 0.7
        self.mutation_limit = 2
        self.model = m
        self.mutation_cnt = []

    def crossover(self, p0, p1):
        child = nx.create_empty_copy(p0)
        p0_edges = set(p0.edges)
        p1_edges = set(p1.edges)
        only_p0_edges = p0_edges - p1_edges
        only_p1_edges = p1_edges - p0_edges
        common_edges = p0_edges & p1_edges
        for u, v in common_edges:
            child.add_edge(u, v)
        
        for u, v in only_p0_edges | only_p1_edges:
            if random.random() < 0.5:
                child.add_edge(u, v)
        
        return child

    def mutation(self, p):
        child = nx.Graph.copy(p)
        s, e = random.sample(child.edges, 1)[0]
        while self.mutation_cnt[s] > self.mutation_limit or self.mutation_cnt[e] > self.mutation_limit:
            s, e = random.sample(child.edges, 1)[0]

        while True:
            new_e = random.sample(child.nodes, 1)[0]
            if (s, new_e) not in child.edges and self.mutation_cnt[new_e] < self.mutation_limit:
                child.remove_edge(s, e)
                child.add_edge(s, new_e)
                self.mutation_cnt[s] += 1
                self.mutation_cnt[new_e] += 1
                return child

    def generate_first_population(self, g):
        graphs = [g]
        while len(graphs) < self.first_population_num:
            child = self.mutation(g)
            graphs.append(child)
        return graphs


    def generate(self, graphs):
        max_index = len(graphs)
        while len(graphs) < self.max_graphs_num:
            # crossover
            if random.random() < self.crossover_rate:
                p0, p1 = random.sample(graphs[:max_index], 2)
                child = self.crossover(p0, p1)
                graphs.append(child)
            # mutate
            else:
                p = random.sample(graphs[:max_index], 1)[0]
                child = self.mutation(p)
                graphs.append(child)
        return graphs

    # using loss function of model
    def get_fitness(self, g, l):
        return random.random()

    # predict graph and compare the result
    def is_adversarial(self, g, l):
        # This is pseudocode
        # pred = model.predict(g)
        # adv_l = np.argmax(pred)
        # return l != adv_l
        return False

    def run(self, graph, label):
        # generate first population
        self.mutation_cnt = [0 for _ in range(len(graph.nodes))]
        graphs = self.generate_first_population(graph)
        graphs = self.generate(graphs)
        for _ in range(self.generation_num):
            graphs_with_fitness = []
            for g in graphs:
                f = self.get_fitness(g, label)
                graphs_with_fitness.append((g, f))
            graphs_with_fitness.sort(key=lambda x:x[1], reverse=True)
            selected_graphs = list(map(lambda x:x[0],
                graphs_with_fitness[:int(self.max_graphs_num * self.select_rate)]))
            if self.is_adversarial(selected_graphs[0], label):
                break
            # generate new population
            graphs = self.generate(selected_graphs)


if __name__ == '__main__':
    g = nx.Graph()
    for i in range(30):
        g.add_node(i)
    for _ in range(40):
        s, e = random.sample(range(30), 2)
        g.add_edge(s, e)
    
    GA(0).run(g, 0)