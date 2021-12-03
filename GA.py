import networkx as nx
import random


class GA:
    def __init__(self, model):
        self.model = model
        self.rounds = 10
        self.population_size = 100
        self.selected_population_size = 50
        self.crossover_rate = 0.7
        self.mutation_rate = 0.05
        self.mutation_limit = 2

    def crossover(self, p0, p1):
        if random.random > self.crossover_rate:
            return p0
        else:
            child = nx.create_empty_copy(p0)
            p0_edges = set(p0.edges)
            p1_edges = set(p1.edges)

            for u, v in p0_edges & p1_edges:
                child.add_edge(u, v)

            for u, v in p0_edges - p1_edges:
                if random.random() < 0.5:
                    child.add_edge(u, v)

            for u, v in p1_edges - p0_edges:
                if random.random() < 0.5:
                    child.add_edge(u, v)

            return child

    def mutation(self, p):
        child = nx.Graph.copy(p)
        for _ in range(self.mutation_limit):
            if random.random() < self.mutation_rate:
                u, v = random.sample(child.edges, 1)[0]
                node = random.sample(child.nodes, 1)[0]
                if random.random() < 0.5 and (u, node) not in child.edges:
                    child.add_edge(u, node)
                    child.remove_edge(u, v)
                elif (node, v) not in child.edges:
                    child.add_edge(node, v)
                    child.remove_edge(u, v)
                else:
                    pass

        return child

    def generate_breeding_population(self, g):
        return [self.mutation(g) for _ in range(self.selected_population_size)]

    def generate(self, graphs):
        population = [
            self.crossover(random.sample(graphs, 2))
            for _ in range(self.population_size)
        ]
        population = [self.mutation(g) for g in population]
        return population

    def get_label(self, g):
        # TODO:
        pass

    # using loss function of model
    def get_fitness(self, g):
        # TODO:
        label = get_label(g)
        return random.random()

    # predict graph and compare the result
    def is_adversarial(self, g, fitness):
        return fitness > 0.8

    def select(self, graphs):
        graphs.sort(key=lambda x: x[1], reverse=True)
        return graphs[:self.selected_population_size]

    def run(self, graph):
        graphs_with_fitness = self.generate_breeding_population(graph)
        for _ in range(self.rounds):
            graphs = [(g, self.get_fitness(g)) for g in self.generate(graphs)]
            for g, fitness in graphs:
                if self.is_adversarial(g, fitness):
                    return g
            else:
                graphs = self.select(graphs)

        return None


if __name__ == '__main__':
    # TODO: load dataset and model
    model = None

    g = nx.Graph()
    for i in range(30):
        g.add_node(i)

    for _ in range(40):
        s, e = random.sample(range(30), 2)
        g.add_edge(s, e)

    adversarial_graph = GA(model).run(g)
    # TODO: print?
