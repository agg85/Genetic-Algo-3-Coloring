from doctest import REPORT_CDIFF
from lib2to3.pgen2.pgen import generate_grammar
from pdb import Restart
from time import time
from Graph_Creator import *
import random
import matplotlib.pyplot as plt

def main():
    start_time = time()
    duration = 45                                           # in secs
    gc = Graph_Creator()

    # Hyperparameters (sort of)
    max_itr = 10
    num_edges = 100
    num_gen = 500
    popln_sz = 500
    gens_before_restart = 30
    mutation_rate = 0.1

    # edges = gc.CreateGraphWithRandomEdges(num_edges)        # Creates a random graph with 50 vertices and 200 edges
    edges = gc.ReadGraphfromCSVfile("./Testcases/{0}".format(num_edges))    # Reads the edges of the graph from a given CSV file

    ga = GeneticAlgorithm(edges, ['r', 'g', 'b'], num_gen, 50, popln_sz, (start_time+duration-1), gens_before_restart, mutation_rate)
    bf_state, bf_vals_per_gen = ga.genetic_algo()           # bf: best_fitness
    time_taken = time() - start_time
    print("Roll no.: 2019A7PS0017G\nNumber of edges: {0}\nBest state: {1}\nFitness value of best state: {2}\nTime taken: {3} secs"\
        .format(num_edges, [{i:bf_state[1][i]} for i in range(50)], bf_state[0], int(time_taken+1)))
    
    plot_bf_vals_per_gen(bf_vals_per_gen)


class GeneticAlgorithm():
    edges = []                                              # edges of the given graph
    state_val = []                                          # values that each variable can take
    state_sz = 0                                            # state size (no. of characters in state string) eg. no. of nodes in graph
    popln_sz = 0                                            # population size (no. of states in current generation)
    max_gen = 0                                             # max no. of generations for which the algo can run
    end_time = 0                                            # max time till which algo can run
    gens_before_restart = 10000                             # generations to wait for before randomly restarting from new state
    mutation_rate = 0                                       # probability with which mutation should be tried in given state
    
    
    def __init__(self, edges, values, num_gen, state_sz, popln_sz, end_time, gens_before_restart, mutation_rate=0):
        self.edges = edges
        self.state_val = values
        self.state_sz = state_sz
        self.popln_sz = popln_sz
        self.max_gen = num_gen
        self.end_time = end_time
        self.gens_before_restart = gens_before_restart
        self.mutation_rate = mutation_rate
    
    
    def genetic_algo(self):
        population = []
        bf_vals_per_gen = []                                # best fitness value generation-wise
        bf_state = [0,""]                                   # best fitness state in curr generation

        restart = 0
        for gen in range(self.max_gen):
            if gen == restart:
                for i in range(self.popln_sz):
                    state = random.choices(population = self.state_val, k = self.state_sz)
                    population += [[self.fitness(state), state]]
            else:
                population = self.nextGen(population, weights)
            
            weights = [indiv[0] for indiv in population]
            gen_bf = max(population)
            bf_vals_per_gen.append(gen_bf[0])
            bf_state = max(bf_state, gen_bf)
            
            # random restart condition
            if gen-restart >= self.gens_before_restart and (bf_vals_per_gen[gen]-bf_vals_per_gen[gen-self.gens_before_restart]) <= 2: 
                restart = gen+1
                population = []

            if time() > self.end_time:
                break
        
        return bf_state, bf_vals_per_gen
    

    def fitness(self, state):
        valid_col = [1 for node in range(self.state_sz)]
        for edge in self.edges:
            if state[edge[0]] == state[edge[1]]:
                valid_col[edge[0]] = valid_col[edge[1]] = 0
        return sum(valid_col)


    def nextGen(self, population, weights):
        next_gen = []
        while len(next_gen) != 2*self.popln_sz:
            parents = random.choices(population = population, weights = weights, k = 2)
            children = self.reproduce([parent[1] for parent in parents])
            children = [self.mutate(child) for child in children]
            next_gen += [[self.fitness(child), child] for child in children]
        
        population += next_gen
        return self.elitismAndCulling(population)


    def reproduce(self, parents):
        cp = random.randint(1,self.state_sz)                # crossover_point
        return [parents[0][:cp]+parents[1][cp:], parents[1][:cp]+parents[0][cp:]]
    
    
    def mutate(self, state):
        if random.random() <= self.mutation_rate:
            mp = random.randint(0,self.state_sz-1)          # mutation_point
            state[mp] = random.choice(self.state_val)
        return state
    

    def elitismAndCulling(self, population):
        population.sort(reverse = True)
        return population[:self.popln_sz]

    
def plot_bf_vals_per_gen(bf_vals_per_gen):
    plt.plot(range(len(bf_vals_per_gen)), bf_vals_per_gen)
    plt.xlabel("generation")
    plt.ylabel("best fitness value")
    plt.show()


def plot_bf_vals_per_num_edges(bf_vals_per_num_edges):
    plt.plot(range(100,600,100), bf_vals_per_num_edges)
    plt.xlabel("no. of edges")
    plt.ylabel("best fitness value")
    plt.show()


if __name__=='__main__':
    main()