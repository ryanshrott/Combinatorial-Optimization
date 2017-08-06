import json
import copy

import numpy as np  # contains helpful math functions like numpy.exp()
import numpy.random  # see numpy.random module
import random  # alternative to numpy.random module

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""Read input data and define helper functions for visualization."""

# Map services and data available from U.S. Geological Survey, National Geospatial Program.
# Please go to http://www.usgs.gov/visual-id/credit_usgs.html for further information
map = mpimg.imread("map.png")  # US States & Capitals map

# List of 30 US state capitals and corresponding coordinates on the map
with open('capitals.json', 'r') as capitals_file:
    capitals = json.load(capitals_file)
capitals_list = list(capitals.items())

def show_path(path, starting_city, w=12, h=8):
    """Plot a TSP path overlaid on a map of the US States & their capitals."""
    x, y = list(zip(*path))
    _, (x0, y0) = starting_city
    plt.imshow(map)
    plt.plot(x0, y0, 'y*', markersize=15)  # y* = yellow star for starting point
    plt.plot(x + x[:1], y + y[:1])  # include the starting point at the end of path
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])
	
def simulated_annealing(problem, schedule):    
    current = problem
    for t in range(1,10000000):
        T = schedule.expDecay(t)
        #print(T)
        if T < 1e-10:
            return current
        next_state = current.successor()
        delta_E = next_state.fitness() - current.fitness()
        if delta_E > 0:
            current = next_state
        else:
            prob = np.exp(delta_E/T)
            u = random.uniform(0,1)
            if u < prob:
                current = next_state
				
class TravelingSalesmanProblem:

    def __init__(self, cities):
        self.path = copy.deepcopy(cities)
    
    def copy(self):
        """Return a copy of the current board state."""
        new_tsp = TravelingSalesmanProblem(self.path)
        return new_tsp
    
    @property
    def names(self):
        names, _ = zip(*self.path)
        return names
    
    @property
    def coords(self):
        _, coords = zip(*self.path)
        return coords
    
    def successor(self, method='reverse'):
        if method == 'reverse':
            ind = sorted(random.sample([i for i,_ in enumerate(self.path)], 2))  
            new_path = self.path[:]
            new_path = new_path[:ind[0]] + new_path[ind[0]:ind[1]][::-1] + new_path[ind[1]:]
            return TravelingSalesmanProblem(new_path)
        elif method == 'permutation':
            new_path = self.path[:]
            random.shuffle(new_path)
            return TravelingSalesmanProblem(new_path)
        elif method == 'adjacent':
            successors = []
            for i in range(len(self.path)-1):
                new_problem = self.copy()
                new_problem.path[i], new_problem.path[i+1] = new_problem.path[i+1], new_problem.path[i]
                successors.append(new_problem)
                
            last_path = self.copy()
            last_path.path[0], last_path.path[-1] = last_path.path[-1], last_path.path[0]
            successors.append(last_path)
            return random.choice(successors)
        else:
            print('No valid method supplied')
            return False

    def fitness(self, metric='euclid'):
        # if the length is shorter, the fitness should be higher 
        # For example, if length = 10000, we return -10000
        # For example, if length = 10, we retun -10
        # Since -10 > -10000, the fitness is higher for the better path 
        def euclid(x, y):
            return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**.5 
        
        def manhattan(x,y):
            return (abs(x[0]-y[0]) + abs(x[1]-y[1])) 
        
        def inf(x,y):
            return (max(abs(x[0]-y[0]), (abs(x[1]-y[1])))) 
            

        if metric == 'euclid':
            norm = euclid 
        elif metric == 'manhattan':
            norm = manhattan
        elif metric == 'inf':
            norm = inf
            
        length = 0
        coords = self.coords
        for i in range(len(coords)-1):
            length += norm(coords[i], coords[i+1])
            
        length += norm(coords[0], coords[-1])
        return -length
    
    def mutate(self): # in place mutation 
        ind = random.sample([i for i,_ in enumerate(self.path)], 2) 
        # swap the cities on the path 
        self.path[ind[0]], self.path[ind[1]] = self.path[ind[1]], self.path[ind[0]]
    
    def reproduce(self, partner): # breeds with parents being the current instance 
                                  # and partner 
        if len(self.path) != len(partner.path):
            print('Cannot breed!')
            return False
        if random.uniform(0,1) > 0.5:
            ind = sorted(random.sample([i for i,_ in enumerate(self.path)], 2))  
            child_path = self.path[ind[0]:ind[1]]
            partners_added = 0
            for x in partner.path:
                if len(child_path) == len(self.path):
                    break
                if x not in child_path: 
                    partners_added += 1
                    if partners_added < ind[0]:
                        child_path.insert(0, x)
                    else:
                        child_path.append(x)
        else:
            ind = sorted(random.sample([i for i,_ in enumerate(partner.path)], 2))  
            child_path = partner.path[ind[0]:ind[1]]
            partners_added = 0
            for x in self.path:
                if len(child_path) == len(self.path):
                    break
                if x not in child_path: 
                    partners_added += 1
                    if partners_added < ind[0]:
                        child_path.insert(0, x)
                    else:
                        child_path.append(x)            
        if len(child_path) != len(set([x[0] for x in child_path])):
            print('Invalid breeding method!')
            return False
        
        return TravelingSalesmanProblem(child_path)
    
    def shuffle(self):
        new_problem = self.copy()
        random.shuffle(new_problem.path)
        return new_problem

        

class SalesmanPopulation:
    
    def __init__(self, population):
        self.pop = population
    
    def averageFitness(self):
        return np.mean([x.fitness() for x in self.pop])
    
    def evolve(self, retain=0.2, random_select=0.05, mutate=0.01):
        agent_performance = [(x.fitness(), x) for x in self.pop]
        sorted_perf = [x[1] for x in sorted(agent_performance, key=lambda x: x[0])][::-1]
        retain_length = int(len(sorted_perf)*retain)
        parents = sorted_perf[:retain_length]
        
        # randomly add other agents to promote genetic diversity 
        for individual in sorted_perf[retain_length:]:
            if random_select > random.random():
                parents.append(individual)
                
        # randomly mutate some individuals                 
        for i, individual in enumerate(parents):
            if mutate > random.random():
                parents[i].mutate()
                
        parents_length = len(parents)
        desired_length = len(self.pop) - parents_length
        children = []
        
        while len(children) < desired_length:
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)
            if male != female:
                male = parents[male]
                female = parents[female]
                child = male.reproduce(female)
                children.append(child)
        evolved_population = SalesmanPopulation(parents + children)
        return evolved_population
    
    def mostFitIndividual(self):
        # returns the fittest individual in the population
        fitness_dict = {x : x.fitness() for x in self.pop}
        return max(fitness_dict, key=fitness_dict.get)
        

class Schedule:
    def __init__(self, alpha, temperature):
        self.alpha = alpha
        self.temperature = temperature
    def expDecay(self, time):
        return self.alpha**(time) * self.temperature
        
	



if __name__ == "__main__":
    
    print('Solving the Genetic Algorithm Approach')
    num_cities = 30
    population_size = 1000
    evolution_cycles = 300
    starting_city = capitals_list[0]
    cities = capitals_list[:num_cities]
    tsp = TravelingSalesmanProblem(cities)
    show_path(tsp.coords, starting_city, w=4, h=3)
    population = SalesmanPopulation([tsp.shuffle() for _ in range(population_size)])
    print(population.averageFitness())
    fitness_history = []
    fitness_history.append(population.averageFitness())
    for i in range(evolution_cycles):
        population = population.evolve()
        avgFitness = population.averageFitness()
        fitness_history.append(avgFitness)
        print(i, population.averageFitness())
    plt.plot(fitness_history)    
    fittest_path = population.mostFitIndividual()
    print('Fittest individual has fitness {:.2f}'.format(fittest_path.fitness()))
    #show_path(fittest_path.coords, starting_city, w=4, h=3)

#    print('Solving the Simulated Annealing Approach')
#    capitals_tsp = TravelingSalesmanProblem(capitals_list[:num_cities])
#    starting_city = capitals_list[0]
#    #print("Initial path value: {:.2f}".format(-capitals_tsp.fitness()))
#    alpha = 0.999
#    temperature=1e20
#    result = simulated_annealing(capitals_tsp, Schedule(alpha, temperature))
#    print("Final path length: {:.2f}".format(result.fitness()))
#    #print(result.path)
#    show_path(result.coords, starting_city, w=4, h=3)
#
