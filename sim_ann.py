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
    """The simulated annealing algorithm, a version of stochastic hill climbing
    where some downhill moves are allowed. Downhill moves are accepted readily
    early in the annealing schedule and then less often as time goes on. The
    schedule input determines the value of the temperature T as a function of
    time. [Norvig, AIMA Chapter 3]
    
    Parameters
    ----------
    problem : Problem
        An optimization problem, already initialized to a random starting state.
        The Problem class interface must implement a callable method
        "successors()" which returns states in the neighborhood of the current
        state, and a callable function "get_value()" which returns a fitness
        score for the state. (See the `TravelingSalesmanProblem` class below
        for details.)

    schedule : callable
        A function mapping time to "temperature". "Time" is equivalent in this
        case to the number of loop iterations.
    
    Returns
    -------
    Problem
        An approximate solution state of the optimization problem
        
    Notes
    -----
        (1) DO NOT include the MAKE-NODE line from the AIMA pseudocode

        (2) Modify the termination condition to return when the temperature
        falls below some reasonable minimum value (e.g., 1e-10) rather than
        testing for exact equality to zero
        
    See Also
    --------
    AIMA simulated_annealing() pseudocode
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Simulated-Annealing.md
    """     
    current = problem
    for t in range(1,10000000):
        T = schedule(t)
        #print(T)
        if T < 1e-10:
            print('Terminated with t', t)
            return current
        next_state = current.successor()
        delta_E = next_state.get_value() - current.get_value()
        if delta_E > 0:
            current = next_state
        else:
            prob = np.exp(delta_E/T)
            u = random.uniform(0,1)
            if u < prob:
                current = next_state
				
class TravelingSalesmanProblem:
    """Representation of a traveling salesman optimization problem.  The goal
    is to find the shortest path that visits every city in a closed loop path.
    
    Students should only need to implement or modify the successors() and
    get_values() methods.
    
    Parameters
    ----------
    cities : list
        A list of cities specified by a tuple containing the name and the x, y
        location of the city on a grid. e.g., ("Atlanta", (585.6, 376.8))
    
    Attributes
    ----------
    names
    coords
    path : list
        The current path between cities as specified by the order of the city
        tuples in the list.
    """
    def __init__(self, cities):
        self.path = copy.deepcopy(cities)
    
    def copy(self):
        """Return a copy of the current board state."""
        new_tsp = TravelingSalesmanProblem(self.path)
        return new_tsp
    
    @property
    def names(self):
        """Strip and return only the city name from each element of the
        path list. For example,
            [("Atlanta", (585.6, 376.8)), ...] -> ["Atlanta", ...]
        """
        names, _ = zip(*self.path)
        return names
    
    @property
    def coords(self):
        """Strip the city name from each element of the path list and return
        a list of tuples containing only pairs of xy coordinates for the
        cities. For example,
            [("Atlanta", (585.6, 376.8)), ...] -> [(585.6, 376.8), ...]
        """
        _, coords = zip(*self.path)
        return coords
    
    def successors(self):
        """Return a list of states in the neighborhood of the current state by
        switching the order in which any adjacent pair of cities is visited.
        
        For example, if the current list of cities (i.e., the path) is [A, B, C, D]
        then the neighbors will include [A, B, D, C], [A, C, B, D], [B, A, C, D],
        and [D, B, C, A]. (The order of successors does not matter.)
        
        In general, a path of N cities will have N neighbors (note that path wraps
        around the end of the list between the first and last cities).

        Returns
        -------
        list<Problem>
            A list of TravelingSalesmanProblem instances initialized with their list
            of cities set to one of the neighboring permutations of cities in the
            present state
        """
        
        successors = []
        for i in range(len(self.path)-1):
            new_problem = self.copy()
            new_problem.path[i], new_problem.path[i+1] = new_problem.path[i+1], new_problem.path[i]
            successors.append(new_problem)
            
        last_path = self.copy()
        last_path.path[0], last_path.path[-1] = last_path.path[-1], last_path.path[0]
        successors.append(last_path)
        
        return successors 
        
            
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
        
        
        
            
    def shuffle(self):
        new_problem = self.copy()
        random.shuffle(new_problem.path)
        return new_problem
            

    def get_value(self, metric='euclid'):
        """Calculate the total length of the closed-circuit path of the current
        state by summing the distance between every pair of adjacent cities.  Since
        the default simulated annealing algorithm seeks to maximize the objective
        function, return -1x the path length. (Multiplying by -1 makes the smallest
        path the smallest negative number, which is the maximum value.)
        
        Returns
        -------
        float
            A floating point value with the total cost of the path given by visiting
            the cities in the order according to the self.cities list
        
        Notes
        -----
            (1) Remember to include the edge from the last city back to the
            first city
            
            (2) Remember to multiply the path length by -1 so that simulated
            annealing finds the shortest path
        """
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
    
    def fitness(self, metric='euclid'):
        # if the length is shorter, the fitness should be higher 
        # For example, if length = 10000, we return -10000
        # For example, if length = 10, we retun -10
        # Since -10 > -10000, the fitness is higher for the better path 
        return self.get_value(metric=metric)
        

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
        fitness_dict = {x : x.fitness() for x in self.pop}
        return max(fitness_dict, key=fitness_dict.get)
        
    

            
                
                

        
        
        
#Construct an instance of the TravelingSalesmanProblem
#test_cities = [('DC', (11, 1)), ('SF', (0, 0)), ('PHX', (2, -3)), ('LA', (0, -4))]
#tsp = TravelingSalesmanProblem(test_cities)
#assert(tsp.path == test_cities)
#
## Test the successors() method -- no output means the test passed
#print(list(tsp.successors()))
#successor_paths = [x.path for x in tsp.successors()]
#assert(all(x in [[('LA', (0, -4)), ('SF', (0, 0)), ('PHX', (2, -3)), ('DC', (11, 1))],
#                 [('SF', (0, 0)), ('DC', (11, 1)), ('PHX', (2, -3)), ('LA', (0, -4))],
#                 [('DC', (11, 1)), ('PHX', (2, -3)), ('SF', (0, 0)), ('LA', (0, -4))],
#                 [('DC', (11, 1)), ('SF', (0, 0)), ('LA', (0, -4)), ('PHX', (2, -3))]]
#          for x in successor_paths))
#
## Test the get_value() method -- no output means the test passed
#assert(np.allclose(tsp.get_value(), -28.97, atol=1e-3))

# These are presented as globals so that the signature of schedule()
# matches what is shown in the AIMA textbook; you could alternatively
# define them within the schedule function, use a closure to limit
# their scope, or define an object if you would prefer not to use
# global variables
alpha = 0.95
temperature=1e4

def schedule(time):
    #return temperature - alpha * time 
    return alpha**(time) * temperature
	
# test the schedule() function -- no output means that the tests passed
#assert(np.allclose(alpha, 0.95, atol=1e-3))
#assert(np.allclose(schedule(0), temperature, atol=1e-3))
#assert(np.allclose(schedule(10), 5987.3694, atol=1e-3))
#
## Failure implies that the initial path of the test case has been changed
#assert(tsp.path == [('DC', (11, 1)), ('SF', (0, 0)), ('PHX', (2, -3)), ('LA', (0, -4))])
#result = simulated_annealing(tsp, schedule)
#print("Initial score: {}\nStarting Path: {!s}".format(tsp.get_value(), tsp.path))
#print("Final score: {}\nFinal Path: {!s}".format(result.get_value(), result.path))
#assert(tsp.path != result.path)
#assert(result.get_value() > tsp.get_value())
#
# Create the problem instance and plot the initial state
#num_cities = 5
#capitals_tsp = TravelingSalesmanProblem(capitals_list[:num_cities])
#starting_city = capitals_list[0]
#print("Initial path value: {:.2f}".format(-capitals_tsp.get_value()))
#print(capitals_list[:num_cities])  # The start/end point is indicated with a yellow star
##show_path(capitals_tsp.coords, starting_city)
#
#
## set the decay rate and initial temperature parameters, then run simulated annealing to solve the TSP
#alpha = 0.95
#temperature=1e10
#result = simulated_annealing(capitals_tsp, schedule)
#print("Final path length: {:.2f}".format(-result.get_value()))
#print(result.path)
#show_path(result.coords, starting_city, w=4, h=3)



if __name__ == "__main__":
    
#    num_cities = 10
#    population_size = 100
#    evolution_cycles = 100
#    starting_city = capitals_list[0]
#    cities = capitals_list[:num_cities]
#    tsp = TravelingSalesmanProblem(cities)
#    show_path(tsp.coords, starting_city, w=4, h=3)
#    population = SalesmanPopulation([tsp.shuffle() for _ in range(population_size)])
#    print(population.averageFitness())
#    
#    for i in range(evolution_cycles):
#        population = population.evolve()
#        print(i, population.averageFitness())
#        
#    fittest_path = population.mostFitIndividual()
#    print('Fittest individual has fitness of', fittest_path.fitness())
#    show_path(fittest_path.coords, starting_city, w=4, h=3)

    num_cities = 10
    capitals_tsp = TravelingSalesmanProblem(capitals_list[:num_cities])
    starting_city = capitals_list[0]
    #print("Initial path value: {:.2f}".format(-capitals_tsp.get_value()))
    alpha = 0.97
    temperature=1e10
    result = simulated_annealing(capitals_tsp, schedule)
    print("Final path length: {:.2f}".format(-result.get_value()))
    #print(result.path)
    show_path(result.coords, starting_city, w=4, h=3)

