import numpy as np

#can use PyTorch, sklearn, NEAT or just numpy
# https://neat-python.readthedocs.io/en/latest/neat_overview.html

"""
approach for self-navigation using evolutionary robotics:

devise genetic representation
build a population
design fitness function
choose selection method
choose crossover & mutation method
choose data analysis method
"""

#use pytorch or sklearn for nn implementation
#input: sensor data (how far walls are)
#genome (size) = weights of nn
#fitness function --> how well does the robot self-navigate? --> measures in the same generation how close the robot got to the rescue ball
#output: driving the robot, speed value for left & right wheel
#use the weights of best robot and then run new generation with that genome

#Evolutionary robotics using NumPy

import numpy as np
import random

class FixedNeuralNetwork:
    def __init__(self, weights):
        # weights = concatenated 1D array of all network weights
        self.input_size = 4 #distances to left, right, top & bottom wall
        self.hidden_size = 5 #can be adjusted
        self.output_size = 2 #left & right wheel speeds

        w1_end = self.input_size * self.hidden_size
        w2_end = w1_end + self.hidden_size * self.output_size

        self.W1 = weights[:w1_end].reshape((self.input_size, self.hidden_size))
        self.W2 = weights[w1_end:w2_end].reshape((self.hidden_size, self.output_size))

    def activate(self, x):
        x = np.array(x)
        h = np.tanh(np.dot(x, self.W1))  # hidden layer activation
        out = np.tanh(np.dot(h, self.W2))  # output layer activation
        return out

genome_size = (self.input_size * self.hidden_size) + (self.hidden_size * self.output_size) #number of weights
population_size = 25 #per generation
generations = 30
mutation_rate = 0.05
elitism = 0.2 #preserves the top individuals by copying them into the next gen

#generate a random genome with random weights
def random_genome():
    return np.random.uniform(-1, 1, genome_size) #or shoudl we do 0 to 1

#mutate & cross-over to mix up genome
def mutate(genome):
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            genome[i] += np.random.normal(0, 0.1)
    return genome

def crossover(mom, dad):
    crossover_point = np.random.randint(1, genome_size-1)
    return np.concatenate([mom[:crossover_point], dad[crossover_point:]])

#evaluate performance of genome: higher fitness indicates closer to rescue target
def evaluate_genome(genome, grid):
    network = FixedNeuralNetwork(genome)
    fitness = 0
    reset_simulation()

    for _ in range(300):  # steps per genome
        sensor_inputs = get_sensor_inputs()  # returns list of 4 wall distances
        output = network.activate(sensor_inputs)
        left_wheel_speed = output[0]
        right_wheel_speed = output[1]
        move_robot(left_wheel_speed, right_wheel_speed)
        update_simulation()

        if reached_target():
            fitness += 1000
            break
        else:
            fitness += compute_progress_toward_goal()

    return fitness

# survival of the fittest
def evolve_population(grid):
    population = [random_genome() for _ in range(population_size)]

    for gen in range(generations):
        genome_scores = [(genome, evaluate_genome(genome, grid)) for genome in population] #returns the genomes with their corresponding fitness
        genome_scores.sort(key=lambda x: x[1], reverse=True)
        best = genome_scores[0][1]
        print(f"Generation {gen + 1}: Best fitness = {best}")

        elites_size = int(elitism*population_size) #how many elitists we want
        elites = [genome for genome, fitness in genome_scores[:elites_size]] #takes the genomes of these elites
        new_population = elites[:] #copy the elites

        while len(new_population) < population_size:
            mom, dad = random.sample(elites, 2)
            child = crossover(mom, dad)
            new_population.append(child)

        population = new_population


#need to code these:
#reset_simulation() – resets robot, map, etc.
#get_sensor_inputs() – returns the 4 wall distance values
#move_robot(left_speed, right_speed)
#update_simulation() – runs a time step of your simulation
#compute_progress_toward_goal() – reward based on distance to target

"""
#or would smth like this be better
for gen in range(generations):
    fitnesses = evaluate_fitness(genome)
    new_population = []

    for _ in range(population_size):
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)

    population = new_population

    best_fitness = max(fitnesses)
    best_genome = population[np.argmax(fitnesses)]
    print(f"Generation {gen}: Best Fitness = {best_fitness:.4f}, Genome = {best_genome}")
"""