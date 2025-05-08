import numpy as np

"""
approach for self-navigation using evolutionary robotics:

devise genetic representation
build a population
design fitness function: phi (bearing) = V*(1- sqrt(dv))(1 - i)
choose selection method
choose crossover & mutation method
choose data analysis method
"""

#GPT

#basics of evolution
population_size = 80
population = [np.random.uniform(0, 1, genome_size) for _ in range(population_size)] #dont understand what the upper & lower bound signify here
#genome = ?
genome_size = 10 #dont understand what the genome should comprise
generations = 35
mutation_rate = 0.05

#fitness function --> how well does the robot self-navigate? phi (bearing) = V*(1- sqrt(dv))(1 - i) was used in lec
def evaluate_fitness(genome):
    total_reward = 0
    for _ in range(episodes):
        reset_environment()
        fitness = simulate_robot_with_genome(genome)
        total_reward += fitness
    return total_reward / episodes

#reproduction
def select_parents(population, fitnesses, k=2):
    # Tournament selection
    selected = []
    for _ in range(k):
        competitors = np.random.sample(list(zip(population, fitnesses)), k=3)
        winner = max(competitors, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

#cross-over & mutation
def crossover(mom, dad):
    crossover_point = np.random.randint(1, genome_size-1)
    return np.concatenate([mom[:crossover_point], dad[crossover_point:]])

def mutate(genome, mutation_rate=0.1):
    for i in range(len(genome)):
        if np.random.rand() < mutation_rate:
            genome[i] += np.random.normal(0, 0.1)
    return genome

#data analysis --> find best fitnesses
best_fitnesses = []
for gen in range(generations):
    fitnesses = [evaluate_fitness(g) for g in population]
    best_fitnesses.append(max(fitnesses))

#this might be better
for gen in range(GENERATIONS):
    fitnesses = [evaluate_fitness(ind) for ind in population]
    new_population = []

    for _ in range(POP_SIZE):
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)

    population = new_population

    best_fitness = max(fitnesses)
    best_genome = population[np.argmax(fitnesses)]
    print(f"Generation {gen}: Best Fitness = {best_fitness:.4f}, Genome = {best_genome}")