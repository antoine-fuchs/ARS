import os
import neat

def write_config(path):
    # Beispiel-Konfiguration für NEAT (XOR-Problem)
    config_content = '''
[NEAT]
fitness_criterion     = max
fitness_threshold     = 3.9
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
# Node activation options
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

# Node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# Genom structural mutation rates
num_hidden              = 0
num_inputs              = 2
num_outputs             = 1
initial_connection      = full

# Gene mutation rates
weight_mutate_rate        = 0.8
weight_replace_rate       = 0.1
weight_mutate_power       = 0.5
conn_add_prob             = 0.05
conn_delete_prob          = 0.03
node_add_prob             = 0.03
node_delete_prob          = 0.01

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
'''.lstrip()
    with open(path, 'w') as f:
        f.write(config_content)


def eval_genomes(genomes, config):
    # Trainingsdaten für das XOR-Problem
    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [0.0, 1.0, 1.0, 0.0]

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 4.0
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)[0]
            genome.fitness -= (output - xo) ** 2


def run_xor():
    config_path = 'config-xor'
    write_config(config_path)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    p = neat.Population(config)
    # Reporter für Konsole und Statistiken
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Lauf bis zum Fitnessziel oder Max-Generationen
    winner = p.run(eval_genomes, n=50)

    print('\nBestes Genom:\n{!s}'.format(winner))

if __name__ == '__main__':
    run_xor()
