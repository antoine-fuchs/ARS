# Using NEAT with a fixed topology to perform the evolutionary robotics self-navigation task
# https://neat-python.readthedocs.io/en/latest/_modules/genome.html

import neat
from Simulation import evaluate_genome

def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(evaluate_genome, n=50) #n = number of generations
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    run('neat-config.txt')