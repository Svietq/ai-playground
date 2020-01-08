from genetic_algorithm import GeneticAlgorithm
import random

# Genetic algorithm to solve travelling salesman problem
# wikipedia link: https://en.wikipedia.org/wiki/Travelling_salesman_problem

NUMBER_OF_CITIES = 7


def create_individual(individual_size):
    individual = list(range(individual_size))
    random.shuffle(individual)
    return individual


ga = GeneticAlgorithm(individual_function=create_individual,
                      individual_size=NUMBER_OF_CITIES,
                      population_size=100,
                      mutation_rate=0.1,
                      max_iterations_number=1000,
                      goal_function=None,
                      fitness_function=sum)

ga.run_algorithm()
