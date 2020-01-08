import numpy as np
from matplotlib import pyplot as plt
import pprint

# Exemplary code for setting up and running genetic algorithm
# ga = GeneticAlgorithm(individual_size=10,
#                       population_size=100,
#                       mutation_rate=0.1,
#                       max_iterations_number=1000)
# ga.run_algorithm()


def default_individual_creation(individual_size):
    return np.random.randint(2, size=individual_size)


def default_mutation(individual):
    random_index = np.random.randint(0, len(individual))
    individual[random_index] = int(not individual[random_index])


def default_crossover(left_individual, right_individual):
    individual_size = len(left_individual)
    pivot = np.random.randint(0, individual_size)
    return np.concatenate([left_individual[0:pivot], right_individual[pivot:individual_size]]), \
           np.concatenate([right_individual[0:pivot], left_individual[pivot:individual_size]])


class GeneticAlgorithm:
    def __init__(self,
                 individual_function=default_individual_creation,
                 individual_size=10,
                 population_size=100,
                 mutation_function=default_mutation,
                 mutation_rate=0.1,
                 max_iterations_number=1000,
                 goal_function=None,
                 fitness_function=sum,
                 crossover_function=default_crossover):
        self.individual_size = individual_size
        self.Population = [individual_function(individual_size) for _ in range(population_size)]
        self.mutation_function = mutation_function
        self.mutation_rate = mutation_rate
        self.iterations_number = 0
        self.max_iterations_number = max_iterations_number
        self.goal_function = goal_function
        self.fitness_function = fitness_function
        self.fitness_values = []
        self.crossover_function = crossover_function

    def is_goal_met(self):
        return ((self.goal_function is not None) and self.goal_function()) or \
                (self.iterations_number >= self.max_iterations_number)

    def calculate_population_fitness(self):
        population_fitness = 0
        for individual in self.Population:
            population_fitness += self.calculate_fitness(individual)

        return population_fitness

    def calculate_fitness(self, individual):
        return self.fitness_function(individual)

    def select_individuals(self):
        # ranking method:
        left_individuals = self.Population[0::2]
        right_individuals = self.Population[1::2]
        assert(len(left_individuals) == len(right_individuals))

        for index in range(len(left_individuals)):
            left_fitness_value = self.calculate_fitness(left_individuals[index])
            right_fitness_value = self.calculate_fitness(right_individuals[index])
            left_individual = right_individuals[index] if left_fitness_value < right_fitness_value else left_individuals[index]
            right_individual = left_individuals[index] if left_fitness_value > right_fitness_value else right_individuals[index]
            self.Population[index*2] = left_individual
            self.Population[index*2+1] = right_individual

    def crossover_individuals(self):
        left_individuals = self.Population[0::2]
        right_individuals = self.Population[1::2]
        assert(len(left_individuals) == len(right_individuals))

        for index in range(len(left_individuals)):
            left_individual = left_individuals[index]
            right_individual = right_individuals[index]
            left_individual, right_individual = self.crossover_function(left_individual, right_individual)
            self.Population[index * 2] = left_individual
            self.Population[index * 2 + 1] = right_individual

    def mutate_individuals(self):
        for individual in self.Population:
            if np.random.uniform() < self.mutation_rate:
                self.mutation_function(individual)

    def print_stats(self):
        # print("---------------------------")
        # print("Iteration: " + str(self.iterations_number))
        population_fitness = self.calculate_population_fitness()
        # print("Total population fitness value: " + str(self.calculate_population_fitness()))
        self.fitness_values.append(population_fitness)

    def plot_stats(self):
        iterations = np.arange(0, self.iterations_number)
        plt.title("Fitness values change")
        plt.xlabel("iteration")
        plt.ylabel("fitness value")
        plt.plot(iterations, self.fitness_values)
        plt.show()

    def run_algorithm(self):
        while not self.is_goal_met():
            self.select_individuals()
            self.crossover_individuals()
            self.mutate_individuals()
            self.iterations_number += 1
            self.print_stats()

        print("Population found: ")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.Population)

        self.plot_stats()
