import numpy as np
from matplotlib import pyplot as plt

# Exemplary code for setting up and running genetic algorithm
# ga = GeneticAlgorithm(individual_size=10,
#                       population_size=100,
#                       mutation_rate=0.1,
#                       max_iterations_number=1000
#                       goal_function = None
#                       fitness_function=sum)
# ga.run_algorithm()


class GeneticAlgorithm:
    def __init__(self, individual_size, population_size, mutation_rate, max_iterations_number, goal_function, fitness_function):
        self.individual_size = individual_size
        self.Population = np.random.randint(2, size=(population_size, individual_size))
        self.mutation_rate = mutation_rate
        self.iterations_number = 0
        self.max_iterations_number = max_iterations_number
        self.goal_function = goal_function
        self.fitness_function = fitness_function
        self.fitness_values = []

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
            pivot = np.random.randint(0, self.individual_size)
            left_individual = left_individuals[index]
            right_individual = right_individuals[index]
            left_individual, right_individual = np.concatenate([left_individual[0:pivot], right_individual[pivot:self.individual_size]]), \
                                                np.concatenate([right_individual[0:pivot], left_individual[pivot:self.individual_size]])

            self.Population[index * 2] = left_individual
            self.Population[index * 2 + 1] = right_individual

    def mutate_individuals(self):
        for individual in self.Population:
            if np.random.uniform() < self.mutation_rate:
                random_index = np.random.randint(0, self.individual_size)
                individual[random_index] = int(not individual[random_index])

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
        print(self.Population)
        self.plot_stats()
