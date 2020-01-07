from genetic_algorithm import GeneticAlgorithm
import numpy as np

# Genetic algorithm to solve knapsack problem
# wikipedia link: https://en.wikipedia.org/wiki/Knapsack_problem

ITEMS_NUMBER = 50
ITEM_MAX_WEIGHT = 200
items_weights = np.random.randint(ITEM_MAX_WEIGHT, size=ITEMS_NUMBER)
TOTAL_MAX_WEIGHT = sum(items_weights)
BACKPACK_CAPACITY = 2000
THRESHOLD = 1


def calc_weight(chromosome):
    bits_sum = 0
    n = 0
    for bit in chromosome:
        bits_sum += bit * items_weights[n]
        n += 1

    return bits_sum


def calc_fit(chromosome):
    total_weight = calc_weight(chromosome)
    epsilon = BACKPACK_CAPACITY - total_weight
    if epsilon >= 0:
        return TOTAL_MAX_WEIGHT - epsilon
    else:
        return TOTAL_MAX_WEIGHT - (BACKPACK_CAPACITY - epsilon)


print("Items weigths: ", items_weights)
print("TOTAL_MAX_WEIGHT: ", TOTAL_MAX_WEIGHT)
print("backpack_capacity: ", BACKPACK_CAPACITY)
ga = GeneticAlgorithm(individual_size=ITEMS_NUMBER,
                      population_size=10,
                      mutation_rate=0.01,
                      max_iterations_number=1000,
                      goal_function=None,
                      fitness_function=calc_fit)
ga.run_algorithm()


def print_found_weights(population):
    print("Found weights: ")
    for individual in population:
        print(calc_weight(individual))


print_found_weights(ga.Population)
