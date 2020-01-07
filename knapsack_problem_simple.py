import numpy as np

ITEMS_NUMBER = 100
items_weights = np.random.randint(20, size=ITEMS_NUMBER)

backpack_weight = 675
threshold = 2


def mutate(chromosome):
    elem = np.random.randint(0, len(chromosome))
    if chromosome[elem] == 1:
        chromosome[elem] = 0
    else:
        chromosome[elem] = 1

    return chromosome


def calc_weight(chromosome):
    bits_sum = 0
    n = 0
    for bit in chromosome:
        bits_sum += bit * items_weights[n]
        n += 1

    return bits_sum


def calc_fit(chromosome):
    total_weight = calc_weight(chromosome)
    epsilon = backpack_weight - total_weight
    if epsilon >= 0:
        return epsilon
    else:
        return backpack_weight - epsilon


def select(left_chromosome, right_chromosome):
    left_fit_value = calc_fit(left_chromosome)
    right_fit_value = calc_fit(right_chromosome)

    if left_fit_value <= right_fit_value:
        return left_chromosome
    else:
        return right_chromosome


def print_stats(chromosome, iteration):
    print("--------------------------------")
    print("Iteration: " + str(iteration))
    print("Fitness value: " + str(calc_fit(chromosome)))
    print("Total weight of items: " + str(calc_weight(chromosome)))
    print("Backpack capacity: " + str(backpack_weight))


def run_algorithm(parent):
    n = 0
    while calc_fit(parent) > threshold and n < 1000:
        child = np.copy(parent)
        child = mutate(child)
        parent = select(parent, child)
        print_stats(parent, n)
        n += 1


run_algorithm(np.random.randint(2, size=ITEMS_NUMBER))
