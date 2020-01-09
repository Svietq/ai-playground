from genetic_algorithm import GeneticAlgorithm
import random
import pandas

# Genetic algorithm to solve travelling salesman problem
# wikipedia link: https://en.wikipedia.org/wiki/Travelling_salesman_problem

cities_costs = pandas.read_excel('salesman_problem.xls', sheet_name='Arkusz1')
NUMBER_OF_CITIES = len(cities_costs.columns) - 1


def create_individual(individual_size):
    individual = list(range(individual_size))
    random.shuffle(individual)
    return individual


def swap_two_elements(individual):
    indices = range(len(individual))
    index1, index2 = random.sample(indices, 2)
    individual[index1], individual[index2] = individual[index2], individual[index1]


def crossover(left_individual, right_individual):
    individual_size = len(left_individual)
    pivot = random.randint(0, individual_size)
    left_individual_tail = left_individual[pivot:individual_size]
    right_individual_tail = right_individual[pivot:individual_size]
    new_left_individual_head = [x for x in left_individual if x not in right_individual_tail]
    new_right_individual_head = [x for x in right_individual if x not in left_individual_tail]
    return (new_left_individual_head + right_individual_tail),\
           (new_right_individual_head + left_individual_tail)


def get_cost_to_city(current_city_index, next_city_index):
    current_city_costs = cities_costs[current_city_index].tolist()
    return current_city_costs[next_city_index]


def calculate_fitness(individual):
    summed_cost = 0
    for index in range(len(individual)-1):
        current_city_index = individual[index]
        next_city_index = individual[index+1]
        summed_cost += get_cost_to_city(current_city_index, next_city_index)

    return -summed_cost


ga = GeneticAlgorithm(individual_function=create_individual,
                      individual_size=NUMBER_OF_CITIES,
                      population_size=100,
                      mutation_function=swap_two_elements,
                      mutation_rate=0.1,
                      max_iterations_number=1000,
                      fitness_function=calculate_fitness,
                      crossover_function=crossover)

ga.run_algorithm()

