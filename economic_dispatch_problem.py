from genetic_algorithm import GeneticAlgorithm
import numpy as np

# Genetic algorithm to solve economic dispatch problem
# link: http://pe.org.pl/articles/2014/11/53.pdf

power_plants_number = 7  # Number of power loss coefficients should be equal to power plants number
power_loss_coefficients = [0.002, 0.003, 0.002, 0.003, 0.004, 0.002, 0.004]
power_plant_bits_number = 5
power_demand = 150
minimal_power = 20
maximal_power = 40
power_tolerance = 10
fuel_cost_coefficients = [1, 2, 4]


def calculate_total_power_loss(power_plants):
    total_power_loss = 0.0

    for index, power_plant in enumerate(power_plants):
        total_power_loss += power_loss_coefficients[index] * (power_plant ** 2)

    return total_power_loss


def calculate_power_balance(power_plants):
    total_power = sum(power_plants)
    total_power_loss = calculate_total_power_loss(power_plants)

    return abs(total_power - total_power_loss - power_demand)


def calculate_fuel_cost(power_plants):
    fuel_cost = 0.0

    for power_plant in power_plants:
        fuel_cost += fuel_cost_coefficients[0] + \
                     fuel_cost_coefficients[1] * power_plant + \
                     fuel_cost_coefficients[2] * (power_plant ** 2)

    return fuel_cost


def binary_list_to_integer(binary_list):
    result = 0
    for element in binary_list:
        result = (result << 1) | element

    return result


def decimal_power_value(power_plant):
    max_power_level = binary_list_to_integer([1 for _ in range(power_plant_bits_number)])
    return minimal_power + (maximal_power - minimal_power) * binary_list_to_integer(power_plant) / max_power_level


def individual_to_decimal_values(individual):
    power_plants = np.array_split(individual, power_plants_number)
    decimal_values = []
    for power_plant in power_plants:
        decimal_values.append(decimal_power_value(power_plant))

    return decimal_values


def calculate_power_cost(individual):
    decimal_values = individual_to_decimal_values(individual)
    power_balance = calculate_power_balance(decimal_values)

    if power_balance <= power_tolerance:
        return -calculate_fuel_cost(decimal_values)
    else:
        return -power_balance*100000


ga = GeneticAlgorithm(individual_size=power_plants_number * power_plant_bits_number,
                      population_size=30,
                      mutation_rate=0.01,
                      max_iterations_number=4000,
                      fitness_function=calculate_power_cost)

best_individual = ga.run_algorithm()
print("Power values on each plant")
print(individual_to_decimal_values(best_individual))
