from genetic_algorithm import GeneticAlgorithm

# Genetic algorithm to solve economic dispatch problem
# link: http://pe.org.pl/articles/2014/11/53.pdf

power_plants_number = 5


def no_crossover(left_individual, right_individual):
    return left_individual, right_individual


ga = GeneticAlgorithm(individual_size=32,
                      population_size=power_plants_number*2,
                      mutation_rate=0.1,
                      max_iterations_number=1000,
                      crossover_function=no_crossover)
ga.run_algorithm()
