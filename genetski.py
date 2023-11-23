import math
import random
import numpy as np
import matplotlib.pyplot as plt

def function(x,y):
    return 3*x**2 + y**4

def generate_individual():
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    return [x,y,function(x,y)]

def generate_initial_population(size_of_population):
    population = [[0,0,0]]*size_of_population
    for i in range(0,size_of_population):
        population[i] = generate_individual()
    return population

def tournament_selection(population,K):
    k_individuals = random.sample(population,K)
    a = [-1]*K
    for i in range(0,K-1):
        a[i] = k_individuals[i][2]
    best_index = np.argmin(a)
    return k_individuals[best_index]

def cumulative_sum(a):
    c_s = []
    j = 0
    for i in range(0, len(a)):
        j += a[i]
        c_s.append(j)
    return c_s

def generation_average(population):
    a = [0]*len(population)
    for i in range(len(population)):
        a[i] = population[i][2]
    return np.sum(a)/len(a)

def roulette_selection(population):
    a = [0]*len(population)
    for i in range(0,len(population)):
        a[i] = 1/(population[i][2])
    population_fitness = np.sum(a)
    cum_sum = cumulative_sum(a)
    r = random.uniform(0,float(population_fitness))
    for i in range(len(a)):
        if r < cum_sum[i]:
            return population[i]

def crossover(parent1,parent2):
    child1 = [parent1[0], parent2[1], function(parent1[0], parent2[1])]
    child2 = [parent1[1], parent2[0], function(parent1[1], parent2[0])]
    return child1,child2

def elitism(population):
    a = [0]*len(population)
    for i in range(len(population)):
        a[i] = population[i][2]
    index_of_first = np.argmin(a)
    a[index_of_first] = 1000000
    index_of_second = np.argmin(a)
    return index_of_first, index_of_second

def mutation(individual,rate):
    for i in range(len(individual)-2):
        random_number = random.random()
        if random_number < rate:
            individual[i]  = individual[i] + random.uniform(-1,1)

def algorithm(selection ,size_of_population=20, number_of_generations=100,K=3, mutation_rate=0.2, err=0.0001):
    population = generate_initial_population(size_of_population)
    best_individual_fitness = []*number_of_generations
    average_generation_fitness = []
    average_generation_fitness.append(generation_average(population))
    el1, el2 = elitism(population)
    best_individual_fitness.append(population[el1][2])
    print("Initial population: ")
    for i in range(0,len(population)):
        print(population[i])
    for i in range(1,number_of_generations-1):
        new_population = []*size_of_population
        el1,el2 = elitism(population)
        best_individual_fitness.append(population[el1][2])
        new_population.append(population[el1])
        new_population.append(population[el2])
        while len(new_population) < len(population):
            if selection == "tournament":
                parent1 = tournament_selection(population, K)
                parent2 = tournament_selection(population, K)
            else:
                parent1 = roulette_selection(population)
                parent2 = roulette_selection(population)
            child1,child2 = crossover(parent1,parent2)
            mutation(child1, mutation_rate)
            mutation(child2, mutation_rate)
            new_population.append(child1)
            new_population.append(child2)
        average_generation_fitness.append(generation_average(population))
        population = new_population
        print(" -------------- Population: " + str(i) + '------------------')
        for j in range(0,len(population)):
            print(population[j])
        if i > 11:
            if abs(best_individual_fitness[i] - best_individual_fitness[i-10]) < err:
                print(best_individual_fitness)
                return best_individual_fitness, average_generation_fitness
    print(best_individual_fitness)
    return best_individual_fitness, average_generation_fitness


best_idividuals_fitness, generation_average_fitness = algorithm("tournament", number_of_generations=50)
plt.plot(best_idividuals_fitness)
plt.plot(generation_average_fitness)
plt.show()