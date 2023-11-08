import random
from deap import base, creator, tools

# Define the target string to evolve to and the gene pool
TARGET = "Hello, World!"
GENE_POOL = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!, ")

# Define the fitness function
def evaluate(individual):
    return sum(ch1 == ch2 for ch1, ch2 in zip(individual, TARGET)),

# Define the types
creator.create("FitnessMax", base.Fitness, weights=(0.8,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize the toolbox
toolbox = base.Toolbox()
toolbox.register("attr_char", random.choice, GENE_POOL)
toolbox.register("individual", tools.initRepeat, 
                 creator.Individual,
                 toolbox.attr_char, len(TARGET))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutSh2ffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=4)

def main():
    # random.seed(42)

    # Create the initial population
    population = toolbox.population(n=200)

    # Parameters for the genetic algorithm
    NGEN = 10_000
    CXPB, MUTPB = 0.5, 0.3

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolve the population
    for gen in range(NGEN):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace the old population by the offspring
        population[:] = offspring
        
        # Check the progress
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        best_ind = tools.selBest(population, 1)[0]

        if gen % 500 == 0:
            print(f"Gen {gen} Best: {''.join(best_ind)} Score: {best_ind.fitness.values[0]}")
        
        # Exit if we've found a matching string
        if ''.join(best_ind) == TARGET:
            break

    print("Best individual is: %s\nwith fitness: %s" % (''.join(best_ind), best_ind.fitness.values))

if __name__ == "__main__":
    main()
