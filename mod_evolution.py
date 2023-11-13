import random
from deap import base, creator, tools
from tools.utils import logger


# -----------------------------------------
# ---| Offspring |---
# -------------------

def mate(parent_1: dict, parent_2: dict):

    """
    Mate two individuals. This function is called by the DEAP toolbox.

    Parameters
    ----------
    parent_1 : dict
        The first parent
    parent_2 : dict
        The second parent

    Returns
    -------
    ind1 : dict
        The first offspring
    ind2 : dict
        The second offspring
    """

    # A simple crossover: swap half of the parameters
    for key in random.sample(list(PARAMETERS.keys()), k=len(PARAMETERS) // 2):
        parent_1[key], parent_2[key] = parent_2[key], parent_1[key]
    return parent_1, parent_2

def mutate(parent: dict):

    """
    Mutate an individual. This function is called by the DEAP toolbox.

    Parameters
    ----------
    parent : dict
        The individual to mutate
    toolbox : object
        The toolbox object from the DEAP library.

    Returns
    -------
    parent : dict
        The mutated individual
    """

    # Mutate a random parameter
    key = random.choice(list(PARAMETERS.keys()))
    parent[key] = getattr(toolbox, key)() # call the sampling function
    return parent,



# -----------------------------------------
# ---| Toolbox |---
# -----------------


def make_toolbox(PARAMETERS: dict,
                 game: object,
                 agent_class: object,
                 strategy: object=None,
                 FIXED_PARAMETERS: dict=None) -> object:

    """
    Create the toolbox object from the DEAP library.

    Parameters
    ----------
    PARAMETERS : dict
        The parameters to sample.
    game : object
        The game on which to evaluate the individuals.
    strategy : object, optional
        The strategy to use for the evolution.
        The default is None.
    agent_class : object
        The agent class to use.
    FIXED_PARAMETERS : dict, optional
        The fixed parameters.
        The default is None.

    Returns
    -------
    toolbox : object
        The toolbox object from the DEAP library.
    """

    if FIXED_PARAMETERS is not None:
        # embed fixed parameters in genome
        for k, v in FIXED_PARAMETERS.items():
            PARAMETERS[k] = lambda: v
        logger.info(f"<fixed parameters>: {tuple(FIXED_PARAMETERS.keys())}")

    # Create the DEAP creator
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", dict, fitness=creator.FitnessMax)

    # Create the toolbox
    toolbox = base.Toolbox()

    # Register each sampling function
    for k, v in PARAMETERS.items():
        toolbox.register(k, v)

    logger.debug(f"<parameters> registered")

    # Function to create a complete individual
    def create_individual():
        return {
            k: getattr(toolbox, k)() for k in PARAMETERS
        }

    # Register the strategy
    if strategy is not None:
        toolbox.register("generate", strategy.generate, creator.Individual,
                         create_individual)
        toolbox.register("strategy", strategy)
    else:
        toolbox.register("individual", tools.initIterate, creator.Individual,
                         create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    logger.debug(f"<individual and population> registered")

    # Register the offspring & selections functions
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament,
                     tournsize=3)

    logger.debug(f"<mate, mutate and select> registered")

    # Register the evaluation function
    def evalModel(genome: dict, game: object):
        model = agent_class(genome)
        fitness = game.run(model)
        return fitness

    evalWrapper = lambda individual: evalModel(individual, 
                                               game)

    toolbox.register("evaluate", evalWrapper)

    logger.debug(f"<evaluate> registered")

    logger.info(f"<toolbox> created")

    return toolbox


# main function
def main(toolbox: object, settings: dict, seed: int = None):

    """
    Main function for the genetic algorithm.

    Parameters
    ----------
    toolbox : object
        The toolbox object from the DEAP library.
    settings : dict
        The simulation settings.
        keys: NPOP, NGEN, CXPB, MUTPB, NLOG
    seed : int, optional
        The seed for the random number generator. 
        The default is None.
    """

    if seed is not None:
        random.seed(seed)

    # Parameters for the genetic algorithm
    NPOP = settings["NPOP"]
    NGEN = settings["NGEN"]
    CXPB = settings["CXPB"]
    MUTPB = settings["MUTPB"]
    NLOG = settings["NLOG"]

    # Create the initial population
    population = toolbox.population(n=NPOP)

    logger.info(f"Starting evolution with {NPOP} individuals")

    # -------------------------------- #

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    logger.info("Evaluated %i individuals" % len(population))

    # -------------------------------- #
    
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

        if gen % NLOG == 0:
            logger.info(f"Gen {gen} Best: {''.join([it[1] for it in best_ind.items()])} Score: {best_ind.fitness.values[0]}")
        
        # Exit if we've found a matching string
        # if ''.join(best_ind) == TARGET:
        #     break

    logger.info("Best individual is: %s\nwith fitness: %s" % (''.join([it[1] for it in best_ind.items()]), best_ind.fitness.values[0]))




if __name__ == "__main__":


    # ---| Agent |---

    PARAMETERS = {
        'char1': lambda: random.choice(list(
            "abcdefghijklmnopqrstuvwxyz ")),
        'char2': lambda: random.choice(list(
            "abcdefghijklmnopqrstuvwxyz ")),
        'char3': lambda: random.choice(list(
            "abcdefghijklmnopqrstuvwxyz ")),
        'char4': lambda: random.choice(list(
            "abcdefghijklmnopqrstuvwxyz ")),
        'char5': lambda: random.choice(list(
            "abcdefghijklmnopqrstuvwxyz ")),
        'char6': lambda: random.choice(list(
            "abcdefghijklmnopqrstuvwxyz "))
    }

    class Agent(object):

        def __init__(self, genome: dict):
            self.genome = genome

        def __getitem__(self, key):
            return self.genome[key]

        def __setitem__(self, key, value):
            self.genome[key] = value

        def __str__(self):
            return "".join([self.genome['char'+str(i+1)] for i in range(6)])

    # ---| Game |---

    class Game(object):
        def __init__(self, target):
            self.target = target

        def run(self, model):
            score = 0
            for i in range(6):
                if model['char'+str(i+1)] == self.target[i]:
                    score += 1
            return (score,)

    game = Game("modena")

    # Create the toolbox
    toolbox = make_toolbox(PARAMETERS=PARAMETERS,
                           game=game,
                           agent_class=Agent,
                           FIXED_PARAMETERS=None)

    # ---| Run |---

    settings = {
        "NPOP": 10,
        "NGEN": 300,
        "CXPB": 0.5,
        "MUTPB": 0.2,
        "NLOG": 5
    }

    main(toolbox=toolbox,
         settings=settings)
