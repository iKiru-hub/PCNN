import random
from numpy import array, around, ndarray
from numpy.random import choice as npchoice
from deap import base, creator, tools
from tools.utils import logger
import os, json, time



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

    # get parameters
    PARAMETERS = parent_1.keys()

    # A simple crossover: swap half of the parameters
    for key in random.sample(list(PARAMETERS), k=len(PARAMETERS) // 2):
        parent_1[key], parent_2[key] = parent_2[key], parent_1[key]
    return parent_1, parent_2

def mutate(parent: dict, toolbox: object):

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

    # get parameters
    PARAMETERS = parent.keys()

    # Mutate a random parameter
    key = random.choice(list(PARAMETERS))
    parent[key] = getattr(toolbox, key)() # call the sampling function
    return parent,


# -----------------------------------------
# ---| Objects |---
# -----------------


class Agent:

    def __init__(self, genome: dict, Model: object):

        """
        An agent.

        Parameters
        ----------
        genome : dict
            The genome.
        Model : object
            The model class.
        """

        # unpack the genome and create the model
        self.genome = genome.copy() 

        # exec callable parameters 
        # NB: the callable parameters are not evolved
        # NB: the callable parameters are not passed to the model
        for k, v in self.genome.items():
            if callable(v):
                genome[k] = v()

        # model instance
        self.model = Model(**genome)

        self.id = ''.join(npchoice(list(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ), 5))

    def __getitem__(self, key):
        return self.genome[key]

    def __setitem__(self, key, value):
        self.genome[key] = value

    def __str__(self):
        return str(self.genome.copy())

    def __repr__(self):
        return f"NetworkSimple(id={self.id}, N={self.genome['N']})"

    def step(self, x: ndarray) -> ndarray:

        """
        Step the agent.

        Parameters
        ----------
        x : np.ndarray
            The input.

        Returns
        -------
        y : np.ndarray
            The output.
        """

        self.model.step(x)

    def reset(self):
        self.model.reset()



# -----------------------------------------
# ---| Toolbox |---
# -----------------


def make_toolbox(PARAMETERS: dict,
                 game: object,
                 model: object,
                 strategy: object=None,
                 FIXED_PARAMETERS: dict=None,
                 fitness_weights: tuple=(1.0,)) -> object:

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
    Model : object
        The model class to use.
    FIXED_PARAMETERS : dict, optional
        The fixed parameters.
        The default is None.

    Returns
    -------
    toolbox : object
        The toolbox object from the DEAP library.
    """

    # --| Model |--

    # check the methods
    if not hasattr(model, "step"):
        raise AttributeError("Model must have a 'step' method.")
    if not hasattr(model, "reset"):
        raise AttributeError("Model must have a 'reset' method.")
    if not hasattr(model, "output"):
        raise AttributeError("Model must have an 'out' method.")

    if FIXED_PARAMETERS is not None:
        # embed fixed parameters in genome
        for k, v in FIXED_PARAMETERS.items():
            PARAMETERS[k] = lambda v=v: v

        logger.info(f"<fixed parameters>: % {tuple(FIXED_PARAMETERS.keys())}")

    # Create the DEAP creator
    creator.create("FitnessMax", base.Fitness, weights=fitness_weights)
    creator.create("Individual", dict, fitness=creator.FitnessMax)

    # Create the toolbox
    toolbox = base.Toolbox()

    # Register each sampling function
    for k, v in PARAMETERS.items():
        toolbox.register(k, v)

    logger.info(f"<parameters> registered")

    # Function to create a complete individual
    def create_individual():
        return {
            k: getattr(toolbox, k)() for k in PARAMETERS
        }

    toolbox.register("individual", tools.initIterate, creator.Individual,
                     create_individual)

    # Register the strategy
    if strategy is not None:
        toolbox.register("generate", strategy.generate, creator.Individual,
                         create_individual)
        toolbox.register("update", strategy.update)
        logger.info(f"<strategy> registered")

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    logger.info(f"<individual and population> registered")

    # Register the offspring & selections functions
    toolbox.register("mate", mate)

    def mutateWrapper(individual):
        return mutate(individual, toolbox)

    toolbox.register("mutate", mutateWrapper)
    toolbox.register("select", tools.selTournament,
                     tournsize=3)

    logger.info(f"<mate, mutate and select> registered")

    # Register the evaluation function
    def evalModel(genome: dict, game: object, Model: object):

        model = Agent(genome, Model)
        fitness = game.run(model)
        return fitness

    evalWrapper = lambda individual: evalModel(individual, 
                                               game,
                                               model)

    toolbox.register("evaluate", evalWrapper)

    logger.info(f"<evaluate> registered")
    logger.info(f"<toolbox> created")

    return toolbox


# main function
def main(toolbox: object, settings: dict, seed: int=None, save: bool=False, **kwargs):

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
    save : bool, optional
        Whether to save the best individual. 
        The default is False.
    **kwargs : dict, optional
        filename : str
            The filename. If None, nothing is saved.
    """

    if seed is not None:
        random.seed(seed)

    if save:
        if "filename" in kwargs:
            filename = kwargs["filename"]
            logger.info(f"Saving best individual as {filename}.json in cache folder.")
        else:
            logger.warning("No filename provided, no individual will be saved.")
            save = False

    # -------------------------------- #

    # Parameters for the genetic algorithm
    NPOP = settings["NPOP"]
    NGEN = settings["NGEN"]
    CXPB = settings["CXPB"]
    MUTPB = settings["MUTPB"]
    NLOG = settings["NLOG"]
    TARGET = settings["TARGET"] if "TARGET" in settings else None
    TARGET_ERROR = settings["TARGET_ERROR"] if "TARGET_ERROR" in settings else 0.

    # Create the initial population
    population = toolbox.population(n=NPOP)

    logger.info(f"--| Evolution |--\n{NPOP=}\n{NGEN=}\n{CXPB=}\n{MUTPB=}\n{NLOG=}")
    logger.info(f"Target: {TARGET} [+/- {TARGET_ERROR}]")

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
            # print(dir(ind))

        # Replace the old population by the offspring
        population[:] = offspring

        # Check the progress
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        best_ind = tools.selBest(population, 1)[0]

        if gen % NLOG == 0:
            logger.info(f"Gen {gen} Score: {around(array(best_ind.fitness.values), 5)}")

            # save the best individual
            if save:

                # save 
                save_best_individual(best_ind=best_ind, 
                                     filename=filename,
                                     path=None)

        if TARGET is not None:
            error = (TARGET - around(array(best_ind.fitness.values), 4))**2
            if (error < TARGET_ERROR).all():
                logger.info(f"Target reached {error:}. Stopping evolution")
                break

    logger.info(f"Best fitness: {around(array(best_ind.fitness.values), 3)}")

    return best_ind


# save the best individual as json
def save_best_individual(best_ind: dict, filename: str, 
                         path: str=None, verbose: bool=False):

    """
    Save the best individual as json. 

    Parameters
    ----------
    best_ind : dict
        The best individual.
    filename : str
        The filename.
    path : str, optional
        The path. If None, use the current working directory.
    verbose : bool, optional
        Whether to print the path and filename. 
        The default is False.
    """

    # add .json extension
    if not filename.endswith(".json"):
        filename += ".json"

    if path is None:
        path = os.getcwd()

        # check if there is a folder named "cache"
        # if not, create it
        if not os.path.exists(os.path.join(path, "cache")):
            os.mkdir(os.path.join(path, "cache"))
            logger.info(f"Created folder {os.path.join(path, 'cache')}")

        path = os.path.join(path, "cache")

    with open(os.path.join(path, filename), 'w') as f:
        json.dump(best_ind, f)

    if verbose:
        logger.info(f"Best individual saved as {filename} in {path}.")


def load_best_individual(filename: str=None, path: str=None):

    """
    Load the best individual as json. 

    Parameters
    ----------
    filename : str
        The filename. If None, the list of available files is printed.
    path : str, optional
        The path. If None, use the current working directory.

    Returns
    -------
    best_ind : dict
        The best individual.
    """

    if path is None:
        path = os.getcwd()
        
        # check if there is a folder named "cache"
        # if not, exit
        if not os.path.exists(os.path.join(path, "cache")):
            logger.error(f"No folder named 'cache' in {path}.")
            return None

        path = os.path.join(path, "cache")

    if filename is None:

        # make a dict of available files .json 
        logger.info(f"Available files in {path}:")
        available_files = {}
        i = 0
        for f in os.listdir(path):
            if f.endswith(".json"):
                available_files[i] = f
                logger.info(f"{i}: {f}")
                i += 1

        # ask the user to choose a file
        while True:
            try:
                i = int(input("Choose a file: "))
                filename = available_files[i]
                break
            except:
                logger.info("Invalid input. Try again. [-1 to exit]]")

            # if i is -1, exit
            if i == -1:
                logger.info("Exiting.")
                return None

    # add .json extension
    if not filename.endswith(".json"):
        filename += ".json"

    with open(os.path.join(path, filename), 'r') as f:
        best_ind = json.load(f)

    logger.info(f"Best individual loaded from {filename} in {path}.")

    return best_ind



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

    FIXED_PARAMETERS = {
        'char1': 'm',
    }

    class Agent(object):

        def __init__(self, genome: dict):
            self.genome = genome

            for k, v in genome.items():
                if callable(v):
                    self.genome[k] = v()

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
                           FIXED_PARAMETERS=FIXED_PARAMETERS)

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

