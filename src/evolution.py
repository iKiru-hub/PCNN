import random
from numpy import array, around, ndarray
import numpy as np
from numpy.random import choice as npchoice
import matplotlib.pyplot as plt
from deap import base, creator, tools
import multiprocessing
from utils import logger, clf
import os, json, time, tqdm

from utils import setup_logger
logger = setup_logger(__name__)

# check if ipython is running
def is_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

IS_IPYTHON = is_ipython()

USE_TQDM = False
FORCE_POOL = False


""" Offspring """

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


""" Objects """


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

    def get_genome(self):
        return self.genome.copy()

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


class Model:

    """
    A model class.
    """

    def __init__(self, **kwargs):

        """
        Initialize the model.

        Parameters
        ----------
        kwargs : dict
            The parameters.
        """

        pass

    def __name__(self):

        return "BaseModel"

    def __repr__(self):

        return "BaseModel"

    def step(self, **kwargs):

        """
        Step the model.

        Parameters
        ----------
        kwargs : dict
            The parameters.
        """

        pass 

    @property
    def output(self):

        pass

    def reset(self):

        """
        Reset the model.
        """

        pass


""" Toolbox """


def _toolbox_evaluate(individual: dict, model: object, game: object):

    """
    Evaluate an individual.

    Parameters
    ----------
    individual : dict
        The individual to evaluate.
    model : object
        The toolbox object from the DEAP library.
    game : object
        The game on which to evaluate the individual.

    Returns
    -------
    fitness : tuple
        The fitness.
    """

    model = Agent(individual, model)
    fitness = game.run(model)
    return fitness


def _evaluate_population(population: list, game: object, 
                         model: object, num_cores=1) -> list:

    """
    Evaluate a population.

    Parameters
    ----------
    population : list
        The population to evaluate.
    game : object
        The game on which to evaluate the individuals.
    model : object
        The model class to use.
    num_cores : int, optional
        The number of cores to use. The default is 1.

    Returns
    -------
    fitnesses : list
        The fitnesses.
    """

    if (num_cores > 1 or (FORCE_POOL and num_cores == 1)) and not USE_TQDM:
        with multiprocessing.Pool(num_cores) as pool:
            fitnesses = pool.starmap(_toolbox_evaluate,
                        [(ind, model, game) for ind in population])
    elif num_cores >= 1:

        logger.warning("# bewarned: no multiprocessing is being used")

        if USE_TQDM:
            fitnesses = list(tqdm.tqdm(map(lambda ind: _toolbox_evaluate(ind, model, game), population), total=len(population)))

        # fitnesses = [_toolbox_evaluate(
        #     individual=individual,
        #     model=model,
        #     game=game) for individual in population]
    else:
        raise ValueError("num_cores must be a positive integer")

    return fitnesses


def make_toolbox(PARAMETERS: dict,
                 game: object,
                 model: object,
                 strategy: object=None,
                 FIXED_PARAMETERS: dict=None,
                 fitness_weights: tuple=(1.0, 1.0), **kwargs) -> object:

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
    fitness_weights : tuple, optional
        The fitness weights.
        The default is (1.0, 1.0).
    **kwargs : dict, optional
        verbose : bool, optional
            Whether to print the toolbox. 
            The default is True.

    Returns
    -------
    toolbox : object
        The toolbox object from the DEAP library.
    """

    verbose = kwargs["verbose"] if "verbose" in kwargs else True

    # --| Model |--

    # check the methods
    if not hasattr(model, "step"):
        raise AttributeError("Model must have a 'step' method.")
    if not hasattr(model, "reset"):
        raise AttributeError("Model must have a 'reset' method.")
    if not hasattr(model, "output"):
        raise AttributeError("Model must have an `output` method.")

    if FIXED_PARAMETERS is not None:
        # embed fixed parameters in genome
        for k, v in FIXED_PARAMETERS.items():
            PARAMETERS[k] = lambda v=v: v

        if verbose:
            logger.info(f"%fixed parameters : {tuple(FIXED_PARAMETERS.keys())}")

    # Create the DEAP creator
    creator.create("FitnessMax", base.Fitness, weights=fitness_weights)
    creator.create("Individual", dict, fitness=creator.FitnessMax)
    if verbose:
        logger.info(f"%fitness created {fitness_weights}")

    # Create the toolbox
    toolbox = base.Toolbox()

    # Register each sampling function
    for k, v in PARAMETERS.items():
        toolbox.register(k, v)

    if verbose:
        logger.info(f"%parameters registered")

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
        if verbose:
            logger.info(f"%strategy registered")

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if verbose:
        logger.info(f"%individual & population registered")

    # Register the offspring & selections functions
    toolbox.register("mate", mate)

    def mutateWrapper(individual):
        return mutate(individual, toolbox)

    toolbox.register("mutate", mutateWrapper)
    toolbox.register("select", tools.selTournament,
                     tournsize=3)

    if verbose:
        logger.info(f"%mate & mutate & select registered")

    # # Register the evaluation function
    # def evalModel(genome: dict, game: object, Model: object):

    #     model = Agent(genome, Model)
    #     fitness = game.run(model)
    #     return fitness

    # evalWrapper = lambda individual: evalModel(individual, 
    #                                            game,
    #                                            model)

    # def evalWrapper(individual):
    #     return evalModel(individual, game, model)

    # toolbox.register("evaluate", evalWrapper)

    # # register the game
    # toolbox.game = game
    # toolbox.register("model", model)

    # Register the evaluation function
    def evalModel(genome: dict, game: object, Model: object):
        model = Agent(genome, Model)
        fitness = game.run(model)
        return fitness

    def evalWrapper(individual):
        return evalModel(individual, toolbox.game, toolbox.model)

    toolbox.register("evaluate", evalWrapper)

    # Assign the game and model directly to the toolbox
    toolbox.game = game
    toolbox.model = model

    if verbose:
        logger.info(f"%evaluate registered")
        logger.info(f"%toolbox created")

    return toolbox


# main function
def main(toolbox: object, settings: dict, seed: int=None,
         save: bool=False, visualizer: object=None, **kwargs):

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
    visualizer : object, optional
        The visualizer object. Default is None.
    **kwargs : dict, optional
        NUM_CORES : int, optional
            The number of cores to use.
            The default is 1.
        filename : str
            The filename. If None, nothing is saved.
        info : dict
            Additional information to save. Default is {}.
        verbose : bool, optional
            Whether to print the best individual.
            The default is False.
        max_duration : int, optional
            The maximum duration of the evolution in minutes.
            Default is None.
        perc_to_save : float, optional
            The percentage of individuals to save [0, 1].
            Default is 0.1.
    """

    # settings
    verbose = kwargs["verbose"] if "verbose" in kwargs else False
    path = kwargs["path"] if "path" in kwargs else None
    if "max_duration" in kwargs:
        max_duration = kwargs["max_duration"]*60 if kwargs["max_duration"] is not None else None
    else:
        max_duration = None
    save_figure = kwargs["save_figure"] if "save_figure" in kwargs else False
    perc_to_save = kwargs.get("perc_to_save", 0.1)
    logger.info(f"% to save: {perc_to_save}")
    perc_to_save = int(perc_to_save * settings["NPOP"])

    if seed is not None:
        random.seed(seed)

    if save:
        if "filename" in kwargs:
            filename = kwargs["filename"]
            if verbose:
                logger.info(
                    f"Saving best individual as `{filename}.json` in cache folder.")
        else:
            if verbose:
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
    NUM_CORES = settings["NUM_CORES"] if "NUM_CORES" in settings else 1

    # check if the number of cores is valid
    if NUM_CORES > multiprocessing.cpu_count():
        logger.warning(f"NUM_CORES ({NUM_CORES}) > cpu_count ({multiprocessing.cpu_count()}). Setting NUM_CORES = cpu_count.")
        NUM_CORES = multiprocessing.cpu_count()

    if verbose:
        try:
            logger.info(f"Evolution settings: {NPOP=}, {NGEN=}, {CXPB=}, {MUTPB=}, {NLOG=}, {NUM_CORES=}")
        except SyntaxError:
            logger.info(f"Evolution settings: pop={NPOP}, gen={NGEN}, {CXPB}, {MUTPB}, {NLOG}, cores={NUM_CORES}")
        logger.info(f"Target: {TARGET} [+/- {TARGET_ERROR}]")
        if max_duration is not None:
            logger.info(f"Max duration: {max_duration//60}min")

    # -------------------------------- #

    # Create the initial population
    population = toolbox.population(n=NPOP)
    model = toolbox.model
    game = toolbox.game

    # -------------------------------- #

    # Evaluate the entire population
    fitnesses = _evaluate_population(population=population,
                                     game=game,
                                     model=model,
                                     num_cores=NUM_CORES)

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    if verbose:
        logger.info("Gen 0 evaluated")

    # -------------------------------- #
    start_time = time.time()
    max_duration_reached = False
    fitness_record = np.zeros((5, NGEN-1))

    # Evolve the population
    for gen in range(1, NGEN):

        record_genome = {}

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
        fitnesses = _evaluate_population(population=invalid_ind,
                                         game=game,
                                         model=model,
                                         num_cores=NUM_CORES)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the old population by the offspring
        population[:] = offspring

        # Check the progress
        fits = [ind.fitness.values[0] for ind in population]
        # length = len(population)
        # mean = sum(fits) / length
        mean = np.mean(fits)

        # -------------------------------- #

        # record the genome of the top individuals
        sorted_population = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)

        logger.debug("Sorted individuals:")
        for i, ind in enumerate(sorted_population[:perc_to_save]):
            record_genome[i] = {"genome": check_genome_content(ind),
                                "fitness": ind.fitness.values}
            print(f"{i+1}: {ind.fitness.values}", end=" | ")
        print()

        best_ind = tools.selBest(population, 1)[0]

        # record fitness
        fitness_record[:, gen-1] = make_stats(fits)

        # -------------------------------- #

        # plot
        if visualizer is not None:
            visualizer.update(population=[ind.fitness.values for ind in population],
                              best_ind=best_ind)
            visualizer.plot()

        # logs
        if gen % NLOG == 0:
            if verbose:
                logger.info(
                    f"Gen {gen} Score: {around(array(best_ind.fitness.values), 5)}")

            # save the best individual
            if save:

                if "info" in kwargs:
                    info = kwargs["info"]
                    info["performance"] = {
                        "gen": str(gen),
                        "fitness": str(around(tuple(best_ind.fitness.values), 5))
                    }
                else:
                    info = {}

                # add fitness record
                info["fitness_record"] = {
                    "mean": fitness_record[0].tolist(),
                    "std": fitness_record[1].tolist(),
                    "max": fitness_record[2].tolist(),
                    "p16": fitness_record[3].tolist(),
                    "p84": fitness_record[4].tolist()
                }

                # add genome record
                info["record_genome"] = record_genome

                # save
                save_best_individual(best_ind=best_ind,
                                     filename=filename,
                                     info=info,
                                     path=path)

                if visualizer and save_figure:
                    visualizer.save_figure(filename=filename + "_plot")

        if TARGET is not None:
            error = (TARGET - around(array(best_ind.fitness.values), 4))**2
            if (error < TARGET_ERROR).all():
                if verbose:
                    logger.info(f"Target reached {error:}. Stopping evolution")
                break

        # check max duration
        if max_duration is not None:
            if time.time() - start_time > max_duration:
                if verbose:
                    logger.warning(f"Max duration reached >>> Exiting")
                max_duration_reached = True
                break

    if verbose:
        logger.info(f"Best fitness: {around(array(best_ind.fitness.values), 3)}")

    if visualizer is not None and not max_duration_reached:
        visualizer.keep_figure()

    if visualizer and save_figure:
        visualizer.save_figure(filename=filename + "_plot")

    return best_ind



""" save and load best individual """


def check_genome_content(genome: dict):

    """
    Check the content of the genome, mainly for non-serializable objects.

    Parameters
    ----------
    genome : dict
        The genome to check.

    Returns
    -------
    genome : dict
        The genome with the non-serializable objects fixed.
    """

    # convert np.ndarray to list
    for k, v in genome.items():
        if isinstance(v, np.ndarray):
            genome[k] = v.tolist()
    return genome


def check_info_dict(info: dict):

    """ convert np.ndarray to list """
    for k, v in info.items():
        if isinstance(v, np.ndarray):
            info[k] = v.tolist()
        elif isinstance(v, dict):
            info[k] = check_info_dict(v)

    return info


# save the best individual as json
def save_best_individual(best_ind: dict, filename: str, 
                         path: str=None, info: dict={},
                         verbose: bool=False):

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
    info : dict, optional
        Additional information to save. Default is {}.
    verbose : bool, optional
        Whether to print the path and filename. 
        The default is False.
    """

    # build file
    file = {
        "info": check_info_dict(info),
        "genome": check_genome_content(best_ind),
    }

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
        try:
            json.dump(file, f)
        except TypeError as e:
            # logger.error(f"Error: {e}")
            # logger.error(f"Could not save the best individual.")
            # logger.debug(f"Best individual: {file}")
            return

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
                i = int(input("Select a file index: "))
                filename = available_files[i]
                break
            except:
                logger.info("Invalid input. Try again. [-1 to exit]]")

            # if i is -1, exit
            if i == -1:
                logger.info("[exit]")
                return None

    # add .json extension
    if not filename.endswith(".json"):
        filename += ".json"

    with open(os.path.join(path, filename), 'r') as f:
        best_ind = json.load(f)

    logger.info(f"loaded `{filename}` from `{path}`")

    return best_ind


def make_stats(values: list):

    """
    Make statistics from a list of values.

    Parameters
    ----------
    values : list
        The list of values.

    Returns
    -------
    stats : dict
        The statistics.
    """

    return np.mean(values), np.std(values), np.max(values), np.percentile(values, 16), np.percentile(values, 84)


""" Other """


class Visualizer:

    """
    A collection of functions to visualize the progress evolution.
    """

    def __init__(self, settings: dict, online: bool=True,
                 fitness_size: int=1, **kwargs):

        """
        A collection of functions to visualize the progress evolution.

        Parameters
        ----------
        settings : dict
            The simulation settings.
            keys: NPOP, NGEN, CXPB, MUTPB, NLOG
        online : bool, optional
            Whether to plot the variables online. 
            The default is True.
        fitness_size : int, optional
            The size of the fitness. Default is 1.
        **kwargs : dict
                optional
            target : tuple, optional
                The target. Default is None.
            k_average : int, optional
                The number of generations to average. 
                The default is 10.
            ylims : tuple, optional
                The y limits of the plot. 
                The default is None.
        """

        # parameters settings
        self.npop = settings["NPOP"]
        self.ngen = settings["NGEN"]
        self.cxpb = settings["CXPB"]
        self.mutpb = settings["MUTPB"]
        self.nlog = settings["NLOG"]
        # self.target = settings["TARGET"] if "TARGET" in settings else None
        # self.target_error = settings["TARGET_ERROR"] if "TARGET_ERROR" in settings else 0.
        self.num_cores = settings["NUM_CORES"] if "NUM_CORES" in settings else 1
        self.path = kwargs["path"] if "path" in kwargs else None

        # add the target only if it is not None and it is numeric
        if kwargs.get("target") is not None:
            self.target = np.array(settings["TARGET"])
            self.target_error = settings["TARGET_ERROR"] if "TARGET_ERROR" in settings else 0.
        else:
            self.target = None
            self.target_error = 0.

        # other parameters
        self.online = online
        self.fitness_size = fitness_size
        self.k_average = kwargs["k_average"] if "k_average" in kwargs else 10
        self._ylims = kwargs["ylims"] if "ylims" in kwargs else None
        self.title_text = kwargs["title_text"] if "title_text" in kwargs else ""

        # variables 
        self.gen = 0
        self.best_fitness = np.zeros((self.fitness_size, self.ngen))
        self.std_fitness = np.zeros((self.fitness_size, self.ngen))
        self.mean_fitness = np.zeros((self.fitness_size, self.ngen))
        self.long_average = np.zeros((self.fitness_size, self.ngen))
        # self.colors = plt.cm.Set1(np.linspace(0, 1, self.fitness_size))
        self.colors = ["black", "red", "blue", "green", "orange", "purple"][:self.fitness_size]
        self.top_fitness = None

        # create the figure
        if online and not IS_IPYTHON:
            self.fig, self.ax = plt.subplots(1, 1)
            logger.info("%figure & axes created")

    def update(self, population: list, best_ind: dict):

        """
        Update the variables.

        Parameters
        ----------
        population : list
            The population as fitnesses.
        best_ind : dict
            The best individual.
        """

        # population as an array (population_size, fitness_size)
        population = np.array(population)
        self.top_fitness = tuple(np.around(best_ind.fitness.values, 2))

        # update the best fitness
        for fi in range(self.fitness_size):
            self.best_fitness[fi, self.gen] = np.max(population[:, fi])
            self.mean_fitness[fi, self.gen] = np.mean(population[:, fi])
            self.std_fitness[fi, self.gen] = np.std(population[:, fi])

        # update the long-term average
        if self.gen > self.k_average:
            self.long_average[:, self.gen] = np.mean(
                self.best_fitness[:, self.gen-self.k_average:self.gen], axis=1)

        # update the generation
        self.gen += 1

    def plot(self, title: str=''):

        """
        Plot the variables.

        Parameters
        ----------
        title : str, optional
            The title of the plot.
            The default is ''.
        ylims : tuple, optional
            The y limits of the plot. 
            The default is None.
        """

        if IS_IPYTHON:
            if self.online:
                clf()
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        else:
            if self.online:
                fig, ax = self.fig, self.ax
                ax.clear()
            else:
                fig, ax = plt.subplots(1, 1)

        # -------------------------------- #

        for i in range(self.fitness_size):

            # plot the best fitness
            ax.plot(range(self.gen), self.best_fitness[i, :self.gen], '-o',
                    color=self.colors[i], alpha=0.5)

            # # plot the mean fitness
            # ax.plot(range(self.gen), self.mean_fitness[i, :self.gen], '--',
            #         color=self.colors[i], alpha=0.9)

            # plot the standard deviation of the fitness
            # as an area below the best fitness
            ax.fill_between(range(self.gen),
                        [self.mean_fitness[i, :self.gen][j] - self.std_fitness[i, :self.gen][j] for j in range(self.gen)],
                            [self.mean_fitness[i, :self.gen][j] + self.std_fitness[i, :self.gen][j] for j in range(self.gen)],
                            alpha=0.1, color=self.colors[i])


            # plot the long-term average
            ax.plot(range(self.gen), self.long_average[i, :self.gen], '-', color=self.colors[i],
                    alpha=0.95, label=f"average [{i}]")

            # plot the target
            if self.target is not None:
                ax.plot(range(self.gen), [self.target[i] for _ in range(self.gen)], 'v', 
                        color=self.colors[i], alpha=0.5, label=f"target [{i}]")
                ax.fill_between(range(self.gen),
                                [self.target[i] - self.target_error for _ in range(self.gen)],
                                [self.target[i] + self.target_error for _ in range(self.gen)],
                                alpha=0.1, color=self.colors[i])

        # -------------------------------- #

        # plot the legend
        ax.legend(loc='lower left')

        # plot the title
        ax.set_title(f"N={self.npop}  - " + \
                     f"Fitness={self.top_fitness} | {self.title_text}")

        # labels
        ax.set_xlabel(f"Generations [max={self.ngen}]")
        ax.set_ylabel("Fitness")

        #
        ax.set_xticks(range(0, self.gen))
        ax.set_xticklabels(range(0, self.gen))


        if self._ylims is not None:
            ax.set_ylim(self._ylims)

        # plot the grid
        ax.grid()

        # plot the show
        if self.online:
            plt.pause(0.1)
        else:
            plt.show()

    def keep_figure(self):

        """
        Keep the figure open.
        """

        if self.online:
            plt.show()

    def save_figure(self, filename: str=None):

        """
        Save the figure.

        Parameters
        ----------
        filename : str, optional
            The filename. If None, use the current time.
        """

        if self.path is None:
            self.path = os.getcwd()
            self.path = os.path.join(self.path, "cache")

        if filename is None:
            filename = time.strftime("%Y%m%d-%H%M%S")

        if not filename.endswith(".png"):
            filename += ".png"

        plt.savefig(os.path.join(self.path, filename))

        logger.info(f"Figure saved as `{filename}`.")


# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #


if __name__ == "__main__":

    """ EXAMPLE: evolution of a string """

    TARGET = "modena"
    LENGTH = len(TARGET)

    PARAMETERS = {
        f'char{i}': lambda: random.choice(list(
            "abcdefghijklmnopqrstuvwxyz ")) for i in range(
            1, LENGTH+1)
    }

    FIXED_PARAMETERS = {
    }

    class Model(object):

        def __init__(self, **kwargs):

            self.genome = kwargs

        def step(self):
            pass

        @property
        def output(self):
            pass

        def reset(self):
            pass

    # ---| Game |---

    class Game(object):

        def __init__(self, target):
            self.target = target

        def run(self, model):
            score = 0
            for i in range(LENGTH):
                if model['char'+str(i+1)] == self.target[i]:
                    score += 1
            return (score, score + np.random.normal(0, 0.1))

    game = Game(TARGET)


    # strategy
    N = len(PARAMETERS) - len(FIXED_PARAMETERS)  # number of parameters   

    from numpy.random import rand
    from numpy import floor, log
    from deap import cma
    MEAN = rand(N)  # Initial mean, could be randomized
    SIGMA = 0.5  # Initial standard deviation
    lambda_ = 4 + floor(3 * log(N))

    # strategy
    strategy = cma.Strategy(centroid=MEAN,
                            sigma=SIGMA, 
                            lambda_=lambda_)


    toolbox = make_toolbox(PARAMETERS=PARAMETERS,
                           game=game,
                           model=Model,
                           strategy=strategy,
                           FIXED_PARAMETERS=FIXED_PARAMETERS,
                           fitness_weights=(1.0, 1.),
                           verbose=False)

    settings = {
        "NPOP": 10,
        "NGEN": 100,
        "CXPB": 0.6,
        "MUTPB": 0.7,
        "NLOG": 1,
        "TARGET": (6., 6.),
        "TARGET_ERROR": 0.,
        "NUM_CORES": 1,
    }

    visualizer = Visualizer(settings=settings, online=True, k_average=20,
                            fitness_size=len(settings["TARGET"]))

    best_ind = main(toolbox=toolbox,
                    settings=settings,
                    save=False,
                    visualizer=visualizer,
                    verbose=True)

    string = "".join(
        [best_ind['char'+str(i+1)] for i in range(LENGTH)])

    logger.info(f"Best individual: {string} [{TARGET}]")

