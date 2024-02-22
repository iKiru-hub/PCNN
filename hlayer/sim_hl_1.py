import numpy as np
import time, sys, os
import random
from deap import base, creator, tools, cma

from tools.utils import logger
import tools.evolutions as me
import inputools.Trajectory as it

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hl_model as hl
import src.models as mm



# ----------------------------------------------------------
# ---| Game |---
# --------------

data_settings = {
    'duration': 5,
    'dt': 1e-3,
    'speed': [0.01, 0.01],
    'prob_turn': 0.005,
    'k_average': 300,
}


def make_2D_data() -> tuple:

    """
    Make the dataset.

    Returns
    -------
    trajectory_pc : np.ndarray
        The trajectory of the place cells.
    trajectory_pc : np.ndarray
        The trajectory of the whole track.
    """

    global data_settings


    # trajectory settings
    duration = data_settings['duration']
    dt = data_settings['dt']
    speed = data_settings['speed']
    prob_turn = data_settings['prob_turn']
    k_average = data_settings['k_average']

    # Create a trajectory
    trajectory = it.make_trajectory(duration=duration,
                                    dt=dt,
                                    speed=speed, 
                                    prob_turn=prob_turn,
                                    k_average=k_average,)

    # make whole track trajectory
    whole_track = it.make_whole_walk(dx=0.01)

    return trajectory, whole_track


class Env:

    """
    `Nj_set` is not present in this variant
    """

    def __init__(self,n_samples: int=1, 
                 make_data: callable=make_2D_data, 
                 eval_func: callable=mm.eval_func_2, 
                 **kwargs):

        """
        The game class.

        Parameters
        ----------
        n_samples : int
            number of samples to average over.
        make_data : callable
            The function to make the dataset.
            Default: make_1D_data
        eval_func : callable
            The evaluation function.
            Default: eval_func_2
        **kwargs : dict
            n_pop : int
                The population size.
            new_dataset_period : int
                The period to make a new dataset.
            target : float 
                target population activity.
            fitness_size : int
                The size of the fitness.
                Default: 1
        """

        # 
        self.fitness_size = kwargs.get("fitness_size", 1)
        self._n_samples = n_samples
        self.new_dataset_period = kwargs.get(
            "new_dataset_period", 3) * kwargs.get("n_pop", 30)
        self._target = kwargs.get("target", None)

        # make dataset
        self._dataset = None
        self._dataset_whole = None
        self._eval_func = eval_func
        self._make_data = make_data
        self._make_new_data()

        # variables
        self.counter = 0

    def __repr__(self):

        return f"Env(fitness_size={self.fitness_size}, n_samples={self._n_samples})"

    def _train(self, agent: object, trajectory: np.ndarray) -> tuple:

        """
        Train a agent on the dataset.

        Parameters
        ----------
        agent : object
            A agent object.
        trajectory : np.ndarray
            A trajectory.

        Returns
        -------
        agent : object
            The trained agent.
        """

        agent.model.reset()

        # train on a single trajectory
        for t in range(len(trajectory)):
            agent.step(trajectory[t].reshape(-1, 1))

        return agent

    def _make_new_data(self, **kwargs):

        """
        Make a new dataset.
        """

        self._dataset = []
        self._dataset_whole = []

        for _ in range(self._n_samples):
            dataset, dataset_whole = self._make_data(**kwargs)
            self._dataset.append(dataset)
            self._dataset_whole.append(dataset_whole)

    def run(self, agent: object) -> float:

        """
        Evaluate a agent on the dataset.

        Parameters
        ----------
        agent : object
            A agent object.

        Returns
        -------
        fitness : float
            The fitness value.
        """

        # make dataset
        if self.counter % self.new_dataset_period == 0:
            self._make_new_data()

        fitness_tot = np.zeros(self.fitness_size)
        for i in range(self._n_samples):

            # test the agent on the dataset
            agent = self._train(agent=agent, 
                                trajectory=self._dataset[i])

            # logger.debug(f"{dir(agent)}")

            # evaluate the agent
            fitness_trial = self._eval_func(
                                        model=agent.model.pcnn,
                                        trajectory=self._dataset[i],
                                        whole_trajectory=self._dataset_whole[i])

            fitness_tot += np.array(fitness_trial)

        # check nan
        if np.isnan(fitness_tot).any():
            fitness_tot = np.ones(self.fitness_size) * -1e3

        fitness = tuple(fitness_tot / self._n_samples)
        assert isinstance(fitness, tuple), "fitness must be a tuple"

        # counter, if counter is equal to the population size then
        # make a new dataset
        self.counter += 1

        if self.counter == self.new_dataset_period:
            self._make_new_data()
            self.counter = 0
            logger.info("New dataset")

        return fitness



# ----------------------------------------------------------
# ---| Genome Setup |---
# ----------------------

# parameters that are not evolved
FIXED_PARAMETERS = {

}


# Define the genome as a dict of parameters 
PARAMETERS = {
    'W': lambda: np.random.normal(0, 0.01, (np.random.randint(30, 100), 2)),
    'activation': lambda: random.choice(('sigmoid', 'positive_tanh', 'relu', 'none')),
}



if __name__ == "__main__" :

    # ---| Setup |---

    fitness_weights = (1., 1., 1., 1.)
    model = hl.ModelHL
    NPOP = 10
    NGEN = 1000
    NUM_CORES = 1  # out of 8

    # Ignore runtime warnings
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # ---| CMA-ES |---

    # parameters
    N_param = len(PARAMETERS) - len(FIXED_PARAMETERS)  # number of parameters
    MEAN = np.random.rand(N_param)  # Initial mean, could be randomized
    SIGMA = 0.8  # Initial standard deviation
    lambda_ = 4 + np.floor(3 * np.log(N_param))

    # strategy
    strategy = cma.Strategy(centroid=MEAN,
                            sigma=SIGMA, 
                            lambda_=lambda_)


    # ---| Game |---
    # -> see above for the specification of the data settings
    n_samples = 2
    # nj_set = [int(i**2) for i in np.linspace(6, 9, n_samples, endpoint=True)]
    env = Env(n_samples=n_samples,
              make_data=make_2D_data,
              eval_func=mm.eval_information_II,
              n_pop=NPOP,
              new_dataset_period=2,
              fitness_size=len(fitness_weights))

    # ---| Evolution |---

    # Create the toolbox
    toolbox = me.make_toolbox(PARAMETERS=PARAMETERS.copy(),
                              game=env,
                              model=model,
                              strategy=strategy,
                              FIXED_PARAMETERS=FIXED_PARAMETERS.copy(),
                              fitness_weights=fitness_weights)

    # ---| Run |---

    settings = {
        "NPOP": NPOP,
        "NGEN": NGEN,
        "CXPB": 0.6,
        "MUTPB": 0.7,
        "NLOG": 1,
        "TARGET": (1.,),
        "TARGET_ERROR": 0.,
        "NUM_CORES": NUM_CORES,
    }

    # ---| Visualisation |---

    visualizer = me.Visualizer(settings=settings, online=True,
                               target=None,
                               k_average=20,
                               fitness_size=len(fitness_weights),
                               ylims=None)

    # ---| save |---
    save = bool(1)

    # filename as best_DDMM_HHMM_r3 
    path = "../cache/"

    # get number of files in the cache
    n_files = len([f for f in os.listdir(path) \
        if os.path.isfile(os.path.join(path, f))])
    filename = str(n_files+1) + "_best_hl"

    # extra information 
    info = {
        "date": time.strftime("%d/%m/%Y") + " at " + time.strftime("%H:%M"),
        "model": model.__name__,
        "game": env.__repr__(),
        "evolved": [key for key in PARAMETERS.keys() if key not in FIXED_PARAMETERS.keys()],
        "data": data_settings,
        "other": "evolving the input layer; fitness: " + \
            "+mean I (traj), -std I (traj), -nb_peaks, +ratio",
    }

    # ---| Run |---
    best_ind = me.main(toolbox=toolbox, settings=settings,
                       info=info, save=save, visualizer=visualizer,
                       filename=filename,
                       verbose=True)

