import numpy as np
import time, os
import random
from deap import base, creator, tools, cma
import src.mod_models as mm
import src.mod_stimulation as ms

from tools.utils import logger
import tools.evolutions as me
import inputools.Trajectory as it


"""
- model: RateNetwork7
- fitness: eval_func 
"""



# ----------------------------------------------------------
# ---| Game |---
# --------------

data_settings = {
    'duration': 40,
    'dt': 0.001,
    'Np': 0.6,  # note that Np is the proportion of Nj for defining the number of patterns
    'cycles': 2,
}


class Env:

    def __init__(self,n_samples: int=1, **kwargs):

        """
        The game class.

        Parameters
        ----------
        n_samples : int
            number of samples to average over.
        **kwargs : dict
            n_pop : int
                The population size.
            new_dataset_period : int
                The period to make a new dataset.

        """

        # 
        self.fitness_size = 2
        self._Np = data_settings['Np']
        self._cycles = data_settings['cycles']
        self._T = data_settings['duration'] / data_settings['dt']
        self._n_samples = n_samples
        self.new_dataset_period = kwargs.get(
            "new_dataset_period", 3) * kwargs.get("n_pop", 30)

        # make dataset
        if kwargs.get("Nj_set", None) is None:
            self._Nj_set = [i**2 for i in range(4, 4 + n_samples)]
        else:
            self._Nj_set = kwargs.get("Nj_set")
        self._dataset = None
        self._local_np = None
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

    def _make_new_data(self):

        """
        Make a new dataset.
        """

        self._dataset = []
        self._local_np = []

        for Nj in self._Nj_set:

            # pick a number of patterns proportional to Nj
            Np = int(Nj * self._Np)

            # adjust T such that it is divided by Nj without remainder
            T = int(self._T - (self._T % Np))

            # make dataset
            self._dataset += [it.make_patterned_input(N=Nj, T=T, Np=Np, 
                                                      binary=True,
                                              cycles=self._cycles)]
            self._local_np += [Np]

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

        fitness_tot = np.zeros(self.fitness_size)
        for i in range(self._n_samples):

            # update the dataset
            dataset = self._dataset[i]

            # update the agent
            N = random.randint(10, 30)
            agent.model.set_dims(N=N, Nj=self._Nj_set[i])

            # test the agent on the dataset
            agent = self._train(agent=agent, 
                                trajectory=dataset)

            # evaluate the agent
            fitness_tot += np.array([mm.eval_func(weights=agent.model.Wff.copy(), 
                                                  wmax=agent.model._wff_max,
                                                  axis=0),
                                     mm.eval_func(weights=agent.model.Wff.copy(),
                                                  wmax=agent.model._wff_max,
                                                  axis=1, Nj_trg=self._local_np[i])])

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
  'gain': 7.0,
  # 'bias': 1.5,
  # 'lr': 0.2,
  # 'tau': 200,
  'wff_std': 0.0,
  'wff_min': 0.0,
  # 'wff_max': 1.,
  # 'wff_tau': 6_000,
  'std_tuning': 0.0,
  # 'soft_beta': 30,
  'dt': 1,
  'N': 5,
  'Nj': 5,
  'DA_tau': 3,
  'bias_scale': 0.0,
  'bias_decay': 100,
  # 'IS_magnitude': 6,
  'is_retuning': False,
  # 'theta_freq': 0.004,
  # 'theta_freq_increase': 0.16,
  # 'nb_per_cycle': 5,
  'plastic': True,
  'nb_skip': 2
}


# Define the genome as a dict of parameters 
PARAMETERS = {
    'gain': lambda: round(random.uniform(0.1, 20.0), 1),
    'bias': lambda: round(random.uniform(0, 30), 1),
    'lr': lambda: round(random.uniform(1e-0, 1e-5), 5),
    'tau': lambda: random.randint(1, 10),
    'wff_std': lambda: round(random.uniform(0, 3.0), 2),
    'wff_min': lambda: round(random.uniform(.0, 1.0), 1),
    'wff_max': lambda: round(random.uniform(1.0, 10.0), 1),
    'wff_tau': lambda: random.choice(range(300, 1500, 50)),
    'std_tuning': lambda: round(random.uniform(0, 1e-3), 4),
    'soft_beta': lambda: round(random.uniform(0, 1e2), 1),
    'dt': lambda: round(random.uniform(0, 1.2), 3),
    'DA_tau': lambda: random.randint(1, 200),
    'bias_decay': lambda: random.randint(1, 400),
    'bias_scale': lambda: round(random.uniform(0.5, 1.5), 2),
    'IS_magnitude': lambda: round(random.uniform(0.1, 15.0), 1),
    'is_retuning': lambda: random.choice((True, False)),
    'theta_freq': lambda: random.choice(np.arange(0, 0.1, 0.001)),
    'theta_freq_increase': lambda: random.uniform(0.01, 0.5),
    'nb_per_cycle': lambda: random.randint(3, 10),
    'nb_skip': lambda: random.randint(1, 5),
}



if __name__ == "__main__" :

    # ---| Setup |---

    fitness_weights = (1., 1.)
    model = mm.PCNNetwork
    NPOP = 30
    NGEN = 1000
    NUM_CORES = 6  # out of 8

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
    n_samples = 6
    nj_set = np.linspace(10, 60, n_samples, endpoint=True).astype(int)
    env = Env(n_samples=n_samples, n_pop=NPOP, new_dataset_period=10)

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
        "TARGET": (1., 1.),
        "TARGET_ERROR": 0.,
        "NUM_CORES": NUM_CORES,
    }

    # ---| Visualisation |---

    visualizer = me.Visualizer(settings=settings, online=True,
                               k_average=20,
                               fitness_size=len(fitness_weights))

    # ---| save |---
    save = bool(1)

    # filename as best_DDMM_HHMM_r3 
    path = "cache/"

    # get number of files in the cache
    n_files = len([f for f in os.listdir(path) \
        if os.path.isfile(os.path.join(path, f))])
    filename = "best_" + str(n_files) + "_pcnn_ptt"

    # extra information 
    info = {
        "date": time.strftime("%d/%m/%Y") + " at " + time.strftime("%H:%M"),
        "model": model.__name__,
        "game": env.__repr__(),
        "evolved": [key for key in PARAMETERS.keys() if key not in FIXED_PARAMETERS.keys()],
        "data": data_settings
    }

    # ---| Run |---
    best_ind = me.main(toolbox=toolbox, settings=settings,
                       info=info, save=save, visualizer=visualizer,
                       filename=filename,
                       verbose=True)
