import numpy as np
import time, os
import random
from deap import base, creator, tools, cma
import src.models as mm

from tools.utils import logger
import tools.evolutions as me
import inputools.Trajectory as it


"""
- model: RateNetwork7
- fitness: eval_information_II
"""



# ----------------------------------------------------------
# ---| Game |---
# --------------

data_settings = {
    'duration': 5,
    'dt': 1e-1,
    'speed': [0.01, 0.05],
    'prob_turn': 0.002,
    'k_average': 300,
    'sigma_pc': 0.01,
    'sigma_bc': 0.05,
    'Npc': 5**2,
    'Nbc': 4*4,
    'layer': None,
}


def make_2D_data(**kwargs) -> tuple:

    """
    Make the dataset.

    Parameters
    ----------
    Nj : int
        Number of neurons.

    Returns
    -------
    trajectory_pc : np.ndarray
        The trajectory of the place cells.
    trajectory_pc : np.ndarray
        The trajectory of the whole track.
    """

    global data_settings

    layer = it.InputNetwork(layers=[
        it.PlaceLayer(N=data_settings['Npc'], 
                      sigma=data_settings['sigma_pc']),
        it.BorderLayer(N=data_settings['Nbc'], 
                       sigma=data_settings['sigma_bc'])
    ])

    # layer = it.InputNetwork(layers=[
    #     it.GridLayer(N=data_settings['Npc'], 
    #                  sigma=data_settings['sigma_pc'],
    #                  scale=np.array([1.1, 1.])),
    #     it.BorderLayer(N=data_settings['Nbc'], 
    #                    sigma=data_settings['sigma_bc'])
    # ])

    data_settings['layer'] = layer.__repr__()

    # trajectory settings
    duration = kwargs.get("duration", data_settings['duration'])
    dt = kwargs.get("dt", data_settings['dt'])
    speed = kwargs.get("speed", data_settings['speed'])
    prob_turn = kwargs.get("prob_turn", data_settings['prob_turn'])
    k_average = kwargs.get("k_average", data_settings['k_average'])

    # Create a trajectory
    trajectory = it.make_trajectory(duration=duration,
                                    dt=dt,
                                    speed=speed, 
                                    prob_turn=prob_turn,
                                    k_average=k_average,)

    # make whole track trajectory
    whole_track = it.make_whole_walk(dx=0.01)

    return layer.parse_trajectory(trajectory=trajectory, disable_tqdm=True), \
        layer.parse_trajectory(trajectory=whole_track, disable_tqdm=True)


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
        self.Nj = data_settings['Npc'] + data_settings['Nbc']

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
        self._make_new_data(dt=agent.model.kwargs['dt'],
                            speed=[agent.model.kwargs['speed'],
                                   agent.model.kwargs['speed']])

        fitness_tot = np.zeros(self.fitness_size)
        for i in range(self._n_samples):

            # update the agent
            N = random.randint(20, 40)
            agent.model.set_dims(N=N, Nj=self.Nj)

            # test the agent on the dataset
            agent = self._train(agent=agent, 
                                trajectory=self._dataset[i])

            # evaluate the agent
            fitness_trial = self._eval_func(
                                        model=agent.model,
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
  # 'gain': 5.0,
  # 'bias': 1.,
  # 'lr': 0.8,
  # 'tau': 200,
  'wff_min': 0.0,
  # 'wff_max': 2.,
  # 'wff_tau': 400,
  # 'soft_beta': 1,
  # 'beta_clone': 0.3,
  'low_bounds_nb': 5,
  'N': 5,
  'Nj': 5,
  # 'DA_tau': 3,
  # 'IS_magnitude': 20,
  'is_retuning': False,
  # 'theta_freq': 0.007,
  'theta_freq_increase': 0.16,
  # 'sigma_gamma': 5e-6,
  'nb_per_cycle': 5,
  'plastic': True,
  'nb_skip': 2
}


# Define the genome as a dict of parameters 
PARAMETERS = {
    'gain': lambda: round(random.uniform(0.1, 20.0), 1),
    'bias': lambda: round(random.uniform(0, 30), 1),
    'lr': lambda: round(random.uniform(1e-0, 1e-5), 5),
    'tau': lambda: random.randint(1, 10),
    'wff_min': lambda: round(random.uniform(.0, 1.0), 1),
    'wff_max': lambda: round(random.uniform(1.0, 10.0), 1),
    'wff_tau': lambda: random.choice(range(300, 1500, 50)),
    'soft_beta': lambda: random.randint(1, 20)/10,
    'beta_clone': lambda: random.randint(1, 20)/10,
    'low_bounds_nb': lambda: random.randint(1, 10),
    'DA_tau': lambda: random.randint(1, 200),
    'bias_decay': lambda: random.randint(1, 400),
    'bias_scale': lambda: round(random.uniform(0.5, 1.5), 2),
    'IS_magnitude': lambda: round(random.uniform(0.1, 15.0), 1),
    'is_retuning': lambda: random.choice((True, False)),
    'theta_freq': lambda: random.choice(np.arange(0.001, 0.01, 0.001)),
    'theta_freq_increase': lambda: random.uniform(0.01, 0.5),
    'sigma_gamma': lambda: random.choice(np.arange(1e-6, 1e-4, 5e-6)),
    'nb_per_cycle': lambda: random.randint(3, 10),
    'nb_skip': lambda: random.randint(1, 5),
    'dt': lambda: random.choice(np.arange(1e-3, 1, 2.5e-3)),
    'speed': lambda: random.choice(np.arange(1e-3, 1e-1, 2.5e-3)),
}



if __name__ == "__main__" :

    # ---| Setup |---

    fitness_weights = (1., 1., 1., 1.)
    model = mm.PCNNetwork
    NPOP = 40
    NGEN = 1000
    NUM_CORES = 6  # out of 8

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
    path = "cache/"

    # get number of files in the cache
    n_files = len([f for f in os.listdir(path) \
        if os.path.isfile(os.path.join(path, f))])
    filename = str(n_files+1) + "_best_pcnn_bpc"

    # extra information 
    info = {
        "date": time.strftime("%d/%m/%Y") + " at " + time.strftime("%H:%M"),
        "model": model.__name__,
        "game": env.__repr__(),
        "evolved": [key for key in PARAMETERS.keys() if key not in FIXED_PARAMETERS.keys()],
        "data": data_settings,
        "other": "trained with Border+PlaceLayer; fitness: " + \
            "+mean I (traj), -std I (traj), -nb_peaks, +ratio",
    }

    # ---| Run |---
    best_ind = me.main(toolbox=toolbox, settings=settings,
                       info=info, save=save, visualizer=visualizer,
                       filename=filename,
                       verbose=True)
