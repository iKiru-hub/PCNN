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

class Env:

    def __init__(self, speed: float, dt: float, duration: int,
                 sigma: float=0.05, n_samples: int=1):

        """
        The game class.

        Parameters
        ----------
        speed : float
            The speed of the trajectory.
        dt : float
            The timestep.
        duration : int
            The duration of the game.
        sigma : float
            The noise level.
        n_samples : int
            number of samples to average over.
        **kwargs : dict
            
        """

        # parameters
        self.speed = speed
        self.dt = dt
        self.duration = duration
        self.sigma = sigma

        # 
        self.fitness_size = 2
        self._n_samples = n_samples

    def __repr__(self):

        return f"Track2D(fitness_size={self.fitness_size}, n_samples={self._n_samples})"

    def _make_data(self, Nj: int) -> np.ndarray:

        """
        Make the dataset.

        Parameters
        ----------
        Nj : int
            Number of neurons.
        **kwargs : dict
            T : int
                The duration of the trajectory.
            dx : float
                The stepsize.
            offset : int
                The offset.

        Returns
        -------
        Z : np.ndarray
            The dataset.
        """

        # Head Direction Layer
        layer = it.HDLayer(N=Nj, sigma=self.sigma)

        return it.make_layer_trajectory(layer=layer, 
                                        duration=self.duration,
                                        dt=self.dt,
                                        speed=self.speed)

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

            # sample new dimensions
            N, Nj = random.randint(4, 30), random.randint(4, 30)

            # update the dataset
            self.dataset = self._make_data(Nj=Nj)

            # update the agent
            agent.model.set_dims(N=N, Nj=Nj)

            # test the agent on the dataset
            agent = self._train(agent=agent, 
                                trajectory=self.dataset)

            # evaluate the agent
            fitness_tot += np.array([mm.eval_func(agent.model, axis=0),
                                     mm.eval_func(agent.model, axis=1)])

        # check nan
        if np.isnan(fitness_tot).any():
            fitness_tot = np.ones(self.fitness_size) * -1e3

        fitness = tuple(fitness_tot / self._n_samples)
        assert isinstance(fitness, tuple), "fitness must be a tuple"
        return fitness



# ----------------------------------------------------------
# ---| Genome Setup |---
# ----------------------

# parameters that are not evolved
FIXED_PARAMETERS = {
  'gain': 10.0,
  'bias': 1.,
  # 'lr': 0.2,
  'tau': 200,
  'wff_std': 0.0,
  'wff_min': 0.0,
  # 'wff_max': 1.,
  'wff_tau': 6_000,
  'std_tuning': 0.0,
  'soft_beta': 10,
  'dt': 1,
  'N': 5,
  'Nj': 5,
  'DA_tau': 3,
  'bias_scale': 0.0,
  'bias_decay': 100,
  # 'IS_magnitude': 6,
  # 'theta_freq': 0.004,
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
    'theta_freq': lambda: random.choice(np.arange(0, 0.1, 0.001)),
    'theta_freq_increase': lambda: random.uniform(0.01, 0.5),
    'nb_per_cycle': lambda: random.randint(3, 10),
}



if __name__ == "__main__" :

    # ---| Setup |---

    fitness_weights = (1., 1.)
    model = mm.RateNetwork7

    # ---| CMA-ES |---

    # parameters
    N_param = len(PARAMETERS) - len(FIXED_PARAMETERS)  # number of parameters   
    MEAN = np.random.rand(N_param)  # Initial mean, could be randomized
    SIGMA = 0.5  # Initial standard deviation
    lambda_ = 4 + np.floor(3 * np.log(N_param))

    # strategy
    strategy = cma.Strategy(centroid=MEAN,
                            sigma=SIGMA, 
                            lambda_=lambda_)


    # ---| Game |---

    T = 8  # s
    dt = 0.0001  # ms
    env = Env(speed=0.1, dt=dt, duration=T, 
              sigma=0.01, n_samples=3)

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
        "NPOP": 20,
        "NGEN": 200,
        "CXPB": 0.6,
        "MUTPB": 2.3,
        "NLOG": 5,
        "TARGET": (1., 1.),
        "TARGET_ERROR": 0.,
    }

    # ---| save |---
    save = False

    # filename as best_DDMM_HHMM_r3 
    path = "cache/"

    # get number of files in the cache
    n_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    filename = "best_" + str(n_files)

    # extra information 
    info = {
        "date": time.strftime("%d/%m/%Y") + " at " + time.strftime("%H:%M"),
        "model": model.__name__,
        "game": env.__repr__(),
        "evolved": [key for key in PARAMETERS.keys() if key not in FIXED_PARAMETERS.keys()],
    }

    # ---| Run |---
    best_ind = me.main(toolbox=toolbox, settings=settings,
                       info=info, save=save, filename=filename,
                       verbose=True)
