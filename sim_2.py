import random
from deap import base, creator, tools, cma
import src.mod_models as mm
import src.mod_evolution as me
import src.mod_stimulation as ms
from tools.utils import logger
import inputools.Trajectory as it
import numpy as np
import time, os


# ----------------------------------------------------------
# ---| Fitness |---
# -----------------

def FitnessFunction(W: np.ndarray, wmax: float) -> tuple:

    """
    Goal:
    - few high valued entries (tuning to a pattern)
    - all the rest low

    Parameters
    ----------
    W : np.ndarray
        The weight matrix.

    Returns
    -------
    fitness : tuple
        The fitness value.
    """

    # # local difference
    # sorted_cols = np.sort(W, axis=1)
    
    # value_1 = sorted_cols[:, -3:].sum() - sorted_cols[:, :-3].sum()

    # # global difference | select the top 2 rows
    # sorted_rows = np.sort(W, axis=0)[-2:, :]

    # value_2 = np.diff(sorted_rows, axis=0).sum() # difference between the two rows

    ### 
    # only one top value per column
    # sorted_cols = np.sort(W, axis=0)
    # value_1 = sorted_cols[-1:, :].sum() - sorted_cols[:-1, :].sum()

    # only one top value per row
    # sorted_rows = np.sort(W, axis=1)
    # value_2 = sorted_rows[:, -1:].sum() - sorted_rows[:, :-1].sum()

    # value_2 = -((W.sum(axis=1) - wmax)**2).sum()

    # total sum close to wmax*Nj
    # value_2 = -(W.sum() - wmax*W.shape[1])**2

    # sort the matrix columns such that it resembles a 
    # diagonal matrix
    I, J = W.shape
    sorted_matrix = np.zeros_like(W)
    max_indices = np.argmax(W, axis=1)

    for i in range(I):
        sorted_matrix[i, min((i, J-1))] = W[i, max_indices[i]]

    # calculate the difference between the sorted matrix
    # and a diagonal matrix
    target = wmax * np.eye(I, J)
    value_3 = -((sorted_matrix - target)**2).sum() / (I*J)

    #
    # value_1 = W.max(axis=1).mean()

    return (value_3,)


# ----------------------------------------------------------
# ---| Game |---
# --------------

class Track2D:

    def __init__(self, Nj: int, fitness_size: int=1, n_samples: int=1,
                 **kwargs):

        """
        The game class.

        Parameters
        ----------
        Nj : int
            Number of neurons.
        fitness_size : int
            The size of the fitness function.
        **kwargs : dict
            dataset parameters.
        """

        # self.Nj = Nj
        self.fitness_size = fitness_size
        self.kwargs = kwargs
        self.dataset = self._make_data(Nj=Nj, **kwargs)

        self._n_samples = n_samples

    def __repr__(self):

        return f"Track2D(fitness_size={self.fitness_size})"

    def _make_data(self, Nj: int=9, **kwargs) -> np.ndarray:

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

        # parameters
        T = kwargs.get('T', 100)
        dx = kwargs.get('dx', 0.1)
        offset = kwargs.get('offset', 0)

        # layer
        layer = lambda x: (np.cos(x + np.linspace(1e-1, offset, Nj)) + 1) / 2

        Z = np.arange(0, T, dx).reshape(-1, 1)
        return layer(Z)

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
            self.dataset = self._make_data(Nj=Nj, **self.kwargs)

            # update the agent
            agent.model.set_dims(N=N, Nj=Nj)

            # test the agent on the dataset
            agent = self._train(agent=agent, 
                                trajectory=self.dataset)

            # evaluate the agent
            fitness = FitnessFunction(W=agent.model.Wff.copy(),
                                      wmax=agent.model._wff_max)

            fitness_tot += np.array(fitness)

        # check nan
        if np.isnan(fitness_tot).any():
            fitness_tot = np.ones(self.fitness_size) * -1e3

        fitness = tuple(fitness_tot / self._n_samples)
        assert isinstance(fitness, tuple), "fitness must be a tuple"
        return fitness

    def run_old(self, agent: object) -> float:

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

        # test the agent on the dataset
        agent = self._train(agent=agent, 
                            trajectory=self.dataset)

        # evaluate the agent
        fitness = FitnessFunction(W=agent.model.Wff.copy(),
                                  wmax=agent.model._wff_max)

        assert isinstance(fitness, tuple), "fitness must be a tuple"
        return fitness


# ----------------------------------------------------------
# ---| Genome Setup |---
# ----------------------

# parameters that are not evolved
FIXED_PARAMETERS = {
    'N': 6,
    'Nj': 6,
    'gain': 30.,
    'bias': 2.5,
    # 'lr': 5e-2,
    'tau': 4.,
    'wff_std': 1e-2,
    'wff_min': 0.,
    'wff_max': 3.5,
    # 'wff_tau': 300,
    'rule': 'hebb',
    'std_tuning': 5e-3,
    'soft_beta': 50.,
    # 'dt': 0.00035,
    'DA_tau': 3,
    'bias_decay': 75,
    'bias_scale': 0.065,
}


# Define the genome as a dict of parameters 
PARAMETERS = {
    'gain': lambda: round(random.uniform(0.1, 20.0), 1),
    'bias': lambda: round(random.uniform(0, 30), 1),
    'lr': lambda: round(random.uniform(1e-0, 1e-5), 5),
    'tau': lambda: random.randint(1, 100),
    'wff_std': lambda: round(random.uniform(0, 3.0), 2),
    'wff_min': lambda: round(random.uniform(.0, 1.0), 1),
    'wff_max': lambda: round(random.uniform(1.0, 10.0), 1),
    'wff_tau': lambda: random.randint(10, 1000),
    'rule': lambda: random.choice(('oja', 'hebb')),
    'std_tuning': lambda: round(random.uniform(0, 1e-2), 4),
    'soft_beta': lambda: round(random.uniform(0, 1e2), 1),
    'dt': lambda: round(random.uniform(0, 1), 3),
    'DA_tau': lambda: random.randint(1, 200),
    'bias_decay': lambda: random.randint(1, 400),
    'bias_scale': lambda: round(random.uniform(0.5, 1.5), 2),
}


if __name__ == "__main__" :

    # ---| Setup |---

    fitness_weights = (1.,)
    model = mm.RateNetwork4

    # ---| CMA-ES |---

    # parameters
    N = len(PARAMETERS) - len(FIXED_PARAMETERS)  # number of parameters   
    MEAN = np.random.rand(N)  # Initial mean, could be randomized
    SIGMA = 0.5  # Initial standard deviation
    lambda_ = 4 + np.floor(3 * np.log(N))

    # strategy
    strategy = cma.Strategy(centroid=MEAN,
                            sigma=SIGMA, 
                            lambda_=lambda_)


    # ---| Game |---

    track = Track2D(Nj=FIXED_PARAMETERS['Nj'], 
                    fitness_size=len(fitness_weights),
                    n_samples=12)

    # ---| Evolution |---

    # Create the toolbox
    toolbox = me.make_toolbox(PARAMETERS=PARAMETERS.copy(),
                              game=track,
                              model=model,
                              strategy=strategy,
                              FIXED_PARAMETERS=FIXED_PARAMETERS.copy(),
                              fitness_weights=fitness_weights)

    # ---| Run |---

    settings = {
        "NPOP": 40,
        "NGEN": 1000,
        "CXPB": 0.6,
        "MUTPB": 2.3,
        "NLOG": 5,
        "TARGET": (0.,),
        "TARGET_ERROR": 0.,
    }

    # ---| save |---

    # filename as best_DDMM_HHMM_r3 
    path = "cache/"

    # get number of files in the cache
    n_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

    filename = "best_" + str(n_files)

    # extra information 
    info = {
        "date": time.strftime("%d/%m/%Y") + " at " + time.strftime("%H:%M"),
        "model": model.__name__,
        "game": track.__repr__(),
        "evolved": tuple(PARAMETERS.keys()),
    }

    # ---| Run |---
    best_ind = me.main(toolbox=toolbox, settings=settings,
                       info=info, save=True, filename=filename)
