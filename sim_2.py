import random
from deap import base, creator, tools, cma
import mod_models as mm
import mod_evolution as me
import mod_stimulation as ms
from tools.utils import logger
import inputools.Trajectory as it
import numpy as np
import time 


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
    sorted_cols = np.sort(W, axis=0)
    value_1 = sorted_cols[-1:, :].sum() - sorted_cols[:-1, :].sum()

    # total sum close to wmax*Nj
    value_2 = -(W.sum() - wmax*W.shape[1])**2

    return (value_1, value_2)


# ----------------------------------------------------------
# ---| Game |---
# --------------

class Track2D:

    def __init__(self, Nj: int, fitness_size: int=1, 
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

        self.Nj = Nj
        self.fitness_size = fitness_size
        self.dataset = self._make_data(Nj=Nj, **kwargs)

    def _make_data(self, T: int=20, dx: float=0.01, 
                   offset: int=4, Nj: int=9) -> np.ndarray:

        """
        Make the dataset.

        Parameters
        ----------
        T : int
            The duration of the trajectory.
        dx : float
            The stepsize.
        offset : int
            The offset.
        Nj : int
            Number of neurons.

        Returns
        -------
        Z : np.ndarray
            The dataset.
        """

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
    'N': 8,
    'Nj': 6,
    # 'gain': 7.,
    # 'bias': 3.,
    # 'lr': 1e-3,
    'tau': 50.,
    'wff_std': 1e-3,
    'wff_min': 0.,
    'wff_max': 3.,
    # 'wff_tau': 100,
    'rule': 'hebb',
    'std_tuning': 1e-3,
    # 'soft_beta': 10.,
    'dt': 0.053,
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
    'id': lambda: ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), 
                                           replace=True, size=7))
}


if __name__ == "__main__" :

    # ---| Setup |---

    fitness_weights = (1., 1.)

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
                    fitness_size=len(fitness_weights))

    # ---| Evolution |---

    # Create the toolbox
    toolbox = me.make_toolbox(PARAMETERS=PARAMETERS.copy(),
                              game=track,
                              model=mm.RateNetwork3,
                              strategy=strategy,
                              FIXED_PARAMETERS=FIXED_PARAMETERS.copy(),
                              fitness_weights=fitness_weights)

    # ---| Run |---

    settings = {
        "NPOP": 100,
        "NGEN": 800,
        "CXPB": 0.5,
        "MUTPB": 0.2,
        "NLOG": 5,
        "TARGET": (100, 0),
        "TARGET_ERROR": 1e-4,
    }

    # save | filename as best_DDMM_HHMM_r3 
    filename = "best_" + time.strftime("%d%m_%H%M") + "_r3"
    best_ind = me.main(toolbox=toolbox, settings=settings, filename=filename)
