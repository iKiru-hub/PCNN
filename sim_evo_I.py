import random
from deap import base, creator, tools
import models as M
import mod_evolution as me
import mod_stimulation as ms
from tools.utils import logger
import inputools.Trajectory as it
import numpy as np




# ----------------------------------------------------------
# ---| Fitness |---
# -----------------

def FitnessFunction(W: np.ndarray, Nj: int=9,
                    wmax: float=4.5) -> np.ndarray:

    """
    The fitness function. It is composed of three terms:
    1. the maximum row entry should be close to wmax 
    2. each row should have only one entry close to wmax
    3. the high-valued entries should be close to Nj

    Parameters
    ----------
    W : np.ndarray
        The weight matrix.
    Nj : int
        Number of neurons.
    wmax : float
        The maximum weight.

    Returns
    -------
    fitness : float
        The fitness value.
    """

    # the maximum row entry should be close to wmax
    value_1 = W.max(axis=1)

    # each row should have at most one entry close to wmax
    # in practice: check the number of entries
    # above wmax/2
    value_2 = - np.where(np.sum(W > wmax/2, axis=1) > 1, 
                       0, 1).sum()

    # the high-valued entries should be close to Nj
    # in practice: check the number of entries
    # above wmax/2
    value_3 = - (Nj - np.sum(W > wmax/2)) ** 2

    return value_1 + value_2 + value_3

# ----------------------------------------------------------
# ---| Game |---
# --------------

class Track2D:

    def __init__(self, dataset: list, Nj: int, wmax: float=4.5):

        """
        A 2D tracking game.

        Parameters
        ----------
        dataset : list
            A list of spiking trajectories.
        Nj : int
            Number of neurons.
        """

        self.dataset = dataset
        self.Nj = Nj
        self.wmax = wmax

    def _train(self, model: object) -> object:

        """
        Train a model on the dataset.

        Parameters
        ----------
        model : object
            A model object.

        Returns
        -------
        model : object
            The trained model.
        """

        model.reset()

        # train on the whole dataset
        for trajectory in self.dataset:

            # train on a single trajectory
            for t in range(len(trajectory)):
                model.step(trajectory[t])

        return model

    def run(self, model: object) -> float:

        """
        Evaluate a model on the dataset.

        Parameters
        ----------
        model : object
            A model object.

        Returns
        -------
        fitness : float
            The fitness value.
        """

        # train the model
        model = self._train(model)

        # evaluate the model
        fitness = FitnessFunction(W=model.Wff, 
                                  Nj=self.Nj,
                                  wmax=self.wmax)

        return (fitness,)



# ----------------------------------------------------------
# ---| Genome Setup |---
# ----------------------

# parameters that are not evolved
FIXED_PARAMETERS = {
    'dim': 2,
    'lr_max': 3e-2,
    'lr_min': 1e-5,
    'lr_tau': 100,
    'wff_const': 5.5,
    'wff_max': 4.5,
}

# Define the genome as a dict of parameters 
PARAMETERS = {
    'tau_u': lambda: random.randint(1, 300),
    'lr_max': lambda: random.uniform(5e-3, 0.1),
    'lr_min': lambda: random.uniform(1e-3, 1e-6),
    'lr_tau': lambda: random.randint(50, 300),
    'wff_const': lambda: random.uniform(4.0, 10.0),
    'wff_max': lambda: random.uniform(2.0, 10.0),
    'wff_min': lambda: random.uniform(0., 2.0),
    'wff_tau_max': lambda: random.randint(1000, 8000),
    'wff_tau_min': lambda: random.randint(10, 500),
    'wff_tau_tau': lambda: random.randint(50, 400),
    'wff_beta': lambda: random.uniform(0.1, 1.0),
    'wr_const': lambda: random.uniform(0.1, 10.0),
    'dim': lambda: random.choice((1, 2)),
    'A': lambda: random.uniform(0.1, 4.0),
    'B': lambda: random.uniform(0.1, 3.0),
    'sigma_exc': lambda: random.uniform(0., 8.0),
    'sigma_inh': lambda: random.uniform(0., 8.0),
    'tau_ff': lambda: random.randint(1, 100),
    'tau_rec': lambda: random.randint(1, 100),
    'syn_ff_tau': lambda: random.randint(1, 100),
    'syn_ff_thr': lambda: random.uniform(0., 1.0),
    'rate_func_beta': lambda: random.uniform(0.1, 1.0),
    'rate_func_alpha': lambda: random.randint(50, 80),
}



if __name__ == "__main__" :

    # ---| Setup |---

    N = 9
    Nj = 9

    # ---| Dataset |---

    # Create an animal
    animal = it.AnimalTrajectory(dt=1, 
                                 prob_turn=0.01, 
                                 prob_speed=0.1,
                                 prob_rest=0.01, 
                                 day_cycle=True)

    # input layer
    layer = ms.InputLayer(N=Nj, kind='place', 
                          bounds=(0.05, 0.95, 0.05, 0.95),
                          sigma=0.04, max_rate=300, min_rate=10)

    dataset = it.make_dataset(n_samples=1,
                              animal=animal,
                              layer=layer,
                              duration=50,
                              timestep=100, dx=0.1)

    track = Track2D(dataset=dataset, Nj=Nj, 
                    wmax=FIXED_PARAMETERS['wff_max'])

    # ---| Evolution |---

    # Create the toolbox
    toolbox = me.make_toolbox(PARAMETERS=PARAMETERS,
                              game=track,
                              agent_class=M.NetworkSimple,
                              FIXED_PARAMETERS=FIXED_PARAMETERS)

    # ---| Run |---

    settings = {
        "NPOP": 10,
        "NGEN": 300,
        "CXPB": 0.5,
        "MUTPB": 0.2,
        "NLOG": 5
    }

    me.main(toolbox=toolbox,
            settings=settings)
