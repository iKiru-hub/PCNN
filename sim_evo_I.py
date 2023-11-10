import random
from deap import base, creator, tools
import models as M
import mod_evolution as me
from tools.utils import logger
import numpy as np




# ----------------------------------------------------------
# ---| Game |---
# --------------

class Track2D:

    def __init__(self, dataset: list, Nj: int):

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


# ----------------------------------------------------------
# ---| Genome Setup |---
# ----------------------

# parameters that are not evolved
FIXED_PARAMETERS = {
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


