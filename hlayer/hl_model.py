import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import models as mm
from tools.utils import logger
import inputools.Trajectory as it





""" Parameters """

PCNN_PARAMS = {'gain': 5,
 'bias': 0.9,
 'lr': 0.6,
 'tau': 10,
 'wff_min': 0.0,
 'wff_max': 0.5,
 'wff_tau': 1500,
 'soft_beta': 1.7,
 'beta_clone': 0.3,
 'low_bounds_nb': 5,
 'DA_tau': 3,
 'bias_decay': 120,
 'bias_scale': 0.84,
 'IS_magnitude': 50,
 'theta_freq': 0.005,
 'theta_freq_increase': 0.075,
 'sigma_gamma': 9.1e-05,
 'nb_per_cycle': 5,
 'nb_skip': 2,
 'dt': 0.001,
 'speed': 0.005,
 'sigma_pc': 0.00251,
 'sigma_bc': 0.005,
 'N': 60,
 'Nj': 121,
 'is_retuning': True,
 'plastic': True,
 'u_trace_decay': 2}




""" Model """

class ModelHL:

    """
    this model is composed by an evolved input layer, which has the purpose of 
    opportunely process the input position, and a downstream layer, which is
    a PCNN with online formed place cells.
    """

    def __init__(self, W: np.ndarray, activation: str,
                 pcnn_params: dict=PCNN_PARAMS):

        """
        Parameters
        ----------
        W : np.ndarray
            the weight matrix of the input layer
        pcnn_params : dict
            the parameters for the PCNN layer.
            Default is PCNN_PARAMS
        """

        # input layer parameters
        self.W = W
        self.activation = self._make_activation_function(activation)
        self.dim_input = W.shape[1]
        self.dim_output = W.shape[0]

        # PCNN layer parameters
        # pcnn_params['Nj'] = self.dim_output
        # self.pcnn = mm.PCNNetwork(**pcnn_params)

    def __repr__(self):

        return f'ModelHL(dim_input={self.dim_input}, dim_output={self.dim_output})'

    def step(self, x: np.ndarray) -> np.ndarray:

        """
        Parameters
        ----------
        x : np.ndarray
            the input position

        Returns
        -------
        np.ndarray
            the output activity
        """

        # input layer
        y = self.activation(self.W @ x)

        # PCNN layer
        # y = self.pcnn.step(x=y)

        return y

    def _make_activation_function(self, kind: str):

        """
        Parameters
        ----------
        kind : str
            the kind of activation function

        Returns
        -------
        callable
            the activation function
        """

        if kind == 'relu':
            return lambda x: np.maximum(0, x)
        elif kind == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif kind == 'positive_tanh':
            return lambda x: (np.tanh(x) + 1) / 2
        else:
            return lambda x: x

    @property
    def output(self):

        """
        return the output layer
        """

        return self.pcnn.out

    def reset(self):

        """
        reset the PCNN layer
        """

        self.pcnn.reset()
