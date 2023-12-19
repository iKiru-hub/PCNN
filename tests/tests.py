import random
import pytest

import sys
import os

# Add the src directory to the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import mod_models as mm


@pytest.fixture
def setup_network():

    # define genome
    genome = {
          'gain': 10.0,
          'bias': 1.,
          'lr': 0.15,
          'tau': 300,
          'wff_std': 0.001,
          'wff_min': 0.0,
          'wff_max': 1.5,
          'wff_tau': 6_000,
          'std_tuning': 0.03,
          'soft_beta': 15,
          'dt': 1,
          'N': N,
          'Nj': Nj,
          'DA_tau': 3,
          'bias_scale': 0.0,
          'bias_decay': 100,
          'IS_magnitude': 10,
          'theta_freq': 0.002,
          'nb_per_cycle': 6,
          'plastic': True,
          'nb_skip': 2,
    }

