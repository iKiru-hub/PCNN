
import numpy as np
import random
import pytest
import sys
import os

import inputools.Trajectory as it

# Add the src directory to the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import mod_models as mm


N = 5
Nj = 9


GENOME = {
          'gain': 10.0,
          'bias': 1.,
          'lr': 0.2,
          'tau': 200,
          'wff_std': 0.0,
          'wff_min': 0.0,
          'wff_max': 1.,
          'wff_tau': 6_000,
          'std_tuning': 0.0,
          'soft_beta': 10,
          'dt': 1,
          'N': N,
          'Nj': Nj,
          'DA_tau': 3,
          'bias_scale': 0.0,
          'bias_decay': 100,
          'IS_magnitude': 6,
          'theta_freq': 0.004,
          'nb_per_cycle': 5,
          'plastic': True,
          'nb_skip': 2,
}


def test_pcnn_init():


    model = mm.PCNNetwork(**GENOME)


def test_training_hd():


    # initialize the model
    model = mm.PCNNetwork(**GENOME)
    layer_hd = it.HDLayer(N=Nj, sigma=0.0075)

    inputs_hd = it.make_layer_trajectory(layer=layer_hd,
                                         duration=5, 
                                         dt=0.01, speed=1.)

    # train the model
    for x in inputs_hd:
        model.step(x=x.reshape(-1, 1))


def test_training_pc():

    # initialize the model
    layer_pc = it.PlaceLayer(N=Nj, sigma=0.01)
    model = mm.PCNNetwork(**GENOME)

    # make data
    whole_track = it.AnimalTrajectory.whole_walk(dx=0.05)
    whole_track_pc = layer_pc.parse_trajectory(trajectory=whole_track)

    # train the model
    for x in whole_track_pc:
        model.step(x=x.reshape(-1, 1))

