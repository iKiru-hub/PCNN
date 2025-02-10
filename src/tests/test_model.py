import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import core.build.pclib as pclib
import pytest
import numpy as np



""" global settings"""

N = 100
local_scale = 0.1





def test_space():


    gcn = pclib.GridNetworkSq([
           pclib.GridLayerSq(sigma=0.04, speed=1.*local_scale, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.8*local_scale, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.7*local_scale, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.5*local_scale, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.3*local_scale, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.05*local_scale, bounds=[-1, 1, -1, 1])])

    space = pclib.PCNNsqv2(N=N,
                           Nj=len(gcn),
                           gain=10.,
                           offset=1.1,
                           clip_min=0.01,
                           threshold=0.35,
                           rep_threshold=0.9,
                           rec_threshold=40.,
                           num_neighbors=5,
                           xfilter=gcn,
                           name="2D")


    assert space.get_size() == N, f"space length is {space.get_size()}"

    upc, ugc = space([0.5, 0.5])

    assert len(upc) == N, f"upc length is {len(upc)}"

    space.update()

    cell_count = len(space)

    assert cell_count == 1, f"cell count is {cell_count}"
