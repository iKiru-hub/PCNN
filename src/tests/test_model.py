import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import core.build.pclib as pclib
print(os.getcwd())
import libs.pclib2 as pclib2
import pytest
import numpy as np



""" global settings"""

N = 100
Nc = 50
local_scale = 0.1


global_parameters = {
    "local_scale_fine": 0.015,
    "local_scale_coarse": 0.006,
    "N": 30**2,
    "Nc": 20**2,
    "rec_threshold_fine": 25.,
    "rec_threshold_coarse": 60.,
    "speed": 1.5,
    "min_weight_value": 0.5
}


parameters = {
    "gain": 102.4,
    "offset": 1.02,
    "threshold": 0.2,
    "rep_threshold": 0.955,
    "rec_threshold": 33,
    "tau_trace": 10,
    "remap_tag_frequency": 1,
    "num_neighbors": 4,
    "min_rep_threshold": 0.99,
    "lr_da": 0.9,
    "lr_pred": 0.95,
    "threshold_da": 0.10,
    "tau_v_da": 1.0,
    "lr_bnd": 0.9,
    "threshold_bnd": 0.1,
    "tau_v_bnd": 1.0,
    "tau_ssry": 437.0,
    "threshold_ssry": 1.986,
    "threshold_circuit": 0.9,
    "rwd_weight": -0.11,
    "rwd_sigma": 96.8,
    "rwd_threshold": 0.,
    "col_weight": -0.53,
    "col_sigma": 16.1,
    "col_threshold": 0.37,
    "rwd_field_mod": 4.6,
    "col_field_mod": 4.4,
    "action_delay": 120.0,
    "edge_route_interval": 50,
    "forced_duration": 19,
    "min_weight_value": 0.1,
    "modulation_options": [True]*4
}



def test_gcn():

    local_scale_coarse = 0.1

    gcn_coarse = pclib.GridNetwork([
           pclib.GridLayer(sigma=0.04, speed=1.*local_scale_coarse,
                           bounds=[-1, 1, -1, 1]),
           pclib.GridLayer(sigma=0.04, speed=0.8*local_scale_coarse,
                           bounds=[-1, 1, -1, 1]),
           pclib.GridLayer(sigma=0.04, speed=0.7*local_scale_coarse,
                           bounds=[-1, 1, -1, 1]),
           pclib.GridLayer(sigma=0.04, speed=0.5*local_scale_coarse,
                           bounds=[-1, 1, -1, 1]),
           pclib.GridLayer(sigma=0.04, speed=0.3*local_scale_coarse,
                           bounds=[-1, 1, -1, 1]),
           pclib.GridLayer(sigma=0.04, speed=0.05*local_scale_coarse,
                           bounds=[-1, 1, -1, 1])])

    out = gcn_coarse([0.5, 0.5])

    assert len(out) == len(gcn_coarse), f"output length is {len(out)}, expected {len(gcn_coarse)}"




def test_modulation():

    da = pclib.BaseModulation(name="DA", size=global_parameters["N"],
                              lr=parameters["lr_da"],
                              lr_pred=0.4,
                              threshold=parameters["threshold_da"],
                              max_w=1.0,
                              tau_v=1.0,
                              eq_v=0.0, min_v=0.0)

    assert len(da) == global_parameters["N"], f"da length is {len(da)}"

    bnd = pclib.BaseModulation(name="BND", size=global_parameters["N"],
                               lr=parameters["lr_bnd"],
                               threshold=parameters["threshold_bnd"],
                               max_w=1.0,
                               tau_v=1.0, eq_v=0.0, min_v=0.0)

    assert len(bnd) == global_parameters["N"], f"bnd length is {len(bnd)}"

    ssry = pclib.StationarySensory(global_parameters["N"],
                                   parameters["tau_ssry"],
                                   parameters["threshold_ssry"],
                                   0.99)
    circuit = pclib.Circuits(da, bnd, parameters["threshold_circuit"])

    assert len(circuit) == 2, f"circuit length is {len(circuit)}, expected 2"



def test_brain():

    brain = pclib2.Brain(
                local_scale=0.001,
                N=global_parameters["N"],
                rec_threshold=0.9,
                speed=global_parameters["speed"],
                min_rep_threshold=parameters["min_rep_threshold"],
                gain=parameters["gain"],
                offset=parameters["offset"],
                threshold=parameters["threshold"],
                rep_threshold=parameters["rep_threshold"],
                tau_trace=parameters["tau_trace"],
                remap_tag_frequency=parameters["remap_tag_frequency"],
                lr_da=parameters["lr_da"],
                lr_pred=parameters["lr_pred"],
                threshold_da=parameters["threshold_da"],
                tau_v_da=parameters["tau_v_da"],
                lr_bnd=parameters["lr_bnd"],
                threshold_bnd=parameters["threshold_bnd"],
                tau_v_bnd=parameters["tau_v_bnd"],
                tau_ssry=parameters["tau_ssry"],
                threshold_ssry=parameters["threshold_ssry"],
                threshold_circuit=parameters["threshold_circuit"],
                rwd_weight=parameters["rwd_weight"],
                rwd_sigma=parameters["rwd_sigma"],
                rwd_threshold=parameters["rwd_threshold"],
                col_weight=parameters["col_weight"],
                col_sigma=parameters["col_sigma"],
                col_threshold=parameters["col_threshold"],
                rwd_field_mod=parameters["rwd_field_mod"],
                col_field_mod=parameters["col_field_mod"],
                action_delay=parameters["action_delay"],
                edge_route_interval=parameters["edge_route_interval"],
                forced_duration=parameters["forced_duration"],
                min_weight_value=parameters["min_weight_value"])

    assert len(brain) == 0, f"brain length is {len(brain)}, expected {global_parameters['N']}"

