import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import core.build.pclib as pclib
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

    "gain_fine": 11.,
    "offset_fine": 1.2,
    "threshold_fine": 0.35,
    "rep_threshold_fine": 0.9,
    "rec_threshold_fine": 60.,
    "tau_trace_fine": 20.0,
    "min_rep_threshold": 0.95,
    "remap_tag_frequency": 1,
    "num_neighbors": 3,

    "gain_coarse": 8.,
    "offset_coarse": 0.9,
    "threshold_coarse": 0.3,
    "rep_threshold_coarse": 0.8,
    "rec_threshold_coarse": 100.,
    "tau_trace_coarse": 20.0,

    "lr_da": 0.3,
    "threshold_da": 0.04,
    "tau_v_da": 1.0,

    "lr_bnd": 0.4,
    "threshold_bnd": 0.05,
    "tau_v_bnd": 2.0,

    "tau_ssry": 100.,
    "threshold_ssry": 0.95,

    "threshold_circuit": 0.2,

    "rwd_weight": 0.03,
    "rwd_sigma": 10.0,
    "col_weight": 0.0,
    "col_sigma": 2.0,

    "action_delay": 50.,
    "edge_route_interval": 50,

    "forced_duration": 100,
    "fine_tuning_min_duration": 10,
}


def test_vspace():

    vspace = pclib.VelocitySpace(size=N, threshold=10.)

    out = vspace([0.5, 0.5])
    assert len(out) == 2, f"output length is {len(out)}, expected 2"

    vspace.update(2, N, np.random.rand(N, 1))
    c = vspace.get_centers()
    assert len(c) == N, f"center length is {len(c)}, expected {N}"


def test_gcn():

    local_scale_coarse = 0.1

    gcn_coarse = pclib.GridNetworkSq([
           pclib.GridLayerSq(sigma=0.04, speed=1.*local_scale_coarse,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.8*local_scale_coarse,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.7*local_scale_coarse,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.5*local_scale_coarse,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.3*local_scale_coarse,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.05*local_scale_coarse,
                             bounds=[-1, 1, -1, 1])])

    out = gcn_coarse([0.5, 0.5])

    assert len(out) == len(gcn_coarse), f"output length is {len(out)}, expected {len(gcn_coarse)}"


def test_space():


    gcn = pclib.GridNetworkSq([
           pclib.GridLayerSq(sigma=0.04, speed=1.*local_scale, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.8*local_scale, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.7*local_scale, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.5*local_scale, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.3*local_scale, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.05*local_scale, bounds=[-1, 1, -1, 1])])

    space = pclib.PCNN(N=N,
                       Nj=len(gcn),
                       gain=10.,
                       offset=1.1,
                       clip_min=0.01,
                       threshold=0.35,
                       rep_threshold=0.9,
                       rec_threshold=40.,
                       min_rep_threshold=parameters["min_rep_threshold"],
                       num_neighbors=parameters["num_neighbors"],
                       xfilter=gcn,
                       tau_trace=20.,
                       remap_tag_frequency=parameters["remap_tag_frequency"],
                       name="2D")


    assert space.get_size() == N, f"space length is {space.get_size()}"

    upc, ugc = space([0.5, 0.5])

    assert len(upc) == N, f"upc length is {len(upc)}"

    space.update()

    cell_count = len(space)

    assert cell_count == 1, f"cell count is {cell_count}"


    local_scale_fine = global_parameters["local_scale_fine"]
    local_scale_coarse = global_parameters["local_scale_coarse"]

    gcn = pclib.GridNetworkSq([
           pclib.GridLayerSq(sigma=0.04, speed=1.*local_scale_fine,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.8*local_scale_fine,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.7*local_scale_fine,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.5*local_scale_fine,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.3*local_scale_fine,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.05*local_scale_fine,
                             bounds=[-1, 1, -1, 1])])

    space_fine = pclib.PCNN(N=global_parameters["N"],
                            Nj=len(gcn),
                            gain=parameters["gain_fine"],
                            offset=parameters["offset_fine"],
                            clip_min=0.01,
                            threshold=parameters["threshold_fine"],
                            rep_threshold=parameters["rep_threshold_fine"],
                            rec_threshold=global_parameters["rec_threshold_fine"],
                            min_rep_threshold=parameters["min_rep_threshold"],
                            num_neighbors=parameters["num_neighbors"],
                            xfilter=gcn,
                            tau_trace=parameters["tau_trace_fine"],
                            remap_tag_frequency=parameters["remap_tag_frequency"],
                            name="2D")

    assert space_fine.get_size() == global_parameters["N"], f"space_fine length is {space_fine.get_size()}"

    space_coarse = pclib.PCNN(N=global_parameters["Nc"],
                           Nj=len(gcn),
                           gain=parameters["gain_coarse"],
                           offset=parameters["offset_coarse"],
                           clip_min=0.01,
                           threshold=parameters["threshold_coarse"],
                           rep_threshold=parameters["rep_threshold_coarse"],
                          rec_threshold=global_parameters["rec_threshold_coarse"],
                           num_neighbors=parameters["num_neighbors"],
                           min_rep_threshold=0.95,
                           xfilter=gcn,
                           name="2D")

    assert space_coarse.get_size() == global_parameters["Nc"], f"space_coarse length is {space_coarse.get_size()}"


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


def test_target_program():


    local_scale_fine = global_parameters["local_scale_fine"]
    local_scale_coarse = global_parameters["local_scale_coarse"]

    gcn = pclib.GridNetworkSq([
           pclib.GridLayerSq(sigma=0.04, speed=1.*local_scale_fine,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.8*local_scale_fine,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.7*local_scale_fine,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.5*local_scale_fine,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.3*local_scale_fine,
                             bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.05*local_scale_fine,
                             bounds=[-1, 1, -1, 1])])

    space_fine = pclib.PCNN(N=global_parameters["N"],
                            Nj=len(gcn),
                            gain=parameters["gain_fine"],
                            offset=parameters["offset_fine"],
                            clip_min=0.01,
                            threshold=parameters["threshold_fine"],
                            rep_threshold=parameters["rep_threshold_fine"],
                            rec_threshold=global_parameters["rec_threshold_fine"],
                            min_rep_threshold=parameters["min_rep_threshold"],
                            num_neighbors=parameters["num_neighbors"],
                            xfilter=gcn,
                            tau_trace=20.,
                            name="2D")

    da = pclib.BaseModulation(name="DA", size=global_parameters["N"],
                              lr=parameters["lr_da"],
                              threshold=parameters["threshold_da"],
                              max_w=1.0,
                              tau_v=1.0,
                              eq_v=0.0, min_v=0.0)


    bnd = pclib.BaseModulation(name="BND", size=global_parameters["N"],
                               lr=parameters["lr_bnd"],
                               threshold=parameters["threshold_bnd"],
                               max_w=1.0,
                               tau_v=1.0, eq_v=0.0, min_v=0.0)


    ssry = pclib.StationarySensory(global_parameters["N"],
                                   parameters["tau_ssry"],
                                   parameters["threshold_ssry"],
                                   0.99)
    circuit = pclib.Circuits(da, bnd, parameters["threshold_circuit"])

    # dpolicy = pclib.DensityPolicy(parameters["rwd_weight"],
    #                               parameters["rwd_sigma"],
    #                               parameters["col_weight"],
    #                               parameters["col_sigma"])

    # assert dpolicy.__str__() == "DensityPolicy", f"dpolicy is {dpolicy.__str__()}"

    expmd = pclib.ExplorationModule(speed=global_parameters["speed"]*2.0,
                                    circuits=circuit,
                                    space_fine=space_fine,
                                    action_delay=parameters["action_delay"],
                                    edge_route_interval=parameters["edge_route_interval"],)

    exout = expmd("new", -1)

    assert isinstance(exout, tuple), f"expmd output is {type(exout)}"


def test_brain():

    brain = pclib.Brain(
                local_scale_fine=global_parameters["local_scale_fine"],
                local_scale_coarse=global_parameters["local_scale_coarse"],
                N=global_parameters["N"],
                Nc=global_parameters["Nc"],
                min_rep_threshold=parameters["min_rep_threshold"],
                num_neighbors=parameters["num_neighbors"],
                rec_threshold_fine=parameters["rec_threshold_fine"],
                rec_threshold_coarse=parameters["rec_threshold_coarse"],
                speed=global_parameters["speed"],
                gain_fine=parameters["gain_fine"],
                offset_fine=parameters["offset_fine"],
                threshold_fine=parameters["threshold_fine"],
                rep_threshold_fine=parameters["rep_threshold_fine"],
                remap_tag_frequency=1,
                tau_trace_fine=parameters["tau_trace_fine"],
                gain_coarse=parameters["gain_coarse"],
                offset_coarse=parameters["offset_coarse"],
                threshold_coarse=parameters["threshold_coarse"],
                rep_threshold_coarse=parameters["rep_threshold_coarse"],
                tau_trace_coarse=parameters["tau_trace_coarse"],
                lr_da=parameters["lr_da"],
                lr_pred=0.4,
                threshold_da=parameters["threshold_da"],
                tau_v_da=parameters["tau_v_da"],
                lr_bnd=parameters["lr_bnd"],
                threshold_bnd=parameters["threshold_bnd"],
                tau_v_bnd=parameters["tau_v_bnd"],
                tau_ssry=parameters["tau_ssry"],
                threshold_ssry=parameters["threshold_ssry"],
                threshold_circuit=parameters["threshold_circuit"],
                remapping_flag=1,
                rwd_weight=parameters["rwd_weight"],
                rwd_sigma=parameters["rwd_sigma"],
                col_weight=parameters["col_weight"],
                col_sigma=parameters["col_sigma"],
                action_delay=parameters["action_delay"],
                edge_route_interval=parameters["edge_route_interval"],
                forced_duration=parameters["forced_duration"],
                fine_tuning_min_duration=parameters["fine_tuning_min_duration"],
                min_weight_value=parameters["fine_tuning_min_duration"])

    assert len(brain) == global_parameters["N"], f"brain length is {len(brain)}, expected {global_parameters['N']}"

