import numpy as np
import simulations as sim

import os
import json
import argparse

from game.constants import GAME_SCALE
from game.envs import get_random_room
import utils



reward_settings = {
    "rw_fetching": "probabilistic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_position_idx": 3,
    "rw_radius": 0.08 * GAME_SCALE,
    "rw_sigma": 0.8,# * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 5,
    "silent_duration": 10_000,
    "fetching_duration": 1,

    "transparent": False,
    "beta": 40.,
    "alpha": 0.06,# * GAME_SCALE,
    "move_threshold": 5,# * GAME_SCALE,
}


game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "rendering": False,
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "max_duration": 20_000,
    "t_teleport": 1_000,
    "limit_position_len": -1,
    "room_thickness": 20,
    "seed": None,
    "pause": -1,
    "verbose": True
}


sim_parameters_2 = {
    'gain_fine': 30.0,
    'offset_fine': 1.0,
    'threshold_fine': 0.4,
    'rep_threshold_fine': 0.89,
    'rec_threshold_fine': 32,
    'tau_trace_fine': 220,
    'remap_tag_frequency': 2,
    'min_rep_threshold': 0.92,
    'gain_coarse': 49.6,
    'offset_coarse': 1.0,
    'threshold_coarse': 0.4,
    'rep_threshold_coarse': 0.74,
    'rec_threshold_coarse': 40,
    'tau_trace_coarse': 203,

    'lr_da': 0.99,
    'lr_pred': 0.8,
    'threshold_da': 0.04,

    'tau_v_da': 2.0,
    'lr_bnd': 0.6,
    'threshold_bnd': 0.3,
    'tau_v_bnd': 4.0,

    'tau_ssry': 28.0,
    'threshold_ssry': 0.975,

    'threshold_circuit': 0.9,
    'remapping_flag': -1,
    'rwd_weight': 4.58,
    'rwd_sigma': 35.0,
    'col_weight': 1.4,
    'col_sigma': 30.9,
    'rwd_field_mod_fine': 0.9,
    'rwd_field_mod_coarse': -1.3,
    'col_field_mod_fine': 1.2,
    'col_field_mod_coarse': 1.0,
    'action_delay': 120.0,
    'edge_route_interval': 5,
    'forced_duration': 1,
    'fine_tuning_min_duration': 88
}


sim_parameters = {
      "gain": 18.6,
      "offset": 1.0,
      "threshold": 0.4,
      "rep_threshold": 0.82,
      "rec_threshold": 55,
      "tau_trace": 20,
      "remap_tag_frequency": 1,
      "num_neighbors": 3,
      "min_rep_threshold": 0.99,
      "lr_da": 0.9,
      "lr_pred": 0.44,
      "threshold_da": 0.05,
      "tau_v_da": 4.0,
      "lr_bnd": 0.9,
      "threshold_bnd": 0.2,
      "tau_v_bnd": 3.0,
      "tau_ssry": 437.0,
      "threshold_ssry": 1.986,
      "threshold_circuit": 0.9,
      "rwd_weight": 0.0,
      "rwd_sigma": 0.0,
      "col_weight": 0.0,
      "col_sigma": 0.0,
      "rwd_field_mod": 1.0,
      "col_field_mod": 1.0,
      "action_delay": 120.0,
      "edge_route_interval": 5275,
      "forced_duration": 19,
      "min_weight_value": 0.1
}


sim_parameters_3 = {
    "gain": 42.0,
    "offset": 1.0,
    "threshold": 0.4,
    "rep_threshold": 0.84,
    "rec_threshold": 94,
    "tau_trace": 20,
    "remap_tag_frequency": 1,
    "num_neighbors": 16,
    "min_rep_threshold": 0.99,
    "lr_da": 0.9,
    "lr_pred": 0.12,
    "threshold_da": 0.05,
    "tau_v_da": 4.0,
    "lr_bnd": 0.9,
    "threshold_bnd": 0.2,
    "tau_v_bnd": 3.0,
    "tau_ssry": 437.0,
    "threshold_ssry": 1.986,
    "threshold_circuit": 0.9,

    # "rwd_weight": -2.68,
    # "rwd_sigma": 67.7,
    # "col_weight": 3.14,
    # "col_sigma": 27.8,
    # "rwd_field_mod": 3.0,
    # "col_field_mod": 0.9,
    # "modulation_option": [True] * 4,

    "rwd_weight": 0.0,
    "rwd_sigma": 0.0,
    "col_weight": 0.0,
    "col_sigma": 0.0,
    "rwd_field_mod": 1.0,
    "col_field_mod": 1.0,
    "modulation_option": [False] * 4,

    "action_delay": 120.0,
    "edge_route_interval": 6991,
    "forced_duration": 19,
    "min_weight_value": 0.1,
}



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=-1)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--rendering", action="store_true")
    parser.add_argument("--interval", type=int, default=20,
                        help="plotting interval")
    parser.add_argument("--room", type=str, default="Square.v0",
                        help='room name: ["Square.v0", "Square.v1",' + \
                         ' "Square.v2","Hole.v0", "Flat.0000", "Flat.0001",' + \
                         '"Flat.0010", "Flat.0011", "Flat.0110", ' + \
                         '"Flat.1000", "Flat.1001", "Flat.1010",' + \
                         '"Flat.1011", "Flat.1110"] or `random`')
    args = parser.parse_args()


    """ setup """

    logger = utils.setup_logger(name="RUN",
                                level=2,
                                is_debugging=True,
                                is_warning=False)


    if args.duration > 0:
        game_settings["max_duration"] = args.duration

    if args.rendering:
        game_settings["rendering"] = True

    game_settings["plot_interval"] = args.interval
    reward_settings["transparent"] = args.transparent

    if args.load:
        parameters = utils.load_parameters(idx=args.idx-1)
        logger.debug(parameters.keys())
    else:
        logger.debug("using local parameters")
        # parameters = sim_parameters
        parameters = sim_parameters_3
        # parameters = sim_parameters_73

    if args.room == "random":
        args.room = get_random_room()
        logger(f"random room: {args.room}")

    """ run """

    parameters['modulation_option'] = [True] * 4

    logger("[@scratch.py]")
    out, _ = sim.run_model(parameters=parameters,
                           global_parameters=sim.global_parameters,
                           reward_settings=reward_settings,
                           game_settings=game_settings,
                           room_name=args.room,
                           limit_position_len=2,
                           verbose=False, pause=game_settings["pause"])

    logger(f"rw_count={out}")
