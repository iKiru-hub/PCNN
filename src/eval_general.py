import numpy as np
import matplotlib.pyplot as plt
import argparse, json
from tqdm import tqdm
import os

import pandas as pd
import scikit_posthocs as ph
from scipy import stats

import game.envs as games

import simulations as sim
import utils


""" settings """

logger = utils.setup_logger(__name__, level=5, is_debugging=False)


GAME_SCALE = games.SCREEN_WIDTH
ENVS = ['Square.v0', 'Flat.0010', 'Flat.1100', 'Flat.1111']
MIN_WEIGHT = 0.01


reward_settings = {
    "rw_fetching": "deterministic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.05 * GAME_SCALE,
    "rw_sigma": 0.8,# * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 2,
    "silent_duration": 5_000,
    "fetching_duration": 2,
    "transparent": False,
    "beta": 40.,
    "alpha": 0.06,# * GAME_SCALE,
    "move_threshold": 4,# * GAME_SCALE,
    "rw_position_idx": 0,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "rendering": False,
    "rendering_pcnn": False,
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "max_duration": 12_500,
    "room_thickness": 20,
    "t_teleport": 1_300,
    "limit_position_len": -1,
    "start_position_idx": 0,
    "seed": None,
    "pause": -1,
    "verbose": False
}

global_parameters = {
    "local_scale": 0.015,
    "N": 23**2,
    "use_sprites": bool(0),
    "speed": 1.,
    "min_weight_value": 0.5
}

# parameters = {
#       "gain": 102.4,
#       "offset": 1.02,
#       "threshold": 0.2,
#       "rep_threshold": 0.955,
#       "rec_threshold": 33,
#       "tau_trace": 10,
#       "remap_tag_frequency": 1,
#       "num_neighbors": 4,
#       "min_rep_threshold": 0.99,
#       "lr_da": 0.9,
#       "lr_pred": 0.95,
#       "threshold_da": 0.10,
#       "tau_v_da": 1.0,
#       "lr_bnd": 0.9,
#       "threshold_bnd": 0.1,
#       "tau_v_bnd": 1.0,
#       "tau_ssry": 437.0,
#       "threshold_ssry": 1.986,
#       "threshold_circuit": 0.9,
#       "rwd_weight": -0.11,
#       "rwd_sigma": 96.8,
#       "rwd_threshold": 0.,
#       "col_weight": -0.53,
#       "col_sigma": 16.1,
#       "col_threshold": 0.37,
#       "rwd_field_mod": 4.6,
#       "col_field_mod": 4.4,
#       "action_delay": 120.0,
#       "edge_route_interval": 50,
#       "forced_duration": 19,
#       "min_weight_value": 0.1,
#     "modulation_options": [True]*4
# }

parameters = utils.load_parameters(idx=90)

def calc_gains(brain: object, threshold: float=0.03):

    # --- indexes
    da = brain.get_da_weights()
    daidx = np.where(da>threshold)[0]
    bnd = brain.get_bnd_weights()
    bndidx = np.where(bnd>threshold)[0]
    noidx = [i for i in np.arange(len(brain.space)).tolist() if i not in bndidx and i not in daidx]

    # --- gains
    gains = brain.get_gain()
    no_g = gains[noidx]
    bnd_g = gains[bndidx]
    da_g = gains[daidx]

    return len(no_g), np.mean(no_g), len(bnd_g), np.mean(bnd_g), len(da_g), np.mean(da_g)


def run_multiple_environments(reps: int):

    record = {"collisions": [],
              "rewards": [],
              "len_no_g": [],
              "mean_no_g": [],
              "len_da_g": [],
              "mean_da_g": [],
              "len_bnd_g": [],
              "mean_bnd_g": []}


    rbar = tqdm(range(reps))

    for rep_i in rbar:

        rbar.set_description(f"rep={rep_i} [{reps}]")

        for room in ENVS:
            out, info = sim.run_model(parameters=parameters,
                                      global_parameters=global_parameters,
                                      reward_settings=reward_settings,
                                      game_settings=game_settings,
                                      room_name=room,
                                      record_flag=True,
                                      limit_position_len=2,
                                      verbose=False,
                                      verbose_min=False,
                                      pause=game_settings["pause"])
            env_ = info['env']
            record['collisions'] += [float(env_.nb_collisions)]
            record['rewards'] += [float(out)]
            gain_info = calc_gains(info['brain'])
            record['len_no_g'] += [int(gain_info[0])]
            record['mean_no_g'] += [float(gain_info[1])]
            record['len_bnd_g'] += [int(gain_info[2])]
            record['mean_bnd_g'] += [float(gain_info[3])]
            record['len_da_g'] += [int(gain_info[4])]
            record['mean_da_g'] += [float(gain_info[5])]

    logger()
    return record


def save(file: dict):

    name = utils.DATA_PATH + "/eval_general_data"

    num = len([f for f in os.listdir(utils.DATA_PATH) if "eval_general_data" in f])

    with open(f"{name}_{num}.json", 'w') as f:
        json.dump(file, f)

    logger("[file saved]")



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    # -- run

    logger("starting simulation..")
    record = run_multiple_environments(reps=args.reps)
    logger("[simulation done]")

    # -- save
    if save:

        save(file=record)

    logger("[done]")

