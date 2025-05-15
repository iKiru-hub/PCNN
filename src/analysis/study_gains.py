import numpy as np
import multiprocessing as mp
import json
import os, sys, time
from functools import partial

sys.path.append(os.path.join(os.getcwd().split("PCNN")[0], "PCNN/src/"))

import utils
import core.build.pclib as pclib
from game.envs import *
from game.constants import ROOMS, GAME_SCALE
import simulations as sim

try:
    # import libs.pclib as pclib
    import core.build.pclib as pclib
    import libs.pclib2 as pclib2
except ImportError:
    import warnings
    warnings.warn("pclib [c++] not found, using python version")
    import libs.pclib1 as pclib

logger = utils.setup_logger('AG', level=2, is_debugging=False)


""" SETTINGS """

reward_settings = {
    "rw_fetching": "deterministic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.05 * GAME_SCALE,
    "rw_sigma": 0.8,# * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 120,
    "silent_duration": 15_000,
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
    "rendering": True,
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "max_duration": 50_000,
    "room_thickness": 30,
    "t_teleport": 2_000,
    "limit_position_len": -1,
    "start_position_idx": 1,
    "seed": None,
    "pause": -1,
    "verbose": True
}

global_parameters = {
    "local_scale": 0.015,
    "N": 30**2,
    "use_sprites": bool(0),
    "speed": 0.7,
    "min_weight_value": 0.5
}

parameters = {
  "gain": 140.0,
  "offset": 1.0,
  "threshold": 0.0,
  "rep_threshold": 0.99,
  "rec_threshold": 50,
  "tau_trace": 99,
  "remap_tag_frequency": 1,
  "num_neighbors": 12,
  "min_rep_threshold": 0.99,

  "lr_da": 0.9,
  "lr_pred": 0.05,
  "threshold_da": 0.05,
  "tau_v_da": 1.0,
  "lr_bnd": 0.9,
  "threshold_bnd": 0.1,
  "tau_v_bnd": 1.0,

  "tau_ssry": 437.0,
  "threshold_ssry": 1.986,
  "threshold_circuit": 0.9,

  "rwd_weight": -2.22,
  "rwd_sigma": 50.4,
  "col_weight": -4.05,
  "col_sigma": 51.5,
  "rwd_field_mod": 1.5,
  "col_field_mod": 5.3,
  "modulation_option": [True] * 4,

  "action_delay": 100.0,
  "edge_route_interval": 15,
  "forced_duration": 19,
  "min_weight_value": 0.01,
}

logger()

def calc_gains(brain: object):

    # --- indexes
    da = brain.get_da_weights()
    daidx = np.where(da>0.05)[0]
    bnd = brain.get_bnd_weights()
    bndidx = np.where(bnd>0.05)[0]
    noidx = [i for i in np.arange(len(brain.space)).tolist() if i not in bndidx and i not in daidx]

    # --- gains
    gains = brain.get_gain()
    no_g = gains[noidx]
    bnd_g = gains[bndidx]
    da_g = gains[daidx]

    return len(no_g), np.mean(no_g), len(bnd_g), np.mean(bnd_g), len(da_g), np.mean(da_g)


def run_experiment(rep_i, room, parameters, global_parameters, reward_settings, game_settings):
    """Runs a single experiment for a given repetition and room."""
    logger(f"REP={rep_i}, {room=}")
    out, info = sim.run_model(parameters=parameters,
                                global_parameters=global_parameters,
                                reward_settings=reward_settings,
                                game_settings=game_settings,
                                room_name=room,
                                record_flag=True,
                                limit_position_len=2,
                                verbose=False,
                                verbose_min=False,
                                pause=game_settings.get("pause"))
    env_ = info['env']
    gain_info = calc_gains(info['brain'])
    return {
        'collisions': env_.nb_collisions,
        'rewards': out,
        'len_no_g': gain_info[0],
        'mean_no_g': gain_info[1],
        'len_bnd_g': gain_info[2],
        'mean_bnd_g': gain_info[3],
        'len_da_g': gain_info[4],
        'mean_da_g': gain_info[5]
    }

if __name__ == "__main__":

    """ args """

    import argparse

    parser = argparse.ArgumentParser(description='study gains')
    parser.add_argument('--reps', type=int, default=4,
                        help='Number of repetitions')
    parser.add_argument('--cores', type=int, default=4,
                        help='Number of cores')
    parser.add_argument('--save', action='store_true',
                        help='save the results')

    args = parser.parse_args()


    """ settings """

    rooms = ['Square.v0', 'Flat.0010', 'Flat.1000', 'Flat.1001', 'Flat.0011']

    game_settings['rendering'] = False
    reward_settings["silent_duration"] = 10_000
    game_settings["max_duration"] = 40_000

    # Create a list of all experiment configurations
    tasks = [(rep_i, room) for rep_i in range(args.reps) for room in rooms]

    # Use a pool of worker processes
    num_cores = args.cores
    logger(f"Running on {num_cores} cores.")
    pool = mp.Pool(processes=num_cores)

    """ run """

    # Prepare the function with static arguments
    partial_run_experiment = partial(run_experiment,
                                     parameters=parameters,
                                     global_parameters=global_parameters,
                                     reward_settings=reward_settings,
                                     game_settings=game_settings)

    # Run the experiments in parallel
    results = pool.starmap(partial_run_experiment, tasks)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    logger("run finished")

    """ save """

    record_final = {
        "collisions": [int(res['collisions']) for res in results],
        "rewards": [float(res['rewards']) for res in results],
        "len_no_g": [int(res['len_no_g']) for res in results],
        "mean_no_g": [float(res['mean_no_g']) for res in results],
        "len_da_g": [int(res['len_da_g']) for res in results],
        "mean_da_g": [float(res['mean_da_g']) for res in results],
        "len_bnd_g": [int(res['len_bnd_g']) for res in results],
        "mean_bnd_g": [float(res['mean_bnd_g']) for res in results]
    }

    # Save the results to a JSON file
    output_filename = f"results_gains_{time.localtime().tm_hour}{time.localtime().tm_min}.json"
    with open(output_filename, 'w') as f:
        json.dump(record_final, f, indent=4)

    logger(f"Results saved to {output_filename}")
