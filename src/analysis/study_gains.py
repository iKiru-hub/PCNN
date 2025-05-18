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
    "rw_position_idx": 1,
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

def calc_da_pos(brain: object):

    #
    da = brain.get_da_weights()
    daidx = np.where(da>0.05)[0]

    centers = brain.get_space_centers()[daidx]

    return centers

def run_experiments_for_reps(reps_range, room, parameters, global_parameters,
                             reward_settings, game_settings, total_reps):
    """Runs a batch of repetitions for a single room."""
    all_room_records = {
        "collisions": [],
        "rewards": [],
        "len_no_g": [],
        "mean_no_g": [],
        "len_da_g": [],
        "mean_da_g": [],
        "len_bnd_g": [],
        "mean_bnd_g": []
    }
    for rep_i in reps_range:
        logger(f"REP={rep_i+1}/{total_reps}, {room=}")
        out, info = sim.run_model(parameters=parameters,
                                    global_parameters=global_parameters,
                                    reward_settings=reward_settings,
                                    game_settings=game_settings,
                                    room_name=room,
                                    record_flag=True,
                                    limit_position_len=2,
                                    verbose=False,
                                    verbose_min=False,
                                    pause=game_settings.get("pause"),
                                    )
        env_ = info['env']
        gain_info = calc_gains(info['brain'])
        all_room_records['collisions'].append(int(env_.nb_collisions))
        all_room_records['rewards'].append(float(out))
        all_room_records['len_no_g'].append(int(gain_info[0]))
        all_room_records['mean_no_g'].append(float(gain_info[1]))
        all_room_records['len_bnd_g'].append(int(gain_info[2]))
        all_room_records['mean_bnd_g'].append(float(gain_info[3]))
        all_room_records['len_da_g'].append(int(gain_info[4]))
        all_room_records['mean_da_g'].append(float(gain_info[5]))
    return all_room_records


if __name__ == "__main__":
    """ args """
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
    reward_settings["silent_duration"] = 10
    game_settings["max_duration"] = 40

    total_reps = args.reps
    num_cores = args.cores
    logger(f"Running {total_reps} repetitions across {num_cores} cores.")
    pool = mp.Pool(processes=num_cores)

    # Calculate how to split the repetitions
    reps_per_core = [total_reps // num_cores] * num_cores
    for i in range(total_reps % num_cores):
        reps_per_core[i] += 1

    # Create tasks for each core
    tasks = []
    start_rep = 0
    for i in range(num_cores):
        end_rep = start_rep + reps_per_core[i]
        rep_range = range(start_rep, end_rep)
        for room in rooms:
            tasks.append((rep_range, room))
        start_rep = end_rep

    # Prepare the function with static arguments
    partial_run_experiments = partial(run_experiments_for_reps,
                                       parameters=parameters,
                                       global_parameters=global_parameters,
                                       reward_settings=reward_settings,
                                       game_settings=game_settings,
                                       total_reps=total_reps)

    # Run the experiments in parallel
    results = pool.starmap(partial_run_experiments, tasks)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    logger("run finished")

    """ save """
    record_final = {
        "collisions": [],
        "rewards": [],
        "len_no_g": [],
        "mean_no_g": [],
        "len_da_g": [],
        "mean_da_g": [],
        "len_bnd_g": [],
        "mean_bnd_g": []
    }

    # Aggregate the results from each core's processing of rooms
    for res in results:
        record_final["collisions"].extend(res["collisions"])
        record_final["rewards"].extend(res["rewards"])
        record_final["len_no_g"].extend(res["len_no_g"])
        record_final["mean_no_g"].extend(res["mean_no_g"])
        record_final["len_da_g"].extend(res["len_da_g"])
        record_final["mean_da_g"].extend(res["mean_da_g"])
        record_final["len_bnd_g"].extend(res["len_bnd_g"])
        record_final["mean_bnd_g"].extend(res["mean_bnd_g"])

    # Save the results to a JSON file
    output_filename = f"results_gains_{time.localtime().tm_hour}{time.localtime().tm_min}.json"
    with open(output_filename, 'w') as f:
        json.dump(record_final, f, indent=4)

    logger(f"Results saved to {output_filename}")
