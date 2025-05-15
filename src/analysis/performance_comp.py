import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os, sys, json
import subprocess
from tqdm import tqdm
import time

sys.path.append(os.path.join(os.getcwd().split("PCNN")[0], "PCNN/src/"))

import utils
import core.build.pclib as pclib
from game.envs import *
from game.constants import ROOMS, GAME_SCALE
import simulations as sim

logger = utils.setup_logger(__name__, level=3)


""" SETTINGS """

reward_settings = {
    "rw_fetching": "probabilistic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.08 * GAME_SCALE,
    "rw_sigma": 0.7,# * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 5,
    "silent_duration": 10_000,
    "fetching_duration": 2,
    "transparent": False,
    "beta": 35.,
    "alpha": 0.06,# * GAME_SCALE,
    "move_threshold": 600,# * GAME_SCALE,
    "rw_position_idx": -1,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "rendering": False,
    "max_duration": 20_000,
    "room_thickness": 20,
    "t_teleport": 1_500,
    "limit_position_len": -1,
    "start_position_idx": 1,
    "seed": None,
    "pause": -1,
    "verbose": False,
    "verbose_min": False
}

global_parameters = {
    "local_scale": 0.015,
    "N": 30**2,
    "use_sprites": False,
    "speed": 0.7,
    "min_weight_value": 0.5
}

PARAMETERS = {
      "gain": 80.0,
      "offset": 1.0,
      "threshold": 0.1,
      "rep_threshold": 0.999,
      "rec_threshold": 70,
      "tau_trace": 20,
      "remap_tag_frequency": 1,
      "num_neighbors": 16,
      "min_rep_threshold": 0.99,

      "lr_da": 0.9,
      "lr_pred": 0.12,
      "threshold_da": 0.05,
      "tau_v_da": 1.0,
      "lr_bnd": 0.9,
      "threshold_bnd": 0.05,
      "tau_v_bnd": 2.0,

      "tau_ssry": 437.0,
      "threshold_ssry": 1.986,
      "threshold_circuit": 0.9,

      "rwd_weight": -2.68,
      "rwd_sigma": 67.7,
      "col_weight": 3.14,
      "col_sigma": 27.8,
      "rwd_field_mod": 3.0,
      "col_field_mod": 0.9,

      "action_delay": 100.0,
      "edge_route_interval": 100,
      "forced_duration": 19,
      "min_weight_value": 0.01
}

# PARAMETERS_NOREMAP = {
#       "gain": 18.6,
#       "offset": 1.0,
#       "threshold": 0.4,
#       "rep_threshold": 0.82,
#       "rec_threshold": 55,
#       "tau_trace": 20,
#       "remap_tag_frequency": 1,
#       "num_neighbors": 3,
#       "min_rep_threshold": 0.99,
#       "lr_da": 0.9,
#       "lr_pred": 0.44,
#       "threshold_da": 0.05,
#       "tau_v_da": 4.0,
#       "lr_bnd": 0.9,
#       "threshold_bnd": 0.2,
#       "tau_v_bnd": 3.0,
#       "tau_ssry": 437.0,
#       "threshold_ssry": 1.986,
#       "threshold_circuit": 0.9,
#       "rwd_weight": 0.0,
#       "rwd_sigma": 0.0,
#       "col_weight": 0.0,
#       "col_sigma": 0.0,
#       "rwd_field_mod": 1.0,
#       "col_field_mod": 1.0,
#       "action_delay": 120.0,
#       "edge_route_interval": 5275,
#       "forced_duration": 19,
#       "min_weight_value": 0.1
# }



""" FUNCTIONS """

OPTIONS = ["chance", "no_remap", "default",
           "only_da", "only_col"]
NUM_OPTIONS = len(OPTIONS)
ROOM_NAME = "Square.v0"
# ROOM_NAME = "Flat.1000"


def change_parameters(params: dict, name: int):

    params['modulation_option'] = [True] * 4

    # baseline
    if name == "chance":
        params["lr_da"] = 0.
        params["lr_pred"] = 0.
        params["lr_bnd"] = 0.
        return params

    # no remap option
    if name == "no_remap":
        params['modulation_option'] = [False] * 4
        params["rwd_weight"] = 0.
        params["rwd_sigma"] = 0.
        params["col_weight"] = 0.
        params["col_sigma"] = 0.
        params["col_field_mod_fine"] = 1.
        params["col_field_mod_coarse"] = 1.
        return params

    # default
    if name == "default":
        return params

    # only da remap
    if name == "only_da":
        params["col_weight"] = 0
        params["col_field_mod_fine"] = 1.
        params["col_field_mod_coarse"] = 1.
        return params

    # only col remap
    if name == "only_col":
        params["rwd_weight"] = 0
        params["rwd_field_mod_fine"] = 1.
        params["rwd_field_mod_coarse"] = 1.
        return params


def convert_numpy(obj):

    """Helper function to convert NumPy arrays to lists for JSON serialization."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def safe_run_model(params, room_name):
    """Runs sim.run_model in a subprocess to prevent crashes."""
    try:
        # Convert agent parameters to JSON, handling NumPy arrays
        params = json.dumps(params, default=convert_numpy)

        # Convert all other settings while handling NumPy arrays
        global_params_json = json.dumps(global_parameters, default=convert_numpy)
        reward_settings_json = json.dumps(reward_settings, default=convert_numpy)
        game_settings_json = json.dumps(game_settings, default=convert_numpy)

        # Construct the command to execute sim.run_model in a subprocess
        command = [
            "python3", "-c",
            f"""
import os, sys
os.chdir("".join((os.getcwd().split("PCNN")[0], "/PCNN/src/")))
import json, simulations, numpy as np
params = json.loads('{params}')
global_parameters = json.loads('{global_params_json}')
reward_settings = json.loads('{reward_settings_json}')
game_settings = json.loads('{game_settings_json}')
result, _ = simulations.run_model(
    parameters=params,
    global_parameters=global_parameters,
    reward_settings=reward_settings,
    game_settings=game_settings,
    room_name='{room_name}',
    verbose=False,
    verbose_min=False
)
print(result)
            """
        ]

        result = subprocess.run(command, check=True, capture_output=True, text=True)

        # Extract the last line of output (assuming sim.run_model()
        # prints only the result last)
        last_line = result.stdout.strip().split("\n")[-1]

        return float(last_line)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error: sim.run_model() crashed with: {e.stderr}")
        return 0
    except ValueError as e:
        logger.warning(f"Warning: sim.run_model() returned unexpected output: {result.stdout.strip()}")
        return 0


def run_local_model(args) -> list:

    results = []

    logger(f"{ROOM_NAME=}")

    for i in tqdm(range(NUM_OPTIONS)):
        params = change_parameters(PARAMETERS.copy(), OPTIONS[i])
        results += [safe_run_model(params, ROOM_NAME)]
        # if OPTIONS[i] == 'no_remap':
        #     results += [safe_run_model(PARAMETERS_NOREMAP, ROOM_NAME)]
        # else:
        #     params = change_parameters(PARAMETERS.copy(), OPTIONS[i])
        #     results += [safe_run_model(params, ROOM_NAME)]

    return results


def update_room_name(room_name):
    global ROOM_NAME
    ROOM_NAME = room_name


if __name__ == "__main__":

    """ args """

    import argparse

    parser = argparse.ArgumentParser(description='study remapping')
    parser.add_argument('--reps', type=int, default=4,
                        help='Number of repetitions')
    parser.add_argument('--cores', type=int, default=4,
                        help='Number of cores')
    parser.add_argument('--save', action='store_true',
                        help='save the results')
    parser.add_argument('--load_idx', type=int, default=-1,
                        help='agent index')

    args = parser.parse_args()

    """ setup """

    NUM_CORES = args.cores
    chunksize = args.reps
    NUM_REPS = NUM_CORES * chunksize

    if args.load_idx > -1:
        idx = int(args.load_idx)
        assert isinstance(idx, int), "agent idx must be int"
        PARAMETERS = utils.load_parameters(idx)
        logger.debug(f"LOADED: {PARAMETERS}")

    logger(f"save={args.save}")
    logger(f"{NUM_CORES=}")
    logger(f"{NUM_REPS=}")
    logger(f"{chunksize=}")
    logger(f"running...")

    """ parallel computation """

    with Pool(processes=NUM_CORES) as pool:
        results = list(
            tqdm(pool.imap(run_local_model,
                           [None] * NUM_REPS,
                           chunksize=chunksize),
                 total=NUM_REPS)
        )

    logger("run finished")

    """ save results """

    if not args.save:
        sys.exit(0)

    # prepare results
    data = {
        "options": OPTIONS,
        "room": ROOM_NAME,
        "parameters": PARAMETERS,
        "global_parameters": global_parameters,
        "results": results
    }

    logger("saving results...")

    # save data
    localtime = time.localtime()
    dataname = os.path.join(os.getcwd().split("PCNN")[0],
                         "PCNN/src/analysis/results/options_eval_")
    dataname += f"{localtime.tm_mday}{localtime.tm_mon}_{localtime.tm_hour}{localtime.tm_min}.json"
    with open(dataname, "w") as f:
        json.dump(data, f)

    logger(f"saved in '{dataname}'")


