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
    "rw_sigma": 0.5,# * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 5,
    "silent_duration": 5_000,
    "fetching_duration": 2,
    "transparent": False,
    "beta": 35.,
    "alpha": 0.06,# * GAME_SCALE,
    "tau": 500,# * GAME_SCALE,
    "move_threshold": 3000,# * GAME_SCALE,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "rendering": False,
    "max_duration": 20_000,
    "room_thickness": 20,
    "t_teleport": 2_000,
    "seed": None,
    "pause": -1,
    "limit_position_len": -1,
    "verbose": False,
    "verbose_min": False
}

global_parameters = {
    "local_scale_fine": 0.015,
    "local_scale_coarse": 0.006,
    "N": 31**2,
    "Nc": 29**2,
    "use_sprites": False,
    "speed": 1.,
    "min_weight_value": 0.5
}

PARAMETERS2 = {
    "gain_fine": 33.3,
    "offset_fine": 1.0,
    "threshold_fine": 0.3,
    "rep_threshold_fine": 0.86,
    "rec_threshold_fine": 50.0,
    "tau_trace_fine": 20.0,

    "remap_tag_frequency": 23,
    "num_neighbors": 10,
    "min_rep_threshold": 35,

    "gain_coarse": 46.5,
    "offset_coarse": 1.0,
    "threshold_coarse": 0.4,
    "rep_threshold_coarse": 0.77,
    "rec_threshold_coarse": 120.0,
    "tau_trace_coarse": 30.0,

    "lr_da": 0.99,
    "lr_pred": 0.3,
    "threshold_da": 0.03,
    "tau_v_da": 4.0,

    "lr_bnd": 0.6,
    "threshold_bnd": 0.3,
    "tau_v_bnd": 4.0,

    "tau_ssry": 100.0,
    "threshold_ssry": 0.995,

    "threshold_circuit": 0.9,
    "remapping_flag": 5,

    "rwd_weight": 4.91,
    "rwd_sigma": 50.0,
    "col_weight": -0.29,
    "col_sigma": 35.0,

    "rwd_field_mod_fine": 1.8,
    "rwd_field_mod_coarse": 1.8,
    "col_field_mod_fine": 0.7,
    "col_field_mod_coarse": 0.5,

    "action_delay": 140.0,
    "edge_route_interval": 3,
    "forced_duration": 100,
    "fine_tuning_min_duration": 10
}

PARAMETERS = {
    "gain_fine": 19.7,
    "offset_fine": 1.0,
    "threshold_fine": 0.4,
    "rep_threshold_fine": 0.8,
    "rec_threshold_fine": 54,
    "tau_trace_fine": 107,
    "remap_tag_frequency": 3,
    "num_neighbors": 15,
    "min_rep_threshold": 0.59,
    "gain_coarse": 25.7,
    "offset_coarse": 1.0,
    "threshold_coarse": 0.4,
    "rep_threshold_coarse": 0.8,
    "rec_threshold_coarse": 36,
    "tau_trace_coarse": 18,
    "lr_da": 0.99,
    "lr_pred": 0.3,
    "threshold_da": 0.04,
    "tau_v_da": 2.0,
    "lr_bnd": 0.6,
    "threshold_bnd": 0.3,
    "tau_v_bnd": 4.0,
    "tau_ssry": 223.0,
    "threshold_ssry": 0.997,
    "threshold_circuit": 0.9,
    "remapping_flag": 2,
    "rwd_weight": 4.3,
    "rwd_sigma": 88.9,
    "col_weight": -2.55,
    "col_sigma": 3.2,
    "rwd_field_mod_fine": -0.7,
    "rwd_field_mod_coarse": 0.2,
    "col_field_mod_fine": 0.7,
    "col_field_mod_coarse": -1.4,
    "action_delay": 120.0,
    "edge_route_interval": 5,
    "forced_duration": 1,
    "fine_tuning_min_duration": 55
}



""" FUNCTIONS """

OPTIONS = ["baseline", "DA-d", "DA-r", "BND-d", "BND-r"]
NUM_OPTIONS = len(OPTIONS)
ROOM_NAME = "Flat.0010"


def change_parameters(params: dict, name: int):

    # baseline
    if name == "DA-d":
        params['modulation_option'] = [True] + [False] * 3
        return params

    # no remap option
    if name == "DA-r":
        params['modulation_option'] = [False, True] + [False] * 2
        return params

    # default
    if name == "BND-d":
        params['modulation_option'] = [False] * 2 + [True, False]
        return params

    # only da remap
    if name == "BND-r":
        params['modulation_option'] = [False] * 3 + [True]
        return params

    # only col remap
    if name == "baseline":
        params['modulation_option'] = [True] * 4
        return params
    else:
        raise NameError(f"{name=}??")


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
    parser.add_argument("--room", type=str, default="none",
                        help=f'room name: {ROOMS} or `random`')
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

    if args.room == "random":
        update_room_name(get_random_room())
        logger(f"random room: {ROOM_NAME}")
    elif args.room in ROOMS:
        update_room_name(args.room)
        logger(f"room: {args.room}")
    elif args.room != "none":
        logger.warning(f"default room: {ROOM_NAME}")


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
                         "PCNN/src/analysis/results/options_eval_v2_")
    dataname += f"{localtime.tm_mday}{localtime.tm_mon}_{localtime.tm_hour}{localtime.tm_min}.json"
    with open(dataname, "w") as f:
        json.dump(data, f)

    logger(f"saved in '{dataname}'")



