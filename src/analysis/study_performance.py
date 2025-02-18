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
    # "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.05 * GAME_SCALE,
    "rw_sigma": 0.75 * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 5,
    "silent_duration": 2_000,
    "fetching_duration": 1,
    "transparent": False,
    "beta": 35.,
    "alpha": 0.06,
}

agent_settings = {
    "init_position": np.array([0.2, 0.2]) * GAME_SCALE,
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "rendering": False,
    "rendering_pcnn": False,
    "max_duration": 8_000,
    "room_thickness": 5,
    "seed": None,
    "pause": -1,
    "verbose": False,
    "verbose_min": False
}

global_parameters = {
    "local_scale_fine": 0.015,
    "local_scale_coarse": 0.006,
    "N": 22**2,
    "Nc": 12**2,
    # "rec_threshold_fine": 26.,
    # "rec_threshold_coarse": 60.,
    "speed": 0.75,
    "min_weight_value": 0.5
}

PARAMETERS = {

    "gain_fine": 10.,
    "offset_fine": 1.0,
    "threshold_fine": 0.3,
    "rep_threshold_fine": 0.88,
    "rec_threshold_fine": 60.,
    "tau_trace_fine": 10.0,
    "min_rep_threshold": 0.95,

    "gain_coarse": 9.,
    "offset_coarse": 1.0,
    "threshold_coarse": 0.3,
    "rep_threshold_coarse": 0.9,
    "rec_threshold_coarse": 100.,
    "tau_trace_coarse": 20.0,

    "lr_da": 0.8,
    "threshold_da": 0.03,
    "tau_v_da": 1.0,

    "lr_bnd": 0.4,
    "threshold_bnd": 0.05,
    "tau_v_bnd": 2.0,

    "tau_ssry": 100.,
    "threshold_ssry": 0.998,

    "threshold_circuit": 0.1,

    "rwd_weight": 0.1,
    "rwd_sigma": 40.0,
    "col_weight": 0.0,
    "col_sigma": 2.0,

    "action_delay": 50.,
    "edge_route_interval": 50,

    "forced_duration": 100,
    "fine_tuning_min_duration": 10,
}


TOT_VALUES = 4
ROOM_NAME = "Square.v0"


""" FUNCTIONS """


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
        agent_settings_json = json.dumps(agent_settings, default=convert_numpy)
        reward_settings_json = json.dumps(reward_settings, default=convert_numpy)
        game_settings_json = json.dumps(game_settings, default=convert_numpy)

        # Construct the command to execute sim.run_model in a subprocess
        command = [
            "python3", "-c",
            f"""
import sys, os
sys.path.append(os.path.join(os.getcwd().split("PCNN")[0], "PCNN/src/"))
import json, simulations, numpy as np
params = json.loads('{params}')
global_parameters = json.loads('{global_params_json}')
agent_settings = json.loads('{agent_settings_json}')
reward_settings = json.loads('{reward_settings_json}')
game_settings = json.loads('{game_settings_json}')
result = simulations.run_model(
    parameters=params,
    global_parameters=global_parameters,
    agent_settings=agent_settings,
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

        # Extract the last line of output (assuming sim.run_model() prints only the result last)
        last_line = result.stdout.strip().split("\n")[-1]

        return float(last_line)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error: sim.run_model() crashed with: {e.stderr}")
        return 0
    except ValueError as e:
        logger.warning(f"Warning: sim.run_model() returned unexpected output: {result.stdout.strip()}")
        return 0


def run_local_model(args):
    return safe_run_model(PARAMETERS, ROOM_NAME)




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
    parser.add_argument("--room", type=str, default="Square.v0",
                        help=f'room name: {ROOMS} or `random`')
    parser.add_argument("--load", action="store_true")

    args = parser.parse_args()

    """ setup """

    NUM_CORES = args.cores
    NUM_REPS = args.reps

    if args.load:
        PARAMETERS = utils.load_parameters()
        logger.debug(f"LOADED: {PARAMETERS}")

    if args.room == "random":
        ROOM_NAME = get_random_room()
        logger(f"random room: {ROOM_NAME}")
    elif args.room in ROOMS:
        ROOM_NAME = args.room
        logger(f"room: {args.room}")
    else:
        logger.warning(f"default room: {ROOM_NAME}")

    """ parallel computation """

    chunksize = NUM_REPS // NUM_CORES  # Divide the workload evenly

    logger(f"save={args.save}")
    logger(f"{NUM_CORES=}")
    logger(f"{NUM_REPS=}")
    logger(f"{chunksize=}")
    logger(f"running...")

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

    logger("saving results...")

    localtime = time.localtime()
    name = f"results/performance_"
    name += f"{localtime.tm_mday}{localtime.tm_mon}_{localtime.tm_hour}{localtime.tm_min}"

    # make folder
    if not os.path.exists(f"results/{name}"):
        os.makedirs(name)

    # save data
    dataname = name + "/data.json"
    with open(dataname, "w") as f:
        json.dump(results, f)

    # save values
    metadata = {
        "room": ROOM_NAME,
        "loaded": args.load,
        "parameters": PARAMETERS,
    }
    with open(name + "/metadata.json", "w") as f:
        json.dump(metadata, f)





















