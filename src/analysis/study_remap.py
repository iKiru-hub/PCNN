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


""" settings """

reward_settings = {
    "rw_fetching": "probabilistic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.1 * GAME_SCALE,
    "rw_sigma": 1.5 * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 150,
    "silent_duration": 5_000,
    "fetching_duration": 1,
    "transparent": False,
    "beta": 30.,
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
    "room_thickness": 30,
    "seed": None,
    "pause": -1,
    "verbose": False
}

global_parameters = {
    "local_scale_fine": 0.015,
    "local_scale_coarse": 0.006,
    "N": 28**2,
    "rec_threshold_fine": 24.,
    "rec_threshold_coarse": 70.,
    "speed": 1.5,
    "min_weight_value": 0.6
}

PARAMETERS = {

    "gain_fine": 11.,
    "offset_fine": 1.2,
    "threshold_fine": 0.4,
    "rep_threshold_fine": 0.9,

    "gain_coarse": 11.,
    "offset_coarse": 1.2,
    "threshold_coarse": 0.4,
    "rep_threshold_coarse": 0.89,

    "lr_da": 0.4,
    "threshold_da": 0.08,
    "tau_v_da": 1.0,

    "lr_bnd": 0.4,
    "threshold_bnd": 0.04,
    "tau_v_bnd": 1.0,

    "tau_ssry": 100.,
    "threshold_ssry": 0.95,

    "threshold_circuit": 0.7,

    "rwd_weight": 0.0,
    "rwd_sigma": 40.0,
    "col_weight": 0.0,
    "col_sigma": 30.0,

    "action_delay": 15.,
    "edge_route_interval": 80,

    "forced_duration": 100,
    "fine_tuning_min_duration": 15
}


TOT_VALUES = 4


""" functions """


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

    values = np.concatenate((np.around(np.linspace(-0.3, 0., TOT_VALUES//2, endpoint=False), 2),
                             np.around(np.linspace(0, 0.3, TOT_VALUES//2, endpoint=False), 2),
                             np.array([0.3])))
    tot = len(values)
    res = np.zeros((tot, tot))

    for i in tqdm(range(tot)):
        for j in range(tot):
            params = PARAMETERS.copy()
            params["rwd_weight"] = values[i]
            params["col_weight"] = values[j]

            result = safe_run_model(params, "Square.v0")

            # result = sim.run_model(
            #             parameters=params,
            #             global_parameters=global_parameters,
            #             agent_settings=agent_settings,
            #             reward_settings=reward_settings,
            #             game_settings=game_settings,
            #             room_name="Square.v0",
            #             verbose=False,
            #             verbose_min=False)
            res[i, j] = result

    return res




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

    args = parser.parse_args()

    """ setup """

    NUM_CORES = args.cores
    NUM_REPS = args.reps


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

    data = np.zeros((results[0].shape), dtype=np.float64)

    for res in results:
        data += res / float(len(results))

    # data /= float(len(results))
    data = data.tolist()

    localtime = time.localtime()
    name = f"results/remap_"
    name += f"{localtime.tm_mday}{localtime.tm_mon}_{localtime.tm_hour}{localtime.tm_min}"

    # make folder
    if not os.path.exists(f"results/{name}"):
        os.makedirs(name)

    # save data
    dataname = name + "/data.json"
    with open(dataname, "w") as f:
        json.dump(data, f)

    # save values
    values = np.concatenate((np.around(np.linspace(-0.5, 0., TOT_VALUES//2, endpoint=False), 2),
                             np.around(np.linspace(0, 0.5, TOT_VALUES//2, endpoint=False), 2),
                             np.array([0.5]))).tolist()
    with open(name + "/values.json", "w") as f:
        json.dump(values, f)





















