import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os, sys, json
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
    "N": 30**2,
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


TOT_VALUES = 6


""" functions """

def run_local_model(args):

    values = np.concatenate((np.around(np.linspace(-0.5, 0., TOT_VALUES//2, endpoint=False), 2),
                             np.around(np.linspace(0, 0.5, TOT_VALUES//2, endpoint=False), 2),
                             np.array([0.5])))
    tot = len(values)
    res = np.zeros((tot, tot))

    for i in tqdm(range(tot)):
        for j in range(tot):
            params = PARAMETERS.copy()
            params["rwd_weight"] = values[i]
            params["col_weight"] = values[j]

            result = sim.run_model(
                        parameters=params,
                        global_parameters=global_parameters,
                        agent_settings=agent_settings,
                        reward_settings=reward_settings,
                        game_settings=game_settings,
                        room_name="Square.v0",
                        verbose=False,
                        verbose_min=False)
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

    args = parser.parse_args()

    """ setup """

    NUM_CORES = args.cores
    NUM_REPS = args.reps


    """ parallel computation """

    chunksize = NUM_REPS // NUM_CORES  # Divide the workload evenly

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

    logger("saving results...")

    data = np.zeros((results[0].shape), dtype=np.float64)

    for res in results:
        data += res / float(len(results))

    # data /= float(len(results))
    data = data.tolist()

    localtime = time.localtime()
    name = f"results/remap_"
    name += f"{localtime.tm_mday}{localtime.tm_mon}_{localtime.tm_hour}{localtime.tm_min}"
    dataname = name + "/data.json"

    # make folder
    if not os.path.exists(f"results/{name}"):
        os.makedirs(name)

    with open(dataname, "w") as f:
        json.dump(data, f)























