import numpy as np
import time, os
import random
from deap import base, creator, tools, cma
import argparse
import json
import subprocess
import string

import tools.evolutions as me
from utils import setup_logger
import simulations as sim
from game.constants import ROOMS, GAME_SCALE


""" SETTINGS """
logger = setup_logger(name="EVO", level=2, is_debugging=False, is_warning=True)

# NUM_SAMPLES = 1
# ROOM_LIST = np.random.choice(ROOMS[1:], size=NUM_SAMPLES-1,
#                              replace=False).tolist() + \
#            ["Square.v0"]

ROOM_LIST = ["Arena.0100", "Arena.0110", "Arena.0111"]
NUM_SAMPLES = len(ROOM_LIST)

MAX_SCORE = 100.

#OPTIONS = [False, False, False, True]
OPTIONS = [True]*4

# ROOM_LIST = ["Square.v0"] * NUM_SAMPLES

EXPL_DURATION = 10_000
TOTAL_DURATION = 30_000
TELEPORT_INTERVAL = 2_500

COLLISION_WEIGHT = -0.5


reward_settings = {
    "rw_fetching": "probabilistic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.08 * GAME_SCALE,
    "rw_sigma": 0.7,# * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 5,
    "silent_duration": EXPL_DURATION,
    "fetching_duration": 2,
    "transparent": False,
    "beta": 35.,
    "alpha": 0.06,# * GAME_SCALE,
    "move_threshold": 150,# * GAME_SCALE,
    "move_threshold": 150,# * GAME_SCALE,
    "rw_position_idx": -1,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "rendering": False,
    "max_duration": TOTAL_DURATION,
    "room_thickness": 20,
    "t_teleport": TELEPORT_INTERVAL,
    "limit_position_len": -1,
    "start_position_idx": 1,
    "seed": None,
    "pause": -1,
    "verbose": False,
    "verbose_min": False
}

global_parameters = {
    "local_scale": 0.015,
    "N": 35**2,
    "use_sprites": False,
    "speed": 1.0,
    "min_weight_value": 0.3
}


""" ENVIRONMENT """


class Model:

    def __init__(self, gain, offset, threshold, rep_threshold,
                 tau_trace, min_rep_threshold, num_neighbors,
                 remap_tag_frequency, rec_threshold,
                 lr_da, lr_pred, threshold_da, tau_v_da,
                 lr_bnd, threshold_bnd, tau_v_bnd,
                 tau_ssry, threshold_ssry, threshold_circuit,
                 rwd_weight, rwd_sigma, col_weight, col_sigma,
                 rwd_field_mod, col_field_mod, action_delay,
                 min_weight_value, edge_route_interval,
                 forced_duration, options=[True]*4):

        self._params = {

            "gain": gain,
            "offset": offset,
            "threshold": threshold,
            "rep_threshold": rep_threshold,
            "rec_threshold": rec_threshold,
            "tau_trace": tau_trace,

            "min_rep_threshold": min_rep_threshold,
            "num_neighbors": num_neighbors,
            "remap_tag_frequency": remap_tag_frequency,

            "lr_da": lr_da,
            "lr_pred": lr_pred,
            "threshold_da": threshold_da,
            "tau_v_da": tau_v_da,

            "lr_bnd": lr_bnd,
            "threshold_bnd": threshold_bnd,
            "tau_v_bnd": tau_v_bnd,

            "tau_ssry": tau_ssry,
            "threshold_ssry": threshold_ssry,
            "threshold_circuit": threshold_circuit,

            "rwd_weight": rwd_weight,
            "rwd_sigma": rwd_sigma,
            "col_weight": col_weight,
            "col_sigma": col_sigma,

            "rwd_field_mod": rwd_field_mod,
            "col_field_mod": col_field_mod,

            "action_delay": action_delay,
            "edge_route_interval": edge_route_interval,

            "forced_duration": forced_duration,
            "min_weight_value": min_weight_value,

            "options": options
        }

        self.name = "".join(np.random.choice(list(string.ascii_uppercase), 5))

    def __repr__(self):
        return "ModelShell"

    def get_params(self):
        return self._params

    def output(self): pass
    def step(self): pass
    def reset(self): pass


class Env:

    def __init__(self, num_samples: int, npop: int):
        self._num_samples = num_samples
        self._npop = npop
        self._counter = 0

    def __repr__(self):
        return f"Env({self._num_samples})"

    def run(self, agent: object) -> tuple:

        fitness = 0
        fitness2 = 0
        nb_zeros = 0
        for i in range(self._num_samples):
            # score = safe_run_model(agent, ROOM_LIST[i])

            score, info = sim.run_model(
                parameters=agent.model.get_params(),
                global_parameters=global_parameters,
                reward_settings=reward_settings,
                game_settings=game_settings,
                room_name=ROOM_LIST[i],
                verbose=False,
                verbose_min=False)

            # cap
            score = min(MAX_SCORE, score)

            # if score == 0:
            #     nb_zeros += 1

            fitness += score
            fitness2 += info['collisions_from_rw']

        # fitness /= max(self._num_samples-nb_zeros, 1)
        # fitness2 /= max(self._num_samples-nb_zeros, 1)
        fitness /= self._num_samples
        fitness2 /= self._num_samples
        return fitness, fitness2 * COLLISION_WEIGHT


""" Game setup """


# parameters that are not evolved
FIXED_PARAMETERS = {

     # 'gain': 33.0,
     # 'offset': 1.0,
     'threshold': 0.4,
     # 'rep_threshold': 0.85,
     # 'rec_threshold': 63,
     # 'tau_trace': 20,

     'remap_tag_frequency': 1,
     'min_rep_threshold': 0.99,

     'lr_da': 0.9,
     'lr_pred': 0.05,
     'threshold_da': 0.05,
     'tau_v_da': 1.0,

     'lr_bnd': 0.9,
     'threshold_bnd': 0.1,
     'tau_v_bnd': 2.0,

     'tau_ssry': 437.0,
     'threshold_ssry': 1.986, # <-----------------
     'threshold_circuit': 0.9,

     # 'rwd_weight': 0.0,
     # 'rwd_sigma': 0.,
     # 'col_weight': 0.0,
     # 'col_sigma': 0.,
     # 'rwd_field_mod': 0.0,
     # 'col_field_mod': 0.0,

     'action_delay': 120.0,
     'edge_route_interval': 50,
     'forced_duration': 19,
     'min_weight_value': 0.1,
     'options': OPTIONS
}


# Define the genome as a dict of parameters
PARAMETERS = {

    "gain": lambda: round(random.uniform(2., 200.), 1),
    "offset": lambda: np.clip(random.uniform(0.9, 1.1), 0.9, 1.02),
    "threshold": lambda: round(random.uniform(0.05, 0.5), 2),
    "rep_threshold": lambda: np.clip(random.uniform(0.8, 1.1), 0.8, 0.999),
    "rec_threshold": lambda: round(random.uniform(20., 120.)),
    "tau_trace": lambda: round(random.uniform(1., 300.)),

    "remap_tag_frequency": lambda: random.choice([1, 2, 3, 4]),
    "num_neighbors": lambda: int(random.randint(3, 20)),
    "min_rep_threshold": lambda: round(random.uniform(0.2, 0.95), 2),

    "lr_da": lambda: round(random.uniform(0.4, 0.99), 2),
    "lr_pred": lambda: round(random.uniform(0.01, 0.4), 2),
    "threshold_da": lambda: round(random.uniform(0.01, 0.5), 2),
    "tau_v_da": lambda: float(random.randint(1, 5)),

    "lr_bnd": lambda: round(random.uniform(0.3, 0.99), 2),
    "threshold_bnd": lambda: round(random.uniform(0.01, 0.5), 2),
    "tau_v_bnd": lambda: float(random.randint(1, 5)),

    "tau_ssry": lambda: float(random.randint(100, 800)),
    "threshold_ssry": lambda: round(random.uniform(0.8, 1.2), 3),
    "threshold_circuit": lambda: round(random.uniform(0.2, 1.3), 2),

    "rwd_weight": lambda: round(random.uniform(-15.0, 5.0), 2),
    "rwd_sigma": lambda: round(random.uniform(1.0, 130.0), 1),
    "col_weight": lambda: round(random.uniform(-15.0, 5.0), 2),
    "col_sigma": lambda: round(random.uniform(1.0, 60.0), 1),

    "rwd_field_mod": lambda: round(random.uniform(-5.0, 5.0), 1),
    "col_field_mod": lambda: round(random.uniform(-5.0, 5.0), 1),

    "action_delay": lambda: round(random.uniform(1., 300.), 1),
    "edge_route_interval": lambda: random.randint(1, 10_000),

    "forced_duration": lambda: random.randint(1, 50),
    "min_weight_value": lambda: random.uniform(0., 0.5),
}



if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--ngen", type=int, default=100)
    parser.add_argument("--npop", type=int, default=1)
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    # ---| Evaluation configs |---

    fitness_weights = (1., 1.)
    # num_samples = 2

    # ---| Evolution configs |---
    NGEN = args.ngen
    NUM_CORES = args.cores  # out of 8
    NPOP = args.npop
    me.USE_TQDM = False
    VISUALIZE = args.visualize

    percent_to_save = 0.9

    # Ignore runtime warnings
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # ---| CMA-ES |---

    # parameters
    N_param = len(PARAMETERS) - len(FIXED_PARAMETERS)  # number of parameters
    MEAN = np.random.rand(N_param)  # Initial mean, could be randomized
    SIGMA = 0.8  # Initial standard deviation
    lambda_ = 4 + np.floor(3 * np.log(N_param))

    # strategy
    strategy = cma.Strategy(centroid=MEAN,
                            sigma=SIGMA,
                            lambda_=lambda_)

    # ---| Game |---
    # -> see above for the specification of the data settings
    env = Env(num_samples=NUM_SAMPLES, npop=NPOP)

    logger(f"Env: {env.__repr__()}")

    # ---| Evolution |---

    # Create the toolbox
    toolbox = me.make_toolbox(PARAMETERS=PARAMETERS.copy(),
                              game=env,
                              model=Model,
                              strategy=strategy,
                              FIXED_PARAMETERS=FIXED_PARAMETERS.copy(),
                              fitness_weights=fitness_weights)

    # ---| Run |---

    settings = {
        "NPOP": NPOP,
        "NGEN": NGEN,
        "CXPB": 0.6,
        "MUTPB": 0.7,
        "NLOG": 1,
        "TARGET": (1.,),
        "TARGET_ERROR": 0.,
        "NUM_CORES": NUM_CORES,
        "USE_TQDM": False,
    }

    # ---| Visualisation |---

    if args.visualize:
        visualizer = me.Visualizer(settings=settings,
                                   online=VISUALIZE,
                                   target=None,
                                   k_average=20,
                                   fitness_size=len(fitness_weights),
                                   ylims=None)
    else:
        visualizer = None

    # ---| save |---
    save = args.save

    # filename as best_DDMM_HHMM_r3
    path = "cache/"

    # get number of files in the cache
    n_files = len([f for f in os.listdir(path) \
        if os.path.isfile(os.path.join(path, f))])
    filename = str(n_files+1) + "_best_agent"

    # extra information
    info = {
        "date": time.strftime("%d/%m/%Y") + " at " + time.strftime("%H:%M"),
        "model": "ModelShell",
        "game": env.__repr__(),
        "evolution": settings,
        "evolved": [key for key in PARAMETERS.keys() if key not in \
            FIXED_PARAMETERS.keys()],
        "data": {"game_settings": game_settings.copy(),
                 "reward_settings": reward_settings.copy(),
                 "global_parameters": global_parameters.copy()},
        "other": f"single space, options={OPTIONS}",
    }
    logger(f"Note: {info['other']}")

    # ---| Run |---
    best_ind = me.main(toolbox=toolbox, settings=settings,
                       info=info, save=save, visualizer=visualizer,
                       filename=filename,
                       perc_to_save=percent_to_save,
                       verbose=True)


