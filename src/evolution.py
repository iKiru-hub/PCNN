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
logger = setup_logger(name="EVO", level=2, is_debugging=True, is_warning=True)

NUM_SAMPLES = 4
ROOM_LIST = np.random.choice(ROOMS[1:], size=NUM_SAMPLES-2, replace=False).tolist() + \
    [ROOMS[0], ROOMS[0]]


reward_settings = {
    "rw_fetching": "probabilistic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.1 * GAME_SCALE,
    "rw_sigma": 1.5 * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 50,
    "silent_duration": 2_000,
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
    "max_duration": 5_000,
    "room_thickness": 30,
    "seed": None,
    "pause": -1,
    "verbose": False,
    "verbose_min": False
}

global_parameters = {
    "local_scale_fine": 0.015,
    "local_scale_coarse": 0.006,
    "N": 30**2,
    "Nc": 20**2,
    "rec_threshold_fine": 26.,
    "rec_threshold_coarse": 60.,
    "speed": 1.5,
    "min_weight_value": 0.5
}



""" ENVIRONMENT """


def convert_numpy(obj):

    """Helper function to convert NumPy arrays to lists for JSON serialization."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def safe_run_model(agent, room_name):
    """Runs sim.run_model in a subprocess to prevent crashes."""
    try:
        # Convert agent parameters to JSON, handling NumPy arrays
        params = json.dumps(agent.model.get_params(), default=convert_numpy)

        # Convert all other settings while handling NumPy arrays
        global_params_json = json.dumps(global_parameters, default=convert_numpy)
        agent_settings_json = json.dumps(agent_settings, default=convert_numpy)
        reward_settings_json = json.dumps(reward_settings, default=convert_numpy)
        game_settings_json = json.dumps(game_settings, default=convert_numpy)

        # Construct the command to execute sim.run_model in a subprocess
        command = [
            "python3", "-c",
            f"""
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

        print(f"\t{agent.model.name} - {room_name} : {last_line}")

        return float(last_line)

    except subprocess.CalledProcessError as e:
        #logger.error(f"Error: sim.run_model() crashed with: {e.stderr}")
        return 0
    except ValueError as e:
        #logger.warning(f"Warning: sim.run_model() returned unexpected output: {result.stdout.strip()}")
        return 0


class Model:

    def __init__(self, gain_fine, offset_fine, threshold_fine, rep_threshold_fine,
                 gain_coarse, offset_coarse, threshold_coarse, rep_threshold_coarse,
                 min_rep_threshold,
                 lr_da, threshold_da, tau_v_da,
                 lr_bnd, threshold_bnd, tau_v_bnd,
                 tau_ssry, threshold_ssry,
                 threshold_circuit,
                 rwd_weight, rwd_sigma, col_weight, col_sigma,
                 action_delay, edge_route_interval,
                 forced_duration, fine_tuning_min_duration):

        self._params = {
            "gain_fine": gain_fine,
            "offset_fine": offset_fine,
            "threshold_fine": threshold_fine,
            "rep_threshold_fine": rep_threshold_fine,
            "min_rep_threshold": min_rep_threshold,
            "gain_coarse": gain_coarse,
            "offset_coarse": offset_coarse,
            "threshold_coarse": threshold_coarse,
            "rep_threshold_coarse": rep_threshold_coarse,
            "lr_da": lr_da,
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
            "action_delay": action_delay,
            "edge_route_interval": edge_route_interval,
            "forced_duration": forced_duration,
            "fine_tuning_min_duration": fine_tuning_min_duration
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
        for i in range(self._num_samples):
            fitness += safe_run_model(agent, ROOM_LIST[i])


        fitness /= self._num_samples
        print(f"#Agent: {agent.model.name} - Fitness: {fitness}")
        return fitness,



""" Game setup """


# parameters that are not evolved
FIXED_PARAMETERS = {

    "gain_fine": 11.,
    "offset_fine": 1.1,
    "threshold_fine": 0.3,
    # "rep_threshold_fine": 0.88,
    # "min_rep_threshold": 0.95,

    # "gain_coarse": 11.,
    "offset_coarse": 0.9,
    "threshold_coarse": 0.3,
    # "rep_threshold_coarse": 0.89,

    # "lr_da": 0.4,
    # "threshold_da": 0.08,
    "tau_v_da": 1.0,

    # "lr_bnd": 0.4,
    "threshold_bnd": 0.04,
    "tau_v_bnd": 1.0,

    "tau_ssry": 100.,
    "threshold_ssry": 0.995,

    # "threshold_circuit": 0.7,

    # "rwd_weight": 0.0,
    # "rwd_sigma": 40.0,
    # "col_weight": 0.0,
    # "col_sigma": 2.0,

    # "action_delay": 30.,
    "edge_route_interval": 50,

    "forced_duration": 100,
    "fine_tuning_min_duration": 20
}


# Define the genome as a dict of parameters
PARAMETERS = {

    "gain_fine": lambda: round(random.uniform(5., 20.), 1),
    "offset_fine": lambda: round(random.uniform(0.5, 2.0), 1),
    "threshold_fine": lambda: round(random.uniform(0.05, 0.6), 2),
    "rep_threshold_fine": lambda: round(random.uniform(0.7, 0.95), 2),
    "min_rep_threshold": lambda: round(random.uniform(0.89, 0.99), 2),

    "gain_coarse": lambda: round(random.uniform(5., 20.), 1),
    "offset_coarse": lambda: round(random.uniform(0.5, 2.0), 1),
    "threshold_coarse": lambda: round(random.uniform(0.05, 0.6), 2),
    "rep_threshold_coarse": lambda: round(random.uniform(0.7, 0.95), 2),

    "lr_da": lambda: round(random.uniform(0.05, 0.99), 2),
    "threshold_da": lambda: round(random.uniform(0.01, 0.5), 2),
    "tau_v_da": lambda: float(random.randint(1, 10)),

    "lr_bnd": lambda: round(random.uniform(0.05, 0.99), 2),
    "threshold_bnd": lambda: round(random.uniform(0.01, 0.5), 2),
    "tau_v_bnd": lambda: float(random.randint(1, 10)),

    "tau_ssry": lambda: float(random.randint(1, 800)),
    "threshold_ssry": lambda: round(random.uniform(0.8, 0.9999), 3),

    "threshold_circuit": lambda: round(random.uniform(0.2, 0.9), 2),

    "rwd_weight": lambda: round(random.uniform(-1.0, 1.0), 2),
    "rwd_sigma": lambda: round(random.uniform(1.0, 80.0), 1),
    "col_weight": lambda: round(random.uniform(-1.0, 1.0), 2),
    "col_sigma": lambda: round(random.uniform(1.0, 30.0), 1),

    "action_delay": lambda: round(random.uniform(1., 200.), 1),
    "edge_route_interval": lambda: random.randint(1, 200),

    "forced_duration": lambda: random.randint(1, 1500),
    "fine_tuning_min_duration": lambda: random.randint(1, 100),
}



if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--ngen", type=int, default=20)
    parser.add_argument("--npop", type=int, default=1)
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    # ---| Evaluation configs |---

    fitness_weights = (1.,)
    # num_samples = 2

    # ---| Evolution configs |---
    NGEN = args.ngen
    NUM_CORES = args.cores  # out of 8
    NPOP = args.npop
    me.USE_TQDM = False
    VISUALIZE = args.visualize

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

    visualizer = me.Visualizer(settings=settings,
                               online=VISUALIZE,
                               target=None,
                               k_average=20,
                               fitness_size=len(fitness_weights),
                               ylims=None)

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
        "evolved": [key for key in PARAMETERS.keys() if key not in FIXED_PARAMETERS.keys()],
        "data": {"game_settings": game_settings.copy(),
                 "reward_settings": reward_settings.copy(),
                 "agent_settings": agent_settings.copy(),
                 "global_parameters": global_parameters.copy()},
        "other": "first try",
    }

    # ---| Run |---
    best_ind = me.main(toolbox=toolbox, settings=settings,
                       info=info, save=save, visualizer=visualizer,
                       filename=filename,
                       verbose=True)


