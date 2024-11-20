import numpy as np
import time, os
import random
from deap import base, creator, tools, cma

import tools.evolutions as me
import run_core as rc
from utils_core import setup_logger
from utils_core import edit_logger as edit_logger_utc
from mod_core import edit_logger as edit_logger_mod




""" SETTINGS """

rc.edit_logger(level=-1, is_debugging=False, is_warning=False)
edit_logger_utc(level=-1, is_debugging=False, is_warning=False)
edit_logger_mod(level=-1, is_debugging=False, is_warning=False)

logger = setup_logger(name="EVO", level=0,
                      is_debugging=True, is_warning=True)

sim_settings = {
    "bounds": np.array([0., 1., 0., 1.]),
    "speed": 0.04,
    "init_position": np.array([0.8, 0.2]),
    "rw_fetching": "probabilistic",
    "rw_behaviour": "dynamic",
    "rw_position": np.array([0.5, 0.8]),
    "rw_radius": 0.1,
    "plot_interval": 8,
    "rendering": False,
    "room": "square",
    "max_duration": 500,
    "seed": None
}

agent_settings = {
    "N": 150,
    "Nj": 13**2,
    "sigma": 0.04,
    "max_depth": 20,
}


""" EVALUATION """

# @brief: reward count
def eval_func_I(agent: object):

    """
    reward count
    """

    return agent.model.get_reward_count()


def eval_func_II(agent: object):

    """
    reward count
    """

    return agent.model.get_reward_count()


""" ENVIRONMENT """


class Model(rc.Simulation):

    def __init__(self, threshold: float,
                 rep_threshold: float,
                 w1: float, w2: float, w3: float,
                 w4: float, w5: float,
                 w6: float, w7: float,
                 w8: float, w9: float,
                 w10: float, w11: float,
                 w12: float,
                 sim_settings: dict=sim_settings,
                 agent_settings: dict=agent_settings):

        self.model_params = {
            "threshold": threshold,
            "rep_threshold": rep_threshold,
            "w1": w1,
            "w2": w2,
            "w3": w3,
            "w4": w4,
            "w5": w5,
            "w6": w6,
            "w7": w7,
            "w8": w8,
            "w9": w9,
            "w10": w10,
            "w11": w11,
            "w12": w12
        }

        super().__init__(sim_settings=sim_settings,
                         agent_settings=agent_settings,
                         model_params=self.model_params)

    def reset(self):

        self.reward_obj.reset()

        self.__init__(**self.model_params,
                      sim_settings=sim_settings,
                      agent_settings=agent_settings)


class Env:

    """
    `Nj_set` is not present in this variant
    """

    def __init__(self, n_samples: int=1,
                 eval_func: callable=eval_func_I):

        #
        self._n_samples = n_samples
        self._eval_func = eval_func

        # variables
        self.counter = 0

    def __repr__(self):

        return f"Env()"

    def _train(self, agent: object) -> tuple:

        done = False
        duration = agent.model.max_duration
        agent.model.reset()
        while not done:
            done = agent.model.update()

        return agent

    def run(self, agent: object) -> float:

        """
        Evaluate a agent on the dataset.

        Parameters
        ----------
        agent : object
            A agent object.

        Returns
        -------
        fitness : float
            The fitness value.
        """

        fitness = 0.
        for i in range(self._n_samples):

            # test the agent on the dataset
            agent = self._train(agent=agent)

            # evaluate the agent
            fitness += self._eval_func(agent=agent)

        fitness /= self._n_samples

        return (fitness,)



""" Game setup """


# parameters that are not evolved
# >>> no ftg weight
FIXED_PARAMETERS = {
    "w4": 0.
}


# Define the genome as a dict of parameters
PARAMETERS = {
    'threshold': lambda: round(random.uniform(0.01, 1.0), 2),
    'rep_threshold': lambda: round(random.uniform(0.01, 1.0), 2),
    'w1': lambda: round(random.uniform(-2.0, 2.0), 2),
    'w2': lambda: round(random.uniform(-2.0, 2.0), 2),
    'w3': lambda: round(random.uniform(-2.0, 2.0), 2),
    'w4': lambda: round(random.uniform(-2.0, 2.0), 2),
    'w5': lambda: round(random.uniform(-2.0, 2.0), 2),
    'w6': lambda: round(random.uniform(-2.0, 2.0), 2),
    'w7': lambda: round(random.uniform(-2.0, 2.0), 2),
    'w8': lambda: round(random.uniform(-2.0, 2.0), 2),
    'w9': lambda: round(random.uniform(-2.0, 2.0), 2),
    'w10': lambda: round(random.uniform(-2.0, 2.0), 2),
    'w11': lambda: round(random.uniform(-2.0, 2.0), 2),
    'w12': lambda: round(random.uniform(-2.0, 2.0), 2),
}



if __name__ == "__main__" :

    # ---| Setup |---

    fitness_weights = (1.,)
    NGEN = 20
    NUM_CORES = 64  # out of 8
    NPOP = NUM_CORES
    me.USE_TQDM = False
    VISUALIZE = False

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
    model = Model
    # -> see above for the specification of the data settings
    n_samples = 4
    env = Env(n_samples=n_samples,
              eval_func=eval_func_I)

    # ---| Evolution |---

    # Create the toolbox
    toolbox = me.make_toolbox(PARAMETERS=PARAMETERS.copy(),
                              game=env,
                              model=model,
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
    }

    # ---| Visualisation |---

    visualizer = me.Visualizer(settings=settings,
                               online=VISUALIZE,
                               target=None,
                               k_average=20,
                               fitness_size=len(fitness_weights),
                               ylims=None)

    # ---| save |---
    save = bool(1)

    # filename as best_DDMM_HHMM_r3
    path = "cache/"

    # get number of files in the cache
    n_files = len([f for f in os.listdir(path) \
        if os.path.isfile(os.path.join(path, f))])
    filename = str(n_files+1) + "_best_pcore"

    # extra information
    info = {
        "date": time.strftime("%d/%m/%Y") + " at " + time.strftime("%H:%M"),
        "model": model.__name__,
        "game": env.__repr__(),
        "evolution": settings,
        "evolved": [key for key in PARAMETERS.keys() if key not in FIXED_PARAMETERS.keys()],
        "data": {"sim_settings": sim_settings.copy(),
                 "agent_settings": agent_settings.copy()},
        "other": "evaluating reward count | ftg weight=0.",
    }

    # ---| Run |---
    best_ind = me.main(toolbox=toolbox, settings=settings,
                       info=info, save=save, visualizer=visualizer,
                       filename=filename,
                       verbose=True)

