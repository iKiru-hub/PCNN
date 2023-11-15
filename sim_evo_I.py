import random
from deap import base, creator, tools, cma
import mod_models as mm
import mod_evolution as me
import mod_stimulation as ms
from tools.utils import logger
import inputools.Trajectory as it
import numpy as np
import time 


# ----------------------------------------------------------
# ---| Fitness |---
# -----------------

def FitnessFunction(W: np.ndarray, activity: np.ndarray, Nj: int=9,
                    wmax: float=4.5, ids: str="", **kwargs) -> np.ndarray:

    """
    The fitness function. It is composed of three terms:
    1. the maximum row entry should be close to wmax 
    2. each row should have only one entry close to wmax
    3. the high-valued entries should be close to Nj

    Parameters
    ----------
    W : np.ndarray
        The weight matrix.
    Nj : int
        Number of neurons.
    wmax : float
        The maximum weight.
    ids : str
        The agent id.

    Returns
    -------
    fitness : float
        The fitness value.
    """

    # the maximum row entry should be close to wmax
    value_1 = - np.sum((wmax - W.max(axis=1)) ** 2)
    # value_1 = - abs(Nj * wmax - W.max(axis=1).sum())
    # value_1 = - abs(Nj * wmax - np.diff((np.sort(W, axis=0)[-2:]), axis=0).sum())

    # value_1 = np.diff((np.sort(W, axis=1)[-2:]), axis=0).sum()

    if np.isnan(value_1):
        logger.error(f"[{ids}] {value_1=}")
        logger.debug(f"[{W=}")
        raise ValueError("value_1 is NaN")

    # each row should have at most one entry close to wmax
    # in practice: the sum of the two highest column entries
    # should be close to wmax
    value_2 = - ((wmax - \
        np.sort(W, axis=0)[-2:].sum(axis=0))**2).sum()

    # each row should have at most one entry close to wmax
    # implementation: the difference between the two highest
    # column entries should be as high as possible
    # value_2 = np.diff((np.sort(W, axis=0)[-2:]), axis=0).sum()


    #
    value_3 = (np.sort(activity, axis=0)[-5:].sum(axis=0) - activity.mean(axis=0)).sum()

    return value_1, value_2, value_3



# ----------------------------------------------------------
# ---| Game |---
# --------------

class Track2D:

    def __init__(self, dataset: list, Nj: int, 
                 wmax: float=4.5, fitness_size: int=2):

        """
        A 2D tracking game.

        Parameters
        ----------
        dataset : list
            A list of spiking trajectories.
        Nj : int
            Number of neurons.
        wmax : float
            The maximum weight.
        fitness_size : int
            The number of fitness values.
        """

        self.dataset = dataset
        self.length = len(dataset)
        self.fitness_size = fitness_size
        self.Nj = Nj
        self.wmax = wmax

    def _train(self, agent: object, trajectory: np.ndarray) -> tuple:

        """
        Train a agent on the dataset.

        Parameters
        ----------
        agent : object
            A agent object.
        trajectory : np.ndarray
            A trajectory.

        Returns
        -------
        agent : object
            The trained agent.
        """

        agent.model.reset()

        activity = np.zeros((len(trajectory), agent.model.N))
        # train on a single trajectory
        for t in range(len(trajectory)):
            agent.step(trajectory[t].reshape(-1, 1))
            activity[t] = agent.model.r.copy().reshape(-1)

        return agent, activity

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

        fitness = np.zeros(self.fitness_size)

        # test the agent on the dataset
        for trajectory in self.dataset:

            # train the agent
            agent, activity = self._train(agent=agent, 
                                trajectory=trajectory)

            # evaluate the agent
            fitness += np.array(FitnessFunction(
                W=agent.model.Wff.copy(), 
                activity=activity.copy(),
                Nj=self.Nj,
                wmax=agent.model.wff_max, 
                ids=agent.id))


        fitness = tuple(fitness / self.length)

        assert isinstance(fitness, tuple), "fitness must be a tuple"
        return fitness



class Agent:

    def __init__(self, genome: dict):

        """
        An agent.

        Parameters
        ----------
        genome : dict
            The genome.
        """

        # unpack the genome and create the model
        self.genome = genome.copy() 

        # exec callable parameters 
        # NB: the callable parameters are not evolved
        # NB: the callable parameters are not passed to the model
        for k, v in self.genome.items():
            if callable(v):
                genome[k] = v()

        # self.model = mm.NetworkSimple(**genome.copy())
        self.model = mm.RateNetworkSimple(**genome.copy())

        self.id = ''.join(np.random.choice(list(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ), 5))

    def __getitem__(self, key):
        return self.genome[key]

    def __setitem__(self, key, value):
        self.genome[key] = value

    def __str__(self):
        return str(self.genome.copy())

    def __repr__(self):
        return f"NetworkSimple(id={self.id}, N={self.genome['N']})"

    def step(self, x: np.ndarray) -> np.ndarray:

        """
        Step the agent.

        Parameters
        ----------
        x : np.ndarray
            The input.

        Returns
        -------
        y : np.ndarray
            The output.
        """

        self.model.step(x)

    def reset(self):
        self.model.reset()



# ----------------------------------------------------------
# ---| Genome Setup |---
# ----------------------

# parameters that are not evolved
FIXED_PARAMETERS = {
    'N': 16,
    'Nj': 9,
    'dim': 2,
    # 'eps': 3,
    # 'tau_u': 100,
    # 'lr_max': 0.03,
    'lr_min': 1e-6,
    # 'lr_tau': 200,
    # 'wff_const': 5.5,
    # 'wff_const_beta': 50,
    # 'wff_const_alpha': 0.8,
    # 'wff_max': 4.5,
    # 'wff_min': 0.075,
    'wff_tau_max': 2000,
    # 'wff_tau_min': 50,
    # 'wff_tau_tau': 204,
    # 'wff_beta': 0.145,
    # 'wr_const': 5.5,
    # 'A': .0,
    # 'B': 1.0,
    # 'sigma_exc': 1.0,
    # 'sigma_inh': 1.0,
    # 'tau_ff': 10,
    # 'tau_rec': 10,
    # 'syn_ff_tau': 10,
    # 'syn_ff_thr': 0.326,
    # 'rate_func_beta': 0.3,
    # 'rate_func_alpha': 60,
}

# Define the genome as a dict of parameters 
PARAMETERS_2 = {
    'tau_u': lambda: random.randint(1, 300),
    'lr_max': lambda: round(random.uniform(5e-3, 0.1), 2),
    'lr_min': lambda: round(random.uniform(1e-3, 1e-6), 8),
    'lr_tau': lambda: random.randint(50, 300),
    'wff_const': lambda: round(random.uniform(4.0, 10.0), 2),
    'wff_const_beta': lambda: random.choice(range(5, 100, 5)),
    'wff_const_alpha': lambda: round(random.uniform(0.5, 1.0), 2),
    'wff_max': lambda: round(random.uniform(2.0, 10.0), 3),
    'wff_min': lambda: round(random.uniform(0., 2.0), 3),
    'wff_tau_max': lambda: random.randint(1000, 8000),
    'wff_tau_min': lambda: random.randint(10, 500),
    'wff_tau_tau': lambda: random.randint(50, 400),
    'wff_beta': lambda: round(random.uniform(0.1, 1.0), 3),
    'wr_const': lambda: round(random.uniform(0.1, 10.0), 3),
    'dim': lambda: random.choice((1, 2)),
    'A': lambda: round(random.uniform(0.1, 4.0), 3),
    'B': lambda: round(random.uniform(0.1, 3.0), 3),
    'sigma_exc': lambda: round(random.uniform(0., 8.0), 3),
    'sigma_inh': lambda: round(random.uniform(0., 8.0), 3),
    'tau_ff': lambda: random.randint(1, 100),
    'tau_rec': lambda: random.randint(1, 100),
    'syn_ff_tau': lambda: random.randint(1, 100),
    'syn_ff_thr': lambda: round(random.uniform(0., 1.0), 3),
    'rate_func_beta': lambda: round(random.uniform(0.1, 1.0), 3),
    'rate_func_alpha': lambda: random.randint(50, 80),
}



# Define the genome as a dict of parameters 
PARAMETERS = {
    'tau_u': lambda: random.randint(1, 300),
    'eps': lambda: round(random.uniform(0.1, 20.0), 2),
    'lr_max': lambda: round(random.uniform(5e-3, 0.1), 2),
    'lr_min': lambda: round(random.uniform(1e-3, 1e-6), 8),
    'lr_tau': lambda: random.randint(50, 1000),
    'wff_const': lambda: round(random.uniform(4.0, 10.0), 2),
    'wff_const_beta': lambda: random.choice(range(5, 100, 5)),
    'wff_const_alpha': lambda: round(random.uniform(0.5, 1.0), 2),
    'wff_max': lambda: round(random.uniform(2.0, 10.0), 3),
    'wff_min': lambda: round(random.uniform(0., 2.0), 3),
    'wff_tau_max': lambda: random.randint(1000, 8000),
    'wff_tau_min': lambda: random.randint(10, 500),
    'wff_tau_tau': lambda: random.randint(50, 400),
    'wff_beta': lambda: round(random.uniform(0.1, 1.0), 2),
    'wr_const': lambda: round(random.uniform(0.1, 10.0), 2),
    'dim': lambda: random.choice((1, 2)),
    'A': lambda: round(random.uniform(0.01, 8.0), 2),
    'B': lambda: round(random.uniform(0.01, 8.0), 25),
    'sigma_exc': lambda: round(random.uniform(0., 8.0), 2),
    'sigma_inh': lambda: round(random.uniform(0., 8.0), 2),
    'tau_ff': lambda: random.randint(1, 100),
    'tau_rec': lambda: random.randint(1, 100),
    'syn_ff_tau': lambda: random.randint(1, 100),
    'syn_ff_thr': lambda: round(random.uniform(0., 1.0), 3),
    'rate_func_beta': lambda: round(random.uniform(0.1, 5.0), 2),
    'rate_func_alpha': lambda: round(random.uniform(0, 10), 2),
    'is_lr_tau_decay': lambda: random.choice((True, False)),
    'is_g_decay': lambda: random.choice((True, False)),
    'is_syn': lambda: random.choice((True, False)),
    'is_eps_scaled': lambda: random.choice((True, False)),
    'rule': lambda: random.choice(('oja', 'hebb')),
}


if __name__ == "__main__" :

    # ---| Setup |---

    fitness_weights = (1., 1., 1.)
    wff_max = 1e2

    # ---| CMA-ES |---

    # parameters
    N = len(PARAMETERS) - len(FIXED_PARAMETERS)  # number of parameters   
    MEAN = np.random.rand(N)  # Initial mean, could be randomized
    SIGMA = 0.5  # Initial standard deviation
    lambda_ = 4 + np.floor(3 * np.log(N))

    # strategy
    strategy = cma.Strategy(centroid=MEAN,
                            sigma=SIGMA, 
                            lambda_=lambda_)


    # ---| Dataset |---

    # Create an animal
    animal = it.AnimalTrajectory(dt=1, 
                                 prob_turn=0.01, 
                                 prob_speed=0.1,
                                 prob_rest=0.01, 
                                 day_cycle=True)

    # input layer
    layer = ms.InputLayer(N=FIXED_PARAMETERS['Nj'],
                          sp=1.,
                          kind='place', 
                          bounds=(0.05, 0.95, 0.05, 0.95),
                          sigma=0.04, max_rate=10, min_rate=0)

    dataset = it.make_dataset(n_samples=2,
                              animal=animal,
                              layer=layer,
                              mode='rate',
                              duration=2_000,
                              whole=True,
                              timestep=1, 
                              dx=0.05)

    track = Track2D(dataset=dataset, 
                    Nj=FIXED_PARAMETERS['Nj'], 

                    fitness_size=len(fitness_weights),
                    wmax=wff_max)

    # ---| Evolution |---

    # Create the toolbox
    toolbox = me.make_toolbox(PARAMETERS=PARAMETERS.copy(),
                              game=track,
                              agent_class=Agent,
                              strategy=strategy,
                              FIXED_PARAMETERS=FIXED_PARAMETERS.copy(),
                              fitness_weights=fitness_weights)

    # ---| Run |---

    settings = {
        "NPOP": 250,
        "NGEN": 800,
        "CXPB": 0.5,
        "MUTPB": 0.2,
        "NLOG": 5,
        # "TARGET": (FIXED_PARAMETERS['Nj']*wff_max, 0.),
        "TARGET": (0, 100, 100),
        "TARGET_ERROR": 1e-4,
    }

    best_ind = me.main(toolbox=toolbox, settings=settings)

    # save 
    filename = "best_ind_" + time.strftime("%H%M") + "_r"
    me.save_best_individual(best_ind=best_ind, 
                            filename=filename,
                            path=None)
