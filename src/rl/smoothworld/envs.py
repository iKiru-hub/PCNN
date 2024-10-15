#atinabox stuff
import ratinabox
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, FieldOfViewBVCs, GridCells
# from ratinabox.contribs.NeuralNetworkNeurons import NeuralNetworkNeurons #for the Actor and Critic
from ratinabox.contribs.TaskEnvironment import (SpatialGoalEnvironment,
    SpatialGoal, Reward)

# rl stuff
from gym import spaces
import gymnasium as gym
from stable_baselines3 import A2C, PPO

#misc
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from gymnasium.spaces import Box
from tqdm import tqdm
import time
from pprint import pprint
import argparse

# local imports
import sys, os
if os.path.exists(os.path.expanduser('~/Research/lab/PCNN/src')):
    sys.path.append(os.path.expanduser('~/Research/lab/PCNN/src'))
elif os.path.exists(os.path.expanduser('~/lab/PCNN/src')):
    sys.path.append(os.path.expanduser('~/lab/PCNN/src'))
else:
    raise ModuleNotFoundError('PCNN not found')

from minimal_model import PCNNrx
import utils
logger = utils.logger
# from utils import Agent
try:
    import inputools.Trajectory as it
except ModuleNotFoundError:
    warnings.warn('`inputools.Trajectory` not found, some functions may not work')



#TASK CONSTANTS
DT = 0.1 # Time step
# GOAL_POS = np.array([0.85, 0.15]) # Goal position
WALL = None
# GOAL_RADIUS = 0.2
REWARD = 1 # Reward
# REWARD_DURATION = int(5/DT) # Reward duration


IS_PCNN = True


""" environment design """


def simple_flat(env: object, GOAL_POS: tuple=None,
                GOAL_RADIUS: float=None,
                REWARD_DURATION: int=50):

    env.add_wall([[0.0, 0.5], [0.5, 0.5]])

    env.exploration_strength = 1

    # Make the reward which is given when a spatial goal is satisfied.
    # attached this goal to the environment
    GOAL_POS = np.array([0.85, 0.15])
    GOAL_RADIUS = 0.15
    reward = Reward(REWARD,
                    decay="none",
                    expire_clock=REWARD_DURATION,
                    dt=DT,)
    goals = [SpatialGoal(env, pos=GOAL_POS,
                         goal_radius=GOAL_RADIUS,
                         reward=reward)]
    env.goal_cache.reset_goals = goals 

    return env, GOAL_POS, GOAL_RADIUS


def flat_one(env: object, GOAL_POS: tuple=None,
             GOAL_RADIUS: float=None,
             REWARD_DURATION: int=50):

    env.exploration_strength = 1

    # Make the reward which is given when a spatial goal is satisfied.
    # Attached this goal to the environment
    GOAL_POS = np.array([0.2, 0.2]) if GOAL_POS is None else GOAL_POS
    GOAL_RADIUS = 0.15 if GOAL_RADIUS is None else GOAL_RADIUS
    reward = Reward(REWARD,
                    decay="none",
                    expire_clock=REWARD_DURATION,
                    dt=DT,)
    goals = [SpatialGoal(env, pos=GOAL_POS,
                         goal_radius=GOAL_RADIUS,
                         reward=reward)]
    env.goal_cache.reset_goals = goals 

    return env, GOAL_POS, GOAL_RADIUS


def flat_two(env: object, GOAL_POS: tuple=None,
             GOAL_RADIUS: float=None,
             REWARD_DURATION: int=50):

    # env.add_wall([[0.5, 0.2], [0.5, 0.85]])
    env.add_wall([[0.5, 0.5], [1., 0.5]])

    env.exploration_strength = 1

    # Make the reward which is given when a spatial goal is satisfied.
    # Attached this goal to the environment
    GOAL_POS = np.array([0.2, 0.2]) if GOAL_POS is None else GOAL_POS
    GOAL_RADIUS = 0.15 if GOAL_RADIUS is None else GOAL_RADIUS
    reward = Reward(REWARD,
                    decay="none",
                    expire_clock=REWARD_DURATION,
                    dt=DT,)
    goals = [SpatialGoal(env, pos=GOAL_POS,
                         goal_radius=GOAL_RADIUS,
                         reward=reward)]
    env.goal_cache.reset_goals = goals 

    return env, GOAL_POS, GOAL_RADIUS


def flat_three(env: object, GOAL_POS: tuple=None,
               GOAL_RADIUS: float=None,
               REWARD_DURATION: int=50):

    door_1 = 0.1

    # left rooms
    env.add_wall([[0.0, 0.3], [0.5, 0.3]])
    env.add_wall([[0.0, 0.6], [0.5, 0.6]])

    env.add_wall([[0.5, 0.0], [0.5, 0.165-door_1/2]])
    env.add_wall([[0.5, 0.165+door_1/2], [0.5, 0.495-door_1/2]])
    env.add_wall([[0.5, 0.495+door_1/2], [0.5, 0.835-door_1/2]])
    env.add_wall([[0.5, 0.835+door_1/2], [0.5, 1.]])

    env.add_wall([[0.65, 0.5], [0.95, 0.5]])

    env.exploration_strength = 1

    # Make the reward which is given when a spatial goal is satisfied.
    # Attached this goal to the environment
    # global GOAL_POS
    # global GOAL_RADIUS
    GOAL_POS = np.array([0.2, 0.2]) if GOAL_POS is None else GOAL_POS
    GOAL_RADIUS = 0.15 if GOAL_RADIUS is None else GOAL_RADIUS
    reward = Reward(REWARD,
                    decay="none",
                    expire_clock=REWARD_DURATION,
                    dt=DT,)
    goals = [SpatialGoal(env, pos=GOAL_POS,
                         goal_radius=GOAL_RADIUS,
                         reward=reward)]
    env.goal_cache.reset_goals = goals 

    return env, GOAL_POS, GOAL_RADIUS


""" environment class """


class SuperEnv(gym.Env):

    """
    Custom environment that follows gym interface.
    """

    def __init__(self, env: object, cells: list, GOAL_POS: np.ndarray=None,
                 GOAL_RADIUS: float=None,
                 max_experiences: int=7, **kwargs):

        """
        Parameters
        ----------
        env : object
            The environment object.
        cells : list
           list of cells (e.g. Place cells) that are used
           to augment the observation space.
        """

        super().__init__()
        self._env = env
        self._cells = cells
        self._cells_names = [cell.__repr__().split(".")[-1].split(" ")[0] \
            for cell in cells]
        self.t = env.t
        self.episodes = env.episodes
        self.GOAL_POS = GOAL_POS
        self.GOAL_RADIUS = GOAL_RADIUS

        # Initialize observation by combining agent's position
        # and zeros for each cell's attributes
        # self.observation = np.concatenate(
        #     [env.Agents[0].pos] + [np.zeros(c.n) for c in self._cells]
        # )

        # ROUNDS SETTINGS
        self._rounds = kwargs.get("rounds", 1)

        # EXPERIENCES SETTINGS
        self._nb_experiences = kwargs.get("nb_experiences", 4)
        self._experience_duration = kwargs.get("experience_duration", 400)
        self._max_experiences = kwargs.get("max_experiences", 7)
        self._experience_t = 0
        self._experience_counter = 0  # all experiences
        self._reward_counter = 0  # positive rewards

        self._max_speed = kwargs.get("max_speed", 0.01)

        # initial position
        self.init_pos = kwargs.get("init_pos", None)
        self.init_pos_radius = kwargs.get("init_pos_radius", None)
        self._is_init_pos = self.init_pos is not None

        # wall hits
        self._max_wall_hits = kwargs.get("max_wall_hits", 20)
        self._wall_hits = 0

        # --- OBSERVATION SPACE ---
        # >> state cells + previous reward
        self.observation = np.concatenate(
            [np.zeros(c.n) for c in self._cells]
        )

        self._obs_size = len(self.observation)

        # Define the observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-100., high=100.,
            shape=(len(self.observation),),
            dtype=np.float64
        )

        # --- ACTION SPACE ---
        # if not IS_PCNN:  # <<< this is the default
        # if True:
        #     self.action_space = gym.spaces.Box(
        #         low=-100., high=100., shape=(2,), dtype=np.float64
        #     )
        # else:  # dx, dy, tau, gate
        #     self.action_space = gym.spaces.Box(
        #         low=-10., high=10., shape=(4,), dtype=np.float64
        #     )

        # >> the agent chooses:
        # x_{1}: a direction [0, 1] -> converted in a [0, 2π]
        # x_{2}: a speed [0, 1] -> converted into a speed as v_{0}+v_{max}*x_{2}

        self.action_space = gym.spaces.Box(
            low=0., high=1., shape=(3,), dtype=np.float64
        )
        # self.action_space = gym.spaces.Box(
        #     low=0., high=1., shape=(2,), dtype=np.float64
        # )
        # self.action_space = gym.spaces.Box(
        #     low=-10., high=10., shape=(2,), dtype=np.float64
        # )

        #
        self.whole_track = it.make_whole_walk(dx=0.015,
                                              bounds=self._env.extent)

        # update the inner environment
        self._env.observation_spaces["agent_0"] = self.observation_space
        self._env.action_spaces["agent_0"] = self.action_space

    def __repr__(self):

        return f"SuperEnv(#experiences={self._nb_experiences}, {IS_PCNN=}, " + \
            f"[{', '.join(self._cells_names)}])"

    def _parse_action(self, action: np.ndarray) -> np.ndarray:

        """
        map the model's output to an action

        Parameters
        ----------
        action : np.ndarray
            The model's output.

        Returns
        -------
        np.ndarray
        """

        # --- MOVEMENT ACTION ---
        # >> action defined as a next position calculated
        #    from the direction and speed
        # direction, speed, var1 = action
        dx, dy, var1 = action

        # map the direction to the angle [0, 1] -> [0, 2π]
        # direction = direction * 2 * np.pi
        dx = dx * 2 - 1
        dy = dy * 2 - 1

        # get the current position
        x, y = self._env.agents_dict['agent_0'].pos.tolist()

        # calculate the new position as action
        return np.array([dx*self._max_speed, dy*self._max_speed])

    def step(self, action: np.ndarray) -> tuple:

        """
        Step the environment by taking an action.

        Parameters
        ----------
        action : np.ndarray
            The action to take in the environment.

        Returns
        -------
        obs : np.ndarray
            The observation after taking the action.
        reward : float
            The reward after taking the action.
        terminate : bool
            Whether the episode has terminated.
        truncated : bool
            Whether the episode was truncated.
        info : dict
            Additional information after taking the action.
        """

        # parse the action
        action = self._parse_action(action=action)

        # Step the environment and get necessary returns
        obs, reward, terminate, truncated, info = self._env.step1(
                            action=action,
                            drift_to_random_strength_ratio=1)
        self.t = self._env.t
        self.episodes = self._env.episodes

        obs = np.array([]).astype(np.float64)

        # Update cells and augment observation
        for c in self._cells:
            c.update()

            if c.name == "PCNNrx":
                obs = np.concatenate((obs,
                                      c.firingrate))
            else:
                obs = np.concatenate((obs,
                                      c.firingrate))

        # --- EXPERIENCE TERMINATION ---

        # EXIT CHECK : the experience duration is reached | --- now disabled
        if terminate and False:
            self._experience_counter += 1

            # max experiences reached
            if self._experience_counter >= self._max_experiences:
                truncated = True 
                terminate = True

            # not yet
            else:
                self.reset(episode_meta_info="new experience",
                           kind="soft")
                terminate = False
                truncated = False

        # --- WALL HITS ---
        # >> check if the agent hits the wall
        if self._env.agents_dict['agent_0'].is_wall_hit:
            self._wall_hits += 1
            if self._wall_hits >= self._max_wall_hits:
                terminate = True
                truncated = True

        # --- REWARD ---

        # if reward:
        #     terminate = True

        # Conflate terminate and truncated into a single 'done' signal
        return obs, reward, terminate, truncated, info

    def reset(self, episode_meta_info: str=None,
              seed: int=None,
              kind: str="soft",
              **kwargs) -> tuple:

        """
        Reset the environment.

        Parameters
        ----------
        episode_meta_info : str
            Additional information to reset the episode.
        seed : int
            Random seed for the environment.
        kind : str
            The kind of reset to perform.
            Default is "soft".

        Returns
        -------
        obs : np.ndarray
            The initial observation after resetting the environment.
        info : dict
            Additional information after resetting the environment.
        """

        if seed is not None:
            super().reset(seed=seed)

        # a totally new episode
        if kind == "soft":

            # reset state cells
            for c in self._cells:
                if c.__class__.__name__ == "PCNNrx":
                    c.reset(kind="soft")
                    continue

                if hasattr(c, "reset"):
                    c.reset()

            self._experience_counter += 1
            self._wall_hits = 0
        else:
            for c in self._cells:
                if c.__class__.__name__ == "PCNNrx":
                    c.reset(kind="hard")
                    continue
            self._experience_counter = 0

        self._env.t = 0

        # Reset the environment state and return the initial observation
        initial_obs = self._env.reset(
            episode_meta_info=episode_meta_info)[0]['agent_0']

        self.observation = initial_obs
        self._wall_hits = 0
        self._experience_t = 0

        # new positions
        if self._is_init_pos:
            new_pos = np.array([
                np.random.uniform(self.init_pos[0],
                                  self.init_pos[0]+self.init_pos_radius),
                np.random.uniform(self.init_pos[1],
                                    self.init_pos[1]+self.init_pos_radius)
            ])
            self._env.agents_dict['agent_0'].pos = new_pos

        # initial observation
        self.observation = np.concatenate([np.zeros(c.n) for c in self._cells])

        return self.observation.astype(np.float64), {}


def generate_navigation_task_env(GOAL_POS: tuple,
                                 IS_PCNN_flag: bool=None,
                                 flat: str="two",
                                 GOAL_RADIUS: float=None,
                                 REWARD_DURATION: int=50,
                                 nb_experiences: int=4,
                                 experience_duration: int=400,
                                 max_experiences: int=7,
                                 init_pos: tuple=None,
                                 init_pos_radius: float=None,
                                 max_wall_hits: int=20,
                                 max_speed: float=0.01,
                                 pcnn_params: dict=None,
                                 cell_types: tuple=None,
                                 return_info: bool=False):

    # if IS_PCNN_flag is not None:
    #     global IS_PCNN
    #     IS_PCNN = IS_PCNN_flag

    # Make the environment and add a wall 
    env = SpatialGoalEnvironment(
        dt=DT,
        teleport_on_reset=init_pos is None,
        episode_terminate_delay=REWARD_DURATION,
        params={"boundary_conditions": "solid"})

    # env.add_wall([[0.0, 0.5], [0.5, 0.5]])
    if flat == "one":
        env, GOAL_POS, GOAL_RADIUS = flat_one(env=env,
                                              GOAL_POS=GOAL_POS,
                                              GOAL_RADIUS=GOAL_RADIUS,
                                              REWARD_DURATION=REWARD_DURATION)
    elif flat == "two":
        env, GOAL_POS, GOAL_RADIUS = flat_two(env=env,
                                              GOAL_POS=GOAL_POS,
                                              GOAL_RADIUS=GOAL_RADIUS,
                                              REWARD_DURATION=REWARD_DURATION)
    elif flat == "three":
        env, GOAL_POS, GOAL_RADIUS = flat_three(env=env,
                                                GOAL_POS=GOAL_POS,
                                                GOAL_RADIUS=GOAL_RADIUS,
                                                REWARD_DURATION=REWARD_DURATION)
    else:
        env, GOAL_POS, GOAL_RADIUS = simple_flat(env=env,
                                                 GOAL_POS=GOAL_POS,
                                                 GOAL_RADIUS=GOAL_RADIUS,
                                                 REWARD_DURATION=REWARD_DURATION)

    env.exploration_strength = 1

    # Make the reward which is given when a spatial goal is satisfied.
    # Attached this goal to the environment
    reward = Reward(REWARD,
                    decay="none",
                    expire_clock=REWARD_DURATION,
                    dt=DT,)

    if GOAL_POS is not None:
        if sum(GOAL_POS.shape) > 2:
            goals = [
                SpatialGoal(env, pos=goal_pos,
                            goal_radius=GOAL_RADIUS,
                            reward=reward)
                for goal_pos in GOAL_POS
            ]
        else:
            goals = [SpatialGoal(env, pos=GOAL_POS,
                                 goal_radius=GOAL_RADIUS,
                                 reward=reward)]
    else:
        goals = [SpatialGoal(env, pos=GOAL_POS,
                             goal_radius=GOAL_RADIUS,
                             reward=reward)]

    logger(f"#goals={len(goals)}")

    env.goal_cache.reset_goals = goals

    # Recruit the agent and add it to environment
    agent = Agent(env, params={'dt':DT,
                               'speed_mean': 0.0025,
                               'speed_std': 0.0005})

    # add cells
    PCs = PlaceCells(
        agent,
        params={
            "n": 100,
            "description": "gaussian_threshold",
            "widths": 0.02,
            "wall_geometry": "geodesic",
            "max_fr": 1.,
            "min_fr": 0.0,
            "color": "C1",
        },
    )

    params = {
        "max_fr": 1,
        "min_fr": 0.1,
        "color": "C1",
    }
    if pcnn_params is not None:
        params["n"] = pcnn_params.get("N", 100)

    PCNN = PCNNrx(
        agent,
        params=params,
        sigma=4e-3,
        pcnn_params=pcnn_params,
    )

    logger(f"{PCNN=}")
    logger.debug(f"pcnn params: {pcnn_params}")
    logger.debug(f"params: {params}")

    #
    fov_params = {
        'object_tuning_type': 0,
        'distance_range': [0.1, 0.3],
        'angle_range': [0, 90],
        'spatial_resolution': 0.01
    }
    fov_ovcs = FieldOfViewBVCs(agent,
                               params=fov_params)
    #
    gc_params = {
                 "n": 40,
        "gridscale_distribution": "modules",
        "gridscale": (0.1),
        "orientation_distribution": "modules",
        "orientation": (0.), #radians
        "phase_offset_distribution": "uniform",
        "phase_offset": (0, 2 * np.pi), #degrees
        "description": "three_rectified_cosines",
        "width_ratio": 4/(3*np.sqrt(3)),
    }
    GCs = GridCells(agent,
                    params=gc_params)

    if cell_types is None:
        cell_types = ("PC", "FOV")

    cell_info = {}
    state_cells = []
    for c in cell_types:
        if c == "PC":
            state_cells.append(PCs)
            cell_info["PC"] = pcnn_params
        elif c == "PCNN":
            state_cells.append(PCNN)
            cell_info["PCNN"] = pcnn_params
        elif c == "GC":
            state_cells.append(GCs)
            cell_info["GC"] = gc_params
        elif c == "FOV":
            state_cells.append(fov_ovcs)
            cell_info["FOV"] = fov_params
        else:
            raise ValueError(f"Invalid cell type {c[0]}")

    # this updates the agent creating an off by one error
    env.add_agents(agent)

    superenv = SuperEnv(env=env, cells=state_cells,
                    GOAL_POS=GOAL_POS,
                    GOAL_RADIUS=GOAL_RADIUS,
                    nb_experiences=nb_experiences,
                    experience_duration=experience_duration,
                    max_experiences=max_experiences,
                    max_speed=max_speed,
                    init_pos=init_pos,
                    init_pos_radius=init_pos_radius,
                    max_wall_hits=max_wall_hits)

    if return_info:
        return superenv, agent, cell_info

    return superenv, agent


def display_reward_patch(fig, ax, reward_pos: np.ndarray,
                         reward_radius: float,
                         **kwargs): #we'll also use this later 
    """Plots the reward patch on the given axis"""
    circle = matplotlib.patches.Circle(reward_pos, radius=reward_radius,
                                       facecolor='r', alpha=0.2, color=None) 
    ax.add_patch(circle)
    return fig, ax


def main(env: object, agent: object, model: object,
         t_timeout: int=15):

    obs, _ = env.reset()
    rewards = 0

    # agent.reset_history()

    t_start = env.t

    while env.t < t_timeout:

        action, _state = model.predict(obs, deterministic=True)
        observation, reward_rate, done, _ , info =  env.step(action=action)

        # exit
        if done:
            break

    # display the trajectory
    slice = agent.get_history_slice(t_start=t_start,
                                    t_end=None,
                                    framerate=30)
    history_data = agent.get_history_arrays() # gets history dataframe as dictionary of arrays
    trajectory = history_data["pos"][slice]

    fig, ax = plt.subplots()
    fig, ax = agent.plot_trajectory(fig=fig, ax=ax, color="changing", t_start=0., t_end=None)
    # env._env.plot_environment(autosave=False, fig=fig, ax=ax)
    # ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', alpha=0.25, lw=1)
    # ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='g', s=100, marker='>')

    # display the reward patch
    # fig, ax = display_reward_patch(fig, ax)

    plt.show()


""" miscellanea """


def load_model(env: object, last: bool=True, model_name: str=None,
               nb_time_steps: int=None, choose: bool=False) -> object:

    """
    Load a model from the models/ppo directory.

    Parameters
    ----------
    env : object
        The environment object.
    last : bool
        Whether to load the last model.
        Default is True.
    model_name : str
        The name of the model to load.
        Default is None.
    nb_time_steps : int
        The number of time steps to load.
        For example, 1000 for iter_1000_steps.
        Default is None.
    choose : bool
        Whether to choose a model from the list.
        Default is False.

    Returns
    -------
    ppo : object
        The PPO model.
    """

    all_models = os.listdir("models/ppo/")
    all_models = sorted(all_models, reverse=True)
    all_models = [model for model in all_models if "agent_" in model]

    if model_name is not None:
        pass
    elif choose:
        a_dict = {i: model for i, model in enumerate(all_models)}
        pprint(a_dict)
        model_name = all_models[int(input("Choose a model: "))]
    elif last:
        model_name = all_models[0]
    else:
        model_name = np.random.choice(all_models)

    all_iters = os.listdir(f"models/ppo/{model_name}")

    if nb_time_steps is not None:
        iter_name = f"iter_{nb_time_steps}_steps"
    elif last:
        iter_name = all_iters[-1]
    else:
        iter_name = np.random.choice(all_iters)

    ppo = PPO.load(f"models/ppo/{model_name}/{iter_name}",
                   env=env)

    return ppo



if __name__ == "__main__":

    """ args """

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10_000, type=int)
    args = parser.parse_args()

    """ train """

    T_TIMEOUT = 1000

    # Optionally check the environment (useful during
    # development)
    env, agent = generate_navigation_task_env(
        IS_PCNN_flag=True,
        flat="three",
        GOAL_POS=np.around([0.5, 0.3]))

    fig, ax = plt.subplots()
    env._env.plot_environment(autosave=False, fig=fig, ax=ax)
    fig, ax = display_reward_patch(fig, ax,
                                   reward_pos=env.GOAL_POS,
                                reward_radius=env.GOAL_RADIUS)
    # plt.show()

    # raise ValueError('This is a test')

    # model = A2C("MlpPolicy", env, verbose=1)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
    )
    model.learn(total_timesteps=args.epochs,
                log_interval=500,
                progress_bar=True)

    # # test
    main(env=env, agent=agent, model=model, t_timeout=T_TIMEOUT)
