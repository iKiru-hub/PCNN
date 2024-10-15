from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

import minigrid
from minigrid.wrappers import ImgObsWrapper

import gymnasium as gym
import torch
import torch as th
import torch.nn as nn
from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Box, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import numpy as np
import pygame
# import matplotlib
# matplotlib.use('cairo')
import matplotlib.pyplot as plt
import argparse
import warnings
from datetime import datetime
from time import time
import os

try:
    from tools.utils import logger, tqdm_enumerate
except ModuleNotFoundError:
    warnings.warn('`tools.utils` not found, using fake logger. Some functions may not work')
    class Logger:

        print('Logger not found, using fake logger')

        def info(self, msg: str):
            print(msg)

        def debug(self, msg: str):
            print(msg)

    logger = Logger()
try:
    import inputools.Trajectory as it
    # from inputools.visualizations import plot_activation
except ModuleNotFoundError:
    raise ModuleNotFoundError('module `inputools` not found')

# append the path one directory up
import sys
sys.path.append("..")
import models as mm


""" settings """

HEIGHT = 13
WIDTH = 13
AGENT_VIEW_SIZE = 5
IMAGE_SIZE = AGENT_VIEW_SIZE * AGENT_VIEW_SIZE * 3 + (WIDTH-2)*(HEIGHT-2)

GAME_NAME = "nRooms"
RENDER_PC = False

""" Rooms """


class FourRoomFlat(MiniGridEnv):

    """
    An environment that resembles a flat with four rooms and
    objects in each room. The agent has to go to the object.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 2,
    }

    def __init__(self, width=13, height=13,
                 agent_start_pos=(9, 4),
                 agent_start_dir=0,
                 max_steps: int | None = None, **kwargs):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES],
        )

        if max_steps is None:
            max_steps = 4 * width*height

        super().__init__(mission_space=mission_space, width=width,
                         height=height, see_through_walls=False,
                         max_steps=max_steps,
                         agent_view_size=5,
                         **kwargs)

    @staticmethod
    def _gen_mission(color: str):
        return f"go to the {color} box"

    def _gen_grid(self, width: int=20, height: int=40):

        # create the grid
        self.grid = Grid(width, height)

        # generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # make walls
        self.grid.vert_wall(width//2, 0, height//4)
        self.grid.vert_wall(width//2, height//4+1, height//2+2)
        # self.grid.vert_wall(width//2, height - height//4,
        #                     height//4-1)

        # make rooms walls
        self.grid.horz_wall(0, height//2, width//2-2)
        self.grid.horz_wall(width//2-1, height//2, width//2-4)

        # Set the 4 object's positions
        ObjPos = [(3, height-3), (width-3, height-3),
                  (width//2-2, height//2), (1, 3)]
        ObjList = [
            {
                "name": "ball",
                "obj": Ball,
            },
            {
                "name": "key",
                "obj": Key,
            },
            {
                "name": "door",
                "obj": Door,
            },
            {
                "name": "ball",
                "obj": Ball,
            },
        ]

        # Generate the object's colors
        Colors = ["red", "blue", "blue", "purple"]
        Objs = [obj for obj in ObjList]
        # while len(Colors) < len(ObjPos):
        #     color = self._rand_elem(COLOR_NAMES)
        #     obj = self._rand_elem(ObjList)
        #     Colors.append(color)
        #     Objs.append(obj)

        # Place the objects in the grid
        for idx, pos in enumerate(ObjPos):
            color = Colors[idx]
            obj = Objs[idx]
            self.grid.set(*pos, obj["obj"](color,))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Select a random target object
        boxIdx = self._rand_int(0, len(ObjPos))
        self.target_pos = ObjPos[boxIdx]
        self.target_obj = Objs[boxIdx]["name"]
        self.target_color = Colors[boxIdx]

        # Generate the mission string
        self.mission = f"Collect the `red` or `purple` ball"

    def step(self, action):

        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        if self.agent_dir == 0:
            next_ax = ax + 1
            next_ay = ay
        elif self.agent_dir == 1:
            next_ax = ax
            next_ay = ay + 1
        elif self.agent_dir == 2:
            next_ax = ax - 1
            next_ay = ay
        elif self.agent_dir == 3:
            next_ax = ax
            next_ay = ay - 1

        # Don't let the agent open any of the doors
        if action == self.actions.toggle:
            terminated = True

        # Reward performing done action in front of the target door
        # if action == self.actions.done:
        if next_ax == tx and next_ay == ty:
            reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info


class TwoRoomFlat(MiniGridEnv):

    """
    An environment that resembles a flat with three rooms and
    objects in each room. The agent has to go to the object.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 2,
    }

    def __init__(self, width=WIDTH, height=HEIGHT, agent_start_pos=(3, 4),
                 agent_start_dir=0,
                 max_steps: int | None = None, **kwargs):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES],
        )

        if max_steps is None:
            max_steps = 4 * width*height

        super().__init__(mission_space=mission_space, width=width,
                         height=height, see_through_walls=False,
                         max_steps=max_steps,
                         **kwargs)

    @staticmethod
    def _gen_mission(color: str):
        return f"go to the {color} box"

    def _gen_grid(self, width: int, height: int):

        # create the grid
        self.grid = Grid(width, height)

        # generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # make rooms
        self.grid.vert_wall(width//2, 0, height//2)
        self.grid.vert_wall(width//2, height//2+2, height - height//2-2)

        # Set the 4 object's positions
        ObjPos = [(WIDTH-2, 7), (5, 8)]
        ObjList = [
            {
                "name": "ball",
                "obj": Ball,
            },
            {
                "name": "key",
                "obj": Key,
            },
        ]

        # Generate the object's colors
        Colors = []
        Objs = []
        while len(Colors) < len(ObjPos):
            color = self._rand_elem(COLOR_NAMES)
            obj = self._rand_elem(ObjList)
            Colors.append(color)
            Objs.append(obj)

        # Place the objects in the grid
        for idx, pos in enumerate(ObjPos):
            color = Colors[idx]
            obj = Objs[idx]
            self.grid.set(*pos, obj["obj"](color,))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Select a random target object
        boxIdx = self._rand_int(0, len(ObjPos))
        self.target_pos = ObjPos[boxIdx]
        self.target_obj = Objs[boxIdx]["name"]
        self.target_color = Colors[boxIdx]

        # Generate the mission string
        self.mission = f"go to the {self.target_color} {self.target_obj}"

    def step(self, action):

        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        if self.agent_dir == 0:
            next_ax = ax + 1
            next_ay = ay
        elif self.agent_dir == 1:
            next_ax = ax
            next_ay = ay + 1
        elif self.agent_dir == 2:
            next_ax = ax - 1
            next_ay = ay
        elif self.agent_dir == 3:
            next_ax = ax
            next_ay = ay - 1

        # Don't let the agent open any of the doors
        if action == self.actions.toggle:
            terminated = True

        # Reward performing done action in front of the target door
        # if action == self.actions.done:
        if next_ax == tx and next_ay == ty:
            reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info


class OneRoomFlat(MiniGridEnv):

    """
    An environment that resembles a flat with three rooms and
    objects in each room. The agent has to go to the object.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 2,
    }

    def __init__(self, width=WIDTH, height=HEIGHT, agent_start_pos=(5, 5),
                 agent_start_dir=0,
                 max_steps: int | None = None, **kwargs):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES])

        if max_steps is None:
            max_steps = 4 * width*height

        super().__init__(mission_space=mission_space, width=width,
                         height=height, see_through_walls=False,
                         max_steps=max_steps,
                         **kwargs)

    @staticmethod
    def _gen_mission(color: str):
        return f"go to the {color} box"

    def _gen_grid(self, width: int, height: int):

        # create the grid
        self.grid = Grid(width, height)

        # generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Set the 4 object's positions
        ObjPos = [(3, 1),
                  (6, 6)]
        ObjList = [
            {
                "name": "ball",
                "obj": Ball,
            },
            {
                "name": "ball",
                "obj": Ball,
            },
        ]

        # Generate the object's colors
        Colors = []
        Objs = []
        while len(Colors) < len(ObjPos):
            color = self._rand_elem(COLOR_NAMES)
            obj = self._rand_elem(ObjList)
            Colors.append(color)
            Objs.append(obj)

        # Place the objects in the grid
        for idx, pos in enumerate(ObjPos):
            color = Colors[idx]
            obj = Objs[idx]
            self.grid.set(*pos, obj["obj"](color,))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Select a random target object
        boxIdx = self._rand_int(0, len(ObjPos))
        self.target_pos = ObjPos[boxIdx]
        self.target_obj = Objs[boxIdx]["name"]
        self.target_color = Colors[boxIdx]

        # Generate the mission string
        self.mission = f"go to the {self.target_color} {self.target_obj}"

    def step(self, action):

        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        if self.agent_dir == 0:
            next_ax = ax + 1
            next_ay = ay
        elif self.agent_dir == 1:
            next_ax = ax
            next_ay = ay + 1
        elif self.agent_dir == 2:
            next_ax = ax - 1
            next_ay = ay
        elif self.agent_dir == 3:
            next_ax = ax
            next_ay = ay - 1

        # Don't let the agent open any of the doors
        if action == self.actions.toggle:
            terminated = True

        # Reward performing done action in front of the target door
        # if action == self.actions.done:
        if next_ax == tx and next_ay == ty:
            reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info



""" Observations """


pc_layer = it.PlaceLayer(N=(WIDTH-2)*(HEIGHT-2), sigma=1,
                         bounds=(1, WIDTH-2, 1, HEIGHT-2))
# # pc_layer = it.PlaceLayer(N=400, sigma=1.,
# #                          bounds=(0, WIDTH-2, 0, HEIGHT-2))
# logger(pc_layer)

Nj = (WIDTH*2)*(2*HEIGHT)
N = WIDTH*HEIGHT
# pc_layer = mm.make_stored_super(N=N, Nj=Nj, sigma=2,
#                                 bounds=(1, WIDTH-2, 1, HEIGHT-2),
#                                 interval=50)
# logger(pc_layer)


fig, ax = plt.subplots(1, 1, figsize=(7, 7))


class ObjObsWrapper(ObservationWrapper):

    def __init__(self, env: object):

        """
        A wrapper that makes image the only observation.

        Parameters
        ----------
        env : object 
            The environment to apply the wrapper
        """

        super().__init__(env)

        self.observation_space = Dict(
            {
                "image": env.observation_space.spaces["image"],
                "mission": Box(low=0.0, high=1.0, 
                               shape=(9,), dtype=np.float32),
                "pc": Box(low=0.0, high=1.0,
                          shape=(pc_layer.N,), dtype=np.float32),
            }
        )

        self.color_one_hot_dict = {
            "red": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "green": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "blue": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "purple": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "yellow": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "grey": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        }

        self.obj_one_hot_dict = {
            "ball": np.array([1.0, 0.0, 0.0]),
            "key": np.array([0.0, 1.0, 0.0]),
            "door": np.array([0.0, 0.0, 1.0]),
        }

    def observation(self, obs):

        mission_array = np.concatenate(
            [
                self.color_one_hot_dict[self.target_color],
                self.obj_one_hot_dict[self.target_obj],
            ]
        )

        # g, v = self.env.gen_obs_grid()
        # o = self.env.gen_obs()

        # pc layer observation
        pc = pc_layer.step(position=np.array(self.env.agent_pos),
                           max_rate=1.).reshape(-1)

        # print(f"{v.shape=}, - \n{obs['image'].shape=}, \n{pc.shape=}")

        # print(np.around(pc.reshape(pc_layer.n, pc_layer.n).T))

        if RENDER_PC:
            it.plot_activation(x=pc, ax=ax, title='Place Cell Layer',
                               shape=(pc_layer.n, pc_layer.n),
                               figsize=(20, 20))
            # ax.clear()
            # ax.imshow(pc_layer.pcnn.Wff, vmin=0, vmax=0.11)
            # plt.pause(0.00001)

        # obs = np.concatenate([obs["image"].flatten(), pc.flatten()], axis=0)
        # print(f"{obs.shape=}")

        # logger.debug(f"{obs['image'].shape=}, {mission_array.shape=}, {pc.shape=}")

        wrapped_obs = {
            "image": obs["image"],
            "mission": mission_array,
            "pc": pc,
        }

        return wrapped_obs


class ObjEnvExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: Dict):

        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        print(f"Observation space: {observation_space}")
        for key, subspace in observation_space.spaces.items():
            print(f"Key: {key}, subspace: {subspace}")
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                # cnn = nn.Sequential(
                #     nn.Conv2d(3, 16, (2, 2)),
                #     nn.ReLU(),
                #     nn.Conv2d(16, 32, (2, 2)),
                #     nn.ReLU(),
                #     nn.Conv2d(32, 64, (2, 2)),
                #     nn.ReLU(),
                #     nn.Flatten(),
                # )
                mlp = nn.Sequential(
                    nn.Linear(IMAGE_SIZE, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                # Compute shape by doing one forward pass
                with th.no_grad():
                    sub_obs = subspace.sample()[None]
                    sub_pc = pc_layer.step(position=np.array([5, 5]))
                    sub_concat = np.concatenate([sub_obs.flatten(),
                                                 sub_pc.flatten()], axis=0).reshape(-1, 1)
                    n_flatten = mlp(
                        th.as_tensor(sub_concat).float()
                    ).shape[1]

                print(f"n_flatten: {n_flatten.shape}")
                linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())
                extractors["image"] = nn.Sequential(*(list(mlp) + list(linear)))
                total_concat_size += 64

            elif key == "mission":
                extractors["mission"] = nn.Linear(IMAGE_SIZE, 32)
                total_concat_size += 32

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:

        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            print(f"observations: {observations[key].shape}")
            encoded_tensor_list.append(extractor(observations[key]))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


class MinigridFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space, features_dim: int = 512,
                 normalized_image: bool = False) -> None:

        logger.debug(f"Observation space: {observation_space}")

        super().__init__(observation_space, features_dim)
        logger.debug(f"Observation space2: {observation_space}")
        observation_space = observation_space['image']
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class CustomCombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):

        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():

            if key == "image":

                cnn = nn.Sequential(
                    nn.Conv2d(subspace.shape[0], 16, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(32, 16, (2, 2)),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(256, 16),
                    nn.ReLU(),
                )

                # append
                extractors[key] = cnn
                # total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
                total_concat_size += 16

            elif key == "vector":

                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


def load_model(last: bool=True, run_name: str=None):

    if run_name is None:
        run_name = [run for run in os.listdir(f"models/ppo") if "nRooms" in run]
        print(run_name)
        run_name = sorted(run_name, reverse=True)[0]

    print(f"{run_name=}")

    runs = os.listdir(f"models/ppo/{run_name}")

    logger.debug(f"{runs=}")

    print(sorted(runs, reverse=True)[0])
    model = sorted(runs, reverse=True)[0][:-4]

    logger.debug(f"{model=}")

    return f"{run_name}/{model}"




""" Main functions """


def main_manual(env: object):

    # wrap the environment
    env = ObjObsWrapper(env)

    logger.debug(f"Observation space: {env.observation_space}")

    # start the manual control
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


def main_train(env: object, **kwargs):

    """
    Main function to train the model
    """

    # policy_kwargs = dict(features_extractor_class=ObjEnvExtractor)
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor)
    logger(f"Policy kwargs: {policy_kwargs}")

    # Create time stamp of experiment
    stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")

    nb_time_steps = int(kwargs.get("nb_time_steps", 1e6))

    checkpoint_callback = CheckpointCallback(
        save_freq=kwargs.get("save_freq", 1e5),
        save_path=f"./models/ppo/{GAME_NAME}_{stamp}/",
        name_prefix="iter",
    )

    # Create the model
    model = PPO(
        "MultiInputPolicy",
        # "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    # Train the model
    model.learn(
        total_timesteps=nb_time_steps,
        tb_log_name=f"{stamp}",
        callback=checkpoint_callback,
        progress_bar=True,
    )


def main_eval(env: object, **kwargs):

    """
    Main function to evaluate the model

    Parameters
    ----------
    env : object
        The environment to evaluate the model
    """

    # settings
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor)

    ppo = PPO("MultiInputPolicy", env,
              policy_kwargs=policy_kwargs, verbose=1)

    # add the experiment time stamp
    if kwargs.get("load_model", False):
        model_name = load_model(last=False)
        ppo = ppo.load(
            f"models/ppo/{model_name}",
                       env=env)
    else:
        # Create the model
        policy_kwargs = dict(features_extractor_class=ObjEnvExtractor)
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

    # ppo = ppo.load(f"models/ppo/{model_name}", env=env)
    # ppo = ppo.load(f"models/ppo/{args.load_model}", env=env)

    obs, info = env.reset()
    rewards = 0

    for i in range(200):
        action, _state = ppo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards += reward

        if terminated or truncated:
            print(f"Test reward: {rewards} - {terminated} - {truncated}")
            obs, info = env.reset()
            rewards = 0
            continue

    print(f"Test reward: {rewards}")

    env.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--manual", action="store_true", help="train the model")
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument("--rooms", type=int, default=1, help="number of room of the flat")
    parser.add_argument("--load_model", action="store_true", help="load a model to evaluate")
    parser.add_argument('--steps', type=int, help='number of steps to train', default=1e6)

    args = parser.parse_args()


    """ set up the environment """

    if args.rooms == 1:
        if args.render:
            env = OneRoomFlat(render_mode="human")
            RENDER_PC = True
        else:
            env = OneRoomFlat()
    elif args.rooms == 2:
        if args.render:
            env = TwoRoomFlat(render_mode="human")
            RENDER_PC = True
        else:
            env = TwoRoomFlat()
    else:
        if args.render:
            env = FourRoomFlat(render_mode="human")
            RENDER_PC = True
        else:
            env = FourRoomFlat()

    # env = gym.make("MiniGrid-LockedRoom-v0", render_mode="human")
    env = ObjObsWrapper(env)
    # env = ImgObsWrapper(env)
    # env = CustomCombinedExtractor(env.observation_space)


    """ set main """

    if args.manual and args.train:
        warnings.warn("Both `manual` and `train` flags are set. Ignoring `train`")

    if args.manual:
        logger("<manual>")
        if not args.render:
            warnings.warn("`render` flag not set, no display will be shown.")
        main_manual(env=env)
    elif args.train:
        logger("<train>")
        main_train(env=env, nb_time_steps=args.steps)
    else:
        logger("<eval>")
        main_eval(env=env, nb_time_steps=args.steps,
                  load_model=args.load_model)

