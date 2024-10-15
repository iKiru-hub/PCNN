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
import argparse
from datetime import datetime
from time import time
import os



""" Environments """

class GoToObjEnv(MiniGridEnv):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(
        self,
        size=11,
        agent_start_pos=(5, 5),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.size = size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES],
        )

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str):
        return f"go to the {color} box"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Set the 4 object's positions
        ObjPos = [(5, 1), (9, 5)]
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


class SimpleEnv(MiniGridEnv):

    def __init__(self, size=50, agent_start_pos=(1, 1), agent_start_dir=0,
        max_steps: int | None = None, **kwargs):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width: int, height: int):

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"



""" Wrappers """


class ObjObsWrapper(ObservationWrapper):
    def __init__(self, env):
        """A wrapper that makes image the only observation.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        self.observation_space = Dict(
            {
                "image": env.observation_space.spaces["image"],
                "mission": Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32),
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
            "box": np.array([0.0, 1.0, 0.0]),
            "key": np.array([0.0, 0.0, 1.0]),
        }

    def observation(self, obs):
        mission_array = np.concatenate(
            [
                self.color_one_hot_dict[self.target_color],
                self.obj_one_hot_dict[self.target_obj],
            ]
        )

        wrapped_obs = {
            "image": obs["image"],
            "mission": mission_array,
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
                cnn = nn.Sequential(
                    nn.Conv2d(3, 16, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, (2, 2)),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                # Compute shape by doing one forward pass
                with th.no_grad():
                    n_flatten = cnn(
                        th.as_tensor(subspace.sample()[None]).float()
                    ).shape[1]

                linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())
                extractors["image"] = nn.Sequential(*(list(cnn) + list(linear)))
                total_concat_size += 64

            elif key == "mission":
                extractors["mission"] = nn.Linear(subspace.shape[0], 32)
                total_concat_size += 32

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


class MinigridFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space, features_dim: int = 512,
                 normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)

        print(f"Observation space: {observation_space}")
        print(f"Features dim: {features_dim}")

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



""" Main """

def main_manual():
    env = SimpleEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

## the `Box` is not used

def load_model(env: object, nb_time_steps: int,
               last: bool=True,
               model_name: str=None) -> object:

    # write docstring
    """
    Load a model from the models/ppo directory

    Parameters
    ----------
    env : object
        The environment object
    nb_time_steps : int
        The number of time steps the model was trained for
    last : bool
        Whether to load the last model
    model_name : str
        The name of the model to load

    Returns
    -------
    ppo : object
        The loaded model
    """

    all_models = os.listdir("models/ppo/")
    all_models = sorted(all_models, reverse=True)
    all_models = [model for model in all_models \
        if "agent_" in model]

    if model_name is not None:
        pass
    elif last:
        model_name = all_models[0]
    else:
        model_name = np.random.choice(all_models)

    all_iter = os.listdir(f"models/ppo/{model_name}")

    if nb_time_steps is not None:
        iter_name = f"iter_{nb_time_steps}_steps"
    elif last:
        iter_name = all_iter[-1]
    else:
        iter_name = np.random.choice(all_iter)

    ppo = PPO.load(f"models/ppo/{model_name}/{iter_name}",
                   env=env)

    return ppo


def main(args):

    # policy_kwargs = dict(features_extractor_class=ObjEnvExtractor)
    policy_kwargs = dict(features_extractor_class=MinigridFeaturesExtractor,
                         features_extractor_kwargs=dict(features_dim=128))
    print(f"Policy kwargs: {policy_kwargs}")

    # Create time stamp of experiment
    stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")

    nb_time_steps = int(args.steps)

    if args.train:
        env = GoToObjEnv()
        env = ImgObsWrapper(env)

        checkpoint_callback = CheckpointCallback(
            save_freq=1e4,
            save_path=f"./models/ppo/minigrid_gotoobj_{stamp}/",
            name_prefix="iter",
        )

        model = PPO(
            # "MultiInputPolicy",
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        model.learn(
            total_timesteps=nb_time_steps,
            tb_log_name=f"{stamp}",
            callback=checkpoint_callback,
            progress_bar=True,
        )
    else:
        if args.render:
            env = GoToObjEnv(render_mode="human")
            print("Rendering the environment")
        else:
            env = GoToObjEnv()
        # env = ObjObsWrapper(env)
        env = ImgObsWrapper(env)

        ppo = PPO(#"MultiInputPolicy", 
                  "CnnPolicy",
                  env,
                  policy_kwargs=policy_kwargs,
                  verbose=1)

        # add the experiment time stamp
        if args.load_model:
            model_name = load_model(last=True)
            ppo = ppo.load(
                f"models/ppo/{model_name}/iter_{nb_time_steps}_steps",
                           env=env)
            # logger.debug(f"Loaded model: {model_name}")

        # ppo = ppo.load(f"models/ppo/{model_name}", env=env)
        # ppo = ppo.load(f"models/ppo/{args.load_model}", env=env)

        obs, info = env.reset()
        rewards = 0

        for i in range(200):

            print(f"{i} ", end="")
            action, _state = ppo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards += reward

            if terminated or truncated:
                print(f"\nTest reward: {rewards} - {terminated} - {truncated}")
                obs, info = env.reset()
                rewards = 0
                continue

        print(f"Test reward: {rewards}")

        env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--load_model", action="store", help="load a model")
    parser.add_argument("--manual", action="store", help="load a model")
    parser.add_argument("--render", action="store_true",
                        help="render trained models")
    parser.add_argument("--timesteps", action="store", 
                        help="number of timesteps to train",
                        default="2_000_000", type=str)
    parser.add_argument('--steps', type=int, help='number of steps to train', default=100_000)
    args = parser.parse_args()


    if args.manual:
        main_manual()
    else:
        main(args=args)
