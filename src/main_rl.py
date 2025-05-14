import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
from tqdm import tqdm
import random
from collections import deque, namedtuple
import copy, os
import argparse, time

from stable_baselines3 import DQN, TD3, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

import gymnasium as gym
from gymnasium import spaces

import game.envs as games
from utils import setup_logger
import simulations as sim
import game.objects as objects

# Setup logging
logger = setup_logger(name="RL", level=3, is_debugging=True, is_warning=True)

RLPATH = os.path.join(os.getcwd().split("PCNN")[0], "PCNN/src/rlmodels")


""" SETTINGS """

GAME_SCALE = games.SCREEN_WIDTH

rl_parameters = {
    'model_type': 'PPO',
    'hidden_dim': 64,
    'learning_rate': 0.001,
    'gamma': 0.99,
    'buffer_size': 10000,
    'batch_size': 64,
    'update_every': 4,
    'training_mode': True,
}

reward_settings = {
    "rw_fetching": "deterministic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_position_idx": 3,
    "rw_radius": 0.08 * GAME_SCALE,
    "rw_sigma": 0.8,# * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 5,
    "silent_duration": 0,
    "fetching_duration": 1,

    "transparent": False,
    "beta": 40.,
    "alpha": 0.06,# * GAME_SCALE,
    "move_threshold": 50,# * GAME_SCALE,
    "rw_position_idx": 2,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "rendering": False,
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "max_duration": 20_000,
    "t_teleport": 0,
    "limit_position_len": -1,
    "room_thickness": 20,
    "seed": None,
    "pause": -1,
    "verbose": True,
    "image_obs": True,
    "resize_to": (15, 15),
}

global_parameters = {
    "use_sprites": bool(0),
    "speed": 1.,
    "min_weight_value": 0.5
}



def make_rl_name(model_str: str, idx: int=-1):
    existing = [fname for fname in os.listdir(RLPATH) if model_str in fname]
    num = len(existing) if idx < 0 else idx
    print(f"{num=}")
    return f"model_{model_str}_{num}"


""" RUN FUNCTIONS """


def setup_env(global_parameters: dict,
              reward_settings: dict,
              game_settings: dict,
              room_name: str,
              pause: int=-1, verbose: bool=True,
              record_flag: bool=False,
              limit_position_len: int=-1,
              preferred_positions: list=None,
              verbose_min: bool=True):

    """ make game environment """

    if verbose and verbose_min:
        logger(f"room_name={room_name}")

    room = games.make_room(name=room_name,
                           thickness=game_settings["room_thickness"],
                           bounds=[0, 1, 0, 1])
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    # ===| objects settings |===

    possible_positions = room.get_room_positions()

    # reward
    if reward_settings['rw_position_idx'] > -1:
        rw_position_idx = reward_settings['rw_position_idx']
    else:
        rw_position_idx = np.random.randint(0, len(possible_positions))

    rw_position = possible_positions[rw_position_idx]

    # agent
    agent_position = room.sample_next_position()
    agent_position_list = [p for p in possible_positions]
    del agent_position_list[rw_position_idx]

    rw_tau = reward_settings["tau"] if "tau" in reward_settings else 100

    # ===| object declaration |===

    reward_obj = objects.RewardObj(
                position=rw_position,
                # possible_positions=constants.POSSIBLE_POSITIONS.copy(),
                possible_positions=possible_positions,
                radius=reward_settings["rw_radius"],
                sigma=reward_settings["rw_sigma"],
                fetching=reward_settings["rw_fetching"],
                value=reward_settings["rw_value"],
                bounds=room_bounds,
                delay=reward_settings["delay"],
                use_sprites=global_parameters["use_sprites"],
                silent_duration=reward_settings["silent_duration"],
                tau=rw_tau,
                preferred_positions=preferred_positions,
                move_threshold=reward_settings["move_threshold"],
                transparent=reward_settings["transparent"])

    body = objects.AgentBody(
                position=agent_position,
                speed=global_parameters["speed"],
                possible_positions=possible_positions,
                use_sprites=global_parameters["use_sprites"],
                bounds=game_settings["agent_bounds"],
                room=room,
                limit_position_len=game_settings["limit_position_len"],
                color=(10, 10, 10))

    # --- env
    env = games.Environment(room=room,
                            agent=body,
                            reward_obj=reward_obj,
                            agent_position_list=agent_position_list,
                            rw_event=game_settings["rw_event"],
                            verbose=False,
                            duration=game_settings["max_duration"],
                            visualize=game_settings["rendering"])

    return env


def main_rl(global_parameters: dict,
            reward_settings: dict,
            game_settings: dict,
            room_name: str,
            num_episodes: int,
            num_envs: int=1,
            save: bool=False,
            load: bool=False,
            idx: int=0,
            pause: int=-1,
            verbose: bool=True,
            record_flag: bool=False,
            limit_position_len: int=-1,
            preferred_positions: list=None,
            verbose_min: bool=True) -> dict:

    # env  = setup_env(global_parameters=global_parameters,
    #                  reward_settings=reward_settings,
    #                  game_settings=game_settings,
    #                  room_name=room_name,
    #                  pause=pause, verbose=verbose,
    #                  record_flag=record_flag,
    #                  limit_position_len=limit_position_len,
    #                  preferred_positions=preferred_positions,
    #                  verbose_min=verbose_min)

    def make_env():
        return setup_env(global_parameters=global_parameters,
                         reward_settings=reward_settings,
                         game_settings=game_settings,
                         room_name=room_name,
                         pause=pause, verbose=verbose,
                         record_flag=record_flag,
                         limit_position_len=limit_position_len,
                         preferred_positions=preferred_positions,
                         verbose_min=verbose_min)


    record, model = run_rl_model_2(make_env=make_env,
                                   load=load,
                                   idx=idx,
                                   num_episodes=num_episodes,
                                   num_envs=num_envs,
                                   model_type=rl_parameters['model_type'],
                                   learning_rate=rl_parameters['learning_rate'],
                                   gamma=rl_parameters['gamma'],
                                   buffer_size=rl_parameters['buffer_size'],
                                   batch_size=rl_parameters['batch_size'],
                                   plot_interval=game_settings['plot_interval'],
                                   t_teleport=game_settings['t_teleport'],
                                   pause=game_settings['pause'],
                                   record_flag=False,
                                   verbose=game_settings['verbose'],
                                   verbose_min=verbose_min,
                                   save=save,
                                   training_mode=rl_parameters['training_mode'])

    return record


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_output_dim: int = 128, mlp_output_dim: int = 32):
        super().__init__(observation_space, features_dim=cnn_output_dim + 3 * mlp_output_dim)

        self.image_space = observation_space["image"]
        self.reward_space = observation_space["reward"]
        self.collision_space = observation_space["collision"]
        self.velocity_space = observation_space["velocity"]

        # Revised CNN for image (smaller kernels and strides)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.image_space.shape[0], 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(288, 32),
            nn.ReLU(),
        )

        # MLP for other inputs
        self.flatten = FlattenExtractor(self.reward_space)
        self.mlp_reward = nn.Sequential(
            nn.Linear(self.flatten.features_dim, mlp_output_dim),
            nn.ReLU()
        )
        self.mlp_collision = nn.Sequential(
            nn.Linear(self.flatten.features_dim, mlp_output_dim),
            nn.ReLU()
        )
        self.mlp_velocity = nn.Sequential(
            nn.Linear(self.velocity_space.shape[0], mlp_output_dim),
            nn.ReLU()
        )

        logger.debug(f"{cnn_output_dim=} {mlp_output_dim=}")

    def _calculate_cnn_output(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            output = self.cnn(dummy_input)
            return output.shape[1]

    def forward(self, observations: dict) -> torch.Tensor:
        image_obs = observations["image"]
        reward_obs = observations["reward"]
        collision_obs = observations["collision"]
        velocity_obs = observations["velocity"]

        image_features = self.cnn(image_obs)
        reward_features = self.mlp_reward(self.flatten(reward_obs))
        collision_features = self.mlp_collision(self.flatten(collision_obs))
        velocity_features = self.mlp_velocity(velocity_obs)

        combined_features = torch.cat([image_features, reward_features, collision_features, velocity_features], dim=1)
        return combined_features


def run_rl_model_2(make_env: callable,
                   num_episodes: int,
                   model_type: str,
                   num_envs: int = 1,
                   load: bool = False,
                   idx: int = 0,
                   learning_rate=0.001,
                   gamma=0.99,
                   buffer_size=10000,
                   batch_size=64,
                   renderer=None,
                   plot_interval=10,
                   t_teleport=100,
                   pause=-1,
                   record_flag=False,
                   verbose=True,
                   verbose_min=True,
                   training_mode=True,
                   save=True):

    def _make_env():
        env = make_env()
        env = games.EnvironmentWrapper(env=env,
                                     resize_to=game_settings['resize_to'])
        env.set_speed(global_parameters["speed"])

        if not isinstance(env, gym.Env):
            logger.error("not gym.Env")
            raise ValueError("Your environment must inherit from gym.Env")
        return env

    if num_envs > 1:
        logger.info(f"Using {num_envs} environments in parallel")

    vec_env = DummyVecEnv([lambda: _make_env()] * num_envs)

    model_classes = {
        "DQN": DQN,
        "TD3": TD3,
        "PPO": PPO
    }

    model_class = model_classes.get(model_type)
    if model_class is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    else:
        logger.info(f"using {model_class=}")

    model_path = f"{RLPATH}/{make_rl_name(model_type, idx)}"

    # Model-specific configuration
    model_kwargs = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "verbose": 1 if verbose else 0,
        "device": "cuda",  # GPU usage
    }

    if model_type in ["DQN", "TD3"]:
        model_kwargs.update({
            "buffer_size": buffer_size,
            "batch_size": batch_size,
        })

    # Determine policy type based on observation space
    observation_space = vec_env.observation_space
    policy_kwargs = {}
    if isinstance(observation_space, spaces.Dict):
        policy_kwargs["features_extractor_class"] = CustomCombinedExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "cnn_output_dim": 32,  # Adjust as needed
            "mlp_output_dim": 8    # Adjust as needed
        }
        policy_kwargs["net_arch"] = [64, 64]  # MLP layers after feature extraction
        policy_class = "MultiInputPolicy"
    else:
        policy_class = "MlpPolicy" # Fallback to MlpPolicy if observation space is not a Dict

    # Load or initialize model
    logger.debug(f"..{model_path=}")
    if load and os.path.exists(model_path + ".zip"):
        model = model_class.load(model_path, env=vec_env, device="cuda", verbose=0)
        logger.info(f"Loaded model from {model_path}")
    else:
        model = model_class(policy_class, vec_env, policy_kwargs=policy_kwargs, **model_kwargs)
        logger.info("new model")

    # Training
    if training_mode:
        model.learn(total_timesteps=num_episodes, progress_bar=True)
        if save:
            os.makedirs(RLPATH, exist_ok=True)
            model.save(model_path)
            logger.info(f"[✓] Trained model saved to {model_path}")

    # Evaluation
    eval_env = _make_env()
    for _i in range(100):
        obs, _ = eval_env.reset()
        record = {"activity": [], "trajectory": []}

        # ---
        for _ in range(eval_env.duration):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(action)

            if record_flag:
                record["trajectory"].append(eval_env.unwrapped.position)

            try:
                if done:
                    break
            except Exception:
                done = done[0]
                if done:
                    break

            if eval_env.visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        break

            if eval_env.visualize and eval_env.t % plot_interval == 0:
                eval_env.render()

        #
        eval_env.render()
        logger.info(f"test {_i} | R={reward}")

    logger.info(f"rw_count={eval_env.rw_count}")

    return record, model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=-1)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--rendering", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--interval", type=int, default=20,
                        help="plotting interval")
    parser.add_argument("--room", type=str, default="Square.v0",
                        help='room name: ["Square.v0", "Square.v1",' + \
                         ' "Square.v2","Hole.v0", "Flat.0000", "Flat.0001",' + \
                         '"Flat.0010", "Flat.0011", "Flat.0110", ' + \
                         '"Flat.1000", "Flat.1001", "Flat.1010",' + \
                         '"Flat.1011", "Flat.1110"] or `random`')
    args = parser.parse_args()

    if args.rendering:
        game_settings['rendering'] = True

    if args.test:
        rl_parameters['training_mode'] = False

    game_settings['max_duration'] = args.duration

    # main()
    main_rl(global_parameters=global_parameters,
            reward_settings=reward_settings,
            game_settings=game_settings,
            room_name=args.room,
            num_episodes=args.episodes,
            num_envs=args.num_envs,
            save=args.save,
            load=args.load,
            idx=args.idx,
            pause=-1,
            verbose=False,
            record_flag=False,
            limit_position_len=2,
            preferred_positions=None,
            verbose_min=True)



