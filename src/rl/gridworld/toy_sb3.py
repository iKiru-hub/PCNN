import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from stable_baselines3.common.base_class import BaseAlgorithm

from stable_baselines3.common.evaluation import evaluate_policy


def evaluate(
    model: BaseAlgorithm,
    num_episodes: int = 100,
    deterministic: bool = True,
) -> float:
    """
    Evaluate an RL agent for `num_episodes`.

    :param model: the RL Agent
    :param env: the gym Environment
    :param num_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or 
        stochastic actions
    :return: Mean reward for the last `num_episodes`
    """
    # This function will only work for a single environment
    vec_env = model.get_env()
    obs = vec_env.reset()
    all_episode_rewards = []
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        # Note: SB3 VecEnv resets automatically:
        # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
        # obs = vec_env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            # `deterministic` is to use deterministic actions
            action, _states = model.predict(obs,
                                deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, _info = vec_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print(f"Mean reward: {mean_episode_reward:.2f} -" + \
        f" Num episodes: {num_episodes}")

    return mean_episode_reward


if __name__ == '__main__':


    """ init """

    env = gym.make("CartPole-v1")

    model = PPO(MlpPolicy, env, verbose=1)


    """ pre-training evaluation """

    # Random Agent, before training
    mean_reward_before_train = evaluate(model,
                    num_episodes=100, deterministic=True)

    print("%Mean reward before training:",
          mean_reward_before_train, "\n-----\n")

    mean_reward, std_reward = evaluate_policy(model,
                    env, n_eval_episodes=100, warn=False)

    print(f"%Mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}", 
          "\n-----\n")


    """ training """

    # Train the agent
    model.learn(total_timesteps=10000)

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model,
                                env, n_eval_episodes=100)

    print(f"\n%Mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")



