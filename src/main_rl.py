import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
from tqdm import tqdm
import random
from collections import deque, namedtuple
import copy, os
import argparse

from stable_baselines3 import DQN, TD3, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

import game.envs as games
from utils import setup_logger
import simulations as sim

# Setup logging
logger = setup_logger(name="RL", level=2, is_debugging=True, is_warning=True)

RLPATH = os.path.join(os.getcwd().split("PCNN")[0], "PCNN/src/rlmodels")


""" SETTINGS """

GAME_SCALE = games.SCREEN_WIDTH

rl_parameters = {
    'model_type': 'TD3',
    'hidden_dim': 64,
    'learning_rate': 0.001,
    'gamma': 0.99,
    'buffer_size': 10000,
    'batch_size': 64,
    'update_every': 4,
    'training_mode': True}


reward_settings = {
    "rw_fetching": "probabilistic",
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
    "move_threshold": 20,# * GAME_SCALE,
}


game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "rendering": False,
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "max_duration": 20_000,
    "t_teleport": 1_000,
    "limit_position_len": -1,
    "room_thickness": 20,
    "seed": None,
    "pause": -1,
    "verbose": True
}

global_parameters = {
    "local_scale": 0.015,
    "N": 25**2,
    "use_sprites": bool(0),
    "speed": 0.7,
    "min_weight_value": 0.5
}


""" RL MODELS """

# Define the experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer to store and sample past experiences"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # Convert to numpy arrays to ensure consistent types
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=min(batch_size, len(self.buffer)))

        # First collect all states, actions, etc. into separate lists
        states_list = [e.state for e in experiences]
        actions_list = [e.action for e in experiences]
        rewards_list = [e.reward for e in experiences]
        next_states_list = [e.next_state for e in experiences]
        dones_list = [e.done for e in experiences]

        # Then convert each list to a single numpy array before creating tensors
        states = torch.FloatTensor(np.array(states_list, dtype=np.float32))
        actions = torch.FloatTensor(np.array(actions_list, dtype=np.float32))
        rewards = torch.FloatTensor(np.array(rewards_list, dtype=np.float32))
        next_states = torch.FloatTensor(np.array(next_states_list, dtype=np.float32))
        dones = torch.FloatTensor(np.array(dones_list, dtype=np.float32))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Deep Q-Network for continuous action space"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    """Deep Q-Network agent implementation"""

    def __init__(self, state_dim, action_dim, action_space, lr=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, update_every=4):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space  # List of possible discrete actions
        self.initial_epsilon = epsilon_start

        # Q-Networks (online and target)
        self.qnetwork_local = QNetwork(state_dim, len(action_space))
        self.qnetwork_target = QNetwork(state_dim, len(action_space))
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.tau = 0.001
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_every = update_every
        self.t_step = 0

        # For compatibility with the custom model interface
        self._internal_state = np.zeros(state_dim)
        self.current_action = np.zeros(action_dim)
        self.representation = np.zeros(64)  # For get_representation() compatibility

    def __str__(self):
        return 'DQN'

    def __call__(self, velocity, prev_velocity, collision, reward_availability):
        """Interface compatible with the custom model"""
        # Combine inputs into a state vector
        state = np.concatenate([
            np.array(velocity),
            np.array([prev_velocity]),
            np.array([collision]),
            np.array([reward_availability])
        ])

        self._internal_state = state

        # Use the current policy to select an action
        action = self.act(state)
        self.current_action = action

        return action

    def act(self, state):
        """Select an action using epsilon-greedy policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                self.qnetwork_local.eval()
                action_values = self.qnetwork_local(state_tensor)
                self.qnetwork_local.train()
                action_idx = torch.argmax(action_values).item()

                # Convert discrete action index to continuous velocity
                # This maps from the discrete action space to continuous velocities
                action = self.action_space[action_idx]
        else:
            # Random action from action space
            action_idx = random.randrange(len(self.action_space))
            action = self.action_space[action_idx]

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return action

    def step(self, state, action, reward, next_state, done):
        """Add experience to memory and learn if it's time"""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def learn(self, experiences):
        """Update value parameters using batch of experience tuples"""
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values for next states from target model
        with torch.no_grad():
            next_q_values = self.qnetwork_target(next_states).detach()
            # Get max Q values for each sample in batch
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            # Compute target Q values
            target_q_values = rewards.unsqueeze(1) + (self.gamma * max_next_q_values * (1 - dones.unsqueeze(1)))

        # Get expected Q values from local model
        q_values = self.qnetwork_local(states)

        # Convert actions to indices for discrete action space
        action_indices = torch.zeros(actions.size(0), dtype=torch.int64)
        for i, action in enumerate(actions):
            # Find the closest action in action space
            distances = [np.linalg.norm(action.numpy() - np.array(a)) for a in self.action_space]
            action_indices[i] = np.argmin(distances)

        expected_q_values = q_values.gather(1, action_indices.unsqueeze(1))

        # Compute loss
        loss = nn.MSELoss()(expected_q_values, target_q_values)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update()

    def soft_update(self):
        """Soft update target network parameters"""
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def get_representation(self):
        """Return a representation compatible with the custom model interface"""
        return self.representation

    def save_model(self, path):
        """Save the local and target Q-network models to the given path"""
        torch.save({
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'qnetwork_target': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load_model(self, path):
        """Load the Q-network models from the given path"""
        checkpoint = torch.load(path)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)

    def reset(self):
        """Reset the agent's state for a new episode"""
        self._internal_state = np.zeros(self.state_dim)
        self.current_action = np.zeros(self.action_dim)
        self.representation = np.zeros(64)
        # Reset the exploration parameter
        self.epsilon = self.initial_epsilon


class TD3Agent:
    """Twin Delayed DDPG (TD3) agent implementation for continuous action space"""

    def __init__(self, state_dim, action_dim, max_action=1.0, lr=0.001, gamma=0.99,
                 tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                 buffer_size=10000, batch_size=64):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.initial_exploration_noise = 0.1

        # Actor network
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic networks (twin critics)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

        # Exploration noise
        self.exploration_noise = 0.1

        # For compatibility with the custom model interface
        self._internal_state = np.zeros(state_dim)
        self.current_action = np.zeros(action_dim)
        self.representation = np.zeros(64)  # For get_representation() compatibility

    def __str__(self):
        return 'TD3'

    def __call__(self, velocity, prev_velocity, collision, reward_availability):
        """Interface compatible with the custom model"""
        # Combine inputs into a state vector
        state = np.concatenate([
            np.array(velocity),
            np.array([prev_velocity]),
            np.array([collision]),
            np.array([reward_availability])
        ])

        self._internal_state = state

        # Use the current policy to select an action
        action = self.act(state)
        self.current_action = action

        return action

    def act(self, state):
        """Select an action using the current policy with noise for exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_tensor).squeeze().numpy()

        # Add exploration noise
        action = action + np.random.normal(0, self.exploration_noise, size=self.action_dim)

        # Clip to ensure action is within valid range
        action = np.clip(action, -self.max_action, self.max_action)

        return action

    def step(self, state, action, reward, next_state, done):
        """Add experience to memory and learn if it's time"""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def learn(self, experiences):
        """Update policy and value parameters using batch of experience tuples"""
        self.total_it += 1
        states, actions, rewards, next_states, dones = experiences

        # Select next actions according to the target policy with noise
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_actions = self.actor_target(next_states) + noise
            next_actions = next_actions.clamp(-self.max_action, self.max_action)

            # Compute target Q value
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q

        # Get current Q estimates
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # Compute critic loss
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic1_target, self.critic1)
            self.soft_update(self.critic2_target, self.critic2)

    def soft_update(self, target, source):
        """Soft update target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def get_representation(self):
        """Return a representation compatible with the custom model interface"""
        return self.representation

    def save_model(self, path):
        """Save actor and critic networks to the given path"""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'total_it': self.total_it
        }, path)

    def load_model(self, path):
        """Load actor and critic networks from the given path"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.total_it = checkpoint.get('total_it', self.total_it)

    def reset(self):
        return


class Actor(nn.Module):
    """Actor (Policy) Model for TD3"""

    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = max_action

    def forward(self, state):
        return self.network(state) * self.max_action


class Critic(nn.Module):
    """Critic (Value) Model for TD3"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

def make_rl_name(model: object):
    model_str = str(model)
    existing = [fname for fname in os.listdir(RLPATH) if model_str in fname]
    num = len(existing)
    return f"{RLPATH}/model_{model_str}_{num}"


""" RUN FUNCTIONS """


def run_rl_model(env: object,
                 num_episodes: int,
                 model_type: str,
                 load: bool=False,
                 idx: int=0,
                 hidden_dim=64,
                 learning_rate=0.001,
                 gamma=0.99,
                 buffer_size=10000,
                 batch_size=64,
                 update_every=4,
                 renderer=None,
                 plot_interval=10,
                 t_teleport=100,
                 pause=-1,
                 record_flag=False,
                 verbose=True,
                 verbose_min=True,
                 training_mode=True):

    """
    Run a standard RL model on the environment

    Args:
        env: Environment to run the model on
        model_type: Type of RL model to use ("DQN" or "TD3")
        hidden_dim: Hidden dimension size for neural networks
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        buffer_size: Replay buffer size
        batch_size: Batch size for learning
        update_every: How often to update the networks
        renderer: Function to render additional information
        plot_interval: How often to render
        t_teleport: How often to teleport the agent
        pause: Time to pause between steps
        record_flag: Whether to record trajectory and activity
        verbose: Whether to print detailed information
        verbose_min: Whether to print minimal information
        training_mode: Whether to train the model or just evaluate

    Returns:
        Dictionary containing recorded data
    """
    # ===| setup |===
    clock = pygame.time.Clock()
    last_position = np.zeros(2)

    # Determine state and action dimensions
    # State: [velocity (2), prev_velocity (1), collision (1), reward_availability (1)]
    state_dim = 5
    action_dim = 2  # [x_velocity, y_velocity]

    # Create the RL model based on model_type
    if model_type == "DQN":
        # For DQN, we'll discretize the action space
        # Create a discrete set of actions (velocities)
        action_space = []
        for vx in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for vy in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                action_space.append([vx, vy])

        model = DQNAgent(state_dim=state_dim,
                         action_dim=len(action_space),
                         action_space=action_space,
                         lr=learning_rate,
                         gamma=gamma,
                         buffer_size=buffer_size,
                         batch_size=batch_size,
                         update_every=update_every)
        logger("%using DQN")

        if load:
            name = f"model_DQN_{idx}"
            model.load_model(f"{RLPATH}/{name}")
            logger(f"loaded DQN-{idx}")

    elif model_type == "TD3":
        model = TD3Agent(state_dim=state_dim,
                        action_dim=action_dim,
                        max_action=1.0,
                        lr=learning_rate,
                        gamma=gamma,
                        buffer_size=buffer_size,
                        batch_size=batch_size)
        logger("%using TD3")

        if load:
            name = f"model_TD3_{idx}"
            model.load_model(f"{RLPATH}/{name}")
            logger(f"loaded TD3-{idx}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # [position, velocity, collision, reward, done, terminated]
    observation = [[0., 0.], 0., 0., False, False]
    prev_position = env.position

    record = {"activity": [],
              "trajectory": []}

    # ===| main loop |===

    for epoch in tqdm(range(num_episodes), desc=f"Game ({model_type})",
                      disable=True):

        env.reset()

        for _ in range(env.duration):

            # Event handling
            if env.visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

            # -check: teleport
            if env.t % t_teleport == 0 and env.reward_obj.is_silent:
                env._reset_agent_position(model, True)

            # velocity
            v = [(env.position[0] - prev_position[0]),
                 (-env.position[1] + prev_position[1])]

            # model step
            try:
                velocity = model(v,
                               observation[1],
                               observation[2],
                               env.reward_availability)
            except IndexError:
                logger.debug(f"IndexError: {len(observation)}")
                raise IndexError

            # For RL learning, we need to store the current state
            current_state = np.concatenate([
                np.array(v),
                np.array([observation[1]]),
                np.array([observation[2]]),
                np.array([env.reward_availability])
            ])

            # store past position
            prev_position = env.position

            # env step
            observation = env(velocity=np.array([velocity[0], -velocity[1]]),
                              brain=model)

            # For RL learning, construct the next state
            next_v = [(env.position[0] - prev_position[0]),
                     (-env.position[1] + prev_position[1])]

            next_state = np.concatenate([
                np.array(next_v),
                np.array([observation[1]]),
                np.array([observation[2]]),
                np.array([env.reward_availability])
            ])

            # Extract reward and done from observation
            reward = float(observation[3]) if observation[3] is not False else 0.0
            done = observation[4]

            # Learn from experience if in training mode
            if training_mode:
                model.step(current_state, velocity, reward, next_state, done)

            # -check: reset agent's brain
            if observation[3]:
                if verbose and verbose_min:
                    logger(f"epoch {epoch+1}|{num_episodes} - score={env.rw_count}")
                break

            # -check: render
            if env.visualize:
                if env.t % plot_interval == 0:
                    env.render()
                    if renderer:
                        renderer()

            # -check: record
            if record_flag:
                record["activity"] += [model.get_representation()]
                record["trajectory"] += [env.position]

            # -check: exit
            if observation[4]:
                if verbose and verbose_min:
                    logger.debug(">> Game terminated <<")
                break

            # pause
            if pause > 0:
                pygame.time.wait(pause)

    pygame.quit()

    return record, model


def main_rl(global_parameters: dict,
            reward_settings: dict,
            game_settings: dict,
            room_name: str,
            num_episodes: int,
            save: bool=False,
            load: bool=False,
            pause: int=-1,
            verbose: bool=True,
            record_flag: bool=False,
            limit_position_len: int=-1,
            preferred_positions: list=None,
            verbose_min: bool=True) -> dict:

    env, reward_obj = sim.setup_env(global_parameters=global_parameters,
                                    reward_settings=reward_settings,
                                    game_settings=game_settings,
                                    room_name=room_name,
                                    pause=pause, verbose=verbose,
                                    record_flag=record_flag,
                                    limit_position_len=limit_position_len,
                                    preferred_positions=preferred_positions,
                                    verbose_min=verbose_min)

    record, model = run_rl_model_2(env=env,
                                 load=load,
                                 num_episodes=num_episodes,
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
                                 training_mode=rl_parameters['training_mode'])

    # record, model = run_rl_model(env=env,
    #                              load=load,
    #                              num_episodes=num_episodes,
    #                              model_type=rl_parameters['model_type'],
    #                              hidden_dim=rl_parameters['hidden_dim'],
    #                              learning_rate=rl_parameters['learning_rate'],
    #                              gamma=rl_parameters['gamma'],
    #                              buffer_size=rl_parameters['buffer_size'],
    #                              batch_size=rl_parameters['batch_size'],
    #                              update_every=rl_parameters['update_every'],
    #                              renderer=None,
    #                              plot_interval=game_settings['plot_interval'],
    #                              t_teleport=game_settings['t_teleport'],
    #                              pause=game_settings['pause'],
    #                              record_flag=False,
    #                              verbose=game_settings['verbose'],
    #                              verbose_min=verbose_min,
    #                              training_mode=rl_parameters['training_mode'])

    logger(f"rw_count={env.rw_count}")

    # if save:
    #     model.save_model(path=make_rl_name(model))

    return record




def run_rl_model_2(env: gym.Env,
                 num_episodes: int,
                 model_type: str,
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
                 training_mode=True):

    env = games.EnvironmentWrapper(env)

    if not isinstance(env, gym.Env):
        raise ValueError("Your environment must inherit from gym.Env")

    vec_env = DummyVecEnv([lambda: env])

    model_classes = {
        "DQN": DQN,
        "TD3": TD3,
        "PPO": PPO
    }

    model_class = model_classes.get(model_type)
    if model_class is None:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_path = f"{RLPATH}/model_{model_type}_{idx}"

    # Define model-specific kwargs
    model_kwargs = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "verbose": 1 if verbose else 0,
        "device": "cuda",  # ✅ Use GPU
    }

    # Add only relevant arguments based on the algorithm
    if model_type in ["DQN", "TD3"]:
        model_kwargs.update({
            "buffer_size": buffer_size,
            "batch_size": batch_size,
        })

    if load and os.path.exists(model_path + ".zip"):
        model = model_class.load(model_path, env=vec_env, device="cuda",
                                 verbose=0)
        logger(f"Loaded model from {model_path}")
    else:
        model = model_class("MlpPolicy", vec_env, **model_kwargs)

    if training_mode:
        model.learn(total_timesteps=num_episodes,
                    progress_bar=True)
        if record_flag:
            model.save(model_path)
            logger(f"Saved model to {model_path}")

    # Evaluation run
    obs = vec_env.reset()
    record = {"activity": [], "trajectory": []}
    for _ in range(env.duration):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        if record_flag:
            record["trajectory"].append(env.unwrapped.position)

        if done:
            break

        # if env.visualize and env.t % plot_interval == 0:
        #     env.render()

    return record, model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=-1)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--rendering", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--interval", type=int, default=20,
                        help="plotting interval")
    parser.add_argument("--room", type=str, default="Square.v0",
                        help='room name: ["Square.v0", "Square.v1",' + \
                         ' "Square.v2","Hole.v0", "Flat.0000", "Flat.0001",' + \
                         '"Flat.0010", "Flat.0011", "Flat.0110", ' + \
                         '"Flat.1000", "Flat.1001", "Flat.1010",' + \
                         '"Flat.1011", "Flat.1110"] or `random`')
    args = parser.parse_args()

    # main()
    main_rl(global_parameters=global_parameters,
            reward_settings=reward_settings,
            game_settings=game_settings,
            room_name=args.room,
            num_episodes=args.episodes,
            save=args.save,
            load=args.load,
            pause=-1,
            verbose=False,
            record_flag=False,
            limit_position_len=2,
            preferred_positions=None,
            verbose_min=True)



