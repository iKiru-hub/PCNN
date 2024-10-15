# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.15.2
# ---

# %%

from gym import spaces
import gymnasium as gym
from stable_baselines3 import A2C, PPO

import sys, os
sys.path.append(os.path.expanduser('~/Research/lab/PCNN/src/rl/smoothworld'))
import envs as se
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.expanduser('~/Research/lab/PCNN'))
import src.utils as utils
import src.visualizations as vis

logger = utils.logger


# %%

""" train PPO agent """

T_TIMEOUT = 1_000
EPOCHS = 100_000

GOAL_POS = np.array([0.2, 0.2])
# GOAL_POS = np.array([[0.7, 0.7],
#                      [0.2, 0.7]])

GOAL_RADIUS = 0.05

INIT_POS = np.array([0.3, 0.7])
INIT_POS_RADIUS = 0.075
# INIT_POS = None
# INIT_POS_RADIUS = None

MAX_WALL_HITS = 10_000

# pcnn params
pcnn_params = {
                "N": 100,
                "Nj": 20**2,
                "alpha": 0.10, # 0.1
                "beta": 20.0, # 20.0
                "lr": 0.00,
                "threshold": 0.4,
                "da_threshold": 0.2,
                "tau_da": 75,
                "eq_da": 1.
}

N_PCNN = pcnn_params["N"]

logger("<settings>")


# Optionally check the environment (useful during
# development
env_params = {
    "IS_PCNN_flag": False,
    "flat": "one",
    "GOAL_POS": GOAL_POS,
    "GOAL_RADIUS": GOAL_RADIUS,
    "REWARD_DURATION": 10,
    "nb_experiences": 2,
    "experience_duration": T_TIMEOUT,
}
cell_types = ["PCNN", "FOV"]
env, agent = se.generate_navigation_task_env(IS_PCNN_flag=True,
                                             flat="two",
                                             GOAL_POS=GOAL_POS,
                                             GOAL_RADIUS=GOAL_RADIUS,
                                             nb_experiences=2,
                                             experience_duration=T_TIMEOUT,
                                             max_experiences=1,
                                             init_pos=INIT_POS,
                                             init_pos_radius=INIT_POS_RADIUS,
                                             max_wall_hits=MAX_WALL_HITS,
                                             pcnn_params=pcnn_params,
                                             cell_types=cell_types)

logger(f"%{env=}")

GOAL_POS, GOAL_RADIUS = env.GOAL_POS, env.GOAL_RADIUS

fig, ax = plt.subplots()
env._env.plot_environment(autosave=False, fig=fig, ax=ax)

if len(GOAL_POS.shape) > 1:
    for goal_pos in GOAL_POS:
        fig, ax = se.display_reward_patch(fig, ax, reward_pos=goal_pos,
                                          reward_radius=GOAL_RADIUS)
else:
    fig, ax = se.display_reward_patch(fig, ax, reward_pos=GOAL_POS,
                                      reward_radius=GOAL_RADIUS)

plt.show()


# %%
# Train RL
# model = A2C("MlpPolicy", env, verbose=1)
model = PPO("MlpPolicy", env, verbose=1)

logger(model)


# %%
# TRAINING
model = model.learn(total_timesteps=EPOCHS,
            log_interval=1_000,
            tb_log_name="ppo_smoothworld",
            progress_bar=True)

logger("trained")

# %%
env._cells[0].__repr__().split(".")[-1].split(" ")[0]

envrec = {
    "cells": [cell.__repr__().split(".")[-1].split(" ")[0] for cell in env._cells],

}

# %%
#### test : run the model over multiple trajectories
logger.info("test")

# figure
if is_plotting:
    plt.show()


# %%
cells = env._cells
dir(cells[0])


# %%
cells[0].plot_place_cell_locations()
plt.show()



# %% [markdown]
# -------------------------
# STUDY OF THE PCNN LAYER
# %%
pcnn = env._cells[0]._pcnn._pcnn

# %%
place_cell_centres = env._cells[0]._pcnn.make_pc_centers(
    trajectory=env.whole_track,
    bounds=tuple(env._env.extent),
    knn=5,
    max_dist=0.13
)

place_cell_connections = env._cells[0]._pcnn.connections.copy()

place_cell_centres


# %%
# plot graph

fig, ax = plt.subplots()
env._env.plot_environment(autosave=False, fig=fig, ax=ax)

# ax.scatter(place_cell_centres[:, 0], place_cell_centres[:, 1])
vis.plot_graph(centers=place_cell_centres,
               connectivity=place_cell_connections,
               bounds=env._env.extent,
               alpha=0.5, ax=ax)
plt.show()


# %%
plt.imshow(pcnn._Wff)
plt.show()



# %%
pcnn = env._cells[0]._pcnn._pcnn

plt.plot(pcnn.tau_record)
plt.show()


# %
pcnn = env._cells[0]


# %%
X = np.random.uniform(0, 1, size=(10, 2))
X


# %%
pcnn = env._cells[0]
out = pcnn.get_state(evaluate_at="all").reshape(100, -1)

z = out.sum(axis=0)
nb_tests = 4
fig, axs = plt.subplots(nrows=2, ncols=nb_tests//2,
                        figsize=(3*nb_tests//2, 5))
axs = axs.flatten()

logger.debug(f"{GOAL_POS=}")

for i in range(nb_tests):

    logger.info(f"run {i}")

    obs, _ = env.reset()
    rewards = 0

    t_start = env.t
    positions = []

    # for t in tqdm(np.arange(t_start, 100+t_start, env._env.dt)):
    for t in tqdm(np.arange(0, T_TIMEOUT+t_start, env._env.dt)):

        try:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward_rate, done, _ , info =  env.step(
                action=action)
        except AttributeError:
            break

        positions += [env._env.agents_dict['agent_0'].pos.tolist()]


        # if t < 2:
        #     print(positions)

        # exit
        # if done or reward_rate:
        #     logger.debug(f"{done=} {reward_rate=} {_=}")
        #     break

    # display the trajectory
    env._env.plot_environment(autosave=False, fig=fig, ax=axs[i])
    fig, xax = se.display_reward_patch(fig, axs[i],
                                      reward_pos=env.GOAL_POS,
                                      reward_radius=env.GOAL_RADIUS)
    # axs[i] = xax

    positions = np.array(positions)

    axs[i].plot(positions[:, 0], positions[:, 1], '-')
    axs[i].set_xlabel(f"pos={positions[-1, :]}")

    logger.debug(f"{positions=}")


plt.show()

# %%
#### test 2 : plot all trajectories experienced by the agent

fig, ax = plt.subplots()
fig, ax = agent.plot_trajectory(fig=fig, ax=ax, t_start=env.t-1000,
                                t_end=env.t,
                                framerate=30,
                                color="changing")
plt.show()


# %%
cell_names = [
    cell.__repr__().split(".")[-1].split(" ")[0] for cell in env._cells
]
cell_names
# %%
#### test 3 : online

obs, _ = env.reset()
cell = env._cells[0]
pcnn = env._cells[0]._pcnn._pcnn
agent = env._env.agents_dict['agent_0']
rewards =  0

pcnn._knn = 15
pcnn._max_dist = 0.3

t_start = env.t
positions = []
actions = []
DA = []
nb_collisions = 0

is_plotting = False

cell.flag_make_pf()

for tc, t in enumerate(np.arange(0, T_TIMEOUT+t_start, env._env.dt)):

    action, _state = model.predict(obs, deterministic=True)
    obs, reward_rate, done, truncated , info =  env.step(
        action=action)

    positions += [env._env.agents_dict['agent_0'].pos.tolist()]
    posr = np.array(positions)
    actions += [action.tolist()]
    DA += [env._cells[0]._pcnn._pcnn._DA]
    nb_collisions += 1*(agent.is_wall_hit)

    # exit
    # if reward_rate:
    #     logger.debug(f"{done=} {reward_rate=}")
    #     break

    if tc > 10 and not is_plotting:
        is_plotting = True
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                       figsize=(15, 5))

    # display the trajectory
    if tc % 200 == 0 and is_plotting:

        ax1.clear()
        ax2.clear()

        tpl = 1000
        env._env.plot_environment(autosave=False, fig=fig, ax=ax1)
        fig, ax1 = se.display_reward_patch(fig, ax1,
                                          reward_pos=env.GOAL_POS,
                                          reward_radius=env.GOAL_RADIUS)
        ax1.plot(posr[-min(tc,tpl):, 0], posr[-min(tc,tpl):, 1], 'b-')
        # print(f"position={tuple(np.around(positions[-1], 2))}")
        ax1.set_title(f"$a=${np.around(action, 2)} | {t=:.1f}s " + \
                      f"| collision={nb_collisions}")

        if env._cells[0]._is_making_pf:
            ax2.imshow(cell._place_field.reshape(100, 100),
                       vmin=0.03, vmax=0.5, cmap="Greens")
            vis.plot_graph(centers=cell._pcnn_centers,
                           connectivity=cell._pcnn_connections,
                           bounds=env._env.extent,
                           alpha=0.5, ax=ax1)
        else:
            ax2.imshow(pcnn._Wff)
            # ax2.imshow(env._cells[0]._pcnn._pcnn.u.reshape(10, 10),
            #            vmin=0, vmax=1, cmap="Greys")

        ax2.set_title("$E_{DA}=$" + f"{pcnn._eq_da}, \n" + \
            f"DA={pcnn._DA:.3f}, " + \
            f"$\langle\Delta W\\rangle$={pcnn._dw_rec:.3f}\n" + \
            "last $t_{dt}=$" + f"{cell._last_makepf}, tot={pcnn._umask.sum()}")

        ax2.axis("off")

        plt.pause(0.0001)

if is_plotting:
    plt.show()


# %%
#### test 4 : online w/ pcnn
obs, _ = env.reset()
agent = env._env.agents_dict['agent_0']
rewards =  0

t_start = env.t
positions = []
actions = []
nb_collisions = 0

is_plotting = False

for tc, t in enumerate(np.arange(0, T_TIMEOUT+t_start, env._env.dt)):

    action, _state = model.predict(obs, deterministic=True)
    obs, reward_rate, done, truncated , info =  env.step(
        action=action)

    positions += [env._env.agents_dict['agent_0'].pos.tolist()]
    posr = np.array(positions)
    actions += [action.tolist()]
    nb_collisions += 1*(agent.is_wall_hit)

    # exit
    if done or reward_rate:
        logger.debug(f"{done=} {reward_rate=}")
        # break

    if tc > 0 and not is_plotting:
        is_plotting = True
        fig, ax1 = plt.subplots(nrows=1, ncols=1)

    # display the trajectory
    if tc % 2 == 0 and is_plotting:

        env._env.plot_environment(autosave=False, fig=fig, ax=ax1)
        fig, ax1 = se.display_reward_patch(fig, ax1,
                                          reward_pos=env.GOAL_POS,
                                          reward_radius=env.GOAL_RADIUS)
        ax1.plot(posr[-min(tc,20):, 0], posr[-min(tc,20):, 1], 'b-')
        # print(f"position={tuple(np.around(positions[-1], 2))}")
        ax1.set_title(f"$a=${np.around(action, 2)} | {t=:.1f}s " + \
                      f"| collision={nb_collisions}")
        ax1.axis("off")

        plt.pause(0.2)

# z -= z.mean(axis=0)
# z = 1/(1 + np.exp(-20*(z-0.1)))

print(z.max(),z.mean())
plt.imshow(z.reshape(100, 100), vmin=0, vmax=1)
plt.axis("off")
plt.show()

# %%



fig, axs = plt.subplots(nrows=10, ncols=10)

for i, ax, in enumerate(axs.flatten()):
    ax.imshow(out[i, :].reshape(100, 100))
    ax.axis("off")

plt.show()
