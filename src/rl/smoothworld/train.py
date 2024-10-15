from gym import spaces
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback

import sys, os
sys.path.append(os.path.expanduser('~/Research/lab/PCNN/src/rl/smoothworld'))
import envs as se
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import argparse
from time import time, sleep
import json
from datetime import datetime

if os.path.exists(os.path.expanduser('~/Research/lab/PCNN')):
    sys.path.append(os.path.expanduser('~/Research/lab/PCNN'))
elif os.path.exists(os.path.expanduser('~/lab/PCNN')):
    sys.path.append(os.path.expanduser('~/lab/PCNN'))
else:
    raise ValueError("PCNN path not found.")
import src.utils as utils
import src.visualizations as vis

logger = utils.logger


if __name__ == "__main__":


    """ args """

    parser = argparse.ArgumentParser()
    parser.add_argument("--EPOCHS", type=int, default=10_000,
                        help="Number of epochs to train the agent.")
    parser.add_argument("--T_TIMEOUT", type=int, default=2000,
                        help="Timeout for the episode.")
    parser.add_argument("--GOAL_POS",
                        type=str, default="0.8,0.8")
    parser.add_argument("--GOAL_RADIUS",
                        type=float, default=0.1)
    parser.add_argument("--INIT_POS",
                        type=str, default="0.3,0.3")
    parser.add_argument("--INIT_POS_RADIUS",
                        type=float, default=0.1)
    parser.add_argument("--MAX_WALL_HITS", type=int, default=2)
    parser.add_argument("--log_int", type=int, default=2)
    parser.add_argument("--save_freq", type=int, default=100_000)
    parser.add_argument("--flat", type=str, default="one")
    parser.add_argument("--save", action="store_true",
                        default=False)
    parser.add_argument("--choose", action="store_true",
                        default=False)
    parser.add_argument("--load", action="store_true",
                        default=False)
    parser.add_argument("--test", action="store_true",
                        default=False)
    args = parser.parse_args()

    EPOCHS = args.EPOCHS
    T_TIMEOUT = args.T_TIMEOUT
    GOAL_POS = np.array([float(x) for x in args.GOAL_POS.split(",")])
    GOAL_RADIUS = args.GOAL_RADIUS
    INIT_POS = np.array([float(x) for x in args.INIT_POS.split(",")])
    INIT_POS_RADIUS = args.INIT_POS_RADIUS
    MAX_WALL_HITS = args.MAX_WALL_HITS


    logger(f"%EPOCHS: {EPOCHS}")
    logger(f"%T_TIMEOUT: {T_TIMEOUT}")
    logger(f"%save: {args.save}")


    """ settings """

    # model
    N_PCNN = 100
    Nj = 25**2

    # pcnn params
    # pcnn_params = {
    #                 "N": 100,
    #                 "Nj": 20**2,
    #                 "alpha": 0.1, # 0.1
    #                 "beta": 20.0, # 20.0
    #                 "lr": 0.0,
    #                 "threshold": 0.4,
    #                 "da_threshold": 0.2,
    #                 "tau_da": 50.,
    #                 "eq_da": 1.

    pcnn_params = {
        "N": N_PCNN,
        "Nj": Nj,
        "tau": 10.0,
        "alpha": 0.2,
        "beta": 20.0,
        "lr": 0.2,
        "threshold": 0.02,
        "ach_threshold": 0.3,
        "da_threshold": 0.5,
        "tau_ach": 200.,  # 2.
        "eq_ach": 1.,
        "tau_da": 200.,  # 2.
        "eq_da": 0.,
        "epsilon": 0.4,
    }
    

    # environment
    env_params = {
        "flat": args.flat,
        "GOAL_POS": GOAL_POS,
        "GOAL_RADIUS": GOAL_RADIUS,
        "nb_experiences": 1,
        "experience_duration": T_TIMEOUT,
        "max_experiences": 3,
        "init_pos": INIT_POS,
        "init_pos_radius": INIT_POS_RADIUS,
        "max_wall_hits": MAX_WALL_HITS,
        "max_speed": 0.07,
        "pcnn_params": pcnn_params,
        "return_info": True,
        "cell_types": ('PCNN', 'PC', 'GC'), # include `PCNN` for PCNN
    }

    IS_PCNN_flag = 'PCNN' in env_params["cell_types"]

    logger(f"%{IS_PCNN_flag=}")

    """ environment """

    env, agent, cell_info = se.generate_navigation_task_env(**env_params)

    GOAL_POS, GOAL_RADIUS = env.GOAL_POS, env.GOAL_RADIUS

    """ agent """

    stamp = datetime.fromtimestamp(time()).strftime(
        "%Y_%m_%d-%H%M%S")

    if args.save:
        save_path = f"./models/ppo/agent_{stamp}/"

        # create directory
        os.mkdir(save_path)

        # save model
        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=save_path,
            name_prefix="iter",
        )

        # save env
        env_params["env"] = env.__repr__()
        env_params["GOAL_POS"] = list(env_params["GOAL_POS"])
        env_params["init_pos"] = list(env_params["init_pos"])
        env_params["cell_info"] = cell_info
        with open(save_path + "info.json", "w") as f:
            json.dump(env_params, f)

    else:
        checkpoint_callback = None

    if args.load:
        model = se.load_model(env=env,
                              last=True,
                              choose=args.choose)
        logger("%PPO agent loaded.")
    else:
        # model = A2C(
        #     "MlpPolicy",
        #     env,
        #     verbose=1,
        # )

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
        )

    logger("%PPO agent created.")


    """ training """

    if not args.load:

        logger("%Training PPO agent...")

        model.learn(total_timesteps=EPOCHS,
                    log_interval=args.log_int,
                    tb_log_name=f"{stamp}",
                    callback=checkpoint_callback,
                    progress_bar=True)

        logger("%PPO agent trained.")

    if args.test:

        logger("%Testing PPO agent...")

        obs, _ = env.reset(kind="hard")
        cell = env._cells[0]

        if IS_PCNN_flag:
            pcnn = cell._pcnn._pcnn
            pcnn._knn = 7
            pcnn._max_dist = 0.5
            cell.flag_make_pf()

        agent = env._env.agents_dict['agent_0']
        rewards =  0

        t_start = env.t
        positions = []
        actions = []
        nb_collisions = 0

        is_plotting = True

        # if IS_PCNN_flag:
        #     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        # else:
        #     fig, ax1 = plt.subplots(nrows=1, ncols=1,
        #                             figsize=(5, 8))

        nb = 0
        env.reset(kind="hard")
        logger(f"%{env}")

        if IS_PCNN_flag:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                           figsize=(8, 5))
        else:
            fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                    figsize=(5, 8))

        # do an infinite amount of episodes
        while True:
            logger(f"plotting {nb=}")
            sleep(1)
            env.reset(kind="hard")
            nb += 1

            ach = []
            DA = []
            for tc, t in enumerate(np.arange(0, T_TIMEOUT, env._env.dt)):

                action, _state = model.predict(obs, deterministic=True)
                obs, reward_rate, done, truncated , info =  env.step(
                    action=action)

                positions += [env._env.agents_dict['agent_0'].pos.tolist()]
                posr = np.array(positions)
                actions += [action.tolist()]
                nb_collisions += 1*(agent.is_wall_hit)
                if len(ach) > 200:
                    del ach[0]

                if IS_PCNN_flag:
                    DA += [env._cells[0]._pcnn._pcnn._DA]
                    ach += [pcnn._ACh]

                # exit
                if done or reward_rate:
                    logger.debug(f"{done=} {reward_rate=}")
                    break

                # display the trajectory
                if tc % 10 == 0:

                    ax1.clear()

                    env._env.plot_environment(autosave=False, fig=fig, ax=ax1)
                    fig, ax1 = se.display_reward_patch(fig, ax1,
                                                      reward_pos=env.GOAL_POS,
                                                      reward_radius=env.GOAL_RADIUS)

                    # graph
                    if IS_PCNN_flag:
                        env._cells[0]._pcnn.plot_graph(ax=ax1)

                    # positions
                    ax1.plot(posr[-min(tc, 100):, 0], posr[-min(tc, 100):, 1], 'b-')
                    ax1.set_title(f"$a=${np.around(action, 2)} | {t=:.1f}s " + \
                                  f"| collision={nb_collisions}")
                    ax1.set_aspect("equal")

                    #
                    if IS_PCNN_flag:
                        ax2.clear()
                        ax2.imshow(pcnn._Wff,
                                   vmin=0,
                                   aspect="equal",
                                   interpolation="nearest")

                        ax2.set_title("$E_{DA}=$" + f"{pcnn._eq_da}, \n" + \
                            f"DA={pcnn._DA:.3f}, " + \
                            f"$N_{{PC}}=${pcnn._umask.sum()} " + \
                            f"$u=${pcnn.u.max():.3f}mV") # + \

                    plt.pause(0.1)

        logger("<done>")
