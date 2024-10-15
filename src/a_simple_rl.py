import numpy as np
import matplotlib.pyplot as plt
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy
import time

from stable_baselines3 import PPO, A2C, TD3

import utils as u
from utils import logger
import policy as p

from tools.utils import clf, tqdm_enumerate, save_image, AnimationMaker


SAVEPATH = "cache/campoverde/"


foo =time.strftime('%H%M%S')


class CampoVerdeEnv(gym.Env):

    def __init__(self, model: object, **kwargs):
        super(CampoVerdeEnv, self).__init__()

        """
        Campo Verde environment

        Parameters
        ----------
        model: object
            PCNN model
        obs_size: int
            number of the top active neurons to observe
        bounds: tuple
            environment boundaries
        min_obs_activity: float
            minimum activity to consider a neuron as active
        isreward: bool
            if True, the reward is provided (i.e. possibly nonzero)
        objective: str
            the objective of the agent, one of ["target", "explore"].
            Default: "target"
        max_steps: int
            maximum number of steps per episode.
            Default: 200
        """

        self.model = model
        self.model_copy = deepcopy(model)

        # environment variables
        self.obs_size = kwargs.get("obs_size", 4)  # half observation
        self.bounds = kwargs.get("bounds", (0, 1, 0, 1))
        self.min_obs_activity = kwargs.get("min_obs_activity", 0.001)
        self.is_policy_first = kwargs.get("is_policy_first", False)

        # episode variables
        self.curr_pos = None
        self.src_pos = None
        self.trg_pos = None
        self.trg_pos_virtual = None
        self.trg_pos_rewarded = None
        self.t = 0
        self.max_steps = kwargs.get("max_steps", 200)

        # reward
        self.isreward = kwargs.get("isreward", False)
        self.objective = kwargs.get("objective", "target")
        self._wall_hits_count = 0
        self._max_wall_hits = kwargs.get("max_wall_hits", 10)
        if not self.isreward:
            self.objective = "free"
        self.nb_pc_start = self.model.__len__()
        self.pc_area_start = estimate_area_grid(points=self.model.centers,
                                                box_size=(self.bounds[1]-self.bounds[0],
                                                        self.bounds[3]-self.bounds[2]),
                                                grid_resolution=100)

        self.record = {
            "est_trg_pos": [],
            "path": [],
            "curr_a": np.zeros(self.model.N),
            "trg_a": np.zeros(self.model.N),
            "obs": np.zeros(self.obs_size),
            "distance": np.inf,
            "reward": []
        }

        # --- continous action space
        # . lambda: weight for the movement
        # . beta: variable strategy parameter
        # . new_local_target: generate a new local target
        # . speed: modulate between speed_base and 10*speed_base
        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0, shape=(4,),
                                       dtype=np.float32)
        self.action_size = self.action_space.shape[0]

        # --- continuous observation space (obs_size)
        self.observation_space = spaces.Box(low=-1.0,
                                            high=1.0,
                                            shape=(self.obs_size*2,),
                                            dtype=np.float32)

    def _one_step(self, action: np.ndarray, curr_pos: np.ndarray,
                  trg_pos: np.ndarray) -> tuple:

        # --- init
        curr_a = self.model(x=curr_pos)

        # estimate the target position from its representation
        # in neural space
        trg_a = self.model(x=trg_pos)
        trg_pos = self.model._calc_average_pos(a=trg_a)

        self.record["est_trg_pos"] += [self.model._calc_average_pos(a=trg_a)]

        # start
        if action is None:

            # make observations
            curr_a_out = np.sort(curr_a.flatten())[-self.obs_size:]
            trg_a_out = np.sort(trg_a.flatten())[-self.obs_size:]
            return curr_pos, curr_a_out, trg_a_out, None

        # update policy
        self.model.policy.set_parameters(action=action)

        # --- drift
        drift_a = p.forward_pc(W=self.model.W_rec,
                               x=curr_a.reshape(-1, 1),
                               alpha=self.model._alpha,
                               beta=self.model._beta, maxval=1.)
        drift_a = self.model.policy.apply_strategy(a=drift_a,
                                                mask=self.model.finite_idx)
        drift_pos = self.model._calc_average_pos(a=drift_a)
        drift_move, _ = p.calc_movement(pos1=curr_pos,
                                        pos2=drift_pos,
                                        speed=self.model.policy.speed)

        # --- direct
        direct_move, distance = p.calc_movement(pos1=curr_pos,
                                                pos2=trg_pos,
                                                speed=self.model.policy.speed)

        # --- update position | ---
        policy_move = self.model.policy(moves=[direct_move, drift_move],
                                 distance=distance)

        # --- THE ACTION COMES DIRECTLY FORM THE ACTION
        agent_move = action[:2] * self.model.policy.speed

        curr_pos += policy_move if self.is_policy_first else agent_move

        curr_pos = self._check_walls(pos=curr_pos)

        # logger.debug(f"action: {action} | move: {move} {curr_pos=}")

        # -- note
        # if it seems still, it might be beacause it is
        # oscillating between two positions

        # check boundaries
        # curr_pos = self.model.policy.check_wall(pos=curr_pos,
        #                                         bounds=self.model.bounds)

        # new representation of the position
        curr_a = self.model(x=curr_pos)

        # --- make observations | the top `obs_size` active neurons
        curr_a = (curr_a - curr_a.min()) / (curr_a.max() - curr_a.min())
        trg_a = (trg_a - trg_a.min()) / (trg_a.max() - trg_a.min())
        curr_a_out = np.sort(curr_a.flatten())[-self.obs_size:]
        trg_a_out = np.sort(trg_a.flatten())[-self.obs_size:]

        # --- record
        self.record["path"] += [curr_pos.tolist()]
        self.record["curr_a"] = curr_a
        self.record["trg_a"] = self.model(x=self.trg_pos_virtual)
        self.record["distance"] = distance

        return curr_pos, curr_a_out, trg_a_out, distance

    def _calc_reward(self, curr_pos: np.ndarray, trg_pos: np.ndarray) -> float:

        """
        calculate the reward
        """

        if self.objective == "target" and self.isreward:
            distance = np.linalg.norm(curr_pos - trg_pos)
            reward = 1.0 if distance < self.model.policy.min_trg_distance else 0.0

        # compare the number of new pc
        elif self.objective == "explore":

            # number of cells
            # reward = self.model.__len__() - self.nb_pc_start

            # area
            pc_area = estimate_area_grid(points=self.model.centers,
                                         box_size=(self.bounds[1]-self.bounds[0],
                                                   self.bounds[3]-self.bounds[2]),
                                         grid_resolution=100)
            reward = pc_area - self.pc_area_start

        else:
            reward = 0.0

        self.record["reward"] += [reward]

        return reward

    def _check_walls(self, pos: np.ndarray) -> np.ndarray:

        """
        check the boundaries, if the walls are hit, bounce back
        """

        if pos[0] < self.bounds[0]:
            pos[0] = self.bounds[0]
            self._wall_hits_count += 1
        elif pos[0] > self.bounds[1]:
            pos[0] = self.bounds[1]
            self._wall_hits_count += 1

        if pos[1] < self.bounds[2]:
            pos[1] = self.bounds[2]
            self._wall_hits_count += 1
        elif pos[1] > self.bounds[3]:
            pos[1] = self.bounds[3]
            self._wall_hits_count += 1

        return pos

    def step(self, action: np.ndarray) -> tuple:

        """
        agent step
        """

        if isinstance(action, tuple):
            action = action[0]

        # --- action pre-processing ---
        if action is not None:
            # pre-process the first action entry from
            # [-1, 1] to [0, 1]
            action[0] = (action[0] + 1) / 2

            # speed
            action[3] = (action[3] + 1) / 2

            # parse flag for the new local target | <<<<<<<<<<<<<
            # if action[2] > 0.:
            #     self.trg_pos, self.trg_pos_virtual = self.model.generate_local_position()

        # --- environment step ---
        self.curr_pos, obs1, obs2, distance = self._one_step(action=action,
                                                             curr_pos=self.curr_pos,
                                                             trg_pos=self.trg_pos)

        # reward
        reward = self._calc_reward(curr_pos=self.curr_pos,
                                   trg_pos=self.trg_pos_rewarded)

        # info
        info = {"distance": distance}
        obs = np.concatenate([obs1, obs2])
        self.record["obs"] = obs

        self.t += 1

        if self.t >= self.max_steps or self._wall_hits_count >= self._max_wall_hits:
            done = True
        elif reward == 1.0 and self.objective == "target":
            done = True
        else:
            done = False

        if np.isnan(obs).any():
            raise ValueError("NaN in the observation")

        truncated = False

        return obs, reward, done, truncated,  info

    def render(self, mode='human', save=False, **kwargs):

        """
        Render the environment (optional)
        """

        vtrg_pos = self.trg_pos_virtual

        trg_pos = self.trg_pos_virtual if not self.isreward else self.trg_pos_rewarded

        animaker = kwargs.get("animaker", None)
        isanim = animaker is not None

        plt.clf()

        if save or isanim:
            fig = plt.figure(figsize=(10, 5))

        # env
        plt.subplot(1, 2, 1)

        # self.model.render(use_a=False, alpha=0.1)
        self.model.render(use_a=True, alpha=0.9)
                          # new_a=self.model.u.flatten().tolist())

        plt.scatter(self.src_pos[0], self.src_pos[1],
                    color='red', marker="o", s=100)
        # plt.scatter(self.trg_pos[0], self.trg_pos[1],
        #             color='blue', marker="x", s=300)
        plt.scatter(vtrg_pos[0],
                    vtrg_pos[1],
                    color='red', marker="x", s=300,
                    label="virtual target")
        plt.scatter(trg_pos[0],
                    trg_pos[1],
                    color='blue', marker="x", s=300,
                    label="true target")

        plt.plot(np.array(self.record["path"])[:, 0],
                 np.array(self.record["path"])[:, 1],
                    "g-", alpha=0.4, lw=2.)
        plt.scatter(np.array(self.record["path"])[-1, 0],
                   np.array(self.record["path"])[-1, 1],
                   color='green', marker="d", s=80,
                   alpha=0.5)

        plt.xlim(self.bounds[0], self.bounds[1])
        plt.ylim(self.bounds[2], self.bounds[3])
        plt.xticks([])
        plt.yticks([])
        plt.legend()
        plt.title(f"t={self.t/10:.1f}s\npos={np.around(self.curr_pos, 2)}" + \
            f"\ntrg={np.around(trg_pos, 2)}")

        # activity
        plt.subplot(2, 2, 2)
        plt.imshow(self.record["curr_a"].reshape(1, self.model.N),
                   cmap="Greys", aspect="auto", interpolation="nearest",
                   vmin=0)
        # plt.title(f"v={self.model.policy.vector[0]:.1f}° - " + \
        #     f"[goal={self.objective}]")
        plt.title("Current representation")
        plt.xticks([])
        plt.yticks([])


        plt.subplot(2, 2, 4)

        plt.imshow(self.record["trg_a"].reshape(1, -1),
                   cmap="Greys", aspect="auto", interpolation="nearest",
                   vmin=0)
        # plt.title(f"Target representation [ACh={self.model._ACh:.2f}]")
        plt.title(f"Target representation")
        # plt.imshow(self.record["obs"].reshape(1, -1),
        #            cmap="Greys", aspect="auto", interpolation="nearest",
        #            vmin=0)
        # plt.title(f"observation [ACh={self.model._ACh:.2f}]")
        plt.xticks([])
        plt.yticks([])

        if not isanim:
            plt.pause(0.001)
        else:
            animaker.add_frame(fig)

        if save:
            if str(input("are u sure?")) in ("yes", "y"):
                fig.savefig("/Users/daniekru/Desktop/episode.svg",
                            format="svg", dpi=300)

    def reset(self, episde_meta_info: str=None, seed: int=None, **kwargs) -> tuple:

        if seed is not None:
            super().reset(seed=seed)

        # self.src_pos = np.around(np.random.uniform(
        #     self.bounds[0], self.bounds[1], 2), 2)

        # use centroid
        # self.src_pos = np.around(
        #     self.model._calc_centroid(), 2)
        # self.src_pos, _ = self.model.generate_local_position()
        self.src_pos = np.array([0.2, 0.2])

        self.curr_pos = self.src_pos.copy()
        # self.trg_pos = np.around(np.random.uniform(
        #     self.bounds[0], self.bounds[1], 2), 2)
        self.trg_pos = np.array([0.8, 0.75])
        self.trg_pos, self.trg_pos_virtual = self.model.generate_local_position()

        # effective position
        a = self.model(x=self.trg_pos)
        self.trg_pos_virtual =  self.model._calc_average_pos(a=a)

        if self.isreward:
            self.trg_pos_rewarded = self.trg_pos_virtual.copy()

        self.t = 0

        # new model
        self.model = deepcopy(self.model_copy)

        self.record = {
            "est_trg_pos": [],
            "path": [],
            "curr_a": np.zeros(self.model.N),
            "trg_a": np.zeros(self.model.N),
            "obs": np.zeros(self.obs_size),
            "distance": np.inf,
            "reward": []
        }

        obs, _, _, _, _ = self.step(action=None)
        return obs, {}

    def close(self):
        # Cleanup resources (optional)
        pass



def estimate_area_grid(points: list, box_size: tuple,
                       grid_resolution: float) -> float:

    """
    Estimate the area covered by a set of points using a grid-based method.

    Parameters
    ----------
    points: list
        list of points
    box_size: tuple
        the size of the box
    grid_resolution: int
        number of cells in the grid

    Returns
    -------
    estimated_area: float
        estimated area
    """

    width, height = box_size

    # Create a 2D grid
    grid = np.zeros((grid_resolution, grid_resolution), dtype=bool)

    # Calculate cell sizes
    cell_width = width / grid_resolution
    cell_height = height / grid_resolution

    # Mark cells containing points
    for x, y in points:
        if 0 <= x <= width and 0 <= y <= height:  # Ensure point is within the box
            i = min(int(y / cell_height), grid_resolution - 1)
            j = min(int(x / cell_width), grid_resolution - 1)
            grid[i, j] = True

    # Count occupied cells
    occupied_cells = np.sum(grid)

    # Calculate area of a single cell
    cell_area = cell_width * cell_height

    # Estimate total area
    estimated_area = occupied_cells * cell_area

    return estimated_area



def run_episode(agent: object, env: object, **kwargs) -> float:

    """
    run an episode
    """

    assert hasattr(agent, "predict"), \
        "agent must have a `predict` method"
    max_steps = env.max_steps

    render = kwargs.get("render", False)
    if render:
        plt.ion()

    # init
    obs, _ = env.reset()

    if kwargs.get("animate", False):
        animation_maker = AnimationMaker(fps=7, use_logger=True,
                            path="/Users/daniekru/Research/lab/PCNN/cache/")
    else:
        animation_maker = None

    # run
    for t in range(max_steps):

        # agent step
        action = agent.predict(obs)

        # environment step
        obs, reward, done, _, info = env.step(action=action)

        if render and t % kwargs.get("tper", 10) == 0:
            env.render(animaker=animation_maker)

        if reward == 1.0:
            logger.info(f"target reached: {reward}")
            if kwargs.get("reward_block", False):
                break

        # check
        if done:
            logger.info("<done>")
            break

    if animation_maker:
        animation_maker.make_animation(
            name=f"rl_run_{time.strftime('%H%M%S')}")
        animation_maker.play_animation(return_Image=False)

    # env.close()
    if render and animation_maker is None:
        plt.show()

    input("press any key to continue...")

    return reward



class DummyAgent:

    def __init__(self, **kwargs):
        # [lambda, beta]
        self.action_size = kwargs.get("action_size", 2)
        self.t = 0

    def __repr__(self):
        return f"DummyAgent({self.action_size})"

    def predict(self, obs: np.ndarray) -> np.ndarray:

        """
        action space: [-1, 1]
        actions:
        - lambda : -1 (drift) ... 1 (target)
        - beta   : -1 (max) ... 1 (min)
        - new_local_target: -1 (no) ... 1 (yes)
        - speed multiplier
        """
        # return np.random.uniform(-1, 1, self.action_size)
        self.t += 1
        # new_trg = 1.0 if self.t % 40 == 0 else -1
        new_trg = -1
        return np.array([0.8, 0., new_trg, 1])

    def reset(self):
        self.t = 0



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--epochs", type=int, default=100_000,
                        help="number of epochs")
    parser.add_argument("--duration", type=int, default=1000,
                        help="episode duration in steps")
    parser.add_argument("--agent", type=str, default="ppo",
                        help="agent type [ppo, a2c, td3, dummy]")

    args = parser.parse_args()

    # --- instantiate the environment ---
    # PCNN model
    pcnn_model = p.make_default_model()
    pcnn_model.load_params(path="cache/campoverde/pcnn_param.json")

    # PCNN policy deault
    pcnn_model.k = 10
    pcnn_model.policy.speed = 0.002
    pcnn_model.policy.speed_max = 0.01
    pcnn_model.policy.strategy = "max"
    pcnn_model.policy.min_trg_distance = 0.01

    # environment
    env = CampoVerdeEnv(model=pcnn_model,
                        obs_size=5,
                        bounds=(0, 1, 0, 1),
                        min_obs_activity=0.01,
                        isreward=True,
                        objective="target",
                        max_steps=args.duration,
                        max_wall_hits=3,
                        is_policy_first=True)
    logger(env)
    logger(f"objectives: `{env.objective}` [rewarded={env.isreward}]")

    # --- instantiate the agent ---


    # --- train the agent ---

    # Train the agent for 10,000 timesteps
    if not args.load and args.agent != "dummy":
        # Create the PPO agent
        if args.agent == "ppo":
            agent = PPO("MlpPolicy", env, verbose=1)
        elif args.agent == "a2c":
            agent = A2C("MlpPolicy", env, verbose=1)
        elif args.agent == "td3":
            agent = TD3("MlpPolicy", env, verbose=1)
        else:
            raise ValueError("agent not found")

        logger(agent)

        agent.learn(total_timesteps=args.epochs,
                    log_interval=1_000,
                    progress_bar=True)
        logger.info("[PPO training done]")

        # save the model
        if args.save:
            filename = f"{SAVEPATH}{args.agent}_campo_verde"
            agent.save(f"{filename}")
            logger.info(f"[{filename} model saved]")

    # custom dummy agent
    elif args.agent == "dummy":
        agent = DummyAgent(action_size=env.action_size)
        logger("[dummy agent loaded]")
        logger(agent)

    # load the model
    else:
        if args.agent == "ppo":
            agent = PPO.load(f"{SAVEPATH}/ppo_campo_verde")
        elif args.agent == "a2c":
            agent = A2C.load(f"{SAVEPATH}/a2c_campo_verde")
        elif args.agent == "td3":
            agent = TD3.load(f"{SAVEPATH}/td3_campo_verde")
        else:
            raise ValueError("agent not found")

        logger.info(f"[{args.agent} model loaded]")


    # --- run an episode ---
    if args.test:

        _ = run_episode(agent=agent,
                        env=env,
                        render=True,
                        tper=2,
                        animate=args.animate)




