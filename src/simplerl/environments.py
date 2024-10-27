import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numba import jit

import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy
from typing import Optional, Tuple, List
import warnings

import os, sys
base_path = os.getcwd().split("PCNN")[0]+"PCNN/src/"
sys.path.append(base_path)
import simplerl.pcnn_wrapper as pw


try:
    from tools.utils import logger
except ModuleNotFoundError:
    warnings.warn('`tools.utils` not found, using fake logger. Some functions may not work')
    class Logger:

        print('Logger not found, using fake logger')

        def __call__(self, msg: str=""):

            self.info(msg=msg)

        def info(self, msg: str):
            print(msg)

        def debug(self, msg: str):
            print(msg)

    logger = Logger()


""" GLOBAL VARIABLES """

ACTION_SIZE = 4
OBS_SIZE = 5

# define general paths
import os
base_path = os.getcwd().split("PCNN")[0]+"PCNN/"

SAVEPATH = base_path + "src/simplerl/models"
PCNN_PATH = base_path + "cache/campoverde"


""" ENVIRONMENT logic """


def detect_line_interesect(p1, p2, p3, p4):
    

    pass


class Wall:
    def __init__(self, point: np.ndarray,
                 orientation: str, length: float = 1., **kwargs):
        self.point = np.array(point)
        self.length = length
        self.thickness = kwargs.get('thickness', 1.)
        self.orientation = orientation
        self._wall_angle = 0. if orientation == "horizontal" else np.pi/2
        self._wall_vector = np.array([
            self.point,
            self.point + np.array([np.cos(self._wall_angle) * length, np.sin(self._wall_angle) * length])
        ])
        self._color = kwargs.get('color', 'black')
        self._bounce_coefficient = kwargs.get('bounce_coefficient', 1.)

    def vector_collide(self, vector: np.ndarray) -> bool:
        """ check if a given [velocity] vector intersects with the wall """

        return segments_intersect(self._wall_vector[0], self._wall_vector[1],
                                  vector[0], vector[1])

    def collide(self, position: np.ndarray, velocity: np.ndarray,
                radius: float) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Check if a circle collides with the wall and return the new velocity and collision angle"""

        # adjust radius of the object with the wall thickness
        # radius += self.thickness

        # Calculate the nearest point on the wall to the circle's center
        wall_vector = self._wall_vector[1] - self._wall_vector[0]
        wall_length = np.linalg.norm(wall_vector)
        wall_unit = wall_vector / wall_length

        relative_position = position - self._wall_vector[0]
        projection = np.dot(relative_position, wall_unit)
        projection = np.clip(projection, 0, wall_length)

        nearest_point = self._wall_vector[0] + projection * wall_unit

        # Check if the circle intersects with the wall
        distance_to_wall = np.linalg.norm(position - nearest_point)
        if distance_to_wall > radius:
            return None, None

        # Calculate reflection
        normal = (position - nearest_point) / distance_to_wall
        reflection = velocity - 2 * np.dot(velocity, normal) * normal
        new_velocity = reflection * self._bounce_coefficient

        # Calculate angle
        angle = np.arctan2(new_velocity[1], new_velocity[0])

        return new_velocity, angle

    def draw(self, ax: plt.Axes=None,
             alpha: float=1.):
        if ax:
            ax.plot([self._wall_vector[0][0],
                     self._wall_vector[1][0]],
                    [self._wall_vector[0][1],
                     self._wall_vector[1][1]],
                    color=self._color, alpha=alpha,
                    lw=self.thickness)
        else:
            plt.plot([self._wall_vector[0][0],
                      self._wall_vector[1][0]],
                    [self._wall_vector[0][1],
                     self._wall_vector[1][1]],
                    color=self._color, alpha=alpha,
                     lw=self.thickness)


class Room:
    def __init__(self, walls: List[Wall], **kwargs):
        self.walls = walls
        # self.bounds = kwargs.get("bounds", [0, 1, 0, 1])

        wdx = 0.05
        self.bounds = kwargs.get("bounds", [wdx, 1.-wdx,
                                            wdx, 1.-wdx])
        self.name = kwargs.get("name", "Base")

        self.nb_collisions = 0
        self.wall_vectors = np.stack([wall._wall_vector for wall in self.walls])

    def __repr__(self):
        return "Room.{}(#walls{})".format(self.name, len(self.walls))

    def check_bounds(self, position: np.ndarray,
                     radius: float,
                     velocity: np.ndarray=None):

        beyond = False
        if position[0] < self.bounds[0] + radius or \
            position[0] > self.bounds[1] - radius:
            if velocity is not None:
                velocity[0] = -velocity[0]
            beyond = True
        if position[1] < self.bounds[2] + radius or \
            position[1] > self.bounds[3] - radius:
            if velocity is not None:
                velocity[1] = -velocity[1]
            beyond = True

        return beyond, velocity

    def handle_collision(self, position: np.ndarray,
                         velocity: np.ndarray,
                         radius: float,
                         stop: bool=False) -> Tuple[np.ndarray,
                            Optional[float], bool]:


        beyond, new_velocity = self.check_bounds(
            position=position, radius=radius,
            velocity=velocity)

        if beyond:
            logger.error(f"OUT OF THE BORDERS")
            self._self_check(position, velocity, radius,
                             stop)
            return new_velocity, None, True

        for wall in self.walls:
            new_velocity, angle = wall.collide(position,
                                               velocity,
                                               radius)
            if new_velocity is not None:
                self._self_check(position, velocity, radius,
                                 stop)
                return new_velocity, angle, True

        self._self_check(position, velocity, radius, stop)
        return velocity, None, False

    def _self_check(self, position, velocity, radius,
                    stop: bool=False):

        if stop: return

        beyond, new_velocity = self.check_bounds(
            position=position+velocity, radius=0.,
            velocity=velocity)

        if beyond:
            logger.error(f"DOOMED TO COLLIDE")
            logger.error(f"[p:{np.around(position, 3)}, v:{np.around(velocity, 3)}]")

    def handle_vector_collision(self, vector: np.ndarray):
        for wall in self.walls:
            if wall.vector_collide(vector):
                return True
        return False

    def draw(self, ax: plt.Axes=None,
             alpha: float=1.):
        for wall in self.walls:
            wall.draw(ax=ax, alpha=alpha)

    def reset(self):
        self.nb_collisions = 0


def make_room(name: str="square", thickness: float=1.):

    walls = [
            Wall([0, 0], orientation="horizontal",
                 thickness=thickness),
            Wall([0, 0], orientation="vertical",
                 thickness=thickness),
            Wall([0, 1], orientation="horizontal",
                 thickness=thickness),
            Wall([1, 0], orientation="vertical",
                 thickness=thickness),
    ]

    if name == "square":
        room = Room(walls=walls, name="Square")
    elif name == "square1":
        walls += [ 
            Wall([0, 0.5], orientation="horizontal",
                 length=0.5, thickness=thickness),
        ]
        room = Room(walls=walls, name="Square1")
    elif name == "square2":
        walls += [
            Wall([0.5, 0.], orientation="vertical",
                 length=0.5, thickness=thickness),
            # Wall([0.5, 0.5], orientation="horizontal",
            #      length=0.5, thickness=thickness),
        ]
        room = Room(walls=walls, name="Square2")
    elif name == "flat":
        walls += [
            Wall([0., 0.33], orientation="horizontal",
                    length=0.6, thickness=thickness),
            Wall([0., 0.66], orientation="horizontal",
                    length=0.6, thickness=thickness),
        ]
        room = Room(walls=walls, name="Flat")
    elif name == "flat2":
        walls += [
            Wall([0., 0.33], orientation="horizontal",
                    length=0.6, thickness=thickness),
            Wall([0., 0.66], orientation="horizontal",
                    length=0.6, thickness=thickness),
            Wall([0.6, 0.], orientation="vertical",
                    length=0.115, thickness=thickness),
            Wall([0.6, 0.215], orientation="vertical",
                    length=0.215, thickness=thickness),
            Wall([0.6, 0.555], orientation="vertical",
                    length=0.215, thickness=thickness),
            Wall([0.6, 0.875], orientation="vertical",
                    length=0.115, thickness=thickness),
        ]
        room = Room(walls=walls, name="Flat")
    else:
        raise NameError("'{}' is not a room".format(name))

    return room



""" RL ENVIRONMENT """


class CampoVerdeEnv(gym.Env):

    def __init__(self, model: object,
                 agent_body: object,
                 room: object,
                 spatial_layers: object=None,
                 **kwargs):
        super(CampoVerdeEnv, self).__init__()

        """
        Campo Verde environment

        Parameters
        ----------
        model: object
            PCNN model
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
        policy_type: str
            type of policy, one of ["forward", "weighted"].
            Default: "forward"
        """

        global OBS_SIZE
        global ACTION_SIZE

        self.model = model
        OBS_SIZE = self.model.N
        self.model_copy = deepcopy(model)
        self.room = room
        self.agent_body = agent_body
        self.spatial_layers = spatial_layers

        # environment variables
        self.obs_size = OBS_SIZE*2  # half observation
        if spatial_layers is not None:
            OBS_SIZE += len(spatial_layers)
            self.obs_size += len(spatial_layers)
            logger.warning(f"OBS_SIZE increased to {OBS_SIZE}")

        self.bounds = room.bounds
        self.min_obs_activity = kwargs.get("min_obs_activity", 0.001)
        self.is_policy_first = kwargs.get("is_policy_first", False)
        self.policy_type = kwargs.get("policy_type", "forward")
        if self.policy_type == "forward":
            assert isinstance(self.model.policy, pw.VelocityPolicy), \
                "Policy must be a VelocityPolicy"
            self.action_space = spaces.Discrete(4)
            self.action_size = 4
            ACTION_SIZE = 4
        elif self.policy_type == "weighted":
            assert isinstance(self.model.policy, pw.MinimalPolicy), \
                "Policy must be a MinimalPolicy"
            self.action_space = spaces.Box(low=0.,
                                           high=1.0,
                                           shape=(ACTION_SIZE,),
                                           dtype=np.float32)
            self.action_size = self.action_space.shape[0]
        else:
            raise NameError(f"Policy type '{self.policy_type}' is not valid")
        logger(f"Policy type: {self.policy_type}")
        logger(f"{ACTION_SIZE=}")

        # episode variables
        self.curr_pos = None
        self.src_pos = None
        self.trg_pos = None
        self.trg_pos_virtual = None
        self.trg_pos_rewarded = None
        self.t = 0
        self.max_steps = kwargs.get("max_steps", 200)
        self.random_start = kwargs.get("random_start", False)
        self.random_target = kwargs.get("random_target", False)
        self.inner_step_duration = kwargs.get("inner_step_duration", 10)

        # reward
        self.isreward = kwargs.get("isreward", False)
        self.objective = kwargs.get("objective", "target")
        self._max_wall_hits = kwargs.get("max_wall_hits", 10)
        if not self.isreward:
            self.objective = "free"
        self.nb_pc_start = self.model.__len__()
        self.pc_area_start = estimate_area_grid(points=self.model.centers,
                                                box_size=(self.bounds[1]-self.bounds[0],
                                                        self.bounds[3]-self.bounds[2]),
                                                grid_resolution=100)
        self.max_reward = 0.0

        self.record = {
            "est_trg_pos": [],
            "path": [],
            "curr_a": np.zeros(self.model.N),
            "trg_a": np.zeros(self.model.N),
            "obs": np.zeros(self.obs_size),
            "distance": np.inf,
            "reward": []
        }

        # --- action space defined by the policy (above)
        # self.action_space = spaces.Box(low=0.,
        #                                high=1.0,
        #                                shape=(ACTION_SIZE,),
        #                                dtype=np.float32)
        self.action_taken = []

        # --- continuous observation space (obs_size)
        self.observation_space = spaces.Box(low=-1.0,
                                            high=1.0,
                                            shape=(self.obs_size,),
                                            dtype=np.float32)

    def __str__(self):
        return "Campo Verde Environment()"

    def _weighted_policy(self, action: np.ndarray,
                          curr_pos: np.ndarray,
                          trg_pos: np.ndarray) -> tuple:

        # --- init ---
        curr_a = self.model(x=curr_pos)

        # --- action ---
        self.model.policy.update_parameters(action=action)

        # --- drift move
        if self.model.is_tuned(min_n=2):
            drift_a = pw.forward_pc(W=self.model.W_rec,
                                    x=curr_a.copy().reshape(-1, 1),
                                    alpha=self.model._alpha,
                                    beta=self.model._beta, maxval=1.)
            drift_pos = self.model._calc_average_pos(a=drift_a)
            drift_move, _ = pw.calc_movement(pos1=curr_pos,
                                             pos2=drift_pos,
                                             speed=self.model.policy.speed)
        else:
            drift_move = None

        # --- direct move
        if trg_pos is not None:
            # estimate the target position from its representation
            # in neural space
            trg_a = self.model(x=trg_pos, frozen=True)
            trg_pos = self.model._calc_average_pos(a=trg_a)

            self.record["est_trg_pos"] += [
                self.model._calc_average_pos(a=trg_a)]

            # calculate the direct move
            direct_move, _ = pw.calc_movement(
                            pos1=curr_pos,
                            pos2=trg_pos,
                            speed=self.model.policy.speed)

        else:
            direct_move, distance = None, None

        # --- spontaneus move
        spontaneus_move = np.array([
            np.cos(self.model.policy.angle_action),
            np.sin(self.model.policy.angle_action)
        ])

        # --- check for NaN
        if check_nan_move(direct_move) or \
            check_nan_move(drift_move):
            truncated = True

            return curr_pos, np.zeros(self.model.N), \
                np.zeros(self.model.N), None, truncated

        # --- final move: defined by the policy
        policy_move = self.model.policy(moves=[direct_move,
                                               drift_move,
                                               spontaneus_move])

        # --- action finalization
        if not self.is_policy_first:
            policy_move = action[:2] * self.model.policy.speed

        return policy_move

    def _forward_policy(self, action: np.ndarray,) -> np.ndarray:

        # self.action_taken += "".join(["{:01.2f} ".format(a) for a in action])
        self.action_taken = str(action)

        return self.model.policy(moves=action)

    def _internal_step(self, action: np.ndarray,
                      curr_pos: np.ndarray,
                      trg_pos: np.ndarray) -> tuple:

        # --- start: no action provided
        if action is None:

            curr_a = self.model(x=curr_pos, frozen=True)
            trg_a = self.model(x=trg_pos, frozen=True)

            # make observations
            curr_a_out = np.sort(curr_a.flatten())
            trg_a_out = np.sort(trg_a.flatten()
                    ) if trg_pos is not None else np.zeros(self.model.N)
            if np.isnan(trg_a_out).any():
                logger.error(f"NaN in the observation:\n{trg_a_out.flatten()}")
                raise ValueError("NaN in the observation [internal step]")
            return curr_pos, curr_a_out, trg_a_out, None, False

        # --- calculate velocity ---
        if self.policy_type == "forward":
            velocity_move = self._forward_policy(action=action)
            distance = None
        elif self.policy_type == "weighted":
            velocity_move  = self._weighted_policy(action=action,
                                                   curr_pos=curr_pos,
                                                   trg_pos=trg_pos)

        # --- action execution [agent body]
        collision, truncated = self.agent_body(velocity=velocity_move)
        curr_pos = self.agent_body.position.copy()

        # % -- note
        # if it seems still, it might be beacause it is
        # oscillating between two positions
        # % --

        # new representation of the position
        curr_a = self.model(x=curr_pos)
        curr_a_or = curr_a.copy()

        # --- make observations | the top `obs_size` active neurons

        # curr representation
        if curr_a.max() != curr_a.min():
            curr_a = (curr_a - curr_a.min()) / (curr_a.max() - curr_a.min())
        curr_a_out = np.sort(curr_a.flatten())

        # trg representation
        if trg_pos is not None:
            trg_a = self.model(x=trg_pos, frozen=True)
            if trg_a.max() != trg_a.min():
                trg_a = (trg_a - trg_a.min()) / (trg_a.max() - trg_a.min())
            trg_a_out = np.sort(trg_a.flatten())
            self.record["trg_a"] = self.model(x=trg_pos)
            distance = np.sqrt(np.sum((curr_pos - trg_pos)**2))
            self.record["distance"] = distance
        else:
            self.record["trg_a"] = np.zeros(self.model.N)
            self.record["distance"] = np.inf
            trg_a_out = np.zeros(self.model.N)

        if np.isnan(trg_a_out).any():
            # logger.error(f"NaN in the observation:\n{trg_a_out.flatten()}")
            # logger.error(f"{trg_a=}")
            # logger.error(f"{trg_pos=}")
            # raise ValueError("NaN in the observation [internal step]2")
            truncated = True

        # --- record
        self.record["path"] += [curr_pos.tolist()]
        self.record["curr_a"] = curr_a

        return curr_pos, curr_a_out, trg_a_out, distance, truncated

    def _calc_reward(self, curr_pos: np.ndarray, trg_pos: np.ndarray) -> float:

        """
        calculate the reward
        """

        if self.objective == "target" and self.isreward and \
            trg_pos is not None:
            distance = np.linalg.norm(curr_pos - trg_pos)
            reward = 1.0 if distance < self.model.policy.min_trg_distance else 0.0

        # compare the number of new pc
        elif self.objective == "explore":

            # number of cells
            # reward = self.model.__len__() - self.nb_pc_start

            # area
            # pc_area = estimate_area_grid(points=self.model.centers,
            #                 box_size=(self.bounds[1]-self.bounds[0],
            #                 self.bounds[3]-self.bounds[2]),
            #                 grid_resolution=100)
            # reward = pc_area - self.pc_area_start
            reward = max((0., self.model._umask.sum()-2))

        else:
            reward = 0.0

        self.record["reward"] += [reward]

        return reward

    def _preprocess_action(self, action: np.ndarray) -> np.ndarray:

        """
        pre-process the action, mainly map from [0, 1] to
        various ranges
        """

        if action is None:
            self.action_taken = [0., 0., 0., 0., 0.]
            return None

        # lambda: minimum at 0.001
        action[1:] = np.maximum(action[1:], 0.0001)

        self.action_taken = "{:04.1f}° L.".format(action[1])
        self.action_taken += "".join(["{:02.2f} ".format(a) for a in action[1:]])

        return action

    def _dilated_step(self, action: np.ndarray) -> tuple:

        """
        agent step
        """

        # --- STEP ---
        self.curr_pos, obs1, obs2, distance, truncated = self._internal_step(
                                action=action,
                                curr_pos=self.curr_pos,
                                trg_pos=self.trg_pos)

        # --- OBSERVATION ---
        # pass through the spatial layers, if present
        obs0 = obs1.copy()
        if self.spatial_layers is not None:
            obs3 = self.spatial_layers(x=self.curr_pos)
            obs0 = np.concatenate([obs3, obs1])

        # make one single observation
        obs = np.concatenate([obs0, obs2])

        # logger.debug(f"{obs.shape=}, {obs1.shape=}, {obs2.shape=}, {obs0.shape=}")

        # --- REWARD ---
        reward = self._calc_reward(curr_pos=self.curr_pos,
                                   trg_pos=self.trg_pos_rewarded)

        reward *= (1 - bool(truncated))  # no reward if truncated

        # --- INFO --
        info = {"distance": distance}
        self.record["obs"] = obs
        self.t += 1

        # --- DONE ---
        # exit 1: max steps or collisions
        if self.t >= self.max_steps or \
            self.room.nb_collisions >= self._max_wall_hits:
            done = True

            if self.objective == "explore":
                reward = max((0., self.max_reward - reward))
                self.max_reward = max(self.max_reward, reward)

        # exit 2: target reached
        elif reward == 1.0 and self.objective == "target":
            done = True
        else:
            done = False

        if np.isnan(obs).any():
            # logger.error(f"NaN in the observation:\n{obs.flatten()}")
            # logger.error(f"{obs1=}")
            # logger.error(f"{obs2=}")
            # logger.error(f"{obs3=}")
            # raise ValueError("NaN in the observation")
            truncated = True

        return obs, reward, done, truncated,  info

    def step(self, action: np.ndarray) -> tuple:

        """
        agent step
        """

        # if isinstance(action, tuple):
        #     action = action[0]

        # --- action pre-processing ---
        if self.policy_type == "weighted":
            action = self._preprocess_action(action=action)

        for _ in range(self.inner_step_duration):
            obs, reward, done, truncated,  info = self._dilated_step(action=action)
            if done or truncated:
                break

        return obs, reward, done, truncated,  info

    def render(self, mode='human', save=False,
               kind: int=0, **kwargs):

        if kind == 0:
            render_env_obs(env=self, save=save, kwargs=kwargs)
        elif kind == 1:
            render_env_pcnn(env=self, save=save, kwargs=kwargs)
        else:
            raise NameError("kind '{}' is invalid".format(kind))

    def reset(self, episde_meta_info: str=None,
              seed: int=None, **kwargs) -> tuple:

        if seed is not None:
            super().reset(seed=seed)

        # self.src_pos = np.around(np.random.uniform(
        #     self.bounds[0], self.bounds[1], 2), 2)

        # use centroid
        # self.src_pos = np.around(
        #     self.model._calc_centroid(), 2)
        # self.src_pos, _ = self.model.generate_local_position()
        if self.random_start:
            self.src_pos = np.around(np.random.uniform(
                self.bounds[0]+0.1, self.bounds[1]-0.1, 2), 2)
        else:
            self.src_pos = np.array([0.2, 0.2])

        # self.src_pos = np.array([0.5, 0.5])

        self.curr_pos = self.src_pos.copy()

        if self.random_target:
            self.trg_pos = np.around(np.random.uniform(
                self.bounds[0]+0.1, self.bounds[1]-0.1, 2), 2)
        else:
            self.trg_pos = np.array([0.8, 0.75])

        # self.trg_pos, self.trg_pos_virtual = self.model.generate_local_position()
        if self.trg_pos is not None:
            # effective position
            a = self.model(x=self.trg_pos)
            self.trg_pos_virtual =  self.model._calc_average_pos(a=a)

        if self.isreward:
            # self.trg_pos_rewarded = self.trg_pos_virtual.copy() if self.trg_pos_virtual is not None else None
            self.trg_pos_rewarded = self.trg_pos.copy()

        self.t = 0

        # new model
        self.model = deepcopy(self.model_copy)

        # update body
        self.agent_body.set_position(position=self.src_pos.copy())
        self.room.reset()

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

    @property
    def description(self):

        """
        return the description of the environment
        """

        return {
            "name": "Campo Verde",
            "bounds": self.bounds,
            "min_obs_activity": self.min_obs_activity,
            "max_wall_hits": self._max_wall_hits,
            "isreward": self.isreward,
            "objective": self.objective,
            "max_steps": self.max_steps,
            "inner_step_duration": self.inner_step_duration,
            "action_space": self.action_space.shape,
            "observation_space": self.observation_space.shape,
            "model": f"{self.model}",
            "room": f"{self.room}",
            "agent_body": f"{self.agent_body}",
            "spatial_layers": f"{self.spatial_layers}",
        }


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

    if points is None:
        return 0.0

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



""" RENDER [visualization] """


def render_env_obs(env: object, save=False, **kwargs):

        trg_pos = env.trg_pos_virtual if not env.isreward else env.trg_pos_rewarded

        plt.clf()

        if save:
            fig = plt.figure(figsize=(10, 5))

        # --- ENV
        plt.subplot(1, 2, 1)

        # room
        env.room.draw()

        # self.model.render(use_a=False, alpha=0.1)
        env.model.render(use_a=True, alpha=0.9)
                          # new_a=self.model.u.flatten().tolist())

        plt.scatter(env.src_pos[0], env.src_pos[1],
                    color='red', marker="o", s=100)
        # plt.scatter(self.trg_pos[0], self.trg_pos[1],
        #             color='blue', marker="x", s=300)
        if trg_pos is not None:
            plt.scatter(trg_pos[0],
                        trg_pos[1],
                        color='blue', marker="x", s=300)

        plt.plot(np.array(env.record["path"])[:, 0],
                 np.array(env.record["path"])[:, 1],
                    "g-", alpha=0.4, lw=2.)
        plt.scatter(np.array(env.record["path"])[-1, 0],
                   np.array(env.record["path"])[-1, 1],
                   color='green', marker="d", s=80,
                   alpha=0.5)

        plt.xlim(env.bounds[0], env.bounds[1])
        plt.ylim(env.bounds[2], env.bounds[3])
        plt.xticks([])
        plt.yticks([])
        if trg_pos is not None:
            plt.title(f"t={env.t/10:.1f}s\n" + \
            f"pos={np.around(env.curr_pos, 2)}" + \
                f"\ntrg={np.around(trg_pos, 2)}")
        else:
            plt.title(f"t={env.t/10:.1f}s\npos={np.around(env.curr_pos, 2)}")

        # --- MEASURES 
        plt.subplot(2, 2, 2)
        plt.imshow(env.record["curr_a"].reshape(1, env.model.N),
                   cmap="Greys", aspect="auto", interpolation="nearest",
                   vmin=0)
        # plt.title(f"v={self.model.policy.vector[0]:.1f}° - " + \
        #     f"[goal={self.objective}]")
        plt.title("Current representation")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 2, 4)

        plt.imshow(env.record["trg_a"].reshape(1, -1),
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

        plt.pause(0.001)

        if save:
            if str(input("are u sure?")) in ("yes", "y"):
                fig.savefig("/Users/daniekru/Desktop/episode.svg",
                            format="svg", dpi=300)


def render_env_pcnn(env: object, save=False, **kwargs):

        # trg_pos = env.trg_pos_virtual if not env.isreward else env.trg_pos_rewarded

        plt.clf()

        if save:
            fig = plt.figure(figsize=(10, 5))

        # --- ENV
        plt.subplot(1, 2, 1)

        # room
        env.room.draw()

        # self.model.render(use_a=False, alpha=0.1)
        env.model.render(use_a=True, alpha=0.9, edge_alpha=0.5)
                          # new_a=self.model.u.flatten().tolist())

        plt.scatter(env.src_pos[0], env.src_pos[1],
                    color='red', marker="o", s=100,
                    label="start")
        # plt.scatter(self.trg_pos[0], self.trg_pos[1],
        #             color='blue', marker="x", s=300)
        if env.objective == "target":
            plt.scatter(env.trg_pos_rewarded[0],
                        env.trg_pos_rewarded[1],
                        color='blue', marker="x",
                        alpha=0.7, s=300,
                        label="target")

        plt.plot(np.array(env.record["path"])[:, 0],
                 np.array(env.record["path"])[:, 1],
                    "g-", alpha=0.4, lw=2.,
                 label="trajectory")
        plt.scatter(np.array(env.record["path"])[-1, 0],
                   np.array(env.record["path"])[-1, 1],
                   color='green', marker="d", s=80,
                   alpha=0.5)

        plt.xlim(env.bounds[0], env.bounds[1])
        plt.ylim(env.bounds[2], env.bounds[3])
        # square box layout
        plt.xticks([])
        plt.yticks([])
        plt.title(f"t={env.t/10:.1f}s " + \
            f"$N_{{PC}}=${env.model.__len__():.0f}" + \
            f"\n{env.action_taken}")
        plt.axis('equal')
        plt.axis("off")
        plt.legend(loc="lower right")

        # --- MEASURES 
        plt.subplot(2, 2, 2)
        da_list = env.model.record['da'][-min(1000,
                            len(env.model.record['da'])-1):]
        ach_list = env.model.record['ach'][-min(1000,
                            len(env.model.record['ach'])-1):]
        plt.plot(range(len(da_list)), da_list, 'g-',
                 alpha=0.5, label="DA")
        plt.plot(range(len(ach_list)), ach_list, 'b-',
                 alpha=0.5, label="ACh")
        # plt.title(f"v={self.model.policy.vector[0]:.1f}° - " + \
        #     f"[goal={self.objective}]")
        plt.title("Concentration of modulators")
        plt.legend(loc="lower right")
        # plt.xlim(0, len(da_list))
        plt.ylim(0, 1.1)
        plt.xticks([])

        plt.subplot(2, 2, 4)

        # plt.imshow(env.model._Wff, cmap="viridis", aspect="auto", interpolation="nearest",
        #            vmin=0.)
        # plt.title(f"Target representation [ACh={self.model._ACh:.2f}]")
        plt.title(f"Activity")
        plt.imshow(env.model.u.reshape(1, -1),
                   cmap="Greys", aspect="auto", interpolation="nearest",
                   vmin=0)
        # plt.imshow(self.record["obs"].reshape(1, -1),
        #            cmap="Greys", aspect="auto", interpolation="nearest",
        #            vmin=0)
        # plt.title(f"observation [ACh={self.model._ACh:.2f}]")
        plt.xticks([])
        plt.yticks([])

        plt.pause(0.001)

        if save:
            if str(input("are u sure?")) in ("yes", "y"):
                fig.savefig("/Users/daniekru/Desktop/episode.svg",
                            format="svg", dpi=300)



""" AGENT """


class AgentBody:
    def __init__(self, room: Room,
                 position: Optional[np.ndarray] = None,
                 **kwargs):
        self.radius = kwargs.get("radius", 0.05)
        self.position = position if position is not None else self._random_position()
        self.velocity = np.zeros(2).astype(float)
        self.color = kwargs.get("color", "red")
        self._room = room
        self.verbose = kwargs.get("verbose", False)
        self.bounce_coefficient = kwargs.get("bounce_coefficient", 0.5)

    def _random_position(self):
        return np.random.rand(2)

    def __call__(self, velocity: np.ndarray):

        self.velocity = velocity
        self.velocity, collision = self._handle_collisions()

        # update position
        # + considering a possible collision
        self.position += self.velocity * (1 + \
                            self.bounce_coefficient * \
                            1 * collision)

        if collision:
            logger.debug(f"new velocity: {np.around(self.velocity, 3)}")
            logger.debug(f"new position: {np.around(self.position, 3)}")

        truncated = not self._room.check_bounds(
                                position=self.position,
                                radius=self.radius*0.2)

        return self.position.copy(), collision, truncated*0

    def _handle_collisions(self) -> Tuple[Optional[float], bool]:
        new_velocity, _, collision = self._room.handle_collision(
            self.position, self.velocity, self.radius)
        if collision:
            self.velocity = new_velocity
            # Move the agent slightly after collision to prevent sticking
            self._room.nb_collisions += 1

            if self.verbose:
                logger.debug("%collision detected%")

        return new_velocity, collision

    def set_position(self, position: np.ndarray):
        self.position = position

    def draw(self, ax: plt.Axes):
        ax.add_patch(Circle(self.position, self.radius,
                            fc=self.color, ec='black'))
        ax.arrow(self.position[0], self.position[1],
                 5*self.velocity[0], 5*self.velocity[1],
                 head_width=0.02, head_length=0.02,
                 fc='black', ec='black')

    def render(self, ax: plt.Axes):
        self._room.draw(ax=ax)
        self.draw(ax=ax)


class Zombie:

    def __init__(self, body: AgentBody,
                 speed: float = 0.1,
                 makefigure: bool = False):

        self.body = body
        self.speed = speed

        self.p = 0.5
        self.velocity = np.zeros(2)

        if makefigure:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig, self.ax = None, None

    def __call__(self, **kwargs):

        self.p += (0.2 - self.p) * 0.02
        self.p = np.clip(self.p, 0.01, 0.99)

        if np.random.binomial(1, self.p):
            angle = np.random.uniform(0, 2*np.pi)
            self.velocity = self.speed * np.array([np.cos(angle),
                                              np.sin(angle)])
            self.p *= 0.2

        collision, _ = self.body(velocity=self.velocity)

        if collision:
            self.velocity = -self.velocity

        return self.velocity, collision

    @property
    def position(self):
        return self.body.position

    def render(self, ax=None):

        if ax is None:
            ax = self.ax
            ax.clear()
        self.body.draw(ax=ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        if ax == self.ax:
            self.fig.canvas.draw()



class DummyAgent:

    def __init__(self, **kwargs):
        # [speed, angle, lambda_1, lambda_2, lambda_3]
        self.action_size = ACTION_SIZE
        self.t = 0

    def __repr__(self):
        return "DummyAgent({})".format(self.action_size)

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
        return np.array([np.random.normal(0, 0.3)**2,
                         action_to_angle(np.random.uniform(-1, 1)),
                         0.5, 0.3, 0.5])

    def reset(self):
        self.t = 0



""" UTILS """

def action_to_angle(value: float) -> float:

    """
    map an action value (float in [-1, 1])
    to and angle in radians (float in [0, 2*pi])
    """

    return (value + 1) * np.pi


@jit(nopython=True)
def two_lines_intersection(p1, p2, p3, p4):
    """
    Check if two lines intersect, where
    (p1, p2) is the first line and (p3, p4) is the second line.
    Returns the point of intersection if the lines intersect, None otherwise.
    """
    (x11, y11), (x12, y12) = p1, p2
    (x21, y21), (x22, y22) = p3, p4

    # Calculate the direction of the lines
    dx1 = x12 - x11
    dy1 = y12 - y11
    dx2 = x22 - x21
    dy2 = y22 - y21

    # Calculate the determinant
    determinant = dx1 * dy2 - dy1 * dx2

    # If the determinant is zero, the lines are parallel or coincident
    if determinant == 0:
        return None

    # Calculate the intersection point
    t1 = ((x21 - x11) * dy2 - (y21 - y11) * dx2) / determinant
    t2 = ((x21 - x11) * dy1 - (y21 - y11) * dx1) / determinant

    # Check if the intersection point lies on both line segments
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        x = x11 + t1 * dx1
        y = y11 + t1 * dy1
        return np.array([x, y])
    else:
        return None


@jit(nopython=True)
def calc_angle(vector: np.ndarray) -> float:

        if vector[1, 0] - vector[0, 0] == 0:
            if vector[1, 1] - vector[0, 1] > 0:
                return np.pi/2
            return 3*np.pi/2

        angle = np.arctan(
            (vector[1, 1] - vector[0, 1]) / \
                (vector[1, 0] - vector[0, 0])
        )

        if angle < 0:
            angle += np.pi

        return angle


@jit(nopython=True)
def calc_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors with the same origin.

    Parameters
    ----------
    v1 : np.ndarray
        The first vector of shape (2, 2).
    v2 : np.ndarray
        The second vector of shape (2, 2).

    Returns
    -------
    float
        The angle between the two vectors in radians.
    """
    # Extract direction vectors
    dir_v1 = v1[1] - v1[0]
    dir_v2 = v2[1] - v2[0]
    
    # Calculate the dot product of the direction vectors
    dot_product = np.dot(dir_v1, dir_v2)
    
    # Calculate the magnitudes of the direction vectors
    norm_v1 = np.linalg.norm(dir_v1)
    norm_v2 = np.linalg.norm(dir_v2)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Ensure the cosine value is within the valid range [-1, 1] to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle = np.arccos(cos_theta)
    
    return angle


@jit(nopython=True)
def vector_norm(vector: np.ndarray):
    return np.sqrt((vector[0, 0] - vector[1, 0])**2 +
                   (vector[0, 1] - vector[1, 1])**2)


@jit(nopython=True)
def reflect_point(point1: np.ndarray,
                  point2: np.ndarray) -> tuple:
    """
    reflect a point (point1) wrt an origin (point2)
    """
    return (point2[0] + (point2[0] - point1[0]),
            point2[1] + (point2[1] - point1[1]))


@jit(nopython=True)
def reflect_vector(vector1: np.ndarray,
                   vector2: np.ndarray,
                   momentum_c: float=1.) -> np.ndarray:

    # Normalize vector2 to get the direction vector
    vector2_dir = vector2[1] - vector2[0]
    vector2_dir /= np.linalg.norm(vector2_dir)

    # Calculate the projection of vector1[1] onto vector2_dir
    projection_length = np.dot(vector1[1] - vector1[0], vector2_dir)
    projection = projection_length * vector2_dir

    # Calculate the reflection point
    reflection_point = 2 * projection - (vector1[1] - vector1[0])
    reflection_point = vector2[0] - momentum_c * reflection_point

    # Create the reflected vector
    reflected_vector = np.stack((vector2[0],
                                 reflection_point))


    return reflected_vector


@jit(nopython=True)
def check_nan_move(move: np.ndarray) -> bool:
    if move is None:
        return False
    return np.isnan(move).any()


@jit(nopython=True)
def line_intersection(p1, p2, p3, p4):

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denom == 0:
        return None  # Lines are parallel

    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom

    return (px, py)

@jit(nopython=True)
def is_point_on_segment(p, segment_start, segment_end):
    x, y = p
    x1, y1 = segment_start
    x2, y2 = segment_end

    # Check if the point is within the bounding box of the segment
    if (min(x1, x2) <= x <= max(x1, x2) and
        min(y1, y2) <= y <= max(y1, y2)):
        return True
    return False

@jit(nopython=True)
def segments_intersect(p1, p2, p3, p4):
    intersection = line_intersection(p1, p2, p3, p4)
    if intersection is None:
        return False  # Parallel or coincident segments

    # Check if the intersection point is on both segments
    return (is_point_on_segment(intersection, p1, p2) and
            is_point_on_segment(intersection, p3, p4))




""" MAIN """


def draw_env(t: int, room: Room, agent: list,
             ax: plt.Axes, **kwargs):

    if t % kwargs.get("fpi", 10) != 0:
        return

    ax.clear()
    room.draw(ax=ax)
    agent.draw(ax=ax)
    ax.set_xlim((room.bounds[0], room.bounds[1]))
    ax.set_ylim((room.bounds[2], room.bounds[3]))
    ax.set_title(kwargs.get("title", ""))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.grid()
    plt.pause(kwargs.get("fpause", 0.01))


def run(room: Room, agent: list,
        duration: int=100,
        **kwargs):


    # plot
    _, ax = plt.subplots()
    title = ""

    logger("running for {}tu".format(duration))

    for t in range(duration):

        title = "t={}".format(t)
        draw_env(t=t, ax=ax, room=room, agent=agent, title=title,
                 fpi=kwargs.get("fpi", 10),
                 fpause=kwargs.get("fpause", 0.01))

        agent()

    logger("done")


def run_episode(agent: object, env: object, **kwargs) -> float:

    """
    run an episode
    """

    assert hasattr(agent, "predict"), \
        "agent must have a `predict` method"
    max_steps = env.max_steps
    logger(f"{max_steps=}")

    render = kwargs.get("render", False)
    render_kind = kwargs.get("render_kind", 0)
    if render:
        plt.ion()

    # init
    obs, _ = env.reset()

    # run
    for t in range(max_steps):

        # agent step
        action = agent.predict(obs)

        if isinstance(action, tuple):
            action = action[0]

        # environment step
        obs, reward, done, truncated, info = env.step(action=action)

        if render and t % kwargs.get("tper", 10) == 0:
            env.render(kind=render_kind)

        if reward == 1.0 and env.objective == "target":
            logger.info("target reached: {}".format(reward))
            if kwargs.get("reward_block", False):
                break

        # check
        if done:
            logger.info("<done>")
            break
        elif truncated:
            logger.info("<truncated>")
            logger.debug(f"position={env.agent_body.position}")
            break

    # env.close()
    if render:
        plt.show()

    input("press any key to continue...")

    return reward



if __name__ == "__main__":

    np.random.seed(0)


    #
    walls = (
        Wall(point=[0, 0], orientation="horizontal"),
        Wall([0, 0], orientation="vertical"),
        Wall([0, 1], orientation="horizontal"),
        Wall([1, 0], orientation="vertical"),
        Wall([0.5, 0], orientation="vertical", length=0.5)
    )

    room = Room(walls=walls)
    print(room)

    colors = ["red", "blue", "green", "yellow", "orange"]

    agents = []
    for i in range(5):
        agent = Agent(room=room,
                      position=np.random.rand(2),
                      radius=np.random.uniform(0.05, 0.1),
                      speed=0.009,
                      color=colors[i])
        print(agent)
        agents.append(agent)


    # fig, ax = plt.subplots()
    # room.draw(ax=ax)
    # agent.draw(ax=ax)
    # plt.show()

    run(room=room, agents=agents, duration=10_000,
        fpi=10, fpause=0.01)






