import numpy as np
from scipy.stats import beta, dirichlet

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
from IPython.display import display, clear_output
from tqdm import tqdm
import os
import time

import pcnn_wrapper as pw
import environments as ev
from tools.utils import AnimationMaker

logger = ev.logger
base_path = os.getcwd().split("PCNN")[0]+"PCNN/"

os.chdir(base_path)
from src.minimal_model import Policy

MEDIA_PATH = os.path.join(base_path, "media/")

def clf():
    clear_output(wait=True)


""" Agent """


class RandomAgent:

    def __init__(self, position: np.ndarray,
                 room: ev.Room,
                 pcnn: pw.PCNNlayer,
                 speed: float=0.01,
                 epsilon: float=0.1,
                 turn_rate: float=0.1, **kwargs):

        # objects
        self._room = room
        self._pcnn = pcnn

        # position
        self.position = position
        self.start_position = position
        self.speed = speed
        self.turn_rate = turn_rate
        angle = np.random.uniform(0, 2*np.pi)
        self.velocity = self.speed * np.array([np.cos(angle),
                                               np.sin(angle)])

        # policy
        self.epsilon = epsilon
        self.param1 = kwargs.get('param1', 1.0)

        # other
        self.radius = kwargs.get('radius', 0.004)
        self.is_training = kwargs.get('is_training', True)

        # record
        self.trajectory = []
        self.curr_representation = np.zeros(self._pcnn.N)
        self.new_representation = np.zeros(self._pcnn.N)
        self.trg_representation = np.zeros(self._pcnn.N)
        self.drift_threshold = kwargs.get("drift_threshold", 0.01)

        # variables
        self.t = 0
        self.delay = 20

        # record
        self.activity_record = np.zeros((self._pcnn.N, 1000))
        self.distance = 1.
        self.new_position = None
        self.velocity_vector = np.zeros((2, 2))
        self.trg_velocity_vector = np.zeros((2, 2))
        self.drift_velocity_vector = np.zeros((2, 2))

    def __repr__(self):
        return f"{self.__class__}"

    def __call__(self, action: int=None,
                 trg_pos: np.ndarray=None,
                 **kwargs):

        _, _, collision = self._room.handle_collision(
            self.position, self.velocity, self.radius*1.05)

        if collision:
            self.velocity = -self.velocity
            self.position += self.velocity

        self._apply_policy(trg_pos=trg_pos,
                           action=action)
        # Ensure constant speed
        if self.velocity.sum() > 0:
            self.velocity = self.speed * self.velocity / np.linalg.norm(self.velocity)

        # Update position
        self.position += self.velocity

        # update pcnn activity
        self._pcnn(x=self.position, frozen=not self.is_training,
                   **kwargs)
        self.curr_representation = self._pcnn.u.copy()

        # record trajectory
        self.trajectory.append(self.position.tolist())
        self.activity_record[:, :-1] = self.activity_record[:, 1:]
        self.activity_record[:, -1] = self.curr_representation.flatten()

        return self.position

    def _random_policy(self):

        # Add some randomness to the velocity direction
        angle_change = np.random.normal(0, self.turn_rate)
        rotation_matrix = np.array([
            [np.cos(angle_change), -np.sin(angle_change)],
            [np.sin(angle_change), np.cos(angle_change)]
        ])
        self.velocity = np.dot(rotation_matrix, self.velocity)

    def _trim_active_neurons(self, representation: np.ndarray):

        """ the active neurons residing beyond a wall
        should not be active """

        active_idx = np.where(representation > self.drift_threshold)[0]
        # logger.debug(f"{active_idx=}")
        for idx in active_idx:
            dx = self._pcnn.centers[idx] - self.position
            vector = np.array([self.position, self._pcnn.centers[idx]])
            # _, _, collision = self._room.handle_collision(
            #     self.position, vector, 0.01)
            collision = self._room.handle_vector_collision(
                vector=vector
            )
            if collision:
                # logger.debug(f"[{idx}] {self._pcnn.centers[idx]} is beyond a wall")
                representation[idx] *= 0
                # input()

            # else:
            #     logger.debug(f"[{idx}] no collision {self._pcnn.centers[idx]}, {self.position}")
        # time.sleep(1.)

        return representation

    def _parse_action(self, action: int):

        if isinstance(action, tuple):
            action = action[0]

        eps = 0.01 if action % 2 == 0 else 1.  # 0.2, 0.8
        lambda_ = 2 * (action // 2 - 0.5) # -1, 1

        return eps, lambda_

    def _apply_policy(self, trg_pos: np.ndarray,
                      action: int=None):

        self.t += 1

        if trg_pos is None:
            self._random_policy()
            return

        # heuristic
        if action is not None:
            if isinstance(action, np.int64):
                self.epsilon, self.param1 = self._parse_action(action)
            else:
                self.epsilon, self.param1 = action
        else:
            raise ValueError("No heuristic defined")

        # --- proximal positions
        drift_representation, drift_position, drift_velocity = self._drift_policy(
                                beta_drift=self.param1)

        # exit 1
        if drift_representation is None:
            self.new_representation = np.zeros(self._pcnn.N)
            self._random_policy()
            return

        # --- target representation
        trg_representation = self._pcnn(x=trg_pos, frozen=True)
        trg_velocity = trg_pos - self.position
        trg_velocity /= trg_velocity.sum()
        self.trg_representation = trg_representation.copy()

        # --- current representation 1 - representation
        # if self.t % self.delay == 0:
        new_representation = self.epsilon * trg_representation + \
            (1 - self.epsilon) * drift_representation
        new_position = self._pcnn._calc_average_pos(a=new_representation)
        # new_velocity = new_position - self.position

        # --- current representation 2 - position
        # new_position = self.epsilon * trg_pos + (1 - self.epsilon) * drift_position
        new_velocity = new_position - self.position
        new_velocity /= new_velocity.sum()
        new_velocity *= self.speed

        # --- current representation 3 - velocity
        # new_velocity = self.epsilon * trg_velocity + (1 - self.epsilon) * drift_velocity
        # new_velocity /= new_velocity.sum()
        # new_velocity *= self.speed
        # new_position = self.position + new_velocity

        # new_position = self.position + new_velocity
        # self.curr_representation = self._pcnn(x=new_position,
        #                                       frozen=True)

        # new_position, self.trg_representation = self._calc_min_drift(
        #     trg_pos=trg_pos, drift_pos=drift_position
        # )
        # self.curr_representation = self.trg_representation.copy()
        self.new_representation = drift_representation #self.trg_representation.flatten()
        self.new_position = new_position

        self.velocity_vector = np.vstack((self.position,
                                          new_position))
        self.trg_velocity_vector = np.vstack((self.position,
                                              trg_pos))
        self.drift_velocity_vector = np.vstack((self.position,
                                                drift_position))

        if new_position is None:
            self._random_policy()
            return
        self.velocity, _ = pw.calc_movement(pos1=self.position,
                                            pos2=new_position,
                                            speed=self.speed)

        self.distance = np.linalg.norm(trg_pos - self.position)

    def _drift_policy(self, beta_drift: float) -> tuple:

        """ calculate the drift representation """

        # --- proximal positions
        drift_representation = self._pcnn.W_rec @ self.curr_representation.reshape(-1, 1) - \
             1. * self.curr_representation.reshape(-1, 1)
        drift_representation = np.maximum(drift_representation, 0)

        # --- exit 1
        if drift_representation.max() == 0.:
            return None, None, None

        # --- only neurons within the fov
        drift_representation = self._trim_active_neurons(
                                    representation=drift_representation)

        # % modulate 1
        # drift_representation = self._max_var_drift(
        #                                 a=drift_representation)

        # % modulate 2
        drift_representation = self._variable_drift(a=drift_representation,
                                                    threshold=0.001,
                                                    beta=beta_drift)
        drift_position = self._pcnn._calc_average_pos(
                                    a=drift_representation.copy())
        drift_velocity = drift_position - self.position
        drift_velocity /= drift_velocity.sum()

        return drift_representation, drift_position, drift_velocity

    def _calc_min_drift(self, trg_pos: np.ndarray,
                        drift_pos: np.ndarray):

        representation = np.zeros(2)
        while np.where(representation > 0.01, 1, 0).sum() < 1:
            trg_pos = trg_pos + (drift_pos - trg_pos) * 0.1
            representation = self._pcnn(x=trg_pos, frozen=True)

        # input(f"{trg_pos=}")
        # print(f"{trg_pos=}")
        time.sleep(0.2)

        return trg_pos, representation

    def _max_var_drift(self, a: np.ndarray):

        z = np.zeros_like(a)
        z[a.argmax()] = 1.
        return z

    def _variable_drift(self, a: np.ndarray, threshold: float=0.1, beta: float=0.1):

        if beta is None:
            beta = 0.

        # minmax normalization
        a = (a - a.min()) / (a.max() - a.min())

        nonzero_idx = np.where(a > threshold)[0]
        v = a[nonzero_idx]
        v = v + (v.mean() - v) * beta
        a[nonzero_idx] = v

        a = (a - a.min()) / (a.max() - a.min())
        return a

    def _update_connections(self):

        # Update the recurrent connections
        self._pcnn._update_recurrent()

        # Update the graph
        self._pcnn._calc_graph()

        self._pcnn.connectivity = pw.remove_wall_intersecting_edges(
            nodes=self._pcnn.centers.copy(),
            connectivity_matrix=self._pcnn.connectivity.copy(),
            walls=self._room.wall_vectors.copy()
        )

        self._pcnn.W_rec *= self._pcnn.connectivity

    def set_policy(self, policy: str):
        self.policy = policy

    def draw(self, ax: plt.Axes=None, show_pcnn: bool=False,
             edge_alpha: float=0.2, **kwargs):

        if show_pcnn:
            self._pcnn.render(ax=ax,
                              alpha=kwargs.get("alpha", 0.5),
                              edge_alpha=edge_alpha, color='blue',
                              use_a=kwargs.get("use_a", True),
                              new_a=self.new_representation)

        if ax is None:
            plt.plot(*np.array(self.trajectory).T, c='red',
                     alpha=0.5, lw=0.8, ls='--')
            plt.gca().add_patch(Circle(self.position,
                                       self.radius, fc="blue", ec='black'))
            plt.plot(self.velocity_vector[:, 0], self.velocity_vector[:, 1],
                     "r-", label="current", alpha=0.5, lw=3)
            plt.plot(self.drift_velocity_vector[:, 0],
                     self.drift_velocity_vector[:, 1],
                     "k-", label="drift", alpha=0.5, lw=3)
            plt.plot(self.trg_velocity_vector[:, 0],
                     self.trg_velocity_vector[:, 1],
                     "b-", label="target", alpha=0.5,
                     lw=3)
            plt.legend(loc="lower right")
        else:
            ax.plot(*np.array(self.trajectory).T, c='red',
                    alpha=kwargs.get("tr_alpha", 0.2),
                    lw=1., ls='-', label="trajectory")
            ax.add_patch(Circle(self.position,
                                self.radius, fc="red", ec='black',
                                label="agent"))

    def reset(self, position: np.ndarray=None):
        self.position = self.start_position.copy() if position is None else position
        self.activity_record = np.zeros((self._pcnn.N, 1000))
        self.trajectory = []
        angle = np.random.uniform(0, 2*np.pi)
        self.velocity = self.speed * np.array([np.cos(angle), np.sin(angle)])

        self.curr_representation = np.zeros(self._pcnn.N)
        self.new_representation = np.zeros(self._pcnn.N)
        self.trg_representation = np.zeros(self._pcnn.N)
        self.distance = 1.



class Reward:

    def __init__(self, start_pos: np.ndarray,
                 onset: int=1000, **kwargs):

        self.position = start_pos if start_pos is not None else np.random.rand(2)
        self.radius = kwargs.get('radius', 0.01)
        self.onset = onset
        self.is_active = False
        self.is_reached = False

    def __call__(self, t: int):
        if t < self.onset:
            return None
        self.is_active = True
        return self.position

    def check(self, position: np.ndarray):
        if not self.is_active:
            return False
        distance = np.linalg.norm(position - self.position)
        self.is_reached = 1 * (distance < self.radius)
        return self.is_reached

    def reset(self):
        self.is_active = False
        self.is_reached = False



class Heuristic:
    def __init__(self, params: np.ndarray=None):

        # assert len(params) == 3, "Heuristic needs three parameters"
        if params is None:
            params = [0.1, 1., 0.1, 1., 1.]

        self.v = 0.
        self.dv = 0.
        self.g = 0.
        self.g_eq = 0.
        self.sign = 1.
        self.eps = 0.
        self.params = params
        self.dist = 1.

    def __call__(self, distance: float):

        self.dv = distance - self.v

        self.v += (distance - self.v) * self.params[0]
        self.g += (self.g_eq - self.g) * 0.01
        self.g_eq = self.g_eq if abs(self.g_eq-self.g) > 0.01 else np.random.uniform(0.5, self.params[4])

        self.sign = np.sign(self.dv) if self.dv != 0 else self.sign  # last sign

        # dv = 1 / (1 + np.exp(-self.params[1]*\
        #     (dv-self.params[2])))

        dist = 1 / (1+np.exp(20*(distance-0.07)))

        noise_flag = np.exp(-self.dv**2/(2*0.003**2))>0

        # self.eps = - noise_flag*self.g + self.params[3] - np.clip(self.dv, self.params[1], self.params[2]) + dist

        p = 0.73*(1-noise_flag*(self.g_eq-self.g))
        self.eps = p*(abs(self.dv)>0.01) + (1-p)*(abs(self.dv)<=0.01)
        self.eps = np.clip(self.eps, 0, 1)
        self.dist = dist

        if distance < 0.03:
            self.eps = 0.95

        # return self.params[2]
        # param1 = self.params[4] * (distance - self.v)

        # return np.clip(self.eps, 0, 1), None
        return max((self.eps,dist)), self.g*noise_flag*2

    def __str__(self):
        return f"v={self.v:.2f}, dv={self.dv:.3f}, eps={self.eps:.2f}, dist={self.dist:.3f}, g={self.g:.2f}"



class Observer:

    def __init__(self, shape_ext: tuple, shape_repr: tuple,
                 tau: float=10):

        self.tau = tau

        self.dist_ext = np.zeros(shape_ext)
        self.repr = np.zeros(shape_repr)
        self.dx_ext = np.zeros(shape_ext)
        self.dx_repr = np.zeros(shape_repr)

    def __call__(self, distance_ext: float, representation: float):

        self.dist_ext += (distance_ext - self.dist_ext) / self.tau
        self.repr += (representation - self.repr) / self.tau

        self.dx_ext = distance_ext - self.dist_ext
        self.dx_repr = np.abs(representation - self.repr)

        if self.dx_repr.sum() == 0:
            obs2 = 0.
        else:
            # obs2 = (self.dx_repr/self.dx_repr.sum()).
            obs2 = self.dx_repr.mean()
            if obs2 > 1:
                logger.debug(f"{self.dx_repr=}, {self.dx_repr.sum=}")
                raise ValueError

        return self.dx_ext.sum(), obs2

    def make_reward(self):

        return self.dx_ext.sum() > 0.001 and self.dx_repr.sum() > 0.001



class HeuristicBayes:

    def __init__(self, obs_params: dict=None):

        self.observer = Observer(**obs_params)
        self.bayes = BetaBayes()

    def __call__(self, distance_ext: float, distance_int: float):

        # observe
        flag_ext, flag_int = self.observer(distance_ext, distance_int)

        # choose action
        action = self.bayes(method="expected" if flag_ext else "sample")

        return action, ext, int



class BetaBayes:

    def __init__(self, lr=0.1, min_value=0.1, eq_beta=1.):

        self.alpha = 1
        self.beta = 1
        self.eq_beta = eq_beta
        self.lr = lr
        self.action = 0
        self.min_value = min_value

    def __call__(self, method="sample"):
        if method == "sample":
            self.action = 1*(beta.rvs(self.alpha, self.beta) > 0.5)
        elif method == "expected":
            self.action = self.alpha / (self.alpha + self.beta)
        else:
            raise ValueError(f"Unknown method: {method}")
        return self.action

    def update(self, reward):
 
        if reward == 0:

            # drift towards equilibrium
            if self.eq_beta is not None:
                self.beta += (self.eq_beta - self.beta) * self.lr
            return

        update_value = self.lr * reward

        if self.action > 0.5:
            self.beta *= 1 - update_value
        else:
            self.beta *= 1 + update_value

        # normalization
        self.beta = np.clip(self.beta, self.min_value,
                            1/self.min_value)



class BayesianPolicy:

    """
    Bayesian policy to choose between two actions:
    - epsilon: the weight for the representation sum
    - lambda: the bias in the drift representation
    """

    def __init__(self, alpha_beta=1, beta_beta=1,
                 alpha_dirichlet=np.array([1, 1, 1]),
                 lr: float=0.1):

        # Initialize priors
        self.alpha_beta = alpha_beta
        self.beta_beta = beta_beta
        self.alpha_dirichlet = alpha_dirichlet
        self.lr = lr

        self.last_action = None

    def __call__(self, method: str="sample"):

        if method == "sample":
            # Sample from Beta distribution for first component
            action1 = beta.rvs(self.alpha_beta, self.beta_beta) > 0.5

            # Sample from Dirichlet distribution for second component
            action2_probs = dirichlet.rvs(self.alpha_dirichlet)[0]
            action2 = np.random.choice(3, p=action2_probs)
        elif method == "expected":
            # Get expected values
            action1, action2 = self.get_expected_values()
        else:
            raise ValueError(f"Unknown method: {method}")

        self.last_action = (action1, action2)

        return action1, action2

    def update(self, reward: int):

        action1, action2 = self.last_action

        if reward == 0:
            return

        update_value = self.lr * reward

        if reward > 0:
            if action1:
                self.alpha_beta += update_value
            else:
                self.beta_beta += update_value
            self.alpha_dirichlet[action2] += update_value
        else:
            if action1:
                self.beta_beta -= update_value
                self.alpha_beta += update_value / 2
            else:
                self.alpha_beta -= update_value
                self.beta_beta += update_value / 2

            penalty = update_value / (len(self.alpha_dirichlet) - 1)
            for i in range(len(self.alpha_dirichlet)):
                if i == action2:
                    self.alpha_dirichlet[i] -= penalty
                else:
                    self.alpha_dirichlet[i] += penalty

        self.alpha_dirichlet = np.maximum(self.alpha_dirichlet, 0.01)
        self.alpha_beta = max(self.alpha_beta, 0.01)
        self.beta_beta = max(self.beta_beta, 0.01)

    def get_expected_values(self):
        ev_action1 = self.alpha_beta / (self.alpha_beta + self.beta_beta)
        ev_action2 = self.alpha_dirichlet / np.sum(self.alpha_dirichlet)
        return ev_action1, ev_action2



class LocalDummy:

    def __init__(self, **kwargs):
        self.position = np.random.random(2)
        self.trajectory = []

    def __repr__(self):
        return "Dummy agent"

    def predict(self, *args, **kwargs):
        self.position = np.random.random(2)
        self.trajectory.append(self.position)
        return np.random.randint(0, 6)

    def draw(self, **kwargs):
        plt.plot(*np.array(self.trajectory).T, c='red',
                    alpha=0.5, lw=0.8, ls='--')



class HeuristicII:
    def __init__(self, tau1: float=10, tau2: float=10,
                 threshold1: float=0.01, threshold2: float=0.01,
                 delay: int=10, **kwargs):

        self.tau1 = tau1
        self.tau2 = tau2
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.delay = delay
        self.delay_bias = 0

        self.eps_eq = 1.
        self.eps = self.eps_eq

        self.lambda_eq = 0.
        self.lambda_ = self.lambda_eq

        self.eps_rec = []
        self.lambda_rec = []
        self.obs1_rec = []

        self.obs2 = 0.

        self.action = np.zeros(2)
        self.action_str = ""

        self.t = 0

    def __repr__(self):
        return f"HeuristicII(delay={self.delay})"
        self.eps_rec = []

    def predict(self, obs):

        # init
        self.t += 1
        obs1, obs2 = obs
        self.obs2 = obs2

        # dynamics
        self.eps += (self.eps_eq - self.eps) / self.tau1 - 1*(np.abs(obs1) < self.threshold1 and \
                                                              (self.eps_eq - self.eps)**2 < 0.001)
        # self.eps += (self.eps_eq - self.eps) / self.tau1 - 1*(obs1<self.threshold1 and \
        #                                         (self.eps_eq - self.eps)**2<0.001)
        self.lambda_ += (self.lambda_eq - self.lambda_) / self.tau2 + (obs2<self.threshold2 and \
                                                abs(self.lambda_eq - self.lambda_)<0.001)

        # record
        self.eps_rec += [self.eps]
        self.lambda_rec += [self.lambda_]
        self.obs1_rec += [obs1]

        # make action only evey delay steps
        # if self.t % self.delay == 0:
            # self.action[0] = 0.95 if self.eps > 0.75 + self.delay_bias * 0.5 else 0.05
        self.action[0] = 0.95 * (self.eps > 0.65) + 0.05 * (self.eps <= 0.65)
        self.action[1] = self.lambda_
            # self.action[1] = np.random.choice([1., 5.]) if self.lambda_ > 0.75 else -1.

            # self.delay_bias += -self.delay_bias / 1000 + (self.action[0] < 0.5)

        # if self.lambda_ > 0.5:
        #     if self.eps < 0.5:
        #         self.action = 0
        #         return self.action
        #     self.action = 1
        #     return self.action
        # if self.eps < 0.5:
        #     self.action = 2
        #     return self.action

        # self.action = 3
        self.action_str = f"{self.action[0]:.2f}, {self.action[1]:.2f}"
        return self.action

    def render(self):

        plt.plot(self.obs1_rec, label=\
                      f"obs1 {np.abs(self.obs1_rec[-1]):.5f}\n" +\
                 f"   $\\theta=${self.threshold1}")
        # plt.axhline(0, color="black", alpha=0.2)
        # plt.axhline(1, color="black", alpha=0.2)
        plt.plot(self.eps_rec, label="eps")
        # plt.plot(self.lambda_rec, label="lambda")
        # plt.ylim(-0.1, 1.1)
        t = len(self.eps_rec)
        plt.xlim(max(0, t-1000), t)
        plt.legend(loc="lower right")
        # plt.title(f"action={self.action}, obs2={self.obs2:.2f}, {self.lambda_eq - self.lambda_:.2f} {self.obs2 < self.threshold2} [{self.threshold2}]")
        plt.title(f"Heuristic - action={self.action_str}\n$\\epsilon=${self.eps:.2f}" +\
            f" $\\lambda=${self.lambda_:.2f}")



class HeuristicIII:
    def __init__(self, params: np.ndarray=None):

        self.v = 0
        self.tau = 10

        self.eps = 0
        self.eq_eps = 0

        self.lambda_ = 0
        self.eq_lambda = 0

        self.delay = 10
        self.t = 0

    def __call__(self, distance: float=None):

        # delay period
        self.t += 1
        if self.t % self.delay != 0:
            return self.eps, self.lambda_

        # target period
        if distance is not None:
            self.v += (distance - self.v) / self.tau

            # stuck
            if abs(distance - self.v) < 0.0001:
                self.eq_eps = 0
            else:
                self.eq_eps = 0.95

        # exploration period
        else:
            self.v -= self.v / self.tau
            self.lambda_ = np.cos(self.t*0.0001) + 1

        return self.eps, self.lambda_



    def __repr__(self):
        return "HeuristicIII"




""" Run agent """



def run_random_agent(agent: RandomAgent, env: object,
                     reward: Reward,
                     heuristic: object=None,
                     duration: int=1000,
                     policy: object=None,
                     **kwargs):

    # Store positions
    positions = np.zeros((duration, 2))

    # plot
    plot = kwargs.get('plot', False)
    interval = kwargs.get('interval', 100)
    animate = kwargs.get('animate', False)
    disable = kwargs.get('disable', False)
    use_clf = kwargs.get('use_clf', False)

    if policy is not None:
        logger(f"Using policy {policy}")

    if plot:

        # --- 2 plots figure
        # fig = plt.figure(figsize=(18, 8))
        # make an axis on one column and two axes on the other
        # as a 1x2 grid
        # gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])
        # gs = fig.add_gridspec(1, 2)
        # ax = fig.add_subplot(gs[0, 0])
        # ax2 = fig.add_subplot(gs[0, 1])

        # fig.set_aspect('auto')
        # fig.tight_layout()

        # --- 1 plot figure
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        if animate:
            animation_maker = AnimationMaker(fps=15,
                                             use_logger=True,
                                             path=MEDIA_PATH)

        env.draw(ax)
        agent.draw(ax)

    rec_size = 700
    umax = []
    achen = []
    R = 0
    eq_da = 0.

    # colors
    # Define the colors for each value
    colors = ['white', 'black', 'red']

    # Create a ListedColormap
    custom_cmap = ListedColormap(colors)

    logger.debug("running agent")

    # Run agent
    for t in tqdm(range(duration), desc="Running agent",
                  disable=disable or use_clf):

        # if t == 50_000:
        #     del env.walls[-1]
        #     logger.debug("% WALL REMOVED %")

        # get target position
        reward_pos = reward(t=t)

        if policy is not None:
            eq_da = policy(pos=agent.position)

        # observation
        if reward_pos is None:
            distance = None
        else:
            distance = np.linalg.norm(reward_pos - agent.position)

        # action
        if heuristic:
            action = heuristic(distance=distance)
        else:
            action = None

        # update agent
        positions[t] = agent(action=action,
                             trg_pos=reward_pos,
                             eq_da=eq_da)

        # check reward
        if reward.check(agent.position):
            logger(f"Reached goal at t={t}")
            R = 1
            break

        # --- record & plot ---
        umax += [agent._pcnn.u.max()]
        achen += [int(agent._pcnn._ach_enabled)]

        if len(umax) > rec_size:
            del umax[0]
            del achen[0]

        if plot and t % interval == 0:# and t > 0.996*reward.onset:
            if not use_clf:
                # ax2.clear()
                ax.clear()
            else:
                clf()
            # ax2.imshow(agent.activity_record,
            #            cmap="Greys",
            #            vmin=0., vmax=1.,
            #            aspect="auto", interpolation="nearest")
            # ax2.set_yticks([])
            # ax2.set_xlabel("time", fontsize=15)
            # ax2.set_ylabel("neurons", fontsize=15)
            # ax2.set_aspect("auto")
            # ax2.set_title(f"Current representation",
            # ax2.set_title(f"{agent.heuristic}",
            #               fontsize=17)

            #
            # ax3.clear()
            # ax3.imshow(agent.trg_representation.reshape(1, -1),
            #            cmap="Greys", vmin=0.,
            #            aspect="auto", interpolation="nearest")
            # ax3.set_yticks([])
            # ax3.set_xlabel("neurons")
            # ax3.set_title("Target representation")

            #
            # title=f"t={t/1000:.1f}, $N_{{PC}}=$" + \
            #     f"{len(agent._pcnn)}" + \
            #     f", $\\epsilon={agent.epsilon:.2f}$",

            # ex.clear()
            # ex.imshow(agent._pcnn._Wff, vmin=0)

            title=f"t={t/100:.1f}s $\\mathbf{{a}}=$" + \
                f"{np.around(action, 2).tolist()}" + \
                f" $Eq_{{DA}}=${eq_da:.2f} {agent._pcnn._DA:.2f}"

            update_plot(ax, agent, env,
                        trg=reward_pos,
                        policy=policy,
                        pause=0.0001,
                        title=title,
                        show_pcnn=True,
                        use_clf=use_clf)


            # animation
            if animate:
                animation_maker.add_frame(fig)

        if t % kwargs.get("uprec", 1000) == 0:
            agent._update_connections()

    if animate:
        animation_maker.make_animation(
            name=f"rlroom_{time.strftime('%H%M%S')}")
        logger(f"animation saved at {MEDIA_PATH}")
        animation_maker.play_animation(return_Image=False)

    if plot:
        plt.show()

    if kwargs.get("save", False):
        name = f"roaming_{time.strftime('%H%M%S')}"
        fig.savefig(f"{MEDIA_PATH}/{name}.png",
                    dpi=400)
        logger(f"plot saved as `{name}`")

    return agent, R



def main_random_agent(seed: int=None, **kwargs):

    if seed is not None:
        np.random.seed(seed)

    onset = 100_000
    duration = kwargs.get("duration", 1_000)

    # Environment
    env = ev.make_room(name="square2", thickness=4.)
    # heuristic = Heuristic(params=[0.01, 2.8, 0.03, 1., 1.])
    heuristic = HeuristicIII()

    # PCNN
    params = {
        "N": 100,
        "Nj": 13**2,
        "tau": 10.0,
        "alpha": 12_000.,
        "beta": 20.0, # 20.0
        "lr": 0.005,
        "threshold": 0.2,#kwargs.get("threshold", 0.09),
        "ach_threshold": 0.7,#args.ach_threshold,
        "da_threshold": 0.5,
        "tau_ach": 200.,  # 2.
        "eq_ach": 1.,
        "tau_da": 10.,#args.tau,  # 2.
        "eq_da": 0.,
        "epsilon": 0.7,
        "rec_epsilon": 0.1,# kwargs.get("rec_epsilon", 0.1),
    }

    pcnn_model = pw.PCNNlayer(pcnn_params=params,
                              sigma=0.01,
                              calc_graph_enable=False)
    pcnn_model._calc_recurrent_enable = False
    logger(f"PCNN: {pcnn_model}")
    agent = RandomAgent(position=np.array([0.8, 0.2]),
                        room=env,
                        pcnn=pcnn_model,
                        policy="drift",
                        speed=0.001,
                        turn_rate=0.18,
                        radius=0.02,
                        epsilon=0.0)

    policy = Policy(eq_da=1.,
                    trg=np.array([0.85, 0.3]),
                    threshold=0.2,
                    startime=1e1)

    # --- pre-training ---

    duration = kwargs.get("duration", 1_000)
    reward = Reward(start_pos=np.array([0.4, 0.5]),
                    onset=duration+1,  # inactive
                    radius=0.03)
    agent, _, = run_random_agent(agent=agent, env=env, reward=reward,
                                 heuristic=heuristic,
                                 policy=policy,
                                 duration=duration,
                                 plot=True,
                                 interval=4000,
                                 uprec=5000,
                                 use_clf=False,
                                 animate=False,
                                 save=kwargs.get("save", False))
    logger("done")



def multiple_runs(agent: RandomAgent, env: object,
                  num_runs: int=10, **kwargs):


    # _, ax = plt.subplots(1, 1, figsize=(8, 8))
    # agent._pcnn.render(ax=ax)
    # plt.show()


    # --- multiple runs ---

    # variable: threshold
    variable_1 = np.linspace(0., 0.7, num_runs, endpoint=True)
    name_1 = "alpha"

    # variable: beta
    variable_2 = np.linspace(-5., 5., num_runs, endpoint=True)
    name_2 = "beta"
    agent.is_training = False

    results = np.zeros((num_runs, num_runs))
    distances = np.zeros((num_runs, num_runs))

    for i in tqdm(range(num_runs), desc="var 1"):
        for j in tqdm(range(num_runs), desc="var 2"):

            # init
            agent.reset(position=np.array([0.4, 0.3]))
            heuristic = Heuristic(params=[0.01,
                                          variable_2[j],
                                          variable_1[i]])
            agent.heuristic = heuristic
            reward = Reward(start_pos=np.array([0.7, 0.8]),
                            onset=1,
                            radius=0.05)

            # run
            _, R = run_random_agent(agent, env, reward,
                                    duration=kwargs.get('duration', 1_000),
                                    plot=False,
                                    interval=20,
                                    uprec=5000,
                                    disable=True)
            results[i, j] = R
            distances[i, j] = agent.distance


    print(results)
    plt.subplot(1, 2, 1)
    plt.imshow(results, cmap="Greys", vmin=0., vmax=1.)
    plt.xlabel(f"{name_1}")
    plt.xticks(range(num_runs), np.around(variable_1, 2))
    plt.ylabel(f"{name_2}")
    plt.yticks(range(num_runs), np.around(variable_2, 2))
    plt.title("Success rate")

    plt.subplot(1, 2, 2)
    plt.imshow(distances, cmap="Greys", vmin=0., vmax=1.)
    plt.colorbar()
    plt.xlabel(f"{name_1}")
    plt.xticks(range(num_runs), np.around(variable_1, 2))
    plt.ylabel(f"{name_2}")
    plt.yticks(range(num_runs), np.around(variable_2, 2))
    plt.title("Distance to target")

    plt.show()



def update_plot(ax: plt.Axes, agent: RandomAgent,
                env: object, **kwargs):

    if not kwargs.get('use_clf', False):
        ax.clear()

    if kwargs.get('trg', None) is not None:
        ax.scatter(*kwargs.get('trg'), c='red', s=100, marker='x')

    if kwargs.get("policy", None) is not None:
        kwargs.get("policy", None).draw(ax=ax,
                                        alpha=0.1)

    env.draw(ax=ax, alpha=1.)
    agent.draw(ax,
               show_pcnn=kwargs.get('show_pcnn', False),
               edge_alpha=0.4, alpha=0.7,
               tr_alpha=0.1,
               use_a=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_aspect('equal')
    ax.set_title(f"{kwargs.get('title', 'Random Agent')}",
                 fontsize=17)
    ax.legend(loc="lower left")
    plt.pause(kwargs.get('pause', 0.01))



def run_simple_episode(agent: RandomAgent,
                       env: object,
                       reward: Reward,
                       duration: int=1000, **kwargs):


    # init
    duration = env.duration
    render = kwargs.get("render", True)
    obs, _ = env.reset()
    use_clf = kwargs.get("use_clf", False)
    use_a = kwargs.get("use_a", False)

    # Run agent
    for t in tqdm(range(duration), desc="Running agent"):

        # agent step
        action = agent.predict(obs)

        # environment step
        obs, reward, done, truncated, info = env.step(action=action)

        if render and t % kwargs.get("tper", 10) == 0:
            env.render(pause=0.001, use_clf=use_clf, t=t,
                       agent=agent, use_a=use_a)

        # exit
        if done and reward > 0:
            logger("reward!")
            break

        if done:
            logger("<done>")
            break

    # env.close()
    if render:
        plt.show()



""" Env """


class CampoBlue(gym.Env):

    def __init__(self, reward: object, agent: object,
                 observer: object, duration: int=1000):

        self.reward = reward
        self.agent = agent
        self.room = agent._room
        self.observer = observer

        self.duration = duration
        self.t = 0

        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(2,))
        self.action_space = spaces.Discrete(4)

        self.name = "campoblu"

    def step(self, action: int):

        self.t += 1

        # step the reward
        trg_pos = self.reward(t=self.t)

        # step the agent
        _ = self.agent(trg_pos=trg_pos,
                       action=action)

        # make observation
        obs1, obs2 = self.observer(
            distance_ext=self.agent.distance,
            representation=self.agent.curr_representation.flatten().copy())
        obs = np.array([obs1, obs2])

        # make reward
        target_reward = self.reward.check(self.agent.position)
        local_reward = self.observer.make_reward()
        reward = local_reward if not target_reward else 2

        # logger.debug(f"{obs=}")

        # info
        info = {"distance": self.agent.distance}

        # exit condition
        done = self.t >= self.duration or reward > 1  # only the target reward

        return obs, reward, done, False, info

    def reset(self, episde_meta_info: str=None,
              seed: int=None, **kwargs) -> tuple:

        if seed is not None:
            super().reset(seed=seed)

        self.agent.reset(position=np.random.uniform(0.1, 0.9, 2))

        return np.ones(2), {}

    def render(self, mode='human', pause=0.01,
               use_clf: bool=False, t: int=0,
               agent: object=None, use_a=False):

        if use_clf:
            clf()

        if agent is not None:
            if hasattr(agent, "render"):
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 2)
                agent.render()
                plt.subplot(1, 2, 1)

        plt.scatter(*self.reward.position, c='red', s=100, marker='x')
        self.room.draw()
        self.agent.draw(show_pcnn=True, use_a=use_a)
        if self.agent.new_position is not None:
            plt.scatter(*self.agent.new_position, c='orange', s=150, marker='v')
            nextpos = np.around(self.agent.new_position, 2)
        else:
            nextpos = None
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks([])
        plt.yticks([])
        # ax.set_aspect('equal')
        title = f"{t=} | $\\epsilon=${self.agent.epsilon} $\\lambda=${self.agent.param1} | " + \
            f"d={self.agent.distance:.3f}"
        # plt.xlabel(f"$\\epsilon_e=${agent.eps:.2f}, $\\lambda_e=${agent.lambda_:.2f} npos={nextpos}")
        plt.title(title,
                  fontsize=17)
        plt.pause(pause)



def train_simple_agent(args):


    # --- instantiate the agent ---
    # PCNN
    params = {
        "N": 180,
        "Nj": 13**2,
        "tau": 10.0,
        "alpha": 16_000.,
        "beta": 20.0, # 20.0
        "lr": 0.2,
        "threshold": 0.7,#kwargs.get("threshold", 0.09),
        "ach_threshold": 0.7,#args.ach_threshold,
        "da_threshold": 0.7,
        "tau_ach": 200.,  # 2.
        "eq_ach": 1.,
        "tau_da": 50.,#args.tau,  # 2.
        "eq_da": 0.,
        "epsilon": 0.3,
        "rec_epsilon": 0.01,# kwargs.get("rec_epsilon", 0.1),
    }

    pcnn_model = pw.PCNNlayer(pcnn_params=params,
                              sigma=0.01,
                              calc_graph_enable=False)
    pcnn_model._calc_recurrent_enable = False

    # load model
    if not args.new_pc:
        # filename = "{}/pcnn_{}".format(ev.SAVEPATH, args.env)
        filename = f"{ev.BASEPATH}/cache/pcnn_param.json"
        pcnn_model.load_params(filename)
        logger.info("PCNN model loaded")

    # room
    room = ev.make_room(name="square1", thickness=4.)
    logger(room)

    logger(f"PCNN: {pcnn_model}")
    agent = RandomAgent(position=np.array([0.4, 0.4]),
                        room=room,
                        pcnn=pcnn_model,
                        heuristic=None,
                        policy="drift",
                        speed=0.001,
                        turn_rate=0.1,
                        radius=0.02,
                        epsilon=0.0)

    # --- pre-training ---

    if args.new_pc:
        reward = Reward(start_pos=np.array([0.8, 0.8]),
                        onset=args.pre_duration+1,  # inactive
                        radius=0.03)
        agent, _, = run_random_agent(agent, room, reward,
                                     duration=args.pre_duration,
                                     plot=False,
                                     interval=7,
                                     uprec=1_000,
                                     animate=False)
        agent._update_connections()
        logger("Pre-training done")

    # --- policy training ---
    agent.reset()
    agent.is_training = False

    # new reward
    reward = Reward(start_pos=np.array([0.8, 0.8]),
                    onset=1,
                    radius=0.03)

    # observer
    observer = Observer(shape_ext=(1,), shape_repr=(pcnn_model.N,),
                        tau=10.0)

    # make env
    env = CampoBlue(reward=reward,
                    agent=agent,
                    observer=observer)


    logger(f"duration: {args.duration}")
    logger(f"epochs: {args.epochs}")

    if args.agent in ["ppo", "a2c", "td3", "ddpg", "dqn"]:
        from stable_baselines3 import PPO, A2C, TD3, DDPG, DQN

    # --- train the agent ---
    # Train the agent for 10,000 timesteps
    if not args.load:

        # Create the PPO agent
        if args.agent == "ppo":
            agent = PPO("MlpPolicy", env, verbose=1)
        elif args.agent == "a2c":
            agent = A2C("MlpPolicy", env, verbose=1)
        elif args.agent == "td3":
            agent = TD3("MlpPolicy", env, verbose=1)
        elif args.agent == "ddpg":
            agent = DDPG("MlpPolicy", env, verbose=1)
        elif args.agent == "dqn":
            agent = DQN("MlpPolicy", env, verbose=1)
        elif args.agent == "dummy":
            agent = LocalDummy()
        elif args.agent == "hII":
            agent = HeuristicII()
        else:
            raise ValueError("agent not found")

        logger(agent)

        if args.train:
            logger("training started [save={}]".format(args.save))
            agent.learn(total_timesteps=args.epochs,
                        log_interval=1_000,
                        progress_bar=True)
            logger.info("[{} training done]".format(args.agent))

            # save the model
            if args.save:
                filename = "{}/{}_{}".format(ev.SAVEPATH,
                                             args.agent,
                                             env.name)
                agent.save("{}".format(filename))
                logger.info("[{} model saved]".format(filename))

    # load the model
    else:
        filename = "{}/{}_{}".format(ev.SAVEPATH, args.agent,
                                     env.name)
        if args.agent == "ppo":
            agent = PPO.load("{}".format(filename))
        elif args.agent == "a2c":
            agent = A2C.load("{}".format(filename))
        elif args.agent == "td3":
            agent = TD3.load("{}".format(filename))
        elif args.agent == "ddpg":
            agent = DDPG.load("{}".format(filename))
        elif args.agent == "dqn":
            agent = DQN.load("{}".format(filename))
        else:
            raise ValueError("agent not found")

        logger.info("[{} model loaded]".format(args.agent))

    # --- run an episode ---
    if args.test:
        logger("testing agent...")
        _ = run_simple_episode(agent=agent,
                               env=env,
                               reward=reward,
                               tper=50)



def simple_run():

    np.random.seed(91)

    onset = 100_000
    duration = 105_000

    # Environment
    env = ev.make_room(name="square1", thickness=4.)
    heuristic = Heuristic(params=[0.01, 2.8, 0.03, 1., 1.])

    # PCNN
    params = {
        "N": 180,
        "Nj": 13**2,
        "tau": 10.0,
        "alpha": 16_000.,
        "beta": 20.0, # 20.0
        "lr": 0.2,
        "threshold": 0.7,#kwargs.get("threshold", 0.09),
        "ach_threshold": 0.7,#args.ach_threshold,
        "da_threshold": 0.7,
        "tau_ach": 200.,  # 2.
        "eq_ach": 1.,
        "tau_da": 50.,#args.tau,  # 2.
        "eq_da": 0.,
        "epsilon": 0.3,
        "rec_epsilon": 0.01,# kwargs.get("rec_epsilon", 0.1),
    }

    pcnn_model = pw.PCNNlayer(pcnn_params=params,
                              sigma=0.01,
                              calc_graph_enable=False)
    pcnn_model._calc_recurrent_enable = False
    logger(f"PCNN: {pcnn_model}")
    agent = RandomAgent(position=np.array([0.4, 0.4]),
                        room=env,
                        pcnn=pcnn_model,
                        heuristic=heuristic,
                        policy="drift",
                        speed=0.001,
                        turn_rate=0.1,
                        radius=0.02,
                        epsilon=0.0)

    # --- pre-training ---

    duration = 40_000
    reward = Reward(start_pos=np.array([0.8, 0.8]),
                    onset=duration+1,  # inactive
                    radius=0.03)
    agent, _, = run_random_agent(agent, env, reward,
                                 duration=duration,
                                 plot=False,
                                 interval=7,
                                 uprec=1_000,
                                 animate=False)
    agent._update_connections()
    logger("Pre-training done")

    # --- Run agent ---
    run = 1

    # single run
    if run == 1:
        reward = Reward(start_pos=np.array([0.8, 0.8]),
                        radius=0.05,
                        onset=200)
        agent.reset(position=np.array([0.1, 0.3]))
        agent.is_training = False

        duration = 5_000
        agent, _, = run_random_agent(agent, env, reward,
                                     duration=duration,
                                     plot=True,
                                     interval=20,
                                     uprec=duration+1,
                                     animate=True)
    # Multiple runs
    elif run == 2:
        multiple_runs(agent, env, num_runs=10)


    logger("Done")



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=100_000,
                        help="number of epochs")
    parser.add_argument("--duration", type=int, default=1_000,
                        help="episode duration in steps")
    parser.add_argument("--pre_duration", type=int, default=40_000,
                        help="episode duration in steps of the pre-training")
    parser.add_argument("--agent", type=str, default="ppo",
                        help="agent type [ppo, a2c, td3, dummy]")
    parser.add_argument("--main", type=int, default=1,
                        help="which main to run: 1 [simple_run], 2 [train_simple_agent]")
    parser.add_argument("--new_pc", action="store_true",
                        help="use an untrained PCNN model")
    args = parser.parse_args()



    if args.main == 1:
        simple_run()
    elif args.main == 2:
        train_simple_agent(args)
    elif args.main == 3:
        main_random_agent(seed=args.seed,
                          duration=args.duration,
                          save=args.save)
    else:
        raise ValueError("Unknown main")

