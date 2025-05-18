import numpy as np
import math
import random
import heapq
import os, sys


sys.path.append(os.path.join(os.getcwd().split("PCNN")[0], "PCNN/src"))
import core.build.pclib as pclib
from utils import setup_logger

logger = setup_logger('PCLIB', level=-2, is_debugging=False)

MAX_ATTEMPTS = 5
MIN_PC_NUMBER = 5
ATTEMPT_PAUSE = 20
TOP_DA_IDX = 5
OVERSHOOT_DURATION = 30


""" FUNCTIONS """


def spatial_shortest_path(connectivity_matrix: np.ndarray,
                          node_coordinates: np.ndarray,
                          node_weights: np.ndarray,
                          start_node: int, end_node: int):
    """
    Calculates the shortest spatial path between two nodes using Dijkstra's algorithm.
    
    Args:
        connectivity_matrix: NxN numpy array with 1 for connected nodes and 0 otherwise.
        node_coordinates: Nx2 numpy array of (x, y) coordinates for each node.
        node_weights: N-length numpy array with optional penalties per node.
        start_node: index of the start node.
        end_node: index of the end node.

    Returns:
        A list of node indices forming the shortest path.
    """

    num_nodes = connectivity_matrix.shape[0]
    distances = [float('inf')] * num_nodes
    parent = [-1] * num_nodes
    finalized = [False] * num_nodes

    # Min-heap priority queue (distance, node)
    pq = []

    distances[start_node] = 0.0
    heapq.heappush(pq, (0.0, start_node))

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        if finalized[current_node] or current_dist > distances[current_node]:
            continue

        finalized[current_node] = True

        if current_node == end_node:
            break

        for neighbor in range(num_nodes):
            if connectivity_matrix[current_node, neighbor] == 1 and not finalized[neighbor]:
                if node_weights[neighbor] < -1000.0:
                    continue
                else:
                    dx = node_coordinates[current_node, 0] - node_coordinates[neighbor, 0]
                    dy = node_coordinates[current_node, 1] - node_coordinates[neighbor, 1]
                    edge_distance = np.sqrt(dx * dx + dy * dy)

                    # New distance could also include node_weights[neighbor] if desired
                    new_distance = distances[current_node] + edge_distance

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parent[neighbor] = current_node
                    heapq.heappush(pq, (new_distance, neighbor))

    # Reconstruct the path
    path = []
    current = end_node

    if distances[end_node] == float('inf'):
        return []

    while current != -1:
        path.append(current)
        current = parent[current]

    path.reverse()

    if not path or path[0] != start_node:
        return []

    return path

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return math.sqrt((v1[0]-v2[0])**2 + (v1[1] - v2[1])**2)


""" NEURONS """


class DensityPolicy2:

    def __init__(self, rwd_weight: float, rwd_sigma: float,
                 col_weight: float, col_sigma: float,
                 rwd_field_mod: float,
                 col_field_mod: float):

        self.rwd_weight = rwd_weight
        self.rwd_sigma = rwd_sigma
        self.col_weight = col_weight
        self.col_sigma = col_sigma

        self.rwd_drive = 0.0
        self.col_drive = 0.0

        self.rwd_field_mod = rwd_field_mod
        self.col_field_mod = col_field_mod

    def __call__(self, space: object, circuits: object, goalmd: object,
             displacement: np.ndarray, curr_da: float, curr_bnd: float,
             reward: float, collision: float):

        # --- reward mod
        if reward > 0.1 and circuits.get_da_leaky_v() > 0.01:

            self.rwd_drive = self.rwd_weight * curr_da

            # +density
            space.remap(circuits.get_da_weights(), displacement,
                        self.rwd_sigma, self.rwd_drive)
            # +gain
            space.modulate_gain(self.rwd_field_mod)

        # --- collision mod
        elif collision > 0.1:

            self.col_drive = self.col_weight * curr_bnd
            neg_disp = [-displacement[0], -displacement[1]]

            # +density
            print(dir(space))
            space.remap(circuits.get_bnd_weights(), neg_disp,
                        self.col_sigma, self.col_drive)
            # +gain
            space.modulate_gain(self.col_field_mod)

    def remap_space(self, block_weights: np.ndarray, space: object, goalmd: object,
                    displacement: np.ndarray, sigma: float, drive: float):

        plan_idxs = goalmd.get_plan_idxs()

        if len(plan_idxs) < 1:
            return

        curr_positions = space.get_position()
        centers = space.get_centers()
        prev_center = centers[plan_idxs[0]]

        for i in range(1, len(plan_idxs)):
            idx = plan_idxs[i]
            curr_center = centers[idx]

            if curr_center[0] < -900.0 or block_weights[i] > 0.0:
                continue

            delta = [
                curr_center[0] - prev_center[0],
                curr_center[1] - prev_center[1]
            ]

            dist_to_pos = math.sqrt(
                (curr_center[0] - curr_positions[0]) ** 2 +
                (curr_center[1] - curr_positions[1]) ** 2
            )
            dist = math.exp(- (dist_to_pos ** 2) / sigma)

            magnitude = dist * drive
            space.single_remap(idx, delta, magnitude)

            prev_center = curr_center

    def __str__(self):
        return "DensityPolicy"

    def __repr__(self):
        return "DensityPolicy"

    def get_rwd_mod(self):
        return self.rwd_drive

    def get_col_mod(self):
        return self.col_drive


""" BEHAVIOUR """


class Plan:

    def __init__(self, space: object, speed: float):

        self._space = space

        # parameters
        self._speed = speed

        # variables
        self._plan_idxs = []
        self._curr_positions = np.zeros(2)
        self._next_position = np.zeros(2)

        self._size = -1
        self._counter = 0
        self._curr_idx = -1
        self._next_idx = -1
        self._trg_idx = -1

        self._action = np.zeros(2)

        # end of plan wherabout
        self._wduration = OVERSHOOT_DURATION
        self._wt = 0
        self.is_overshooting = False

    def step_plan(self) -> tuple:

        distance = self._calculate_distance()

        # same next position
        # if distance > 0.01 and self._counter > 0:
        if self._curr_idx != self._next_idx:
            logger(f"\tdistance={distance:0.3f} counter={self._counter} " + \
                   f"next={self._next_idx}")
            return self._make_velocity(), True

        # check: end of the plan
        if self._counter > self._size or self._curr_idx == self._trg_idx or self.is_overshooting:
            logger(f"[Plan] end | counter={self._counter > self._size}" + \
                   f" curr={self._curr_idx} trg={self._trg_idx}")

            if self._wt < self._wduration:
                self._wt += 1
                logger.debug(f"[Plan] ... overshoot [{self._wt}|{self._wduration}]")
                self.is_overshooting = True
                return self._action, True

            self.reset()
            return np.zeros(2), False

        # retrieve next position
        self._counter += 1
        try:
            self._next_idx = self._plan_idxs[self._counter]
            self._next_position = np.array(self._space.get_centers()[self._next_idx])

            logger(f"[Plan] next idx: {self._next_idx}")
        except IndexError as e:
            print(f"Index Error, {self._counter=} | {self._size=}")
            raise IndexError

        return self._make_velocity(), True

    def _make_velocity(self) -> tuple:

        dx = self._next_position - self._curr_position
        norm = np.sqrt(dx[0]**2 + dx[1]**2)

        if norm < self._speed:
            self._action =  dx
        else:
            self._action = (self._speed * dx) / norm

        return self._action.copy()

    def _calculate_distance(self) -> float:
        self._curr_position = self._space.get_position()
        self._curr_idx = self._space.calculate_closest_index(self._curr_position)
        logger(f"[Plan] curr={self._curr_idx}")
        return euclidean_distance(self._curr_position, self._next_position)

    def set_plan(self, plan_idxs: list):
        self._plan_idxs = plan_idxs
        self._size = len(plan_idxs)
        self._counter = 0
        self._curr_idx = -1
        self._next_idx = plan_idxs[0]
        self._trg_idx = plan_idxs[-1]

        logger(f"[Plan] new plan: {plan_idxs}")

    def is_finished(self) -> bool:
        return self._counter > self._size

    def get_next_positions(self) -> np.ndarray:
        return self._next_position.copy()

    def reset(self):
        self._plan_idxs = []
        self._size = -1
        self._counter = 0
        self._curr_idxs = -1
        self._next_idx = -1
        self._trg_idx = -1
        self._wt = 0
        self.is_overshooting = False


class RewardObject:

    def __init__(self, min_weight_value: float=0.01):
        self.trg_idx = -1
        self.trg_value = 0.0
        self.min_weight_value = min_weight_value

    def update(self, da_weights, space, trigger=True):
        # exit: no trigger
        if not trigger:
            return -1

            # exit: weights not strong enough
            if np.max(da_weights) < self.min_weight_value:
                logger(f"weak weights | {max(da_weights):.3f}" + \
                    f" < {self.min_weight_value}")
                return -1

        # --- update the target representation ---
        # method 1: take the center of mass
        self.trg_idx = self._converge_to_trg_index(da_weights, space)

        # try method 2
        if self.trg_idx < 0:
            # method 2: take the argmax of the weights
            max_index = np.argmax(da_weights)
            self.trg_idx = int(max_index)

        # exit: no trg index
        if self.trg_idx < 0:
            logger("[RwO] no trg index")
            return -1

        # exit: low trg value
        if da_weights[self.trg_idx] < 0.00001:
            logger(f"[RwO] low trg value : {da_weights[self.trg_idx]:.5f} max={da_weights.max():.5f} argmax={da_weights.argmax()}")
            return -1

        # update the target value
        self.trg_value = da_weights[self.trg_idx]

        return self.trg_idx

    def _converge_to_trg_index(self, da_weights, space):

        # weights for the centers
        cx, cy = 0.0, 0.0
        sum_weights = np.sum(da_weights)

        if sum_weights == 0.0:
            # logger("[-] sum is zero")
            return -1

        # method 2 : argmax
        # return da_weights.argmax()

        # top 5 DA cells
        top_idx = np.argsort(da_weights)[-TOP_DA_IDX:]

        # method 1 : weighted center
        centers = space.get_centers()
        for idx in top_idx:
            cx += da_weights[idx] * centers[idx, 0]
            cy += da_weights[idx] * centers[idx, 1]

        # centers of mass
        cx /= sum_weights
        cy /= sum_weights

        # get closest center
        closest_idx = space.calculate_closest_index([cx, cy])
        return closest_idx


class GoalModule:

    def __init__(self, space: object, circuits: object, speed: float):

        # External components
        self.space = space
        self.circuits = circuits

        # Internal components
        self.plan = Plan(space, speed)

        # Variables
        self.fine_tuning_time = 0
        self.final_idx = -1

    def make_plan(self, space, space_weights, trg_idx, curr_idx=-1):

        # Current index and value
        if curr_idx == -1:
            curr_idx = space.calculate_closest_index(space.get_position())

        # Check: current position at the boundary
        if space_weights[curr_idx] < -1000.0:
            logger("curr idx has bad value")
            return [], False

        # Make plan path
        plan_idxs = spatial_shortest_path(
            space.get_connectivity(),
            space.get_centers(),
            space_weights,
            curr_idx, trg_idx
        )

        # Check if the plan is valid, i.e., size > 1
        if len(plan_idxs) < 1:
            logger(f"[Goal] plan too short | {plan_idxs}")
            return [], False

        logger("[Goal] new plan ############################### " + \
            "return bnd.get_weights()(idx); :")
        logger(" ")

        # Check bnd value of each index
        for i in range(len(plan_idxs)):
            logger(f"[goal] idx={plan_idxs[i]} | " + \
                f"value={space_weights[plan_idxs[i]]} | " + \
                f"bnd v={self.circuits.get_bnd_value(plan_idxs[i])}")

        return plan_idxs, True

    def update(self, trg_idx: int, goal_directed: bool):
        # -- Propose a coarse plan --

        # Plan from the current position
        results = self.make_plan(
            self.space,
            self.circuits.make_value_mask(goal_directed),
            trg_idx
        )

        # Check: failed coarse planning
        if not results[1]:  # Second element is the success flag
            logger("[Goal] failed planning")
            return False

        # Extract the last index of the coarse plan
        curr_idx = self.space.calculate_closest_index([
            self.space.get_centers()[results[0][-1], 0],
            self.space.get_centers()[results[0][-1], 1]
        ])

        # Record
        self.plan.set_plan(results[0])

        return True

    def step_plan(self, obstacle: bool=False):

        # Exit: active
        if self.plan.is_finished():
            logger("[Goal] planned finished")
            return [0.0, 0.0], False

        local_velocity = [0.0, 0.0]

        # -- Coarse plan
        if not obstacle:
            action, flag = self.plan.step_plan()

            logger(f"[Goal] progress={flag}")

            # Exit: coarse action
            if flag and action.sum()**2 > 0:
                logger(f"[Goal] action {action} [{flag=}] /{action.sum()=}")
                return action, flag

        logger(f"[Goal] obstacle={obstacle}")
        # logger(f"[Goal] fine_tuning={self.is_tuning}")
        return [0.0, 0.0], False

    def reset(self):
        self.plan.reset()

    def is_active(self):
        return not self.plan.is_finished() or self.plan.is_overshooting

    def get_plan_idxs(self):
        return self.plan._plan_idxs


class ExplorationModule:

    def __init__(self, speed: float, circuits: object, space: object,
                 action_delay: float=1.0, edge_route_interval: int=100):

        # external components
        self.circuits = circuits
        self.space = space

        # parameters
        self.speed = speed
        self.action_delay = action_delay
        self.open_threshold = 2.0  # radians
        self.rejected_indexes = np.zeros(space.get_size(), dtype=np.float32)
        self.sparsity_threshold = 5

        # plan
        self.t = 0
        self.action = [0.1, 0.1]
        self.edge_idx = -1

        # alternate between random walk and edge exploration
        self.edge_route_time = 0
        self.edge_route_interval = edge_route_interval

        self.new_plan = True

    def __call__(self, directive: str, random_walk_directive: bool=False):

        # check ongoing plan
        action, plan_status = self._step_random_plan()

        # new plan | directive of plan end
        if directive == "new" or plan_status:

            self.new_plan = True
            # plan_flag = self._make_plan(rejected_idx=rejected_idx)

            if self.edge_route_time > self.edge_route_interval and not \
                random_walk_directive:
                logger.debug(f"edge route time < {self.edge_route_time}")
                self.edge_route_time = 0
                return [-1.0, -1.0], True
            else:
                self._random_action_plan()
                action, _ = self._step_random_plan()
                self.edge_route_time += 1
                return action, False

        self.new_plan = False
        return action, False

    def _random_action_plan(self):

        angle = random.uniform(0.0, 2.0 * math.pi)
        self.action = [math.cos(angle) * self.speed, math.sin(angle) * self.speed]
        self.t = 0

    def _step_random_plan(self):

        if self.t > (self.action_delay - 1):
            logger.debug(f"PLAN END | {self.edge_route_time=}")
            return self.action, True

        self.t += 1
        return self.action, False

    def _sample_random_idx(self, num_attempts: int):

        if num_attempts > 20:
            return -1

        idx = random.randint(0, self.space.__len__() - 1)

        if idx == 0:
            return -1

        if self.rejected_indexes[idx] > 0.0:
            return self._sample_random_idx(num_attempts + 1)

        if self.circuits.get_bnd_weights()[idx] < 0.01 or \
                self.space.get_trace_value(idx) > 0.001:
            return idx

        return self._sample_random_idx(num_attempts + 1)

    def is_edge_position(self):
        curr_idx = self.space.calculate_closest_index(self.space.get_position())
        neighbourhood_degree = self.space.get_neighbourhood_node_degree(curr_idx)
        return neighbourhood_degree < self.sparsity_threshold

    def str(self):
        return "ExplorationModule"

    def repr(self):
        return "ExplorationModule"

    def confirm_edge_walk(self):
        self.edge_route_time = 0

    def reset_rejected_indexes(self):
        self.rejected_indexes = np.zeros(self.space.get_size(), dtype=np.float32)

    def get_edge_representation(self):
        edge_rep = np.zeros(self.space.get_size(), dtype=np.float32)
        if self.edge_idx < 0:
            return edge_rep
        edge_rep[self.edge_idx] = 1.0
        return edge_rep


""" BRAIN """


class Brain:

    def __init__(self,
                 local_scale,
                 N,
                 rec_threshold,
                 speed,
                 min_rep_threshold,
                 gain,
                 offset,
                 threshold,
                 rep_threshold,
                 tau_trace,
                 remap_tag_frequency,
                 lr_da,
                 lr_pred,
                 threshold_da,
                 tau_v_da,
                 lr_bnd,
                 threshold_bnd,
                 tau_v_bnd,
                 tau_ssry,
                 threshold_ssry,
                 threshold_circuit,
                 rwd_weight,
                 rwd_sigma,
                 col_weight,
                 col_sigma,
                 rwd_field_mod,
                 col_field_mod,
                 action_delay,
                 edge_route_interval,
                 forced_duration,
                 min_weight_value,
                 options=[True]*4):

        self.clock = 0
        self.forced_duration = forced_duration

        # Initialize modulations and sensory systems
        self.da = pclib.BaseModulation("DA", N, lr_da, lr_pred,
                                       threshold_da, 1.0,
                                tau_v_da, 0.0, 0.4, 0.1)
        self.bnd = pclib.BaseModulation("BND", N, lr_bnd, 0.0,
                                        threshold_bnd, 1.0,
                                 tau_v_bnd, 0.0, 0.1)
        self.ssry = pclib.StationarySensory(N, tau_ssry,
                                            threshold_ssry, 0.99)
        self.circuits = pclib.Circuits(self.da, self.bnd, threshold_circuit)

        gc_scales = [1., 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.07]
        self.space, self.gcn = pclib.make_space(0.4, gc_scales,
                                      local_scale, N, rec_threshold, speed,
                                      min_rep_threshold, gain,
                                      offset, threshold, rep_threshold,
                                      tau_trace, remap_tag_frequency)

        # Initialize modules
        self.goalmd = GoalModule(self.space, self.circuits, speed)
        self.rwobj = RewardObject(min_weight_value)
        self.dpolicy = pclib.DensityPolicy(rwd_weight, rwd_sigma,
                                           col_weight, col_sigma,
                                           rwd_field_mod, col_field_mod,
                                           options)
        self.expmd = ExplorationModule(speed * 2.0, self.circuits,
                                       self.space, action_delay,
                                       edge_route_interval)

        # failed planning despite goal
        self._t_attempt = 0

        # Variables
        self.curr_representation = None
        self.directive = "new"
        self.trg_plan_end = 0
        self.forced_exploration = -1
        self.tmp_trg_idx = -1
        self.action = [0.0, 0.0]
        self.expmd_res = ([0.0, 0.0], 0)  # (action, index)

        self.state = {
            "representation": [0.] * self.space.get_size(),
            "internal_state": [0., 0.],
            "reward": 0.0,
            "collision": 0.0,
        }

        logger.debug(f"{offset=}")

    def __call__(self, velocity: list, collision: float,
                 reward: float, trigger: bool):

        self.clock += 1

        if reward > 0.0:
            logger("[Brain] reward received")

        # === STATE UPDATE ==============================================


        self.space.update()

        # :space
        u, _ = self.space(velocity)
        self.curr_representation = u

        # if collision > 0.0001:
        #     logger.debug("+collision")
        if len(self.space) == self.space.get_size():
            logger.warning(f"the space is full | N={len(self.space)}")

        # :circuits
        self.state['internal_state'] = self.circuits(u, collision, reward, False)
        self.state['reward'] = reward
        self.state['collision'] = collision

        # :dpolicy fine space || update relative to the previous state
        self.dpolicy(self.space,
                     self.circuits,
                     velocity,
                     self.state['internal_state'][1],
                     self.state['internal_state'][0],
                     self.state['reward'],
                     self.state['collision'])

        if reward > 0.0:
            logger(f"{self.state['internal_state']=}")
            logger(f"da_v={self.circuits.get_da_leaky_v():.3f}")

        # Check: stillness | wrt the fine space
        if self.forced_exploration < self.forced_duration:
            self.forced_exploration += 1
            self.goalmd.reset()
            logger("[Brain] forced exploration")
            return self._explore(collision)
        elif self.ssry(self.curr_representation):
            logger(f"[Brain] forced exploration : v={self.ssry.get_v()}")
            self.forced_exploration = 0
            return self._explore(collision)

        # === GOAL-DIRECTED BEHAVIOUR ====================================

        # --- Current target plan

        # Check: current trg plan
        if self.goalmd.is_active():
            logger("[Brain] active goal plan")
            self.trg_plan_end = 0
            obstacle = collision > 0.0
            progress = self.goalmd.step_plan(collision > 0.0)

            # Keep going
            if progress[1]:
                self.action = progress[0]
                return self._finalize()

            # End or fail -> random walk
            self.forced_exploration = 0
            logger("[Brain] end or fail -> random walk")

        # Time since the last trg plan ended
        self.trg_plan_end += 1

        # --- New target plan: REWARD ------------------------------------

        # :reward object | new reward trg index wrt the fine space
        self.tmp_trg_idx = self.rwobj.update(self.circuits.get_da_weights(),
                                             self.space, trigger)

        # check: attempt pause
        if self.clock - self._t_attempt > ATTEMPT_PAUSE:

            if self.tmp_trg_idx > -1:

                # Check new reward trg plan
                valid_plan = self.goalmd.update(self.tmp_trg_idx, True)

                # [+] reward trg plan
                if valid_plan:
                    logger("[Brain] valid goal plan")
                    self.directive = "trg"
                    progress = self.goalmd.step_plan(collision > 0.0)
                    if progress[1]:
                        self.action = progress[0]
                        return self._finalize()

                    self.forced_exploration = 0

                # [-] failed trg plan
                self._t_attempt = self.clock
                logger("[Brain] invalid goal plan")

        # Fall through to exploration
        return self._explore(collision)

    def _make_prediction(self):

        # Simulate a step
        next_representation = self.space.simulate_one_step(self.action)

        # Make prediction
        self.circuits.make_prediction(next_representation)

    def _explore(self, collision: float):

        # === EXPLORATIVE BEHAVIOUR =======================================

        # Check: collision
        if collision > 0.0:
            self.directive = "new"
            logger.debug('NEW EXPL PLAN')
        else:
            self.directive = "continue"

        # :experience module
        exp_action, edge_route_flag = self.expmd(directive=self.directive,
                                                 random_walk_directive=False)

        # Check: plan to go to the open boundary
        # if flag > -1 or flag == -2:
        if edge_route_flag:
            logger.debug("[Brain] exploration->boundary plan")
            self.action = self._attempt_boundary_plan()
        else:
            self.action = exp_action

        return self._finalize()

    def _finalize(self):

        # Make prediction
        self._make_prediction()
        return self.action

    def _attempt_boundary_plan(self):

        logger.debug(f"attempting boundary plan | {self.expmd.edge_route_time=}")

        expl_idx, sample_flag = self._sample_expl_idx()

        # Attempt for a bunch of times | use 404 as final rejection
        if sample_flag: # only if enough pc are formed

            for i in range(MAX_ATTEMPTS):

                # Attempt a plan
                self.goalmd.reset()
                valid_plan = self.goalmd.update(expl_idx, False)

                # Valid plan
                if valid_plan:

                    self.directive = "trg ob"
                    goal_action, goal_flag = self.goalmd.step_plan(False)

                    # Confirm the edge walk
                    logger.debug(f"valid plan | {goal_flag} | {goal_action=}")
                    if goal_flag:
                        return goal_action

                # Invalid plan -> try again
                expl_idx, _ = self._sample_expl_idx()

        logger.debug("\tfailed")
        # Tried too many times, make a random walk plan instead
        exp_action, _ = self.expmd(directive="new",
                                   random_walk_directive=True)
        logger.debug(f"{exp_action=}")

        return exp_action

    def _sample_expl_idx(self):

        """
        sample an index from the top 30% of pc with
        low trace v2
        """

        if len(self.space) < MIN_PC_NUMBER:
            return -1, False

        traces = self.space.get_trace_v2()
        idxs =  np.argsort(traces[:len(self.space)])
        idx = np.random.choice(idxs[:len(traces)//3])

        return idx, True

    def __str__(self):
        return "Brain"

    def __repr__(self):
        return "Brain"

    def __len__(self):
        return len(self.space)

    def is_space_full(self):
        return len(self.space) == self.space.get_size()

    def get_trg_representation(self):
        trg_representation = np.zeros(self.space.get_size())
        trg_representation[self.rwobj.trg_idx] = 1.0
        return trg_representation

    def get_trg_idx(self):
        return self.rwobj.trg_idx

    def get_leaky_v(self):
        return self.circuits.get_leaky_v()

    def get_representation(self):
        return self.curr_representation

    def get_trg_position(self):
        centers = self.space.get_centers()
        return [centers[self.rwobj.trg_idx, 0],
                centers[self.rwobj.trg_idx, 1]]

    def get_expmd(self):
        return self.expmd

    def get_directive(self):
        return self.directive

    def get_space_size(self):
        return self.space.get_size()

    def get_space_count(self):
        return len(self.space)

    def get_plan_idxs(self):
        return self.goalmd.get_plan_idxs()

    def get_space_position(self):
        return self.space.get_position()

    def get_space_centers(self):
        return self.space.get_centers()

    def get_space_connectivity(self):
        return self.space.get_connectivity()

    def get_space_wrec(self):
        return self.space.get_wrec()

    def get_space_wff(self):
        return self.space.get_wff()

    def get_da_weights(self):
        return self.circuits.get_da_weights()

    def get_bnd_weights(self):
        return self.circuits.get_bnd_weights()

    def get_gain(self):
        return self.space.get_gain()

    def get_gc_network(self):
        return self.gcn

    def make_space_edges(self):
        return self.space.make_edges()

    def get_edge_representation(self):
        return self.expmd.get_edge_representation()

    def reset(self):
        self.goalmd.reset()
        self.circuits.reset()
        self.space.reset()
        logger.debug("[Brain reset]")


""" main """


if __name__ == "__main__":


    brain = Brain(local_scale=0.1,
                  N=100,
                  rec_threshold=1.,
                  speed=1.,
                  min_rep_threshold=1.,
                  gain=10.,
                  offset=1.,
                  threshold=1.,
                  rep_threshold=1.,
                  tau_trace=2.,
                  remap_tag_frequency=0.2,
                  lr_da=0.1,
                  lr_pred=0.1,
                  threshold_da=0.1,
                  tau_v_da=10.,
                  lr_bnd=0.1,
                  threshold_bnd=0.2,
                  tau_v_bnd=2.,
                  tau_ssry=3.,
                  threshold_ssry=0.2,
                  threshold_circuit=10.,
                  rwd_weight=1.,
                  rwd_sigma=1.,
                  col_weight=1.,
                  col_sigma=1.,
                  rwd_field_mod=1.,
                  col_field_mod=1.,
                  action_delay=20.,
                  edge_route_interval=0.1,
                  forced_duration=1.,
                  min_weight_value=0.3)

    # test
    v = brain(np.random.randn(2).tolist(), 0.1, 0., False)
