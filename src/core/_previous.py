
""" from `mod_core` """

class LeakyVariable:

    def __init__(self, eq: float=0., tau: float=10,
                 name: str="leaky", ndim: int=1,
                 visualize: bool=False,
                 number: int=None,
                 max_record: int=100):

        """
        Parameters
        ----------
        eq : float
            Default 0.
        tau : float
            Default 10
        threshold : float
            Default 0.
        """

        self.name = name
        self.eq = np.array([eq]*ndim).reshape(-1, 1) if ndim > 1 else np.array([eq])
        self.ndim = ndim
        self._v = np.ones(1)*self.eq if ndim == 1 else np.ones((ndim, 1))*self.eq
        self.tau = tau
        self.record = []
        self._max_record = max_record
        self._visualize = False 
        self._number = number

        # figure configs
        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))

    def __repr__(self):
        return f"{self.name}(eq={self.eq}, tau={self.tau})"

    def __call__(self, x: float=0., eq: np.ndarray=None,
                 simulate: bool=False):

        if simulate:
            if eq is not None:
                self.eq = eq
            return self._v + (self.eq - self._v) / self.tau + x

        if eq is not None:
            self.eq = eq
        self._v += (self.eq - self._v) / self.tau + x
        self._v = np.maximum(0., self._v)
        # self.v = np.clip(self.v, -1, 1.)
        self.record += [self._v.tolist()]
        if len(self.record) > self._max_record:
            del self.record[0]

        return self._v

    def get_v(self):
        return self._v

    def reset(self):
        self._v = self.eq
        self.record = []

    def render(self):

        if not self._visualize:
            return

        self.ax.clear()
        self.ax.plot(range(len(self.record)), self.record)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.grid()
        self.ax.set_title(f"{self.name} |" +
            f" v={np.around(self._v, 2).tolist()}")

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/{self._number}.png")
            return
        self.fig.canvas.draw()


class ExperienceModule2(ModuleClass):

    """
    Input:
        x: 2D position [array]
        mode: mode [str] ("current" or "proximal")
    Output:
        representation: output [array]
        current position: [array]
    """

    def __init__(self, pcnn: pcnn.PCNN,
                 circuits: Circuits,
                 trg_module: object,
                 weight_policy: object,
                 pcnn_plotter: object=None,
                 action_delay: float=2.,
                 visualize: bool=False,
                 visualize_action: bool=False,
                 number: int=None,
                 weights: np.ndarray=None,
                 speed: int=0.005,
                 max_depth: int=10):

        super().__init__()
        self.pcnn = pcnn
        self.circuits = circuits
        self.pcnn_plotter = pcnn_plotter
        self.trg_module = trg_module
        self.weight_policy = weight_policy
        self.output = {
                "u": np.zeros(pcnn.get_size()),
                "delta_update": np.zeros(pcnn.get_size()),
                "velocity": np.zeros(2),
                "action_idx": None,
                "score": None,
                "depth": None,
                "score_values": np.zeros((9, 4)).tolist(),
                "action_values": np.zeros(9).tolist()}

        # --- policies
        # self.random_policy = RandomWalkPolicy(speed=0.005)
        self.action_policy = SamplingPolicy(speed=speed,
                                    visualize=visualize_action,
                                            number=number,
                                            name="SamplingMain")
        self.action_policy_int = SamplingPolicy(speed=speed,
                                    visualize=False,
                                    name="SamplingInt")
        self.action_space_len = len(self.action_policy)

        self.action_delay = action_delay # ---
        self.action_threshold_eq = -0.001
        self.action_threshold = self.action_threshold_eq

        self.max_depth = max_depth

        self.weights = weights
        if self.weights is None:
            self.weights = np.ones(len(self.circuits)) / \
                len(self.circuits)

        # --- visualization
        self.visualize = visualize
        # if visualize:
        #     self.fig, self.ax = plt.subplots(figsize=(4, 3))
        #     logger(f"%visualizing {self.__class__}")
        # else:
        #     self.fig, self.ax = None, None
        self.record = []

    def _logic(self, observation: dict,
               directive: str="keep"):

        """
        the level of acetylcholine is determined
        affects the next weight update
        """

        # --- update random policy in case of collision
        if observation["collision"]:
            self.action_policy.has_collided()

        # --- get representations
        spatial_repr = self.pcnn(x=observation["position"])
        observation["u"] = spatial_repr.copy()

        # --- generate an action
        # TODO : use directive
        # action_ext, action_idx, score = self._generation_action(
        #                         observation=observation,
        #                         directive=directive)
        action_ext, action_idx, score, depth, score_values, action_values = self._generate_action_from_simulation(
                                observation=observation,
                                directive=directive)

        # --- output
        self.record += [observation["position"].tolist()]
        self.output = {
                "u": spatial_repr,
                "delta_update": self.pcnn.delta_update(),
                "velocity": action_ext,
                "action_idx": action_idx,
                "score": score,
                "depth": depth,
                "score_values": score_values,
                "action_values": action_values}

        possible_actions = "|".join([f"{r[0]*1000:.2f},{r[1]*1000:.2f}" for r in self.action_policy._samples])

        logger(f"action: {action_ext} | " + \
               f"score: {score} | " + \
               f"depth: {depth} | " + \
               f"action_idx: {action_idx} | " + \
               f"action_values: {action_values} | " + \
            f"actions: {possible_actions}")

    def _generate_action(self, observation: dict,
                           directive: str="new") -> np.ndarray:

        self.action_policy_int.reset()
        done = False # when all actions have been tried
        score_values = [[], [], [], [], [], [], [], [], []]

        while True:

            # --- generate a random action
            if directive == "new":
                action, done, action_idx, action_values = self.action_policy_int()
            elif directive == "keep":
                action = observation["velocity"]
                action_idx = observation["action_idx"]
            else:
                raise ValueError("directive must be " + \
                    "'new' or 'keep'")

            # --- simulate its effects

            # new position if the action is taken
            new_position = observation["position"] + \
                action * self.action_delay

            # new observation/effects
            if new_position is None:
                u = np.zeros(self.pcnn.get_size())
            else:
                u = self.pcnn.fwd_ext(x=new_position)

            new_observation = {
                "u": u,
                "position": new_position,
                "velocity": action,
                "collision": False,
                "reward": 0.,
                "delta_update": 0.}

            modulation = self.circuits(
                            observation=new_observation,
                            simulate=True)
            trg_modulation = self.trg_module(
                            observation=new_observation,
                            directive="compare")

            if isinstance(trg_modulation, np.ndarray):
                trg_modulation = trg_modulation.item()

            # --- evaluate the effects
            self.weight_policy(
                        circuits_dict=self.circuits.circuits,                               trg_module=self.trg_module)
            score = 0

            # relevant modulators
            values = [-0*modulation["Bnd"].item(),
                      0*modulation["dPos"].item(),
                      0*modulation["Pop"].item(),
                      -trg_modulation["score"]]
            score += sum(values)
            # score_values += [values]

            if action_idx == 4:
                score = -1.

            # ---
            score_values[action_idx] += [trg_modulation["score"],
                                         score]

            # ---

            # the action is above threshold
            # if score > self.action_threshold:
            if False:

                # lower the threshold
                self.action_threshold = min((score,
                                self.action_threshold_eq))
                break

            # it is the best available action
            elif done:

                # set new threshold
                # [account for a little of bad luck]
                self.action_threshold = score*1.1
                break

            directive = "new"

            # try again
            self.action_policy_int.update(score=score)

        # ---
        return action, action_idx, score, score_values, action_values

    def _simulation_loop(self, observation: dict,
                      depth: int,
                      threshold: float=0.,
                      max_depth: int=10) -> dict:

        position = observation["position"]
        action = observation["velocity"]
        action_idx = observation["action_idx"]

        # --- simulate its effects
        action, action_idx, score, score_values, action_values = self._generate_action(
                                observation=observation,
                                directive="keep")
        # action, action_idx, score = self._generate_action(
        #                         observation=observation,
        #                         directive="keep")
        observation["position"] += action
        observation["velocity"] = action
        observation["action_idx"] = action_idx

        # if score > threshold:
        #     return score, True, depth, score_values, action_values

        if depth >= max_depth:
            return score, False, depth, score_values, action_values

        return self._simulation_loop(observation=observation,
                               depth=depth+1,
                               threshold=threshold,
                               max_depth=max_depth)

    def _generate_action_from_simulation(self,
            observation: dict, directive: str) -> tuple:

        done = False # when all actions have been tried
        self.action_policy.reset()
        assert len(self.action_policy._available_idxs) == 9, \
            "action policy must have 9 actions"
        counter = 0
        while True:

            # save state
            action_threshold_or = self.action_threshold

            # --- generate a random action
            if directive == "new":
                action, done, action_idx, action_values_main = self.action_policy()
                print(f"({counter}) | new: {action_idx=} {done=}")
            elif directive == "keep":
                action = observation["velocity"]
                action_idx = observation["action_idx"]
                done = False
                print(f"(#) | keep: {action_idx=}")
            else:
                raise ValueError("directive must be " + \
                    "'new' or 'keep'")

            # --- simulate its effects over a few steps

            observation["velocity"] = action
            observation["action_idx"] = action_idx

            # drive toward a target
            self.trg_module(observation=observation,
                            directive="calculate")
            self.weight_policy()

            # roll out
            score, success, depth, score_values, action_values = self._simulation_loop(
                            observation=observation,
                            depth=0,
                            threshold=self.action_threshold_eq,
                            max_depth=self.max_depth)

            # restore state
            self.action_threshold = action_threshold_or

            # the action is above threshold
            if score > self.action_threshold and False:

                # lower the threshold
                self.action_threshold = min((score,
                                self.action_threshold_eq))
                # break

            # it is the best available action
            elif done:

                # set new threshold
                # [account for a little of bad luck]
                self.action_threshold = score*1.1
                break

            directive = "new"

            # update
            self.action_policy.update(score=score)
            counter += 1

            print(f".. score: {np.around(score, 3)}")

        # ---
        self.action_policy.reset()

        return action, action_idx, score, depth, score_values, action_values

    def render(self, ax=None, **kwargs):

        self.action_policy.render(values=self.output["score_values"], action_values=self.output["action_values"])
        self.trg_module.render()
        self.weight_policy.render()

        # if ax is None and self.visualize:
        #     ax = self.ax

        if self.pcnn_plotter is not None:
            self.pcnn_plotter.render(ax=None,
                trajectory=kwargs.get("trajectory", False),
                new_a=1*self.circuits.circuits["DA"].output)

    def reset(self, complete: bool=False):
        super().reset(complete=complete)
        if complete:
            self.record = []


class ExperienceModule(ModuleClass):

    """
    Input:
        x: 2D position [array]
        mode: mode [str] ("current" or "proximal")
    Output:
        representation: output [array]
        current position: [array]
    """

    def __init__(self, pcnn: pcnn.PCNN,
                 circuits: Circuits,
                 trg_module: object,
                 weight_policy: object,
                 pcnn_plotter: object=None,
                 action_delay: float=2.,
                 visualize: bool=False,
                 visualize_action: bool=False,
                 number: int=None,
                 weights: np.ndarray=None,
                 speed: int=0.005,
                 max_depth: int=10):

        super().__init__()
        self.pcnn = pcnn
        self.circuits = circuits
        self.pcnn_plotter = pcnn_plotter
        self.trg_module = trg_module
        self.weight_policy = weight_policy
        self.output = {
                "u": np.zeros(pcnn.get_size()),
                "delta_update": np.zeros(pcnn.get_size()),
                "velocity": np.zeros(2),
                "action_idx": None,
                "score": None,
                "depth": None,
                "score_values": np.zeros((9, 4)).tolist(),
                "action_values": np.zeros(9).tolist()}

        # --- policies
        # self.random_policy = RandomWalkPolicy(speed=0.005)
        self.action_policy = SamplingPolicy(speed=speed,
                                    visualize=visualize_action,
                                            number=number,
                                            name="SamplingMain")
        self.action_policy_int = SamplingPolicy(speed=speed,
                                    visualize=False,
                                    name="SamplingInt")
        self.action_space_len = len(self.action_policy)

        self.action_delay = action_delay # ---
        self.action_threshold_eq = -0.001
        self.action_threshold = self.action_threshold_eq

        self.max_depth = max_depth

        self.weights = weights
        if self.weights is None:
            self.weights = np.ones(len(self.circuits)) / \
                len(self.circuits)

        # --- visualization
        self.visualize = visualize
        # if visualize:
        #     self.fig, self.ax = plt.subplots(figsize=(4, 3))
        #     logger(f"%visualizing {self.__class__}")
        # else:
        #     self.fig, self.ax = None, None
        self.record = []

    def _logic(self, observation: dict,
               directive: str="keep"):

        """
        the level of acetylcholine is determined
        affects the next weight update
        """

        # --- get representations
        spatial_repr = self.pcnn(x=observation["position"])
        observation["u"] = spatial_repr.copy()

        # --- generate an action
        action_ext, action_idx, score, depth, score_values, action_values = self._generate_action_from_simulation(
                                observation=observation,
                                directive=directive)

        # --- output & logs
        self.record += [observation["position"].tolist()]
        self.output = {
                "u": spatial_repr,
                "delta_update": self.pcnn.get_delta_update(),
                "velocity": action_ext,
                "action_idx": action_idx,
                "score": score,
                "depth": depth,
                "score_values": score_values,
                "action_values": action_values}

        possible_actions = "|".join([f"{r[0]*1000:.2f},{r[1]*1000:.2f}" for r in self.action_policy._samples])

        # logger(f"action: {action_ext} | " + \
        #        f"score: {score} | " + \
        #        f"depth: {depth} | " + \
        #        f"action_idx: {action_idx} | " + \
        #        f"action_values: {action_values} | " + \
        #     f"actions: {possible_actions}")

    def _generate_action(self, observation: dict,
                           directive: str="new") -> np.ndarray:

        self.action_policy_int.reset()
        done = False # when all actions have been tried
        score_values = [[], [], [], [], [], [], [], [], []]

        while True:

            # --- generate a random action
            if directive == "new":
                action, done, action_idx, action_values = self.action_policy_int()
            elif directive == "keep":
                action = observation["velocity"]
                action_idx = observation["action_idx"]
            else:
                raise ValueError("directive must be " + \
                    "'new' or 'keep'")

            # --- simulate its effects

            # new position if the action is taken
            new_position = observation["position"] + \
                action * self.action_delay

            # new observation/effects
            if new_position is None:
                u = np.zeros(self.pcnn.get_size())
            else:
                u = self.pcnn.fwd_ext(x=new_position)

            new_observation = {
                "u": u,
                "position": new_position,
                "velocity": action,
                "collision": False,
                "reward": 0.,
                "delta_update": 0.}

            modulation = self.circuits(
                            observation=new_observation,
                            simulate=True)
            trg_modulation = self.trg_module(
                            observation=new_observation,
                            directive="compare")

            if isinstance(trg_modulation, np.ndarray):
                trg_modulation = trg_modulation.item()

            # --- evaluate the effects
            self.weight_policy(
                        circuits_dict=self.circuits.circuits,                               trg_module=self.trg_module)
            score = 0

            # relevant modulators
            values = [-0*modulation["Bnd"].item(),
                      0*modulation["dPos"].item(),
                      0*modulation["Pop"].item(),
                      -trg_modulation["score"]]
            score += sum(values)
            # score_values += [values]

            if action_idx == 4:
                score = -1.

            # ---
            score_values[action_idx] += [trg_modulation["score"],
                                         score]

            # ---

            # the action is above threshold
            # if score > self.action_threshold:
            if False:

                # lower the threshold
                self.action_threshold = min((score,
                                self.action_threshold_eq))
                break

            # it is the best available action
            elif done:

                # set new threshold
                # [account for a little of bad luck]
                self.action_threshold = score*1.1
                break

            directive = "new"

            # try again
            self.action_policy_int.update(score=score)

        # ---
        return action, action_idx, score, score_values, action_values

    def _simulation_loop(self, observation: dict,
                      depth: int,
                      threshold: float=0.,
                      max_depth: int=10) -> dict:

        position = observation["position"]
        action = observation["velocity"]
        action_idx = observation["action_idx"]

        # --- simulate its effects
        action, action_idx, score, score_values, action_values = self._generate_action(
                                observation=observation,
                                directive="keep")
        # action, action_idx, score = self._generate_action(
        #                         observation=observation,
        #                         directive="keep")
        observation["position"] += action
        observation["velocity"] = action
        observation["action_idx"] = action_idx

        # if score > threshold:
        #     return score, True, depth, score_values, action_values

        if depth >= max_depth:
            return score, False, depth, score_values, action_values

        return self._simulation_loop(observation=observation,
                               depth=depth+1,
                               threshold=threshold,
                               max_depth=max_depth)

    def _generate_action_from_simulation(self,
            observation: dict, directive: str) -> tuple:

        done = False # when all actions have been tried
        self.action_policy.reset()
        assert len(self.action_policy._available_idxs) == 9, \
            "action policy must have 9 actions"
        counter = 0
        while True:

            # --- generate a random action
            if directive == "new":
                action, done, action_idx, action_values_main = self.action_policy()
                print(f"({counter}) | new: {action_idx=} {done=}")

            elif directive == "keep":
                action = observation["velocity"]
                action_idx = observation["action_idx"]
                done = False
                print(f"(#) | keep: {action_idx=}")
            else:
                raise ValueError("directive must be " + \
                    "'new' or 'keep'")

            # --- simulate its effects over a few steps

            observation["velocity"] = action
            observation["action_idx"] = action_idx

            # drive toward a target
            self.trg_module(observation=observation,
                            directive="calculate")
            self.weight_policy()

            # roll out
            score, success, depth, score_values, action_values = self._simulation_loop(
                            observation=observation,
                            depth=0,
                            threshold=self.action_threshold_eq,
                            max_depth=self.max_depth)

            # restore state
            # self.action_threshold = action_threshold_or

            # the action is above threshold
            if score > self.action_threshold and False:

                # lower the threshold
                self.action_threshold = min((score,
                                self.action_threshold_eq))
                # break

            # it is the best available action
            elif done:

                # set new threshold
                # [account for a little of bad luck]
                self.action_threshold = score*1.1
                break

            directive = "new"

            # update
            self.action_policy.update(score=score)
            counter += 1

            print(f".. score: {np.around(score, 3)}")

        # ---
        self.action_policy.reset()

        return action, action_idx, score, depth, score_values, action_values

    def render(self, ax=None, **kwargs):

        self.action_policy.render(values=self.output["score_values"], action_values=self.output["action_values"])
        self.trg_module.render()
        self.weight_policy.render()

        # if ax is None and self.visualize:
        #     ax = self.ax

        if self.pcnn_plotter is not None:
            self.pcnn_plotter.render(ax=None,
                trajectory=kwargs.get("trajectory", False),
                new_a=1*self.circuits.circuits["DA"].output)

    def reset(self, complete: bool=False):
        super().reset(complete=complete)
        if complete:
            self.record = []


class WeightsPolicy:

    def __init__(self, circuits_dict: dict,
                 trg_module: object,
                 visualize: bool=False,
                 number: int=None):

        self.objects = circuits_dict | {"trg": trg_module}
        self.weights = np.ones(len(self.objects)) / \
            len(self.objects)
        self.windex = {name: i for i, name in \
            enumerate(self.objects.keys())}

        self.visualize = visualize
        self._number = number
        if visualize:
            self.fig, self.ax = plt.subplots(figsize=FIGSIZE)
            logger(f"%visualizing {self.__class__}")

    def __str__(self):
        return f"{self.__class__}(#N={len(self.objects)})"

    def __call__(self, **kwargs):

        return self._assign()

    def _assign(self):

        # --- retrieve
        # for i, (_, obj) in enumerate(self.objects.items()):
        #     self.weights[i] = obj.score_weight

        # --- logic
        # self.weights[self.windex["trg"]] = 20 * \
        #     (self.objects["Ftg"].output > 0.4)
        # self.weights[self.windex["Pop"]] = 1 * \
        #     (self.objects["Ftg"].output < 0.4)
        # self.weights[self.windex["Bnd"]] = 1 if self.objects["Ftg"].output < 0.4 else 0.1

        # --- normalize
        self.weights = self.weights / self.weights.sum()

        # --- update
        for i, (_, obj) in enumerate(self.objects.items()):
            obj.score_weight = self.weights[i]

        return self.weights.copy()

    def render(self):

        if not self.visualize:
            return

        self.ax.clear()
        self.ax.bar(range(len(self.objects)), self.weights,
                    color="blue")
        self.ax.set_xticks(range(len(self.objects)))
        self.ax.set_xticklabels([f"{n}" for n in self.objects.keys()])
        self.ax.set_title("Weights")
        self.ax.grid()
        # self.ax.set_ylim(0, 5)

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return

        self.fig.canvas.draw()


class RandomWalkPolicy:

    def __init__(self, speed: float=0.1):

        self.speed = speed

        self.p = 0.5
        self.velocity = np.zeros(2)

    def __call__(self):

        self.p += (0.2 - self.p) * 0.02
        self.p = np.clip(self.p, 0.01, 0.99)

        if np.random.binomial(1, self.p):
            angle = np.random.uniform(0, 2*np.pi)
            self.velocity = self.speed * np.array([np.cos(angle),
                                              np.sin(angle)])
            self.p *= 0.2

        return self.velocity

    def has_collided(self):
        self.velocity = -self.velocity



class ExploratoryModule(ModuleClass):

    def __init__(self, exp_module: ExperienceModule):

        super().__init__()
        self.exp_module = exp_module
        self.output = None

    def _logic(self, x: np.ndarray, mode: str="repr"):

        """
        the level of acetylcholine is determined
        affects the next weight update
        """

        if mode == "repr":
            _, self.output = self.exp_module(x=x)
        elif mode == "trajectory":
            self.output = self._loop(x=x, tape=[], duration=10)
        else:
            raise ValueError("mode must be 'repr' or 'trajectory'")

    def _loop(self, x: np.ndarray,
              tape: list,
              duration: int=10):

        """
        generate a trajectory
        """

        _, x = self.exp_module(x=x)
        tape += [x]

        if len(tape) < duration:
            self._loop(x=x, tape=tape, duration=duration)

        return tape


class SelfModule(ModuleClass):

    def __init__(self, expl_module: ExploratoryModule):

        super().__init__()
        self.expl_module = expl_module
        self.speed = 0.01

    def _logic(self, x: dict):

        """
        the level of acetylcholine is determined
        affects the next weight update
        """

        position = x["position"]
        collision = x["collision"]

        new_position = self.expl_module(x=position, mode="repr")
        self.output = self._velocity_fn(p1=position,
                                     p2=new_position,
                                     speed=self.speed)

    def _velocity_fn(self, p1: np.ndarray,
                   p2: np.ndarray,
                   speed: float) -> float:

        """
        action function and return as a velocity
        """

        theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        if theta < 0:
            theta += 2 * np.pi

        return np.array([np.cos(theta),
                             np.sin(theta)])

"""
IDEAS
-----

- lock a selected action so to match its predicted value
with the future value at time t_a, where t_a is the depth
of the simulation generating the action
"""



""" from `run_core` """


def make_trajectory(plot: bool=False, speed: float=0.001,
                   duration: int=2, Nj: int=13**2, sigma: float=0.01,
                   **kwargs) -> tuple:

    """
    make a trajectory parsed through a place cell layer.

    Parameters
    ----------
    plot: bool
        plot the trajectory.
    speed: float
        speed of the trajectory.
    duration: int
        duration of the trajectory.
    Nj: int
        number of place cells.
    sigma: float
        variance of the place cells.
    **kwargs
        is2d: bool
            is the trajectory in 2D.
            Default is True.
        dx: float
            step size of the trajectory.
            Default is 0.005.

    Returns
    -------
    trajectory: np.ndarray
        trajectory of the agent.
    whole_track: np.ndarray
        whole track of the agent.
    inputs: np.ndarray
        inputs to the place cell layer.
    whole_track_layer: np.ndarray
        whole track of the place cell layer.
    layer: PlaceLayer
        place cell
    """

    # settings
    bounds = kwargs.get("bounds", (0, 1, 0, 1))
    is2d = kwargs.get("is2d", True)
    dx = kwargs.get("dx", 0.005)

    # make activations
    layer = it.PlaceLayer(N=Nj,
                          sigma=sigma,
                          bounds=bounds)

    trajectory, inputs, whole_track, whole_track_layer = utils.make_env(
        layer=layer, duration=duration, speed=0.1,
        dt=None, distance=None, dx=1e-2,
        plot=False,
        verbose=True,
        bounds=bounds,
        line_env=False,
        make_full=kwargs.get("make_full", False),
        dx_whole=5e-3)

    if plot:
        plt.figure(figsize=(3, 3))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'k-')
        plt.xticks(())
        plt.yticks(())
        plt.axis('off')
        plt.show()

    return trajectory, whole_track, inputs, whole_track_layer, layer


def train(inputs: np.ndarray, layer_centers: np.ndarray,
          params: dict):

    model = PCNN(**params)
    for x in inputs:
        model(x=x.reshape(-1, 1))

    info = {"centers": calc_centers_from_layer(wff=model._Wff,
                                   centers=layer_centers),
            "connectivity": model._Wrec.copy(),
            "count": len(model)}

    return info


def plot_network(centers, connectivity, ax):

    """
    plot the network
    """

    ax.plot(centers[:, 0], centers[:, 1], 'ko', markersize=2)
    for i in range(connectivity.shape[0]):
        for j in range(connectivity.shape[1]):
            if connectivity[i, j] > 0:
                ax.plot([centers[i, 0], centers[j, 0]],
                        [centers[i, 1], centers[j, 1]], 'k-',
                        alpha=0.2, lw=0.5)

    ax.axis('off')

    return ax


def simple_run(args):

    np.random.seed(0)
    duration = args.duration
    N = args.N
    Nj = 13**2

    logger(f"{duration}")
    logger(f"{N=}")
    logger(f"{Nj=}")

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.2,
        "beta": 20.0,
        "threshold": 0.1,
        "rep_thresold": 0.5,
        "rec_threshold": 0.1,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "eq_ach": 1.,
        "tau_ach": 2.,
        "ach_threshold": 0.9,
    }

    # make trajectory
    trajectory, _, _, _, _ = make_trajectory(
                            plot=False,
                            Nj=Nj,
                            duration=duration,
                            is2d=True,
                            sigma=0.01)

    # pc filter
    pclayer = pcnn.PClayer(n=13, sigma=0.01)
    logger.debug(f"{pclayer=}")
    params["xfilter"] = pclayer

    # model
    model = pcnn.PCNN(**params)

    # train
    fig, ax = plt.subplots(figsize=(6, 6))

    for t, x in tqdm_enumerate(trajectory):
        model(x=x.reshape(-1, 1))

        if t % 50 == 0:
            ax.clear()
            # ax.imshow(model._Wff, cmap='viridis', aspect='auto')
            ax.plot(trajectory[:t, 0], trajectory[:t, 1], 'r-',
                    lw=0.5, alpha=0.4)
            centers = pcnn.calc_centers_from_layer(wff=model._Wff,
                                              centers=model.xfilter.centers)
            plot_network(centers=centers,
                         connectivity=model._Wrec,
                         ax=ax)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"t={t} | {len(model)} neurons")
            plt.pause(0.005)


def experimentI():

    np.random.seed(0)
    duration = 10
    N = 80
    Nj = 13**2

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.2,
        "beta": 20.0,
        "threshold": 0.1,
        "rep_thresold": 0.5,
        "rec_threshold": 0.1,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "eq_ach": 1.,
        "tau_ach": 2.,
        "ach_threshold": 0.9,
    }

    # make trajectory
    trajectory, _, inputs, _, _ = make_trajectory(
                            plot=False,
                            Nj=Nj,
                            duration=duration,
                            is2d=True,
                            sigma=0.01)

    xlayer = PClayer(n=13, sigma=0.01)
    params["xfilter"] = xlayer

    D = 10
    var_name1 = "rec_threshold"
    var_values1 = np.linspace(0., 1., D)

    var_name2 = "threshold"
    var_values2 = np.linspace(0., 1., D)

    results = []
    avg_neighbors = np.zeros((D, D))
    info = []
    logger("training..")
    max_cell_count = 0
    for i, value1 in tqdm_enumerate(var_values1):
        for j, value2 in tqdm_enumerate(var_values2):
            params[var_name1] = value1
            params[var_name2] = value2
            info = train(inputs=trajectory, layer_centers=xlayer.centers,
                         params=params)
            results += [info]
            avg_neighbors[i, j] = info["connectivity"].sum() / info["count"]

            max_cell_count = max(max_cell_count, info["count"])

    logger.debug(f"max cell count: {max_cell_count}")
    logger("plotting..")

    # plot
    plt.figure(figsize=(9, 9))
    plt.imshow(np.flip(avg_neighbors, axis=0),
               cmap='viridis')
    plt.colorbar()
    plt.xlabel(var_name2)
    plt.ylabel(var_name1)
    xtickslab = [" "] * D
    xtickslab[0] = f"{var_values2[0]:.2f}"
    xtickslab[-1] = f"{var_values2[-1]:.2f}"
    plt.xticks(range(D), xtickslab)
    ytickslab = [" "] * D
    ytickslab[-1] = f"{var_values1[0]:.2f}"
    ytickslab[0] = f"{var_values1[-1]:.2f}"
    plt.yticks(range(D), ytickslab)
    plt.title("Average number of neighbors")

    plt.show()


def experimentII(args):

    np.random.seed(0)
    duration = args.duration
    N = args.N
    Nj = 13**2

    logger(f"{duration}")
    logger(f"{N=}")
    logger(f"{Nj=}")

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.2,
        "beta": 20.0,
        "threshold": 0.3,
        "rep_thresold": 0.8,
        "rec_threshold": 0.1,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "name": "PCNN"
    }

    # make trajectory
    trajectory, _, _, _, _ = make_trajectory(
                            plot=False,
                            Nj=Nj,
                            duration=duration,
                            is2d=True,
                            sigma=0.01)

    # pc filter
    pclayer = pcnn.PClayer(n=13, sigma=0.01)
    logger.debug(f"{pclayer=}")
    params["xfilter"] = pclayer

    # model
    model = pcnn.PCNN(**params)
    model_plotter = pcnn.PlotPCNN(model=model)
    modulators_list = [mod.BoundaryMod(N=N),
                       mod.Acetylcholine()]

    for modulator in modulators_list:
        logger.debug(f"{modulator} keys: {modulator.input_key}")

    modulators = mod.Modulators(modulators=modulators_list)

    exp_module = mod.ExperienceModule(pcnn=model,
                                      modulators=modulators_list)

    # train
    for t, x in tqdm_enumerate(trajectory):

        exp_module(x=x.reshape(-1, 1))
        modulators(u=exp_module.output[0],
                   position=x,
                   delta_update=exp_module.output[2],
                   collision=False)

        if t % 100 == 0:
            # exp_module.render()
            modulators.render()
            model_plotter.render(trajectory=trajectory[:t])
            plt.pause(0.005)


def experimentIV(args):

    # --- settings
    np.random.seed(args.seed)
    evc.set_seed(seed=args.seed)
    duration = args.duration

    # --- brain
    N = args.N
    Nj = 13**2

    logger(f"{duration}")
    logger(f"{N=}")
    logger(f"{Nj=}")

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.3,
        "beta": 20.0,
        "clip_min": 0.005,
        "threshold": 0.4,
        "rep_threshold": 0.7,
        "rec_threshold": 0.99,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "name": "PCNN"
    }

    # pc filter
    pclayer = pcnn.PClayer(n=13, sigma=0.01)
    logger.debug(f"{pclayer=}")
    params["xfilter"] = pclayer

    # pcnn
    model = pcnn.PCNN(**params)
    model_plotter = pcnn.PlotPCNN(model=model,
                                  visualize=True)
    modulators_dict = {"Bnd": mod.BoundaryMod(N=N,
                                              visualize=False),
                       "Ach": mod.Acetylcholine(visualize=False),
                       "ET": mod.EligibilityTrace(N=N,
                                                  visualize=False),
                       "dPos": mod.PositionTrace(visualize=False)}

    for _, modulator in modulators_dict.items():
        logger.debug(f"{modulator} keys: {modulator.input_key}")

    # other components
    modulators = mod.Modulators(modulators_dict=modulators_dict,
                                visualize=True)
    exp_module = mod.ExperienceModule(pcnn=model,
                                      pcnn_plotter=model_plotter,
                                      modulators=modulators,
                                      speed=0.01)
    brain = mod.Brain(exp_module=exp_module,
                      modulators=modulators)

    # --- agent & env
    agent = evc.AgentBody(brain=brain)

    # --- run
    evc.main(agent=agent,
             duration=duration)


def experimentV(args):

    # --- settings
    np.random.seed(args.seed)
    evc.set_seed(seed=args.seed)
    duration = args.duration

    # --- brain
    N = args.N
    Nj = 13**2

    logger(f"{duration}")
    logger(f"{N=}")
    logger(f"{Nj=}")

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.3,
        "beta": 20.0,
        "clip_min": 0.005,
        "threshold": 0.4,
        "rep_threshold": 0.9,
        "rec_threshold": 0.99,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "name": "PCNN"
    }

    # pc filter
    pclayer = pcnn.PClayer(n=13, sigma=0.01)
    logger.debug(f"{pclayer=}")
    params["xfilter"] = pclayer

    # pcnn
    model = pcnn.PCNN(**params)
    model_plotter = pcnn.PlotPCNN(model=model,
                                  visualize=True,
                                  number=0)
    modulators_dict = {"Bnd": mod.BoundaryMod(N=N,
                                              visualize=True,
                                              number=2),
                       "dPos": mod.PositionTrace(visualize=False)}

    for _, modulator in modulators_dict.items():
        logger.debug(f"{modulator} keys: {modulator.input_key}")

    # other components
    modulators = mod.Modulators(modulators_dict=modulators_dict,
                                visualize=True,
                                number=1)
    exp_module = mod.ExperienceModule(pcnn=model,
                                      pcnn_plotter=model_plotter,
                                      modulators=modulators,
                                      speed=0.009)
    brain = mod.Brain(exp_module=exp_module,
                      modulators=modulators)

    # --- agent & env
    agent = evc.AgentBody(brain=brain)

    # --- run
    evc.main(agent=agent,
             duration=duration)



