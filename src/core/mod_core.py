import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tools.utils import logger
import pcnn_core as pcnn

import matplotlib
matplotlib.use("TkAgg")


FIGPATH = "dashboard/media"


def set_seed(seed: int=0):
    np.random.seed(seed)


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


class Modulation(ABC):

    """
    Actions:
    - "plasticity"
    - "fwd"
    """

    def __init__(self, **kwargs):

        self.N = kwargs.get("N", 1)
        self.leaky_var = None
        self.action = None
        self.input_key = None
        self.output = None
        self.value = None
        self.score_weight = kwargs.get("score_weight", 1.)
        self._visualize = kwargs.get("visualize", False)
        self._number = kwargs.get("number", None)
        self.mod_weight = kwargs.get("mod_weight", 1.)

    def __repr__(self):
        return f"Mod.{self.leaky_var.name}"

    def __call__(self, inputs: float=0.,
                 simulate: bool=False):
        self._logic(inputs=inputs,
                    simulate=simulate)
        return self.score_weight * self.output

    @abstractmethod
    def _logic(self, inputs: float):
        pass

    def get_leaky_v(self):
        return self.leaky_var._v

    def render(self):
        self.leaky_var.render()

    def render_field(self, pcnn_plotter: object):

        self.ax.clear()
        pcnn_plotter.render(ax=self.ax,
                            trajectory=None,
                            new_a=self.weights,
                            edges=False,
                            alpha_nodes=0.8,
                            cmap="Greens",
                            title=f"{self.leaky_var.name}")

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return

    def reset(self, complete: bool=False):
        self.leaky_var.reset()
        if complete:
            self.output = None
            self.state = (0., 0.)


class Program(ABC):

    """
    Actions:
    - "plasticity"
    - "fwd"
    """

    def __init__(self, **kwargs):

        self.N = kwargs.get("N", 1)
        self.name = kwargs.get("name", "program")
        self.input_key = None
        self.output = None
        self.value = None
        self.score_weight = kwargs.get("score_weight", 1.)
        self._visualize = kwargs.get("visualize", False)
        self._number = kwargs.get("number", None)
        self.mod_weight = kwargs.get("mod_weight", 1.)

    def __repr__(self):
        return f"Prog.{self.name}"

    def __call__(self, inputs: float=0.,
                 simulate: bool=False):
        self._logic(inputs=inputs,
                    simulate=simulate)
        return self.score_weight * self.output

    @abstractmethod
    def _logic(self, inputs: float):
        pass

    def render(self):
        pass

    def reset(self, complete: bool=False):
        self.leaky_var.reset()
        if complete:
            self.output = None
            self.state = (0., 0.)


class ModuleClass(ABC):

    def __init__(self):

        self.circuits = None
        self.output = None
        self.pcnn = None
        self._number = None
        self.score_weight = 1.

    def __call__(self, observation: dict,
                 **kwargs):

        # self._apply_modulators()

        self._logic(observation=observation,
                    **kwargs)
        return self.output

    def _apply_circuits(self):
        if self.pcnn is None or self.circuits is None:
            return

        for _, modulator in self.circuits.circuits.items():

            if isinstance(modulator, np.ndarray):
                output = modulator.output.copy().reshape(-1, 1)
            else:
                output = modulator.output

            if modulator.action == "plasticity":
                self.pcnn.add_update(x=output)
            elif modulator.action == "fwd":
                self.pcnn.add_input(x=output)
            elif modulator.action == "fwd":
                pass
            else:
                raise ValueError(
                    "leaky_var action must" + \
                    " be 'plasticity' or 'fwd'")

    @abstractmethod
    def _logic(self, observation: float, **kwargs):
        pass

    def fwd_pcnn(self, x: np.ndarray) -> np.ndarray:
        return self.pcnn.fwd_ext(x=x)



""" --- some leaky_var classes --- """


class Dopamine(Modulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.leaky_var = LeakyVariable(eq=0., tau=5,
                                       name="DA",
                                       max_record=500)
        self.threshold = 0.7
        self.action = "fwd"
        self.input_key = ("u", "reward", None)
        self.weights = np.zeros(self.N)
        self.eta = 0.1
        self.output = np.zeros((self.N, 1))

        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
            logger(f"%visualizing {self.__class__}")

    def _logic(self, inputs: tuple,
               simulate: bool=False) -> np.ndarray:

        u = inputs[0].flatten()

        if inputs[1] > 0:
            v = np.clip(self.leaky_var(x=inputs[1], simulate=simulate),
                        0, 1)

            # potentiation
            self.weights += self.eta * v * np.where(u < 0.1, 0, u)

            # depression
            self.weights -= self.eta * (1 - v) * np.where(u < 0.1, 0, u)
        else:
            v = np.clip(self.leaky_var(x=0, simulate=simulate),
                        0, 1)
        # self.weights = np.clip(self.weights, 0., 1.)

        out = [(self.weights * u).reshape(-1, 1),
                self.get_leaky_v()]
        self.output = out[0]
        self.value = out[0].max()
        self.var = u.copy()

        return out

    def render(self):

        if not self._visualize:
            return

        self.ax.clear()
        self.ax.imshow(self.weights.reshape(1, -1),
                           aspect="auto", cmap="Greens_r",
                           vmin=0., vmax=4.)
        self.ax.set_title(f"Dopamine Modulation | " + \
            f"max:{self.weights.max():.2f} [{np.argmax(self.weights)}]")
        self.ax.set_yticks([])

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return

        self.fig.canvas.draw()


class Acetylcholine(Modulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.leaky_var = LeakyVariable(eq=1., tau=10,
                                       name="ACh",
                                       max_record=500)
        self.threshold = 0.9
        self.action = "plasticity"
        self.input_key = ("delta_update", None)
        self.output = 0.

    def _logic(self, inputs: tuple,
               simulate: bool=False) -> np.ndarray:
        v = self.leaky_var(x= -1 * inputs[0], simulate=simulate)

        out = [1*(v > self.threshold),
                self.get_leaky_v()]
        self.output = out[0]
        self.value = out[0]
        return out


class FatigueMod(Modulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.leaky_var = LeakyVariable(eq=1., tau=1000,
                                       name="Ftg",
                                       max_record=500)
        self.threshold = 0.9
        self.action = "fwd"
        self.input_key = ("reward", None)
        self.output = np.zeros((1, 1))

        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
            logger(f"%visualizing {self.__class__}")

    def _logic(self, inputs: tuple,
               simulate: bool=False) -> np.ndarray:

        if inputs[0] > 0:
            v = self.leaky_var(x=-1, simulate=simulate)
        else:
            v = self.leaky_var(simulate=simulate)

        self.output = self.score_weight * v.item()
        self.value = v.item()
        return


class BoundaryMod(Modulation):

    """
    Input:
        x: representation `u` [array]
        collision: collision [bool]
    Output:
        representation (output) [array]
    """

    def __init__(self, eta: float=0.1,
                 threshold: float=0.01,
                 **kwargs):

        super().__init__(**kwargs)
        self.leaky_var = LeakyVariable(eq=0., tau=3,
                                       name="Bnd")
        self.action = "fwd"
        self.input_key = ("u", "collision")
        self.threshold = threshold
        self.weights = np.zeros(self.N)
        self.eta = eta
        self.output = np.zeros((self.N, 1))

        if self._visualize:
            # self.fig_bnd, self.axs_bnd = plt.subplots(
            #     2, 1, figsize=(4, 3))
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
            logger(f"%visualizing {self.__class__}")

        self.var = np.zeros(self.N)

    def _logic(self, inputs: tuple, simulate: bool=False):

        u = inputs[0].flatten()

        if inputs[1]:
            v = self.leaky_var(x=1, simulate=simulate)
            self.weights += self.eta * v * \
                np.where(u < self.threshold, 0, u)
            self.weights = np.clip(self.weights, 0., 1.)
        else:
            v = self.leaky_var(x=0, simulate=simulate)

        out = [self.weights.reshape(1, -1) @ u.reshape(-1, 1),
                self.get_leaky_v()]
        self.output = out[0]
        self.value = out[0]
        self.var = u.copy()
        return out

    def super_render(self):

        if not self._visualize:
            return

        self.axs_bnd[0].clear()
        self.axs_bnd[1].clear()
        self.axs_bnd[0].imshow(self.weights.reshape(1, -1),
                           aspect="auto", cmap="RdBu",
                               vmin=-0.1, vmax=0.1)
        self.axs_bnd[0].set_title(f"Boundary Modulation")
        self.axs_bnd[0].set_yticks([])
        self.axs_bnd[0].set_xticks([])

        # self.axs_bnd[1].imshow(self.output.reshape(1, -1),
        #                    aspect="auto", cmap="RdBu",
        #                        vmin=-0.1, vmax=0.1)
        self.axs_bnd[1].bar(range(self.N), self.var.flatten(),
                            color="grey")
        # self.axs_bnd[1].set_title(f"Output")
        self.axs_bnd[1].set_ylim(0, 0.035)
        self.axs_bnd[1].axhline(y=0.005, color="red",
                                alpha=0.4, linestyle="--")
        self.axs_bnd[1].grid()

        if self._number is not None:
            self.fig_bnd.savefig(f"{FIGPATH}/fig{self._number}.png")
            return

        self.fig_bnd.canvas.draw()


class PositionTrace(Modulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.leaky_var = LeakyVariable(eq=0., tau=50,
                                       name="dPos",
                                       ndim=2,
                                       max_record=500)
        self.output = self.get_leaky_v().reshape(-1, 1)
        self.threshold = 0.9
        self.action = "fwd"
        self.input_key = ("position", None)
        self.output = 1.
        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
            logger(f"%visualizing {self.__class__}")

    def _logic(self, inputs: tuple, simulate: bool=False):

        v = self.leaky_var(eq=inputs[0].reshape(-1, 1),
                           simulate=simulate)

        res = np.abs(inputs[0].reshape(-1, 1) - v)

        out = [res.sum(),
               self.get_leaky_v()]
        self.output = out[0] * (out[0] > 0.25)
        self.value = out[0]
        return out

    def render(self):

        if not self._visualize:
            return

        self.ax.clear()

        self.ax.bar(range(2), self.output,
                    color="orange")
        self.ax.set_ylim(0., 0.8)
        self.ax.set_title(f"Position Trace " + \
            f"{np.abs(self.output).max():.2f}")
        self.fig.canvas.draw()


class Regions(Modulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.leaky_var = LeakyVariable(eq=0., tau=50,
                                       name="Red",
                                       ndim=2,
                                       max_record=500)
        self.output = self.get_leaky_v().reshape(-1, 1)
        self.threshold = 0.9
        self.action = "fwd"
        self.input_key = ("position", None)
        self.output = np.zeros((self.N, 1))
        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
            logger(f"%visualizing {self.__class__}")

    def _logic(self, inputs: tuple, simulate: bool=False):

        v = self.leaky_var(eq=inputs[0].reshape(-1, 1),
                           simulate=simulate)

        res = np.abs(inputs[0].reshape(-1, 1) - v)

        out = [res.sum(),
               self.get_leaky_v()]
        self.output = out[0] * (out[0] > 0.25)
        self.value = out[0]
        return out

    def render(self):

        if not self._visualize:
            return

        self.ax.clear()

        self.ax.bar(range(2), self.output,
                    color="orange")
        self.ax.set_ylim(0., 0.8)
        self.ax.set_title(f"Position Trace " + \
            f"{np.abs(self.output).max():.2f}")
        self.fig.canvas.draw()



# --- PROGRAMS


class PopulationProgMax(Program):

    def __init__(self, **kwargs):

            super().__init__(**kwargs)
            self.input_key = ("u", None)
            self.output = np.zeros((self.N, 1))
            self._activity = np.zeros(self.N)

            if self._visualize:
                self.fig, self.ax = plt.subplots(figsize=(4, 3))
                logger(f"%visualizing {self.__class__}")

    def _logic(self, inputs: tuple, simulate: bool=False):

        self.output = inputs[0].max()
        self.value = inputs[0].max()
        if not simulate:
            self.activity = inputs[0].flatten()
        return self.output

    def render(self):

        if not self._visualize:
            return

        self.ax.clear()

        self.ax.bar(range(self.N), self._activity,
                    color="grey")
        self.ax.grid()
        self.ax.set_ylim(0, 1.)
        self.ax.set_title(f"Population Activity " + \
            f"$u_{{max}}$={self._activity.max():.2f}")

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return
        self.fig.canvas.draw()


class TargetProg(Program):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.input_key = ("u", None)
        self.output = np.zeros((self.N, 1))
        self._activity = np.zeros(self.N)

        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
            logger(f"%visualizing {self.__class__}")

    def _logic(self, inputs: tuple, simulate: bool=False):

        self.output = inputs[0].max()
        self.value = inputs[0].max()
        if not simulate:
            self.activity = inputs[0].flatten()
        return self.output

    def render(self):

        if not self._visualize:
            return

        self.ax.clear()

        self.ax.bar(range(self.N), self._activity,
                    color="grey")
        self.ax.grid()
        self.ax.set_ylim(0, 1.)
        self.ax.set_title(f"Population Activity " + \
            f"$u_{{max}}$={self._activity.max():.2f}")

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return
        self.fig.canvas.draw()



# --- COLLECTION


class Circuits:

    def __init__(self, circuits_dict: dict,
                 visualize: bool=False,
                 number: int=None):

        self.circuits = circuits_dict
        self.names = tuple(circuits_dict.keys())
        logger(f"circuits : {self.names}")
        self.output = {name: circuits_dict[name].output \
            for name in self.names}
        self.values = np.zeros(len(self.names))

        self._number = number
        self.visualize = visualize
        if visualize:
            self.fig, self.axs = plt.subplots(figsize=(4, 4))
            logger(f"%visualizing {self.__class__}")

    def __len__(self):
        return len(self.circuits)

    def __call__(self, observation: dict,
                 simulate: bool=False) -> dict:

        for i, (name, circuit) in enumerate(
                                self.circuits.items()):
            inputs = []
            for key in circuit.input_key:
                if key is not None:
                    inputs += [observation[key]]
            inputs += [None]
            _ = circuit(inputs=inputs,
                        simulate=simulate)
            self.output[name] = circuit.output
            self.values[i] = circuit.value

        return self.output

    def render(self, pcnn_plotter: object=None):

        # render singular circuit
        for _, circuit in self.circuits.items():
            if hasattr(circuit, "weights") and \
                pcnn_plotter is not None:
                circuit.render_field(pcnn_plotter=pcnn_plotter)
            else:
                circuit.render()

        # render dashboard
        if self.visualize:

            self.axs.clear()
            self.axs.bar(range(len(self.names)), self.values,
                         color="purple")
            self.axs.set_xticks(range(len(self.names)))
            self.axs.set_xticklabels([f"{n}\n{v:.2f}" for n, v \
                in zip(self.names, self.values)])
            self.axs.set_title("Circuits")
            self.axs.set_ylim(-0.1, 1.3)

            if self._number is not None:
                self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
                return

            self.fig.canvas.draw()



""" --- some MODULE classes --- """


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
                "u": np.zeros(pcnn.N),
                "delta_update": np.zeros(pcnn.N),
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
        spatial_repr = self.pcnn.fwd_ext(
                            x=observation["position"])
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
                "delta_update": self.pcnn.delta_update,
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
                u = np.zeros(self.pcnn.N)
            else:
                u = self.pcnn.fwd_ext(x=new_position,
                                      frozen=True)

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
        # logger.warning(f"score: {score_values} | ")

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
        logger.debug(f"---- generate action [{directive=}] ...")
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

        logger.debug(f"#generated action: {action} | " + \
                     f"score: {score} | " + \
                     f"depth: {depth} | " + \
                     f"[M] action_idx: {action_idx} | " + \
                     f"[M] action_values: {action_values_main}")

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


class TargetModule(ModuleClass):

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
                 pcnn_plotter: object=None,
                 visualize: bool=True,
                 visualize_action: bool=False,
                 speed: int=0.005,
                 max_depth: int=10,
                 score_weight: float=1.,
                 number: int=None):

        super().__init__()
        self.pcnn = pcnn
        self.circuits = circuits
        self.pcnn_plotter = pcnn_plotter
        self.speed = speed
        self.max_depth = max_depth
        self.score_weight = score_weight
        self.output = {
                "u": np.zeros(pcnn.N),
                "trg_position": np.zeros(2),
                "velocity": np.zeros(2),
                "score": 0.}

        # --- visualization
        self.visualize = visualize
        self._number = number
        if visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
            logger(f"%visualizing {self.__class__}")

    def _logic(self, observation: dict,
               directive: str=None):

        # compare a queries velocity with the
        # calculated target velocity
        if directive == "compare":

            # only if fatigue is high
            if self.circuits.circuits["Ftg"].value > 0.4:
                score = pcnn.cosine_similarity_vec(
                                self.output["velocity"],
                                observation["velocity"])

                self.output["score"] = np.array([-1 * score]) #* self.score_weight
                # ptrg_vs_actionsrint(f"compared: {observation['velocity']} | " + \
                #       f"target: {self.output['velocity']} | " + \
                #       f"score: {self.output['score']}")
            else:
                self.score_weight = np.ones(1)
                self.output["score"] = 0.

            return

        # calculate the target position

        # --- modulation
        modulation = self.circuits.circuits["DA"].weights

        # --- get representations
        curr_repr = self.pcnn.fwd_ext(
                            x=observation["position"],
                            frozen=True)
        curr_pos = observation["position"]

        trg_repr, flag = self._converge_to_location(
                x=np.zeros((self.pcnn.N, 1)),
                depth=0,
                modulation=modulation,
                threshold=0.8)
        centers = self.pcnn._centers

        # set -np.inf values to zero
        centers = np.where(centers == -np.inf, 0, centers)
        trg_pos = pcnn.calc_position_from_centers(a=trg_repr,
                                    centers=centers)

        # velocity
        velocity = trg_pos - curr_pos
        velocity = velocity / np.linalg.norm(velocity) * self.speed

        # intensity
        if self.circuits.circuits["Ftg"].value > 0.4:
            score = -1 * int(flag) * self.circuits.circuits["Ftg"].value
        else:
            self.score_weight = 1.
            score = 0.

        # --- output
        self.output = {
                "u": trg_repr,
                "trg_position": trg_pos,
                "velocity": velocity,
                "score": np.array([score])}

    def _converge_to_location(self, x: np.ndarray,
                              depth: int,
                              modulation: np.ndarray,
                              threshold: float=0.1):

        u = self.pcnn._Wrec @ (x + modulation.reshape(-1, 1))

        c = pcnn.cosine_similarity_vec(u, x)
        if c > threshold:
            return u, True

        if depth >= self.max_depth:
            return u, False

        return self._converge_to_location(x=u,
                                          depth=depth+1,
                                          modulation=modulation,
                                          threshold=threshold)

    def render(self, **kwargs):

        if not self.visualize:
            return


        # if self.pcnn_plotter is not None:
        #     self.pcnn_plotter.render(ax=self.ax,
        #         trajectory=kwargs.get("trajectory", None),
        #         new_a=1*self.output["u"])

        self.ax.clear()
        self.ax.scatter(self.output["trg_position"][0],
                    self.output["trg_position"][1],
                    marker="x", color="red",
                    s=100*np.abs(self.output["score"]))
        self.ax.set_title(f"Target Module | " + \
                          f" I={np.around(self.output['score'], 2)}")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect("equal")
        self.ax.set_xlabel(f"trg_pos={np.around(self.output['trg_position'], 3)}")
        self.ax.grid()

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return

    def reset(self, complete: bool=False):
        super().reset(complete=complete)


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
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
            logger(f"%visualizing {self.__class__}")

    def __str__(self):
        return f"{self.__class__}(#N={len(self.objects)})"

    def __call__(self, **kwargs):

        return self._assign()

    def _assign(self):

        # --- retrieve
        for i, (_, obj) in enumerate(self.objects.items()):
            self.weights[i] = obj.score_weight

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
        # self.ax.set_ylim(0, 5)

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return

        self.fig.canvas.draw()



""" --- high level MODULES --- """



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


class Brain:

    def __init__(self, exp_module: ExperienceModule,
                 circuits: Circuits,
                 number: int=None,
                 plot_intv: int=1):

        self.exp_module = exp_module
        self.circuits = circuits

        self.movement = None
        self.observation_ext = {
            "position": np.zeros(2).astype(float),
            "collision": False
        }

        self.observation_int = {
            "u": np.zeros(exp_module.pcnn.N),
            "delta_update": 0.,
            "velocity": np.zeros(2),
            "action_idx": 0,
            "depth": 0,
            "score": 0,
            "onset": 0,
            "trg_u": np.zeros(exp_module.pcnn.N),
            "trg_position": np.zeros(2),
            "trg_velocity": np.zeros(2),
            "importance": 0,
        }

        self.state = self.observation_ext | self.observation_int

        self.directive = "new"
        self._elapsed_time = 0
        self._plot_intv = plot_intv
        self.t = -1
        self.number = number
        self.frequencies = np.zeros(self.exp_module.action_space_len)

        if number is not None:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))

        # record
        self.record = {"trajectory": []}

    def __call__(self, observation: dict):

        self.t += 1
        self.observation_ext = observation
        self.state = self.observation_ext | self.observation_int

        self.record["trajectory"] += [
                    self.observation_ext["position"].tolist()]

        # --- update modulation
        mod_observation = self.circuits(observation=self.state)

        # TODO : add directive depending on modulation


        # set new directive
        if self.t == 0:
            self.directive = "new"

        if self.directive == "force_keep" and \
            (self.t - self.observation_int["onset"]) < \
            self.observation_int["depth"]:
            print("force keep ...")

            return self.movement

#         elif self.directive == "force_keep":
#             self.directive = "new"

        elif mod_observation["dPos"] < 0.05:
            self.directive = "new"

        else:
            self.directive = "keep"

        logger.debug(f"Brain || directive: {self.directive}")

        # --- update experience module
        # full output
        self.observation_int = self.exp_module(
                            observation=self.state,
                            directive=self.directive)

        # --- update output
        self.observation_int["onset"] = self.t
        self.directive = "force_keep"
        self.movement = self.observation_int["velocity"]

        self.frequencies[self.observation_int["action_idx"]] += 1

        logger.debug(f"keep: {self.observation_int['onset']}"+\
            f" | t={self.t} dir: {self.directive}")

        return self.movement

    def routines(self, wall_vectors: np.ndarray):

        # pcnn routine
        self.exp_module.pcnn.clean_recurrent(
                                wall_vectors=wall_vectors)

    def render(self, **kwargs):

        if self.t % self._plot_intv == 0:

            self.circuits.render(
                pcnn_plotter=self.exp_module.pcnn_plotter)
            self.exp_module.render(ax=kwargs.get("ax", None),
                              trajectory=np.array(
                                   self.record["trajectory"]))

            # if self.number is not None:

            #     self.ax.clear()
            #     self.ax.bar(range(len(self.frequencies)),
            #                 self.frequencies / self.frequencies.sum(),
            #                 color="black")
            #     self.ax.set_xticks(range(len(self.frequencies)))
            #     self.ax.set_xticklabels([f"{i}" for i in range(len(self.frequencies))])
            #     self.ax.set_title("Action Frequencies")
            #     self.ax.set_ylim(0, 1)

            #     self.fig.savefig(f"{FIGPATH}/fig{self.number}.png")

    @property
    def render_values(self):
        return self.circuits.values.copy(), self.circuits.names




""" policies """


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



class SamplingPolicy:

    def __init__(self, samples: list=None,
                 speed: float=0.1,
                 visualize: bool=False,
                 number: int=None,
                 name: str=None):

        """
        Parameters
        ----------
        samples : list, optional
            List of samples. The default is None.
        speed : float, optional
            Speed of the agent. The default is 0.1.
        visualize : bool, optional
            Visualize the policy. The default is False.
        number : int, optional
            Number of the figure. The default is None.
        name : str, optional
            Name of the policy. The default is None.
        """

        self._name = name if name is not None else "SamplingPolicy"
        self._samples = samples
        if samples is None:
            self._samples = [np.array([-speed/np.sqrt(2),
                                       speed/np.sqrt(2)]),
                             np.array([0., speed]),
                             np.array([speed/np.sqrt(2),
                                       speed/np.sqrt(2)]),
                             np.array([-speed, 0.]),
                             np.array([0., 0.]),
                             np.array([speed, 0.]),
                             np.array([-speed/np.sqrt(2),
                                       -speed/np.sqrt(2)]),
                             np.array([0., -speed]),
                             np.array([speed/np.sqrt(2),
                                       -speed/np.sqrt(2)])]
            logger(f"{self.__class__} using default samples [2D movements]")

        # np.random.shuffle(self._samples)

        self._num_samples = len(self._samples)
        self._samples_indexes = list(range(self._num_samples))

        self._idx = None
        self._available_idxs = list(range(self._num_samples))
        self._p = np.ones(self._num_samples) / self._num_samples
        self._velocity = self._samples[0]
        self._values = np.zeros(self._num_samples)

        # render
        self._number = number
        self.visualize = visualize
        if visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 6))
            logger(f"%visualizing {self.__class__}")

    def __len__(self):
        return self._num_samples

    def __str__(self):

        return f"{self._name}(#samples={self._num_samples})"

    def __call__(self, keep: bool=False) -> tuple:

        # --- keep the current velocity
        if keep and self._idx is not None:
            return self._velocity.copy(), False, self._idx

        # --- first sample
        if self._idx is None:
            self._idx = np.random.choice(
                            self._samples_indexes, p=self._p)
            self._available_idxs.remove(self._idx)
            self._velocity = self._samples[self._idx]
            return self._velocity.copy(), False, self._idx, self._values

        # --- all samples have been tried
        if len(self._available_idxs) == 0:

            # self._idx = np.random.choice(self._num_samples,
            #                                   p=self._p)

            if np.where(self._values == 0)[0].size > 1:
                self._idx = np.random.choice(
                                np.where(self._values == 0)[0])
            else:
                self._idx = np.argmax(self._values)

            self._velocity = self._samples[self._idx]
            # print(f"{self._name} || selected: {self._idx} | " + \
            #     f"{self._values.max()} | values: {np.around(self._values, 2)} v={np.around(self._velocity*1000, 2)}")
            return self._velocity.copy(), True, self._idx, self._values

        # --- sample again
        p = self._p[self._available_idxs].copy()
        p /= p.sum()
        self._idx = np.random.choice(
                        self._available_idxs,
                        p=p)
        self._available_idxs.remove(self._idx)
        self._velocity = self._samples[self._idx]

        return self._velocity.copy(), False, self._idx, self._values
    def update(self, score: float):

        # --- normalize the score
        # score = pcnn.generalized_sigmoid(x=score,
        #                                  alpha=-0.5,
        #                                  beta=1.)

        self._values[self._idx] = score

        # --- update the probability
        # a raw score of 0. becomes 0.5 [sigmoid]
        # and this ends in a multiplier of 1. [id]
        # self._p[self._idx] *= (0.5 + score)

        # normalize
        # self._p = self._p / self._p.sum()

    def get_state(self):

        return {"values": self._values,
                "idx": self._idx,
                "p": self._p,
                "velocity": self._velocity,
                "available_idxs": self._available_idxs}

    def set_state(self, state: dict):

        self._values = state["values"]
        self._idx = state["idx"]
        self._p = state["p"]
        self._velocity = state["velocity"]
        self._available_idxs = state["available_idxs"]

    def render(self, values: np.ndarray=None,
               action_values: np.ndarray=None):

        if not self.visualize:
            return

        # self._values = (self._values.max() - self._values) / \
        #     (self._values.max() - self._values.min())
        # self._values = np.where(np.isnan(self._values), 0,
        #                         self._values)

        self.ax.clear()

        if action_values is not None:
            self.ax.imshow(action_values.reshape(3, 3),
                           cmap="RdBu_r", vmin=-1.1, vmax=1.1,
                           aspect="equal",
                           interpolation="nearest")
        else:
            self.ax.imshow(self._values.reshape(3, 3),
                           cmap="RdBu_r", vmin=-3.1, vmax=3.1,
                           aspect="equal",
                           interpolation="nearest")

        # labels inside each square
        for i in range(3):
            for j in range(3):

                if values is not None:
                    text = "".join([f"{np.around(v, 2)}\n" for v in values[3*i+j]])
                else:
                    text = f"{self._samples[3*i+j][1]:.3f}\n" + \
                          f"{self._samples[3*i+j][0]:.3f}"
                self.ax.text(j, i, f"{text}",
                             ha="center", va="center",
                             color="black",
                             fontsize=13)

        # self.ax.bar(range(self._num_samples), self._values)
        # self.ax.set_xticks(range(self._num_samples))
        # self.ax.set_xticklabels(["stay", "up", "right",
        #                          "down", "left"])
        # self.ax.set_xticklabels(np.around(self._values, 2))
        self.ax.set_xlabel("Action")
        self.ax.set_title(f"Action Space")
        self.ax.set_yticks(range(3))
        # self.ax.set_ylim(-1, 1)
        self.ax.set_xticks(range(3))

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return

        self.fig.canvas.draw()

    def reset(self):
        self._idx = None
        self._available_idxs = list(range(self._num_samples))
        self._p = np.ones(self._num_samples) / self._num_samples
        self._velocity = self._samples[0]
        self._values = np.zeros(self._num_samples)

    def has_collided(self):

        self._velocity = -self._velocity







"""
IDEAS
-----

- lock a selected action so to match its predicted value
with the future value at time t_a, where t_a is the depth
of the simulation generating the action
"""

