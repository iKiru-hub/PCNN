import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import pcnn_core as pcore
import utils_core as utils
import pclib



""" INITIALIZATION """

FIGPATH = "dashboard/media"
FIGSIZE = (4, 4)


def set_seed(seed: int=0):
    np.random.seed(seed)

logger = utils.setup_logger(name="MOD",
                            level=-1,
                            is_debugging=False,
                            is_warning=False)


""" ABSTRACT CLASSES """

class LeakyVariableWrapper1D(pclib.LeakyVariable1D):

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

        super().__init__(name=name, eq=eq, tau=tau,
                         min_v=0.)

        self.name = name
        self.record = []
        self._max_record = max_record
        self._visualize = False
        self._number = number

        # figure configs
        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=FIGSIZE)

    def __call__(self, x: float=0.,
                 simulate: bool=False):

        if simulate:
            return super().__call__(x, True)

        self.record += [super().__call__(x)]
        if len(self.record) > self._max_record:
            del self.record[0]

        return self.record[-1]

    def reset(self):
        super().reset()
        self.record = []

    def render(self, return_fig: bool=False):

        if not self._visualize:
            return

        self.ax.clear()
        self.ax.plot(range(len(self.record)), self.record)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.grid()
        self.ax.set_title(f"{self.name} |" +
            f" v={np.around(self.get_v(), 2).tolist()}")

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/{self._number}.png")
            return
        self.fig.canvas.draw()

        if return_fig:
            return self.fig


class LeakyVariableWrapperND(pclib.LeakyVariableND):

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

        super().__init__(name=name, eq=eq, tau=tau,
                         ndim=ndim, min_v=0.)

        self.name = name
        self.record = []
        self._max_record = max_record
        self._visualize = False
        self._number = number

        # figure configs
        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=FIGSIZE)

    def __call__(self, x: float=0.,
                 simulate: bool=False):

        if simulate:
            return super().__call__(x, True)

        self.record += [super().__call__(x).tolist()]
        if len(self.record) > self._max_record:
            del self.record[0]

        return self.record[-1]

    def reset(self):
        super().reset()
        self.record = []

    def render(self, return_fig: bool=False):

        if not self._visualize:
            return

        self.ax.clear()
        self.ax.plot(range(len(self.record)), self.record)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.grid()
        self.ax.set_title(f"{self.name} |" +
            f" v={np.around(self.get_v(), 2).tolist()}")

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/{self._number}.png")
            return
        self.fig.canvas.draw()

        if return_fig:
            return self.fig


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
        self._visualize = kwargs.get("visualize", False)
        self._number = kwargs.get("number", None)
        self.mod_weight = kwargs.get("mod_weight", 1.)

        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=FIGSIZE)
            logger(f"%visualizing {self.__class__}")

    def __repr__(self):
        return f"Mod.{self.leaky_var.name}"

    def __call__(self, inputs: float=0.,
                 simulate: bool=False):
        self._logic(inputs=inputs,
                    simulate=simulate)
        return self.output

    @abstractmethod
    def _logic(self, inputs: float):
        pass

    def get_leaky_v(self):
        return self.leaky_var.get_v()

    def render(self, return_fig: bool=False):
        self.leaky_var.render(return_fig=return_fig)

    def render_field(self, pcnn_plotter: object,
                     bounds: np.ndarray=None,
                     return_fig: bool=False):

        self.ax.clear()
        pcnn_plotter.render(ax=self.ax,
                            trajectory=None,
                            new_a=self.weights,
                            edges=False,
                            alpha_nodes=0.8,
                            cmap="Greens",
                            customize=True,
                            title=f"{self.leaky_var.name}")

        # if bounds is not None:
        #     self.ax.set_xlim(bounds[0], bounds[1]*1.2)
        #     self.ax.set_ylim(bounds[2], bounds[3]*1.2)

        # self.ax.set_xticks([bounds[0], bounds[1]])
        # self.ax.set_yticks([bounds[2], bounds[3]])
        # self.ax.set_xticklabels([f"{bounds[0]:.2f}", f"{bounds[1]:.2f}"])
        # self.ax.set_yticklabels([f"{bounds[2]:.2f}", f"{bounds[3]:.2f}"])

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return

        if return_fig:
            return self.fig

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
        self._visualize = kwargs.get("visualize", False)
        self._number = kwargs.get("number", None)
        self.mod_weight = kwargs.get("mod_weight", 1.)

    def __repr__(self):
        return f"Prog.{self.name}"

    def __call__(self, inputs: float=0.,
                 simulate: bool=False):
        self._logic(inputs=inputs,
                    simulate=simulate)
        return self.output

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


""" --- MODULATORS & PROGRAMS --- """


class Dopamine(Modulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.leaky_var = LeakyVariableWrapper1D(eq=0., tau=5,
                                       name="DA",
                                       max_record=500)
        self.threshold = 0.7
        self.action = "fwd"
        self.input_key = ("u", "reward", None)
        self.weights = np.zeros(self.N)
        self.eta = 0.1
        self.output = np.zeros((self.N, 1))

    def _logic(self, inputs: tuple,
               simulate: bool=False) -> np.ndarray:

        u = inputs[0].flatten()

        if inputs[1] > 0:
            v = np.clip(self.leaky_var(x=inputs[1],
                                       simulate=simulate),
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

    def render(self, return_fig: bool=False):

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

        if return_fig:
            return self.fig


class Acetylcholine(Modulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.leaky_var = LeakyVariableWrapper1D(eq=1., tau=10,
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
        self.leaky_var = LeakyVariableWrapper1D(eq=1., tau=1000,
                                       name="Ftg",
                                       max_record=500)
        self.action = "fwd"
        self.input_key = ("reward", None)
        self.output = np.zeros((1, 1))

    def _logic(self, inputs: tuple,
               simulate: bool=False) -> np.ndarray:

        if inputs[0] > 0:
            v = self.leaky_var(x=-1, simulate=simulate)
        else:
            v = self.leaky_var(simulate=simulate)

        self.output = v
        self.value = v
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
        self.leaky_var = LeakyVariableWrapper1D(eq=0., tau=3,
                                       name="Bnd")
        self.action = "fwd"
        self.input_key = ("u", "collision")
        self.threshold = threshold
        self.weights = np.zeros(self.N)
        self.eta = eta
        self.output = np.zeros((self.N, 1))

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

    def super_render(self, return_fig: bool=False):

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

        if return_fig:
            return self.fig_bnd


class PositionTrace(Modulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.leaky_var = LeakyVariableWrapperND(eq=0., tau=50,
                                       name="dPos",
                                       ndim=2,
                                       max_record=500)
        self.output = self.get_leaky_v().reshape(-1, 1)
        self.threshold = 0.9
        self.action = "fwd"
        self.input_key = ("position", None)
        self.output = 1.

    def _logic(self, inputs: tuple, simulate: bool=False):

        self.leaky_var.set_eq(inputs[0].reshape(-1, 1))
        v = self.leaky_var(x=np.zeros(len(self.leaky_var)),
                           simulate=simulate)

        res = np.abs(inputs[0].reshape(-1, 1) - v)

        out = [res.sum(),
               self.get_leaky_v()]
        self.output = out[0] * (out[0] > 0.25)
        self.value = out[0]
        return out

    def render(self, return_fig: bool=False):

        if not self._visualize:
            return

        self.ax.clear()

        self.ax.bar(range(2), self.output,
                    color="orange")
        self.ax.set_ylim(0., 0.8)
        self.ax.set_title(f"Position Trace " + \
            f"{np.abs(self.output).max():.2f}")
        self.fig.canvas.draw()

        if return_fig:
            return self.fig


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
                self.fig, self.ax = plt.subplots(figsize=(4, 4))
                logger(f"%visualizing {self.__class__}")

    def _logic(self, inputs: tuple, simulate: bool=False):

        self.output = inputs[0].max()
        self.value = inputs[0].max()
        if not simulate:
            self.activity = inputs[0].flatten()
        return self.output

    def render(self, return_fig: bool=False):

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

        if return_fig:
            return self.fig


class ActionSmoothness:

    def __init__(self, gain: float=None):

        self.prev_action = None

    def __call__(self, action: np.ndarray):

        if self.prev_action is None:
            self.prev_action = action
            return 0.

        similarity = utils.cosine_similarity_vec(
                        self.prev_action, action)

        return similarity

    def update(self, action: np.ndarray):
        self.prev_action = action

    def reset(self):
        self.prev_action = None


class TargetProg(Program):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.input_key = ("u", None)
        self.output = np.zeros((self.N, 1))
        self._activity = np.zeros(self.N)

        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=FIGSIZE)
            logger(f"%visualizing {self.__class__}")

    def _logic(self, inputs: tuple, simulate: bool=False):

        self.output = inputs[0].max()
        self.value = inputs[0].max()
        if not simulate:
            self.activity = inputs[0].flatten()
        return self.output

    def render(self, return_fig: bool=False):

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

        if return_fig:
            return self.fig


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
            self.fig, self.ax = plt.subplots(figsize=FIGSIZE)
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

    def render(self, pcnn_plotter: object=None,
               bounds: np.ndarray=None,
               return_fig: bool=False):

        # render singular circuit
        cir_figs = []
        for _, circuit in self.circuits.items():
            if hasattr(circuit, "weights") and \
                pcnn_plotter is not None:
                cir_figs += [circuit.render_field(
                            pcnn_plotter=pcnn_plotter,
                            bounds=bounds,
                            return_fig=return_fig)]
            else:
                cir_figs += [circuit.render(return_fig=return_fig)]

        # render dashboard
        if self.visualize:

            self.ax.clear()
            self.ax.bar(range(len(self.names)), self.values,
                         color="purple")
            self.ax.set_xticks(range(len(self.names)))
            self.ax.set_xticklabels([f"{n}\n{v:.2f}" for n, v \
                in zip(self.names, self.values)])
            self.ax.set_title("Circuits")
            self.ax.set_ylim(0., 2.)
            self.ax.grid()

            if self._number is not None:
                self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
                return

            self.fig.canvas.draw()

            if return_fig:
                return [self.fig] + cir_figs
        if return_fig:
            return cir_figs


""" --- some MODULE --- """


class ExperienceModule(ModuleClass):

    """
    Input:
        x: 2D position [array]
        mode: mode [str] ("current" or "proximal")
    Output:
        representation: output [array]
        current position: [array]
    """

    def __init__(self, pcnn: pcore.PCNN,
                 circuits: Circuits,
                 trg_module: object,
                 pcnn_plotter: object=None,
                 action_delay: float=2.,
                 weights: dict=None,
                 speed: int=0.005,
                 max_depth: int=10,
                 visualize: bool=False,
                 visualize_action: bool=False,
                 number: int=None,
                 **kwargs):

        super().__init__()
        self.pcnn = pcnn
        self.circuits = circuits
        self.pcnn_plotter = pcnn_plotter
        self.trg_module = trg_module
        self._action_smoother = ActionSmoothness()
        self.output = {
                "delta_update": np.zeros(pcnn.get_size()),
                "velocity": np.zeros(2),
                "action_idx": None,
                "score": None,
                "depth": action_delay}

        # --- action generation configuration
        # self.action_policy_main = ActionSampling2D(speed=speed,
        #                                          visualize=visualize_action,
        #                                          number=number,
        #                                          name="SamplingMain")
        # self.action_policy_int = ActionSampling2D(speed=speed,
        #                                         visualize=False,
        #                                         number=None,
        #                                         name="SamplingInt")
        self.action_policy_main = ActionSampling2DWrapper(speed=speed,
                                                 visualize=visualize_action,
                                                 number=number,
                                                 name="SamplingMain")
        self.action_policy_int = ActionSampling2DWrapper(speed=speed,
                                                visualize=False,
                                                number=None,
                                                name="SamplingInt")
        self.action_space_len = len(self.action_policy_int)
        self.action_delay = action_delay # ---
        self.max_depth = max_depth
        self.weights = weights

        # internal directives
        self.directive = {
            "state": "new",
            "onset": 0,
            "action_t": 0,
            "depth": 0,
        }
        self.t = 0

        # --- visualization
        self.record = []
        self.rollout = {
            "trajectory": [],
            "action_sequence": [],
            "score_sequence": [],
            "index_sequence": []}

    def _logic(self, observation: dict):

        """
        the level of acetylcholine is determined
        affects the next weight update
        """

        # --- get representations
        # spatial_repr = self.pcnn(x=observation["position"])
        # observation["u"] = spatial_repr.copy()

        # the target is factored in
        self.trg_module(observation=observation)

        # --- generate an action
        # self._generate_action(observation=observation)
        self._generate_action_from_rollout(observation=observation)

        # --- output & logs
        self.record += [observation["position"].tolist()]
        # self.output["u"] = spatial_repr
        self.output["delta_update"] = self.pcnn.get_delta_update()
        self.t += 1

    def _generate_action(self, observation: dict,
                         returning: bool=False) -> np.ndarray:

        """
        generate a new action given an observation
        """

        self.action_policy_int.reset()
        done = False # when all actions have been tried

        while True:

            # --- generate a random action
            action, done, action_idx = self.action_policy_int()


            # --- simulate its effects
            # new position if the action is taken
            new_position = observation["position"] + action

            score, values = self._evaluate_action(position=new_position,
                                          action=action,
                                          action_idx=action_idx)

            # score = 1 / (1 + np.exp(-score))

            # it is the best available action
            if done:
                break

            # try again
            self.action_policy_int.update(score=score)

        # ---
        if returning:
            return action, action_idx, score, values

        self.output["velocity"] = action
        self.output["action_idx"] = action_idx
        self.output["score"] = score

    def _evaluate_action(self, position: np.ndarray,
                         action: np.ndarray,
                         action_idx: int) -> float:

        # new observation/effects
        u = self.pcnn.fwd_ext(x=position)

        new_observation = {
            "u": u,
            "position": position,
            "velocity": action,
            "collision": False,
            "reward": 0.,
            "delta_update": 0.}

        modulation = self.circuits(
                        observation=new_observation,
                        simulate=True)
        trg_modulation = self.trg_module.evaluate_direction(
                        velocity=action)

        # --- evaluate the effects
        # relevant modulators
        values = np.array([modulation["Bnd"].item(),
                           modulation["dPos"].item(),
                           modulation["Pop"].item(),
                           trg_modulation,
                           self._action_smoother(action=action)])
        score = (values @ self.weights.T)# / np.abs(self.weights).sum()

        if action_idx == 4:
            score = -1.

        return score, np.around(values*self.weights, 3)

    def _action_rollout(self, observation: dict,
                        action: np.ndarray,
                        action_idx: int) -> tuple:

        #
        rollout_scores = []
        rollout_values = []
        action_seq = [action]
        index_seq = [action_idx]

        # first step + evaluate
        observation["position"] = observation["position"] + action
        trajectory = [observation["position"]]
        score, values = self._evaluate_action(position=observation["position"],
                                                 action=action,
                                                 action_idx=action_idx)
        rollout_scores += [score]
        rollout_values += [values.tolist()]
        rollout_scores[0] = round(rollout_scores[0], 4)

        # action smoothing
        self._action_smoother.reset()
        self._action_smoother(action=action)

        # --- simulate its effects over the next steps
        for t in range(self.max_depth):

            # get the current action
            action, action_idx, score, values = self._generate_action(
                                observation=observation,
                                returning=True)

            # step
            observation["position"] = observation["position"] + action
            self._action_smoother.update(action=action)

            # save the score
            rollout_scores += [round(score, 4)]
            trajectory += [observation["position"]]
            action_seq += [action]
            index_seq += [action_idx]
            rollout_values += [values.tolist()]

        return rollout_scores, trajectory, action_seq, index_seq, rollout_values

    def _generate_action_from_rollout(self,
                        observation: dict) -> np.ndarray:

        """
        generate a new action given an observation
        through a rollout over many steps
        """

        # --- `keep` current directive
        if self.directive["state"] == "keep" and \
            not observation["collision"]:

            # apply the plan
            self.output["velocity"] = self.rollout["action_sequence"
                            ][self.directive["action_t"]]
            self.output["action_idx"] = self.rollout["index_sequence"
                            ][self.directive["action_t"]]

            # check whether to keep or renew
            if self.directive["action_t"] >= self.directive["depth"]:
                self.directive["state"] = "new"
            self.directive["action_t"] += 1

            return

        # --- `new` directive
        self.action_policy_main.reset()
        done = False

        best_score = -np.inf
        depth = 0
        best_rollout = []
        all_scores = []
        while True:

            # --- generate random main action
            action, done, action_idx = self.action_policy_main()

            # --- evaluate action
            results = self._action_rollout(observation=observation.copy(),
                                                  action=action,
                                                  action_idx=action_idx)
            self.action_policy_main.update()

            rollout_scores = results[0]
            trajectory = results[1]
            action_seq = results[2]
            index_seq = results[3]
            rollout_values = results[4]

            # evaluate the best action
            if np.sum(rollout_scores[1:]) > best_score:
                best_score = np.sum(rollout_scores[1:])
                best_rollout = [np.stack(trajectory), rollout_scores,
                                action_seq, index_seq,
                                len(rollout_scores[1:]),
                                rollout_values]

            if done:
                break
            all_scores += [[action, rollout_scores]]

        # if best_score <= 0.:
        #     logger.warning("no good action found")
        #     for a, scores in all_scores:
        #         logger.debug(f"{a} | {scores}")

        #     raise ValueError("no good action found")

        # --- record result
        depth = best_rollout[4]
        self.output["velocity"] = best_rollout[2][0]
        self.output["action_idx"] = best_rollout[3][0]
        self.output["score"] = best_rollout[1][0]
        self.directive["state"] = "keep"
        self.directive["onset"] = self.t
        self.directive["depth"] = depth
        self.directive["action_t"] = 1

        self.rollout["trajectory"] = best_rollout[0][:depth+2]
        self.rollout["score_sequence"] = best_rollout[1][:depth+2]
        self.rollout["action_sequence"] = best_rollout[2][:depth+2]
        self.rollout["index_sequence"] = best_rollout[3][:depth+2]

        # logger.debug(f"rollout [{depth=}] ||\n" + \
        #     f"actions: {self.rollout['action_sequence']}\n" + \
        #     f"indexes: {self.rollout['index_sequence']}\n"
        #     f"scores: {best_rollout[1]}\n" + \
        #     f"values: {best_rollout[5]}")
        # logger.debug(f"best action: {best_action} | {best_action_idx} | {best_score}")
        logger.debug(f"plan xy:\n{np.around(self.rollout['trajectory'], 4).tolist()}")

    def render(self, ax=None, **kwargs):

        return_fig = kwargs.get("return_fig", False)

        fig_api = self.action_policy_int.render(return_fig=return_fig)
        fig_tm = self.trg_module.render(return_fig=return_fig)

        if self.pcnn_plotter is not None:
            title = f"$t=${self.t} | #PCs={len(self.pcnn)}"
            fig_pp = self.pcnn_plotter.render(ax=None,
                trajectory=kwargs.get("trajectory", False),
                rollout=[self.rollout["trajectory"],
                         self.rollout["score_sequence"]],
                new_a=1*self.circuits.circuits["DA"].output,
                return_fig=return_fig,
                alpha_nodes=kwargs.get("alpha_nodes", 0.8),
                alpha_edges=kwargs.get("alpha_edges", 0.5),
                customize=True,
                title=title)

        if return_fig:
            return fig_pp, fig_tm, fig_api



    def reset(self, complete: bool=False):
        super().reset(complete=complete)
        if complete:
            self.record = []


class TargetModule(ModuleClass):

    """
    Input:
        x: 2D position [array]
        mode: mode [str] ("current" or "proximal")
    Output:
        representation: output [array]
        current position: [array]
    """

    def __init__(self, pcnn: pcore.PCNN,
                 circuits: Circuits,
                 visualize: bool=True,
                 threshold: float=0.4,
                 speed: int=0.005,
                 max_depth: int=10,
                 number: int=None):

        super().__init__()
        self.pcnn = pcnn
        self.circuits = circuits
        self.speed = speed
        self.max_depth = max_depth
        self.threshold = threshold
        self.output = {
                "u": np.zeros(pcnn.get_size()),
                "trg_position": np.zeros(2),
                "velocity": np.zeros(2),
                "score": 0.}

        # --- visualization
        self.visualize = visualize
        self._number = number
        if visualize:
            self.fig, self.ax = plt.subplots(figsize=FIGSIZE)
            logger(f"%visualizing {self.__class__}")

    def _logic(self, observation: dict):

        # --- modulation
        modulation = self.circuits.circuits["DA"].weights

        # --- get representations
        curr_repr = self.pcnn.fwd_ext(
                            x=observation["position"])
        curr_pos = observation["position"]

        trg_repr, flag = self._converge_to_location(
                x=np.zeros((self.pcnn.get_size(), 1)),
                depth=0,
                modulation=modulation,
                threshold=0.8)
        centers = self.pcnn.get_centers()

        # set -np.inf values to zero
        centers = np.where(centers == -np.inf, 0, centers)
        trg_pos = pcore.calc_position_from_centers(a=trg_repr,
                                    centers=centers)

        # velocity
        velocity = trg_pos - curr_pos
        velocity = velocity / np.linalg.norm(velocity) * self.speed

        # intensity
        if self.circuits.circuits["Ftg"].value > self.threshold and flag:
            score = 1 * int(flag) * self.circuits.circuits["Ftg"].value
        else:
            score = 0.

        # --- output
        assert isinstance(score, float), f"{type(score)=}"
        self.output = {
                "u": trg_repr,
                "trg_position": trg_pos,
                "velocity": velocity,
                "score": score}

    def _converge_to_location(self, x: np.ndarray,
                              depth: int,
                              modulation: np.ndarray,
                              threshold: float=0.1):

        # u = self.pcnn._Wrec @ (x + modulation.reshape(-1, 1))
        u = self.pcnn.fwd_int(x.reshape(-1) + modulation.reshape(-1))

        c = pcore.cosine_similarity_vec(u, x)
        if c > threshold:
            return u, True

        if depth >= self.max_depth:
            return u, False

        return self._converge_to_location(x=u,
                                          depth=depth+1,
                                          modulation=modulation,
                                          threshold=threshold)

    def evaluate_direction(self, velocity: np.ndarray) -> float:

        # compare a queried velocity with the
        # calculated target velocity
        if self.output["score"] > 0.:
            similarity = utils.cosine_similarity_vec(
                            self.output["velocity"],
                            velocity)

            self.output["similarity"] = np.array([-1 * similarity])
        else:
            similarity = 0.

        assert isinstance(similarity, float), f"wrong type {type(similarity)=}"

        return similarity

    def render(self, **kwargs):

        if not self.visualize:
            return

        self.ax.clear()
        self.ax.scatter(*self.output["trg_position"].flatten(),
                    marker="x", color="red",
                    s=100)
        self.ax.set_title(f"Target Module | " + \
            f" I={self.output['score']:.4f}")
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

        if kwargs.get("return_fig", False):
            return self.fig

    def reset(self, complete: bool=False):
        super().reset(complete=complete)



""" --- high level MODULES --- """


class Brain:

    def __init__(self, exp_module: ExperienceModule,
                 circuits: Circuits,
                 pcnn2D: pcore.PCNN,
                 number: int=None,
                 plot_intv: int=1):

        self.exp_module = exp_module
        self.circuits = circuits
        self.pcnn2D = pcnn2D

        # self.movement = None
        # self.observation_ext = {
        #     "position": np.zeros(2).astype(float),
        #     "collision": False
        # }

        self.state = {
            "u": np.zeros(pcnn2D.get_size()),
            "delta_update": 0.,
            "velocity": np.zeros(2),
            "action_idx": 0,
            "depth": 0,
            "score": 0,
            "onset": 0,
            "trg_u": np.zeros(exp_module.pcnn.get_size()),
            "trg_position": np.zeros(2),
            "trg_velocity": np.zeros(2),
            "importance": 0,
        }

        # self.state = self.observation_ext | self.observation_int

        self.directive = "new"
        self._elapsed_time = 0
        # self._plot_intv = plot_intv  # no longer used
        self.t = -1
        self.number = number
        self.frequencies = np.zeros(self.exp_module.action_space_len)

        if number is not None:
            self.fig, self.ax = plt.subplots(figsize=FIGSIZE)

        # record
        self.record = {"trajectory": []}

    def __call__(self, observation: dict):

        self.t += 1
        # self.observation_ext = observation

        # forward current position
        # spatial_repr = self.pcnn2D(x=observation["position"])
        # observation["u"] = spatial_repr

        self.state["u"] = self.pcnn2D(x=observation["position"])
        self.state = self.state | observation

        # --- update modulation
        # mod_observation = self.circuits(observation=self.state)
        mod_observation = self.circuits(observation=self.state)
        self.state = self.state | mod_observation

        # self.state = self.observation_ext | self.observation_int

        # TODO : add directive depending on modulation

        # set new directive
        # if self.t == 0:
        #     self.directive = "new"

        # if self.directive == "force_keep" and \
        #     (self.t - self.observation_int["onset"]) < \
        #     self.observation_int["depth"]:

        #     return self.movement

#         elif self.directive == "force_keep":
#             self.directive = "new"

        # elif mod_observation["dPos"] < 0.05:
        #     self.directive = "new"

        # else:
        #     self.directive = "keep"

        # --- update experience module
        # full output
        # self.observation_int = self.exp_module(observation=observation)
        exp_output = self.exp_module(observation=self.state)

        # --- update output
        # self.observation_int["onset"] = self.t
        # self.directive = "force_keep"
        self.movement = exp_output["velocity"]

        self.record["trajectory"] += [observation["position"].tolist()]
        # self.frequencies[self.observation_int["action_idx"]] += 1
        self.state = observation | exp_output

        return exp_output["velocity"]

    def routines(self, wall_vectors: np.ndarray):

        # pcnn routine
        # self.exp_module.pcnn.clean_recurrent(
        #                         wall_vectors=wall_vectors)
        pass

    def render(self, **kwargs):

        fig_cir = self.circuits.render(
            pcnn_plotter=self.exp_module.pcnn_plotter,
            bounds=self.exp_module.pcnn_plotter._bounds,
            return_fig=kwargs.get("return_fig", False))

        fig_exp = self.exp_module.render(ax=kwargs.get("ax", None),
                          trajectory=np.array(
                               self.record["trajectory"]) * \
                                         kwargs.get("use_trajectory", False),
                                     return_fig=kwargs.get("return_fig", False),
                            alpha_nodes=kwargs.get("alpha_nodes", 0.8),
                            alpha_edges=kwargs.get("alpha_edges", 0.5))

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

        if kwargs.get("return_fig", False):
            return list(fig_exp) + fig_cir

    @property
    def render_values(self):
        return self.circuits.values.copy(), self.circuits.names


""" policies """


class ActionSampling2D:

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
            self.fig, self.ax = plt.subplots(figsize=FIGSIZE)
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
            return self._velocity.copy(), False, self._idx

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
            return self._velocity.copy(), True, self._idx

        # --- sample again
        p = self._p[self._available_idxs].copy()
        p /= p.sum()
        # self._idx = np.random.choice(
        #                 self._available_idxs,
        #                 p=p)
        self._idx = np.random.choice(
                        self._available_idxs)
        self._available_idxs.remove(self._idx)
        self._velocity = self._samples[self._idx]

        return self._velocity.copy(), False, self._idx

    def update(self, score: float=0.):

        # --- normalize the score

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
               action_values: np.ndarray=None,
               return_fig: bool=False):

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
            self._values[4] = (self._values[:4].sum() + self._values[5:].sum()) / 8
            self.ax.imshow(self._values.reshape(3, 3),
                           cmap="Blues_r",
                           aspect="equal",
                           interpolation="nearest")

        # labels inside each square
        for i in range(3):
            for j in range(3):
                if values is not None:
                    text = "".join([f"{np.around(v, 2)}\n" for v in values[3*i+j]])
                else:
                    # text = f"{self._samples[3*i+j][1]:.3f}\n" + \
                    #       f"{self._samples[3*i+j][0]:.3f}"
                    text = f"{self._values[3*i+j]:.3f}"
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

        if return_fig:
            return self.fig

    def reset(self):
        self._idx = None
        self._available_idxs = list(range(self._num_samples))
        self._p = np.ones(self._num_samples) / self._num_samples
        self._velocity = self._samples[0]
        self._values = np.zeros(self._num_samples)

    def has_collided(self):

        self._velocity = -self._velocity


class ActionSampling2DWrapper(pclib.ActionSampling2D):

    def __init__(self, speed: float, name: str,
                 visualize: bool=False,
                 number: int=None):

        """
        Parameters
        ----------
        speed : float, optional
            Speed of the agent. The default is 0.1.
        name : str, optional
            Name of the policy. The default is None.
        visualize : bool, optional
            Visualize the policy. The default is False.
        number : int, optional
            Number of the figure. The default is None.
        """

        super().__init__(speed=speed, name=name)

        # render
        self._number = number
        self.visualize = visualize
        if visualize:
            self.fig, self.ax = plt.subplots(figsize=FIGSIZE)
            logger(f"%visualizing {self.__class__}")

    def __call__(self, **kwargs):
        out = list(super().__call__(*kwargs))
        out[0] = np.array(out[0])
        return out

    def render(self, values: np.ndarray=None,
               return_fig: bool=False):

        if not self.visualize:
            return

        self.ax.clear()

        values = values if values is not None else self.get_values()

        values[4] = (values[:4].sum() + values[5:].sum()) / 8
        self.ax.imshow(values.reshape(3, 3),
                       cmap="Blues_r",
                       aspect="equal",
                       interpolation="nearest")

        # labels inside each square
        for i in range(3):
            for j in range(3):
                if values is not None:
                    text = "".join([f"{np.around(v, 2)}\n" for v in values[3*i+j]])
                else:
                    text = f"{self._values[3*i+j]:.3f}"
                self.ax.text(j, i, f"{text}",
                             ha="center", va="center",
                             color="black",
                             fontsize=13)

        self.ax.set_xlabel("Action")
        self.ax.set_title(f"Action Space")
        self.ax.set_yticks(range(3))
        self.ax.set_xticks(range(3))

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return

        self.fig.canvas.draw()

        if return_fig:
            return self.fig
