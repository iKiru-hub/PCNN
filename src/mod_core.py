import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import time

import utils_core as utc

try:
    import libs.pclib as pclib
except ImportError:
    import warnings
    warnings.warn("pclib [c++] not found, using python version")
    import libs.pclib1 as pclib


""" INITIALIZATION """

FIGPATH = utc.FIGPATH

FIGSIZE = (4, 4)


def set_seed(seed: int=0):
    np.random.seed(seed)

logger = utc.setup_logger(name="MOD",
                          level=-1,
                          is_debugging=True,
                          is_warning=False)

def edit_logger(level: int=-1,
                is_debugging: bool=True,
                is_warning: bool=False):
    global logger
    logger.set_level(level)
    logger.set_debugging(is_debugging)
    logger.set_warning(is_warning)


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
        self._fig_standalone = kwargs.get("fig_standalone", False)
        self.mod_weight = kwargs.get("mod_weight", 1.)
        self._pcnn_plotter2d = kwargs.get("pcnn_plotter2d", None)

        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=FIGSIZE)
            logger(f"%visualizing {self.__class__}")

            logger.debug(f"{self.__class__.__name__} ax={self.ax}")

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

    def render_field(self, pcnn_plotter: object=None,
                     return_fig: bool=False, **kwargs):

        if not self._visualize:
            return

        self.ax.clear()

        if self._pcnn_plotter2d is not None:
            self._pcnn_plotter2d.render(ax=self.ax,
                                trajectory=None,
                                new_a=self.weights,
                                edges=False,
                                alpha_nodes=0.8,
                                cmap="Greens",
                                customize=True,
                                title=f"{self.leaky_var.name} |" + \
                    f" $w_{{max}}=${self.weights.max():.3f}")


        elif pcnn_plotter is not None:
            pcnn_plotter.render(ax=self.ax,
                                trajectory=None,
                                new_a=self.weights,
                                edges=False,
                                alpha_nodes=0.8,
                                cmap="Greens",
                                customize=True,
                                title=f"{self.leaky_var.name} |" + \
                    f" $w_{{max}}=${self.weights.max():.3f}")

        # if bounds is not None:
        #     self.ax.set_xlim(bounds[0], bounds[1]*1.2)
        #     self.ax.set_ylim(bounds[2], bounds[3]*1.2)

        # self.ax.set_xticks([bounds[0], bounds[1]])
        # self.ax.set_yticks([bounds[2], bounds[3]])
        # self.ax.set_xticklabels([f"{bounds[0]:.2f}", f"{bounds[1]:.2f}"])
        # self.ax.set_yticklabels([f"{bounds[2]:.2f}", f"{bounds[3]:.2f}"])

        if self._fig_standalone:
            self.fig.canvas.draw()

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
        self.threshold = kwargs.get("threshold", 0.01)
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

            if not simulate:

                # potentiation
                self.weights += self.eta * v * \
                    np.where(u < self.threshold, 0, u)

                # depression
                # self.weights -= self.eta * (1 - v) * \
                #     np.where(u < self.threshold, 0, u)
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
        self.leaky_var = LeakyVariableWrapper1D(eq=1.,
                                    tau=kwargs.get("tau", 700),
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
                 tau: float=3, **kwargs):

        super().__init__(**kwargs)
        self.leaky_var = LeakyVariableWrapper1D(eq=0., tau=tau,
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

            if not simulate:
                self.weights += self.eta * v * \
                    np.where(u < self.threshold, 0, u)
                self.weights = np.clip(self.weights, 0., 1.)
        else:
            v = self.leaky_var(x=0, simulate=simulate)

        out = self.weights.reshape(1, -1) @ u.reshape(-1, 1)
        out = utc.generalized_sigmoid(x=out, alpha=0.5,
                                      beta=200., clip_min=0.001).item()
        self.output = out
        self.value = out
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
        self.output = out[0] * (out[0] > 0.25).item()
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

        if np.all(self.prev_action) == None:
            self.prev_action = action
            return 0.

        similarity = utc.cosine_similarity_vec(
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
                 other_circuits: dict=None,
                 visualize: bool=False,
                 number: int=None):

        self.circuits = circuits_dict
        self.names = tuple(circuits_dict.keys())
        logger(f"circuits : {self.names}")
        self.output = {name: circuits_dict[name].output \
            for name in self.names}
        self.values = np.zeros(len(self.names))

        # other circuits
        self.other_circuits = other_circuits
        if other_circuits is not None:
            self.names = self.names + tuple(other_circuits.keys())

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

        return self.output, self.values

    def render_circuits(self, pcnn_plotter: object=None,
                      return_fig: bool=False):

        # render singular circuit
        cir_figs = []
        for _, circuit in self.circuits.items():
            if hasattr(circuit, "weights"):# or \
                # pcnn_plotter is not None:
                cir_figs += [circuit.render_field(
                            pcnn_plotter=pcnn_plotter,
                            return_fig=return_fig)]
            else:
                cir_figs += [circuit.render(return_fig=return_fig)]

        return cir_figs

    def render(self, pcnn_plotter: object=None,
               bounds: np.ndarray=None,
               return_fig: bool=False):

        # render singular circuit
        cir_figs = self.render_circuits(pcnn_plotter=pcnn_plotter,
                           return_fig=return_fig)

        # render dashboard
        if self.visualize:

            if self.other_circuits is not None:
                values = self.values.tolist() + \
                    [obj.get_value() for obj in self.other_circuits.values()]
            else:
                values = self.values

            self.ax.clear()
            self.ax.bar(range(len(self.names)), values,
                         color="purple")
            self.ax.set_xticks(range(len(self.names)))
            self.ax.set_xticklabels([f"{n}\n{v:.2f}" for n, v \
                in zip(self.names, values)])
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

    def __init__(self, pcnn: object,
                 circuits: Circuits,
                 trg_module: object,
                 pcnn_plotter: object=None,
                 action_delay: float=1.,
                 weights: dict=None,
                 speed: int=0.005,
                 max_depth: int=10,
                 visualize: bool=False,
                 visualize_action: bool=False,
                 number: int=None,
                 number2: int=None,
                 **kwargs):

        super().__init__()
        self.pcnn = pcnn
        self.circuits = circuits
        self.pcnn_plotter = pcnn_plotter
        self.trg_module = trg_module
        self._action_smoother = ActionSmoothness()
        self.output = {
                "velocity": np.zeros(2),
                "action_idx": None,
                "score": None,
                "depth": action_delay}

        # --- action generation configuration
        self.speed = speed
        self.max_depth = max_depth
        self.action_delay = action_delay # ---
        self.action_policy_main = ActionSampling2DWrapper(speed=speed * action_delay,
                                                 visualize=visualize_action,
                                                 number=None,
                                                 name="SamplingMain")
        self.action_policy_int = ActionSampling2DWrapper(speed=speed * action_delay,
                                                visualize=False,
                                                number=None,
                                                name="SamplingInt")
        self.action_space_len = len(self.action_policy_int)

        if isinstance(weights, dict):
            self.eval_network = pclib.TwoLayerNetwork(
                    weights["hidden"].tolist(),
                    weights["output"].tolist())
            self.using_mlp = True
        else:
            self.eval_network = OneLayerNetworkWrapper(weights=weights.tolist())
            self.using_mlp = False

        # internal directives
        self.directive = {
            "state": "new",
            "onset": 0,
            "trg_position": None,
            "action_t": 0,
            "depth": 0,
        }
        self.t = 0

        # --- visualization
        self.visualize = visualize
        self._number = number
        self._number2 = number2
        if visualize:
            self.fig, self.ax1 = plt.subplots(figsize=FIGSIZE)
            self.fig2, self.ax2 = plt.subplots(figsize=FIGSIZE)
            logger(f"%visualizing {self.__class__}")

        self.record = []
        self.rollout = {
            "trajectory": [],
            "action_sequence": [],
            "score_sequence": [],
            "index_sequence": [],
            "hidden_sequence": [],
            "values_sequence": [],
            "score": 0.0,
        }
        self._mod_names = ("Bnd", "dPos", "Pop", "Trg", "Act", "nxP")

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
        self._planning(observation=observation)

        # --- output & logs
        self.record += [observation["position"].tolist()]
        self.t += 1

    def _planning(self, observation):


        """
        generate a new action given an observation
        through a rollout over many steps
        """

        # --- `keep` current directive
        if self.directive["state"] == "keep" and \
            not observation["collision"]:

            # check whether it's time to step to the next action
            action, completed, valid = self._check_plan_step(observation=observation)

            # exit 1: invalid action
            if valid:

                # the step is completed
                if completed:

                    # exit 2: the plan is completed
                    if self.directive["action_t"] >= self.directive["depth"]:
                        self.directive["state"] = "new"

                    # go to the next step
                    else:
                        self.directive["trg_position"] = self.rollout["trajectory"][self.directive["action_t"]]
                        action, _ = self._make_action(
                            position=observation["position"],
                            target=self.directive["trg_position"])
                        self.output["velocity"] = action
                        self.directive["action_t"] += 1
                        return

                # the step is not completed
                else:
                    self.output["velocity"] = action
                    self.directive["state"] = "keep"
                    return

       # --- `new` directive
        self._generate_action_from_rollout(observation=observation)

    def _generate_action(self, observation: dict,
                         returning: bool=False,
                         position_list: []=None) -> np.ndarray:

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

            evaluation = self._evaluate_action(
                            position=new_position,
                            action=action,
                            action_idx=action_idx)
            score, hidden, values = evaluation

            # score = 1 / (1 + np.exp(-score))

            # check that the new position is actually new
            if position_list is not None:
                if new_position.tolist() in position_list:
                    score = -1.
                    # values += [-1.]

            # it is the best available action
            if done:
                break

            # try again
            self.action_policy_int.update(score=score)

        # ---
        if returning:
            return action, action_idx, score, hidden, values

        self.output["velocity"] = action
        self.output["action_idx"] = action_idx
        self.output["score"] = score

    def _evaluate_action(self, position: np.ndarray,
                         action: np.ndarray,
                         action_idx: int) -> float:

        if action_idx == 4:
            # assuming no MLP
            return -np.inf, np.zeros(5), [0., 0., 0., 0., 0.]

        # new observation/effects
        u = self.pcnn.fwd_ext(x=position)

        new_observation = {
            "u": u,
            "position": position,
            "velocity": action,
            "collision": False,
            "reward": 0.,
            "delta_update": 0.}

        modulation, _ = self.circuits(
                        observation=new_observation,
                        simulate=True)
        trg_modulation = self.trg_module.evaluate_direction(
                        velocity=action)

        # --- evaluate the effects
        # relevant modulators
        values = [modulation["Bnd"],
                  modulation["dPos"],
                  modulation["Pop"] * int(trg_modulation <= 0.),
                  trg_modulation,
                  self._action_smoother(action=action)]

        score, hidden = self.eval_network(values)

        return score, hidden, values

    def _action_rollout(self, observation: dict,
                        action: np.ndarray,
                        action_idx: int) -> tuple:

        #
        action_seq = [action]
        index_seq = [action_idx]

        # first step + evaluate
        observation["position"] = observation["position"] + action
        trajectory = [observation["position"]]
        position_list = [observation["position"].tolist()]
        evaluation = self._evaluate_action(position=observation["position"],
                                           action=action,
                                           action_idx=action_idx)
        rollout_scores = [round(evaluation[0], 4)]
        rollout_hidden = [evaluation[1]]
        rollout_values = [evaluation[2]]
        # rollout_scores[0] = round(rollout_scores[0], 4)

        # action smoothing
        self._action_smoother.reset()
        self._action_smoother(action=action)

        # --- simulate its effects over the next steps
        for t in range(self.max_depth):

            # get the current action
            results = self._generate_action(
                                observation=observation,
                                returning=True,
                                position_list=position_list)
            action, action_idx, score, hidden, values = results

            # step
            observation["position"] = observation["position"] + action
            position_list += [observation["position"].tolist()]
            self._action_smoother.update(action=action)

            # save the score
            rollout_scores += [round(score, 4)]
            trajectory += [observation["position"]]
            action_seq += [action]
            index_seq += [action_idx]
            rollout_hidden += [hidden]
            rollout_values += [values]

        return rollout_scores, trajectory, action_seq, index_seq, rollout_hidden, rollout_values

    def _generate_action_from_rollout(self,
                        observation: dict) -> np.ndarray:

        """
        generate a new action given an observation
        through a rollout over many steps
        """

        # --- `new` directive
        self.action_policy_main.reset()
        done = False

        best_score = -np.inf
        best_min = -np.inf
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
            rollout_hidden = results[4]
            rollout_values = results[5]

            # evaluate the best action
            if np.sum(rollout_scores[1:]) > best_score and \
                np.min(rollout_scores) > best_min:

                best_score = np.sum(rollout_scores[1:])
                best_min = np.min(rollout_scores[1:])
                best_rollout = [np.stack(trajectory),
                                rollout_scores,
                                action_seq, index_seq,
                                len(rollout_scores[1:]),
                                rollout_hidden,
                                rollout_values]

            if done:
                break
            all_scores += [[action, rollout_scores]]

        # --- record result
        depth = best_rollout[4]

        self.rollout["trajectory"] = best_rollout[0][:depth+2]
        self.rollout["score_sequence"] = best_rollout[1][:depth+2]
        self.rollout["action_sequence"] = best_rollout[2][:depth+2]
        self.rollout["index_sequence"] = best_rollout[3][:depth+2]
        self.rollout["hidden_sequence"] = np.array(best_rollout[5])
        self.rollout["values_sequence"] = np.array(best_rollout[6])
        self.rollout["score"] = best_score

        self.directive["state"] = "keep"
        self.directive["onset"] = self.t
        self.directive["depth"] = depth
        self.directive["action_t"] = 1
        self.directive["trg_position"] = self.rollout["trajectory"][0]

        action, _ = self._make_action(
            position=observation["position"],
            target=self.directive["trg_position"])

        self.output["velocity"] = action
        self.output["action_idx"] = self.rollout["index_sequence"][0]
        self.output["score"] = self.rollout["score_sequence"][0]
        self.output["depth"] = depth

    def _check_plan_step(self, observation: dict):

        """
        check if the current plan is still valid

        Returns: (action, complete, validity)
        """

        # modulation error
        bnd_err = abs(observation["Bnd"] - \
                      self.rollout["values_sequence"][self.directive["action_t"]][0])
        dpos_err = abs(observation["dPos"] - \
                      self.rollout["values_sequence"][self.directive["action_t"]][1])
        pop_err = abs(observation["Pop"] - \
                      self.rollout["values_sequence"][self.directive["action_t"]][2])

        # exit 1: boundary hit
        if bnd_err > 0.01:
            return None, None, False

        # distance from next checkpoint
        action, distance = self._make_action(
            position=observation["position"],
            target=self.rollout["trajectory"][self.directive["action_t"]]
        )

        # exit 2: checkpoint reached
        if distance < 0.001:
            return None, True, True

        # exit 3: calculate an action
        return action, False, True

    def _make_action(self, position: np.ndarray,
                     target: np.ndarray) -> np.ndarray:

        distance_vector = target - position
        distance = np.linalg.norm(distance_vector)

        if distance < self.speed:
            return distance_vector, distance

        return distance_vector / distance * self.speed, distance

    def render(self, ax=None, **kwargs):

        if ax is not None:

            ax.plot(self.rollout["trajectory"][:, 0],
                   self.rollout["trajectory"][:, 1],
                    '-', color="blue", alpha=0.5, lw=2)
            ax.scatter(self.rollout["trajectory"][:, 0],
                       self.rollout["trajectory"][:, 1],
                       c=np.array(self.rollout["score_sequence"])/10,
                       cmap="hot",
                       s=30)
            return

        return_fig = kwargs.get("return_fig", False)

        fig_api = self.action_policy_main.render(
                            return_fig=return_fig)
        fig_tm = self.trg_module.render(return_fig=return_fig)

        if self.pcnn_plotter is not None:
            title = f"$t=${self.t} | #PCs={len(self.pcnn)}"
            fig_pp = self.pcnn_plotter.render(ax=None,
                trajectory=kwargs.get("trajectory", False),
                rollout=[self.rollout["trajectory"],
                         self.rollout["score_sequence"]],
                new_a=1*self.circuits.circuits["DA"].output,
                return_fig=return_fig,
                render_elements=True,
                alpha_nodes=kwargs.get("alpha_nodes", 0.8),
                alpha_edges=kwargs.get("alpha_edges", 0.5),
                customize=True,
                title=title)

        # visualize the score sequence of the current plan
        if self.visualize:

            # rollout trajectory
            length = len(self.rollout["score_sequence"])
            self.ax1.clear()
            self.ax1.plot(self.rollout["score_sequence"],
                         '-', color="blue", alpha=0.7, lw=2)

            if not self.using_mlp:
                for i, v in enumerate(self.rollout["hidden_sequence"].T):
                    self.ax1.scatter(range(length), v, s=30,
                                    alpha=0.8, label=self._mod_names[i])
                self.ax1.legend(loc="lower right")

            self.ax1.axvline(x=self.directive["action_t"],
                            color="red", linestyle="--")
            self.ax1.set_ylim(-3., 6.)
            self.ax1.set_xlabel("Time")
            self.ax1.grid()
            self.ax1.set_title(f"Behaviour Score Sequence")

            self.fig.canvas.draw()

            # display the values' hidden space
            if self.using_mlp:
                self.ax2.clear()
                self.ax2.plot(*self.rollout["hidden_sequence"].T,
                              '-o', color="blue", alpha=0.7, lw=2)
                self.ax2.set_ylim(-1.5, 3.5)
                self.ax2.set_xlim(-1.5, 3.5)
                # self.ax2.set_yticks([-2., 0., 2.])
                # self.ax2.set_xticks([-2., 0., 2.])
                # self.ax2.set_xticklabels(["-2", "0", "2"])
                # self.ax2.set_yticklabels(["-2", "0", "2"])
                self.ax2.set_aspect("equal")
                self.ax2.grid()
                self.ax2.set_title(f"Values hidden space")
                self.fig2.canvas.draw()
            # plot weights
            else:
                self.eval_network.render(ax=self.ax2,
                                         labels=self._mod_names[:-1])

            if self._number is not None:
                self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
                self.fig2.savefig(f"{FIGPATH}/fig{self._number2}.png")

            if return_fig:
                return fig_pp, fig_tm, fig_api, self.fig

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

    def __init__(self, pcnn: object,
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

    def __str__(self) -> str:
        return f"TargetModule(threshold={self.threshold})"

    def _logic(self, observation: dict):

        """
        trg_repr <- converge(modulation)
        enabled
        --
        value <- compare velocities()
        """

        # --- modulation
        modulation = self.circuits.circuits["DA"].weights

        # --- get representations
        # curr_repr = self.pcnn.fwd_ext(
        #                     x=observation["position"])
        curr_pos = observation["position"]

        trg_repr, flag = self._converge_to_location(
                x=np.zeros((self.pcnn.get_size(), 1)),
                depth=0,
                modulation=modulation,
                threshold=0.8)
        centers = self.pcnn.get_centers()

        # set -np.inf values to zero
        centers = np.where(centers == -np.inf, 0, centers)
        trg_pos = utc.calc_position_from_centers(a=trg_repr,
                                    centers=centers)

        # velocity
        velocity = trg_pos - curr_pos
        velocity = velocity / np.linalg.norm(velocity) * self.speed

        # intensity
        if self.circuits.circuits["Ftg"].value > self.threshold and flag:
            score = int(flag) * self.circuits.circuits["Ftg"].value
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

        c = utc.cosine_similarity_vec(u, x)
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
            similarity = utc.cosine_similarity_vec(
                            self.output["velocity"],
                            velocity)

            self.output["similarity"] = np.array([-1 * similarity])
        else:
            similarity = 0.

        assert isinstance(similarity, float), \
            f"wrong type {type(similarity)=}"

        return similarity

    def render(self, **kwargs):

        if not self.visualize:
            return

        self.ax.clear()
        self.ax.scatter(*self.output["trg_position"].flatten(),
                    marker="x", color="red",
                    s=100)
        self.ax.set_title(f"Target Module | " + \
            f"$\\leq${self.threshold}"
            f" $I=${self.output['score']:.2f}")
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
                 pcnn2D: object,
                 densitymod: object=None):

        self.exp_module = exp_module
        self.circuits = circuits
        self.pcnn2D = pcnn2D
        self.densitymod = densitymod

        self.state = {
            "position": np.zeros(2),
            "collision": False,
            "reward": 0.,
            "u": np.zeros(pcnn2D.get_size()),
            "delta_update": 0.,
            "velocity": np.zeros(2),
            "action_idx": 0,
            "depth": 0,
            "score": 0,
        }

        # self.state = self.observation_ext | self.observation_int

        self.directive = "new"
        self._elapsed_time = 0
        self.t = -1
        self.frequencies = np.zeros(self.exp_module.action_space_len)

        # record
        self.record = {"trajectory": []}

    def __str__(self) -> str:
        return f"Brain({self.exp_module}, {self.circuits}, " + \
            f"{self.pcnn2D})"

    def __call__(self, observation: dict):

        """
        forward the observation through the brain
        """

        self.t += 1
        self.record["trajectory"] += [observation["position"].tolist()]
        self.state["position"] = observation["position"]
        self.state["collision"] = observation["collision"]
        self.state["reward"] = observation["reward"]

        # >> forward current position
        # self.state["u"] = self.pcnn2D(x=observation["position"])
        # u, _ = self.pcnn2D(v=observation["position"])
        # logger.debug(f"{self.pcnn2D}")
        # logger.debug(f"velocity: {self.state['velocity']}")
        u, y = self.pcnn2D(self.state["velocity"])
        # logger.debug(f"GCN:\n{np.around(self.pcnn2D.get_activation_gcn(), 2)}")
        self.pcnn2D.update(*observation["position"])
        self.state["u"] = np.array(u)
        self.state["delta_update"] = self.pcnn2D.get_delta_update()

        # --- update modulation
        c_out, c_val = self.circuits(observation=self.state)
        for i, (key, out) in enumerate(c_out.items()):
            self.state[key] = out

        ach = self.densitymod(x=c_val) if self.densitymod is not None else 1.

        # self.pcnn2D.ach_modulation(ach=ach)
        # self.pcnn2D.update(*observation["position"])

        # --- update experience module
        exp_output = self.exp_module(observation=self.state)
        for (key, out) in exp_output.items():
            self.state[key] = out

        return exp_output["velocity"]

    def reset(self, position: list):
        self.pcnn2D.reset_gcn(position)

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

        if kwargs.get("return_fig", False):
            return list(fig_exp) + fig_cir


class randBrain:

    def __init__(self, speed: float,
                 pcnn2D: object):

        self.speed = speed
        self.pcnn2D = pcnn2D
        self.circuits = None

        self.state = {
            "position": np.zeros(2),
            "collision": False,
            "reward": 0.,
            "u": np.zeros(pcnn2D.get_size()),
            "delta_update": 0.,
            "velocity": np.zeros(2),
            "action_idx": 0,
            "depth": 0,
            "score": 0,
        }

        # self.state = self.observation_ext | self.observation_int

        self.directive = "new"
        self._elapsed_time = 0
        self.t = -1

        # record
        self.record = {"trajectory": []}

    def __str__(self) -> str:
        return f"Brain({self.pcnn2D})"

    # def __call__(self, observation: dict):
    def __call__(self, v: list, collision: float,
                 reward: float=0.0, position: list=[-1.0, -1.0]):

        """
        forward the observation through the brain
        """

        if collision:
            self.state["velocity"] *= -1

        self.t += 1
        if self.t % 100 == 0:
            self.state["velocity"] = np.random.uniform(
                -self.speed, self.speed, 2)
            self.state["velocity"] = self.state["velocity"]/\
                np.linalg.norm(self.state["velocity"]) * \
                self.speed

        # >> forward current position
        u, y = self.pcnn2D(self.state["velocity"])
        self.pcnn2D.update(*position)

        # logger.debug(f"[{self.t}] y: {np.around(y, 2)}")

        return self.state["velocity"]

    def reset(self, position: list):
        self.pcnn2D.reset_gcn(position)
        # pass

    def render(self, **kwargs):

        pass


""" policies """


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


""" local utils """

class OneLayerNetworkWrapper(pclib.OneLayerNetwork):

    def __init__(self, weights: list):

        super().__init__(weights=weights)

    def render(self, ax: plt.Axes,
               labels: list):

        ax.clear()
        ax.bar(range(5), self.get_weights(),
               color="blue", alpha=0.7)
        ax.set_title("Perceptron Weights")
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels)
        ax.grid()







