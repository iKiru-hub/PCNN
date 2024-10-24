import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tools.utils import logger
import pcnn_core as pcnn




class LeakyVariable:

    def __init__(self, eq: float=0., tau: float=10,
                 name: str="leaky",
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
        self.eq = eq
        self.v = self.eq
        self.output = 0.
        self.tau = tau
        self.record = []
        self._max_record = max_record

        # figure configs
        self.fig, self.ax = plt.subplots(figsize=(4, 3))

    def __repr__(self):
        return f"{self.name}(eq={self.eq}, tau={self.tau})"

    def __call__(self, x: float=0.):
        self.v += (self.eq - self.v) / self.tau + x
        self.v = max(0., self.v)
        self.record += [self.v]
        if len(self.record) > self._max_record:
            del self.record[0]

    def reset(self):
        self.v = self.eq
        self.record = []

    def render(self):

        self.ax.clear()
        self.ax.plot(range(len(self.record)), self.record)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.grid()
        self.ax.set_title(f"{self.name} | v={self.v:.3f}")
        self.fig.canvas.draw()


class Modulation(ABC):

    """
    Actions:
    - "plasticity"
    - "fwd"
    """

    def __init__(self, N: int=0):

        self.N = N
        self.leaky_var = None
        self.output = 0.
        self.action = None
        self.input_key = None

    def __repr__(self):
        return f"Mod.{self.leaky_var.name}"

    def __call__(self, inputs: float=0.):
        self._logic(inputs=inputs)
        return self.output

    @abstractmethod
    def _logic(self, inputs: float):
        pass

    def render(self):
        self.leaky_var.render()

    def reset(self, complete: bool=False):
        self.leaky_var.reset()
        if complete:
            self.output = None


class ModuleClass(ABC):

    def __init__(self):

        self.modulators = None
        self.output = None
        self.pcnn = None

    def __call__(self, x,
                 **kwargs):

        self._apply_modulators(**kwargs)
        self._logic(x, **kwargs)
        return self.output

    def _apply_modulators(self):
        if self.pcnn is None or self.modulators is None:
            return

        for modulator in self.modulators:
            if modulator.action == "plasticity":
                self.pcnn.add_update(x=modulator.output)
            elif modulator.action == "fwd":
                self.pcnn.add_input(x=modulator.output)
            else:
                raise ValueError(
                    "leaky_var action must be 'plasticity' or 'fwd'")

    @abstractmethod
    def _logic(self, x: float, **kwargs):
        pass




""" --- some leaky_var classes --- """


class Acetylcholine(Modulation):

    def __init__(self):

        super().__init__()
        self.leaky_var = LeakyVariable(eq=1., tau=10,
                                       name="ACh",
                                       max_record=500)
        self.output = self.leaky_var.v
        self.threshold = 0.9
        self.action = "plasticity"
        self.input_key = ("delta_update", None)

    def _logic(self, inputs: tuple):
        self.leaky_var(x= -1 * inputs[0])
        self.output = 1*(self.leaky_var.v > self.threshold)


class BoundaryMod(Modulation):

    """
    Input:
        x: representation `u` [array]
        collision: collision [bool]
    Output:
        representation (output) [array]
    """

    def __init__(self, N: int, eta: float=0.1):

        super().__init__()
        self.leaky_var = LeakyVariable(eq=0., tau=3,
                                       name="Bnd")
        self.action = "fwd"
        self.input_key = ("u", "collision")
        self.weights = np.zeros(N)
        self.eta = eta

        self.fig_bnd, self.axs_bnd = plt.subplots(
            2, 1, figsize=(4, 3))

    def _logic(self, inputs: tuple):

        x = inputs[0].flatten()

        if inputs[1]:
            self.leaky_var(x=1)
            self.weights += self.eta * self.leaky_var.v * \
                np.where(x < 0.3, 0, x)
            self.weights = np.clip(self.weights, 0., 1.)
        else:
            self.leaky_var(x=0)

        self.output = (-1 * self.weights * \
            x).reshape(-1, 1)

    def super_render(self):

        self.axs_bnd[0].clear()
        self.axs_bnd[1].clear()
        self.axs_bnd[0].imshow(self.weights.reshape(1, -1),
                           aspect="auto", cmap="RdBu",
                               vmin=-0.1, vmax=0.1)
        self.axs_bnd[0].set_title(f"Boundary Modulation")

        self.axs_bnd[1].imshow(self.output.reshape(1, -1),
                           aspect="auto", cmap="RdBu",
                               vmin=-0.1, vmax=0.1)
        self.axs_bnd[1].set_title(f"Output")

        self.fig_bnd.canvas.draw()



class Modulators:

    def __init__(self, modulators: list):

        self.modulators = modulators

    def __call__(self, **kwargs):

        for modulator in self.modulators:
            inputs = []
            for key in modulator.input_key:
                if key is not None:
                    inputs += [kwargs[key]]
            inputs += [None]
            modulator(inputs=inputs)

    def render(self):
        for modulator in self.modulators:
            modulator.render()
            if hasattr(modulator, "super_render"):
                modulator.super_render()


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
                 modulators: list,
                 pcnn_plotter: object=None,
                 makefig: bool=False):

        super().__init__()
        self.pcnn = pcnn
        self.modulators = modulators
        self.pcnn_plotter = pcnn_plotter
        self.output = (np.zeros(pcnn.N),
                       np.zeros(2), 0.)

        if makefig:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
        else:
            self.fig, self.ax = None, None
        self.record = []

    def _logic(self, x: float, mode: str="current",
               **kwargs):

        """
        the level of acetylcholine is determined
        affects the next weight update
        """

        # --- logic
        if mode == "current":
            self.pcnn.fwd_ext(x=x)
        elif mode == "proximal":
            self.pcnn.fwd_int(x=x)

        # --- output
        position = self.pcnn.current_position()
        self.record += [position.tolist()]
        self.output = (self.pcnn.representation, position,
                       self.pcnn.delta_update)

    def render(self, ax=None, **kwargs):

        if ax is None:
            ax = self.ax

        if self.pcnn_plotter is not None:
            self.pcnn_plotter.render(ax=ax,
                    trajectory=kwargs.get("trajectory", False),
                    new_a=-1*self.modulators[0].output)


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



class Agent:

    def __init__(self, pcnn_model: object,
                 ):

        self.module = self_module









