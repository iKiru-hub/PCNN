import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tools.utils import logger
import pcnn_core as pcnn




class LeakyVariable:

    def __init__(self, eq: float=0., tau: float=10,
                 name: str="leaky", ndim: int=1,
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
        self.ndim = ndim
        self.v = np.ones(1)*self.eq if ndim == 1 else np.ones((ndim, 1))*self.eq
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
        # self.v = np.clip(self.v, -1, 1.)
        self.record += [self.v.tolist()]
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
        self.ax.set_title(f"{self.name} | v={np.around(self.v, 2).tolist()}")
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
        self.state = (0., 0.)
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
            self.state = (0., 0.)


class ModuleClass(ABC):

    def __init__(self):

        self.modulators = None
        self.output = None
        self.pcnn = None

    def __call__(self, x,
                 **kwargs):

        self._apply_modulators()

        self._logic(x, **kwargs)
        return self.output

    def _apply_modulators(self):
        if self.pcnn is None or self.modulators is None:
            return

        for modulator in self.modulators:

            if isinstance(modulator, np.ndarray):
                output = modulator.output.copy().reshape(-1, 1)
            else:
                output = modulator.output

            if modulator.action == "plasticity":
                self.pcnn.add_update(x=output)
            elif modulator.action == "fwd":
                self.pcnn.add_input(x=output)
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
        self.state = (1*(self.leaky_var.v > self.threshold),
                       self.leaky_var.v)
        self.output = self.state[0]


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

        self.state = ((-1 * self.weights * \
            x).reshape(-1, 1), self.leaky_var.v)
        self.output = self.state[0].copy()

    def super_render(self):

        self.axs_bnd[0].clear()
        self.axs_bnd[1].clear()
        self.axs_bnd[0].imshow(self.weights.reshape(1, -1),
                           aspect="auto", cmap="RdBu",
                               vmin=-0.1, vmax=0.1)
        self.axs_bnd[0].set_title(f"Boundary Modulation")
        self.axs_bnd[0].set_yticks([])

        self.axs_bnd[1].imshow(self.output.reshape(1, -1),
                           aspect="auto", cmap="RdBu",
                               vmin=-0.1, vmax=0.1)
        self.axs_bnd[1].set_title(f"Output")
        self.axs_bnd[1].set_yticks([])

        self.fig_bnd.canvas.draw()


class EligibilityTrace(Modulation):

    def __init__(self, N: int):

        super().__init__()
        self.leaky_var = LeakyVariable(eq=0., tau=50,
                                       name="ET",
                                       ndim=N,
                                       max_record=500)
        self.output = self.leaky_var.v.reshape(-1, 1)
        self.threshold = 0.9
        self.action = "fwd"
        self.input_key = ("u", None)
        self.fig, self.ax = self.leaky_var.fig, self.leaky_var.ax

    def _logic(self, inputs: tuple):

        self.leaky_var(x=inputs[0].reshape(-1, 1)*0.1)
        self.state = ((self.leaky_var.v-self.leaky_var.v.min()), None)
        self.output = -1*self.state[0].reshape(-1, 1) * 10

    def render(self):

        self.ax.clear()

        self.ax.imshow(np.abs(self.output).reshape(1, -1),
                       aspect="auto", cmap="Reds")
        self.ax.set_yticks([])
        self.ax.set_title(f"Eligibility Trace " + \
            f"{np.abs(self.output).max():.2f}")
        self.fig.canvas.draw()


class Modulators:

    def __init__(self, modulators: list):

        self.modulators = modulators
        self.names = tuple(modulators.keys())
        self.output = {name: None for name in self.names}

    def __call__(self, observation: dict):

        for name, modulator in self.modulators.items():
            inputs = []
            for key in modulator.input_key:
                if key is not None:
                    inputs += [observation[key]]
            inputs += [None]
            modulator(inputs=inputs)
            self.output[name] = modulator.output

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
                 modulators: Modulators,
                 pcnn_plotter: object=None,
                 makefig: bool=False):

        super().__init__()
        self.pcnn = pcnn
        self.modulators = modulators
        self.pcnn_plotter = pcnn_plotter
        self.output = {"u": np.zeros(pcnn.N),
                       "position": np.zeros(2),
                       "delta_update": 0.,
                       "velocity": np.zeros(2)}

        # --- policies
        self.policy = RandomWalkPolicy(speed=0.005)

        # --- visualizationo
        if makefig:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
        else:
            self.fig, self.ax = None, None
        self.record = []

    def _logic(self, observation: dict, mode: str="current"):

        """
        the level of acetylcholine is determined
        affects the next weight update
        """

        # --- input
        collision = kwargs.get("collision", False)

        # --- logic
        if mode == "current":
            self.pcnn.fwd_ext(x=x)
        elif mode == "proximal":
            self.pcnn.fwd_int(x=x)

        position = x
        if collision: self.policy.has_collided()

        # get representations
        repr_ext = self.pcnn.representation
        repr_int = np.array([self.modulators[1].state[1],
                             self.modulators[0].state[1]])

        # move in internal space
        action_int = self._generate_internal_action(z=repr_int)
        pred_repr_int = repr_int + action_int

        # move in external space
        action_ext = self._generate_external_action(
                                z_int=pred_repr_int,
                                curr_position=position)

        # --- output
        self.record += [position.tolist()]
        self.output = {"u": repr_ext,
                       "position": position,
                       "delta_update": self.pcnn.delta_update,
                       "velocity": action_ext}

    def _generate_internal_action(self, z: np.ndarray) -> np.ndarray:

        """
        step in the internal space

        Parameters
        ----------
        z : np.ndarray
            predicted internal representation

        Returns
        -------
        np.ndarray
            action in the internal space
        """

        # goal:
        # - ACh = 1.0
        # - Bnd = 0.0
        new_z = z + (np.array([1., 0.]) - z) * 0.5

        return new_z

    def _evaluate_action(self, action: np.ndarray,
                         current_position: np.ndarray) -> np.ndarray:

        """
        evaluate the action

        Parameters
        ----------
        action : np.ndarray
            action in the external space

        Returns
        -------
        np.ndarray
            evaluated action
        """

        new_position = current_position + action

    def _generate_external_action(self, z_int: np.ndarray,
                                  curr_position: np.ndarray) -> np.ndarray:

        """
        step in the external space

        Parameters
        ----------
        z_int : np.ndarray
            predicted internal representation
        curr_position : np.ndarray
            external representation

        Returns
        -------
        np.ndarray
            action in the external space
        """

        # proximal representation of the current position
        self.pcnn.add_input(x=self.modulators[2].output)
        representation = self.pcnn.fwd_int(x=curr_position)
        mean_position = self.pcnn.current_position(
                        u=representation)

        if len(self.pcnn) < 10 :#or position is None:
            return self.policy()

        if mean_position is None:
            return self.policy()

        speed = 0.005
        action = mean_position - curr_position.flatten()
        action = np.around(speed * action / np.abs(action).sum(), 4)

        assert np.abs(action).sum() <= speed*1.1, f"action {np.abs(action).sum()} [{action}] has greater magnitude than {speed=}"

        return action

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





