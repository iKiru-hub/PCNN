import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tools.utils import logger
import pcnn_core as pcnn




class LeakyVariable:

    def __init__(self, eq: float=0., tau: float=10,
                 name: str="leaky", ndim: int=1,
                 visualize: bool=False,
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
        self._visualize = visualize

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
        self.fig.canvas.draw()


class Modulation(ABC):

    """
    Actions:
    - "plasticity"
    - "fwd"
    """

    def __init__(self, N: int=0,
                 visualize: bool=False):

        self.N = N
        self.leaky_var = None
        self.action = None
        self.input_key = None
        self.output = None

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
        return self.leaky_var._v

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

    def __call__(self, observation: dict,
                 **kwargs):

        self._apply_modulators()

        self._logic(observation=observation,
                    **kwargs)
        return self.output

    def _apply_modulators(self):
        if self.pcnn is None or self.modulators is None:
            return

        for _, modulator in self.modulators.modulators.items():

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


class Acetylcholine(Modulation):

    def __init__(self, visualize: bool=True):

        super().__init__()
        self.leaky_var = LeakyVariable(eq=1., tau=10,
                                       name="ACh",
                                       max_record=500,
                                       visualize=visualize)
        self.threshold = 0.9
        self.action = "plasticity"
        self.input_key = ("delta_update", None)
        self.output = 0.
        self._visualize = visualize

    def _logic(self, inputs: tuple,
               simulate: bool=False) -> np.ndarray:
        v = self.leaky_var(x= -1 * inputs[0], simulate=simulate)

        out = [1*(v > self.threshold),
                self.get_leaky_v()]
        self.output = out[0]
        return out


class BoundaryMod(Modulation):

    """
    Input:
        x: representation `u` [array]
        collision: collision [bool]
    Output:
        representation (output) [array]
    """

    def __init__(self, N: int, eta: float=0.1,
                 visualize: bool=True):

        super().__init__(N=N)
        self.leaky_var = LeakyVariable(eq=0., tau=3,
                                       name="Bnd",
                                       visualize=visualize)
        self.action = "fwd"
        self.input_key = ("u", "collision")
        self._visualize = visualize
        self.weights = np.zeros(N)
        self.eta = eta
        self.output = np.zeros((N, 1))

        if visualize:
            self.fig_bnd, self.axs_bnd = plt.subplots(
                2, 1, figsize=(4, 3))

        self.var = np.zeros(N)

    def _logic(self, inputs: tuple, simulate: bool=False):

        x = inputs[0].flatten()

        if inputs[1]:
            v = self.leaky_var(x=1, simulate=simulate)
            self.weights += self.eta * v * \
                np.where(x < 0.1, 0, x)
            self.weights = np.clip(self.weights, 0., 1.)
        else:
            v = self.leaky_var(x=0, simulate=simulate)

        out = [(-1 * self.weights * x).reshape(-1, 1),
                self.get_leaky_v()]
        self.output = out[0]
        self.var = x.copy()
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

        # self.axs_bnd[1].imshow(self.output.reshape(1, -1),
        #                    aspect="auto", cmap="RdBu",
        #                        vmin=-0.1, vmax=0.1)
        self.axs_bnd[1].bar(range(self.N), self.var.flatten(),
                            color="grey")
        self.axs_bnd[1].set_title(f"Output")
        self.axs_bnd[1].set_ylim(0, 0.035)
        self.axs_bnd[1].axhline(y=0.005, color="red",
                                alpha=0.4, linestyle="--")
        self.axs_bnd[1].grid()

        self.fig_bnd.canvas.draw()


class EligibilityTrace(Modulation):

    def __init__(self, N: int, visualize: bool=True):

        super().__init__(N=N)
        self.leaky_var = LeakyVariable(eq=0., tau=50,
                                       name="ET",
                                       ndim=N,
                                       max_record=500,
                                       visualize=visualize)
        self.output = self.get_leaky_v().reshape(-1, 1)
        self._visualize = visualize
        self.threshold = 0.9
        self.action = "fwd"
        self.input_key = ("u", None)
        self.output = np.zeros((N, 1))

        if visualize:
            self.fig, self.ax = self.leaky_var.fig, self.leaky_var.ax

    def _logic(self, inputs: tuple, simulate: bool=False):

        v = self.leaky_var(x=inputs[0].reshape(-1, 1)*0.1,
                           simulate=simulate)
        out = [-10 * (v - v.min()).reshape(-1, 1),
                self.get_leaky_v()]
        self.output = out[0]
        return out

    def render(self):

        if not self._visualize:
            return

        self.ax.clear()

        self.ax.imshow(np.abs(self.output).reshape(1, -1),
                       aspect="auto", cmap="Reds")
        self.ax.set_yticks([])
        self.ax.set_title(f"Eligibility Trace " + \
            f"{np.abs(self.output).max():.2f}")
        self.fig.canvas.draw()



class PositionTrace(Modulation):

    def __init__(self, visualize: bool=True):

        super().__init__(N=1)
        self.leaky_var = LeakyVariable(eq=0., tau=50,
                                       name="dPos",
                                       ndim=2,
                                       max_record=500,
                                       visualize=visualize)
        self.output = self.get_leaky_v().reshape(-1, 1)
        self._visualize = visualize
        self.threshold = 0.9
        self.action = "fwd"
        self.input_key = ("position", None)
        self.output = 1.
        if self._visualize:
            self.fig, self.ax = self.leaky_var.fig, self.leaky_var.ax

    def _logic(self, inputs: tuple, simulate: bool=False):

        v = self.leaky_var(eq=inputs[0].reshape(-1, 1),
                           simulate=simulate)

        res = np.abs(inputs[0].reshape(-1, 1) - v)

        out = [res.sum(),
               self.get_leaky_v()]
        self.output = out[0]
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



class Modulators:

    def __init__(self, modulators_dict: dict):

        self.modulators = modulators_dict
        self.names = tuple(modulators_dict.keys())
        self.output = {name: modulators_dict[name].output \
            for name in self.names}

    def __call__(self, observation: dict,
                 simulate: bool=False) -> dict:

        for name, modulator in self.modulators.items():
            inputs = []
            for key in modulator.input_key:
                if key is not None:
                    inputs += [observation[key]]
            inputs += [None]
            output = modulator(inputs=inputs,
                               simulate=simulate)
            self.output[name] = modulator.output

        return self.output

    def render(self):
        for _, modulator in self.modulators.items():
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
        # self.random_policy = RandomWalkPolicy(speed=0.005)
        self.action_policy = SamplingPolicy(speed=0.005)

        # --- visualizationo
        if makefig:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
        else:
            self.fig, self.ax = None, None
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
        spatial_repr = self.pcnn.fwd_ext(x=observation["position"])

        # --- generate an action
        # TODO : use directive
        action_ext = self._generation_action(
                                observation=observation,
                                directive=directive)

        # --- output
        self.record += [observation["position"].tolist()]
        self.output = {
                    "u": spatial_repr,
                    "position": [observation["position"].tolist()],
                    "delta_update": self.pcnn.delta_update,
                    "velocity": action_ext}

    def _generation_action(self, observation: dict,
                           threshold: float=0.5,
                           directive: str="new") -> np.ndarray:

        done = False  # when all actions have been tried
        while not done:

            # --- generate a random action
            if directive == "new":
                action, done = self.action_policy()
            elif directive == "keep":
                action = observation["velocity"]
            else:
                raise ValueError("directive must be " + \
                    "'new' or 'keep'")

            # --- simulate its effects
            new_position = observation["position"] + action

            if new_position is None:
                u = np.zeros(self.pcnn.N)
            else:
                u = self.pcnn.fwd_ext(x=new_position)

            new_observation = {
                "u": u,
                "position": new_position,
                "velocity": action,
                "collision": False,
                "delta_update": 0.}

            modulation = self.modulators(
                            observation=new_observation,
                            simulate=True)

            # --- evaluate
            score = 1
            score -= 0.5*new_observation["u"].max()
            bnd_mod = np.abs(modulation["Bnd"]).max()
            score *= np.clip(1 - 10*bnd_mod, 0, 1)

            logger(f"score: {score:.4f}")

            # the action is good enough
            if score > threshold or done:
                break
            directive = "new"

            # try again
            self.action_policy.update(score=0.5*score)

        # ---
        self.action_policy.reset()

        return action

    def render(self, ax=None, **kwargs):

        self.action_policy.render()

        if ax is None:
            ax = self.ax

        if self.pcnn_plotter is not None:
            self.pcnn_plotter.render(ax=ax,
                trajectory=kwargs.get("trajectory", False),
                new_a=-1*self.modulators.modulators["Bnd"].output)


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

    def __init__(self, exp_module: ExperienceModule,
                 modulators: Modulators):

        self.exp_module = exp_module
        self.modulators = modulators

        self.movement = None
        self.output = {"u": np.zeros(exp_module.pcnn.N),
                       "delta_update": 0.,
                       "velocity": np.zeros(2)}

    def __call__(self, observation: dict):

        # --- first move
        if self.movement is None:
            directive = "new"
        else:
            directive = "keep"

        # --- update modulation
        mod_observation = self.modulators(observation=observation)

        # TODO : add directive depending on modulation

        if mod_observation["dPos"] < 0.05:
            directive = "new"

        # --- update experience module
        logger(f"AGENT -> {directive}")
        exp_output = self.exp_module(
                            observation=observation,
                            directive=directive)

        self.output = {
            "u": exp_output["u"],
            "delta_update": exp_output["delta_update"],
            "velocity": exp_output["velocity"]}

        self.movement = self.output["velocity"]

        return self.output

    def render(self, ax: object, trajectory: np.ndarray):

        self.modulators.render()
        self.exp_module.render(ax=ax,
                          trajectory=np.array(trajectory))





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

    def __init__(self, speed: float=0.1):

        self._samples = [np.array([0., speed]),
                         np.array([speed/np.sqrt(2),
                                   speed/np.sqrt(2)]),
                         np.array([speed, 0.]),
                         np.array([speed/np.sqrt(2),
                                   -speed/np.sqrt(2)]),
                         np.array([0., -speed]),
                         np.array([-speed/np.sqrt(2),
                                   -speed/np.sqrt(2)]),
                         np.array([-speed, 0.]),
                         np.array([-speed/np.sqrt(2),
                                   speed/np.sqrt(2)])]

        self._num_samples = len(self._samples)

        self._idx = None
        self._available_idxs = list(range(self._num_samples))
        self._p = np.ones(self._num_samples) / self._num_samples
        self._velocity = self._samples[0]

        # render
        self.fig, self.ax = plt.subplots(figsize=(4, 3))

    def __call__(self, keep: bool=False) -> tuple:

        # --- keep the current velocity
        if keep and self._idx is not None:
            return self._velocity.copy(), False

        # --- first sample
        if self._idx is None:
            self._idx = np.random.choice(
                            self._num_samples, p=self._p)
            self._available_idxs.remove(self._idx)
            self._velocity = self._samples[self._idx]
            return self._velocity.copy(), False

        # --- all samples have been tried
        if len(self._available_idxs) == 0:

            self._idx = np.random.choice(self._num_samples,
                                              p=self._p)
            self._velocity = self._samples[self._idx]
            return self._velocity.copy(), True

        # --- sample again
        p = self._p[self._available_idxs].copy()
        p /= p.sum()
        self._idx = np.random.choice(
                        self._available_idxs,
                        p=p)
        self._available_idxs.remove(self._idx)
        self._velocity = self._samples[self._idx]

        return self._velocity.copy(), False

    def update(self, score: float):

        # update the probability
        self._p[self._idx] += score

        # normalize
        self._p = self._p / self._p.sum()

    def render(self):

        self.ax.clear()
        self.ax.bar(range(self._num_samples), self._p)
        self.ax.set_xticks(range(self._num_samples))
        # self.ax.set_xticklabels(["stay", "up", "right",
        #                          "down", "left"])
        self.ax.set_xlabel("Action")
        self.ax.set_title(f"Action Space")
        self.ax.set_ylim(0, 1)
        self.fig.canvas.draw()

    def reset(self):
        self._idx = None
        self._available_idxs = list(range(self._num_samples))
        self._p = np.ones(self._num_samples) / self._num_samples
        self._velocity = self._samples[0]

    def has_collided(self):

        self._velocity = -self._velocity





