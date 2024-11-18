import numpy as np
import matplotlib.pyplot as plt
import utils_core as utils
from tools.utils import logger


class TargetModule():

    """
    Input:
        x: 2D position [array]
        mode: mode [str] ("current" or "proximal")
    Output:
        representation: output [array]
        current position: [array]
    """

    def __init__(self, speed: int=0.005,
                 max_depth: int=10,
                 score_weight: float=1.,
                 visualize: bool=True,
                 number: int=None):

        super().__init__()
        self._speed = speed
        self._max_depth = max_depth
        self._score_weight = score_weight
        self.output = {
                "u": None,
                "trg_position": np.zeros(2),
                "velocity": np.zeros(2),
                "importance": 0.}

        #
        self._num_nodes = None
        self._nodes = None
        self._edges = None
        self._value_weights = None

        # --- visualization
        self._visualize = visualize
        self._number = number
        if visualize:
            self._fig, self._ax = plt.subplots(figsize=(4, 3))
            logger(f"%visualizing {self.__class__}")

    def __str__(self):
        return f"TargetModule(weight={self._score_weight})"

    def __call__(self, observation: dict):

        """
        observation should contain:
        - `position` : position
        - `Ftg` : value of the Ftg circuit
        """

        # current representations
        curr_pos = observation["position"]

        # target representations
        trg_repr, flag = self._converge_to_location(
                x=np.zeros((self._num_nodes, 1)),
                depth=0,
                modulation=self._value_weights,
                threshold=0.8)
        trg_pos = utils.calc_position_from_centers(a=trg_repr,
                                             centers=self._nodes)

        # velocity
        velocity = trg_pos - curr_pos
        velocity = velocity / np.linalg.norm(velocity) * self._speed

        # intensity
        if observation["Ftg"] > 0.4:
            importance = -1 * int(flag) * observation["Ftg"]
        else:
            importance = 0.

        # --- output
        assert isinstance(importance, float), f"{type(importance)=}"
        self.output = {
                "u": trg_repr,
                "trg_position": trg_pos,
                "trg_velocity": velocity,
                "importance": importance}

        logger(f"trg_module || " + \
            f"v={np.around(velocity*1000, 2).tolist()} | " + \
            f"importance={importance:.3f}")

    def _converge_to_location(self, x: np.ndarray,
                              depth: int,
                              modulation: np.ndarray,
                              threshold: float=0.1):

        u = self._edges @ (x + modulation.reshape(-1, 1))

        c = utils.cosine_similarity_vec(u, x)
        if c > threshold:
            return u, True

        if depth >= self._max_depth:
            return u, False

        return self._converge_to_location(x=u,
                                          depth=depth+1,
                                          modulation=modulation,
                                          threshold=threshold)

    def compare(self, velocity: np.ndarray) -> float:

        """
        compare a queries velocity with the
        calculated target velocity
        """

        score = utils.cosine_similarity_vec(
                        self.output["trg_velocity"],
                        velocity)
        score *= self._score_weight * self.output["importance"]

        assert isinstance(score, float), f"{type(score)=}"

        return score

    def update(self, nodes: np.ndarray,
               edges: np.ndarray,
               value_weights: np.ndarray):

        self._num_nodes = nodes.shape[0]
        self._nodes = nodes.copy()
        self._edges = edges.copy()
        self._value_weights = value_weights.copy()

    def render(self, **kwargs):

        if not self._visualize:
            return

        self._ax.clear()
        self._ax.scatter(self.output["trg_position"][0],
                    self.output["trg_position"][1],
                    marker="x", color="red",
                    s=100*np.abs(self.output["importance"]))
        self._ax.set_title(f"Target Module | " + \
                          f" I={np.around(self.output['importance'], 2)}")
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 1)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.set_aspect("equal")
        self._ax.set_xlabel(f"trg_pos={np.around(self.output['trg_position'], 3)}")
        self._ax.grid()

        if self._number is not None:
            self._fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return

    def reset(self):
        raise NotImplementedError




if __name__ == "__main__":

    # --- test
    nodes = np.random.rand(10, 2)
    edges = np.random.rand(10, 10)
    value_weights = np.random.rand(10)

    target_module = TargetModule()
    target_module.update(nodes=nodes,
                         edges=edges,
                         value_weights=value_weights)

    target_module(observation={"position": np.random.rand(2),
                               "Ftg": 0.5})
    target_module.render()
    print(target_module.output)
    print(target_module.compare(velocity=np.random.rand(2)))
    plt.show()

