import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tqdm import tqdm



""" functions """

class OnlineFigure(ABC):

    """
    an object responsible for plotting online data
    in different figures
    """

    def __init__(self):

        self.fig, self.ax = plt.subplots()

    @abstractmethod
    def update(self):
        pass


def cosine_similarity(M: np.ndarray):

    """
    calculate the cosine similarity
    """

    # normalized matrix dot product
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    M = np.where(np.isnan(M), 0., M)
    return (M @ M.T) * (1 - np.eye(M.shape[0]))


def cosine_similarity_vec(x: np.ndarray,
                         y: np.ndarray) -> np.ndarray:

    """
    calculate the cosine similarity
    """

    y = y.reshape(-1, 1)
    x = x.reshape(1, -1)

    z = (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
    if np.isnan(z):
        return 0.
    return z.item()


def calc_position_from_centers(a: np.ndarray,
                               centers: np.ndarray) -> np.ndarray:

    """
    calculate the position of the agent from the
    activations of the neurons in the layer
    """

    return (centers * a.reshape(-1, 1)).sum(axis=0) / a.sum()


""" analysis """


def multiple_simulation(N: int, simulator: object):

    """
    run multiple simulations
    """

    def run(simulator: object):

        done = False
        while not done:
            done = simulator.update()
        return simulator.get_trajectory()

    data = []
    for _ in tqdm(range(N)):
        data.append(np.array(run(simulator)))
        simulator.reset(seed=np.random.randint(0, 10000))

    # plot
    fig, ax = plt.subplots(nrows=N, ncols=1, figsize=(13, 5))
    for i, trajectory in enumerate(data):
        ax[i].plot(trajectory[:, 0], trajectory[:, 1])
        ax[i].set_title(f"simulation {i}")

    plt.tight_layout()
    plt.show()


