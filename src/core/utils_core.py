import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod






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

