import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tqdm import tqdm
import logging, coloredlogs



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


""" logger """

def setup_logger(name: str="MAIN",
                 colored: bool=True,
                 level: int=0,
                 is_debugging: bool=True,
                 is_warning: bool=True) -> logging.Logger:

    """
    this function sets up a logger

    Parameters
    ----------
    name : str
        name of the logger. Default="MAIN"
    colored : bool
        use colored logs. Default=True
    level : int
        the level that is currently used.
        Default=0
    is_debugging : bool
        use debugging mode. Default=True
    is_warning : bool
        use warning mode. Default=True

    Returns
    -------
    logger : object
        logger object
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create a custom formatter
    if colored:
        formatter = coloredlogs.ColoredFormatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # create a colored stream handler with the custom formatter
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        # add the handler to the logger and disable propagation
        logger.addHandler(handler)

    logger.propagate = False

    # wrapper class
    class LoggerWrapper:
        def __init__(self, logger,
                     level: int,
                     is_debugging: bool=False,
                     is_warning: bool=False):
            self.logger = logger
            self.level = level
            self.is_debugging = is_debugging
            self.is_warning = is_warning

            self.logger.info(self)

        def __repr__(self):

            return f"LoggerWrapper(name={self.logger.name}," + \
                   f"level={self.level}, " + \
                   f"debugging={self.is_debugging})" + \
                   f"warning={self.is_warning})"

        def __call__(self, msg: str="", level: int=0):
            if level <= self.level:
                self.logger.info(msg)

        def info(self, msg: str="", level: int=0):
            self(msg, level)

        def warning(self, msg: str="", level: int=0):
            if level <= self.level and self.is_warning:
                self.logger.warning(msg)

        def error(self, msg: str="", level: int=0):
            if level <= self.level:
                self.logger.error(msg)

        def debug(self, msg, level: int=0):
            if level <= self.level and self.is_debugging:
                self.logger.debug(msg)

        def set_debugging(self, is_debugging: bool):
            self.is_debugging = is_debugging

        def set_warning(self, is_warning: bool):
            self.is_warning = is_warning

        def set_level(self, level: int):
            self.level = level

    return LoggerWrapper(logger=logger, level=level,
                         is_debugging=is_debugging,
                         is_warning=is_warning)


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
        # return simulator.get_pcnn_graph()

    data = []
    for _ in tqdm(range(N)):

        # data += [run(simulator)]

        data.append(np.array(run(simulator)))
        simulator.reset(init_position=np.random.uniform(0.1, 0.9, 2))

    # plot
    ncols = min((N, 5))
    nrows = N // 5

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(13, 5))

    for i, ax in enumerate(axs.flatten()):

        if i < len(data):

            # trajectory
            ax.plot(data[i][:, 0], data[i][:, 1],
                       lw=0.5)

            # graph
            # nodes, edges = data[i]
            # ax.scatter(nodes[:, 0], nodes[:, 1], s=10)

            ax.set_title(f"{i=}")
            # ax[i].axis("off")
            ax.set_aspect("equal")
            ax.set_xlim(0., 1.)
            ax.set_ylim(0., 1.)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ax.axis("off")

    plt.tight_layout()
    plt.show()


