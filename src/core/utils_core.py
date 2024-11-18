import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
import logging, coloredlogs


FIGPATH = "dashboard/cache/"


""" LOGGER """


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
                   f"debugging={self.is_debugging}, " + \
                   f"warning={self.is_warning})"

        def __call__(self, msg: str="", level: int=1):
            if level <= self.level:
                self.logger.info(msg)

        def info(self, msg: str="", level: int=1):
            self(msg, level)

        def warning(self, msg: str=""):
            if self.is_warning:
                self.logger.warning(msg)

        def error(self, msg: str=""):
            if self.is_warning:
                self.logger.error(msg)

        def debug(self, msg):
            if self.is_debugging:
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


logger = setup_logger(name="UTILS", colored=True,
                      level=0, is_debugging=False,
                      is_warning=False)


def edit_logger(level: int=-1,
                is_debugging: bool=True,
                is_warning: bool=False):
    global logger
    logger.set_level(level)
    logger.set_debugging(is_debugging)
    logger.set_warning(is_warning)


""" VISUALIZATION """


class PlotPCNN:

    def __init__(self, model: object,
                 visualize: bool=True,
                 number: int=None,
                 edges: bool=True,
                 bounds: tuple=(0, 1, 0, 1),
                 cmap: str='viridis'):

        self._model = model
        self._number = number
        self._bounds = bounds
        self._elements = []
        self.visualize = visualize
        if visualize:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        # else:
        #     self._fig, self._ax = None, None

    def add_element(self, element: object):
        assert hasattr(element, "render"), \
            "element must have a render method"

        self._elements += [element]

    def render(self, trajectory: np.ndarray=None,
               rollout: tuple=None,
               edges: bool=True, cmap: str='RdBu_r',
               ax=None, new_a: np.ndarray=None,
               alpha_nodes: float=0.1,
               alpha_edges: float=0.2,
               return_fig: bool=False,
               render_elements: bool=False,
               customize: bool=False,
               title: str=None):

        new_ax = True
        if ax is None:
            # fig, ax = plt.subplots(figsize=(6, 6))
            fig, ax = self._fig, self._ax
            ax.clear()
            new_ax = False

        # new_a = new_a if new_a is not None else self._model.u

        # render other elements
        if render_elements:
            for element in self._elements:
                element.render(ax=ax)

        # --- trajectory
        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-',
                          lw=0.5, alpha=0.5 if new_a is not None else 0.9)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                        c='k', s=100, marker='x')

        # --- rollout
        if rollout is not None and len(rollout[0]) > 0:
            rollout_trj, rollout_vals = rollout
            ax.plot(rollout_trj[:, 0], rollout_trj[:, 1], 'b',
                    lw=1, alpha=0.5, linestyle='--')
            for i, val in enumerate(rollout_vals):
                ax.scatter(rollout_trj[i, 0], rollout_trj[i, 1],
                            c='b', s=10*(2+val), alpha=0.7,
                           marker='o')

        # --- network
        centers = self._model.get_centers()
        connectivity = self._model.get_wrec()

        ax.scatter(centers[:, 0],
                   centers[:, 1],
                   c=new_a if new_a is not None else None,
                   s=40, cmap=cmap,
                   vmin=0, vmax=0.04,
                   alpha=alpha_nodes)

        if edges and new_a is not None:
            for i in range(connectivity.shape[0]):
                for j in range(connectivity.shape[1]):
                    if connectivity[i, j] > 0:
                        ax.plot([centers[i, 0], centers[j, 0]],
                                [centers[i, 1], centers[j, 1]],
                                'k-',
                                alpha=alpha_edges,
                                lw=0.5)

        #
        # ax.axis('off')
        if customize:
            # ax.axis('off')
            ax.set_xlim(self._bounds[0], self._bounds[1])
            ax.set_ylim(self._bounds[2], self._bounds[3])
            ax.set_xticks(())
            ax.set_yticks(())

        if title is None:
            title = f"PCNN | N={len(self._model)}"
        ax.set_title(title, fontsize=14)

        if self._number is not None and not new_ax:
            try:
                fig.savefig(f"{FIGPATH}fig{self._number}.png")
            except Exception as e:
                logger.debug(f"{e=}")
                return
            plt.close()
            return

        if not new_ax:
            fig.canvas.draw()

        # if ax == self._ax:
        #     self._fig.canvas.draw()

        if return_fig:
            return fig


""" FUNCTIONS """


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

    norms = (np.linalg.norm(x) * np.linalg.norm(y))

    if norms == 0:
        return 0.

    z = (x @ y) / norms
    if np.isnan(z):
        return 0.
    return z.item()


def calc_position_from_centers(a: np.ndarray,
                               centers: np.ndarray) -> np.ndarray:

    """
    calculate the position of the agent from the
    activations of the neurons in the layer
    """

    if a.sum() == 0:
        return np.array([np.nan, np.nan])

    return (centers * a.reshape(-1, 1)).sum(axis=0) / a.sum()


@jit(nopython=True)
def generalized_sigmoid(x: np.ndarray,
                        alpha: float,
                        beta: float,
                        clip_min: float=0.,
                        gamma: float=1.
                        ) -> np.ndarray:

    """
    generalized sigmoid function and set values below
    a certain threshold to zero.

    Parameters
    ----------
    x : np.ndarray
        the input
    alpha : float
        the threshold
    beta : float
        the slope
    gamma : float
        the intensity (height).
        Default is 1.
    clip_min : float
        the minimum value to clip.
        Default is 0.

    Returns
    -------
    np.ndarray
        The output array.
    """

    x = gamma / (1.0 + np.exp(-beta * (x - alpha)))

    return np.where(x < clip_min, 0., x)


""" ANALYSIS """


def _multiple_simulations(N: int, simulator: object,
                          use_tqdm: bool=True):

    """
    run multiple simulations
    """

    # --- INITIALIZATION
    # define initial positions as N points on a grid
    # over a box [0, 1] x [0, 1]

    # approximate N to the nearest square number
    if np.sqrt(N) % 1 != 0:
        N = int(np.sqrt(N)) ** 2
        logger.warning(f"Approximated N to the" + \
            f" nearest square number: {N}")
    xg = np.linspace(0.1, 0.9, int(np.sqrt(N)))
    yg = np.linspace(0.1, 0.9, int(np.sqrt(N)))
    all_init_positions = np.array(np.meshgrid(xg, yg)).T.reshape(-1, 2)

    # --- SIMULATION
    def run(simulator: object):

        done = False
        while not done:
            done = simulator.update()

        return simulator.get_trajectory(), simulator.get_reward_visit()
        # return simulator.get_pcnn_graph()

    data = []
    for i in tqdm(range(N), disable=not use_tqdm):

        # data += [run(simulator)]
        simulator.reset(init_position=all_init_positions[i])
        data += [run(simulator)]

    return N, data


def analysis_I(N: int, simulator: object):

    """
    plot the start and end positions of the trajectory,
    GOAL: highlight how the agent stays within the reward area
    """

    # --- RUN
    N, data = _multiple_simulations(N, simulator)

    # --- PLOT
    if len(data) > 10:
        num_per_col = 10
    else:
        num_per_col = len(data)
    ncols = min((N, num_per_col))
    nrows = N // num_per_col

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(13, 5))

    # reward
    rw_position, rw_radius = simulator.get_reward_info()
    for i, ax in enumerate(axs.flatten()):

        if i < len(data) and data[i][1]:

            # reward area
            ax.add_patch(plt.Circle(rw_position, rw_radius,
                                    color="green", alpha=0.1))

            # trajectory
            trajectory = np.array(data[i][0])
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                       lw=0.5, alpha=0.7)

            # start and end
            ax.scatter(trajectory[0, 0], trajectory[0, 1],
                       marker="o", color="white", s=40,
                       edgecolor="red")
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                       marker="o", color="red", s=40,
                       edgecolor="red")

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


def analysis_II(N: int, simulator: object):

    """
    GOAL: highlight the fraction of simulations whose last part
    of the trajectory is spend within the reward area, given
    different reward positions
    """

    assert int(np.sqrt(N)) ** 2 == N, "N must be a perfect square"

    # reward positions
    # all_rw_positions = [
    #     [0.1, 0.1], [0.2, 0.5],
    #     [0.1, 0.9], [0.5, 0.7],
    #     [0.9, 0.9], [0.7, 0.5],
    #     [0.9, 0.1], [0.5, 0.2],
    #     [0.5, 0.5]
    # ]

    all_rw_positions = [
        [0.2, 0.2], [0.2, 0.8],
        [0.8, 0.8], [0.8, 0.2],
        [0.2, 0.5], [0.5, 0.7],
        [0.7, 0.2], [0.5, 0.2],
        [0.5, 0.5], [-10., -10.]
    ]
    rw_radius = simulator.get_reward_info()[1]
    NUM_TRIALS = len(all_rw_positions)

    # run & plot
    fig, axs = plt.subplots(nrows=2, ncols=NUM_TRIALS, figsize=(13, 5))
    fig.suptitle("Reward reaching accuracy in different positions",
                 fontsize=16)
    for i in tqdm(range(NUM_TRIALS)):

        simulator.set_rw_position(rw_position=all_rw_positions[i])
        _, data = _multiple_simulations(N, simulator, use_tqdm=False)

        # process:
        # average residuals for the last 70% of the trajectory
        residuals = []
        avg_end_positions = []
        all_positions = []
        for trajectory, _ in data:

            # average position
            avg_pos = np.array(trajectory[int(0.7 * len(trajectory)):]).mean(axis=0)
            avg_end_positions += [avg_pos]
            all_positions += [avg_pos.tolist()]

            # check if the average position is within the reward area
            # if np.linalg.norm(avg_pos - all_rw_positions[i]) < rw_radius:
            #     num_within += 1
            rw_position_i = all_rw_positions[i] if i < NUM_TRIALS - 1 else np.array([0.5, 0.5])
            residuals += [np.linalg.norm(avg_pos - rw_position_i)]

        accuracy = 1. - np.array(residuals)
        accuracy = np.flip(np.sort(accuracy))[:int(0.9 * N)]
        variance = np.var(all_positions, axis=0).mean()

        # plot:
        # A) plot
        # axs[0, i].bar(0, accuracy, color="green", alpha=0.8)
        axs[0, i].plot(range(len(accuracy)), accuracy, color="green", alpha=0.8)

        # variance as a shaded area around the mean
        axs[0, i].fill_between(range(len(accuracy)),
                              np.mean(accuracy) - variance,
                              np.mean(accuracy) + variance,
                              color="red", alpha=0.1)
        axs[0, i].axhline(np.mean(accuracy), color="red", lw=2.)

        if i == NUM_TRIALS - 1:
            axs[0, i].set_title(f"[baseline]\n{np.mean(accuracy):.2f}")
        else:
            axs[0, i].set_title(f"{np.mean(accuracy):.2f}")
        # axs[0, i].axis("off")
        axs[0, i].set_ylim(0., 1.)
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([0, 1.])
        axs[0, i].set_yticklabels(["0", "1"])
        axs[0, i].grid(True)

        # B) scatter plot of the reward area and the average end positions
        axs[1, i].add_patch(plt.Circle(rw_position_i, rw_radius,
                                      color="green", alpha=0.2))
        avg_end_positions = np.array(avg_end_positions)
        axs[1, i].scatter(avg_end_positions[:, 0], avg_end_positions[:, 1],
                         color="red", s=5)
        axs[1, i].set_aspect("equal")
        axs[1, i].set_xlim(0., 1.)
        axs[1, i].set_ylim(0., 1.)
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

    # save fig
    fig.savefig("reward_reaching_accuracy.png")
    print("Figure saved as 'reward_reaching_accuracy.png'")

    plt.tight_layout()
    plt.show()





