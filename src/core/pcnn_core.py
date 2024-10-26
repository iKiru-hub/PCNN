import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from abc import ABC, abstractmethod

# set off runtime warnings
np.seterr(divide='ignore', invalid='ignore')

from tools.utils import clf, tqdm_enumerate, logger
import inputools.Trajectory as it

import os, sys
base_path = os.getcwd().split("PCNN")[0]+"PCNN/src"
sys.path.append(base_path)
import utils




class PCNN():

    """
    Learning place fields
    - formation: dependant on ACh levels (exploration)
    - remapping: dependant on DA levels (reward)

    Important parameters:
    - learning rate : speed of remapping
    - threshold : area of the place field
    - epsilon : repulsion threshold
    """

    def __init__(self, N: int, Nj: int,
                 xfilter: object=None, **kwargs):

        """
        Parameters
        ----------
        N: int
            number of neurons
        Nj: int
            number of place cells
        xfilter: object
            input filter object.
            Default is None
        **kwargs: dict
            additional parameters for the `minPCNN`
            alpha: float
                threshold parameter of the sigmoid. Default is 0.1
            beta: float
                gain parameter of the sigmoid. Default is 20.0
            threshold: float
                threshold for the neuronal activity. Default is 0.3
            rep_threshold: float
                threshold for the lateral repulsion. Default is 0.3
            rec_threshold: float
                threshold for the recurrent connections. Default is 0.2
            k_neighbors: int
                maximum number of neighbors
            eq_ach: float
                equilibrium of acetilcholine. Default is 1.0
            tau_ach: float
                time constant for the acetilcholine. Default is 50
            ach_threshold: float
                threshold for the acetilcholine. Default is 0.5
        """

        super().__init__()

        # parametes
        self.name = kwargs.get('name', "PCNN")
        self.N = N
        self.Nj = Nj
        self.xfilter = xfilter
        self._alpha = kwargs.get('alpha', 0.3)
        self._beta = kwargs.get('beta', 20.0)
        self._clip_min = kwargs.get('clip_min', 0.005)
        self._threshold = kwargs.get('threshold', 0.3)
        self._rep_threshold = kwargs.get('rep_threshold', 0.3)
        self._indexes = np.arange(N)

        # recurrent connections
        self._rec_threshold = kwargs.get('rec_threshold', 0.2)
        self._k_neighbors = kwargs.get('k_neighbors', 7)
        self._calc_recurrent_enable = kwargs.get(
                            'calc_recurrent_enable', True)

        # variables
        self.u = np.zeros((N, 1))
        self.mod_input = np.zeros((N, 1))
        self.mod_update = 1.
        self._umask = np.zeros((N, 1))
        self._Wff = np.zeros((N, Nj))
        self._Wff_backup = np.zeros((N, Nj))
        self._Wrec = np.zeros((N, N))
        self._is_plastic = True

        # record
        self.cell_count = 0
        self.record = {
            "u": [],
            "umax": [],
            "dw": [0.]
        }
        self.info = kwargs
        self.info["N"] = N
        self.info["Nj"] = Nj

    def __repr__(self):
        return f"PCNN(N={self.N}, Nj={self.Nj})"

    def __call__(self, x: np.ndarray,
                 frozen: bool=False):

        """
        update the network state

        Parameters
        ----------
        x: np.ndarray
            input to the network
        frozen: bool
            if True, the network will not update
            the weights. Default is False
        **kwargs
            tau: float
                time constant for the policy.
                Default is None
            eq_da: float
                equilibrium value for the dopamine.
                Default is 1.0
        """

        if self.xfilter:
            x = self.xfilter(x=x)

        if (sum(x.shape)-1) > self.Nj:
            x = x / x.sum(axis=0) * 1.
            x = x.reshape(1, self.Nj, -1)
        else:
            x = x / x.sum() * 1.

        # normalize
        # x = x / x.sum()

        # step `u` | x : PC Nj > neurons N
        u = self._Wff @ x.reshape(-1, 1) + \
            self.mod_input.reshape(-1, 1)
        self.u = generalized_sigmoid(x=u,
                                     alpha=self._alpha,
                                     beta=self._beta,
                                     clip_min=self._clip_min)
        self.mod_input *= 0

        # update modulators
        if self._is_plastic and not frozen:
            stb, plc = self._calc_indexes()
            if stb is None:
                self._update(x=x, idx=plc)

        # record
        # self.record["u"] += [self.u.tolist()]
        # self.record["umax"] += [self.u.max()]

    def __len__(self):

        """
        number of tuned neurons
        """

        return self.cell_count

    def _calc_indexes(self) -> tuple:

        """
        calculate the indexes of the stable neurons and the winner
        """

        # calculate stable indexes as the index of the neurons
        # with weights close to wmax | skip the index 0
        stable_indexes = np.where(self._Wff.sum(axis=1) > 0.99)[0]
        empty_indexes = self._indexes[~np.isin(self._indexes,
                                               stable_indexes)]

        # --- make mask
        # no tuned neurons
        if len(stable_indexes) == 0:
            stb_winner = None

        # select eventual stable winner
        else:
            # update mask
            self._umask[stable_indexes] = 1.
            self._umask[empty_indexes] = 0.

            # maximally active stable neuron
            try:
                u_vals = self.u[stable_indexes]
                stb_winner = stable_indexes[np.argmax(u_vals)]
            except IndexError:
                raise IndexError(f"{stable_indexes=}, {self.u.shape=}")

            if stb_winner == 0:
                stb_winner = None
            elif self.u[stb_winner] < self._threshold:
                stb_winner = None
            else:
                return stb_winner, None

        # --- define plastic neurons
        # argmax of the activations of non-stable neurons
        plastic_indexes = [i for i in range(self.N) \
                                if i not in stable_indexes]

        if len(plastic_indexes) == 0:
            return None, None

        plastic_winner = np.random.choice(plastic_indexes)

        return None, plastic_winner

    def _update(self, x: np.ndarray, idx: int):

        """
        update the weights of the network

        Parameters
        ----------
        x : np.ndarray
            stimulus
        idx : int
            index of the selected plastic neuron
        """

        assert type(idx) == int or idx is None or \
            type(idx) == np.int64, f"{idx=}, {type(idx)=}"

        # --- FORMATION ---
        # there is an un-tuned neuron ready to learn
        if idx is None:
            return

        # self._Wff_backup = self._Wff.copy()

        # if the levels of ACh are high enough, update
        # the plastic neuron
        dw = (x.flatten() - self._Wff[idx]) * self.mod_update
        self._Wff[idx, :] += dw
        self.mod_update = 1.
        self.record["dw"] += [dw.sum()]

        # --- add new neuron ---
        if dw.sum() > 0.0:
            # this trick ensures that it isn't close to others
            # || this is necessary only because of a lurking bug
            # || somewhere, ACh and rps are not always enough!
            similarity = cosine_similarity(
                M=self._Wff.copy())[:, idx].max()

            # revert back
            if similarity > self._rep_threshold:
                # logger.debug(f"reverting back.. {similarity=}")
                self._Wff = self._Wff_backup.copy()
                self.record["dw"][-1] = 0.
                return

            logger.debug(f"adding new neuron.. {similarity=}")

            # proceed
            # self._Wff_backup = self._Wff.copy()
            self.cell_count += 1

            self._Wff_backup = self._Wff.copy()

            if self._calc_recurrent_enable:
                self._update_recurrent()

    def _update_recurrent(self, **kwargs):

        self._Wrec = calc_weight_connectivity(M=self._Wff,
                                    threshold=self._rec_threshold)

        # max number of neighbors
        if self._k_neighbors is not None:
            self._Wrec = k_most_neighbors(M=self._Wrec,
                                          k=self._k_neighbors)

    def add_input(self, x: np.ndarray):

        """
        add an input to the network

        Parameters
        ----------
        x : np.ndarray
            input to the network
        """

        self.mod_input += x

    def add_update(self, x: np.ndarray):

        """
        add an update to the network

        Parameters
        ----------
        x : np.ndarray
            update to the network
        """

        self.mod_update *= x

    def fwd_ext(self, x: np.ndarray):
        self(x=x.reshape(-1, 1))
        return self.representation

    def fwd_int(self, x: np.ndarray):

        # u = self.fwd_ext(x=x)
        self.u = self._Wrec @ self.u.reshape(-1, 1) + self.mod_input
        self.mod_input *= 0
        return self.representation

    def freeze(self, cut_weights: bool=False):

        self._is_plastic = False

        if cut_weights:
            self._Wff[~self._umask.flatten().astype(bool), :] = 0.

    def unfreeze(self):

        self._is_plastic = True

    @property
    def representation(self):
        return self.u.flatten().copy()

    @property
    def delta_update(self):
        return self.record["dw"][-1]

    def current_position(self, u: np.ndarray=None):
        if u is None:
            u = self.u

        if u.sum() <= 0.:
            return None

        idxs = np.where(self._umask.flatten() > 0)[0]
        centers = self._centers[idxs]

        # set nan to zero
        position = (centers.reshape(-1, 2) * \
            u[idxs].reshape(-1, 1)).sum(axis=0) / \
            u[idxs].sum()

        return position

    @property
    def _centers(self) -> np.ndarray:
        return calc_centers_from_layer(wff=self._Wff,
                                       centers=self.xfilter.centers)

    def reset(self, complete: bool=False):

        """
        reset the network

        Parameters
        ----------
        complete : bool
            whether to have a complete reset.
            Default False.
        """

        self.u = np.zeros((self.N, 1))

        self.record = {
            "u": [],
            "umax": [],
            "dw": []
        }

        if complete:
            self._umask = np.zeros((self.N, 1))
            self._Wff = np.zeros((self.N, self.Nj))
            self._Wrec = np.zeros((self.N, self.N))


class InputFilter(ABC):

    """
    Abstract class for filtering inputs
    """

    @abstractmethod
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def __call__(self, x: np.ndarray):
        pass


class PClayer(InputFilter):

    def __init__(self, n: int, sigma: int,
                 policy: object=None, **kwargs):

        self.N = n**2
        self.n = n
        self.bounds = kwargs.get("bounds", (0, 1, 0, 1))
        self._k = kwargs.get("k", 4)
        self.sigma = sigma
        self.spacing = None
        self._endpoints = kwargs.get("endpoints", True)
        self.centers = self._make_centers()
        self.centers = np.around(self.centers, 3)

    def __repr__(self):
        return f"PClayer(N={self.N}, s={self.sigma})"

    def _make_centers(self) -> np.ndarray:

        """
        Make the tuning function for the neurons in the layer.

        Returns
        -------
        centers : numpy.ndarray
            centers for the neurons in the layer.
        """

        x_min, x_max, y_min, y_max = self.bounds

        # check if it is a 2D grid or 1D
        if x_min == x_max:

            # Define the centers of the tuning functions
            # over a 1D grid
            x_centers = np.array([x_min] * self.N)
            y_centers = np.linspace(y_min, y_max, self.N)
            dim = 1

        elif y_min == y_max:

            # Define the centers of the tuning functions
            # over a 1D grid
            x_centers = np.linspace(x_min, x_max, self.N)
            y_centers = np.array([y_min] * self.N)
            dim = 1

        else:

            # Define the centers of the tuning functions
            x_centers = np.linspace(x_min, x_max, self.n,
                                    endpoint=self._endpoints)
            y_centers = np.linspace(y_min, y_max, self.n,
                                    endpoint=self._endpoints)
            dim = 2

        # Make the tuning function
        centers = np.zeros((self.N, 2))
        for i in range(self.N):

            if dim == 1:
                centers[i] = (x_centers[i], y_centers[i])
                continue
            centers[i] = (x_centers[i // self.n], 
                          y_centers[i % self.n])

        return centers

    def __call__(self, x: np.ndarray) -> np.ndarray:

        """
        Activation function of the neurons in the layer.

        Parameters
        ----------
        x : numpy.ndarray
            Input to the activation function. Shape (n, 2)

        Returns
        -------
        activation : numpy.ndarray
            Activation of the neurons in the layer.
        """

        return np.exp(-np.linalg.norm(
            x.reshape(-1, 2) - self.centers.reshape(-1, 1,
                                        2), axis=2)**2 / self.sigma)

    def plot(self):
        plt.scatter(self.centers[:, 0], self.centers[:, 1], s=10)
        plt.show()


class PlotPCNN:

    def __init__(self, model: PCNN,
                 makefig: bool=True):

        self._model = model
        if makefig:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        else:
            self._fig, self._ax = None, None

    def render(self, trajectory: np.ndarray=None,
               ax=None, new_a: np.ndarray=None):

        if ax is None:
            ax = self._ax
            ax.clear()

        new_a = new_a if new_a is not None else self._model.u

        # --- trajectory
        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-',
                          lw=0.5, alpha=0.6)

        # --- network
        centers = self._model._centers
        connectivity = self._model._Wrec

        ax.scatter(centers[:, 0],
                   centers[:, 1],
                   c=new_a.flatten(),
                   s=40, cmap='viridis',
                   vmin=0, vmax=0.04)

        for i in range(connectivity.shape[0]):
            for j in range(connectivity.shape[1]):
                if connectivity[i, j] > 0:
                    ax.plot([centers[i, 0], centers[j, 0]],
                            [centers[i, 1], centers[j, 1]], 'k-',
                            alpha=0.2, lw=0.5)

        #
        ax.axis('off')
        ax.set_ylim((0, 1))
        ax.set_xlim((0, 1))
        ax.set_title(f"PCNN | N={len(self._model)}")

        if ax == self._ax:
            self._fig.canvas.draw()





""" local utils """


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


@jit(nopython=False)
def calc_weight_connectivity(M: np.ndarray,
                             threshold: float=0.5,
                             triangular: bool=False) -> np.ndarray:
    """
    calculate the connectivity of the weights
    ---
    >>> TODO <<<
    consider using the `cosine_similarity` function below
    """

    # Manually compute the norm along axis 1
    norm_M = np.sqrt((M ** 2).sum(axis=1))

    # Compute the cosine similarity
    cosine_sim = (M @ M.T) / (norm_M[:, None] * norm_M[None, :])

    # make it lower triangular
    if triangular:
        cosine_sim *= np.tril(np.ones_like(cosine_sim), k=-1)

    # Set the diagonal to zero
    cosine_sim = np.where(np.isnan(cosine_sim), 0., cosine_sim)

    cosine_sim *= 1 - np.eye(cosine_sim.shape[0])
    return np.where(cosine_sim > (1-threshold), cosine_sim, 0.)


def calc_repulsion(M: np.ndarray,
                   threshold: float) -> np.ndarray:

    """
    calculate the repulsion level for
    every pair i,j in the matrix
    """

    # normalized matrix dot product
    repulsion = np.max(cosine_similarity(M=M), axis=0)

    return np.where(repulsion < threshold, 1., 0.).reshape(-1, 1)


def cosine_similarity(M: np.ndarray):

    """
    calculate the cosine similarity
    """

    # normalized matrix dot product
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    M = np.where(np.isnan(M), 0., M)
    return (M @ M.T) * (1 - np.eye(M.shape[0]))


def k_most_neighbors(M: np.ndarray, k: int):

    """
    set at most k neighbors for each tuned neuron
    """

    # Get the absolute values of the matrix
    abs_matrix = np.abs(M)

    # Sort indices of each row based on the absolute values
    sorted_indices = np.argsort(abs_matrix, axis=1)

    # Create a mask to zero out elements
    mask = np.ones(M.shape, dtype=bool)

    # For each row, set the smallest elements to zero
    # if the count exceeds k
    for i in range(M.shape[0]):
        if np.count_nonzero(M[i]) > k:
            mask[i, sorted_indices[i, :-k]] = False

    # Apply the mask to the original matrix
    M[~mask] = 0

    return M


@jit(nopython=True)
def calc_centers_from_layer(wff: np.ndarray, centers: np.ndarray):
    """
    calculate the centers of the downstream pc from the centers of
    the input pc layer and the feedforward weight matrix
    """

    # get the positions of the centers along the two axes
    X = centers[:, 0]
    Y = centers[:, 1]

    x = (wff * X).sum(axis=1) / wff.sum(axis=1)
    y = (wff * Y).sum(axis=1) / wff.sum(axis=1)

    x = np.where(np.isnan(x), -np.inf, x)
    y = np.where(np.isnan(y), -np.inf, y)

    return np.column_stack((x, y))



""" for testing """""

def make_trajectory(plot: bool=False, speed: float=0.001,
                   duration: int=2, Nj: int=13**2, sigma: float=0.01,
                   **kwargs) -> tuple:

    """
    make a trajectory parsed through a place cell layer.

    Parameters
    ----------
    plot: bool
        plot the trajectory.
    speed: float
        speed of the trajectory.
    duration: int
        duration of the trajectory.
    Nj: int
        number of place cells.
    sigma: float
        variance of the place cells.
    **kwargs
        is2d: bool
            is the trajectory in 2D.
            Default is True.
        dx: float
            step size of the trajectory.
            Default is 0.005.

    Returns
    -------
    trajectory: np.ndarray
        trajectory of the agent.
    whole_track: np.ndarray
        whole track of the agent.
    inputs: np.ndarray
        inputs to the place cell layer.
    whole_track_layer: np.ndarray
        whole track of the place cell layer.
    layer: PlaceLayer
        place cell
    """

    # settings
    bounds = kwargs.get("bounds", (0, 1, 0, 1))
    is2d = kwargs.get("is2d", True)
    dx = kwargs.get("dx", 0.005)

    # make activations
    layer = it.PlaceLayer(N=Nj,
                          sigma=sigma,
                          bounds=bounds)

    trajectory, inputs, whole_track, whole_track_layer = utils.make_env(
        layer=layer, duration=duration, speed=0.1,
        dt=None, distance=None, dx=1e-2,
        plot=False,
        verbose=True,
        bounds=bounds,
        line_env=False,
        make_full=kwargs.get("make_full", False),
        dx_whole=5e-3)

    if plot:
        plt.figure(figsize=(3, 3))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'k-')
        plt.xticks(())
        plt.yticks(())
        plt.axis('off')
        plt.show()

    return trajectory, whole_track, inputs, whole_track_layer, layer


def train(inputs: np.ndarray, layer_centers: np.ndarray,
          params: dict):

    model = PCNN(**params)
    for x in inputs:
        model(x=x.reshape(-1, 1))

    info = {"centers": calc_centers_from_layer(wff=model._Wff,
                                   centers=layer_centers),
            "connectivity": model._Wrec.copy(),
            "count": len(model)}

    return info


def plot_network(centers: np.ndarray,
                 connectivity: np.ndarray, ax: object=None):

    """
    plot the network
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(centers[:, 0], centers[:, 1], 'ko', markersize=2)
    for i in range(connectivity.shape[0]):
        for j in range(connectivity.shape[1]):
            if connectivity[i, j] > 0:
                ax.plot([centers[i, 0], centers[j, 0]],
                        [centers[i, 1], centers[j, 1]], 'k-',
                        alpha=0.2, lw=0.5)

    ax.axis('off')

    return ax


def simple_run():

    np.random.seed(0)
    duration = 20
    N = 70
    Nj = 13**2

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.2,
        "beta": 20.0,
        "clip_min": 0.005,  # "clip_min": 0.005,
        "threshold": 0.3,
        "rep_threshold": 0.8,
        "rec_threshold": 0.1,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
    }

    # make trajectory
    trajectory, _, inputs, _, layer = make_trajectory(
                            plot=False,
                            Nj=Nj,
                            duration=duration,
                            is2d=True,
                            sigma=0.01)

    pclayer = PClayer(n=13, sigma=0.007)
    use_pc = 1
    if use_pc:
        params["xfilter"] = pclayer
        logger.debug(f"{pclayer=}")
    model = PCNN(**params)

    # train
    fig, ax = plt.subplots(figsize=(6, 6))

    if not use_pc:
        for t, x in tqdm_enumerate(inputs):
            model(x=x.reshape(-1, 1))

            if t % 50 == 0:
                ax.clear()
                # ax.imshow(model._Wff, cmap='viridis', aspect='auto')
                ax.plot(trajectory[:t, 0], trajectory[:t, 1], 'r-',
                        lw=0.5, alpha=0.4)
                centers = calc_centers_from_layer(wff=model._Wff,
                                                  centers=layer.centers)
                                                  # centers=model.xfilter.centers)
                plot_network(centers=centers,
                             connectivity=model._Wrec,
                             ax=ax)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(f"t={t} | {len(model)} neurons")
                plt.pause(0.005)
    else:
        for t, x in tqdm_enumerate(trajectory):
            model.add_update(1.)
            model(x=x.reshape(-1, 1))

            if t % 50 == 0:
                ax.clear()
                # ax.imshow(model._Wff, cmap='viridis', aspect='auto')
                ax.plot(trajectory[:t, 0], trajectory[:t, 1], 'r-',
                        lw=0.5, alpha=0.4)
                centers = calc_centers_from_layer(wff=model._Wff,
                                                  # centers=layer.centers)
                                                  centers=model.xfilter.centers)
                plot_network(centers=centers,
                             connectivity=model._Wrec,
                             ax=ax)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(f"t={t} | {len(model)} neurons")
                plt.pause(0.005)


def experimentI():

    np.random.seed(0)
    duration = 10
    N = 80
    Nj = 13**2

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.2,
        "beta": 20.0,
        "threshold": 0.1,
        "rep_thresold": 0.5,
        "rec_threshold": 0.1,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "eq_ach": 1.,
        "tau_ach": 2.,
        "ach_threshold": 0.9,
    }

    # make trajectory
    trajectory, _, inputs, _, _ = make_trajectory(
                            plot=False,
                            Nj=Nj,
                            duration=duration,
                            is2d=True,
                            sigma=0.01)

    xlayer = PClayer(n=13, sigma=0.01)
    params["xfilter"] = xlayer

    D = 10
    var_name1 = "rec_threshold"
    var_values1 = np.linspace(0., 1., D)

    var_name2 = "threshold"
    var_values2 = np.linspace(0., 1., D)

    results = []
    avg_neighbors = np.zeros((D, D))
    info = []
    logger("training..")
    max_cell_count = 0
    for i, value1 in tqdm_enumerate(var_values1):
        for j, value2 in tqdm_enumerate(var_values2):
            params[var_name1] = value1
            params[var_name2] = value2
            info = train(inputs=trajectory, layer_centers=xlayer.centers,
                         params=params)
            results += [info]
            avg_neighbors[i, j] = info["connectivity"].sum() / info["count"]

            max_cell_count = max(max_cell_count, info["count"])

    logger.debug(f"max cell count: {max_cell_count}")
    logger("plotting..")

    # plot
    plt.figure(figsize=(9, 9))
    plt.imshow(np.flip(avg_neighbors, axis=0),
               cmap='viridis')
    plt.colorbar()
    plt.xlabel(var_name2)
    plt.ylabel(var_name1)
    xtickslab = [" "] * D
    xtickslab[0] = f"{var_values2[0]:.2f}"
    xtickslab[-1] = f"{var_values2[-1]:.2f}"
    plt.xticks(range(D), xtickslab)
    ytickslab = [" "] * D
    ytickslab[-1] = f"{var_values1[0]:.2f}"
    ytickslab[0] = f"{var_values1[-1]:.2f}"
    plt.yticks(range(D), ytickslab)
    plt.title("Average number of neighbors")

    plt.show()




if __name__ == "__main__":


    simple_run()

    # experimentI()


