import numpy as np
from abc import ABC, abstractmethod
from numba import jit
import sys
sys.path.append("../")
from utils_core import setup_logger


logger = setup_logger(name="PCNN",
                      level=-1,
                      is_debugging=False,
                      is_warning=False)

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
        self._trace_tau = kwargs.get('trace_tau', 100)

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
        self._connectivity = np.zeros((N, Nj))
        self._Wrec = np.zeros((N, N))
        self._is_plastic = True

        self.trace = np.zeros((N, 1))

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

    def __str__(self):
        return f"PCNN.([py])"

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
        self.trace += (self.u - self.trace) / self._trace_tau
        self.mod_input *= 0

        # update modulators
        if self._is_plastic and not frozen:
            stb, plc = self._calc_indexes()
            if stb is None:
                self._update(x=x, idx=plc)

        # record
        # self.record["u"] += [self.u.tolist()]
        # self.record["umax"] += [self.u.max()]

        return self.u.copy().flatten()

    def __len__(self):
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

            logger.debug(f" [+1] adding new neuron: {similarity=}")

            # proceed
            # self._Wff_backup = self._Wff.copy()
            self.cell_count += 1

            self._Wff_backup = self._Wff.copy()

            if self._calc_recurrent_enable:
                self._update_recurrent()

    def _update_recurrent(self, **kwargs):

        self._Wrec = calc_weight_connectivity(M=self._Wff,
                                    threshold=self._rec_threshold)
        self._connectivity = np.where(self._Wrec > 0, 1., 0.)

        # max number of neighbors
        if self._k_neighbors is not None:
            self._Wrec = k_most_neighbors(M=self._Wrec,
                                          k=self._k_neighbors)

    def fwd_ext(self, x: np.ndarray):
        self(x=x.reshape(-1, 1), frozen=True)
        return self.representation

    def fwd_int(self, u: np.ndarray):

        # u = self.fwd_ext(x=x)
        self.u = self._Wrec @ u.reshape(-1, 1) + self.mod_input
        self.mod_input *= 0
        return self.representation

    @property
    def representation(self):
        return self.u.flatten().copy()

    def get_size(self):
        return self.N

    def get_delta_update(self):
        return self.record["dw"][-1]

    def get_wrec(self):
        return self._Wrec.copy()

    def get_wff(self):
        return self._Wff.copy()

    def get_connectivity(self):
        return self._connectivity.copy()

    def get_trace(self):
        return self.trace.copy()

    def get_centers(self) -> np.ndarray:
        return calc_centers_from_layer(wff=self._Wff,
                                       centers=self.xfilter.centers)

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


class PCLayer(InputFilter):

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

    def render(self):
        plt.scatter(self.centers[:, 0], self.centers[:, 1], s=10)
        plt.axis('off')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()


class LeakyVariable1D:

    def __init__(self, name: str, eq: float,
                 tau: float, min_v: float=0.):

        """
        Parameters
        ----------
        name : str
            name of the variable
        eq : float
            Default 0.
        tau : float
            Default 10
        min_v : float
            Default 0.
        """

        self.name = name
        self._eq = np.array([eq])
        self._v = np.array([eq])
        self._tau = tau
        self._min_v = min_v

    def __repr__(self):
        return f"{self.name}(eq={self.eq}, tau={self.tau})"

    def __call__(self, x: float=0.,
                 simulate: bool=False):

        if simulate:
            v = self._v + (self._eq - self._v) / self._tau + x
            v = np.maximum(self._min_v, v)
            return v

        self._v += (self._eq - self._v) / self._tau + x
        self._v = np.maximum(0., self._v)

        return self._v

    def __len__(self):
        return len(self._v)

    def get_name(self):
        return self.name

    def set_eq(self, eq: float):
        self._eq = eq

    def get_v(self):
        return self._v

    def reset(self):
        self._v = self._eq


class LeakyVariableND:

    def __init__(self, name: str, eq: np.ndarray,
                 tau: float, ndim: int, min_v: float=0.):

        """
        Parameters
        ----------
        name : str
            name of the variable
        eq : np.ndarray
            Default 0.
        tau : float
            Default 10
        min_v : float
            Default 0.
        """

        self.name = name
        self._eq = eq
        self._v = np.ones(ndim) * eq
        self._tau = tau
        self._min_v = min_v
        self._ndim = ndim

    def __str__(self):
        return f"LeakyVariableND.{self.name}()"

    def __repr__(self):
        return f"LeakyVariableND.{self.name}()"

    def __len__(self):
        return self._ndim

    def __call__(self, x: np.ndarray=None,
                 simulate: bool=False):

        if simulate:
            v = self._v + (self._eq - self._v) / self._tau + x
            v = np.maximum(self._min_v, v)
            return v

        self._v += (self._eq - self._v) / self._tau + x

        return self._v

    def set_eq(self, eq: np.ndarray):
        self._eq = eq.flatten()

    def get_v(self):
        return self._v

    def reset(self):
        self._v = np.ones(self._ndim) * self._eq


class ActionSampling2D:

    def __init__(self, speed: float=0.1,
                 name: str=None):

        """
        Parameters
        ----------
        samples : list, optional
            List of samples. The default is None.
        speed : float, optional
            Speed of the agent. The default is 0.1.
        visualize : bool, optional
            Visualize the policy. The default is False.
        number : int, optional
            Number of the figure. The default is None.
        name : str, optional
            Name of the policy. The default is None.
        """

        self._name = name if name is not None else "SamplingPolicy"
        self._samples = [np.array([-speed/np.sqrt(2),
                                   speed/np.sqrt(2)]),
                         np.array([0., speed]),
                         np.array([speed/np.sqrt(2),
                                   speed/np.sqrt(2)]),
                         np.array([-speed, 0.]),
                         np.array([0., 0.]),
                         np.array([speed, 0.]),
                         np.array([-speed/np.sqrt(2),
                                   -speed/np.sqrt(2)]),
                         np.array([0., -speed]),
                         np.array([speed/np.sqrt(2),
                                   -speed/np.sqrt(2)])]

        # np.random.shuffle(self._samples)

        self._num_samples = len(self._samples)
        self._samples_indexes = list(range(self._num_samples))

        self._idx = None
        self._available_idxs = list(range(self._num_samples))
        self._velocity = self._samples[0]
        self._values = np.zeros(self._num_samples)

    def __len__(self):
        return self._num_samples

    def __str__(self):

        return f"{self._name}(#samples={self._num_samples})"

    def __call__(self, keep: bool=False) -> tuple:

        # --- keep the current velocity
        if keep and self._idx is not None:
            return self._velocity.copy(), False, self._idx

        # --- first sample
        if self._idx is None:
            self._idx = np.random.choice(
                            self._samples_indexes, p=self._p)
            self._available_idxs.remove(self._idx)
            self._velocity = self._samples[self._idx]
            return self._velocity.copy(), False, self._idx

        # --- all samples have been tried
        if len(self._available_idxs) == 0:

            if np.where(self._values == 0)[0].size > 1:
                self._idx = np.random.choice(
                                np.where(self._values == 0)[0])
            else:
                self._idx = np.argmax(self._values)

            self._velocity = self._samples[self._idx]
            return self._velocity.copy(), True, self._idx

        # --- sample again
        self._idx = np.random.choice(
                        self._available_idxs)
        self._available_idxs.remove(self._idx)
        self._velocity = self._samples[self._idx]

        return self._velocity.copy(), False, self._idx

    def update(self, score: float=0.):

        # --- normalize the score

        self._values[self._idx] = score

    def reset(self):
        self._idx = None
        self._available_idxs = list(range(self._num_samples))
        self._p = np.ones(self._num_samples) / self._num_samples
        self._velocity = self._samples[0]
        self._values = np.zeros(self._num_samples)


class TwoLayerNetwork:

    def __init__(self, w_hidden: np.ndarray,
                 w_output: np.ndarray):

        w_hidden = np.array(w_hidden)
        w_output = np.array(w_output)

        assert w_hidden.shape == (5, 2), \
            f"{w_hidden.shape=} | expected (5, 2)"
        assert w_output.shape == (2,), \
            f"{w_output.shape=} | expected (2)"


        self.w_hidden = w_hidden.reshape(2, 5)
        self.w_output = w_output.reshape(1, 2)

    def __str__(self):
        return "TwoLayerNetwork"

    def __call__(self, x: np.ndarray):

        h = self.w_hidden @ x

        return (self.w_output @ h).item(), h.flatten().tolist()


class OneLayerNetwork:

    def __init__(self, weights: list):

        self.weights = np.array(weights)

    def __str__(self):
        return "OneLayerNetwork"

    def __call__(self, x: np.ndarray) -> float:

        x = np.array(x).reshape(-1, 1)

        return (self.weights @ x).item(), (self.weights * x.flatten()).tolist()

    def get_weights(self):
        return self.weights



""" FUNCTIONS """


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
    norm_M = np.linalg.norm(M, axis=1, keepdims=True)
    norm_M = np.where(norm_M == 0, 1., norm_M)

    M = M / norm_M
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


def calc_position_from_centers(a: np.ndarray,
                               centers: np.ndarray) -> np.ndarray:

    """
    calculate the position of the agent from the
    activations of the neurons in the layer
    """

    if a.sum() == 0:
        return np.array([np.nan, np.nan])

    return (centers * a.reshape(-1, 1)).sum(axis=0) / a.sum()



""" from evolution """

def load_model_settings(idx: int=None, verbose: bool=True) -> tuple:

    """
    load the results from an evolutionary search and
    return it as `sim_settings`, `agent_settings`, and
    `model_params`
    """

    import json
    def log(msg: str):
        if verbose:
            print(msg)

    # list all files
    file_list = os.listdir(EVOPATH)
    if len(file_list) == 0:
        return None, None, None

    log(f">>> files: {file_list}")

    if idx is None:
        idx = len(file_list) - 1
    log(f">>> idx: {idx}")
    filename = file_list[idx]

    log(f">>> loading file: {filename}")

    # load the file
    with open(f"{EVOPATH}/{filename}", "r") as f:
        data = json.load(f)

    info = data["info"]
    model_params = data["genome"]

    log(f"date: {info['date']}")
    log(f"evolution: {info['evolution']}")
    log(f"evolved: {info['evolved']}")
    log(f"other: '{info['other']}'")
    log(f"performance: {info['performance']}")

    sim_settings = info["data"]["sim_settings"]
    agent_settings = info["data"]["agent_settings"]

    log(f"sim_settings: {sim_settings}")
    log(f"agent_settings: {agent_settings}")
    log(f"model_params: {model_params}")

    # make some lists into numpy arrays
    sim_settings["bounds"] = np.array(sim_settings["bounds"])
    sim_settings["rw_position"] = np.array(sim_settings["rw_position"])
    sim_settings["init_position"] = np.array(sim_settings["init_position"])

    evo_info = {
        "date": info["date"],
        "evolution": info["evolution"],
        "evolved": info["evolved"],
        "other": info["other"],
        "performance": info["performance"]
    }

    return info["data"]["sim_settings"], \
              info["data"]["agent_settings"], \
                model_params, evo_info


