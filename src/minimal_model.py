import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import convolve2d
import copy
from numba import jit

import json, os

import ratinabox
from ratinabox.Neurons import Neurons

try:
    import utils
    import visualizations as vis
except ImportError:
    try:
        import sys
        exp_path = os.getcwd().split("PCNN")[0] + "PCNN"

        sys.path.append(os.path.expanduser(exp_path))
        import src.utils as utils
        import src.visualizations as vis
    except ImportError:
        raise ImportError("`utils.py` not found in `/src`")

try:
    import inputools.Trajectory as it
except ImportError:
    raise ImportError("`inputools` not found")


DEBUG = False
logger = utils.logger




""" minimal PCNN model """


class Recorder:

    def __init__(self, max_size: int=None):

        self.indexes = []
        self._centers = []
        self._connections = []

        # activity
        self._max_size = max_size
        self._activity = {'u': [],
                          'da': [],
                          'ach': []}

    def record_activity(self, u: np.ndarray=None,
                        da: float=None, ach: float=None):

        """
        record the activity of the network

        Parameters
        ----------
        u: np.ndarray
            activations of the neurons
        da: float
            dopamine levels
        ach: float
            acetilcholine levels
        """

        if self._max_size is not None:
            if len(self._activity['u']) > self._max_size:
                self._activity['u'].pop(0)
                self._activity['da'].pop(0)
                self._activity['ach'].pop(0)

        if u is not None:
            if isinstance(u, np.ndarray):
                u = u.flatten().tolist()
            self._activity['u'].append(u)

        if da is not None:
            self._activity['da'].append(da)

        if ach is not None:
            self._activity['ach'].append(ach)

    def get_activity(self, key: str) -> list:

        """
        get the activity of the network

        Parameters
        ----------
        key: str
            key of the activity

        Returns
        -------
        list
            activity of the network
        """

        return self._activity[key]

    def __len__(self):

        if len(self._activity['u']) == 0:
            return 0

        return len(np.array(self._activity['u'])[:, 0])

    @property
    def centers(self):
        return np.array(self._centers)

    @property
    def connections(self):
        return np.array(self._connections)

    def add_many_centers(self, centers: np.ndarray):

        """
        add many centers to the list of centers

        Parameters
        ----------
        centers: np.ndarray
            centers to add
        """

        for i, center in enumerate(centers):
            self.add_center(center=center, idx=i)

    def add_center(self, center: list,
                   idx: int):

        """
        add a center to the list of centers

        Parameters
        ----------
        center: list
            center to add
        idx: int
            index of the center
        """

        self._centers.append(center)
        self.indexes.append(idx)

    def add_connection(self, src: int, trg: int):

        """
        add a connection between two centers

        Parameters
        ----------
        src: int
            source index
        trg: int
            target index
        """

        self._connections.append((src, trg))

    def has_centers(self):
        return len(self.centers) > 0

    def has_connections(self):
        return len(self.connections) > 0

    def set_centers(self, centers: list):

        """
        set the centers

        Parameters
        ----------
        centers: list
            list of centers
        """

        self._centers = np.array(centers) if isinstance(centers,
                                    list) else centers


# || note: `Recorder` is not used in the current implementation
# ||       but it is kept for future implementations
# ||       . before it was used to record the centers and connections

class minPCNN():

    """
    Learning place fields
    - formation: dependant on ACh levels (exploration)
    - remapping: dependant on DA levels (reward)

    Important parameters:
    - learning rate : speed of remapping
    - threshold : area of the place field
    - epsilon : repulsion threshold
    """

    def __init__(self, N: int, Nj: int, **kwargs):

        """
        Parameters
        ----------
        N: int
            number of neurons
        Nj: int
            number of place cells
        **kwargs: dict
            additional parameters for the `minPCNN`
            alpha: float
                threshold parameter of the sigmoid. Default is 0.1
            beta: float
                gain parameter of the sigmoid. Default is 20.0
            lr: float
                learning rate. Default is 0.012
            threshold: float
                threshold for the winner. Default is 0.3
            tau_da: float
                time constant for the dopamine. Default is 150
        """

        super().__init__()

        # parametes
        self.N = N
        self.Nj = Nj
        self._tau = kwargs.get('tau', 10.0)
        self._alpha = kwargs.get('alpha', 0.1)
        self._beta = kwargs.get('beta', 20.0)
        self._lr = kwargs.get('lr', 0.012)
        self._threshold = kwargs.get('threshold', 0.3) # density of the fields
        self._indexes = np.arange(N)
        self._upper_fr = kwargs.get('upper_fr', 0.05)

        self._epsilon = kwargs.get('epsilon', 0.01)
        self._rec_epsilon = kwargs.get('rec_epsilon', 0.0001)
        self._k_neighbors = kwargs.get('k_neighbors', 7)

        self._calc_recurrent_enable = kwargs.get('calc_recurrent_enable', True)

        # variables
        self.u = np.zeros((N, 1))
        self._umask = np.zeros((N, 1))
        self._Wff = np.zeros((N, Nj))
        self._Wff_backup = np.zeros((N, Nj))
        self.W_rec = np.zeros((N, N))

        # learning
        self._dw_rec = None
        self._is_plastic = True
        self.is_idx = False

        # acetilcholine
        self._eq_ach_or = kwargs.get('eq_ach', 1.0)
        self._eq_ach = self._eq_ach_or
        self._tau_ach_or = kwargs.get('tau_ach', 50)
        self._tau_ach = self._tau_ach_or
        self._ach_threshold = kwargs.get('ach_threshold', 0.5)
        self._ACh = self._eq_ach
        self._ach_enabled = 0

        # dopamine
        self._eq_da_or = kwargs.get('eq_da', 0.) 
        self._eq_da = self._eq_da_or
        self._tau_da_or = kwargs.get('tau_da', 1.)
        self._tau_da = self._tau_da_or
        self._da_threshold = kwargs.get('da_threshold', 0.99)
        self._DA = self._eq_da
        self._da_enabled = 0

        self._adaptive_threshold = np.zeros(self.N) + \
            self._threshold

        # record
        self.var1 = 0.
        # self.dwda = np.zeros((N, Nj))
        self.dwda = 0.
        self.flag = False
        self.record = {
            "u": [],
            "umax": [],
            "da": [],
            "ach": [],
        }

        self.rps = np.zeros((N, 1))
        self.birth_similarity = np.zeros((N, 1))
        self.cell_count = 0

    def __repr__(self):

        return f"minPCNN(N={self.N}, Nj={self.Nj}, lr={self._lr})"

    def __call__(self, x: np.ndarray, frozen: bool=False,
                 **kwargs):

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

        # self._tau_da = kwargs.get("tau", self._tau_da_or)
        self._eq_da = kwargs.get("eq_da", self._eq_da_or)

        if (sum(x.shape)-1) > self.Nj:
            x = x / x.sum(axis=0) * 1.
            x = x.reshape(1, self.Nj, -1)
        else:
            x = x / x.sum() * 1.

        # step `u` | x : PC Nj > neurons N
        self.u = utils.generalized_sigmoid(x=self._Wff @ x,
                                           alpha=self._alpha,
                                           beta=self._beta,
                                           clip_min=1e-3)
        # self.u = np.where(self.u < 2e-2, 0.0, self.u)
        self.u = np.where(self.u < 2e-3, 0.0, self.u)

        # [0, 1] normalization
        # self.u /= max((self.u.max(), self._upper_fr))

        # update modulators
        if self._is_plastic and not frozen:
            active_idx = self._update_modulators()
            self._update(x=x, idx=active_idx)
        # else:
        #     stb_idx, _ = self._calc_indexes()

        # record activity
        # self.record_activity(u=self.u,
        #                      da=self._DA,
        #                      ach=self._ACh)

        if kwargs.get("return_u", False):
            return self.u.reshape(self.N, -1).copy()

    def __len__(self):

        """
        number of tuned neurons
        """

        return self._umask.sum()

    @property
    def _repulsion(self):

        """
        calculate the repulsion
        """

        # return np.where(calc_repulsion(self._Wff).sum(axis=1) > 0.8,
        #                 0., 1.).reshape(-1, 1)
        return np.where((self._Wff @ self._Wff.T - \
            np.eye(self.N)).sum(axis=1) > 0.0,
                        0., 1.).reshape(-1, 1)

    def _calc_indexes(self) -> tuple:

        """
        calculate the indexes of the stable neurons and the winner
        """

        # calculate stable indexes as the index of the neurons
        # with weights close to wmax | skip the index 0
        # stable_indexes = np.where(np.abs(1 - self._Wff.sum(axis=1)) < 0.001)[0]
        stable_indexes = np.where(self._Wff.sum(axis=1) > 0.99)[0]
        empty_indexes = self._indexes[~np.isin(self._indexes, stable_indexes)]

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

            if stb_winner == 0: # !
                stb_winner = None
            # elif self.u[stb_winner] < self._threshold: # <----------------------------------
            elif self.u[stb_winner] < self._adaptive_threshold[stb_winner]: # <----------------------------------
                stb_winner = None
            else:
                return stb_winner, None

        # --- define plastic neurons
        # argmax of the activations of non-stable neurons
        plastic_indexes = [i for i in range(self.N) if i not in stable_indexes]

        if len(plastic_indexes) == 0:
            return None, None

        plastic_winner = np.random.choice(plastic_indexes)
        # plastic_winner = np.argmax(self.u[plastic_indexes])

        return None, plastic_winner

    def _update_modulators(self):

        """
        update the modulators
        """

        self._ACh += (self._eq_ach - self._ACh) / self._tau_ach
        self._DA += (self._eq_da - self._DA) / self._tau_da

        # determine the active neuron
        stb_idx, plastic_idx = self._calc_indexes()


        # record
        self.record["u"] += [self.u.tolist()]
        self.record["umax"] += [self.u.max()]
        self.record["da"] += [self._DA]
        self.record["ach"] += [self._ACh]

        # current from the stb_idx
        if stb_idx:
            self._ach_enabled = 0
            return None

        self._ach_enabled = 1

        return plastic_idx

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

        # --- REMAPPING ---
        # the stable neurons get closer to the current
        # location `x`
        if idx is None:

            # if np.where(self.u > 0.04, 1, 0).sum() > 1:
            #     logger.debug(f"{np.where(self.u > 0.04, 1, 0).sum()}")
            #     return

            # if self.flag: return
            da_enabled = (self._DA > self._da_threshold) * self._umask
            #     (x.flatten() - self._Wff) * self._umask

            self.rps = calc_repulsion(M=self._Wff.copy(),
                                      epsilon=self._epsilon)
            attr = calc_attraction(M=self._Wff.copy(),
                                   x=x,
                                   beta=20.,
                                   alpha=0.7)

            # weight update: magnitude * distance function
            dw = (self._Wff - x.reshape(1, -1))
                # np.exp(-((self._Wff - x.reshape(1, -1))**2)) * \
                # np.exp(-np.linalg.norm(self._Wff - x.flatten(),
                #             axis=1)**2).reshape(-1, 1)
            # dw *= utils.generalized_sigmoid(x=np.abs(dw), beta=-30_000,
            #                                 alpha=self._epsilon, clip_min=0.1)

            # dw = dw * dist.reshape(-1, 1) * self._lr
            # self._dw_rec = dw.sum()
            # dw = dw * calc_repulsion(M=dw,
            #                          epsilon=self._epsilon)
            dw *= da_enabled.reshape(-1, 1) * self._lr * self.rps.reshape(-1, 1) * attr.reshape(-1, 1) #* calc_repulsion(M=self._Wff,
                                        # epsilon=self._epsilon)

            # --- update the DA levels
            if dw.sum() > 0.0:
                self._DA *= 0.
                # logger.debug(f"DA. REMAPPING {dw.max():.5f}")
                # logger.debug(f"sum: {dw.sum():.6f}")
                # # logger.debug(f"attraction: {np.around(attr.flatten(), 3)} {attr.sum():.8f}")

                # z = (self._Wff @ x) / (np.linalg.norm(self._Wff, axis=1) *
                #                        np.linalg.norm(x)).reshape(-1, 1)
                # z = np.where(np.isnan(z), 0., z)

                # logger.debug(f"attr: {attr.max():.3f}")
                # logger.debug(f"z max: {z.max():.3f}")

            # if self._DA > 0.05:
            #     logger.debug(f"# DA present")
            #     logger.debug(f"sum: {dw.sum():.6f}")
            #     logger.debug(f"attraction: {np.around(attr.flatten(), 3)} {attr.sum():.8f}")
            #     logger.debug(f"rep: {self.rps.sum():.3f}, z max: {z.max():.3f} x max: {x.max():.2f}")

            # self.dwda = max((dw.max(), self.dwda))
            self._Wff -= dw  #* utils.generalized_softmax(
                # x=dw, beta=1., clip_min=0.0)
            self._Wff_backup = self._Wff.copy()

            # self._Wff *= calc_repulsion(M=self._Wff,
            #                             epsilon=self._epsilon)

            # if da_enabled.sum() > 0:
            #     self.flag = True

            # --- update the adaptive threshold
            # idea: lower the threshold the more a neuron has remapped
            dat = 1 + 10*dw.max(axis=1)
            dat = np.where(dat < 0.4, 0.5, dat)
            # self._adaptive_threshold *= dat
            self._adaptive_threshold *= 1 - 0.5 * attr.flatten()

        # --- FORMATION ---
        # there is an un-tuned neuron ready to learn
        else:

            # if the levels of ACh are high enough, update
            # the plastic neuron
            dw = 1 * (self._ACh > self._ach_threshold) * \
                ((x.flatten() - self._Wff[idx])) * \
                self._ach_enabled

            self.rps = calc_repulsion(M=self._Wff,
                                      epsilon=self._epsilon)

            self._Wff[idx, :] += dw * self.rps[idx]

            # --- update the ACh levels
            if dw.sum() > 0.0:
                # this trick ensures that it isn't close to others
                # || this is necessary only because of a lurking bug
                # || somewhere, ACh and rps are not always enough!
                similarity = max_similarity(M=self._Wff.copy(),
                                                            idx=idx)

                # revert back
                if similarity > 0.6:
                    self._Wff = self._Wff_backup.copy()

                # proceed
                else:
                    self._Wff_backup = self._Wff.copy()
                    self.birth_similarity[idx] = similarity
                    # logger.debug(f"{self.cell_count}. similarity={self.birth_similarity[idx]}")

                    self._ACh *= 0.
                    self.cell_count += 1

                    if self._calc_recurrent_enable:
                        self._update_recurrent()

        if DEBUG:
            print(f"%update: {self._ACh=}, dW={dW.sum():.3f}")

    def step(self, x, **kwargs):

        return self.__call__(x=x, **kwargs)

    def clear_connections(self, epsilon: float=None):

        """
        clear the connections

        Parameters
        ----------
        epsilon : float
            threshold for the weights.
            Default is 0.01
        """

        self._Wff = clear_connectivity(M=self._Wff,
                                       epsilon=epsilon or self._epsilon)

    def set_off(self, cut_weights: bool=False):

        """
        turn off plasticity
        """

        self._is_plastic = False

        if cut_weights:
            self._Wff[~self._umask.flatten().astype(bool), :] = 0.

    def freeze(self, cut_weights: bool=False):

        self._is_plastic = False

        if cut_weights:
            self._Wff[~self._umask.flatten().astype(bool), :] = 0.

    def unfreeze(self):

        self._is_plastic = True

    def _update_recurrent(self, **kwargs):

        self.W_rec = calc_weight_connectivity(M=self._Wff,
                                              threshold=self._rec_epsilon)

        # max number of neighbors
        if self._k_neighbors is not None:
            self.W_rec = k_most_neighbors(matrix=self.W_rec,
                                          k=self._k_neighbors)

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
        self._eq_ach = self._eq_ach_or
        self._eq_da = self._eq_da_or
        self._ACh = self._eq_ach
        self._DA = self._eq_da

        self.record = {
            "u": [],
            "umax": [],
            "da": [],
            "ach": [],
        }

        if complete:
            self._umask = np.zeros((self.N, 1))
            self._Wff = np.zeros((self.N, self.Nj))
            self.W_rec = np.zeros((self.N, self.N))
            self._adaptive_threshold = np.zeros(self.N) + \
                self._threshold


class fullPCNN:

    """
    a `minPCNN` endowed with an input layer of place cells, such
    that it just receives an input position and returns the 
    activation of the learned place cells
    """

    def __init__(self, params: dict, sigma: float=4e-3,
                 bounds: tuple=(0, 1, 0, 1)):

        """
        Parameters
        ----------
        N: int
            number of neurons
        Nj: int
            number of place cells
        sigma: float
            std of the place cells.
            Default is 4e-3
        bounds: tuple
            bounds of the place cells.
            Default is (0, 1, 0, 1)
        **kwargs: dict
            additional parameters for the `minPCNN`
            alpha: float
                threshold parameter of the sigmoid. Default is 0.1
            beta: float
                gain parameter of the sigmoid. Default is 20.0
            lr: float
                learning rate. Default is 0.012
            threshold: float
                threshold for the winner. Default is 0.3
            tau_da: float
                time constant for the dopamine. Default is 150
        """

        self.N = params.get('N', 300)
        self.Nj = params.get('Nj', 25**2)

        # place cells input layer
        self._pc_in = it.PlaceLayer(N=self.Nj, sigma=sigma, bounds=bounds)

        # minimal PCNN
        self._pcnn = minPCNN(**params)

        # copy variables
        self.u = self._pcnn.u
        self._pc_x = np.zeros((self.Nj, 1))
        self._umask = self._pcnn._umask
        self._Wff = self._pcnn._Wff
        self._DA = self._pcnn._DA
        # self._tau_da_baseline = params["tau_da"]
        self._eq_da_baseline = params["eq_da"]

        self.centers = None
        self.connections = None

    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:

        """
        Parameters
        ----------
        x: np.ndarray
            position of the agent
        tau: float
            time constant for the policy. Default is None
        eq_da: float
            equilibrium value for the dopamine. Default is 1.0

        Returns
        -------
        np.ndarray
            activation of the place cells
        """

        if x.shape[0] == 2:
            x = self._pc_in.step(x.reshape(-1, 1))
            x = x.reshape(-1, 1)
        elif x.shape[0] > 2:
            x = self._pc_in.step(x.reshape(-1, 2, 1))
        else:
            x = x.reshape(-1, 1)

        self._pc_x = x

        self._pcnn(x=x, **kwargs)

        # copy variables
        self.u = self._pcnn.u.copy()
        self._umask = self._pcnn._umask.copy()
        self._Wff = self._pcnn._Wff.copy()
        self._DA = self._pcnn._DA

        return self.u.copy().flatten()

    def __repr__(self):

        return f"fullPCNN({self._pcnn}, {self._pc_in})"

    def step(self, x: np.ndarray, tau: float=None,
             eq_da: float=1., **kwargs) -> np.ndarray:

        """
        step the network

        Parameters
        ----------
        x: np.ndarray
            position of the agent
        tau: float
            time constant for the policy. Default is None
        eq_da: float
            equilibrium value for the dopamine. Default is 1.0

        Returns
        -------
        np.ndarray
            activation of the place cells
        """

        return self.__call__(x=x, tau=tau, eq_da=eq_da, **kwargs)

    def set_eq_tau(self, eq: float):

        """
        set the equilibrium value for the dopamine
        and the time constant

        Parameters
        ----------
        eq: float
            equilibrium value for the dopamine
        """

        # self._pcnn._tau_da = max((1,
        #             self._tau_da_baseline * utils.generalized_sigmoid(
        #                           x=tau, alpha=0., beta=1.
        #                           )))
        # self._pcnn._eq_da = max((1,
        #             self._eq_da_baseline * \
        #                          utils.generalized_sigmoid(
        #                           x=eq, alpha=0., beta=1.
        #                           ) * 2))
        pass

    def set_off(self):

        """
        turn off plasticity
        """

        self._pcnn.set_off()

    def make_pc_centers_old(self, trajectory: np.ndarray, bounds: tuple, **kwargs):

        """
        calculate the centers of the generated place cells

        Parameters
        ----------
        trajectory: np.ndarray
            trajectory of the agent
        bounds: tuple
            bounds of the environment
        **kwargs: dict
            additional parameters for the `train_whole_track`
            knn: int
                number of neighbours. Default is 5
            max_dist: float
                maximum distance. Default is 0.13
        """

        pc_in_activations = self._pc_in.parse_trajectory(trajectory=trajectory)

        record = utils.train_whole_track(model=self._pcnn,
                                         whole_track=trajectory,
                                         whole_track_layer=pc_in_activations,
                                         use_a=False)

        centers, connections = vis.plot_centers(model=self._pcnn,
                                                trajectory=pc_in_activations,
                                                track=trajectory,
                                                kernel=np.ones((20)),
                                                plot=False,
                                                threshold=0,
                                                record=record,
                                                use_knn=True,
                                                knn_k=kwargs.get('knn', 5),
                                                max_dist=kwargs.get(
                                                    'max_dist', 0.13),
                                                bounds=bounds)

        self.centers = centers
        self.connections = connections

        return centers

    def plot_graph(self, ax: plt.Axes):

        vis.plot_c(W=self._Wff,
                   layer=self._pc_in,
                   color="orange",
                   k=5,
                   max_dist=0.25,
                   ax=ax,
                   show=False,
                   alpha=0.1)


class PCNNrx(Neurons):

    default_params = {
        "n": 10,
        "name": "PCNNrx",
        "color": "blue",
        "min_fr": 0,
        "max_fr": 1,
    }

    def __init__(self, Agent: object, params: dict={}, 
                 sigma: float=None,
                 pcnn_params: dict=None):

        # `ratinabox` api
        self.Agent = Agent
        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        super().__init__(Agent, self.params)

        # default pcnn_params
        if pcnn_params is None:
            pcnn_params = {
                    "N": self.params["n"],
                    "Nj": 25**2,
                    "alpha": 0.15, # 0.1
                    "beta": 20.0, # 20.0
                    "lr": 0.012,
                    "threshold": 0.3,
                    "da_threshold": 0.5,
                    "tau_da": 50,
                    "eq_da": 1.0,
            }
        else:
            self.n = pcnn_params["N"]

        sigma = sigma if sigma is not None else 5e-3

        self._pcnn_full_params = {
            "pcnn_params": pcnn_params,
            "sigma": sigma,
            "bounds": tuple(self.Agent.Environment.extent)
        }

        self._pcnn = fullPCNN(params=pcnn_params,
                              sigma=sigma,
                              bounds=tuple(self.Agent.Environment.extent))

        self._place_field = np.zeros((int(np.sqrt(pcnn_params["N"])),
                                   int(np.sqrt(pcnn_params["N"]))))
        self._pcnn_centers = None
        self._pcnn_connections = None

        self._is_making_pf = False
        self._makepf_t = 0
        self._last_makepf = 0

        self._knn = 10
        self._max_dist = 0.35

        if ratinabox.verbose is True:
            logger.info("PCNNrx successfully initialised")
        return

    def __repr__(self):

        return f"PCNNrx({self._pcnn})"

    def _make_place_centers(self):

        whole_track = self.Agent.Environment.whole_track.copy()

    def _make_place_field(self):

        firingrate, track = self.get_state(evaluate_at="all",
                                           makepf=False,
                                           makepf_two=True)

        self._place_field = firingrate.reshape(self._pcnn._pcnn.N,
                                                -1).sum(axis=0)
        self._place_field -= self._place_field.mean(axis=0)
        # self._place_field = 1/(1 + np.exp(-15*(self._place_field-0.18)))
        self._place_field = np.where(self._place_field < 0.01,
                                     0., self._place_field * (1 + \
                                     np.exp(-5*self._place_field)))

        self._last_makepf = self._makepf_t

        # make centers
        positions, _, _ = vis.calc_cell_tuning(
            model=self._pcnn, track=track,
            kernel=np.ones((20)),
            threshold=0, record=firingrate.reshape(self._pcnn._pcnn.N, -1))

        connectivity = utils.calc_knn(centers=positions,
                                      k=self._knn,
                                      max_dist=self._max_dist)

        self._pcnn_centers = positions
        self._pcnn_connections = connectivity

    def get_state(self, evaluate_at="agent", **kwargs) -> np.ndarray:

        """Returns the firing rate of the place cells.
        By default position is taken from the Agent and used
        to calculate firinr rates.
        This can also by passed directly (evaluate_at=None,
        pos=pass_array_of_positions)
        or ou can use all the positions in the environment (evaluate_at="all").

        Returns
        -------
        firingrates: np.ndarray
            firing rates
        """

        if evaluate_at == "agent":  # single position
            pos = self.Agent.pos
        elif evaluate_at == "all":   # array of positions 
            pos = self.Agent.Environment.flattened_discrete_coords
        else:
            pos = kwargs["pos"]

        pos = np.array(pos)

        # if multiple positions are provided, the step is frozen
        frozen = pos.shape[0] != 2
        firingrate = self._pcnn(x=pos,
                                frozen=frozen)

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]

        # make the place field
        if kwargs.get("makepf", True) and self._makepf_t%50 == 0. and \
            self._is_making_pf:
            self._make_place_field()

        if kwargs.get("makepf_two", False):
            return firingrate, pos.reshape(-1, 2)

        self._makepf_t += 1

        return firingrate # (#cells, #positions)

    def plot_place_cell_locations(
        self,
        whole_track: np.ndarray,
        fig=None,
        ax=None,
        autosave=None,
    ):
        """
        Scatter plots where the centre of the place cells are

        %% not implemented yet %%
        """

        if self._pcnn.centers is None:

            self.place_cell_centres = self._pcnn.make_pc_centers(
                trajectory=self.Agent.Environment.whole_track,
                bounds=tuple(self.Agent.Environment.extent),
                knn=5,
                max_dist=0.13
            )

        if fig is None and ax is None:
            fig, ax = self.Agent.Environment.plot_environment(autosave=False)
        else:
            _, _ = self.Agent.Environment.plot_environment(
                fig=fig, ax=ax, autosave=False
            )
        place_cell_centres = self.place_cell_centres

        x = place_cell_centres[:, 0]
        if self.Agent.Environment.dimensionality == "1D":
            y = np.zeros_like(x)
        elif self.Agent.Environment.dimensionality == "2D":
            y = place_cell_centres[:, 1]

        ax.scatter(
            x,
            y,
            c="C1",
            marker="x",
            s=15,
            zorder=2,
        )
        if autosave:
            ratinabox.utils.save_figure(fig, "place_cell_locations", save=autosave)

        return fig, ax

    def plot_graph(self, ax: plt.Axes=None,
                   fig: plt.Figure=None):

        """
        plot the graph of the place cells
        """

        # make graph


        return fig, ax

    def set_eq_tau(self, eq: float):

        """
        set the equilibrium value for the dopamine
        and the time constant

        Parameters
        ----------
        eq_da: float
            equilibrium value for the dopamine
        """

        self._pcnn.set_eq_tau(eq=eq)

    def flag_make_pf(self, flag: bool=None):

        """
        flag to make the place field
        """

        self._is_making_pf = not self._is_making_pf if flag is None else flag

    def remap(self):
        logger.warning("remap not implemented for PCNNrx")
        return

    def reset(self, kind: str="soft"):

        """
        reset the model by making a new network
        """

        if kind == "soft":
            self._pcnn._pcnn.reset()
        else:
            self._pcnn = fullPCNN(params=self._pcnn_full_params["pcnn_params"],
                                  sigma=self._pcnn_full_params["sigma"],
                                  bounds=self._pcnn_full_params["bounds"])

        self._makepf_t = 0

        return


class Policy:

    """
    Implement the effect of a rewarded position in the
    environment by modulating the dopamine levels
    """

    def __init__(self, tau: float=None, eq_da: float=1.,
                 trg: np.ndarray=np.array([0.5, 0.5]),
                 threshold: float=0.3,
                 startime: int=0):

        self._tau = tau
        self.x = tau
        self._eq_da = eq_da
        self._trg = trg
        self._startime = startime
        self._threshold = threshold
        self.t = 0

    def __repr__(self):

        return f"Policy(eq_da={self._eq_da}, trg={self._trg}" + \
                f", startime={self._startime}ms, threshold={self._threshold})"

    def __call__(self, pos: np.ndarray, **kwargs):

        self.t += 1

        if self.t < self._startime:
            return 0.

        # calculate the distance to the target
        dist = np.linalg.norm(pos - self._trg)

        return self._eq_da * (1 - \
           utils.generalized_sigmoid(x=dist, alpha=self._threshold,
                                     beta=50., clip_min=0.))

    def draw(self, ax: object, alpha: float=0.9):

        if self.t > self._startime:
            # ax.scatter(self._trg[0], self._trg[1], s=200,
            #            marker="x", color="green",
            #            alpha=alpha)
            ax.add_patch(Circle(self._trg,
                                self._threshold, fc="green", ec='green',
                                alpha=alpha, label="reward"))


class PlaceCells(Neurons):

    """
    The PlaceCells class defines a population of PlaceCells.
    This class is a subclass of Neurons() and inherits it properties/plotting functions.

       Must be initialised with an Agent and a 'params' dictionary.

       PlaceCells defines a set of 'n' place cells scattered across the environment.
       The firing rate is a functions of the distance from the Agent to the place cell centres.
       This function (params['description'])can be:
           • gaussian (default)
           • gaussian_threshold
           • diff_of_gaussians
           • top_hat
           • one_hot

       List of functions:
           • get_state()
           • plot_place_cell_locations()

       default_params = {
               "n": 10,
               "name": "PlaceCells",
               "description": "gaussian",
               "widths": 0.20,
               "place_cell_centres": None,  # if given this will overwrite 'n',
               "wall_geometry": "geodesic",
               "min_fr": 0,
               "max_fr": 1,
               "name": "PlaceCells",
           }
    """

    default_params = {
        "n": 10,
        "name": "PCNNrx",
        "description": "gaussian",
        "widths": 0.20,  # the radii
        "place_cell_centres": None,  # if given this will overwrite 'n',
        "wall_geometry": "geodesic",
        "min_fr": 0,
        "max_fr": 1,
        "name": "PlaceCells",
    }

    def __init__(self, Agent, params={}):
        """Initialise PlaceCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}.
        """

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        if self.params["place_cell_centres"] is None:
            self.params["place_cell_centres"] = self.Agent.Environment.sample_positions(
                n=self.params["n"], method="uniform_jitter"
            )
        elif type(self.params["place_cell_centres"]) is str:
            if self.params["place_cell_centres"] in [
                "random",
                "uniform",
                "uniform_jitter",
            ]:
                self.params[
                    "place_cell_centres"
                ] = self.Agent.Environment.sample_positions(
                    n=self.params["n"], method=self.params["place_cell_centres"]
                )
            else:
                raise ValueError(
                    "self.params['place_cell_centres'] must be None, an array of locations or one of the instructions ['random', 'uniform', 'uniform_jitter']"
                )
        else:
            self.params["n"] = self.params["place_cell_centres"].shape[0]
        self.place_cell_widths = self.params["widths"] * np.ones(self.params["n"])

        super().__init__(Agent, self.params)

        # Assertions (some combinations of boundary condition and
        # wall geometries aren't allowed)
        if self.Agent.Environment.dimensionality == "2D":
            if all(
                [
                    (
                        (self.wall_geometry == "line_of_sight")
                        or ((self.wall_geometry == "geodesic"))
                    ),
                    (self.Agent.Environment.boundary_conditions == "periodic"),
                    (self.Agent.Environment.dimensionality == "2D"),
                ]
            ):
                print(
                    f"{self.wall_geometry} wall geometry only possible in 2D when the boundary conditions are solid. Using 'euclidean' instead."
                )
                self.wall_geometry = "euclidean"
            if (self.wall_geometry == "geodesic") and (
                len(self.Agent.Environment.walls) > 5
            ):
                print(
                    "'geodesic' wall geometry only supported for enivironments with 1 additional wall (4 bounding walls + 1 additional). Sorry. Using 'line_of_sight' instead."
                )
                self.wall_geometry = "line_of_sight"

        if ratinabox.verbose is True:
            print(
                "PlaceCells successfully initialised. You can see where they are centred at using PlaceCells.plot_place_cell_locations()"
            )
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Returns the firing rate of the place cells.
        By default position is taken from the Agent and used to calculate firinf rates. This can also by passed directly (evaluate_at=None, pos=pass_array_of_positions) or ou can use all the positions in the environment (evaluate_at="all").

        Returns:
            firingrates: an array of firing rates
        """
        if evaluate_at == "agent":
            pos = self.Agent.pos
        elif evaluate_at == "all": #
            pos = self.Agent.Environment.flattened_discrete_coords  # array of positions 
        else:
            pos = kwargs["pos"]
        pos = np.array(pos)

        # place cell fr's depend only on how far the agent is from cell centres (and their widths)
        dist = (
            self.Agent.Environment.get_distances_between___accounting_for_environment(
                self.place_cell_centres, pos, wall_geometry=self.wall_geometry
            )
        )  # distances to place cell centres
        widths = np.expand_dims(self.place_cell_widths, axis=-1)

        if self.description == "gaussian":
            firingrate = np.exp(-(dist**2) / (2 * (widths**2)))
        if self.description == "gaussian_threshold":
            firingrate = np.maximum(
                np.exp(-(dist**2) / (2 * (widths**2))) - np.exp(-1 / 2),
                0,
            ) / (1 - np.exp(-1 / 2))
        if self.description == "diff_of_gaussians":
            ratio = 1.5
            firingrate = np.exp(-(dist**2) / (2 * (widths**2))) - (
                1 / ratio**2
            ) * np.exp(-(dist**2) / (2 * ((ratio * widths) ** 2)))
            firingrate *= ratio**2 / (ratio**2 - 1)
        if self.description == "one_hot":
            closest_centres = np.argmin(np.abs(dist), axis=0)
            firingrate = np.eye(self.n)[closest_centres].T
        if self.description == "top_hat":
            firingrate = 1 * (dist < self.widths)

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate # (#cells, #positions)

    def plot_place_cell_locations(
        self,
        fig=None,
        ax=None,
        autosave=None,
    ):
        """Scatter plots where the centre of the place cells are

        Args:
            fig, ax: if provided, will plot fig and ax onto these instead of making new.
            autosave (bool, optional): if True, will try to save the figure into `ratinabox.figure_directory`.Defaults to None in which case looks for global constant ratinabox.autosave_plots

        Returns:
            _type_: _description_
        """
        if fig is None and ax is None:
            fig, ax = self.Agent.Environment.plot_environment(autosave=False)
        else:
            _, _ = self.Agent.Environment.plot_environment(
                fig=fig, ax=ax, autosave=False
            )
        place_cell_centres = self.place_cell_centres

        x = place_cell_centres[:, 0]
        if self.Agent.Environment.dimensionality == "1D":
            y = np.zeros_like(x)
        elif self.Agent.Environment.dimensionality == "2D":
            y = place_cell_centres[:, 1]

        ax.scatter(
            x,
            y,
            c="C1",
            marker="x",
            s=15,
            zorder=2,
        )
        ratinabox.utils.save_figure(fig, "place_cell_locations", save=autosave)

        return fig, ax

    def remap(self):
        """Resets the place cell centres to a new random distribution. These will be uniformly randomly distributed in the environment (i.e. they will still approximately span the space)"""
        self.place_cell_centres = self.Agent.Environment.sample_positions(
            n=self.n, method="uniform_jitter"
        )
        np.random.shuffle(self.place_cell_centres)
        return


def calc_repulsion(M: np.ndarray,
                   epsilon: float) -> np.ndarray:

    """
    calculate the repulsion level for
    every i,j in the matrix
    """

    # normalized matrix dot product
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    M = np.where(np.isnan(M), 0., M)
    repulsion = (M @ M.T) * (1 - np.eye(M.shape[0]))

    repulsion = np.max(repulsion, axis=0)

    return np.where(repulsion < epsilon, 1., 0.).reshape(-1, 1)
    # return np.where(repulsion < 0, 1., 0.).reshape(-1, 1)
    # return 1.


def max_similarity(M: np.ndarray, idx: int):

    # normalized matrix dot product
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    M = np.where(np.isnan(M), 0., M)
    similarity = ((M @ M.T) * (1 - np.eye(M.shape[0])))[:, idx]

    return similarity.max()


def calc_attraction(M: np.ndarray,
                    x: np.ndarray,
                    beta: float=1.,
                    alpha: float=0.7) -> np.ndarray:

    """
    calculate the attraction level for
    every i,j in the matrix
    """

    # normalized matrix dot product
    z = (M @ x.reshape(-1, 1)) / \
            (np.linalg.norm(M, axis=1) * \
            np.linalg.norm(x)).reshape(-1, 1)

    z = np.where(np.isnan(z), 0., z)

    return utils.generalized_sigmoid(z,
                                  alpha=alpha,
                                  beta=beta).reshape(-1, 1)

    # return np.where(v < 0.6, 0., 1.)



@jit(nopython=False)
def clear_connectivity(M: np.ndarray,
                       epsilon: float=1e-3) -> np.ndarray:

    """
    if the distance between two non-zero rows is less
    than epsilon, set the one row to zero

    Parameters
    ----------
    M: np.ndarray
        matrix to clear
    epsilon: float
        threshold for the distance

    Returns
    -------
    np.ndarray
        cleared matrix
    """

    mask = np.zeros(M.shape[0])

    for i, row_i in enumerate(M):
        for j, row_j in enumerate(M):
            if i == j or row_i.sum() == 0 or row_j.sum() == 0:
                continue
            if cosine_similarity(row_i, row_j) < epsilon:
                mask[j] = 1.

    return M * mask.reshape(-1, 1)

@jit(nopython=False)
def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:

    """
    calculate the cosine similarity
    between two vectors
    """

    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

@jit(nopython=False)
def calc_weight_connectivity(M: np.ndarray,
                             threshold: float=0.5) -> np.ndarray:
    """
    calculate the connectivity of the weights
    """

    # Manually compute the norm along axis 1
    norm_M = np.sqrt((M ** 2).sum(axis=1))

    # Compute the cosine similarity
    cosine_sim = (M @ M.T) / (norm_M[:, None] * norm_M[None, :])
    # cosine_sim *= (1 - np.eye(cosine_sim.shape[0]))
    cosine_sim *= np.tril(np.ones_like(cosine_sim), k=-1)
    cosine_sim = np.where(np.isnan(cosine_sim), 0., cosine_sim)

    return np.where(cosine_sim < threshold, 0., cosine_sim)


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


# @jit(nopython=False)
def make_edge_list(M: np.ndarray, centers: np.ndarray) -> tuple:

    """
    make a list of edges given the weight (connectivity) matrix
    and the node scenters

    Returns
    -------
    list
        arrays of edges
    np.ndarray
        connectivity matrix
    """

    edges = []
    C = np.zeros((M.shape[0], M.shape[1]))

    for i, row in enumerate(M):
        for j, col in enumerate(row):
            if col > 0.0:
                # edges += [centers[i].tolist() + centers[j].tolist()]
                edges += [np.array([[centers[i][0], centers[i][1]],
                          [centers[j][0], centers[j][1]]])]
                C[i, j] = 1

    return edges, C


# @jit(nopython=True)
def k_most_neighbors(matrix: np.ndarray, k: int):

    # Get the absolute values of the matrix
    abs_matrix = np.abs(matrix)

    # Sort indices of each row based on the absolute values
    sorted_indices = np.argsort(abs_matrix, axis=1)

    # Create a mask to zero out elements
    mask = np.ones(matrix.shape, dtype=bool)

    # For each row, set the smallest elements to zero
    # if the count exceeds k
    for i in range(matrix.shape[0]):
        if np.count_nonzero(matrix[i]) > k:
            mask[i, sorted_indices[i, :-k]] = False

    # Apply the mask to the original matrix
    matrix[~mask] = 0

    return matrix


if __name__ == "__main__":

    model = minPCNN(N=100, Nj=25**2)

    print(dir(model))

    # alpha = 1.0
    # beta = 1.0
    # x = 0.

    # for _ in range(30):
    #     print(f"{x=}")
    #     x = utils.generalized_sigmoid(x, alpha, beta)
