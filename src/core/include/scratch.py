
class SamplingPolicy:

    def __init__(self, samples: list=None,
                 speed: float=0.1,
                 visualize: bool=False,
                 number: int=None,
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
        self._samples = samples
        if samples is None:
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
            logger(f"{self.__class__} using default samples [2D movements]")

        # np.random.shuffle(self._samples)

        self._num_samples = len(self._samples)
        self._samples_indexes = list(range(self._num_samples))

        self._idx = None
        self._available_idxs = list(range(self._num_samples))
        self._p = np.ones(self._num_samples) / self._num_samples
        self._velocity = self._samples[0]
        self._values = np.zeros(self._num_samples)

        # render
        self._number = number
        self.visualize = visualize
        if visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 6))
            logger(f"%visualizing {self.__class__}")

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
            return self._velocity.copy(), False, self._idx, self._values

        # --- all samples have been tried
        if len(self._available_idxs) == 0:

            # self._idx = np.random.choice(self._num_samples,
            #                                   p=self._p)

            if np.where(self._values == 0)[0].size > 1:
                self._idx = np.random.choice(
                                np.where(self._values == 0)[0])
            else:
                self._idx = np.argmax(self._values)

            self._velocity = self._samples[self._idx]
            # print(f"{self._name} || selected: {self._idx} | " + \
            #     f"{self._values.max()} | values: {np.around(self._values, 2)} v={np.around(self._velocity*1000, 2)}")
            return self._velocity.copy(), True, self._idx, self._values

        # --- sample again
        p = self._p[self._available_idxs].copy()
        p /= p.sum()
        self._idx = np.random.choice(
                        self._available_idxs,
                        p=p)
        self._available_idxs.remove(self._idx)
        self._velocity = self._samples[self._idx]

        return self._velocity.copy(), False, self._idx, self._values

    def update(self, score: float):

        # --- normalize the score
        # score = pcnn.generalized_sigmoid(x=score,
        #                                  alpha=-0.5,
        #                                  beta=1.)

        self._values[self._idx] = score

        # --- update the probability
        # a raw score of 0. becomes 0.5 [sigmoid]
        # and this ends in a multiplier of 1. [id]
        # self._p[self._idx] *= (0.5 + score)

        # normalize
        # self._p = self._p / self._p.sum()

    def get_state(self):

        return {"values": self._values,
                "idx": self._idx,
                "p": self._p,
                "velocity": self._velocity,
                "available_idxs": self._available_idxs}

    def set_state(self, state: dict):

        self._values = state["values"]
        self._idx = state["idx"]
        self._p = state["p"]
        self._velocity = state["velocity"]
        self._available_idxs = state["available_idxs"]

    def render(self, values: np.ndarray=None,
               action_values: np.ndarray=None):

        if not self.visualize:
            return

        # self._values = (self._values.max() - self._values) / \
        #     (self._values.max() - self._values.min())
        # self._values = np.where(np.isnan(self._values), 0,
        #                         self._values)

        self.ax.clear()

        if action_values is not None:
            self.ax.imshow(action_values.reshape(3, 3),
                           cmap="RdBu_r", vmin=-1.1, vmax=1.1,
                           aspect="equal",
                           interpolation="nearest")
        else:
            self.ax.imshow(self._values.reshape(3, 3),
                           cmap="RdBu_r", vmin=-3.1, vmax=3.1,
                           aspect="equal",
                           interpolation="nearest")

        # labels inside each square
        for i in range(3):
            for j in range(3):

                if values is not None:
                    text = "".join([f"{np.around(v, 2)}\n" for v in values[3*i+j]])
                else:
                    text = f"{self._samples[3*i+j][1]:.3f}\n" + \
                          f"{self._samples[3*i+j][0]:.3f}"
                self.ax.text(j, i, f"{text}",
                             ha="center", va="center",
                             color="black",
                             fontsize=13)

        # self.ax.bar(range(self._num_samples), self._values)
        # self.ax.set_xticks(range(self._num_samples))
        # self.ax.set_xticklabels(["stay", "up", "right",
        #                          "down", "left"])
        # self.ax.set_xticklabels(np.around(self._values, 2))
        self.ax.set_xlabel("Action")
        self.ax.set_title(f"Action Space")
        self.ax.set_yticks(range(3))
        # self.ax.set_ylim(-1, 1)
        self.ax.set_xticks(range(3))

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return

        self.fig.canvas.draw()

    def reset(self):
        self._idx = None
        self._available_idxs = list(range(self._num_samples))
        self._p = np.ones(self._num_samples) / self._num_samples
        self._velocity = self._samples[0]
        self._values = np.zeros(self._num_samples)

    def has_collided(self):

        self._velocity = -self._velocity



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

    def render(self):
        plt.scatter(self.centers[:, 0], self.centers[:, 1], s=10)
        plt.axis('off')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()



class LeakyVariable:

    def __init__(self, eq: float=0., tau: float=10,
                 name: str="leaky", ndim: int=1,
                 visualize: bool=False,
                 number: int=None,
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
        self._visualize = False
        self._number = number

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

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/{self._number}.png")
            return
        self.fig.canvas.draw()



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

        # max number of neighbors
        if self._k_neighbors is not None:
            self._Wrec = k_most_neighbors(M=self._Wrec,
                                          k=self._k_neighbors)

    def clean_recurrent(self, wall_vectors: np.ndarray):

        self._Wrec = remove_wall_intersecting_edges(
            nodes=self._centers.copy(),
            connectivity_matrix=self._Wrec.copy(),
            walls=wall_vectors
        )

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

    def fwd_ext(self, x: np.ndarray,
                frozen: bool=False):
        self(x=x.reshape(-1, 1), frozen=frozen)
        return self.representation

    def fwd_int(self, u: np.ndarray):

        # u = self.fwd_ext(x=x)
        self.u = self._Wrec @ u.reshape(-1, 1) + self.mod_input
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




def make_orthonormal(matrix):
    v1, v2 = matrix.T
    v2 = v2 - (v1 @ v2) / (v1 @ v1) * v1
    matrix[:, 0] = v1
    matrix[:, 1] = v2
    for i in range(len(matrix.T)):
        matrix[i, :] /= matrix[i, :].sum()
    return matrix

    for i in range(len(matrix)):
        if matrix[i, 1] < 0:
            if np.random.random() < 0.5:
                matrix[i, 1] = 0
            else:
                matrix[i, 1] *= -1
                matrix[i, 0] = 0

    return matrix





