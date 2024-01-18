import numpy as np 
import matplotlib.pyplot as plt 
from itertools import product as iterprod
import warnings
try:
    from tools.utils import logger, tqdm_enumerate
except ModuleNotFoundError:
    warnings.warn('`tools.utils` not found, using fake logger. Some functions may not work')
    class Logger:

        print('Logger not found, using fake logger')

        def info(self, msg: str):
            print(msg)

        def debug(self, msg: str):
            print(msg)

    logger = Logger()
try:
    import inputools.Trajectory as it
except ModuleNotFoundError:
    warnings.warn('`inputools.Trajectory` not found, some functions may not work')


def random_id(length: int=5) -> str:

    """
    Generate a random id of a given length.

    Parameters
    ----------
    length : int
        Length of the id. Default: 5

    Returns
    -------
    id : str
        Random id.
    """

    return ''.join(np.random.choice(list(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    ), length))



# Network classes
# ----------------

# ## Deleted features:

# > adaptive threshold
# ```
# self._bias_max = kwargs.get('bias', 3)
# self._bias_decay = kwargs.get('bias_decay', 100)
# self._bias_scale = kwargs.get('bias_scale', 1.0)

# self._bias += (self._bias_max - self._bias) / self._bias_decay + self._bias_scale * self.u 
# ```

# > dt 
# ```
# self.t += self._dt #* (1 - DA_block)
# ```



class PCNNetwork:

    def __init__(self, N: int, Nj: int, **kwargs):

        """
        Network class 

        Parameters
        ----------
        N: int
            Number of neurons
        Nj: int
            Number of input neurons
        **kwargs: dict
            gain: float
                Gain of the activation function. Default: 7
            bias: float
                Bias of the activation function. Default: 0
            lr: float
                Learning rate. Default: 0.01
            tau: float
                Time constant. Default: 30
            plastic: bool
                Whether the network is plastic. Default: True
            std_tuning: float
                Standard deviation of the tuning curve
            soft_beta: float
                Beta for the softmax function. Default: 10
            wff_max: float
                Maximum value of the feedforward weights. Default: 3
            wff_min: float
                Minimum value of the feedforward weights. Default: 0.0
            wff_tau: float
                Time constant for the feedforward weights decay.
                Default: 500
            plastic: bool
                Whether the network is plastic. Default: True
            nb_per_cycle: int
                Number of neurons per cycle. Default: 6
            nb_skip: int
                Number of cycles to skip. Default: 1
            theta_freq: int
                Frequency of the cycles. Default: 1
            theta_freq_increase: float
                Increase of the frequency of the cycles. Default: 0
            IS_magnitude: float
                Magnitude of the oscillatory stimulation. Default: 3
            DA_tau: float
                Time constant of the dopamine. Default: 100
            is_retuning: bool
                Whether to re-tune the neurons. Default: False
            seed: int
                Seed for the random number generator. Default: None 
        """

        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)

        # General 
        self.N = N
        self.Nj = Nj

        # define instance id as a string of alphanumeric characters
        self.id = random_id()

        # Internal dynamics
        self.u = np.zeros((N, 1))
        self.Ix = np.zeros((N, 1))
        self.Is = np.zeros((N, 1))
        self._lr = kwargs.get('lr', 0.01)
        self._tau = kwargs.get('tau', 30)

        # activation function
        self._gain = kwargs.get('gain', 7)
        self._bias_max = kwargs.get('bias', 3)
        self._bias = self._bias_max * np.ones((N, 1))
        self.activation_func = lambda x: 1 / (
            1 + np.exp(-self._gain * (x - \
                self._bias)))

        # Feedforward weights
        self.Wff = np.zeros((N, Nj))
        self._wff_min = kwargs.get('wff_min', 0.0)
        self._wff_max = kwargs.get('wff_max', 3)
        self._wff_tau = kwargs.get('wff_tau', 500)

        # plasticity
        self.temp = np.ones((N, 1))*1e-4
        self.temp_past = np.ones((N, 1))*1e-4
        self._plastic = kwargs.get('plastic', True)

        # tuning 
        self._nb_per_cycle = self._adjust_num_per_cycle(
            k=kwargs.get('nb_per_cycle', 6), N=self.N)
        self._nb_skip = kwargs.get('nb_skip', 1)
        self._theta_freq = kwargs.get('theta_freq', 1)
        self._theta_freq_increase = kwargs.get('theta_freq_increase', 0.)
        self.tuning = calc_tuning(N=N, K=self._nb_per_cycle, b=self._theta_freq)
        self._IS_magnitude = kwargs.get('IS_magnitude', 3)
        self._range = np.arange(self.N).reshape(-1, 1)
        self._is_retuning = kwargs.get('is_retuning', False)

        # softmax
        beta = kwargs.get('soft_beta', 10)
        self.softmax = lambda x: np.exp(beta*x) / np.exp(beta*x).sum(axis=1,
                                                                     keepdims=True)

        # internal clock
        self.t = 0.
        self._dt = 1.

        # modulation
        self.DA = 1.
        self._DA_tau = kwargs.get('DA_tau', 100)

        # weight update control 
        self.W_old = self.Wff.copy()
        self.W_deriv = np.zeros((self.N, self.Nj))
        self.W_clone = self.Wff.copy()
        self.W_cold_mask = np.zeros((self.N, 1))

        #
        self.var1 = None
        self.var2 = None

    def __repr__(self):

        return f"PCNNetwork(N={self.N}, Nj={self.Nj}) [{self.id}]"

    def _adjust_num_per_cycle(self, k: int, N: int) -> int:

        """
        optimize the number of neurons per cycles given an initial preferred number 

        Parameters
        ----------
        k: int
            Preferred number of neurons per cycle
        N: int
            Number of neurons

        Returns
        -------
        j: int
            Number of neurons per cycle
        """

        # case: N smaller than k
        if N < k:
            return N

        j = 0
        distance = N
        for pair in [[N%i, i] for i in range(1, N)]:
            if pair[0] == 0 and abs(k - pair[1]) < distance:
                j = pair[1]
                distance = abs(k - pair[1])
        return j  

    def _re_tuning(self):

        """
        Calculate the tuning of the neurons, given that some neurons 
        reached a certain temperature
        """

        # get idx of the neurons that reached the temperature
        idx_selective = np.where(self.temp == 1.)[0]
        idx_non_selective = np.where(self.temp != 1.)[0]

        # increase the theta frequency
        self._theta_freq *= (1 + self._theta_freq_increase)
        self._dt *= (1 - self._theta_freq_increase*1.1)

        # calculate the new tuning
        new_tuning = calc_tuning(N=self.N - len(idx_selective),
                                 K=self._nb_per_cycle, b=self._theta_freq)

        # update the tuning
        self.tuning[idx_non_selective] = new_tuning
        self.tuning[idx_selective] = 0.

    def _update(self, x: np.ndarray):

        """
        Update function

        Parameters
        ----------
        x: np.ndarray
            Input from other neurons
        """

        # weight update
        delta = self.u * x.T

        # update weights
        self.Wff += self._lr * delta * self.softmax(self.Wff) * self.DA * (1 - 1*(self.temp == 1.))

        # clip weights
        self.Wff = self.Wff.clip(min=self._wff_min, 
                                 max=self._wff_max)

        # weight decay
        self.Wff += (- self.Wff / self._wff_tau) * (1 - self.temp)

        # temperture
        self.temp = (self.Wff.max(axis=1) / self._wff_max).reshape(-1, 1)

        # calculate the weight cold mask
        self.W_cold_mask = 1 == (self.temp * (1 - np.around(self.W_deriv.sum(axis=1), 3).reshape(-1, 1)))            

        # weight derivative 
        self.W_deriv += - self.W_deriv / 10 + np.abs(self.Wff - self.W_old)
        self.W_old = self.Wff.copy()

        # weight clone
        self.W_clone = self.softmax(self.Wff) 
        self.W_clone = np.where(self.W_clone < 0.1, 0., self.W_clone)
        # normalize such that each row sums to 1
        self.W_clone = 1.*self.W_clone / self.W_clone.sum(axis=1, keepdims=True)
        # set nan to 0
        self.W_clone = np.nan_to_num(self.W_clone, nan=0, posinf=0, neginf=0)
        # self.W_clone = (self.W_clone - self.W_clone.min(axis=1, keepdims=True)) / (
        #     self.W_clone.max(axis=1, keepdims=True) - self.W_clone.min(axis=1, keepdims=True))

    def step(self, x: np.ndarray=None):

        """
        Step function

        Parameters
        ----------
        x: np.ndarray
            Input from other neurons
        """

        # calculate input current
        if x is not None:

            # define weights
            W = self.Wff * (1 - self.W_cold_mask) + self.W_clone * self.W_cold_mask
            self.var1 = W.copy()

            # step
            self.Ix = W @ x * (1 - self.W_cold_mask) + cosine_similarity(W, x) * self.W_cold_mask
            # self.Ix = cosine_similarity(W, x) * self.W

        # calculate synaptic current
        self.Is = self._IS_magnitude * (1 - self.temp) * calc_osc(
                N=self.N, t=self.t,
                I=self._range, O=self.tuning, K=self._nb_per_cycle,
                b=self._theta_freq, nb_skip=self._nb_skip)

        # activation
        self.u = self.activation_func(
            self.u - self.u / self._tau + self.Ix + self.Is
        )

        # update DA
        ut_block = (self.u * self.temp).max()
        DA_block = ut_block * 1*(ut_block >= 0.99)
        self.DA += (1 - DA_block - self.DA) / self._DA_tau
        self.var2 = DA_block

        # update weights
        if self._plastic:
            self._update(x=x)

        # update internal clock
        self.t += 1

    @property
    def output(self):

        """
        Return the output of the network
        """

        return self.u.copy()

    def set_dims(self, N: int, Nj: int):

        """
        Set the dimensions of the network

        Parameters
        ----------
        N: int
            Number of neurons
        Nj: int
            Number of input neurons
        """

        self.N = N
        self.Nj = Nj

        self.reset()

    def reset(self):

        """
        Reset function
        """

        # Internal dynamics
        self.u = np.zeros((self.N, 1))
        self.Ix = np.zeros((self.N, 1))
        self.Is = np.zeros((self.N, 1))

        # weight reset 
        self.Wff = np.zeros((self.N, self.Nj))
        self.W_old = self.Wff.copy()
        self.W_deriv = np.zeros((self.N, self.Nj))
        self.W_clone = self.Wff.copy()

        # tuning
        self.temp = np.ones((self.N, 1))*1e-3
        self._bias = self._bias_max * np.ones((self.N, 1))
        self.t = 0.

        # re-update the tuning
        self._range = np.arange(self.N).reshape(-1, 1)
        self._nb_per_cycle = self._adjust_num_per_cycle(
            k=self._nb_per_cycle, N=self.N)
        self.tuning = calc_tuning(N=self.N, K=self._nb_per_cycle,
                                  b=self._theta_freq)



""" Model functions """


def calc_turn(N: int, t: int, i: int=0, K: int=6, b: int=1, 
              nb_skip: int=1) -> np.ndarray:

    """
    function tat calculates when a certain cycle of neurons should be active 

    Parameters
    ----------
    N : int
        Number of neurons.
    t : int
        Current time step.
    i : int
        Index of the cycle.
    K : int
        Number of neurons per cycle. Default: 6
    b : int
        Frequency scale. Default: 1
    nb_skip : int
        Number of cycles to skip. Default: 1

    Returns
    -------
    turn : np.ndarray
        Array of 1s and 0s indicating whether the cycle should be active or not.
    """

    # size of a cycle in rad
    cycle_size = np.pi / b  
    
    # number of cycles
    nb_cycles = np.ceil(N / K) * nb_skip  

    # total length of the grand cycle
    tot_cycle_len = cycle_size * nb_cycles  

    # map t onto the grand cycle 
    t = t % tot_cycle_len  

    # map i onto the grand cycle
    i = i // K * nb_skip  

    # calculate the current cycle idx and return
    # 1 if it matches the input cycle idx
    return 1*((t // cycle_size) == i)


def calc_tuning(N: int, K: int, b: int) -> np.ndarray:
    
    """
    calculate the tuning of the neurons 

    Parameters
    ----------
    N : int
        Number of neurons.
    K : int
        Number of neurons per cycle.
    b : int
        Frequency of the cycles.

    Returns
    -------
    tuning : np.ndarray
        Tuning of the neurons.
    """

    # calculate the partition over a cycle
    partitions = np.linspace(-np.pi/2/b+1*np.pi/2/b/K,
                             np.pi/2/b+1*np.pi/2/b/K, K, endpoint=0)
    return np.array([partitions[i%K] for i in range(0, N)]).reshape(-1, 1)


def calc_gamma(t: int, O: int, b: int) -> np.ndarray:

    """
    calculate the activity as gamma cycles 

    Parameters
    ----------
    t : int
        Current time step.
    O : int
        Phase offset.
    b : int
        Frequency of the cycles.

    Returns
    -------
    gamma : np.ndarray
        Activity as gamma cycles.
    """

    t = t % (np.pi / b) 
    return np.exp(-(np.sin(b*(t - O))-1)**2 / 1e-6)


def calc_osc(N: int, t: int, I: int, O: int, K: int, 
             b: int, nb_skip: int=1) -> np.ndarray:

    """
    calculate the oscillatory stimulation 

    Parameters
    ----------
    N : int
        Number of neurons.
    t : int
        Current time step.
    I : int
        Index of the cycle.
    O : int
        Phase offset.
    K : int
        Number of neurons per cycle.
    b : int
        Frequency of the cycles.
    nb_skip : int
        Number of cycles to skip. Default: 1

    Returns
    -------
    osc : np.ndarray
        Oscillatory stimulation.
    """

    return calc_turn(N=N, t=t, i=I, K=K, b=b, 
                     nb_skip=nb_skip) * calc_gamma(t=t, O=O, b=b)



""" Connectivity functions """


def mexican_hat_1D(N: int, A: int, B: int, sigma_exc: float, 
                sigma_inh: float) -> np.ndarray:

    """
    Generate a Mexican hat connectivity pattern for N neurons.

    Parameters
    ----------
    N : int
        Number of neurons.
    A : int
        Amplitude of excitatory connections.
    B : int
        Amplitude of inhibitory connections.
    sigma_exc : float
        Standard deviation of excitatory connections.
    sigma_inh : float
        Standard deviation of inhibitory connections.

    Returns
    -------
    W : np.ndarray
        Connectivity matrix.
    """

    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d_ij = np.abs(i - j)
            W[i, j] = A * np.exp(-d_ij**2 / (2 * sigma_exc**2)
                                 ) - B * np.exp(
                    -d_ij**2 / (2 * sigma_inh**2))

    return W

# Re-define the mexican hat weights function for 2D
def mexican_hat_2D(N: int, A: int, B: int, sigma_exc: int, sigma_inh: int) -> np.ndarray:

    """
    Generate a Mexican hat connectivity pattern for a 2D grid of neurons.

    Parameters
    ----------
    dims : tuple
        Dimensions of the grid (rows, columns).
    A : int
        Amplitude of excitatory connections.
    B : int
        Amplitude of inhibitory connections.
    sigma_exc : int
        Standard deviation of excitatory connections.
    sigma_inh : int
        Standard deviation of inhibitory connections.

    Returns
    -------
    W : np.ndarray
        Connectivity matrix.
    """

    ns = int(np.sqrt(N))
    W = np.zeros((N, N))

    # all neurons positions as all possible combinations of x and y
    ids = [*iterprod(range(ns), range(ns))]

    # for each neuron i
    for i in range(N):

        # for each neuron j
        for j in range(N):

            # skip if i == j
            if i == j:
                continue

            # Calculate Euclidean distance
            d_ij = np.sqrt((ids[i][0] - ids[j][0])**2 + (ids[i][1] - ids[j][1])**2)
            W[i, j] = A * np.exp(-d_ij**2 / (2 * sigma_exc**2)) - B * np.exp(-d_ij**2 / (2 * sigma_inh**2))

    return W



""" Training functions """""


def train_model(genome: dict, N: int, Nj: int, data: np.ndarray=None,
                Model=PCNNetwork, **kwargs):

    """
    Train the model with a specific value of N and Nj

    Parameters
    ----------
    genome : dict
        Genome of the model.
    N : int
        Number of neurons.
    Nj : int
        Number of input neurons.
    data : np.ndarray
        Input data. Default: None
    Model : class
        Model class. Default: mm.RateNetwork6
    **kwargs : dict
        Tmax : int
            Maximum time. Default: 10
        dt : float
            Time step. Default: 0.0007
        sigma : float
            Standard deviation of the input. Default: 0.015
        nb : int
            Number of trials. Default: 1
        display : bool
            Whether to display the progress bar. Default: False
        func : function
            Function to calculate the metric. Default: None
        ignore_zero : bool
            Whether to ignore the zero weights entries. Default: False

    Returns
    -------
    model : Model
        Trained model.
    record : np.ndarray
        Record of the metric.
    """

    # Settings
    genome['N'] = N
    genome['Nj'] = Nj
    nb = kwargs.get('nb', 1)
    display = kwargs.get('display', False)
    logger(f"Training {Model=} with {N=} and {Nj=}, [{nb} trials]")

    if kwargs.get('func', None) is None:
        func = eval_func
        logger(f"Using {func=}")

    # input
    if data is None:
        layer = it.HDLayer(N=Nj, sigma=kwargs.get('sigma', 0.015))
        trajectory = np.arange(0, kwargs.get('Tmax', 10), kwargs.get('dt', 0.0007))
        data = layer.parse_trajectory(trajectory=trajectory)
        logger(f"Using {layer=}")
    else:
        assert data.shape[1] == Nj, f"Input data [{data.shape[1]}] should have the same number of neurons as Nj [{Nj}]"
    
    record1 = []
    record2 = []

    for _ in range(nb):

        # initialization
        model = Model(**genome)

        for t, x in tqdm_enumerate(data, disable=not display):
            model.step(x=x.reshape(-1, 1))

        # calc metric
        record1 += [func(model, axis=0, ignore_zero=kwargs.get('ignore_zero', False))]
        record2 += [func(model, axis=1, ignore_zero=kwargs.get('ignore_zero', False))]

    # average over trials
    record1 = np.array(record1).mean()
    record2 = np.array(record2).mean()
    
    return model, (record1, record2)


def eval_func(weights: np.ndarray, wmax: float, axis: int=1,
              ignore_zero: bool=False, Nj_trg: int=None) -> float:

    """
    evaluate the model by its weight matrix 

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix of the model.
    wmax : float
        Maximum weight.
    axis : int
        Axis to evaluate the model on. Default: 1
    ignore_zero : bool
        Whether to ignore the zero weights entries.
        Default: False
    Nj_trg : int
        Number of input neurons. Default: None

    Returns
    -------
    score : float
        Score associated to the weights.
    """

    # settings
    ni, nj = weights.shape

    # overridde ni if Nj_trg is given
    nj = Nj_trg if Nj_trg is not None else nj

    # places that are legitimitely empty
    nb_empty = 1*((axis==0)*(nj>ni)*(nj - ni) + (axis==1)*(nj<ni)*(ni - nj))

    # sum of weights (along axis) that are above 85% of the max weight
    W_sum = np.where(weights > wmax * 0.85, 1, 0).sum(axis=axis) 

    # error: where there is more than one connection per neuron
    e_over = W_sum[W_sum > 1] -1

    # error: where there is less than one connection per neuron
    nb_under = (W_sum < 1).sum() - nb_empty if not ignore_zero else 0

    # total error: sum of the two errors, kind of
    err = nb_under + ((e_over.sum() - nb_under) > 0)*(e_over.sum() - nb_under) 

    # fraction of neurons that are doing ok
    return 1 - abs((err.sum())/ni)



def cosine_similarity(v: np.ndarray, w: np.ndarray) -> float:

    """
    Calculate the cosine similarity between two vectors.

    Parameters
    ----------
    v : np.ndarray
        First vector.
    w : np.ndarray
        Second vector.

    Returns
    -------
    similarity : float
        Cosine similarity between the two vectors.
    """

    # if v is a matrix, calculate the cosine similarity between each row of v and w
    if len(v.shape) > 1:
        result = v @ w / (np.linalg.norm(v, axis=1, keepdims=True) * np.linalg.norm(w))

    else:
        result = v @ w / (np.linalg.norm(v) * np.linalg.norm(w))

    # if inf or nan, return 0
    return np.nan_to_num(result, nan=0, posinf=0, neginf=0)

