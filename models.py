import numpy as np 
import matplotlib.pyplot as plt 
#from tools.utils import logger



#---------------------------------
# ---| Network implementation |---
#---------------------------------


class Network:

    def __init__(self, N: int=1, Nj: int=1, **kwargs):

        """
        Neuron class

        Parameters
        ----------
        N: int
            Number of neurons
        Nj: int
            Number of input neurons
        **kwargs: dict
            Erest: float
                Resting potential
            lr: float
                Learning rate
            learnable: bool
                Whether the network is learnable
            tau: float
                Time constant
            wff_const: float
                Constant for feedforward weights
        """

        # hyperparameters
        self._Erest = kwargs.get('Erest', -70)
        self._lr = kwargs.get('lr', 0.01)
        self._learnable = kwargs.get('learnable', False)

        # parameters
        self.N = N
        self.Nj = Nj
        self.tau = kwargs.get('tau_u', 10)
        self.wr_const = kwargs.get('wr_const', 3)
        self.wff_const = kwargs.get('wff_const', 5)
        self.wff_max = kwargs.get('wff_max', 5)
        self.wff_min = kwargs.get('wff_min', 0.05)
        self.wff_tau_const = kwargs.get('wff_tau', 100)
        self.wff_tau = np.ones((N, 1)) * self.wff_tau_const

        # state variables
        self.u = np.ones((N, 1)) * self._Erest
        self.s = np.zeros((N, 1))

        # connectivity
        self.Wff = np.ones((N, Nj)) / Nj * self.wff_const 
        self.Wr = (np.ones((N, N)) - np.eye(N)) * self.wr_const

        # input synapses
        self._syn_ff = np.zeros((N, Nj))
        self._syn_ff_tau = kwargs.get('syn_ff_tau', 10)
        self._syn_ff_min = kwargs.get('syn_ff_min', 0.05)

        # 
        self._wff_beta = kwargs.get('wff_beta', 0.1)
        self._wff_func = lambda x: np.exp(self._wff_beta * x) / np.exp(self._wff_beta * x).sum(axis=0)
        self._wff_tau_func = lambda x: 1 / (1 + np.exp(-23 * (x -0.7)))

        # rate function
        self.rate_func = lambda x: 1 / (1 + np.exp(-0.25 * (x + 58)))

        # record
        self.record = np.zeros((self.N, 3))

        print(self.__repr__())

    def __repr__(self):

        return f'Network(N={self.N}, Nj={self.Nj})'

    def _update(self, Sj: np.ndarray):

        """
        Update function

        Parameters
        ----------
        Sj: np.ndarray
            Input from other neurons
        """

        # weight decay
        self.Wff += (self.wff_const / self.Nj - self.Wff) / self.wff_tau

        # Oja rule
        dWff = self.s * Sj.T - self.Wff @ (Sj * Sj) 

        # FF synapses | local accumulator
        self._syn_ff += (- self._syn_ff + dWff) / self._syn_ff_tau

        # update weights
        self.Wff += self._lr * dWff * (np.abs(self._syn_ff) > self._syn_ff_min)

        # normalize weights
        self._normalize()

        # update weight decay time constant
        dtau = self._wff_tau_func(self.Wff.max(axis=1, keepdims=True)/self.wff_max)
        self.wff_tau += (dtau*2000 + (1-dtau)*self.wff_tau_const - self.wff_tau) / 50

    def _normalize(self):

        """
        Normalize the weights 
        """

        # normalize weights across neurons | winner-take-all
        sofWff = self._wff_func(self.Wff)
        self.Wff = sofWff / sofWff.max(axis=0) * self.Wff

        # normalize weights within a neuron | constant sum
        self.Wff = self.Wff / self.Wff.sum(axis=1, 
                                keepdims=True) * self.wff_const

        # clip weights | min and max
        self.Wff = self.Wff.clip(min=self.wff_min,
                                 max=self.wff_max)

    def step(self, Sj: np.ndarray):

        """
        Step function

        Parameters
        ----------
        Sj: np.ndarray
            Input from other neurons
        """

        exc = self.Wff @ Sj
        inh = self.Wr @ self.s

        # update state variables
        self.u += (self._Erest - self.u + exc - inh) / self.tau 

        # spike generation
        self.s = np.random.binomial(1, self.rate_func(self.u))

        # update weights
        if self._learnable:
            self._update(Sj=Sj)

        # record
        self.record[:, 0] = self._wff_tau_func(self.Wff.max(axis=1).reshape(-1)/self.wff_max)
        # self.record[:, 0] = self.Wff.max(axis=1).reshape(-1) / self.wff_max 

    def reset(self):

        """
        Reset function
        """

        self.u = np.ones((self.N, 1)) * self._Erest
        self.s = np.zeros((self.N, 1))
        self.Wff = np.ones((self.N, self.Nj)) / self.Nj * self.wff_const



#--------------------------------
# ---| Connectivity patterns |---
#--------------------------------

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
            d_ij = np.abs(i - j)
            W[i, j] = A * np.exp(-d_ij**2 / (2 * sigma_exc**2)
                                 ) - B * np.exp(
                    -d_ij**2 / (2 * sigma_inh**2))

    return W

# Re-define the mexican hat weights function for 2D
def mexican_hat_2D(dims: tuple, A: int, B: int, sigma_exc: int, sigma_inh: int) -> np.ndarray:

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

    rows, cols = dims
    N = rows * cols
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # Convert index to 2D coordinates
            x_i, y_i = i // cols, i % cols
            x_j, y_j = j // cols, j % cols
            # Calculate Euclidean distance
            d_ij = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            W[i, j] = A * np.exp(-d_ij**2 / (2 * sigma_exc**2)) - B * np.exp(-d_ij**2 / (2 * sigma_inh**2))

    return W
