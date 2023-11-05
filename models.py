import numpy as np 
import matplotlib.pyplot as plt 
#from tools.utils import logger


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
        self.tau = kwargs.get('tau', 10)
        self.wr_const = kwargs.get('wr_const', 3)
        self.wff_const = kwargs.get('wff_const', 5)
        self.wff_max = kwargs.get('wff_max', 5)

        # state variables
        self.u = np.ones((N, 1)) * self._Erest
        self.s = np.zeros((N, 1))

        # connectivity
        self.Wff = np.ones((N, Nj)) / Nj * self.wff_const 
        self.Wr = (np.ones((N, N)) - np.eye(N)) * self.wr_const
        self.clip_func = lambda x: (self.wff_max*np.exp(x) - self.wff_max*np.exp(-x)) / (np.exp(x) - np.exp(-x)).clip(min=0)

        # rate function
        self.rate_func = lambda x: 1 / (1 + np.exp(-0.25 * (x + 58)))

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

        # Oja rule
        dWff = self.s * Sj.T - self.Wff @ (Sj * Sj) 

        # update weights
        self.Wff += self._lr * dWff

        self.Wff = self.Wff.clip(min=0)

        # clip weights
        # self.Wff = self.clip_func(self.Wff)

        # normalize weights
        self.Wff = self.Wff / self.Wff.sum(axis=1, 
                                keepdims=True) * self.wff_const

    def step(self, Sj: np.ndarray):

        """
        Step function

        Parameters
        ----------
        Sj: np.ndarray
            Input from other neurons
        """

        # update state variables
        self.u += (self._Erest - self.u) / self.tau + self.Wff @ Sj - self.Wr @ self.s

        # spike generation
        self.s = np.random.binomial(1, self.rate_func(self.u))

        # update weights
        if self._learnable:
            self._update(Sj=Sj)

    def reset(self):

        """
        Reset function
        """

        self.u = np.ones((self.N, 1)) * self._Erest
        self.s = np.zeros((self.N, 1))
        self.Wff = np.ones((self.N, self.Nj)) / self.Nj * self.wff_const

