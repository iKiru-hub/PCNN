import numpy as np 
import matplotlib.pyplot as plt 
from itertools import product as iterprod
try:
    from tools.utils import logger
except ModuleNotFoundError:
    class Logger:

        print('Logger not found, using fake logger')

        def info(self, msg: str):
            print(msg)

        def debug(self, msg: str):
            print(msg)

    logger = Logger()


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
        self._lr_const = kwargs.get('lr', 0.01)
        self._learnable = kwargs.get('learnable', False)

        # parameters
        self.N = N
        self.Nj = Nj
        self.tau = kwargs.get('tau_u', 10)
        self.wr_const = kwargs.get('wr_const', 3)
        self.wff_const = kwargs.get('wff_const', 5)
        self.wff_max = kwargs.get('wff_max', 5)
        self.wff_min = kwargs.get('wff_min', 0.05)
        self.wff_tau_const = kwargs.get('wff_tau', 50)
        self.wff_tau = np.ones((N, 1)) * self.wff_tau_const

        # state variables
        self.u = np.ones((N, 1)) * self._Erest
        self.s = np.zeros((N, 1))

        # conductances
        self.g_ff = np.zeros((N, 1))
        self.g_rec = np.zeros((N, 1))
        self.tau_ff = kwargs.get('tau_ff', 20)
        self.tau_rec = kwargs.get('tau_rec', 75)

        # connectivity
        self.Wff = np.ones((N, Nj)) / Nj * self.wff_const 
        self.Wrec = (np.ones((N, N)) - np.eye(N)) * self.wr_const

        # input synapses
        self._syn_ff = np.zeros((N, Nj))
        self._syn_ff_tau = kwargs.get('syn_ff_tau', 10)
        self._syn_ff_thr = kwargs.get('syn_ff_min', 0.05)
        self._lr = np.ones((N, 1)) * self._lr_const

        # 
        self._wff_beta = kwargs.get('wff_beta', 0.1)
        self._wff_func = lambda x: np.exp(self._wff_beta * x) / np.exp(self._wff_beta * x).sum(axis=0)
        # self._wff_tau_func = lambda x: 1 / (1 + np.exp(-23 * (x -0.7)))
        self._wff_decay_func = lambda x: 1 / (1 + np.exp(-23 * (x -0.7)))

        # rate function
        self.rate_func = lambda x: 1 / (1 + np.exp(-0.3 * (x + 60)))

        # colors for plotting | rainbow
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.N))

        # record
        self.record = np.zeros((self.N, 3))

        logger.info(self.__repr__())

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
        self.Wff += self._lr * dWff * (np.abs(self._syn_ff) > self._syn_ff_thr)

        # normalize weights
        self._normalize()

        # update weight decay components
        decay_speed = self._wff_decay_func(self.Wff.max(axis=1, keepdims=True)/self.wff_max)

        # decay time constant
        self.wff_tau += (decay_speed*2000 + (1-decay_speed)*self.wff_tau_const - self.wff_tau) / 50

        # learning rate
        self._lr += (decay_speed * 1e-5 + (1-decay_speed) * self._lr_const - self._lr) / 100

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

        # update conductances
        self.g_ff += (- self.g_ff + self.Wff @ Sj) / self.tau_ff
        self.g_rec = (self.g_rec - self.g_rec / self.tau_rec + self.Wrec @ self.s).clip(-10, 10)

        # update state variables
        self.u += (self._Erest - self.u + self.g_ff + self.g_rec) / self.tau 

        # spike generation
        self.s = np.random.binomial(1, self.rate_func(self.u))

        # reset
        # self.u = self.u * (1 - self.s) + self._Erest * self.s

        # update weights
        if self._learnable:
            self._update(Sj=Sj)

        # record
        # self.record[:, 0] = self._wff_tau_func(self.Wff.max(axis=1).reshape(-1)/self.wff_max)
        self.record[:, 0] = self.g_ff.reshape(-1)
        self.record[:, 1] = self.g_rec.reshape(-1)
        self.record[:, 2] = self.Wrec[1, :] * self.s.reshape(-1) 

    def set_wrec(self, Wrec: np.ndarray):

        """
        Set recurrent weights

        Parameters
        ----------
        Wrec: np.ndarray
            Recurrent weights
        """

        assert Wrec.shape == (self.N, self.N), 'Wrec must be of shape (N, N)'

        # delete diagonal 
        self.Wrec = self.Wrec * (np.ones((self.N, self.N)) - np.eye(self.N, self.N))

        self.Wrec = Wrec.copy() * self.wr_const

        logger.info(f"Recurrent weights set {Wrec.shape}")

    def reset(self):

        """
        Reset function
        """

        self.u = np.ones((self.N, 1)) * self._Erest
        self.s = np.zeros((self.N, 1))
        self.g_ff = np.zeros((self.N, 1))
        self.g_rec = np.zeros((self.N, 1))
        self.Wff = np.ones((self.N, self.Nj)) / self.Nj * self.wff_const

        self._lr = self._lr_const
        self._wff_tau = self.wff_tau_const

        self.u[0, 0] = -50


class NetworkSimple:
    
    _Erest = -70

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
            plastic: bool
                Whether the network is plastic.
                Default: True
        """

        # define instance id as a string of alphanumeric characters
        self.id = ''.join(np.random.choice(list(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ), 5))
        self.plastic = kwargs.get('plastic', True)

        # parameters
        self.N = N
        self.n = int(np.sqrt(N))
        self.Nj = Nj
        self.tau = kwargs.get('tau_u', 10)
        self._lr_max = kwargs.get('lr_max', 0.01)
        self._lr_min = kwargs.get('lr_min', 1e-5)
        self._lr_tau = kwargs.get('lr_tau', 100)

        # Input connection parameters
        self.wff_const = kwargs.get('wff_const', 5)
        self.wff_max = kwargs.get('wff_max', 5)
        self.wff_min = kwargs.get('wff_min', 0.05)
        self.wff_tau_min = kwargs.get('wff_tau_min', 50)
        self.wff_tau_max = kwargs.get('wff_tau_max', 2000)
        self.wff_tau_tau = kwargs.get('wff_tau_tau', 50)
        self._wff_beta = kwargs.get('wff_beta', 0.1)
        self._wff_func = lambda x: np.exp(self._wff_beta * x) / np.exp(
            self._wff_beta * x).sum(axis=0)
        self._wff_decay_func = lambda x: 1 / (1 + np.exp(-23 * (x -0.7)))

        self.wff_tau = np.ones((N, 1)) * self.wff_tau_min
        self.Wff = np.ones((N, Nj)) / Nj * self.wff_const 

        self._wff_const_beta = kwargs.get('wff_const_beta', 40)
        self._wff_const_alpha = kwargs.get('wff_const_alpha', 0.85)
        self._wff_const_func = lambda x: 1 / (1 + np.exp(
            - self._wff_const_beta * (x - \
                self._wff_const_alpha * self.wff_max)))

        # recurrent connection parameters
        self.wr_const = kwargs.get('wr_const', 3)
        self.Wrec = self._make_mexican_hat(
            dim=kwargs.get('dim', 1),
            A=kwargs.get('A', 1),
            B=kwargs.get('B', 1),
            sigma_exc=kwargs.get('sigma_exc', 1),
            sigma_inh=kwargs.get('sigma_inh', 1)
        ) * self.wr_const

        # state variables
        self.u = np.ones((N, 1)) * self._Erest
        self.s = np.zeros((N, 1))

        # conductances
        self.g_ff = np.zeros((N, 1))
        self.g_rec = np.zeros((N, 1))
        self.tau_ff = kwargs.get('tau_ff', 20)
        self.tau_rec = kwargs.get('tau_rec', 75)

        # input synapses
        self._syn_ff = np.zeros((N, Nj))
        self._syn_ff_tau = kwargs.get('syn_ff_tau', 10)
        self._syn_ff_thr = kwargs.get('syn_ff_thr', 0.05)
        self._lr = np.ones((N, 1)) * self._lr_max

        # rate function
        rate_func_beta = kwargs.get('rate_func_beta', 0.3)
        rate_func_alpha = kwargs.get('rate_func_alpha', 60)
        self.rate_func = lambda x: 1 / (1 + np.exp(-rate_func_beta * (x + rate_func_alpha)))

        # 
        self.kwargs = kwargs

        self.c = 0

    def __repr__(self):

        return f'NetworkSimple(N={self.N}, Nj={self.Nj}) [{self.id}]'

    def _make_mexican_hat(self, dim: int, A: int, B: int, sigma_exc: float,
                          sigma_inh: float):

        """
        Make a Mexican hat connectivity pattern

        Parameters
        ----------
        dim: int
            dimensions of the connectivity pattern, 1 or 2
        **kwargs: dict
            A: int
                Amplitude of excitatory connections
            B: int
                Amplitude of inhibitory connections
            sigma_exc: float
                Standard deviation of excitatory connections
            sigma_inh: float
                Standard deviation of inhibitory connections
        """

        if dim == 1:

            W = np.zeros((self.N, self.N))
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    d_ij = np.abs(i - j)
                    W[i, j] = A * np.exp(-d_ij**2 / (2 * sigma_exc**2)
                                         ) - B * np.exp(
                            -d_ij**2 / (2 * sigma_inh**2))

        elif dim == 2:

            ns = int(np.sqrt(self.N))
            W = np.zeros((self.N, self.N))

            # all neurons positions as all possible combinations of x and y
            ids = [*iterprod(range(ns), range(ns))]

            # for each neuron i
            for i in range(self.N):

                # for each neuron j
                for j in range(self.N):

                    # skip if i == j
                    if i == j:
                        continue

                    # Calculate Euclidean distance
                    d_ij = np.sqrt((ids[i][0] - ids[j][0])**2 + (ids[i][1] - ids[j][1])**2)
                    W[i, j] = A * np.exp(-d_ij**2 / (2 * sigma_exc**2)) - B * np.exp(-d_ij**2 / (2 * sigma_inh**2))

        else:

            raise ValueError(f'dim must be 1 or 2, received {dim}')

        return W

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
        self.Wff += self._lr * dWff * (np.abs(self._syn_ff) > self._syn_ff_thr)

        # normalize weights
        self._normalize()

        # update weight decay components
        decay_speed = self._wff_decay_func(self.Wff.max(axis=1, keepdims=True)/self.wff_max)

        # decay time constant
        self.wff_tau += (decay_speed * self.wff_tau_max + (1-decay_speed)*self.wff_tau_min - self.wff_tau) / self.wff_tau_tau

        # learning rate
        self._lr += (decay_speed * self._lr_min + (1-decay_speed) * self._lr_max - self._lr) / self._lr_tau

    def _normalize(self):

        """
        Normalize the weights 
        """

        # ---| global normalization |---
        # normalize weights across neurons | winner-take-all 
        sofWff = self._wff_func(self.Wff)

        # self.c += 1

        # logger.debug(f"Wff: \n{self.Wff}")
        # logger.debug(f"\n{np.exp(self._wff_beta * self.Wff)}")
        # logger.debug(f"\n{np.exp(self._wff_beta * self.Wff).sum(axis=0)}")
        # logger.debug(f"\n----\n{self._wff_const_func(self.Wff.max(axis=0)).sum()}")
        # logger.debug(f"\n{self.wff_const=}")

        # if self.c == 5:
        #     raise ValueError(f'stop {self.c=}')

        # else:
        #     logger.debug(f"{self.c=}")


        self.Wff = sofWff / sofWff.max(axis=0) * self.Wff

        # ---| local normalization |---

        # update ff weight constant
        # self.wff_const *= max((self._wff_const_func(self.Wff.max(axis=0)).sum() / self.Nj,
        #                        self.wff_min * self.N))
        wff_const = max((
            self.wff_const * self._wff_const_func(self.Wff.max(axis=0)).sum() / self.Nj, 
            self.wff_min * self.N
        )) 

        # normalize weights within a neuron | constant sum
        self.Wff = self.Wff / self.Wff.sum(axis=1, 
                                keepdims=True) * wff_const

        # clip weights | min and max
        self.Wff = self.Wff.clip(min=self.wff_min,
                                 max=self.wff_max)

        # if there are nan values, set all to zero 
        if np.isnan(self.Wff).any() or np.isinf(self.Wff).any():
            self.Wff = np.zeros((self.N, self.Nj))

    def step(self, Sj: np.ndarray):

        """
        Step function

        Parameters
        ----------
        Sj: np.ndarray
            Input from other neurons
        """

        # update conductances
        self.g_ff += (- self.g_ff + self.Wff @ Sj) / self.tau_ff
        self.g_rec = (self.g_rec - self.g_rec / self.tau_rec + self.Wrec @ self.s).clip(-10, 10)

        # update state variables
        self.u += (self._Erest - self.u + self.g_ff + self.g_rec) / self.tau 

        if np.isnan(self.u).any() or np.isinf(self.u).any():
            self.u = -1e3
            

        # spike generation
        self.s = np.random.binomial(1, self.rate_func(self.u))

        # update weights
        if self.plastic:
            self._update(Sj=Sj)

    def set_plastic(self, plastic: bool):

        """
        Set plasticity

        Parameters
        ----------
        plastic: bool
            Whether the network is plastic
        """

        self.plastic = plastic

    def get_kwargs(self):

        """
        Get the kwargs
        """

        return self.kwargs

    def reset(self, bias: bool=False):

        """
        Reset function

        Parameters
        ----------
        bias: bool
            if True, set the first neuron to -50
        """

        self.u = np.ones((self.N, 1)) * self._Erest
        self.s = np.zeros((self.N, 1))
        self.g_ff = np.zeros((self.N, 1))
        self.g_rec = np.zeros((self.N, 1))
        self.Wff = np.ones((self.N, self.Nj)) / self.Nj * self.wff_const * 0

        self._lr = np.ones((self.N, 1)) * self._lr_max
        self._wff_tau = self.wff_tau_min

        if bias:
            self.u[0, 0] = -50




class RateNetwork:
    
    _Erest = -70

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
            plastic: bool
                Whether the network is plastic.
                Default: True
        """

        # define instance id as a string of alphanumeric characters
        self.id = ''.join(np.random.choice(list(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ), 5))
        self.plastic = kwargs.get('plastic', True)

        # parameters
        self.N = N
        self.n = int(np.sqrt(N))
        self.Nj = Nj

        # internal dynamics
        self.tau = kwargs.get('tau_u', 10)
        self._eps = np.ones((N, 1)) * kwargs.get('eps', 0.1)  # noise amplitude
        self._is_eps_scaled = kwargs.get('is_eps_scaled', False)

        # Learning rate
        self._lr_max = kwargs.get('lr_max', 0.01)
        self._lr_min = kwargs.get('lr_min', 1e-5)
        self._lr_tau = kwargs.get('lr_tau', 100)

        # plasticity rule 
        self._rule = kwargs.get('rule', 'oja')
        assert self._rule in ['oja', 'hebb'], 'rule must be oja or hebb'

        # decay check
        self._is_lr_tau_decay = kwargs.get('is_lr_tau_decay', True)
        self._is_g_decay = kwargs.get('is_g_decay', True)
        self._is_syn = kwargs.get('is_syn', True)
        self._zero_wff = kwargs.get('zero_wff', False)

        # Input connection parameters
        self.wff_const = kwargs.get('wff_const', 5)
        self.wff_max = kwargs.get('wff_max', 5)
        self.wff_min = kwargs.get('wff_min', 0.05)
        self.wff_tau_min = kwargs.get('wff_tau_min', 50)
        self.wff_tau_max = kwargs.get('wff_tau_max', 2000)
        self.wff_tau_tau = kwargs.get('wff_tau_tau', 50)
        self._wff_beta = kwargs.get('wff_beta', 0.1)
        self._wff_func = lambda x: np.exp(self._wff_beta * x) / np.exp(
            self._wff_beta * x).sum(axis=0)

        self._wff_decay_func = lambda x: 1 / (1 + np.exp(-23 * (x -0.7)))

        self.wff_tau = np.ones((N, 1)) * self.wff_tau_min

        if self._zero_wff:
            self.Wff = np.zeros((N, Nj))
        else:
            self.Wff = np.ones((N, Nj)) / Nj * self.wff_const 

        self._wff_const_beta = kwargs.get('wff_const_beta', 40)
        self._wff_const_alpha = kwargs.get('wff_const_alpha', 0.85)
        self._wff_const_func = lambda x: 1 / (1 + np.exp(
            - self._wff_const_beta * (x - \
                self._wff_const_alpha * self.wff_max)))

        # recurrent connection parameters
        self.wr_const = kwargs.get('wr_const', 3)
        self.Wrec = self._make_mexican_hat(
            dim=kwargs.get('dim', 1),
            A=kwargs.get('A', 1),
            B=kwargs.get('B', 1),
            sigma_exc=kwargs.get('sigma_exc', 1),
            sigma_inh=kwargs.get('sigma_inh', 1)
        ) * self.wr_const

        # state variables
        self.u = np.ones((N, 1)) * self._Erest
        self.s = np.zeros((N, 1))

        # conductances
        self.g_ff = np.zeros((N, 1))
        self.g_rec = np.zeros((N, 1))
        self.tau_ff = kwargs.get('tau_ff', 20)
        self.tau_rec = kwargs.get('tau_rec', 75)

        # input synapses
        self._syn_ff = np.zeros((N, Nj))
        self._syn_ff_tau = kwargs.get('syn_ff_tau', 10)
        self._syn_ff_thr = kwargs.get('syn_ff_thr', 0.05)
        self._lr = np.ones((N, 1)) * self._lr_max

        # rate function
        rate_func_beta = kwargs.get('rate_func_beta', 0.3)
        rate_func_alpha = kwargs.get('rate_func_alpha', 60)
        self.rate_func = lambda x: 1 / (1 + np.exp(-rate_func_beta * (x + rate_func_alpha)))

        # 
        self.kwargs = kwargs

        self.c = 0

    def __repr__(self):

        return f'RateNetwork(N={self.N}, Nj={self.Nj}) [{self.id}]'

    def _make_mexican_hat(self, dim: int, A: int, B: int, sigma_exc: float,
                          sigma_inh: float):

        """
        Make a Mexican hat connectivity pattern

        Parameters
        ----------
        dim: int
            dimensions of the connectivity pattern, 1 or 2
        **kwargs: dict
            A: int
                Amplitude of excitatory connections
            B: int
                Amplitude of inhibitory connections
            sigma_exc: float
                Standard deviation of excitatory connections
            sigma_inh: float
                Standard deviation of inhibitory connections
        """

        if dim == 1:

            W = np.zeros((self.N, self.N))
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    d_ij = np.abs(i - j)
                    W[i, j] = A * np.exp(-d_ij**2 / (2 * sigma_exc**2)
                                         ) - B * np.exp(
                            -d_ij**2 / (2 * sigma_inh**2))

        elif dim == 2:

            ns = int(np.sqrt(self.N))
            W = np.zeros((self.N, self.N))

            # all neurons positions as all possible combinations of x and y
            ids = [*iterprod(range(ns), range(ns))]

            # for each neuron i
            for i in range(self.N):

                # for each neuron j
                for j in range(self.N):

                    # skip if i == j
                    if i == j:
                        continue

                    # Calculate Euclidean distance
                    d_ij = np.sqrt((ids[i][0] - ids[j][0])**2 + (ids[i][1] - ids[j][1])**2)
                    W[i, j] = A * np.exp(-d_ij**2 / (2 * sigma_exc**2)) - B * np.exp(-d_ij**2 / (2 * sigma_inh**2))

        else:

            raise ValueError(f'dim must be 1 or 2, received {dim}')

        return W

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
        if self._rule == 'oja':
            dWff = self.s * Sj.T - self.Wff @ (Sj * Sj) 
        elif self._rule == 'hebb':
            dWff = self.s * Sj.T

        if self._is_syn:

            # FF synapses | local accumulator
            self._syn_ff += (- self._syn_ff + dWff) / self._syn_ff_tau

            # update weights
            self.Wff += self._lr * dWff * (np.abs(self._syn_ff) > self._syn_ff_thr)

        else:

            # update weights
            self.Wff += self._lr * dWff

        # normalize weights
        self._normalize()

        # update weight decay components
        decay_speed = self._wff_decay_func(self.Wff.max(axis=1, keepdims=True)/self.wff_max)

        if self._is_lr_tau_decay:

            # decay time constant
            self.wff_tau += (decay_speed * self.wff_tau_max + \
                (1-decay_speed)*self.wff_tau_min - self.wff_tau) / self.wff_tau_tau

            # learning rate
            self._lr += (decay_speed * self._lr_min + (1-decay_speed) * self._lr_max \
                - self._lr) / self._lr_tau

    def _normalize(self):

        """
        Normalize the weights 
        """

        # ---| global normalization |---
        # normalize weights across neurons | winner-take-all 
        sofWff = self._wff_func(self.Wff)

        # self.c += 1

        # logger.debug(f"Wff: \n{self.Wff}")
        # logger.debug(f"\n{np.exp(self._wff_beta * self.Wff)}")
        # logger.debug(f"\n{np.exp(self._wff_beta * self.Wff).sum(axis=0)}")
        # logger.debug(f"\n----\n{self._wff_const_func(self.Wff.max(axis=0)).sum()}")
        # logger.debug(f"\n{self.wff_const=}")

        # if self.c == 5:
        #     raise ValueError(f'stop {self.c=}')

        # else:
        #     logger.debug(f"{self.c=}")


        self.Wff = sofWff / sofWff.max(axis=0) * self.Wff

        # ---| local normalization |---

        # update ff weight constant
        # self.wff_const *= max((self._wff_const_func(self.Wff.max(axis=0)).sum() / self.Nj,
        #                        self.wff_min * self.N))
        wff_const = max((
            self.wff_const * self._wff_const_func(self.Wff.max(axis=0)).sum() / self.Nj, 
            self.wff_min * self.N
        )) 

        # normalize weights within a neuron | constant sum
        self.Wff = self.Wff / self.Wff.sum(axis=1, 
                                keepdims=True) * wff_const

        # clip weights | min and max
        self.Wff = self.Wff.clip(min=self.wff_min,
                                 max=self.wff_max)

        # if there are nan values, set all to zero 
        if np.isnan(self.Wff).any() or np.isinf(self.Wff).any():
            self.Wff = np.zeros((self.N, self.Nj))

    def step(self, Sj: np.ndarray):

        """
        Step function

        Parameters
        ----------
        Sj: np.ndarray
            Input from other neurons
        """

        if self._is_g_decay:

            # update conductances
            self.g_ff += (- self.g_ff + self.Wff @ Sj) / self.tau_ff
            self.g_rec = (self.g_rec - self.g_rec / self.tau_rec + self.Wrec @ self.s).clip(-10, 10)

        else:

            # update conductances
            self.g_ff = self.Wff @ Sj
            self.g_rec = (self.Wrec @ self.s).clip(-10, 10)

        # update state variables
        self.u += (self._Erest - self.u + self.g_ff + self.g_rec) / self.tau + \
            self._eps * np.random.randn(self.N, 1)

        if np.isnan(self.u).any() or np.isinf(self.u).any():
            self.u = -1e3

        # spike generation
        self.s = self.rate_func(self.u)

        # eps 
        if self._is_eps_scaled:
            self._eps = self._eps * (1 - self.Wff.max(axis=1) / self.wff_max).reshape(-1, 1)

        # update weights
        if self.plastic:
            self._update(Sj=Sj)

    def set_plastic(self, plastic: bool):

        """
        Set plasticity

        Parameters
        ----------
        plastic: bool
            Whether the network is plastic
        """

        self.plastic = plastic

    def get_kwargs(self):

        """
        Get the kwargs
        """

        return self.kwargs

    def reset(self, bias: bool=False):

        """
        Reset function

        Parameters
        ----------
        bias: bool
            if True, set the first neuron to -50
        """

        self.u = np.ones((self.N, 1)) * self._Erest
        self.s = np.zeros((self.N, 1))
        self.g_ff = np.zeros((self.N, 1))
        self.g_rec = np.zeros((self.N, 1))

        if self._zero_wff:
            self.Wff = np.zeros((self.N, self.Nj))
        else:
            self.Wff = np.ones((self.N, self.Nj)) / self.Nj * self.wff_const 

        self._lr = np.ones((self.N, 1)) * self._lr_max
        self._wff_tau = self.wff_tau_min

        if bias:
            self.u[0, 0] = -50



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
