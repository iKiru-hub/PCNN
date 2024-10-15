import numpy as np
import matplotlib.pyplot as plt
from itertools import product as iterprod
from scipy.signal import correlate2d, find_peaks, convolve2d
import json, sys, os
from datetime import datetime
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from tools.utils import logger, tqdm_enumerate
except ModuleNotFoundError:
    warnings.warn('`tools.utils` not found, using fake logger. Some functions may not work')
    class Logger:

        print('Logger not found, using fake logger')

        def __call__(self, msg: str=""):

            self.info(msg=msg)

        def info(self, msg: str):
            print(msg)

        def debug(self, msg: str):
            print(msg)

    logger = Logger()
try:
    import inputools.Trajectory as it
except ModuleNotFoundError:
    warnings.warn('`inputools.Trajectory` not found, some functions may not work')

# suppress RuntimeWarning
np.seterr(divide='ignore', invalid='ignore')

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
            soft_beta: float
                Beta for the softmax function. Default: 10
            beta_clone: float
                Beta for the clone of the weights. Default: 0.5
            low_bounds_nb: int
                Number of neurons to consider for the lower bound.
                Default: 6
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
            sigma_gamma: float
                Standard deviation of the gamma cycles. Default: 5e-6
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
        self._w_kernel = kwargs.get('w_kernel', np.ones((4, 4)))

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
        # self.Wff = np.abs(np.random.normal(0, 0.0001, size=(N, Nj)))
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
        self._sigma_gamma = kwargs.get('sigma_gamma', 5e-6)
        self.tuning = calc_tuning(N=N, K=self._nb_per_cycle, 
                                  b=self._theta_freq)
        self._IS_magnitude = kwargs.get('IS_magnitude', 3)
        self._range = np.arange(self.N).reshape(-1, 1)
        self._is_retuning = kwargs.get('is_retuning', False)

        # softmax
        self._beta = kwargs.get('soft_beta', 15)
        self._beta_clone = kwargs.get('beta_clone', 0.5)
        self._low_bounds_nb = kwargs.get('low_bounds_nb', 6)
        # self.softmax = lambda x: np.exp(self._beta*x) / np.exp(self._beta*x).sum(axis=1,
        #                                                              keepdims=True)

        # internal clock
        self.t = 0.
        self._dt = 1.

        # modulation
        self.DA = 1.
        self._DA_tau = kwargs.get('DA_tau', 100)
        self.DA_block = 0

        # weight update control 
        self.W_old = self.Wff.copy()
        self.W_deriv = np.zeros((self.N, self.Nj))
        self.W_clone = self.Wff.copy()
        self.W_cold_mask = np.zeros((self.N, 1))

        self.W_final = self.Wff.copy()
        self.idx_selective = []

        # reccurent connections
        self.W_rec = np.zeros((N, N))
        self.u_trace = np.zeros((N, 1))
        self._u_trace_decay = kwargs.get('u_trace_decay', 10)

        #
        self.kwargs = kwargs
        self.var1 = None
        self.var2 = None
        self.var3 = []
        self.var4 = None
        self.idx_selective = 0
        self.selective_neurons = []

        self.loaded = False

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
        idx_non_selective = np.where(self.temp < 1.)[0]

        if idx_selective.size > self.idx_selective:
            self.idx_selective = idx_selective.size
        else:
            return

        # increase the theta frequency
        if idx_selective.size > 0:
            self._theta_freq = min((self._theta_freq * (1 + self._theta_freq_increase),
                                   0.05))
            self._theta_freq_increase *= 0.95
        # self._dt *= (1 - self._theta_freq_increase*1.1)

        # calculate the new tuning
        # new_tuning = calc_tuning(N=self.N - len(idx_selective),
        #                          K=self._nb_per_cycle, b=self._theta_freq, 
        #                          sigma=self._sigma_gamma)
        new_tuning = calc_tuning(N=self.N - len(idx_selective), 
                                 K=self._nb_per_cycle, 
                                 b=self._theta_freq)

        # update the tuning
        self.tuning[idx_non_selective] = new_tuning
        self.tuning[idx_selective] = -1

    def _softmax(self, x: np.ndarray, beta: float=1, axis: int=1) -> np.ndarray:

        """
        Parameters
        ----------
        x: np.ndarray
            Input array
        beta: float
            Beta for the softmax function. Default: 1
        axis: int
            Axis along which to calculate the softmax. Default: 1

        Returns
        -------
        x: np.ndarray
            Output array
        """

        return np.exp(beta*x) / np.exp(beta*x).sum(axis=axis, keepdims=True)

    def _softmax_plus(self, x: np.ndarray, threshold: float=0.035) -> np.ndarray:

        """
        A wrapper around the softmax function that set to zero the values
        that are below a certain threshold. Its purpose is to avoid that output 
        is not influenced by the array size.

        Parameters
        ----------
        x: np.ndarray
            Input array
        threshold: float
            Threshold. Default: 1e-1

        Returns
        -------
        x: np.ndarray
            Output array
        """

        # calculate beta for each row
        beta = 1 + self._beta * self.temp

        # apply softmax
        x = np.exp(beta*x) / np.exp(beta*x).sum(axis=1, keepdims=True)

        # set to 0 the values below the threshold
        # x = np.where(x < threshold, 1e-5, x)

        # normalize such that each row sums to 1
        # x = x / x.sum(axis=1, keepdims=True)

        # set nan to 0
        # x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)

        return x

    def _calc_clone(self, low_bounds_nb: int=6, beta: float=1.,
                   threshold: float=0.04) -> np.ndarray:

        """
        Calculate the clone of the weights

        Parameters
        ----------
        low_bounds_nb: int
            Number of neurons to consider for the lower bound.
            Default: 6
        beta: float
            Beta for the softmax function. Default: 1
        threshold: float
            Threshold. Default: 1e-1

        Returns
        -------
        W_clone: np.ndarray
            Clone of the weights

        Notes
        -----
        Two options:
        - calculate a lower bound as the highest nth value of 
          the softmax 
        - use a threshold
        """



        # calculate the softmax of the weights
        # w_soft = self._softmax(x=self.Wff, beta=beta)

        # calculate the lower bound of the weights
        # low_bounds = np.sort(self._softmax(x=self.Wff),
        #                      axis=1)[:, -low_bounds_nb].reshape(-1, 1)



        # threshold = self.Wff.max(axis=1, keepdims=True) * 0.3
        threshold = self.Wff.max(axis=1, keepdims=True) * 0.5 # <---------------------

        w_trim = np.where(self.Wff < threshold, 0., self.Wff)
        # w_trim = self._softmax(x=W, beta=beta)
        # low_bounds = np.sort(self._softmax(x=W),
        #                      axis=1)[:, -low_bounds_nb].reshape(-1, 1)

        # trim the weights
        # w_trim = np.where(w_soft < low_bounds, 0, w_soft)

        # clip the weights
        # threshold = self.Wff.max(axis=1, keepdims=True) * 0.3
        # w_trim = np.where(w_soft < threshold, 0., w_soft)

        # normalize such that each row sums to 1
        w_trim = w_trim / w_trim.sum(axis=1, keepdims=True)

        # set nan to 0
        return np.nan_to_num(w_trim, nan=0, posinf=0, neginf=0)

    def _update(self, x: np.ndarray):

        """
        Update function

        Parameters
        ----------
        x: np.ndarray
            Input from other neurons
        """

        # tweak the input current in order to elicit plasticity even 
        # for weak inputs
        x = x if x.max() < 0.1 else x / x.sum()

        # calculate the softmax of the weights
        # w_soft = self.softmax(self.Wff)
        # w_soft = self._softmax_plus(self.Wff)
        w_soft = self._softmax(x=self.Wff, beta=1 + self._beta * self.temp)

        # update weights | NB: `w_soft` is omitted
        self.Wff += self._lr * self.u * x.T * self.DA * w_soft * \
            (1 - 1*(self.temp == 1.)) * (1 - self.DA_block)

        # self.var1 = w_soft.copy()

        # clip weights
        self.Wff = self.Wff.clip(min=self._wff_min, 
                                 max=self._wff_max)

        # weight decay
        self.Wff += (- self.Wff / self._wff_tau) * (1 - self.temp) 
        # self.Wff = self.Wff - 0.5*self.Wff * (1 - (self.Wff.sum(axis=1, keepdims=True) > 0.2))

        # temperature
        # self.temp = (self.Wff.max(axis=1) / self._wff_max).reshape(-1, 1) 
        self.temp =  self.Wff.sum(axis=1) / self._wff_max
        self.temp = np.where(self.temp > 0.95, 1, self.temp).reshape(-1, 1)

        # calculate the weight cold mask
        self.W_cold_mask = 1 == (self.temp * (1 - \
            np.around(self.W_deriv.sum(axis=1), 3).reshape(-1, 1)))

        # weight derivative 
        self.W_deriv += - self.W_deriv / 10 + np.abs(self.Wff - self.W_old)
        self.W_old = self.Wff.copy()

        # weight clone
        self.W_clone = self._calc_clone(low_bounds_nb=self._low_bounds_nb, 
                                        beta=self._beta_clone)
        # self.W_clone = self.softmax(self.Wff)
        # self.W_clone = np.exp(self.Wff) / np.exp(self.Wff).sum(axis=1, keepdims=True)
        # self.W_clone = self._softmax(x=self.Wff, beta=0.3)
        # self.W_clone = np.where(self.W_clone < 0.04, 0., self.W_clone)

        # normalize such that each row sums to 1
        # self.W_clone = 1.*self.W_clone / self.W_clone.sum(axis=1, keepdims=True)

        # set nan to 0
        # self.W_clone = np.nan_to_num(self.W_clone, nan=0, posinf=0, neginf=0)
        # self.W_clone = (self.W_clone - self.W_clone.min(axis=1, keepdims=True)) / (
        #     self.W_clone.max(axis=1, keepdims=True) - self.W_clone.min(axis=1, keepdims=True))

        # ---| recurrent connections 
        # update trace
        # self.u_trace += (self.u * (self.temp>0.9) - self.u_trace) / self._u_trace_decay
        # self.u_trace = np.where(self.u_trace < 0.001, 0, 0.1)

        # update recurrent weights
        # self.W_rec += self._lr * (self.u_trace @ self.u_trace.T) * (1 - np.eye(self.N)) 

        # self.W_rec *= self.temp

        if np.where(self.temp == 1.)[0].size > self.selective_neurons.__len__():
            self.W_rec = make_wrec2D(weights=self.W_clone.copy(),
                                     kernel=self._w_kernel.copy(),  # this is why I hate python
                                     threshold=0.5).copy()

            self.selective_neurons = np.where(self.temp == 1.)[0]

        # re-tuning
        if self._is_retuning:
            self._re_tuning()

    def _cosine_similarity_mv(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:

        """
        Calculate the cosine similarity between one vector and a matrix

        Parameters
        ----------
        v: np.ndarray
            Vector
        w: np.ndarray
            Matrix

        Returns
        -------
        sim: np.ndarray
            Cosine similarity
        """

        return (v @ w.T) / (np.linalg.norm(v) * np.linalg.norm(w, axis=1))

    def get_wrec(self):

        self.W_rec = make_wrec2D(weights=self.W_final.copy(),
                                        kernel=np.ones((4, 4)),
                                        threshold=0.5).copy()

    def _update_recurrent(self, kernel: np.ndarray=None, threshold: float=0.5):

        """
        Update function
        """

        # max_distance = 0.

        # update list of selective neurons
        # all_selective = np.where(self.temp == 1.)[0]

        # # get the indices of the newly selective neurons
        # new_selective = [i for i in all_selective if i not in self.selective_neurons]

        # if len(new_selective) > 0:

        if kernel is None:
            kernel = np.ones((4, 4))
        if threshold is None:
            threshold = 0.5

        self.W_rec = make_wrec2D(weights=self.W_final.copy(),
                                 kernel=kernel,
                                 threshold=threshold).copy()

            # self.selective_neurons = all_selective


        # if self.t % 100 == 0:

            # W = np.where(self._softmax(self.W_final > 0.3), self.W_final, 0)

            # for all newly selective neurons, update the recurrent weights
            # for i in range(self.N):

                # calculate the euclidean distance wrt all other neurons in input space
                # for j in range(self.N):
                #     if i == j or j not in all_selective: continue

                    # Wi = np.where(self._softmax(self.W_clone[i] > 0., axis=0),
                    #               self.W_clone[i], 0)
                    # Wj = np.where(self._softmax(self.W_clone[j] > 0., axis=0),
                    #               self.W_clone[j], 0)
                    # Wi = np.where(self.W_clone[i] > 0.0, self.W_clone[i], 0)
                    # Wj = np.where(self.W_clone[j] > 0.0, self.W_clone[j], 0)

                    # dist = cosine_similarity(Wi, Wj)
                    # dist = cosine_similarity(Wi*self.u_trace[i], 
                    #                          Wj*self.u_trace[j])

                    # dist = np.linalg.norm(self.Wff[i] - self.Wff[j])

                    # if dist > max_distance:
                    #     self.W_rec[i, j] = dist

            # update the list 
            # self.selective_neurons = np.where(self.temp == 1.)[0]

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
            if not self.loaded:
                self.W_final = self.Wff * (1 - self.W_cold_mask) + \
                    self.W_clone * self.W_cold_mask

            # step
            self.Ix = self.W_final @ x * (1 - self.W_cold_mask) + \
                cosine_similarity(self.W_final.T, x) * self.W_cold_mask
            # self.Ix = cosine_similarity(W, x) * self.W

            self.var1 = self.W_final.copy()

        # calculate synaptic current
        self.var4 = calc_osc(
                N=self.N, t=self.t,
                I=self._range, O=self.tuning, K=self._nb_per_cycle,
                b=self._theta_freq, nb_skip=self._nb_skip,
                sigma=self._sigma_gamma
        )

        self.Is = self._IS_magnitude * (1 - self.temp) * calc_osc(
                N=self.N, t=self.t,
                I=self._range, O=self.tuning, K=self._nb_per_cycle,
                b=self._theta_freq, nb_skip=self._nb_skip,
                sigma=self._sigma_gamma
        )


        # self.Is = self._IS_magnitude * (1 - self.temp) * calc_osc(
        #         N=self.N, t=self.t,
        #         I=self._range, O=self.tuning, K=self._nb_per_cycle,
        #         b=self._theta_freq, nb_skip=self._nb_skip,
        #         sigma=self._sigma_gamma
        # )

        # activation
        self.u = self.activation_func(
            self.u - self.u / self._tau + self.Ix + self.Is
        )

        # trace
        self.u_trace += (self.Ix * (self.temp>0.9) - \
            self.u_trace) / self._u_trace_decay

        # update DA
        ut_block = (self.u * self.temp).max()  # float [0, 1]
        DA_block = ut_block * 1*(ut_block > 0.99) * (
            self.DA < 0.0)  # bool <----------------- !!

        self.DA_block = DA_block
        # self.DA += (1 - DA_block - self.DA) / self._DA_tau
        self.DA += (1 - ut_block - self.DA) / self._DA_tau
        self.var2 = ut_block

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

    def set_da_tau(self, DA_tau: float):

        """
        Set the time constant of the dopamine

        Parameters
        ----------
        DA_tau: float
            Time constant of the dopamine
        """

        self._DA_tau = DA_tau

    def set_u_tau(self, u_tau: float):

        """
        Set the time constant of the internal dynamics

        Parameters
        ----------
        u_tau: float
            Time constant of the internal dynamics
        """

        self._tau = u_tau

    def set_off(self, bias: float=None, gain: float=None):

        """
        Turn off plasticity

        Parameters
        ----------
        bias: float
            Bias of the activation function.
            Default: None
        gain: float
            Gain of the activation function.
            Default: None
        """

        self._plastic = False
        self._IS_magnitude = 0

        self.Wff = np.where(self.Wff < 0.005, 0, self.Wff)
        # loop over all neurons
        for i in range(self.N):
            if self.temp[i] < 1:
                self.Wff[i] *= 0
                continue


        if bias is not None:
            self._bias = bias * np.ones((self.N, 1))
        if gain is not None:
            self._gain = gain

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

    def set_weights(self, Wff: np.ndarray, W_rec: np.ndarray=None):

        """
        Set the weights of the network

        Parameters
        ----------
        Wff: np.ndarray
            Feedforward weights
        W_rec: np.ndarray
            Recurrent weights
        """

        # self.Wff = Wff.copy()
        # self.W_final = Wff.copy()
        # if W_rec is not None:
        #     self.W_rec = W_rec

        self.loaded = True

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
        self.W_cold_mask = np.zeros((self.N, 1))

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


""" PC + PCNN """


class SuperPCNN:

    def __init__(self, layer: object, pcnn: PCNNetwork, interval: int=1):

        """
        SuperPC class

        Parameters
        ----------
        layer: object
            Layer object
        pcnn: PCNNetwork
            PCNNetwork object
        interval: int
            time taken to process each step.
            Default: 1
        """

        self.layer = layer
        self.pcnn = pcnn

        self.N = pcnn.N
        self.n = int(np.sqrt(pcnn.N))
        self._interval = interval

    def __repr__(self):

        return f"SuperPCNN(layer={self.layer}, pcnn={self.pcnn})"

    def step(self, position: np.ndarray=None, max_rate: float=1.) -> np.ndarray:

        """
        Step function

        Parameters
        ----------
        x: np.ndarray
            Input from other neurons

        Returns
        -------
        output: np.ndarray
            Output of the network
        """

        # step
        for _ in range(self._interval):
            y = self.layer.step(position=position, max_rate=1.)
            self.pcnn.step(x=y.reshape(-1, 1))

        return self.pcnn.output * max_rate

    def set_off(self, **kwargs):

        """
        Turn off plasticity

        Parameters
        ----------
        **kwargs: dict
            Keyword arguments
        """

        self.pcnn.set_off(**kwargs)

    def reset(self):

        """
        Reset function
        """

        self.pcnn.reset()



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


def calc_gamma(t: int, O: int, b: int, sigma: float=5e-6) -> np.ndarray:

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
    sigma : float
        Standard deviation of the activity. 
        Default: 1e-6

    Returns
    -------
    gamma : np.ndarray
        Activity as gamma cycles.
    """

    try:
        t = t % (np.pi / b) 
    except ZeroDivisionError:
        logger.debug(f"ZeroDivisionError: {b=}, {t=}")
        raise ZeroDivisionError
    return np.exp(-(np.sin(b*(t - O))-1)**2 / sigma)


def calc_osc(N: int, t: int, I: int, O: int, K: int, 
             b: int, nb_skip: int=1, sigma: float=5e-6) -> np.ndarray:

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
    sigma : float
        Standard deviation of the gamma cycles. 
        Default: 1e-6

    Returns
    -------
    osc : np.ndarray
        Oscillatory stimulation.
    """

    return calc_turn(N=N, t=t, i=I, K=K, b=b, 
                     nb_skip=nb_skip) * calc_gamma(t=t, O=O, b=b, 
                                                   sigma=sigma)


def make_wrec2D(weights: np.ndarray,
                kernel: np.ndarray,
                threshold: float,
                threshold_2: float=1e-3) -> np.ndarray:

    """
    make the recurrent connections 

    Parameters
    ----------
    weights : np.ndarray
        Weights of the network.
    kernel : np.ndarray
        Kernel for the convolution.
        Default: np.ones((4, 4))
    threshold : float
        Threshold for the weights. 
        Default: 0
    threshold_2 : float
        Threshold for the weights. 
        Default: 1e-3

    Returns
    -------
    W : np.ndarray
        Recurrent connections.
    """

    wff_or = weights.copy()
    N = wff_or.shape[0]
    n = int(np.sqrt(wff_or.shape[1]))
    wff = np.empty((N, (n - kernel.shape[0]+1)**2))
    # logger.debug(f"{wff_or.shape=}, {n=}, {wff.shape=} {kernel.shape=}")

    # apply a 1D conv over each row
    for i in range(N):
        try:
            wff[i] = convolve2d(wff_or[i].reshape(n, n),
                                kernel, mode='valid').flatten()
        except ValueError:
            logger.debug(f"ValueError: {wff_or[i].shape=}, {n=} {wff[i].shape=}")
            raise ValueError

    W = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            if wff[i].sum() < threshold_2 or \
                wff[j].sum() < threshold_2:
                continue
            W[i,j] = cosine_similarity(wff[i].flatten(), 
                                       wff[j].flatten())

    W = W * (1 - np.eye(N))
    W = np.where(W > threshold, W, 0)

    return W.copy()


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
        # result = v @ w / (np.linalg.norm(v, axis=1, keepdims=True) * np.linalg.norm(w))
        result = (v.T @ w) / (np.linalg.norm(v.T, axis=1, keepdims=False).reshape(-1, 1) * \
            np.linalg.norm(w.T, axis=1, keepdims=False).reshape(-1, 1))

    else:
        result = v @ w / (np.linalg.norm(v) * np.linalg.norm(w))

    # if inf or nan, return 0
    return np.nan_to_num(result, nan=0, posinf=0, neginf=0)


def save_model(model: object, params: dict, name: str, path=None):

    """
    Save the model weights as a json

    Parameters
    ----------
    model : object
        Model to save.
    params : dict
        Parameters of the model.
    name : str
        Name of the file.
    path : str
        Path to the file. Default: None
    """

    if path is None:
        path = f"cache/{name}.json"

    file = {
        'wff': model.Wff.tolist(),
        'params': params,
        'date': str(datetime.now()),
    }

    with open(path, 'w') as f:
        json.dump(file, f)

    logger(f"Model saved as {path}")


def load_model(name: str, path=None) -> object:

    """
    Load the model weights from a json

    Parameters
    ----------
    name : str
        Name of the file.
    path : str
        Path to the file. Default: None

    Returns
    -------
    model : object
        Model.
    """

    if path is None:
        logger.debug(os.getcwd())
        # os.chdir('lab/PCNN/cache')
        path = f"{name}.json"
        logger.debug(os.listdir())

    with open(path, 'r') as f:
        file = json.load(f)

    # os.chdir('src/')

    model = PCNNetwork(**file['params'])

    if 'wrec' in file:
        model.set_weights(Wff=np.array(file['wff']),
                          W_rec=np.array(file['wrec']))
    else:
        model.set_weights(Wff=np.array(file['wff']))

    logger(f"Model loaded from {path}")

    return model



""" parameters """

genome = {'gain': 6.0,
 'bias': 0.9,
 'lr': 0.9,
 'tau': 40,
 'wff_min': 0.0,
 'wff_max': 0.13,
 'wff_tau': 2000.,
 'soft_beta': 1.7,
 'beta_clone': 0.3,
 'low_bounds_nb': 5,
 'DA_tau': 200,
 'bias_decay': 120,
 'bias_scale': 0.84,
 'IS_magnitude': 40,
 'theta_freq': 0.01,
 'theta_freq_increase': 0.,
 'sigma_gamma': 9.1e-05,
 'nb_per_cycle': 5,
 'nb_skip': 1,
 'dt': 1.,
 'speed': 0.001,
 'N': 20,
 'Nj': 121,
 'is_retuning': True,
 'plastic': True}


def make_stored_model(N: int, Nj: int):

    """
    Make a stored model

    Parameters
    ----------
    N : int
        Number of neurons.
    Nj : int
        Number of input neurons.
    """

    genome['N'] = N
    genome['Nj'] = Nj

    return PCNNetwork(**genome)


def make_stored_super(N: int, Nj: int, sigma: float=0.01,
                      bounds: tuple=(0, 1, 0, 1), **kwargs) -> object:

    """
    Make a stored super model

    Parameters
    ----------
    N : int
        Number of neurons.
    Nj : int
        Number of input neurons.
    sigma : float
        Standard deviation of the input.
        Default: 0.01
    bounds : tuple
        Bounds of the input.
        Default: (0, 1, 0, 1)
    **kwargs : dict
    """

    pcnn = make_stored_model(N=N, Nj=Nj)
    layer = it.PlaceLayer(N=Nj, sigma=0.015, bounds=bounds)

    return SuperPCNN(layer=layer, pcnn=pcnn, **kwargs)


if __name__ == "__main__":

    sp = make_stored_super(N=100, Nj=121)

    print(sp)
