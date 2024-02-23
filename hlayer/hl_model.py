import sys
import os
import numpy as np
from scipy.signal import correlate2d, find_peaks

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import models as mm
from tools.utils import logger
import inputools.Trajectory as it





""" Parameters """

PCNN_PARAMS = {'gain': 5,
 'bias': 0.9,
 'lr': 0.6,
 'tau': 10,
 'wff_min': 0.0,
 'wff_max': 0.5,
 'wff_tau': 1500,
 'soft_beta': 1.7,
 'beta_clone': 0.3,
 'low_bounds_nb': 5,
 'DA_tau': 3,
 'bias_decay': 120,
 'bias_scale': 0.84,
 'IS_magnitude': 50,
 'theta_freq': 0.005,
 'theta_freq_increase': 0.075,
 'sigma_gamma': 9.1e-05,
 'nb_per_cycle': 5,
 'nb_skip': 2,
 'dt': 0.001,
 'speed': 0.005,
 'sigma_pc': 0.00251,
 'sigma_bc': 0.005,
 'N': 60,
 'Nj': 121,
 'is_retuning': True,
 'plastic': True,
 'u_trace_decay': 2}




""" Model """

class ModelHL:

    """
    this model is composed by an evolved input layer, which has the purpose of 
    opportunely process the input position, and a downstream layer, which is
    a PCNN with online formed place cells.
    """

    def __init__(self, W: np.ndarray, activation: str,
                 pcnn_params: dict=PCNN_PARAMS):

        """
        Parameters
        ----------
        W : np.ndarray
            the weight matrix of the input layer
        pcnn_params : dict
            the parameters for the PCNN layer.
            Default is PCNN_PARAMS
        """

        # input layer parameters
        self.W = W
        self.activation = self._make_activation_function(activation)
        self.dim_input = W.shape[1]
        self.dim_output = W.shape[0]

        # PCNN layer parameters
        pcnn_params['Nj'] = self.dim_output
        self.pcnn = mm.PCNNetwork(**pcnn_params)

    def __repr__(self):

        return f"ModelHL(dim 1={self.dim_input}, dim 2={self.dim_output}," + \
               f" PCNN={self.pcnn})"

    def step(self, x: np.ndarray):

        """
        Parameters
        ----------
        x : np.ndarray
            the input position
        """

        # input layer
        y = self.activation(self.W @ x)

        # PCNN layer
        self.pcnn.step(x=y)

    def _make_activation_function(self, kind: str):

        """
        Parameters
        ----------
        kind : str
            the kind of activation function

        Returns
        -------
        callable
            the activation function
        """

        if kind == 'relu':
            return lambda x: np.maximum(0, x)
        elif kind == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif kind == 'positive_tanh':
            return lambda x: (np.tanh(x) + 1) / 2
        else:
            return lambda x: x

    @property
    def output(self):

        """
        return the output layer
        """

        return self.pcnn.out

    def set_off(self):

        """
        set the PCNN layer off
        """

        self.pcnn.set_off()

    def reset(self):

        """
        reset the PCNN layer
        """

        self.pcnn.reset()



""" Evaluation """



def eval_field_modality_hl(activation: np.ndarray, indices: bool=False) -> int:

    """
    calculate how many peaks are in the activation map

    Parameters
    ----------
    activation : np.ndarray
        Activation map.
    indices : bool
        Whether to return the indices of the peaks. 
        Default: False

    Returns
    -------
    nb_peaks : int
        Number of peaks.
    peaks_indices : np.ndarray
        Indices of the peaks.
    """

    # reshape the activation map
    n = int(np.sqrt(activation.shape[0]))
    activation = activation.reshape(n, n)

    # Correlate the distribution with itself
    autocorrelation = correlate2d(activation, activation, mode='full')

    # Find peaks in the autocorrelation map
    peaks_indices = find_peaks(autocorrelation[autocorrelation.shape[0]//2], 
                               height=autocorrelation.max()/3)[0]

    if indices:
        return len(peaks_indices), peaks_indices
    return len(peaks_indices)



def eval_information_II_hl(model: object, trajectory: np.ndarray, 
                           whole_trajectory: np.ndarray, **kwargs) -> tuple:

    """
    Evaluate the model by its information content on the trajectory
    and its place field on the whole trajectory.:
    - mean information content
    - standard deviation of the information content
    - number of peaks in the activation map

    Parameters
    ----------
    model : object
        The model object.
    trajectory : np.ndarray
        The input trajectory.
    whole_trajectory : np.ndarray
        The whole input trajectory.

    Returns
    -------
    mean_IT : float
        Mean information content.
    std_IT : float
        Standard deviation of the information content.
    nb_peaks : int
        Number of peaks in the activation map.
    """

    # ------------------------------------------------------------ #
    # evaluate the shape of the place field

    # record the population activity for the whole track
    A = np.empty(len(whole_trajectory))
    model.set_off()

    for t, x in enumerate(whole_trajectory):
        model.step(x=x.reshape(-1, 1))   
        A[t] = model.pcnn.u.sum()

    # number of peaks in the activation map
    nb_peaks, peaks = eval_field_modality_hl(activation=A, indices=True)

    if len(peaks) > 0:
        
        lim_dx = max((peaks[0] - 10, 0))
        lim_sx = min((peaks[0] + 10, len(A)))

        # get the area (+- 20 units) around the main peak
        on_area = A[lim_dx:lim_sx]
        off_area = np.concatenate((A[:lim_dx], A[lim_sx:]))

        # calculate the ratio of the area around the main peak
        a_ratio = on_area.sum() / off_area.sum()

    else:
        a_ratio = 0

    # peaks of different neurons should be as far as possible
    # across the network
    mean_peak_position = np.array([np.argmax(A[i::model.pcnn.N]) \
        for i in range(model.pcnn.N)]).mean()

    # mean distance between peaks
    var_peaks = mean_peak_position.var()


    # ------------------------------------------------------------ #
    # evaluate the information content

    # record the population activity for the trajectory
    AT = []
    A = np.empty(len(trajectory))

    for t, x in enumerate(trajectory):
        model.step(x=x.reshape(-1, 1))   
        AT += [tuple(np.around(model.pcnn.u.flatten(), 1))]
        A[t] = model.pcnn.u.sum()

    # assign a probability (frequency) for each unique pattern
    AP = {}
    for a in AT:
        if a in AP.keys():
            AP[a] += 1
            continue
        AP[a] = 1

    for k, v in AP.items():
        AP[k] = v / AT.__len__()

    # compute the information content at each point in the track 
    IT = np.empty(len(trajectory))
    for t, a in enumerate(AT):
        IT[t] = - np.log2(AP[a])

    # calculate and return the information content-related metrics
    mean = np.clip(IT, -5, 5).mean()
    std = IT.std()

    return mean, -std, -nb_peaks, -var_peaks#a_ratio


