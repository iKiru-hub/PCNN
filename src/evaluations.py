import numpy as np
from scipy.signal import correlate2d, find_peaks
from scipy.ndimage import convolve1d





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

    # places that are legitimately empty
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


def eval_func_2(model: object, trajectory: np.ndarray, target: float=None) -> float:

    """
    Evaluate the model by its activity.
    Two options:
    - if target is given, calculate the error between the target and the activity
    - if target is not given, calculate the sum of the activity

    Parameters
    ----------
    model : object
        The model object.
    trajectory : np.ndarray
        The input trajectory.
    target : float
        The target value. Default: None

    Returns
    -------
    score : float
        Score associated to the activity.
    """


    # set model
    model._plastic = False
    model._bias = 0.8
    model._gain = 5

    # 
    score = 0.

    # evaluate
    for t, x in enumerate(trajectory):

        # step
        model.step(x=x.reshape(-1, 1))   

        # calculate the norm of the population activity
        u_norm = np.linalg.norm(model.u)

        if target is not None:
            error += (target - u_norm)*2
        else:
            score += u_norm

    # return the score normalized by the number of time steps
    return score / len(trajectory)


def eval_field_modality(activation: np.ndarray, indices: bool=False) -> int:

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


def eval_information(model: object, trajectory: np.ndarray, 
                     **kwargs) -> tuple:

    """
    Evaluate the model by its information content.

    Parameters
    ----------
    model : object
        The model object.
    trajectory : np.ndarray
        The input trajectory.

    Returns
    -------
    max_value : float
        Maximum information content.
    diff : float
        Difference between the maximum information content and the mean.
        NB: the lower the better.
    max_wi : float
        Maximum weight sum over i.
    """

    # record the population activity for the whole track
    AT = []
    A = np.empty(len(trajectory))
    model._plastic = False

    for t, x in enumerate(trajectory):
        model.step(x=x.reshape(-1, 1))   
        AT += [tuple(np.around(model.u.flatten(), 1))]
        A[t] = model.u.sum()

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

    # another metric, regarding the weights 
    # square of the difference between the maximum weight sum over i 
    # and the maximum weight
    # max_wi = model.Wff.sum(axis=0).max()

    # number of peaks in the activation map
    nb_peaks = eval_field_modality(activation=A)

    return mean, -std, -nb_peaks


def eval_information_II(model: object, trajectory: np.ndarray, 
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
        A[t] = model.u.sum()

    # number of peaks in the activation map
    nb_peaks, peaks = eval_field_modality(activation=A, indices=True)

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
    mean_peak_position = np.array([np.argmax(A[i::model.N]) for i in range(model.N)]).mean()

    # mean distance between peaks
    var_peaks = mean_peak_position.var()


    # ------------------------------------------------------------ #
    # evaluate the information content

    # record the population activity for the trajectory
    AT = []
    A = np.empty(len(trajectory))

    for t, x in enumerate(trajectory):
        model.step(x=x.reshape(-1, 1))   
        AT += [tuple(np.around(model.u.flatten(), 1))]
        A[t] = model.u.sum()

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

    return mean, -std, -nb_peaks, -var_peak#a_ratio


""" meant to be for `HLayer` """


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


KERNEL = np.ones(20) 
SIZE_PERCENT = 0.02

# def calc_cell_tuning(model, trajectory, track, kernel, threshold, threshold_s, record):
def eval_field_modality_hl_II(activation: np.ndarray) -> tuple:

    """
    Evaluate the modality of the place field:
    - mean of the top 2% of the activation map
    - mean of the rest 98% of the activation map

    Parameters
    ----------
    activation : np.ndarray
        Activation map.

    Returns
    -------
    top2 : float
        Mean of the top 2% of the activation map.
    bottom98 : float
        Mean of the rest 98% of the activation map.
    """

    top2 = 0.
    bottom98 = 0.
    nb_max = int(activation.shape[1] * SIZE_PERCENT)
    xrange = np.arange(activation.shape[1])

    for i in range(activation.shape[0]):

        # make convolution
        z = convolve1d(activation[i], KERNEL)

        # find idx of the maximum
        max_idxs = np.argsort(z)[-nb_max:]

        # find the top 2% and the rest 98%
        top2 += z[max_idxs].mean()
        bottom98 += z[np.setdiff1d(xrange, max_idxs)].mean()

    return top2, bottom98


def eval_information_III_hl(model: object, trajectory: np.ndarray, 
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
    A = np.empty((model.N, len(whole_trajectory)))
    model.set_off()

    for t, x in enumerate(whole_trajectory):
        model.step(x=x.reshape(-1, 1))   
        A[:, t] = model.pcnn.u.copy().flatten()

    # difference between the mean of the top 2% and mean of the rest 98%
    top2, bottom98 = eval_field_modality_hl_II(activation=A)

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

    return mean, -std, top2, -bottom98


