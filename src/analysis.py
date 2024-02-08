import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt






def mutual_information(joint_xy, marginal_x, marginal_y) -> float:

    """
    Calculate mutual information between two random variables.

    Parameters
    ----------
    joint_xy : np.ndarray
        Joint probability distribution of two random variables.
    marginal_x : np.ndarray
        Marginal probability distribution of first random variable.
    marginal_y : np.ndarray
        Marginal probability distribution of second random variable.

    Returns
    -------
    MI : float
        Mutual information between two random variables.
    """

    mi = 0

    # Loop over all entries in joint distribution
    for i in range(len(marginal_x)):
        for j in range(len(marginal_y)):

            # Calculate MI for this entry
            if joint_xy[i, j] > 0:  # avoid log(0)
                mi += joint_xy[i, j] * np.log2(joint_xy[i, j] / (marginal_x[i] * marginal_y[j]))

    return mi


def calc_MI(X: np.ndarray, Y: np.ndarray, bins: int=10) -> float:

    """
    Calculate mutual information between two random variables.

    Parameters
    ----------
    X : np.ndarray
        First random variable, shape (Nx, T).
    Y : np.ndarray
        Second random variable, shape (Ny, T).
    bins : int, optional
        Number of bins for discretization, by default 10.

    Returns
    -------
    MI : float
        Mutual information between two random variables,
        shape (T,).
    """

    Nx, T = X.shape
    Ny, T = Y.shape

    # Discretization: Convert continuous data to discrete bins
    # X_discrete = np.digitize(X, np.linspace(0, 1, bins))
    # Y_discrete = np.digitize(Y, np.linspace(0, 1, bins))

    MI_t = np.zeros(T)

    # Loop over time steps
    for t in range(T):

        # joint_xy_t = np.zeros((bins, bins))
        joint_xy_t = np.zeros((Nx, Ny))

        # Loop over entries i in X
        for i in range(Nx):

            # Loop over entries j in X
            for j in range(Ny):

                # Update joint distribution
                # joint_xy_t[X_discrete[i, t]-1, Y_discrete[j, t]-1] += 1
                joint_xy_t[i, j] += 1

        # Normalize the joint distribution for this time step
        joint_xy_t /= joint_xy_t.sum()

        # Marginal distributions for this time step
        marginal_x_t = joint_xy_t.sum(axis=1)
        marginal_y_t = joint_xy_t.sum(axis=0)

        # Calculate MI for this time step
        MI_t[t] = mutual_information(joint_xy=joint_xy_t,
                                     marginal_x=marginal_x_t,
                                     marginal_y=marginal_y_t)

    return MI_t



def plot_MI(X: np.ndarray, Y: np.ndarray, bins: int=10) -> None:

    """
    Plot mutual information between two random variables.

    Parameters
    ----------
    X : np.ndarray
        First random variable, shape (Nx, T).
    Y : np.ndarray
        Second random variable, shape (Ny, T).
    bins : int, optional
        Number of bins for discretization, by default 10.
    """

    # calculate MI
    MI = calc_MI(X=X, Y=Y, bins=bins)

    # reshape as a square matrix
    size = int(np.sqrt(len(MI)))
    MI = MI.reshape((size, size))
    print(MI)

    plt.figure()
    plt.imshow(MI, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Mutual Information')
    plt.show()



if __name__ == '__main__':

    # Generate two random variables
    T = 11**2

    # correlated random variables
    c = 1
    Ni = 10
    Nj = 10

    # Generate random data for X and Y
    np.random.seed(0)
    X = np.random.rand(Ni, T)  # Random data for r
    Y = np.random.rand(Nj, T)  # Random data for Y

    # Introduce correlation between X and Y
    Y = c * X + (1 - c) * Y


    # Plot mutual information
    plot_MI(X=X, Y=Y, bins=100)
