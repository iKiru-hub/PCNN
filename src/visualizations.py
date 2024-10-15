import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.ndimage import convolve1d

from IPython.display import clear_output 

import sys, os
if os.path.exists(os.path.expanduser('~/Research/lab/PCNN/src')):
    sys.path.append(os.path.expanduser('~/Research/lab/PCNN/src'))
elif os.path.exists(os.path.expanduser('~/lab/PCNN/src')):
    sys.path.append(os.path.expanduser('~/lab/PCNN/src'))
else:
    raise FileNotFoundError("Path to the package not found.")

import utils as utils
from tqdm import tqdm


def clf():
    clear_output(wait=True)


def plot_graph_colored_1D(W: np.ndarray):

    """
    Plot a graph of the neurons with edges colored according to their weight.

    Parameters
    ----------
    W : np.ndarray
        The weight matrix of the network.
    """

    # Create a graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(len(W)))

    # Add weighted edges
    for i in range(len(W)):
        for j in range(i+1, len(W)):  # Only upper triangle needed for undirected graph
            weight = W[i, j]
            G.add_edge(i, j, weight=weight)

    # Position nodes in a circle
    pos = nx.circular_layout(G)

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_color='k', node_size=50)

    # Draw the edges with a color map
    edges = G.edges(data=True)
    colors = [d['weight'] for (u, v, d) in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors,
                           edge_cmap=plt.cm.RdYlGn, edge_vmin=-1, edge_vmax=1)

    # Show the plot
    plt.title('Network Graph with Weight-Colored Connections')
    plt.axis('off')
    plt.show()

# Define a function to plot the graph with a 2D structure
def plot_graph_colored_2D(W: np.ndarray, grid_dimensions, threshold=0.):

    """
    Plot a graph of the neurons in a 2D grid with edges colored according to their weight.

    Parameters
    ----------
    W : np.ndarray
        The weight matrix of the network.
    grid_dimensions : tuple
        The dimensions of the grid.
    threshold : float
        The threshold for the weight to be considered significant.
    """

    rows, cols = grid_dimensions
    N = rows * cols

    # Create a graph
    G = nx.Graph()

    # Add nodes with 2D positions
    for i in range(N):
        x, y = i // cols, i % cols  # Convert index to 2D position
        G.add_node(i, pos=(x, y))

    # Add weighted edges considering the threshold
    for i in range(N):
        for j in range(i+1, N):  # Only upper triangle needed for undirected graph
            weight = W[i, j]
            if abs(weight) > threshold:  # Only consider significant weights
                G.add_edge(i, j, weight=weight)

    # Get positions from node data
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_color='k', node_size=100)

    # Draw the edges with a color map
    edges = G.edges(data=True)
    colors = [d['weight'] for (u, v, d) in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors,
                           edge_cmap=plt.cm.RdYlGn, edge_vmin=-1, edge_vmax=1)

    # Show the plot with color bar
    plt.title('2D Grid Graph with Weight-Colored Connections')
    plt.axis('equal')  # Ensure x and y scales are equal to avoid distortion
    plt.axis('off')  # Turn off the axis
    plt.show()


def plot_weight_matrix(W: np.ndarray):

    """
    Plot the weight matrix as a heatmap.

    Parameters
    ----------
    W : np.ndarray
        The weight matrix of the network.
    """

    plt.imshow(W, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.title('Weight Matrix')
    plt.colorbar()
    plt.axis('off')
    plt.show()


def plotting(model: object, X: np.ndarray, t: int, record: np.ndarray, 
             Ix: np.ndarray, colors: list, **kwargs):

    """
    Plot the input, weights, weight matrix, and u - DA.

    Parameters
    ----------
    model : object
        The model object.
    X : np.ndarray
        The input data.
    t : int
        The current time step.
    record : np.ndarray
        The record of the network activity.
    Ix : np.ndarray
        The input current.
    colors : list
        The colors for the neurons.
    **kwargs : dict
        animaker : object
            The animation object.
            Default: None
        subtitle_2 : str
            The subtitle for the second subplot.
            Default: None
        subtitle_3 : str
            The subtitle for the third subplot.
        winsize_1 : int
            The window size for the first subplot.
            Default: 20
    """

    Nj = model.Nj
    N = model.N

    winsize_1 = kwargs.get('winsize_1', 20)
    animaker = kwargs.get('animaker', None)
    is_anim = bool(animaker)

    if kwargs.get('subtitle_2', None) is None:
        subtitle_2 = f"Weight matrix"
    else:
        subtitle_2 = kwargs.get('subtitle_2', None)
    if kwargs.get('subtitle_3', None) is None:
        subtitle_3 = f"u - DA"
    else:
        subtitle_3 = kwargs.get('subtitle_3', None)

    clf()
    fig = plt.figure(figsize=(20, 6))
    plt.tight_layout()

    ### input
    plt.subplot(221)
    plt.imshow(X.T[:, t-winsize_1:t], cmap="Greys")
    plt.xticks(())
    plt.title(f"{t=}")
    plt.xlabel("ms")
    plt.grid()

    ### weights
    plt.subplot(232)
    plt.axvline(0, color='black', alpha=0.3)
    plt.axvline(model._wff_max, color='red', alpha=0.9)
    for i in range(N):
        # plt.plot(np.flip(model.Wff[i], axis=0), range(Nj), '-', color=colors[i], alpha=0.3)
        # plt.plot(np.flip(model.Wff[i], axis=0), range(Nj), 'o', color=colors[i], alpha=0.5)
        plt.plot(np.flip(model.W_clone[i], axis=0), range(Nj), '-', color=colors[i], alpha=0.3)
        plt.plot(np.flip(model.W_clone[i], axis=0), range(Nj), 'o', color=colors[i], alpha=0.5)
        plt.title(f"Weights")
    plt.yticks(())
    plt.xlim((-0.1, 3))
    plt.grid()

    ### weight matrix
    plt.subplot(233)
    plt.imshow(model.Wff.T, cmap="plasma")
    plt.title(subtitle_2)
    plt.xlabel("i")
    plt.ylabel("j")
    plt.grid()

    ### u - DA
    plt.subplot(212)
    tm = min((t, 300))
    for i in range(N):
        plt.plot(range(t-tm, t), record[i+1, t-tm:t], color=colors[i], alpha=0.75)
        plt.plot(range(t-tm, t), Ix[i, t-tm:t], '--', color=colors[i], alpha=0.85)

    plt.fill_between(range(t-tm, t), record[0, t-tm:t], color='green', alpha=0.1, 
                     label=f"DA={np.around(model.DA, 2)} [{model.var2:.2f}]")

    # plt.ylabel(f"$u$={np.around(model.u.T[0], 1)}")
    plt.ylim((0, 4.3))
    plt.xlabel('ms')
    plt.title(subtitle_3)
    plt.legend()
    plt.grid()
    plt.pause(0.001)

    # save animation
    if is_anim:
        animaker.add_frame(fig)


def plot_activation(activation: np.ndarray, shape=None, ax: object=None):

    size, N = activation.shape
    sqrtsize = int(np.sqrt(size))

    if shape is None:
        nr, nc = int(np.sqrt(N)), int(np.sqrt(N))
        shape = (nc, nr)
    else:
        nr, nc = shape
    
    _, axs = plt.subplots(nr, nc, figsize=(shape[1], shape[0]))
    i = 0
    for row in axs:
        for ax in row:
            ax.imshow(np.flip(activation[:, i].reshape(sqrtsize, sqrtsize), axis=0), cmap='plasma')
            ax.axis('off')
            i += 1
    plt.show()


def calc_cell_tuning(model, track, kernel, threshold, record, **kwargs):

    """ apply convolution """

    record_conv1d = np.empty((model.N, len(track)))
    centers = np.empty((model.N, 2))
    zero_idxs = []
    for i in range(model.N):
        record_conv1d[i] = convolve1d(record[i], kernel)
        if record_conv1d[i].max() > threshold:
            idx = record_conv1d[i].argmax()
            if track[idx, 0] == 0. or track[idx, 1] == 0.:  # exclude the origin
                zero_idxs += [i]
                continue
            centers[i] = track[idx]
        else:
            zero_idxs += [i]

    # trim positions near the origin
    pruned_idxs = [
        i for i, p in enumerate(centers) if p[0] > 0.001 and p[1] > 0.001
    ]

    centers = centers[pruned_idxs]

    return centers, record_conv1d, pruned_idxs


""" place fields and connectivity """


def plot_nodes_edges(centers: np.ndarray, connectivity: np.ndarray,
                     bounds: tuple=(0, 1, 0, 1), ax: plt.Axes=None,
                     c: np.ndarray=None,
                     marker_size: int=10):

    """
    Plot the nodes and edges of the network.

    Parameters
    ----------
    centers : np.ndarray
        The centers of the weight matrix.
    connectivity : np.ndarray
        The connectivity matrix.
    ax : object
        The axis object.
        Default is None.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(len(centers)):
        for j in range(len(centers)):
            if connectivity[i, j] != 0:  # If nodes i and j are connected
                ax.plot([centers[i][0], centers[j][0]],
                        [centers[i][1], centers[j][1]], 'ko-',
                        alpha=0.5, linewidth=1*connectivity[i,j])  # Draw edge

    for position in centers:
        if c is not None:
            ax.scatter(position[0], position[1], marker='o',
                    c=c, cmap='plasma',
                    markersize=marker_size)
        else:
            ax.plot(position[0], position[1], marker='o',
                    color='black', markersize=marker_size)  # 'bo' makes the markers blue circles

    ax.set_title('Graph Visualization', fontsize=13)
    ax.grid(True)
    ax.set_xlim((bounds[0], bounds[1]))
    ax.set_ylim((bounds[2], bounds[3]))
    ax.set_xticks(())
    ax.set_yticks(())
    if ax is None:
        plt.show()


def plot_place_fields(N: int, trajectory: np.ndarray,
                      whole_track: np.ndarray, record: np.ndarray):

    """ plot each cell's preferred location over the trajectory """
    #n_to_plot = min((model.N, 40))
    nrows = 5
    ncols = N // nrows
    fig_0, axs = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    i = 0
    for row in tqdm(axs):
        for ax in row:
            ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.2)
            ax.scatter(whole_track[:, 0], whole_track[:, 1], c=record[i], s=10, cmap='Reds',
               edgecolors='black', linewidths=0)
            #ax.axis('off')
            ax.set_ylim((-0.01, 1.01))
            ax.set_xlim((-0.01, 1.01))
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(f"{i}")
            i += 1

    plt.suptitle("Tuning position for each cell", fontsize=17)
    plt.show()


def plot_one_pc(record: np.ndarray, whole_track: np.ndarray, trajectory: np.ndarray,
                idx: int):

    """ plot one cell's preferred location over the trajectory """

    fig, ax = plt.subplots(figsize=(6, 6))
    z = np.where(record[idx] > 0.01, record[idx], 0)

    ax.scatter(whole_track[:, 0], whole_track[:, 1], c=z, s=20,
               # cmap="plasma",
               edgecolors='black', linewidths=0, alpha=0.2, vmin=0.011)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'w-', lw=2,
            alpha=0.7, label="trajectory")
    ax.set_ylim((-0.01, 1.01))
    ax.set_xlim((-0.01, 1.01))
    ax.set_yticks(())
    ax.set_xticks(())
    ax.grid()
    ax.axis('off')
    # ax.set_title(f"The field of a place cell")
    # ax.legend(loc="upper right")
    plt.show()

    return fig


def plot_centers(model, trajectory, track, kernel, threshold,
                 record,
                 threshold_s=0., ax=None, kernel2=None, plot=False,
                 alpha=0.9,
                 use_knn: bool=False, knn_k: int=3, max_dist: float=0.1,
                 bounds: tuple=(0, 1, 0, 1), **kwargs):

    """
    Plot the centers and edges of the network.

    Parameters
    ----------
    model : object
        The model object.
    trajectory : np.ndarray
        The trajectory of the agent.
    track : np.ndarray
        The track of the environment.
    kernel : np.ndarray
        The convolution kernel.
    threshold : float
        The threshold for the convolution.
    threshold_s : float
        The threshold for the connectivity.
        Default is 0.
    ax : object
        The axis object.
        Default is None.
    kernel2 : np.ndarray
        The second convolution kernel.
        Default is None.
    plot : bool
        Whether to plot the graph.
        Default is False.
    alpha : float
        The transparency of the nodes and edges.
        Default is 0.9.
    use_knn : bool
        Whether to use k-nearest neighbors.
        Default is False.
    knn_k : int
        The number of nearest neighbors.
        Default is 3.
    max_dist : float
        The maximum distance between the centers.
        Default is 0.1.
    bounds : tuple
        The bounds of the plot.
        Default is (0, 1, 0, 1).
    **kwargs : dict
        grid : bool
            Whether to show the grid.
            Default
    """

    # make centers
    positions, _, pruned_idxs = calc_cell_tuning(
        model=model, trajectory=trajectory, track=track, kernel=kernel,
        threshold=threshold, threshold_s=threshold_s, record=record,
        kernel2=kernel2)

    # make connectivity
    if not use_knn:
        model._update_recurrent(kernel=np.ones((4, 4)), threshold=threshold_s)
        connectivity = model.W_rec.copy()

        connectivity = connectivity[pruned_idxs][:, pruned_idxs]
    #
    if use_knn:
        connectivity = utils.calc_knn(centers=positions, k=knn_k, max_dist=max_dist)

        # make it upper triangular
        connectivity *= np.tril(np.ones(connectivity.shape), k=1)

    if not plot:
        if kwargs.get('get_pruned', False):
            return positions, connectivity, pruned_idxs
        return positions, connectivity

    if plot:
        plot_graph(centers=positions, connectivity=connectivity,
                   bounds=bounds, ax=ax, alpha=alpha,
                   grid=kwargs.get('grid', True),
                   color=kwargs.get('color', 'blue'),
                   plot_connections=kwargs.get('plot_connections', True))

    if kwargs.get('get_pruned', False):
        return positions, connectivity, pruned_idxs

    return positions, connectivity


def plot_centers_knn(centers: np.ndarray, k: int=3,
                     max_dist: float=2.,
                     color: str="blue", ax=None,
                     title: str="Graph",
                     show: bool=False,
                     alpha: float=0.5):

    """
    Plot the k-nearest neighbors of the centers.

    Parameters
    ----------
    centers : np.ndarray
        The centers of the weight matrix.
    k : int
        The number of nearest neighbors to consider.
        Default is 3.
    max_dist : float
        The maximum distance between the centers.
        Default is 2..
    color : str
        The color of the nodes.
        Default is 'blue'.
    ax : object
        The axis object.
        Default is None.
    title : str
        The title of the plot.
        Default is 'Graph'.
    show : bool
        Whether to show the plot.
        Default is False.
    alpha : float
        The transparency of the nodes and edges.
        Default is 0.5.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    knn = utils.calc_knn(centers=centers,
                         k=k,
                         max_dist=max_dist)

    ax.scatter(centers[:, 0], centers[:, 1], c=color,
               s=60, marker='o', alpha=alpha)
    for i in range(centers.shape[0]):
        for j in range(centers.shape[0]):
            if knn[i, j] == 1:
                ax.plot([centers[i, 0], centers[j, 0]],
                        [centers[i, 1], centers[j, 1]],
                        '-', color=color, alpha=alpha,
                        linewidth=1)
    ax.set_title(title)

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xticks(())
    ax.set_yticks(())

    # set aspect ratio
    ax.set_aspect('equal', adjustable='box')

    if show:
        plt.show()


def plot_fields_trajectory(record: np.ndarray, whole_track: np.ndarray,
                           whole_track_layer: np.ndarray,
                           trajectory: np.ndarray, model: object,
                           knn_k: int=8, max_dist: float=0.1,
                           bounds: tuple=(0, 1, 0, 1), **kwargs):

    """ plot all cells at the same time """

    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 7))

    z = np.zeros(record.shape[1])
    for i in range(model.N):
        #if i > 7: break
        if record[i].max() > 2: continue
        z += np.where(record[i] > 0., record[i], 0)

    # z = kwargs.get('stb', None) if kwargs.get('stb', None) is not None else z

    ax1.plot(trajectory[:, 0], trajectory[:, 1],
             'b-', lw=5, alpha=0.15, label="trajectory")
    # ax1.scatter(trajectory[:, 0], trajectory[:, 1], c=kwargs('stb', None),
    #             lw=5, alpha=0.15, label="trajectory")

    #z = z.clip(0, 0.8)
    #z = (z.max() - z)/(z.max() - z.min())
    #plt.scatter(whole_track[:, 0], whole_track[:, 1], c=record[7], s=20, cmap='Reds')
    # cax = ax1.scatter(whole_track[:, 0], whole_track[:, 1], c=z, s=20, cmap='Reds',
    #            edgecolors='black', linewidths=0, alpha=0.2, label="place fields",
    #                   vmin=-0.1, vmax=1.1)
    #             # vmin=0.9, vmax=0.92)
    # fig.colorbar(cax, ax=ax1, orientation='vertical')

    #ax1.scatter(trajectory[:, 0], trajectory[:, 1], c=zpc, s=5, cmap='Blues',
    #                edgecolors='black', linewidths=0, marker='o', alpha=0.15, label="trajectory")

    if bounds is not None:
        ax1.set_ylim((bounds[2], bounds[3]))
        ax1.set_xlim((bounds[0], bounds[1]))
    else:
        ax1.set_ylim((-0.01, 1.01))
        ax1.set_xlim((-0.01, 1.01))

    ax1.set_yticks(())
    ax1.set_xticks(())
    ax1.grid()

    #ax1.scatter(whole_track[:, 0], whole_track[:, 1], c=whole_track_layer.sum(axis=1), cmap='Greys', edgecolors='white', alpha=0.3)
    centers, connections = plot_centers(model=model, trajectory=whole_track_layer,
                                        track=whole_track, kernel=np.ones((20)),
                                        threshold=0, threshold_s=0.1,
                                        ax=ax1, record=record, kernel2=np.ones((4, 4)),
                                        alpha=0.5, plot=True, use_knn=True,
                                        knn_k=knn_k, max_dist=max_dist,
                                        bounds=bounds)

    # ax1.set_title("Tuning positions over the trajectory")
    # ax1.legend(loc="upper right")
    plt.show()

    return fig


def plot_pc_diagram(centers: np.ndarray, bounds: tuple=(0, 1, 0, 1)):

    """
    Plot a diagram of the place cells.

    Parameters
    ----------
    centers : np.ndarray
        The centers of the weight matrix.
    bounds : tuple
        The bounds of the plot.
        Default is (0, 1, 0, 1).
    """

    # make linear meshgrid of the environment
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[2], bounds[3], 100)
    X, Y = np.meshgrid(x, y)

    # array of positions
    positions = np.vstack([X.ravel(), Y.ravel()]).T


def plot_graph(centers=None, connectivity=None,
               bounds=(0, 1, 0, 1), ax=None,
               fig=None,
               alpha: float=0.5,
               grid: bool=False,
               color: str='blue',
               marker: str='o',
               label: str=None,
               plot_connections: bool=True,
               plot_centers: bool=False):

    """
    Plot the graph of the network.

    Parameters
    ----------
    centers : np.ndarray
        The centers of the weight matrix.
    connectivity : np.ndarray
        The connectivity matrix.
    bounds : tuple
        The bounds of the plot.
        Default is (0, 1, 0, 1).
    ax : object
        The axis object.
        Default is None.
    alpha : float
        The transparency of the nodes and edges.
        Default is 0.5.
    grid : bool
        Whether to show the grid.
        Default is True.
    color : str
        The color of the nodes.
        Default is 'blue'.
    """

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        new = True
    else:
        new = False

    islabel = False
    if plot_connections:
        for i in range(len(centers)):
            for j in range(len(centers)):
                if connectivity[i, j] != 0:  # If nodes i and j are connected
                    if not islabel:
                        ax.plot([centers[i][0], centers[j][0]], 
                            [centers[i][1], centers[j][1]], 'ko-', 
                            alpha=alpha, linewidth=1*connectivity[i,j], label="place cell")  # Draw edge
                        islabel=True
                        continue
                    ax.plot([centers[i][0], centers[j][0]],
                            [centers[i][1], centers[j][1]], linestyle='-', 
                            color=color, alpha=0.3, linewidth=1,#*connectivity[i,j],
                            markersize=1)  # Draw edge

    # Plot nodes on top of the edges to make them more visible
    if plot_centers:
        # print("plotting centers..")
        for k, position in enumerate(centers):
            if label is not None and k == 0:
                ax.plot(position[0], position[1],
                        marker=marker,
                        color=color, markersize=5,
                        alpha=alpha, label=label)
                continue
            ax.plot(position[0], position[1],
                    marker=marker,
                    color=color, markersize=5,
                    alpha=alpha)  # 'bo' makes the markers blue circles

    # ax.set_title('Graph Visualization')
    ax.grid(grid)
    ax.set_xlim((bounds[0], bounds[1]))
    ax.set_xticks(())
    ax.set_ylim((bounds[2], bounds[3]))
    ax.set_yticks(())

    # equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    if new:
        fig.suptitle("Graph Visualization")
        plt.show()


def plot_c(W, layer, color="blue", alpha=1, show=True,
           k: int=4, max_dist: int=0.1, ax=None,
           **kwargs):

    """
    Plot the centers and connections of the network.

    Parameters
    ----------
    W : np.ndarray
        The weight matrix.
    layer : object
        The layer object.
    color : str
        The color of the nodes.
        Default is 'blue'.
    alpha : float
        The transparency of the nodes.
        Default is 1.
    show : bool
        Whether to show the plot.
        Default is True.
    k : int
        The number of nearest neighbors.
        Default is 4.
    max_dist : float
        The maximum distance between the centers.
        Default is 0.1.
    ax : object
        The axis object.
        Default is None.
    kwargs : dict
        title : str
            The title of the plot.
            Default is None.
        alpha : float
            The transparency of the nodes.
            Default is 0.5.
    """

    # make nodes
    X = layer.centers[:, 0]
    Y = layer.centers[:, 1]

    x = (W * X).sum(axis=1) / W.sum(axis=1)
    y = (W * Y).sum(axis=1) / W.sum(axis=1)

    x = np.where(np.isnan(x), 0, x)
    y = np.where(np.isnan(y), 0, y)

    centers = np.hstack([x[:, None], y[:, None]])

    # title
    if kwargs.get('title', None) is None:
        title = f"Graph Visualization"
    else:
        title = kwargs.get('title', None)
    title += f" | $N_{{PC}}=${np.where(x > 0, 1, 0).sum()}"

    plot_centers_knn(centers=centers,
                     k=k, max_dist=max_dist,
                     color=color, ax=ax,
                     title=title,
                     alpha=kwargs.get('alpha', 0.5))


def make_network_graph(W: np.ndarray, layer: object,
                       k: int=3, max_dist: float=0.1) -> tuple:

    """
    Make a network graph.

    Parameters
    ----------
    W : np.ndarray
        The weight matrix.
    layer : object
        The layer object.
    k : int
        The number of nearest neighbors.
        Default is 3.
    max_dist : float
        The maximum distance between the centers.
        Default is 0.1.
    """

    # make nodes
    X = layer.centers[:, 0]
    Y = layer.centers[:, 1]

    x = (W * X).sum(axis=1) / W.sum(axis=1)
    y = (W * Y).sum(axis=1) / W.sum(axis=1)

    x = np.where(np.isnan(x), 0, x)
    y = np.where(np.isnan(y), 0, y)

    nodes = np.hstack([x[:, None], y[:, None]])

    edges = utils.calc_knn(centers=centers,
                         k=k,
                         max_dist=max_dist)

    return nodes, edges

