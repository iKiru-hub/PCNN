import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from IPython.display import clear_output 

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
    """

    Nj = model.Nj
    N = model.N
    animaker = kwargs.get('animaker', None)
    is_anim = bool(animaker)

    if kwargs.get('subtitle_2', None) is None:
        subtitle_2 = np.around(model.temp.flatten(), 2)
    else:
        subtitle_2 = kwargs.get('subtitle_2', None)
    
    clf()
    fig = plt.figure(figsize=(20, 6))
    plt.tight_layout()

    ### input
    plt.subplot(221)
    plt.imshow(X.T[:, t-20:t], cmap="Greys")
    plt.xticks(())
    plt.title(f"{t=}")
    plt.xlabel("ms")
    plt.grid()

    ### weights
    plt.subplot(232)
    plt.axvline(0, color='black', alpha=0.3)
    plt.axvline(model._wff_max, color='red', alpha=0.9)
    for i in range(N):
        plt.plot(np.flip(model.Wff[i], axis=0), range(Nj), '-', color=colors[i], alpha=0.3)
        plt.plot(np.flip(model.Wff[i], axis=0), range(Nj), 'o', color=colors[i], alpha=0.5)        
        plt.title(f"Weights")
    plt.yticks(())
    plt.xlim((-0.1, 5))
    plt.grid()

    ### weight matrix
    plt.subplot(233)
    plt.imshow(model.Wff.T, cmap="plasma")
    plt.title(f"Temperatures: {subtitle_2}")
    plt.xlabel("i")
    plt.ylabel("j")
    plt.grid()

    ### u - DA
    plt.subplot(212)
    tm = min((t, 300))
    for i in range(N):
        plt.plot(range(t-tm, t), record[i+1, t-tm:t], color=colors[i], alpha=0.75)
        plt.plot(range(t-tm, t), Ix[i, t-tm:t], '--', color=colors[i], alpha=0.85)

    plt.fill_between(range(t-tm, t), record[0, t-tm:t], color='green', alpha=0.1, label='DA')

    plt.ylabel(f"$u$={np.around(model.u.T[0], 1)}")
    plt.ylim((0, 4.3))
    plt.xlabel('ms')
    plt.legend()
    plt.grid()
    plt.pause(0.001)

    # save animation
    if is_anim:
        animaker.add_frame(fig)


