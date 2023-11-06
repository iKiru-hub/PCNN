import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_graph_colored(W: np.ndarray):

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

