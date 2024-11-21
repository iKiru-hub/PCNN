import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_surface(points: np.ndarray, h: np.ndarray):

    """
    make a surface from a set of points
    """

    # add a z component = 0.
    points = np.hstack((points,
                        h.reshape(-1, 1)))

    from scipy.spatial import Delaunay

    tri = Delaunay(points[:, :2])

    # --- PLOT
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

    # plot wireframe
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                    triangles=tri.simplices, linewidth=1.,
                    edgecolor='k',
                    antialiased=True, alpha=0.2)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.axis('off')
    plt.show()

if __name__ == '__main__':

    n = 30

    # generate some random points
    points = np.random.rand(n, 2)
    h = np.array([np.random.rand()/4 for i in range(4)] + [0 for i in range(n-4)])

    make_surface(points, h)
