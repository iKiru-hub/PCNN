import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from utils import logger





class SpatialCellLayer(ABC):

    def __init__(self, nx: int, ny: int, sigma: float,
                 offset: np.ndarray=np.array([0, 0]),
                 bounds: tuple=(0, 1, 0, 1)):

        """
        Generate a spatial cell layer with N cells.

        Parameters
        ----------
        nx: int
            the number of cells in the x direction.
        ny: int
            the number of cells in the y direction.
        sigma: float
            the variance of the gaussian.
        bounds: tuple
            the bounds of the grid layer.
            Default is (0, 1, 0, 1)
        """

        self.N = nx * ny
        self.sigma = sigma
        self.bounds = bounds
        self.offset = offset
        self.centers = self._make_centers(nx=nx, ny=ny)

        # add offset
        self.centers += offset

    def __call__(self, x: np.ndarray):

        if isinstance(x, list):
            x = np.array(x).reshape(-1, 2)
        elif len(x.shape) == 1:
            x = x.reshape(-1, 2)

        a = np.exp(-np.sum((x - self.centers)**2,
                           axis=1) / (2*self.sigma**2))

        return a

    def __len__(self):
        return self.N

    def __str__(self):
        return f"{self.__class__.__name__}" + \
            f"(N={len(self)}, s={self.sigma}, o={self.offset.tolist()})"

    @abstractmethod
    def _make_centers(self, nx: int, ny: int):
        pass

    def plot(self, ax: plt.Axes=None):

        if ax is None:
            fig, ax = plt.subplots()
            show = True
        else:
            show = False

        # create meshgrid as a list of points
        x = np.linspace(0, 1, 150)
        y = np.linspace(0, 1, 150)
        X, Y = np.meshgrid(x, y)
        points = np.c_[X.ravel(), Y.ravel()]

        # compute grid cell activity
        Z = np.zeros((len(points), self.N))

        for i, p in enumerate(points):
            Z[i] = self(p).flatten()

        ax.scatter(points[:, 0], points[:, 1],
                   c=Z.sum(axis=1), s=2)
        ax.set_title(f"{self}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('equal')
        ax.axis('off')

        if show:
            plt.show()



class GridCellLayer(SpatialCellLayer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_centers(self, nx: int, ny: int):

        """Generate triangular grid of cell centers."""

        n = np.sqrt(self.N).astype(int)
        x_min, x_max, y_min, y_max = self.bounds
        x_spacing = (x_max - x_min) / (nx - 1)
        y_spacing = (y_max - y_min) / (ny - 1)

        centers = []
        extra = 0
        for j in range(n):
            for i in range(n):
                x = x_min + i * x_spacing
                y = y_min + j * y_spacing
                if j % 2 == 1:
                    x += x_spacing / 2

                if x_min <= x <= x_max and y_min <= y <= y_max:
                    extra += 1
                centers.append(np.array([x, y]))

        if len(centers) != self.N:
            logger.warning(f"{extra} neurons are " + \
                f"out of the provided bounds")

        return np.array(centers)



class PlaceCellLayer(SpatialCellLayer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_centers(self, nx: int, ny: int):

        xx, yy = np.meshgrid(np.linspace(self.bounds[0],
                                         self.bounds[1], nx),
                             np.linspace(self.bounds[2],
                                         self.bounds[3], ny))

        return np.vstack([xx.ravel(), yy.ravel()]).T



class LayerGroup:

    def __init__(self, layers: list):

        self.layers = layers
        self.N = sum([len(layer) for layer in layers])

    def __call__(self, x: np.ndarray):

        a = np.zeros(0)
        for layer in self.layers:
            a = np.hstack([a, layer(x)])

        return a

    def __len__(self):
        return self.N

    def __str__(self):
        return f"{self.__class__.__name__}" + \
            f"(N={len(self)}, layers={len(self.layers)})"

    def plot(self):

        fig, ax = plt.subplots(1, len(self.layers),
                               figsize=(4*len(self.layers), 4))

        for i, layer in enumerate(self.layers):
            layer.plot(ax=ax[i])

        plt.show()




if __name__ == "__main__":

    gc = [
        GridCellLayer(nx=5, ny=5, sigma=0.05),
        GridCellLayer(nx=5, ny=5, sigma=0.05,
                      offset=np.array([0., 0.13])),
        GridCellLayer(nx=4, ny=4, sigma=0.1)
    ]

    group = LayerGroup(layers=gc)

    print(group)

    a = group(np.array([0.5, 0.5]))
    print(a.shape)

    group.plot()





