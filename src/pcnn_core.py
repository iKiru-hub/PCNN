import numpy as np
import matplotlib.pyplot as plt
try:
    import libs.pclib as pclib
except ImportError:
    try:
        import src.libs.pclib1 as pclib
    except ImportError:
        import warnings
        warnings.warn("pclib [c++] not found, using python version")
        import libs.pclib1 as pclib






class GridLayerWrapper(pclib.GridLayer):

    def __init__(self, N: int, sigma: float,
                 speed: float, kind: str="square"):

        super().__init__(N, sigma, speed, kind)
        self.N = N

    def render_tuning(self):

        dx = 0.01
        x, y = np.meshgrid(np.arange(0, self.N, dx),
                           np.arange(0, self.N, dx))
        points = np.vstack([x.ravel(), y.ravel()]).T

        activity = np.zeros((self.N, len(points)))
        for i, p in enumerate(points):
            activity[:, i] = self(p)

        n = int(np.sqrt(self.N))
        fig, ax = plt.subplots(n, n, figsize=(10, 10))
        ax = ax.flatten()
        for i in range(n*n):
            ax[i].imshow(activity[:, i].reshape(self.N, self.N),
                         cmap="hot", origin="lower")
            ax[i].axis("off")

        plt.show()

