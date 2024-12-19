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
                 speed: float, boundary_type: str="square",
                 positions_type: str="square"):

        super().__init__(N, sigma, speed,
                         boundary_type, positions_type)
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


class PCNNwrapper:

    def __init__(self, pcnn2D: object,
                 max_iter: int=1000):

        self.pcnn2D = pcnn2D
        self.max_iter = max_iter

        self.record_pcnn = []
        self.record_xfl = []

    def __call__(self, x: list):

        u, y = self.pcnn2D(x)
        self.pcnn2D.update()
        self.record_pcnn.append(u.tolist())
        self.record_xfl.append(y.tolist())

        if len(self.record_pcnn) > self.max_iter:
            self.record_pcnn.pop(0)
            self.record_xfl.pop(0)

    def render(self, ax1: plt.Axes, ax2: plt.Axes):

        ax1.imshow(np.array(self.record_pcnn).T, cmap="plasma", origin="lower",
                   aspect="auto")
        ax1.set_title(f"PCNN {np.max(self.record_pcnn[-1]):.2f}")
        ax1.axis("off")
        ax2.imshow(np.array(self.record_xfl).T, cmap="plasma", origin="lower",
                   aspect="auto")
        ax2.set_title(f"XFL {np.max(self.record_xfl[-1]):.2f}")
        ax2.axis("off")



if __name__ == '__main__':

    """ online """

    #hexagon = pcr.pclib.Hexagon()
    gc_list = [
        pclib.GridLayer(25, 0.05, 0.9, "square", "square"),
        pclib.GridLayer(25, 0.2, 0.7, "square", "random_square"),
        pclib.GridLayer(25, 0.03, 0.9, "hexagon", "random_circle"),
    ]
    gcn = pclib.GridNetwork(gc_list)
    pcnn_ = pclib.PCNNgrid(N=30, Nj=len(gcn), gain=3., offset=1.,
                        clip_min=0.09,
                        threshold=0.07,
                        rep_threshold=0.3,
                        rec_threshold=0.1,
                        num_neighbors=8, trace_tau=0.1,
                        xfilter=gcn, name="2D")

    pcnn2d = PCNNwrapper(pcnn_,
                         max_iter=100)


    a = []
    #p = np.array([0.2, 0.2])
    x, y = 0.2, 0.2
    x0, y0 = 0.2, 0.2
    traj = [[x, y]]
    acc = np.zeros((len(gcn), 100))

    size = 20
    s = np.array([0.005, 0.005])
    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(9, 3))

    for t in range(200_000):

        x += s[0]
        y += s[1]

        # hexagon boundaries & gc
        #x, y = hexagon(x, y)
        #acc[:, :-1] = acc[:, 1:]
        #acc[:, -1] = gcn([x - x0, y - y0]).flatten()
        #posj += [gch.get_positions()[4]]
        pcnn2d([x - x0, y - y0])

        x0 = x
        y0 = y


        if t % 20 == 0:
            s = np.random.uniform(-1, 1, 2)
            s = 0.1 * s / np.abs(s).sum()

        # hit wall
        if x <= 0 or x >= size:
            s[0] *= -1
            x += s[0]
        elif y <= 0 or y >= size:
            s[1] *= -1
            y += s[1]

        traj += [[x, y]]
        if t % 10 == 0:


            ax.clear()
            ax1.clear()
            ax2.clear()

            ax.plot(np.array(traj)[:, 0], np.array(traj)[:, 1], "k")
            ax.set_xlim(0, size)
            ax.set_ylim(0, size)
            ax.axis("off")

            pcnn2d.render(ax1, ax2)

            plt.pause(0.001)



    print("done")

