import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
try:
    # import libs.pclib as pclib
    import core.build.pclib as pclib
    print("imported pclib [c++]")
except ImportError:
    try:
        import src.libs.pclib1 as pclib
    except ImportError:
        import warnings
        warnings.warn("pclib [c++] not found, using python version")
        import libs.pclib1 as pclib


logger = utils.setup_logger(__name__, level=3)



class GridLayerWrapper(pclib.GridLayerSq):

    def __init__(self, N: int, sigma: float,
                 speed: float, init_bounds: list=[-1., 1., -1., 1.],
                boundary_type: str="square",
                 positions_type: str="square"):

        super().__init__(N, sigma, speed, init_bounds,
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

        fig, self.axs = plt.subplots(8, 8, figsize=(9, 9),
                                   sharex=True, sharey=True)
        fig.tight_layout()
        # _, self.axgc = plt.subplots(figsize=(3, 3))
        self.axs = self.axs.flatten()

    def __call__(self, x: list, position: list=[-1., -1.]):

        u, y = self.pcnn2D(x)
        # self.pcnn2D.update(*position)
        self.pcnn2D.update()
        self.record_pcnn.append(u.tolist())
        self.record_xfl.append(y.tolist())
        # if len(self.record_pcnn) > self.max_iter:
        #     self.record_pcnn.pop(0)

        if len(self.record_xfl) > 200:
            self.record_xfl.pop(0)

    def render(self, traj):

        traj = np.array(traj)[-len(self.record_pcnn):]
        record = np.array(self.record_pcnn)
        # self.axgc.clear()
        # self.axgc.imshow(np.array(self.record_xfl).T,
        #                  cmap="plasma", aspect="auto")
        # self.axgc.set_title(f"GC activity {np.sum(self.record_xfl[-1]):.3f}")
        # self.axgc.axis("off")

        centers = self.pcnn2D.get_centers()

        for i, ax in enumerate(self.axs):

            ax.clear()

            ax.plot(traj[:, 0], traj[:, 1],
                    "k", alpha=0.3, lw=0.4)
            ax.scatter(traj[:, 0], traj[:, 1],
                       c=record[:, i], cmap="plasma",
                       s=5*record[:, i],
                       alpha=0.4)

            ax.scatter(centers[i, 0], centers[i, 1],
                       c="k", s=15)
            ax.axis("off")
            # ax.set_title(f"{i}")
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)

        plt.pause(0.001)


class PCNNplotter:

    def __init__(self, pcnn2D: object,
                 max_iter: int=1000):

        self.pcnn2D = pcnn2D
        self.max_iter = max_iter

        self.record_pcnn = []
        self.trajectory = []

        fig, self.axs = plt.subplots(9, 9, figsize=(9, 9),
                                   sharex=True, sharey=True)
        fig.tight_layout()
        # _, self.axgc = plt.subplots(figsize=(3, 3))
        self.axs = self.axs.flatten()

    def __call__(self, u: list, position):

        self.record_pcnn.append(u.tolist())
        self.trajectory.append(position)
        if len(self.record_pcnn) > self.max_iter:
            self.record_pcnn.pop(0)
            self.trajectory.pop(0)

    def render(self):

        traj = np.array(self.trajectory)
        record = np.array(self.record_pcnn)

        centers = self.pcnn2D.get_centers()

        for i, ax in enumerate(self.axs):

            ax.clear()

            ax.plot(traj[:, 0], traj[:, 1],
                    "k", alpha=0.3, lw=0.4)
            ax.scatter(traj[:, 0], traj[:, 1],
                       c=record[:, i], cmap="plasma",
                       s=5*record[:, i],
                       alpha=0.4)

            ax.scatter(centers[i, 0], centers[i, 1],
                       c="k", s=15)
            ax.axis("off")
            ax.set_title(f"{np.sum(record[-1, i]):.3f}")
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)

        # plt.pause(0.001)


def run_pcnn(duration=100000):

    #hexagon = pcr.pclib.Hexagon()
    # gc_list = [
    #     pclib.GridLayer(40, 0.01, 0.01, [-1., 0., -1., 0.],
    #                     "square", "square"),
    #     pclib.GridLayer(30, 0.01, 0.05, [0., 1., 0., 1.],
    #                     "square", "square"),
    #     pclib.GridLayer(30, 0.01, 0.1, [-1., 0., 0., 1.],
    #                     "square", "square")
    #     # pclib.GridLayer(30, 0.03, 0.5, "square", "random_square"),
    #     # pclib.GridLayer(25, 0.02, 0.4, "hexagon", "random_circle"),
    # ]

    idx = 1

    if idx == 0:
        gcn = pclib.GridNetwork([pclib.GridLayer(N=9, sigma=0.04, speed=0.1, init_bounds=[-1, 0, -1, 0],
                                  boundary_type="square"),
                   pclib.GridLayer(N=9, sigma=0.04, speed=0.1, init_bounds=[0, 1, -1, 0],
                       boundary_type="square"),
                   pclib.GridLayer(N=9, sigma=0.04, speed=0.1, init_bounds=[-1, 0, 0, 1],
                       boundary_type="square"),
                   pclib.GridLayer(N=9, sigma=0.04, speed=0.1, init_bounds=[0, 1, 0, 1],
                       boundary_type="square"),
                   pclib.GridLayer(N=9, sigma=0.04, speed=0.05, init_bounds=[-1, 0, -1, 0],
                                  boundary_type="square"),
                   pclib.GridLayer(N=9, sigma=0.04, speed=0.05, init_bounds=[0, 1, -1, 0],
                      boundary_type="square"),
                   pclib.GridLayer(N=9, sigma=0.04, speed=0.05, init_bounds=[-1, 0, 0, 1],
                      boundary_type="square"),
                   pclib.GridLayer(N=9, sigma=0.04, speed=0.05, init_bounds=[0, 1, 0, 1],
                      boundary_type="square"),
                   pclib.GridLayer(N=9, sigma=0.04, speed=0.025, init_bounds=[-1, 0, -1, 0],
                                  boundary_type="square"),
                   pclib.GridLayer(N=9, sigma=0.04, speed=0.025, init_bounds=[0, 1, -1, 0],
                      boundary_type="square"),
                   pclib.GridLayer(N=9, sigma=0.04, speed=0.025, init_bounds=[-1, 0, 0, 1],
                      boundary_type="square"),
                   pclib.GridLayer(N=9, sigma=0.04, speed=0.025, init_bounds=[0, 1, 0, 1],
                                  boundary_type="square")])

        pcnn_ = pclib.PCNNgrid(N=25, Nj=len(gcn), gain=7., offset=1.1,
                               clip_min=0.01,
                               threshold=0.4,
                               rep_threshold=0.5,
                               rec_threshold=0.1,
                               num_neighbors=8, trace_tau=0.1,
                               xfilter=gcn, name="2D")

    elif idx == 1:
        gcn = pclib.GridNetworkSq([
                   pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[-1, 0, -1, 0]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[0, 1, -1, 0]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[-1, 0, 0, 1]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[0, 1, 0, 1]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.07, bounds=[-1, 0, -1, 0]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.07, bounds=[0, 1, -1, 0]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.07, bounds=[-1, 0, 0, 1]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.07, bounds=[0, 1, 0, 1]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.03, bounds=[-1, 0, -1, 0]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.03, bounds=[0, 1, -1, 0]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.03, bounds=[-1, 0, 0, 1]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.03, bounds=[0, 1, 0, 1]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.015, bounds=[-1, 0, -1, 0]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.015, bounds=[0, 1, -1, 0]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.015, bounds=[-1, 0, 0, 1]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.015, bounds=[0, 1, 0, 1]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.005, bounds=[-1, 0, -1, 0]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.005, bounds=[0, 1, -1, 0]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.005, bounds=[-1, 0, 0, 1]),
                   pclib.GridLayerSq(sigma=0.04, speed=0.005, bounds=[0, 1, 0, 1])
        ])

        pcnn_ = pclib.PCNNsq(N=13**2, Nj=len(gcn), gain=7., offset=1.4,
                               clip_min=0.01,
                               threshold=0.5,
                               rep_threshold=0.85,
                               rec_threshold=0.1,
                               num_neighbors=8, trace_tau=0.1,
                               xfilter=gcn, name="2D")
 
    pcnn2d = PCNNwrapper(pcnn_,
                         max_iter=100_000)
 
    print(pcnn2d)

    a = []
    #p = np.array([0.2, 0.2])
    x, y = 0.2, 0.2
    x0, y0 = 0.2, 0.2
    traj = [[x, y]]
    acc = np.zeros((len(gcn), 100))

    size = 20
    s = np.array([0.005, 0.005])

    for t in tqdm(range(duration)):

        x += s[0]
        y += s[1]

        # hexagon boundaries & gc
        pcnn2d([x - x0, y - y0], [x, y])

        x0 = x
        y0 = y

        if t % 100 == 0:
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
        if t % 10000 == 0:
            pcnn2d.render(traj)

    print("done")
    # plt.show()

    # centers = []
    # for c in pcnn2d.pcnn2D.get_centers():
    #     if c[0] > 0 and c[1] > 0:
    #         centers.append(c)

    # print("c: ", pcnn2d.pcnn2D.get_centers())

    centers = np.array(pcnn2d.pcnn2D.get_centers(nonzero=True))
    print("centers: ", centers, centers.shape)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(*np.array(traj).T, "-k", lw=1, alpha=0.4)
    ax.scatter(centers[:, 0], centers[:, 1], c="r", s=15)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"N={len(centers)}")

    plt.show()


def run_pcnn_hex(duration: int=100_000):

    gcn = pclib.GridNetworkHex([
                pclib.GridLayerHex(sigma=0.04, speed=0.6),
                pclib.GridLayerHex(sigma=0.04, speed=0.4),
                pclib.GridLayerHex(sigma=0.04, speed=0.3),
                pclib.GridLayerHex(sigma=0.04, speed=0.2),
                pclib.GridLayerHex(sigma=0.04, speed=0.1),
                pclib.GridLayerHex(sigma=0.04, speed=0.08),
                pclib.GridLayerHex(sigma=0.04, speed=0.06),
                pclib.GridLayerHex(sigma=0.04, speed=0.03)])

    pcnn_ = pclib.PCNN(N=100,
                       Nj=len(gcn),
                       gain=10.,
                       offset=1.2,
                       clip_min=0.01,
                       threshold=0.3,
                       rep_threshold=0.8,
                       rec_threshold=1.0,
                       min_rep_threshold=0.99,
                       xfilter=gcn,
                       tau_trace=2.0,
                       name="2D")

    pcnn2d = PCNNwrapper(pcnn_,
                         max_iter=duration)
 
    a = []
    x, y = 0.2, 0.2
    x0, y0 = 0.2, 0.2
    traj = [[x, y]]
    acc = np.zeros((len(gcn), 100))

    size = 30
    s = np.array([0.005, 0.005])

    logger("[running pcnn hex]")

    for t in tqdm(range(duration)):

        x += s[0]
        y += s[1]

        # hexagon boundaries & gc
        pcnn2d([x - x0, y - y0])#, [x, y])
        # pcnn2d([s[0], s[1]], [x, y])

        x0 = x
        y0 = y

        if t % 100 == 0:
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
        if t % 10000 == 0:
            pcnn2d.render(traj)

    logger("[done]")
    # logger(pcnn2d.pcnn2D.get_centers(),
    #       pcnn2d.pcnn2D.get_centers().shape)
    pcnn2d.render(traj)

    centers = []
    for c in pcnn2d.pcnn2D.get_centers():
        if c[0] > 0 and c[1] > 0:
            centers.append(c)

    centers = np.array(centers)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(*np.array(traj).T, "-k", lw=1, alpha=0.4)
    ax.scatter(centers[:, 0], centers[:, 1], c="r", s=15)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"N={len(centers)}")

    plt.show()


def run_gcn():

    #hexagon = pcr.pclib.Hexagon()

    gcn = pclib.GridNetwork([pclib.GridLayer(N=9, sigma=0.04, speed=0.1, init_bounds=[-1, 0, -1, 0],
                              boundary_type="square"),
               pclib.GridLayer(N=9, sigma=0.04, speed=0.1, init_bounds=[0, 1, -1, 0],
                   boundary_type="square"),
               pclib.GridLayer(N=9, sigma=0.04, speed=0.1, init_bounds=[-1, 0, 0, 1],
                   boundary_type="square"),
               pclib.GridLayer(N=9, sigma=0.04, speed=0.1, init_bounds=[0, 1, 0, 1],
                   boundary_type="square"),
               pclib.GridLayer(N=9, sigma=0.04, speed=0.05, init_bounds=[-1, 0, -1, 0],
                              boundary_type="square"),
               pclib.GridLayer(N=9, sigma=0.04, speed=0.05, init_bounds=[0, 1, -1, 0],
                  boundary_type="square"),
               pclib.GridLayer(N=9, sigma=0.04, speed=0.05, init_bounds=[-1, 0, 0, 1],
                  boundary_type="square"),
               pclib.GridLayer(N=9, sigma=0.04, speed=0.05, init_bounds=[0, 1, 0, 1],
                  boundary_type="square"),
               pclib.GridLayer(N=9, sigma=0.04, speed=0.025, init_bounds=[-1, 0, -1, 0],
                              boundary_type="square"),
               pclib.GridLayer(N=9, sigma=0.04, speed=0.025, init_bounds=[0, 1, -1, 0],
                  boundary_type="square"),
               pclib.GridLayer(N=9, sigma=0.04, speed=0.025, init_bounds=[-1, 0, 0, 1],
                  boundary_type="square"),
               pclib.GridLayer(N=9, sigma=0.04, speed=0.025, init_bounds=[0, 1, 0, 1],
                              boundary_type="square")])

    # gcn = GridLayerWrapper(9, 0.01, 0.4, "square", "square")


    a = []
    #p = np.array([0.2, 0.2])
    x, y = 0.2, 0.2
    x0, y0 = 0.2, 0.2
    traj = [[x, y]]
    acc = []

    size = 100
    s = np.array([0.09, 0.09])

    # fig, axs = plt.subplots(5, 5, figsize=(6, 6))
    # axs = axs.flatten()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for t in range(200_000):

        x += s[0]
        y += s[1]

        # gc
        acc += [gcn([x - x0, y - y0]).tolist()]

        if len(acc) > 200_000:
            acc.pop(0)

        x0 = x
        y0 = y


        traj += [[x, y]]

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

        if t % 1000 == 0:

            trajj = np.array(traj)[-len(acc):]
            accc = np.array(acc)

            ax.clear()
            ax.plot(trajj[:, 0], trajj[:, 1], "k", alpha=0.4, lw=0.5)
            for l in range(len(gcn)):
                ax.scatter(trajj[:, 0], trajj[:, 1],
                           c=accc[:, l], cmap="plasma",
                           s=7*accc[:, l]*(accc[:, l]>0.1))

            ax.set_xlim(0, size)
            ax.set_ylim(0, size)
            ax.axis("off")
            plt.pause(0.001)

            # for l, ax in enumerate(axs):

            #     ax.clear()

            #     ax.plot(trajj[:, 0], trajj[:, 1], "k", alpha=0.4, lw=0.5)
            #     ax.scatter(trajj[:, 0], trajj[:, 1],
            #                c=accc[:, l], cmap="plasma",
            #                s=7*accc[:, l]*(accc[:, l]>0.1))
            #     ax.set_xlim(0, size)
            #     ax.set_ylim(0, size)
            #     ax.axis("off")

            #     plt.pause(0.001)


def run_model(duration: int=100_000):

    gcn = pclib.GridHexNetwork(
           [pclib.GridHexLayer(sigma=0.04, speed=0.1),
            pclib.GridHexLayer(sigma=0.04, speed=0.07),
            pclib.GridHexLayer(sigma=0.04, speed=0.03),
            pclib.GridHexLayer(sigma=0.04, speed=0.025)])

    pcnn_ = pclib.PCNNgridhex(N=25, Nj=len(gcn), gain=8.,
                              offset=1.3,
                              clip_min=0.01,
                              threshold=0.3,
                              rep_threshold=0.4,
                              rec_threshold=0.1,
                              num_neighbors=8, trace_tau=0.1,
                              xfilter=gcn, name="2D")

    pcnn2d = PCNNwrapper(pcnn_,
                         max_iter=duration)
 
    print(pcnn2d)

    a = []
    x, y = 0.2, 0.2
    x0, y0 = 0.2, 0.2
    traj = [[x, y]]
    acc = np.zeros((len(gcn), 100))

    size = 20
    s = np.array([0.005, 0.005])

    for t in range(duration):

        x += s[0]
        y += s[1]

        # hexagon boundaries & gc
        pcnn2d([x - x0, y - y0])

        x0 = x
        y0 = y

        if t % 100 == 0:
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
        # if t % 50_000 == 0:
        #     pcnn2d.render(traj)
        #     print(f"update time: {t/1000:.0f} sec")

    pcnn2d.render(traj)
    plt.show()


def run_gcn_hex():

    #hexagon = pcr.pclib.Hexagon()

    gcn = pclib.GridHexNetwork([
                pclib.GridHexLayer(sigma=0.04, speed=0.13),
                pclib.GridHexLayer(sigma=0.04, speed=0.1),
                pclib.GridHexLayer(sigma=0.04, speed=0.08),
                pclib.GridHexLayer(sigma=0.04, speed=0.06),
                pclib.GridHexLayer(sigma=0.04, speed=0.01)])
    # gcn = GridLayerWrapper(9, 0.01, 0.4, "square", "square")


    a = []
    #p = np.array([0.2, 0.2])
    x, y = 0.2, 0.2
    x0, y0 = 0.2, 0.2
    traj = [[x, y]]
    acc_size = 10_000
    acc = np.zeros((len(gcn), acc_size))

    size = 100
    s = np.array([0.09, 0.09])

    # fig, axs = plt.subplots(5, 5, figsize=(6, 6))
    # axs = axs.flatten()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for t in range(200_000):

        x += s[0]
        y += s[1]

        # gc
        # a_ = gcn([x - x0, y - y0]).tolist()
        a_ = gcn([s[0], s[1]]).tolist()
        print(f"{np.around(a_, 2)}")

        acc[:, :-1] = acc[:, 1:]
        acc[:, -1] = a_

        x0 = x
        y0 = y


        traj += [[x, y]]

        if t % 1000 == 0:
            s = np.random.uniform(-1, 1, 2)
            s = 0.1 * s / np.abs(s).sum()

        # hit wall
        if x <= 0 or x >= size:
            s[0] *= -1
            x += s[0]
        elif y <= 0 or y >= size:
            s[1] *= -1
            y += s[1]

        if t % 1000 == 0:

            trajj = np.array(traj)[-len(acc):]
            # accc = np.array(acc)

            ax.clear()
            ax.imshow(acc, cmap="plasma", aspect="auto")
            # ax.plot(trajj[:, 0], trajj[:, 1], "k", alpha=0.4, lw=0.5)
            # for l in range(len(gcn)):
            #     ax.scatter(trajj[:, 0], trajj[:, 1],
            #                c=accc[:, l], cmap="plasma",
            #                s=7*accc[:, l]*(accc[:, l]>0.1))

            ax.set_xlim(0, acc_size)
            # ax.set_ylim(0, acc.s)
            # ax.axis("off")
            ax.set_title(f"t={t}")
            plt.pause(0.001)

            # for l, ax in enumerate(axs):

            #     ax.clear()

            #     ax.plot(trajj[:, 0], trajj[:, 1], "k", alpha=0.4, lw=0.5)
            #     ax.scatter(trajj[:, 0], trajj[:, 1],
            #                c=accc[:, l], cmap="plasma",
            #                s=7*accc[:, l]*(accc[:, l]>0.1))
            #     ax.set_xlim(0, size)
            #     ax.set_ylim(0, size)
            #     ax.axis("off")

            #     plt.pause(0.001)



def run_model_sq(duration: int=100_000):

    gcn = pclib.GridNetworkSq([
               pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[-1, 0, -1, 0]),
               pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[0, 1, -1, 0]),
               pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[-1, 0, 0, 1]),
               pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[0, 1, 0, 1]),
               pclib.GridLayerSq(sigma=0.04, speed=0.05, bounds=[-1, 0, -1, 0]),
               pclib.GridLayerSq(sigma=0.04, speed=0.05, bounds=[0, 1, -1, 0]),
               pclib.GridLayerSq(sigma=0.04, speed=0.05, bounds=[-1, 0, 0, 1]),
               pclib.GridLayerSq(sigma=0.04, speed=0.05, bounds=[0, 1, 0, 1]),
               pclib.GridLayerSq(sigma=0.04, speed=0.025, bounds=[-1, 0, -1, 0]),
               pclib.GridLayerSq(sigma=0.04, speed=0.025, bounds=[0, 1, -1, 0]),
               pclib.GridLayerSq(sigma=0.04, speed=0.025, bounds=[-1, 0, 0, 1]),
               pclib.GridLayerSq(sigma=0.04, speed=0.025, bounds=[0, 1, 0, 1])])

    pcnn_ = pclib.PCNNsq(N=40, Nj=len(gcn), gain=7., offset=1.1,
                           clip_min=0.01,
                           threshold=0.4,
                           rep_threshold=0.5,
                           rec_threshold=0.1,
                           num_neighbors=8, trace_tau=0.1,
                           xfilter=gcn, name="2D")

    pcnn2d = PCNNwrapper(pcnn_,
                         max_iter=duration)
 
    print(pcnn2d)

    a = []
    x, y = 0.2, 0.2
    x0, y0 = 0.2, 0.2
    traj = [[x, y]]
    acc = np.zeros((len(gcn), 100))

    size = 20
    s = np.array([0.005, 0.005]) * 100.

    for t in tqdm(range(duration)):

        x += s[0]
        y += s[1]

        # hexagon boundaries & gc
        pcnn2d([x - x0, y - y0])

        x0 = x
        y0 = y

        if t % 100 == 0:
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
        if t % 100 == 0:
            pcnn2d.render(traj)
            print(f"update time: {t/1000:.0f} sec")

    # pcnn2d.render(traj)
    # plt.show()



if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=100_000)

    args = parser.parse_args()


    np.random.seed(0)

    # run_pcnn(duration=args.duration)
    run_pcnn_hex(duration=args.duration)
    # run_gcn_hex()
    #run_gcn()
    # run_model_sq()
