import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
import time, json

import utils as u
from utils import logger
import minimal_model as mm
import inputools.Trajectory as it

try:
    import main
except ImportError:
    import os, sys
    try:
        sys.path.append(os.getcwd())
        import main
    except ImportError:
        logger.warning(f"Cannot find the main module, {os.getcwd()}")
        try:
            os.chdir("../")
            import main
        except ImportError:
           logger.warning(f"Cannot find the 'main' module, {os.getcwd()}")


""" PC layer class """

SAVEPATH = "cache/campoverde/"


class PClayer:

    def __init__(self, n: int, sigma: int,
                 policy: object=None, **kwargs):

        self.N = n**2
        self.n = n
        self.name = kwargs.get("name", "PClayer")
        self.bounds = kwargs.get("bounds", (0, 1, 0, 1))
        self._k = kwargs.get("k", 4)
        self.sigma = sigma
        self.spacing = None
        self._endpoints = kwargs.get("endpoints", True)
        self.centers = self._centers_(endpoints=self._endpoints)
        self.centers = np.around(self.centers, 3)

        # print(len(self.centers))
        e, w = self._make_connections(K=kwargs.get("K", 4))
        self.edges = e
        self.weights = w

        self.a = np.zeros(self.N)

        # policy
        self.policy = policy

    def __repr__(self):
        return f"PClayer(N={self.N}, s={self.sigma})"

    def __call__(self, x: np.ndarray):

        if isinstance(x, list):
            x = np.array(x).reshape(-1, 2)
        elif len(x.shape) == 1:
            x = x.reshape(-1, 2)

        self.a = np.exp(-np.sum((x - self.centers)**2, axis=1) / (2*self.sigma**2))

        # logger.debug(f"{self.name}_max={self.centers[np.argmax(self.a), :]} [{np.around(x, 2)}]")

        return self.a.copy()

    def forward(self, z: np.ndarray):

        """
        forward pass of an input targeting all neurons
        individually
        """

        self.a = z.copy()
        # self.a += -self.a / 2 + z

        return self.a.copy()

    def _calc_movement(self, a1: np.ndarray, a2: np.ndarray, speed: float = 1.,
                       flag: str="direct") -> np.ndarray:

        """
        Calculate a movement in space (the reference ambient space)
        given two population vectors [activations]
        """

        pos1 = self._calc_average_pos(a=a1)
        pos2 = self._calc_average_pos(a=a2)

        if pos2 is None or pos1 is None:
            return np.array([0, 0]), None
        elif pos2[0] - pos1[0] == 0:
            return np.array([0, 0]), None

        # Calculate the angle between the two vectors
        angle = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])

        # ensure angle is in range [0, 2π]
        if angle < 0:
            angle += 2 * np.pi

        #
        angle_deg = np.degrees(angle)

        # Calculate the distance between the two positions
        distance = np.linalg.norm(pos1 - pos2)

        # make speed proportional to the distance
        speed = max((distance/2, 0.01)) if speed > distance else speed

        # Calculate the movement as a new position with a speed of `speed`
        dx = speed * np.cos(angle)
        dy = speed * np.sin(angle)

        return np.array([dx, dy]), distance

    def _calc_average_pos(self, a: np.ndarray) -> np.ndarray:

        """
        calculate the average position of the layer
        """

        if a.sum() == 0:
            return None

        xc = self.centers[:, 0]
        yc = self.centers[:, 1]

        return np.array([np.sum(xc * a) / np.sum(a),
                         np.sum(yc * a) / np.sum(a)])

    def calc_path(self, a1: np.ndarray=None, a2: np.ndarray=None,
                  x1: np.ndarray=None, x2: np.ndarray=None,
                  max_steps: int=100, eps: float=0.1,
                  plot: bool=False, **kwargs) -> np.ndarray:

        if x1 is not None:
            # compute the activity for the source and
            # target positions
            a1 = self(x=x1)
            a2 = self(x=x2)

            true_src_pos = x1
            true_trg_pos = x2
        else:
            if a1 is None:
                raise ValueError("at least 'a1' and 'a2' must be provided")

            true_src_pos = kwargs.get("true_src_pos", None)
            true_trg_pos = kwargs.get("true_trg_pos", None)

        src_pos = self._calc_average_pos(a=a1)
        trg_pos = self._calc_average_pos(a=a2)
        path = [src_pos]
        curr_pos = src_pos.copy()
        logger.debug(f"src={np.around(src_pos, 2)}, trg={np.around(trg_pos, 2)}")

        # activation
        src_a = self(x=src_pos)
        trg_a = self(x=trg_pos)
        curr_a = src_a.copy()
        a = src_a

        # plot
        if plot:
            fig = plt.figure(figsize=(10, 5))

            # Create a GridSpec with 2 rows and 2 columns
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2])

            # Create the subplots
            ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
            ax2 = fig.add_subplot(gs[1, 0])  # Second row, first column
            ax3 = fig.add_subplot(gs[:, 1])  # First and second row, second column

        # runtime
        counter = 0
        focused = False
        distance = np.inf
        min_trg_distance = self.policy.min_trg_distance if \
            self.policy is not None else 0.01
        while distance >= min_trg_distance:

            # current position based on the population activity
            # curr_pos = self._calc_average_pos(a=curr_a)

            # --- lateral contribution ---
            drift_a = (self.weights @ curr_a.reshape(-1, 1)).flatten()

            # apply policy
            drift_a = self.policy.apply_strategy(a=drift_a)

            drift_pos = self._calc_average_pos(a=drift_a)
            lateral_move, lat_distance = calc_movement(
                            pos1=curr_pos,
                            pos2=drift_pos,
                            speed=self.policy.speed,
                            use_distance=self.policy.use_distance)

            # --- directional contribution ---
            # calculate the position
            direct_move, distance = calc_movement(
                            pos1=curr_pos,
                            pos2=trg_pos,
                            speed=self.policy.speed,
                            use_distance=self.policy.use_distance)

            # --- simple policy over `eps` ---
            # eps *= (1 + 0.1*(distance < 1) * (1 - distance))
            # eps = min((eps, 0.7))

            # --- update the position ---
            if self.policy is None:
                move = eps*direct_move + (1-eps)*lateral_move
            else:
                move = self.policy(moves=[direct_move, lateral_move],
                                   distance=distance)
            curr_pos += move
            curr_a = self(x=curr_pos)
            path += [curr_pos.tolist()]

            # calculate the distance to the target
            # in neural space
            goal = u.cosine_similarity(u=curr_a, v=trg_a)
            logger.debug(f"distance={distance:.4f}")

            counter += 1
            if counter > max_steps:
                logger.error("breaking")
                break

            if plot:
                ax1.clear()
                ax2.clear()
                ax3.clear()

                ax1.imshow(trg_a.reshape(1, -1), cmap="Greys", vmin=0,
                           aspect="auto", interpolation="nearest")
                ax1.set_title(f"target - t={counter} {distance=:.3f}")
                ax1.set_xticks([])
                ax1.set_yticks([])

                ax2.imshow(curr_a.reshape(1, -1), cmap="Greys", vmin=0,
                           aspect="auto", interpolation="nearest")
                ax2.set_title("current")
                ax2.set_xticks([])
                ax2.set_yticks([])


                if true_src_pos is not None:
                    ax3.scatter(true_src_pos[0], true_src_pos[1],
                                color='grey', marker="o", s=100, alpha=0.4)
                    ax3.scatter(true_trg_pos[0], true_trg_pos[1],
                                color='grey', marker="x", s=300, alpha=0.4)

                self.plot_online(ax=ax3, use_a=1, new_a=curr_a)
                ax3.scatter(src_pos[0], src_pos[1],
                            color='red', marker="o", s=100)
                ax3.scatter(trg_pos[0], trg_pos[1],
                            color='blue', marker="x", s=300)
                ax3.plot(np.array(path)[:, 0], np.array(path)[:, 1],
                            "g-", alpha=0.5)
                ax3.plot(np.array(path)[:, 0], np.array(path)[:, 1],
                            "gv", alpha=0.5)
                ax3.set_xlim(0., 1.)
                ax3.set_ylim(0., 1.)
                ax3.set_xticks([])
                ax3.set_yticks([])

                plt.pause(0.2)

        if plot:
            plt.show()

        return np.array(path)

    def calc_node_path2(self, a1: np.ndarray, a2: np.ndarray,
                       speed: float=0.1, max_steps: int=100) -> np.ndarray:

        src_pos = self._calc_average_pos(a=a1)
        trg_pos = self._calc_average_pos(a=a2)

        z = (a1 + a2) / 2  # this is the input to the layer

        # activation
        a = a1

        curr_idx = np.argmax(a1)
        seq_idx = [curr_idx]

        path = [src_pos]
        curr_pos = src_pos.copy()
        logger.debug(f"src={src_pos}, trg={trg_pos}")

        removed = []
        counter = 0
        focused = False
        while ((trg_pos - curr_pos)**2).sum() > 0.001:

            # get the neighbors -- idx
            neighbors = np.where(self.edges[curr_idx] > 0)[0]

            # remove already visited nodes -- idx
            neighbors = [n for n in neighbors if n not in removed]

            if len(neighbors) == 0:
                logger.error("No neighbors")
                break

            # get the best neighbor -- idx
            best_neighbor_idx = np.argmax([a[n] for n in neighbors])

            # move to the best neighbor
            if not focused:
                logger.debug(f"{best_neighbor_idx=}")
                curr_idx = neighbors[best_neighbor_idx]  # idx
                curr_trg = self.centers[curr_idx]  # pos
                logger.debug(f"curr_trg={np.around(curr_trg, 2)} [idx={curr_idx}] [best_neighbor={best_neighbor_idx}]")
            seq_idx += [curr_idx]  # idx
            # path += [self._calc_average_pos(a=a)] # pos

            # move in the direction of the best neighbor
            # with a speed of `speed`
            distance = curr_trg - curr_pos
            dx = np.array(speed * distance)
            path += [path[-1] + dx]
            curr_pos = path[-1]

            # remove the current node from the neighbors if the distance
            # to it is less than 0.01
            distance = np.sqrt((distance**2).sum())  # float
            if distance < 0.02:
                removed += [curr_idx]  # idx
                logger.debug(f"### dist={np.around(distance, 2)} [{removed}]")
                focused = False
            else:
                focused = True

            # new activation
            a = self(x=curr_pos) + a2

            logger(f"idx={curr_idx} dx={np.around(dx, 2)} [x={np.around(curr_pos, 2)}, y={np.around(curr_trg, 2)}] dist={np.around(distance, 2)}")
            # logger(f"neighbors={np.around(a[neighbors], 2)}")

            counter += 1
            if counter > max_steps:
                logger.error("breaking")
                break

        return np.array(path)

    def _centers_(self, endpoints: bool=False):

        if not endpoints:
            limx1 = self.bounds[0] + 1/(self.n+1)
            limy1 = self.bounds[2] + 1/(self.n+1)
        else:
            limx1 = self.bounds[0]
            limy1 = self.bounds[2]

        xx, yy = np.meshgrid(np.linspace(limx1, self.bounds[1],
                                         self.n, endpoint=endpoints),
                             np.linspace(limy1, self.bounds[3],
                                         self.n, endpoint=endpoints))

        self.spacing = (self.bounds[1] - limx1) / (self.n-1)

        return np.vstack([xx.ravel(), yy.ravel()]).T

    def _make_connections(self, K: int=4):

        """
        use KNN to make intra-layer connections
        """

        # max_dist = 1.3/self.n if self._endpoints else 0.8/(self.n-1)

        e, w = u.calc_knn(centers=self.centers,
                          k=self._k,
                          max_dist=3.*self.spacing,
                          return_weights=True)
        return e, w

    def plot_online(self, ax=None, use_a: bool=False,
                    new_a: np.ndarray=None):

        if ax is None:
            fig, ax = plt.subplots()
            ishow = True
        else:
            ishow = False

        a = self.a if new_a is None else new_a

        # edges
        for i, e in enumerate(self.edges):
            for j, w in enumerate(e):
                if w > 0:
                    ax.plot([self.centers[i, 0], self.centers[j, 0]],
                            [self.centers[i, 1], self.centers[j, 1]],
                            color='black',
                            alpha=min((1, a[i]/2)) if use_a else 0.5,
                            lw=0.3)
                            # lw=np.exp(-self.weights[i, j]))

        ax.scatter(self.centers[:, 0], self.centers[:, 1],
                   s=20,
                   c=a if use_a else 'grey',
                   cmap="Greys" if use_a else None,
                   alpha=0.9)
                   # edgecolors='grey' if use_a else None,)

        ax.set_xlim(0., 1.)
        ax.set_ylim(0., 1.)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        if ishow:
            plt.show()

    def plot(self, ax=None, use_a: bool=False,
             new_a: np.ndarray=None):

        if ax is None:
            fig, ax = plt.subplots()
            ishow = True
        else:
            ishow = False

        a = self.a if new_a is None else new_a

        # edges
        for i, e in enumerate(self.edges):
            for j, w in enumerate(e):
                if w > 0:
                    ax.plot([self.centers[i, 0], self.centers[j, 0]],
                            [self.centers[i, 1], self.centers[j, 1]],
                            color='black',
                            alpha=0.5,
                            lw=0.3)
                            # lw=np.exp(-self.weights[i, j]))

        ax.scatter(self.centers[:, 0], self.centers[:, 1],
                   s=20,
                   c=a if use_a else 'grey',
                   alpha=0.9,
                   edgecolors='grey')

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        if ishow:
            plt.show()

    def reset(self):

        self.a = np.zeros(self.N)


class PCNNlayer(mm.minPCNN):

    """
    a super class for the minPCNN model made to be used
    as the hard-coded PClayer
    """

    def __init__(self, pcnn_params: dict,
                 policy: object=None,
                 **kwargs):

        # check if the parameters are valid
        # assert np.sqrt(pcnn_params["N"]) % 1 == 0, "'N' must be a perfect square"

        super(PCNNlayer, self).__init__(**pcnn_params)

        self.N = pcnn_params["N"]
        self.name = kwargs.get("name", "PClayer")
        self.bounds = kwargs.get("bounds", (0, 1, 0, 1))

        # pc input layer
        self.Nj = pcnn_params.get("Nj", 100)
        self.sigma = kwargs.get("sigma", 0.004)
        self._input_layer = it.PlaceLayer(N=self.Nj,
                                          sigma=self.sigma,
                                          bounds=self.bounds)
        logger(f"{self._input_layer} - sigma={self._input_layer.sigma}")

        # other
        self.centers = None
        self.centers_finite = None
        self.finite_idx = None
        self.mask = None
        # self.nodes = np.zeros((self.N, 2))
        self.edges = []
        self.connectivity = None
        self.policy = policy

    def __repr__(self):
        return f"PCNNlayer(N={self.N}, Nj={self.Nj}, lr={self._lr})"

    def __call__(self, x: np.ndarray, frozen: bool=False,
                 **kwargs):

        """
        update the network state

        Parameters
        ----------
        x: np.ndarray
            input to the network
        frozen: bool
            if True, the network will not update
            the weights. Default is False
        """

        # pass through the input layer
        x = self._input_layer.step(position=x)

        x = x / x.sum()

        # step `u` | x : PC Nj > neurons N
        self.u = u.generalized_sigmoid(x=self._Wff @ x,
                                       alpha=self._alpha,
                                       beta=self._beta,
                                       clip_min=1e-3)
        # self.u = np.where(self.u < 2e-2, 0.0, self.u)
        self.u = np.where(self.u < 2e-3, 0.0, self.u)

        # [0, 1] normalization
        # self.u /= max((self.u.max(), self._upper_fr))

        # update modulators
        if self._is_plastic and not frozen:
            active_idx = self._update_modulators()
            self._update(x=x, idx=active_idx)

        # centers
        if self.mask is not None:
            if len(self.mask) != self._umask.sum():
                # logger.debug(f"#centers={len(self.mask)} " +\
                #     f"#stb={self._umask.sum()}")
                self._calc_graph()

        return self.u.copy()

    def _calc_centroid(self):

        """
        calculate the centroid of the layer wrt
        its centers
        """

        xc = self.centers_finite[:, 0]
        yc = self.centers_finite[:, 1]

        return np.array([np.sum(xc) / len(xc),
                         np.sum(yc) / len(yc)])

    def _calc_average_pos(self, a: np.ndarray) -> np.ndarray:

        """
        calculate the average position of the layer
        """

        a = a.flatten() * self.mask
        ao = a.copy()

        if a.sum() == 0:
            logger.warning("a.sum() == 0")
            return None

        xc = self.centers[:, 0]
        yc = self.centers[:, 1]

        # no infinities
        xc = np.where(np.isfinite(xc), xc, 0).flatten()
        yc = np.where(np.isfinite(yc), yc, 0).flatten()

        # xc = self.centers_finite[:, 0]
        # yc = self.centers_finite[:, 1]
        # a = a[self.finite_idx].flatten()

        pos = np.array([np.sum(xc * a) / np.sum(a),
                        np.sum(yc * a) / np.sum(a)])

        if pos[0] > 1 or pos[0] < 0:
            logger.debug(f"a={a.flatten()}")
            logger.debug(f"{xc=}\n{yc=}")
            logger.debug(f"nonzero ={len(self.edges)}, {a.shape=}, {xc.shape=}")
            raise ValueError(f"estimated pos={pos} over the bounds")

        if np.sum(a) == 0:
            logger.error(f"{ao=} [max {ao.max()}, max_idx {ao[self.finite_idx].max()}]")
        return pos

    def _calc_graph(self):

        """
        calculate the center and edges of the network wrt
        the physical space
        """

        # calc centers positions
        self.centers = mm.calc_centers_from_layer(wff=self._Wff,
                                        centers=self._input_layer.centers)

        # discard the infinities
        self.centers_finite = self.centers[np.isfinite(self.centers[:, 0])]
        self.finite_idx = np.where(np.isfinite(self.centers[:, 0]))[0]
        self.mask = np.zeros(self.N)
        self.mask[self.finite_idx] = 1.

        self.edges, self.connectivity = mm.make_edge_list(M=self.W_rec,
                                                          centers=self.centers)

    def calc_path(self, x1: np.ndarray, x2: np.ndarray,
                  max_steps: int=100,
                  plot: bool=False, **kwargs) -> np.ndarray:

        # compute the activity for the source and
        # target positions
        src_a = self(x=x1, frozen=True)
        trg_a = self(x=x2, frozen=True)

        true_src_pos = x1
        true_trg_pos = x2

        # calculate the average position of the source and target
        # from the population activity
        src_pos = self._calc_average_pos(a=src_a)
        trg_pos = self._calc_average_pos(a=trg_a)
        if src_pos[0] == -np.inf:
            src_pos = self._calc_centroid()
            logger.warning(f"[centroid] src_pos={src_pos}")
        path = [src_pos]
        curr_pos = src_pos.copy()
        logger.debug(f"src={np.around(src_pos, 2)}," + \
            f" trg={np.around(trg_pos, 2)}")

        # calculate back the activity for the source and target
        # positions | this is done only for 'realism'
        src_a = self(x=src_pos)
        trg_a = self(x=trg_pos)
        pos2_trg = self._calc_average_pos(a=trg_a)
        curr_a = src_a.copy()
        a = src_a

        # plot
        if plot:
            fig = plt.figure(figsize=(10, 5))

            # Create a GridSpec with 2 rows and 2 columns
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2])

            # Create the subplots
            ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
            ax2 = fig.add_subplot(gs[1, 0])  # Second row, first column
            ax3 = fig.add_subplot(gs[:, 1])  # First and second row, second column

        # runtime
        counter = 0
        focused = False
        distance = np.inf
        eps = 0.1
        min_trg_distance = self.policy.min_trg_distance if \
            self.policy is not None else 0.01

        def exit_condition(distance: float,
                           min_trg_distance: float) -> bool:
            if self.policy.use_distance:
                return distance >= min_trg_distance
            return True

        while exit_condition(distance, min_trg_distance):

            # logger(f"[{counter}]")

            # --- lateral contribution ---
            # drift_a = (self.W_rec @ curr_a.reshape(-1, 1)).flatten()
            drift_a = forward_pc(W=self.W_rec, x=curr_a.reshape(-1, 1),
                                 alpha=self._alpha,
                                 beta=self._beta, maxval=1.)

            # apply policy
            drift_a = self.policy.apply_strategy(a=drift_a,
                                                   mask=self.finite_idx)

            drift_pos = self._calc_average_pos(a=drift_a)
            drift_move, lat_distance = calc_movement(
                            pos1=curr_pos,
                            pos2=drift_pos,
                            speed=self.policy.speed,
                            use_distance=self.policy.use_distance)

            # --- directional contribution ---
            direct_move, distance = calc_movement(
                            pos1=curr_pos,
                            pos2=trg_pos,
                            speed=self.policy.speed,
                            use_distance=self.policy.use_distance)

            # --- simple policy over `eps` ---
            # eps *= (1 + 0.1*(distance < 1) * (1 - distance))
            # eps = min((eps, 0.7))

            #
            if np.isnan(drift_move).any():
                logger.debug(f"drift_a:\n{drift_a.flatten()}")
                logger.debug(f"drift_pos:\n{drift_pos.flatten()}")
                logger.debug(f"activation:\n{curr_a.flatten()}")

            # --- update ---
            if self.policy is None:
                move = eps*direct_move + (1-eps)*drift_move
            else:
                move = self.policy(moves=[direct_move, drift_move],
                                   distance=distance)
            curr_pos += move

            # check boundaries
            curr_pos = self.policy.check_wall(pos=curr_pos,
                                              bounds=self.bounds)

            curr_a = self(x=curr_pos)
            path += [curr_pos.tolist()]

            # --- log ---
            if self.policy.use_distance:
                distance = round(distance, 4)
                logger(f"pos={np.around(curr_pos, 2)}," + \
                    f" distance={distance:.4f}")
            else:
                distance = np.inf
                logger(f"pos={np.around(curr_pos, 2)} " + \
                    f" highest_a={curr_a.max():.4f}")

            # ---
            counter += 1
            if counter > max_steps:
                logger.error("breaking")
                break

            # --- plot ---
            if plot:
                ax1.clear()
                ax2.clear()
                ax3.clear()

                ax1.imshow(self._Wff, cmap="viridis", vmin=0,
                           aspect="auto", interpolation="nearest")
                # ax1.imshow(trg_a.reshape(1, -1), cmap="Greys", vmin=0,
                #            aspect="auto", interpolation="nearest")
                ax1.set_title(f"Wff - t={counter} {distance=}")
                ax1.set_xticks([])
                ax1.set_yticks([])

                ax2.imshow(curr_a.reshape(1, -1), cmap="Greys", vmin=0,
                           aspect="auto", interpolation="nearest")
                ax2.set_title("current")
                ax2.set_xticks([])
                ax2.set_yticks([])

                self.plot_online(ax=ax3, use_a=False, alpha=0.1)
                self.plot_online(ax=ax3, use_a=True,
                                 new_a=curr_a.flatten().tolist())

                if self.policy.isgoal:
                    ax3.scatter(src_pos[0], src_pos[1],
                                color='red', marker="o", s=100)
                    ax3.scatter(trg_pos[0], trg_pos[1],
                                color='blue', marker="x", s=300)

                    if true_src_pos is not None:
                        ax3.scatter(true_src_pos[0], true_src_pos[1],
                                    color='grey', marker="o", s=100,
                                    alpha=0.4)
                        ax3.scatter(true_trg_pos[0], true_trg_pos[1],
                                    color='grey', marker="x", s=300,
                                    alpha=0.4)

                ax3.plot(np.array(path)[:, 0], np.array(path)[:, 1],
                            "g-", alpha=0.1, lw=2.)
                # ax3.plot(np.array(path)[:, 0], np.array(path)[:, 1],
                #             "gv", alpha=0.5)
                ax3.scatter(np.array(path)[-1, 0], np.array(path)[-1, 1],
                            color='green', marker="d", s=80,
                            alpha=0.5)
                ax3.set_xlim(0., 1.)
                ax3.set_ylim(0., 1.)
                ax3.set_xticks([])
                ax3.set_yticks([])
                ax3.set_title(f"N={len(self.finite_idx)}," +\
                    f" DA={self._DA:.2f}, ACh={self._ACh:.2f}")

                plt.pause(0.001)

        # if plot:
        #     plt.show()

        return np.array(path)

    def generate_local_position(self) -> tuple:

        """
        generate a random position based on the
        visited locations (from the neurons centers).
        The two-steps estimates is necessary since
        the nodes at the boundaries might lead to an
        imprerfect estimate
        """

        # from a random node to a population vector
        pos = self.centers[np.random.choice(self.finite_idx)]

        # population vector
        a = self(x=pos)

        # estimate the position
        pos = self._calc_average_pos(a=a)

        a = self(x=pos)

        # effective position
        pos_virtual = self._calc_average_pos(a=a)

        return pos, pos_virtual

    def plot_online(self, ax=None, use_a: bool=False,
                    new_a: np.ndarray=None, edges: bool=True,
                    alpha: float=0.9):

        if ax is None:
            fig, ax = plt.subplots()
            ishow = True
        else:
            ishow = False

        a = self.u.flatten().tolist() if new_a is None else new_a

        # edges
        if edges:
            for i, e in enumerate(self.connectivity):
                for j, w in enumerate(e):
                    if w > 0:
                        ax.plot([self.centers[i, 0], self.centers[j, 0]],
                                [self.centers[i, 1], self.centers[j, 1]],
                                color='black',
                                alpha=min((1, a[i]/2)) if use_a else 0.5,
                                lw=0.3)
                            # lw=np.exp(-self.weights[i, j]))

        ax.scatter(self.centers[:, 0], self.centers[:, 1],
                   s=20,
                   c=a if use_a else 'grey',
                   cmap="Greys" if use_a else None,
                   alpha=alpha)
                   # edgecolors='grey' if use_a else None,)

        ax.set_xlim(0., 1.)
        ax.set_ylim(0., 1.)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        if ishow:
            plt.show()

    def render(self, use_a: bool=False, new_a: np.ndarray=None,
               alpha: float=0.9):

        a = self.u if new_a is None else new_a

        # edges
        for i, e in enumerate(self.connectivity):
            for j, w in enumerate(e):
                if w > 0:
                    plt.plot([self.centers[i, 0], self.centers[j, 0]],
                             [self.centers[i, 1], self.centers[j, 1]],
                             color='black',
                             alpha=0.2,
                             lw=0.3)

        plt.scatter(self.centers[:, 0], self.centers[:, 1],
                    s=20,
                    c=a if use_a else 'grey',
                    alpha=alpha,
                    cmap="Greys" if use_a else None)
                    # edgecolors='grey')

    def plot(self, ax=None, use_a: bool=False,
             new_a: np.ndarray=None, save: bool=False,
             path: str=None, ishow: bool=True, is_env: bool=False):

        if ax is None:
            fig, ax = plt.subplots()
        else:
            ishow = False

        a = self.u if new_a is None else new_a

        # edges
        for i, e in enumerate(self.connectivity):
            for j, w in enumerate(e):
                if w > 0:
                    ax.plot([self.centers[i, 0], self.centers[j, 0]],
                            [self.centers[i, 1], self.centers[j, 1]],
                            color='black',
                            alpha=0.5,
                            lw=0.3)
                            # lw=np.exp(-self.weights[i, j]))

        ax.scatter(self.centers[:, 0], self.centers[:, 1],
                   s=20,
                   c=a if use_a else 'grey',
                   alpha=0.9,
                   edgecolors='grey')

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        if ishow:
            plt.show()

        if save:
            if path is None:
                path = r"/cache/layer.png"
            fig.savefig(path)

    def save_params(self, path: str=None):

        """
        save the parameters of the network
        """

        params = {
            "N": self.N,
            "Nj": self.Nj,
            "alpha": self._alpha,
            "beta": self._beta,
            "lr": self._lr,
            "threshold": self._threshold,
            "tau_da": self._tau_da,
            "eq_da": self._eq_da,
            "tau_ach": self._tau_ach,
            "eq_ach": self._eq_ach,
            "ach_threshold": self._ach_threshold,
            "da_threshold": self._da_threshold,
            "epsilon": self._epsilon,
            "rec_epsilon": self._rec_epsilon,
            "k_neighbors": self._k_neighbors,
            "Wff": self._Wff.tolist(),
            "W_rec": self.W_rec.tolist(),
            "umask": self._umask.tolist(),
            "bounds": self.bounds,
            "sigma": self.sigma
        }

        if path is None:
            path = SAVE_PATH + "params.json"

        logger.debug(f"{os.getcwd()}")
        with open(path, "w") as f:
            json.dump(params, f)

        self.plot(save=True, path=f"{SAVE_PATH}pcnn_layer.png", ishow=False)

        logger(f"Parameters saved to {path}")

    def load_params(self, path: str=None):

        """
        load the parameters of the network
        """

        if path is None:
            path = SAVE_PATH + "params.json"

        with open(path, "r") as f:
            params = json.load(f)

        self.N = params["N"]
        self.Nj = params["Nj"]
        self._alpha = params["alpha"]
        self._beta = params["beta"]
        self._lr = params["lr"]
        self._threshold = params["threshold"]
        self._tau_da = params["tau_da"]
        self._eq_da = params["eq_da"]
        self._tau_ach = params["tau_ach"]
        self._eq_ach = params["eq_ach"]
        self._ach_threshold = params["ach_threshold"]
        self._da_threshold = params["da_threshold"]
        self._epsilon = params["epsilon"]
        self._rec_epsilon = params["rec_epsilon"]
        self._k_neighbors = params["k_neighbors"]
        self._Wff = np.array(params["Wff"])
        self._umask = np.array(params["umask"])
        self.W_rec = np.array(params["W_rec"])
        self.bounds = params["bounds"]
        self.sigma = params["sigma"]

        self.u = np.zeros(self.N)
        self._indexes = np.arange(self.N)
        self._adaptive_threshold = np.zeros(self.N) + \
            self._threshold
        self.rps = np.zeros((self.N, 1))

        # make new input pc layer
        self._input_layer = it.PlaceLayer(N=self.Nj,
                                          sigma=self.sigma,
                                          bounds=self.bounds)

        self._calc_graph()

        logger(f"Parameters loaded from {path}")



""" Stack of PC layers """


class Layers:

    def __init__(self, N: list, S: list, K: list, Ed: list=None,
                 bounds: tuple=(0, 1, 0, 1)):

        self.N = N
        self.S = S
        Ed = Ed if Ed is not None else [False] * len(N)
        self.bounds = bounds
        self._layers = [
            PClayer(n=N[i], sigma=S[i], k=K[i],
                    endpoints=Ed[i],
                    name=f"PClayer_{i+1}",
                    bounds=bounds) \
            for i in range(len(N))
        ]
        self.len = len(self._layers)
        self.w = []

        self.edges = []
        self.weights = []
        self.edges_w = []
        self.connections = []
        self.indexes = []
        for i in range(len(self._layers)-1, 0, -1):
            c, e, w, ew, ix = intra_layer_connections(
                nodes1=self._layers[i].centers,
                nodes2=self._layers[i-1].centers,
                max_dist=2.*self._layers[i].sigma)
                # max_dist=1./N[i])

            self.edges.append(e)
            self.weights.append(w)
            self.edges_w.append(ew)
            self.connections.append(c)
            self.indexes.append(ix)

    def __repr__(self):
        return f"Layers(" + ", ".join([f"(N={n}, S={s})" \
            for n, s in zip(self.N, self.S)]) + ")"

    def __len__(self):
        return len(self._layers)

    def __call__(self, x: np.ndarray):

        self.a = [
            layer(x) for layer in self._layers
        ]
        return self.a.copy()

    def expand_positions(self, x: np.ndarray):

        """
        given a position `x` all layers get activated,
        then the top layers project downstream
        """

        # generalized forward pass
        a_fwd = self(x=x)

        # project activations downstream
        z = [self.a[-1]]
        for i in range(self.len-1, 0, -1):
            layer = self._layers[i-1]
            y = self.weights[self.len-i-1] @ self.a[i]
            z += [layer.forward(z=y)]

        # record in `a`
        # z += [self.a[-1]]
        self.a = []
        for k in range(self.len-1, -1, -1):
            self.a += [np.around(z[k], 2)]

        return self.a, a_fwd

    def calc_path(self, x1: np.ndarray, x2: np.ndarray,
                  speed: float=0.1, max_steps: int=100,
                  plot: bool=False):

        """
        calculate the path from x1 to x2
        """

        a1, a_fwd_1 = self.expand_positions(x=x1)
        self.reset()
        a2, a_fwd_2 = self.expand_positions(x=x2)

        path = self._layers[0].calc_path(a1=a_fwd_1[0], a2=a_fwd_2[0],
                                              speed=speed,
                                              max_steps=max_steps,
                                              plot=plot,
                                              true_src_pos=x1,
                                              true_trg_pos=x2)

        return path

    def plot(self, use_a: bool=False,
             axs=None, trajectory: np.ndarray=None,
             new_a: np.ndarray=None, showaxes: bool=False):

        if axs is None:
            fig, axs = plt.subplots(1, len(self._layers))
            ishow = True
        else:
            ishow = False

        for i, layer in enumerate(self._layers):
            # plot trajectory
            if trajectory is not None:
                axs[i].plot(trajectory[:, 0], trajectory[:, 1],
                        color='red', lw=0.5, alpha=0.5,
                        label="trajectory")

            layer.plot_online(ax=axs[i], use_a=use_a,
                              new_a=new_a[i] if new_a is not None else None)

        if ishow:
            plt.show()

    def plot_graph(self, use_a: bool=False):

        """
        plot the edges between layers
        """

        # us
        # colors = plt.cm.rainbow(np.linspace(0, 1, self.len))
        colors = plt.cm.winter(np.linspace(0, 1, self.len))
        makers = ["o", "v", "s", "p"][:self.len]

        # plot nodes
        fig, axs = plt.subplots(1, self.len-1)
        ishow = True

        for i, ax in enumerate(axs):

            # plot nodes i
            ax.scatter(self._layers[self.len-i-1].centers[:, 0],
                       self._layers[self.len-i-1].centers[:, 1],
                       s=70,
                       c=self._layers[self.len-i-1].a if use_a else colors[self.len-i-1],
                       alpha=0.3,
                       marker=makers[self.len-i-1],
                       label=f"Layer {i}")

            # plot nodes i+1
            ax.scatter(self._layers[self.len-i-2].centers[:, 0],
                       self._layers[self.len-i-2].centers[:, 1],
                       s=30,
                       c=self._layers[self.len-i-2].a if use_a else colors[self.len-i-2],
                       marker=makers[self.len-i-2],
                       alpha=0.7,
                       label=f"Layer {i+1}")

            # plot edges
            for l, e in enumerate(self.edges[i]):
                ax.plot([e[0], e[2]], [e[1], e[3]],
                        color='black',
                        alpha=0.8,
                        linestyle='--' if l % 2 == 0 else '-',
                        lw=0.5)

            ax.set_xlim(self.bounds[0], self.bounds[1])
            ax.set_ylim(self.bounds[2], self.bounds[3])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            ax.legend(loc="lower right")

        plt.show()

    def plot_graph_3d(self, use_a: bool=False,
                      trajectory: np.ndarray=None,
                      ax=None, show_edges: bool=False,
                      layer_idx: int=None):

        """
        Plot the layers in 3D with connections projected to the bottom layer.
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ishow = True
        else:
            ishow = False

        # Plot trajectory
        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                    np.zeros(trajectory.shape[0]),
                    color='red', lw=0.5, alpha=0.5,
                    label="trajectory")

        # Define colors and markers for each layer
        # colors = ["red", "green", "blue", "orange"][:self.len]
        colors = plt.cm.winter(np.linspace(0, 1, self.len))
        markers = ["o", "v", "s", "p"][:self.len]

        # Plot each layer at different heights
        for i, layer in enumerate(self._layers):
            z = np.full(layer.centers.shape[0], i)  # Height of the layer
            ax.scatter(layer.centers[:, 0], layer.centers[:, 1], z,
                       s=20+30*i,
                       color=[colors[i][:-1]]*layer.N if not use_a else layer.a, 
                       cmap="Greys",
                       # alpha=self.a[i].reshape(-1) if use_a else 0.7,
                       # edgecolors=colors[i] if show_edges else None,
                       # linewidths=0.5 if show_edges else 0.0,
                       marker=markers[i],
                       label=f"Layer {i}")

        # Plot connections to the bottom layer
        if show_edges:
            for i in range(len(self._layers) - 1):
                if layer_idx is not None:
                    if i != layer_idx:
                        continue
                for k, (edge, edge_w) in enumerate(zip(self.edges[i],
                                        self.edges_w[i])):
                    x = [edge[0], edge[2]]
                    y = [edge[1], edge[3]]
                    z = [self.len - i - 1, self.len - i - 2]
                    wa = self._layers[i].a[self.indexes[i][k]]/1.5 if use_a else 0.5
                    ax.plot(x, y, z,
                            # color=colors[self.len-i-1],
                            color='grey',
                            alpha=min((wa, 1.)),
                            linestyle='-', lw=1.)

        # ax.set_xlim(-0.2, 1.2)
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_zlim(-0.2, self.len)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Layer')
        ax.set_axis_off()

        ax.legend(loc="lower right")

        if ishow:
            plt.show()

    def reset(self):

        for layer in self._layers:
           layer.reset()


def intra_layer_connections(nodes1: np.ndarray, nodes2: np.ndarray,
                            max_dist: float=1.0, k: int=None):

    connectivity = np.zeros((len(nodes2), len(nodes1)))
    edges = []
    weights = np.zeros((len(nodes2), len(nodes1)))
    edges_w = []
    indexes = []

    for j, node1 in enumerate(nodes1):
        edges_i = []
        edges_wi = []
        indexes_i = []
        for i, node2 in enumerate(nodes2):
            w = np.linalg.norm(node1 - node2)
            if w < max_dist:
                connectivity[i, j] = 1
                weights[i, j] = np.exp(-w)
                edges_i += [node1.tolist() + node2.tolist()]
                edges_wi += [weights[i, j]]
                indexes_i += [j]
                # edges_i += [[node1[0], node1[1],
                #              node2[0]*np.random.normal(1, 0.02),
                #              node2[1]*np.random.normal(1, 0.02)]]
                # print(f"{node1} -> {node2} : {edges_i[-1]}")

        edges += edges_i
        edges_w += edges_wi
        indexes += indexes_i

    # logger.debug(edges)
    edges = np.array(edges)

    return connectivity, edges, weights, edges_w, indexes


""" policy """


class Policy:

    def __init__(self, lambdas: list, isgoal: bool, min_dx: float=0.01,
                 speed: float=1.0, **kwargs):

        self.lambdas = lambdas
        self.lambdas_zero = lambdas.copy()
        self.min_dx = min_dx
        self.min_trg_distance = kwargs.get("min_trg_distance", 0.01)
        self.speed = speed
        self.speed_base = speed
        self.speed_max = kwargs.get("speed_max", 0.1)

        self.isgoal = None  # ignore it
        self.strategy = kwargs.get("strategy", "max")
        self.k = kwargs.get("k", 3)
        self.threshold = kwargs.get("threshold", 0.001)

        # local target
        self._u = 0
        self._tau = kwargs.get("tau", 1)
        self.current_move = None

        # variable strategy
        # . negative: enhance the most active neurons
        # . zero: no change
        # . =1: all neurons have the same activity [mean]
        # . >1: enhance the least active neurons
        self.beta = kwargs.get("beta", 0.)

        self.vector = [0., 0.]

    def __repr__(self):
        return f"Policy(lambdas=({np.around(self.lambdas, 3)}," + \
            f" speed={self.speed}, goal={self.isgoal})"

    def __call__(self, moves: list, distance: float) -> np.ndarray:

        """
        calculate the policy
        """

        # update the policy, if there's a goal
        # self._update_policy(distance=distance)

        # keep the same plan, if possible  | no plan
        # move = self._keep_plan()
        # if move is not None:
            # return move

        # calculate the new move
        self.current_move = np.zeros(2)
        for i, (l, m) in enumerate(zip(self.lambdas, moves)):
            self.current_move += l * m

        # self.current_move = np.where(np.abs(self.current_move) < \
        #                              self.min_dx,
        #                 np.sign(self.current_move) * self.min_dx,
        #                 self.current_move)
        # make it a unit vector
        self.current_move = self.current_move / np.linalg.norm(self.current_move) * self.speed

        # logger.debug(f"policy_move={self.current_move} [{self.speed}]")

        # check
        if np.isnan(self.current_move).any():
            raise ValueError(f"policy_move={self.current_move}" + \
                f" {moves=}, {distance=}")

        # explicitly calculate angle and magnitude
        self.vector = [
            np.degrees(np.arctan2(self.current_move[1], self.current_move[0])),
            np.linalg.norm(self.current_move)
        ]

        return self.current_move

    def _update_policy(self, distance: float):  # ignore it

        if self.isgoal:
            self._attraction_policy(distance=distance)

    def _attraction_policy(self, distance: float):  # ignore it

        """
        update the lambda values
        """

        attraction = u.generalized_sigmoid(x=distance,
                                           alpha=0.3,
                                           beta=-10.,
                                           gamma=0.5)

        self.lambdas[0] = min((self.lambdas_zero[0] + attraction, 0.99))
        self.lambdas[1] = 1 - self.lambdas[0]

    def set_parameters(self, action: np.ndarray):

        """
        action: [lambda, beta]
        """

        self.lambdas[0] = action[0]
        self.lambdas[1] = 1 - action[0]
        self.beta = action[1]

        # speed
        self.speed = action[3] * self.speed_base + (1 - action[3]) * self.speed_max

    def set_strategy(self, strategy: str):
            self.strategy = strategy

    def check_wall(self, pos: np.ndarray, bounds: tuple) -> np.ndarray:

        """
        revert the current move policy
        """

        if pos[0] <= bounds[0] or pos[0] >= bounds[1]:
            self.current_move[0] *= -1
            pos += self.current_move
        elif pos[1] <= bounds[2] or pos[1] >= bounds[3]:
            self.current_move[1] *= -1
            pos += self.current_move
        return pos

    @property
    def use_distance(self):
        return self.isgoal

    def _keep_plan(self):

        # update
        self._u -= self._u / self._tau

        # keep the same target
        if self._u > 0.1:
            return self.current_move

        # make a new target
        self._u = 1
        return None

    def apply_strategy(self, a: np.ndarray,
                         **kwargs) -> np.ndarray:

        """
        given an activity population vector, it changes it
        such to bias certain neurons (directions in neural space)

        Parameters
        ----------
        a: np.ndarray
            a population vector

        Returns
        -------
        np.ndarray
            a population vector with the bias
        """

        # just select the most active neuron
        if self.strategy == "max":
            return np.where(a == a.max(), a, 0)

        # threshold to the top-k most active neurons
        elif self.strategy == "topk":
            topk = np.argsort(a)[-self.k:]
            return np.where(np.isin(np.arange(len(a)), topk), a, 0)

        # select the least active non-zero neurons
        elif self.strategy == "min":
            min_nonzero = np.where(a > self.threshold,
                                   a, 0).argmin()
            return np.where(np.arange(len(a)) == min_nonzero, 1, 0)

        # threshold to the top-k least active neurons
        elif self.strategy == "bottomk":
            a *= np.random.normal(1, 1, a.shape)  # add noise
            idx = a[a > self.threshold].argsort()[:self.k]
            a[~idx] = 0
            return a

        elif self.strategy == "variable":
            nonzero_idx = np.where(a > self.threshold)[0]
            v = a[nonzero_idx]
            v = v + (v.mean() - v) * self.beta
            a[nonzero_idx] = v
            # return np.maximum(a, 0.)
            return a

        # select a random neuron
        mask = kwargs.get("mask", None)

        if mask is not None:
            idx = np.random.choice(mask)
        else:
            idx = np.random.randint(0, len(a))
        return np.where(np.arange(len(a)) == idx, 1, 0)



""" local utils """


def calc_movement(pos1: np.ndarray, pos2: np.ndarray,
                  speed: float, use_distance: bool=True) -> np.ndarray:

    """
    Calculate a movement in space (the reference ambient space)
    given two population vectors [activations]
    """

    if pos2 is None or pos1 is None:
        return np.array([0, 0]), None
    elif pos2[0] - pos1[0] == 0:
        return np.array([0, 0]), None

    # Calculate the angle between the two vectors
    angle = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])

    # ensure angle is in range [0, 2π]
    if angle < 0:
        angle += 2 * np.pi

    #
    angle_deg = np.degrees(angle)

    # Calculate the distance between the two positions
    if use_distance:
        distance = np.linalg.norm(pos1 - pos2)

        # make speed proportional to the distance
        speed = distance/2 if speed > distance else speed
    else:
        distance = np.inf

    # Calculate the movement as a new position
    # with a speed of `speed`
    dx = speed * np.cos(angle)
    dy = speed * np.sin(angle)

    return np.array([dx, dy]), distance


@jit(nopython=True)
def forward_pc(W: np.ndarray, x: np.ndarray,
               alpha: float, beta: float,
               maxval: 1.) -> np.ndarray:

    """
    forward pass of the PC layer
    """

    # min-max normalization in [0, maxval]
    # x = (x - x.min()) / (x.max() - x.min()) * maxval
    # x = x / x.sum()

    z = u.generalized_sigmoid(x=W @ x,
                              alpha=alpha,
                              beta=beta,
                              clip_min=1e-3)
    z = np.where(z < 2e-3, 0.0, z)

    return z


def boundary_conditions(x: np.ndarray, bounds: tuple) -> np.ndarray:

    """
    apply boundary conditions to the position
    """

    x = np.where(x < bounds[0], bounds[0], x)
    x = np.where(x > bounds[1], bounds[1], x)

    return x


def calc_average_pos(self, a: np.ndarray,
                     centers: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:

    """
    calculate the average position of the layer
    """


    a = a.flatten() * mask
    ao = a.copy()

    if a.sum() == 0:
        logger.warning("a.sum() == 0")
        return None

    xc = centers[:, 0]
    yc = centers[:, 1]

    # no infinities
    xc = np.where(np.isfinite(xc), xc, 0).flatten()
    yc = np.where(np.isfinite(yc), yc, 0).flatten()

    # weighted average
    pos = np.array([np.sum(xc * a) / np.sum(a),
                    np.sum(yc * a) / np.sum(a)])

    if pos[0] > 1 or pos[0] < 0:
        logger.debug(f"a={a.flatten()}")
        logger.debug(f"{xc=}\n{yc=}")
        raise ValueError(f"estimated pos={pos} over the bounds")

    return pos


class Environment:

    def __init__(self, model: object, **kwargs):

        self.model = model

        # environment variables
        self.bounds = kwargs.get("bounds", (0, 1, 0, 1))

        # episode variables
        self.curr_pos = None
        self.trg_pos = None

    def _one_step(self, action: np.ndarray, curr_pos: np.ndarray,
                  trg_pos: np.ndarray) -> tuple:

        # --- init
        curr_a = self.model(x=curr_pos)

        # estimate the target position from its representation
        # in neural space
        trg_a = self(x=trg_pos)
        trg_pos = self.model._calc_average_pos(a=trg_a)

        # start
        if action is None:
            return curr_pos, curr_a, trg_a, None

        # update policy
        model.policy.set_parameters(action=action)

        # --- drift
        drift_a = forward_pc(W=self.model.W_rec,
                             x=curr_a.reshape(-1, 1),
                             alpha=self.model._alpha,
                             beta=self.model._beta, maxval=1.)
        drift_a = model.policy.apply_strategy(a=drift_a,
                                                mask=self.model.finite_idx)
        drift_pos = model._calc_average_pos(a=drift_a)
        drit_move, _ = calc_movement(pos1=curr_pos,
                                     pos2=drift_pos,
                                     speed=self.model.policy.speed,
                                     use_distance=self.model.policy.use_distance)

        # --- direct
        direct_move, distance = calc_movement(pos1=curr_pos,
                                              pos2=trg_pos,
                                              speed=self.model.policy.speed,
                                              use_distance=self.model.policy.use_distance)

        # --- update position
        curr_pos += move

        # check boundaries
        curr_pos = self.model.policy.check_wall(pos=curr_pos,
                                                bounds=self.model.bounds)

        # new representation of the position
        curr_a = self.model(x=curr_pos)

        return curr_pos, curr_a, trg_a, distance

    def _calc_reward(self, curr_pos: np.ndarray, trg_pos: np.ndarray) -> float:

        """
        calculate the reward
        """

        distance = np.linalg.norm(curr_pos - trg_pos)
        reward = 1.0 if distance < self.model.policy.min_trg_distance else 0.0

        return reward

    def run(self, agent: object, mode: str="target", **kwargs):

        assert hasattr(agent, "step"), \
            "agent must have a `step` method"
        max_steps = kwargs.get("max_steps", 200)

        # init
        self.reset()

        pos, obs1, obs2, distance = self._one_step(action=None,
                                                   curr_pos=self.curr_pos,
                                                   trg_pos=self.trg_pos)

        # run
        for t in range(max_steps):

            # merge observations
            obs = np.concatenate([obs1, obs2])

            # agent step
            action = agent.step(obs=obs)

            # environment step
            self.curr_pos, obs1, obs2, distance = \
                self._one_step(action=action,
                               curr_pos=self.curr_pos,
                               trg_pos=self.trg_pos)

            # reward
            reward = self._calc_reward(curr_pos=self.curr_pos,
                                       trg_pos=self.trg_pos)

            # check
            if reward == 1.0:
                logger.info("reached the target")
                break

        return reward

    def reset(self):

        self.curr_pos = np.random.uniform(
            self.bounds[0], self.bounds[1], 2)
        self.trg_pos = np.random.uniform(
            self.bounds[0], self.bounds[1], 2)


def make_default_model():

    # policy
    policy = Policy(lambdas=[0.3, 0.5],
                    min_dx=0.001,
                    speed=0.005,
                    strategy="max",
                    k=3,
                    threshold=0.001,
                    min_trg_distance=0.01,
                    isgoal=True,
                    tau=5,
                    beta=-2.)
    logger(policy)

    """ pcnn """

    N = 50
    Nj = 12**2
    sigma = 0.005

    params = {
        "N": N,
        "Nj": Nj,
        "tau": 10.0,
        "alpha": 0.19,
        "beta": 20.0, # 20.0
        "lr": 0.1,
        "threshold": 0.2,
        "ach_threshold": 0.2,
        "da_threshold": 0.5,
        "tau_ach": 100.,  # 2.
        "eq_ach": 1.,
        "tau_da": 2.,  # 2.
        "eq_da": 0.,
        "epsilon": 0.01,
        "rec_epsilon": np.exp(-8),
        "k_neighbors": None,
    }
    # *rec_epsilon*
    # low: more connections
    # exp(-x): x=4 -> 0.0183, x=5 -> 0.0067, x=6 -> 0.0025
    #          x=7 -> 0.0009, x=8 -> 0.0003, x=9 -> 0.0001

    model = PCNNlayer(pcnn_params=params,
                      policy=policy)

    return model


""" main methods """


def run_trajectory(layers: Layers, is3d: bool=False):

    # make trajectory
    duration = 60
    _, trajectory = u.make_any_walk(duration=duration)

    num_steps = len(trajectory)

    # online roaming
    if is3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, axs = plt.subplots(1, layers.len)

    # run
    for t in range(num_steps):

        # _ = layers.expand_positions(x=trajectory[t])
        _ = layers(x=trajectory[t])

        if t % 20 == 0:

            if not is3d:
                for i, ax in enumerate(axs):
                    ax.clear()
                    ax.set_title(f"Layer {i}")
                #     ax.plot([0, 1], [0, 0], [0, 0], color='black', lw=0.5)
                #     ax.plot([0, 1], [1, 1], [0, 0], color='black', lw=0.5)
                #     ax.plot([0, 0], [0, 1], [0, 0], color='black', lw=0.5)
                #     ax.plot([1, 1], [0, 1], [0, 0], color='black', lw=0.5)
                layers.plot(use_a=True,
                            trajectory=trajectory[:t],
                            axs=axs)
            else:
                ax.clear()
                layers.plot_graph_3d(use_a=True,
                                     trajectory=trajectory[:t],
                                     ax=ax, show_edges=True)

                # plot a square in the box (0, 1, 0, 1) with 4 lines
                ax.plot([0, 1], [0, 0], [0, 0], color='black', lw=0.5)
                ax.plot([0, 1], [1, 1], [0, 0], color='black', lw=0.5)
                ax.plot([0, 0], [0, 1], [0, 0], color='black', lw=0.5)
                ax.plot([1, 1], [0, 1], [0, 0], color='black', lw=0.5)

            plt.pause(0.001)

    plt.show()


def test_calc_path(model: object, max_steps: int=200,
                   trg: np.ndarray=None, src: np.ndarray=None,
                   axs=None, online_plot: bool=False, plot: bool=False,
                   eps: float=0.1, **kwargs):

    if plot:
        if axs is None:
            fig, axs = plt.subplots()
            ishow = True
        else:
            ishow = False

    if src is None:
        src = np.array([0.1, 0.1])
    if trg is None:
        trg = np.around(np.random.uniform(0.4, 0.99, 2), 2)
    # src = np.around(np.random.rand(2), 2)
    # trg = np.array([0.9, 0.1])
    logger.info(f"true: src={src.tolist()}, trg={trg.tolist()}")

    # run
    path = model.calc_path(x1=src,
                           x2=trg,
                           eps=eps,
                           max_steps=1000,
                           plot=online_plot,
                           **kwargs)

    # plot
    if plot:
        axs.scatter(src[0], src[1], color="red", alpha=0.9,
                    marker="o", s=100, label="start")
        axs.scatter(trg[0], trg[1], color="blue", alpha=0.9,
                    marker="x", s=300, label="target")
        axs.plot(path[:, 0], path[:, 1], "gv", alpha=0.5)
        axs.plot(path[:, 0], path[:, 1], "g-", alpha=0.3)
        axs.set_xlim(0., 1.)
        axs.set_ylim(0., 1.)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_aspect('equal')

        axs.legend(loc="lower right")

        if ishow:
            plt.show()


def main_one():

    # N = [9, 5, 2]
    N = [18, 10, 4]

    # model = Layers(N=N,
    #                S=[0.15, 0.2, 0.4],
    #                K=[5, 5, 5],
    #                Ed=[True, True, True],
    #                bounds=(-1, 2, -1, 2))

    policy = Policy(lambdas=[0.2, 0.8],
                    min_dx=0.001,
                    speed=0.1,
                    strategy="random",
                    k=3,
                    threshold=0.001)
    logger(policy)

    model = PClayer(n=9, sigma=0.15, k=5, endpoints=True,
                    policy=policy)
    logger(model)

    """ connectivity graph """
    # model.plot_graph_3d(show_edges=True, use_a=False, layer_idx=None)  # 3d connectivity
    # model.plot_graph()  # 2d inter-connectivity
    # model.plot()  # 2d connectivity

    """ misc """
    # model(x=np.random.rand(2))

    # z = model.expand_positions(x=np.random.rand(2))

    # for ai in model.a:
    #     logger.debug(f"{np.around(ai, 3)}")

    # model.plot_graph_3d(use_a=True)

    """ main run """

    # --- 1
    # run_trajectory(model, is3d=False)

    # --- 2
    test_calc_path(model,
                   eps=0.01,
                   max_steps=400,
                   online_plot=True,
                   plot=False,
                   min_distance=0.01)

    # --- 3
    # fig, axs = plt.subplots(1, 1)
    # for _ in range(10):
    #     axs.clear()
    #     test_calc_path(model,
    #                    speed=0.3,
    #                    max_steps=400,
    #                    axs=axs)
   #     plt.pause(3)


def main_two(args):

    # policy
    policy = Policy(lambdas=[0.3, 0.5],
                    min_dx=0.001,
                    speed=0.005,
                    strategy="max",
                    k=3,
                    threshold=0.001,
                    min_trg_distance=0.01,
                    isgoal=True,
                    tau=5,
                    beta=-2.)
    logger(policy)

    """ pcnn """

    N = 40
    Nj = 10**2
    sigma = 0.009

    params = {
        "N": N,
        "Nj": Nj,
        "tau": 10.0,
        "alpha": 0.19,
        "beta": 20.0, # 20.0
        "lr": 0.1,
        "threshold": 0.2,
        "ach_threshold": 0.3,
        "da_threshold": 0.5,
        "tau_ach": 150.,  # 2.
        "eq_ach": 1.,
        "tau_da": 10.,  # 2.
        "eq_da": 0.,
        "epsilon": 1.,
        "rec_epsilon": np.exp(-7),
        "k_neighbors": None,
    }
    # *rec_epsilon*
    # low: more connections
    # exp(-x): x=4 -> 0.0183, x=5 -> 0.0067, x=6 -> 0.0025
    #          x=7 -> 0.0009, x=8 -> 0.0003, x=9 -> 0.0001

    model = PCNNlayer(pcnn_params=params,
                      policy=policy)

    # main settings
    settings = main.Settings()
    settings.duration = args.duration
    settings.online = False
    logger(f"{settings.duration=}")

    if args.load:
        model.load_params(path=f"{SAVEPATH}/pcnn_param.json")

        model._calc_graph()
        model.plot(ishow=True, save=args.save,
                   path=f"{SAVEPATH}/pcnn_graph.png")

    # --- random walk ---
    if not args.load:
        main.main_alternative(args=settings,
                              model=model,
                              input_type="trajectory")
        if settings.online:
            input()
            plt.close()
        # if args.online:
        #     plt.show()
        logger(f"random walk done")

        model._calc_graph()
        model.plot(ishow=True, save=args.save,
                   path=r"cache/pcnn_graph.png")

        # save
        if args.save:
            model.save_params(path=f"{SAVEPATH}/pcnn_param.json")

    # --- behaviour ---
    if args.test:
        test_calc_path(model,
                       eps=0.01,
                       max_steps=400,
                       online_plot=True,
                       plot=False,
                       min_distance=0.0,
                       src=np.array([0.2, 0.2]),
                       trg=np.array([0.2, 0.6]))

    # time.sleep(1)
    # plt.close()



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=int, default=0)
    parser.add_argument("--rep", type=int, default=1,
                        help="number of repetitions")
    parser.add_argument("--duration", type=int, default=10,
                        help="duration of the random walk")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    # seed
    if args.seed is not None:
        np.random.seed(args.seed)

    for _ in range(args.rep):
        logger(" ")
        if args.main == 0:
            logger.info("[main one]")
            main_one()
        else:
            logger.info("[main two]")
            main_two(args=args)

    logger.info("[done]")

