import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
from pprint import pprint
import time
import src.models as mm
from src.models import logger
from src.minimal_model import minPCNN, fullPCNN, Policy
import src.minimal_model as mm
import src.visualizations as vis
import src.utils as utils
import inputools.Trajectory as it
from tools.utils import clf, tqdm_enumerate, save_image, AnimationMaker
from tools.evolutions import load_best_individual


""" utilities """

ANIM_PATH = os.path.join(os.path.dirname(__file__), "media/")


def make_trajectory(plot: bool=False, speed: float=0.001,
                   duration: int=2, Nj: int=13**2, sigma: float=0.01,
                   **kwargs) -> tuple:

    """
    make a trajectory parsed through a place cell layer.

    Parameters
    ----------
    plot: bool
        plot the trajectory.
    speed: float
        speed of the trajectory.
    duration: int
        duration of the trajectory.
    Nj: int
        number of place cells.
    sigma: float
        variance of the place cells.
    **kwargs
        is2d: bool
            is the trajectory in 2D.
            Default is True.
        dx: float
            step size of the trajectory.
            Default is 0.005.

    Returns
    -------
    trajectory: np.ndarray
        trajectory of the agent.
    whole_track: np.ndarray
        whole track of the agent.
    inputs: np.ndarray
        inputs to the place cell layer.
    whole_track_layer: np.ndarray
        whole track of the place cell layer.
    layer: PlaceLayer
        place cell
    """

    # settings
    bounds = kwargs.get("bounds", (0, 1, 0, 1))
    is2d = kwargs.get("is2d", True)
    dx = kwargs.get("dx", 0.005)

    # make activations
    layer = it.PlaceLayer(N=Nj,
                          sigma=sigma,
                          bounds=bounds)

    # make trajectory
    # if is2d:
    #     whole_track = it.make_whole_walk(dx=dx)
    #     trajectory = it.make_trajectory(duration=duration, dt=1.,
    #                                     speed=[speed, speed], 
    #                                     prob_turn=0.01, k_average=200)[400:]

    #     inputs = layer.parse_trajectory(trajectory=trajectory)
    #     whole_track_layer = layer.parse_trajectory(trajectory=whole_track)
    # else:
    #     bounds = (0, 1, 0, 1)
    #     trajectory, inputs, whole_track, whole_track_layer = utils.make_env(
    #         layer=layer, duration=None, speed=None, dt=None,
    #         distance=None, dx=1e-4,
    #         plot=False, verbose=True, bounds=bounds,
    #         line_env=True, 
    #         dx_whole=5e-3)

    trajectory, inputs, whole_track, whole_track_layer = utils.make_env(
        layer=layer, duration=duration, speed=0.1,
        dt=None, distance=None, dx=1e-2,
        plot=False,
        verbose=kwargs.get("verbose", False),
        bounds=bounds,
        line_env=False,
        make_full=kwargs.get("make_full", True),
        dx_whole=5e-3)

    if plot:
        plt.figure(figsize=(3, 3))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'k-')
        plt.xticks(())
        plt.yticks(())
        plt.axis('off')
        plt.show()

    return trajectory, whole_track, inputs, whole_track_layer, layer


def make_centers(model: object, trajectory: np.ndarray,
                 whole_track: np.ndarray, whole_track_layer: np.ndarray,
                 plot: bool=False, ax: plt.Axes=None,
                 show: bool=False, color: str='blue',
                 alpha: float=0.5,
                 old_centers: np.ndarray=None,
                 old_connections: np.ndarray=None,
                 plot_connections: bool=False,
                 verbose: bool=True) -> np.ndarray:

    record = utils.train_whole_track(model=model, whole_track=whole_track,
                                     whole_track_layer=whole_track_layer,
                                     verbose=verbose)

    # trajectory
    if plot:
        if ax is None:
            _, ax = plt.subplots()
    else:
        ax = None

    if color == 'blue' and plot:
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                'b-', alpha=0.1, lw=1, label="trajectory")

    # calculating new centers and connections
    centers, connections, idxs = vis.plot_centers(model=model,
                                  trajectory=whole_track_layer,
                                  track=whole_track, kernel=np.ones((20)),
                                  threshold=0, threshold_s=0.1,
                                  ax=ax, record=record,
                                            kernel2=np.ones((4, 4)),
                                  alpha=alpha, plot=False, use_knn=True,
                                  knn_k=6, max_dist=0.35,
                                  grid=False, color=color,
                                  get_pruned=True)

    # overwrite new centers and connections
    # if old_connections is not None:
    #     connections = old_connections
    #     centers = old_centers

    # # plot
    # if plot:
    #     # ax.scatter(centers[:, 0], centers[:, 1], c=color, s=20, 
    #     #            alpha=alpha)
    #     if plot_connections:
    #         vis.plot_graph(centers=centers, connectivity=connections,
    #                        ax=ax, alpha=alpha, color=color, grid=False)

    # if plot:
    #     ax.set_title(f"{len(centers)} place cells")
    #     plt.legend()
    #     if show:
    #         plt.show()

    centers_mask = np.zeros(model.N)
    centers_mask[idxs] = 1
    connections_mask = np.zeros((model.N, model.N))
    # for i, idx in enumerate(idxs):
    #     for j in connections[i]:
    #         connections_mask[idx, j] = 1

    return centers, connections, centers_mask, connections_mask


def calc_nodes_edges(Wff: np.ndarray, Wrec: np.ndarray,
                     centers: np.ndarray) -> tuple:

    nodes = mm.calc_centers_from_layer(wff=Wff,
                                       centers=centers)
    edges, connections = mm.make_edge_list(M=Wrec, centers=nodes)

    nodes[:, 0] = np.where(np.isinf(nodes[:, 0]), -np.inf, nodes[:, 0])
    nodes[:, 1] = np.where(np.isinf(nodes[:, 1]), -np.inf, nodes[:, 1])

    return nodes, edges, connections


def online_plot(t: int, tper: int, model: object,
                trajectory: np.ndarray, policy_eq: float,
                policy_time: int, with_policy: bool,
                ax1: plt.Axes, ax2: plt.Axes,
                fig: plt.Figure,
                umax: list, da: list, ach: list,
                dwd: list=None,
                centers: np.ndarray=None,
                connections: np.ndarray=None,
                **kwargs):

    if t%tper == 0:

        ax1.clear()

        if kwargs.get("plot_trajectory", True):
            ax1.plot(trajectory[:t, 0], trajectory[:t, 1],
                     'b-', alpha=0.2, lw=1)
            # ax1.scatter(trajectory[:len(da), 0],
            #             trajectory[:len(da), 1],
            #             c=da, s=40, cmap="Greens",
            #             alpha=0.1, marker='o',
            #             vmin=0., vmax=1.)
            if kwargs.get("sho", True):
                ax1.plot(trajectory[t-1, 0], trajectory[t-1, 1],
                         marker='v', c='b', alpha=1., ms=10)
        if centers is not None:
            if connections is not None:
                vis.plot_graph(centers=centers, connectivity=connections,
                               ax=ax1, alpha=0.9, color="black", grid=False,
                               plot_connections=True,
                               plot_centers=True)
            else:
                ax1.scatter(centers[:, 0], centers[:, 1],
                            c='b', s=20, alpha=0.5)
        elif model._umask.sum() > 0:
            ax1.scatter(model.centers[:, 0], model.centers[:, 1])

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xticks(())
        ax1.set_yticks(())
        ax1.grid(True)
        ax1.set_aspect('equal', 'box')
        if ith_policy:
            ax1.set_title(f"t={t/1000:.1f}s " + \
                f"| N={model._umask.sum():.0f} - " + \
                "$Eq_{\\text{ACh}}$=" + f"{policy_eq:.2f}"+ \
                f" [$t_p=${policy_time/1000:.0f}s]")
        # else:
        #     ax1.set_title(f"t={t/1000:.1f}s | " + \
        #         f"N={model._umask.sum():.0f}")

        #
        ax2.clear()
        # ax2.imshow(model.dwda)
        # ax2.colorbar()
        ax2.set_title(f"Activity")
        #
        # ax2.imshow(model._Wff, cmap='viridis', vmin=0.,
        #            aspect='auto', interpolation='nearest',
        #            label="")
        ax2.imshow(np.array(umax).T, cmap='Greys',
                   aspect='auto', interpolation='kaiser',
                   vmin=0., vmax=0.1,
                   label="")

        # ax2.axhline(model._da_threshold, color='g', alpha=0.5,
        #             linestyle='--', lw=1)
        # ax2.plot(range(len(ach)), ach, 'b-', label=f"ACh={model._ACh:.3f}",
        #          alpha=0.5) # range(len(da))
        # print(len(da), len(umax), len(umax[0]))
        # print(np.linspace(0, len(umax), len(da)), len(da*model.N))
        ax2.plot(np.linspace(0, max(len(umax), 1), len(da)),
                 np.array(da)*model.N,
                 'g-', label=f"DA={model._DA:.3f}",
                 alpha=0.4)
        # ax2.plot(np.linspace(0, max(len(umax), 1), len(ach)),
        #             ach*model.N,
        #             'b-', label=f"ACh={model._ACh:.3f}",
        #             alpha=0.4)
        # ax2.plot(range(len(dwd)), dwd, 'r-',
        #          label=f"$\Delta W$={model.dwda.max():.3f}",
        #          alpha=0.5)
        ax2.set_xticks(())
        ax2.set_ylim(0, model.N)

        # ax2.plot(range(len(umax)), umax, 'k-', label="most active $u$",
        #          alpha=0.7)
        # ax2.set_ylim(0, 1.1)
        # ax2.legend()
        # ax2.grid(True)

        if kwargs.get("show", True):
            fig.canvas.draw()
            plt.pause(0.0001)


def make_graph_record(centers: np.ndarray,
                      connections: np.ndarray,
                      graph: dict=None):

    if graph is None:
        new = True

        graph = {
            "nodes": {
                i: center.tolist() for i, center in enumerate(centers)
            },
            "edges": {
                i: [int(j) for j in idxs] for i, idxs in enumerate(connections)
            },
            "centers": centers,
            "connections": connections,
        }
    else:
        new = False
        old_centers = graph["centers"]
        old_connections = graph["connections"]

        # determine the preserved old centers
        mask = np.isclose(centers, old_centers, atol=1e-3).all(axis=1)
        centers = centers[mask]


def calculate_area(p1, p2, p3):
    """Calculate the area of a triangle given its vertices."""
    return 0.5 * np.abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))


def remove_zero_rows_cols(matrix):
    # Find indices of rows that have at least one non-zero element
    non_zero_indices = np.where(np.any(matrix != 0, axis=1))[0]
    
    # Use these indices to select both rows and columns (since it's symmetric)
    return matrix[non_zero_indices][:, non_zero_indices]


def plot_triangles(centers, connectivity_matrix, ax):
    """Plot triangles formed by nodes and edges, colored by their area."""
    triangles = []
    areas = []

    centers = centers[np.where(connectivity_matrix.sum(axis=1) > 0)]

    # eliminate rows and columns with no connections
    connectivity_matrix = remove_zero_rows_cols(connectivity_matrix)

    num_nodes = len(centers)

    # Find all triangles
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if connectivity_matrix[i, j]:
                for k in range(j + 1, num_nodes):
                    if connectivity_matrix[i, k] and connectivity_matrix[j, k]:
                        p1, p2, p3 = centers[i], centers[j], centers[k]
                        triangles.append([p1, p2, p3])
                        areas.append(calculate_area(p1, p2, p3))

    # Normalize areas for coloring
    areas = np.array(areas)
    print(len(areas), areas, connectivity_matrix.shape)
    if len(areas) == 0:
        return
    if areas.max() != areas.min():
        norm_areas = (areas - areas.min()) / (areas.max() - areas.min())
    else:
        norm_areas = np.zeros_like(areas)

    # Create a PolyCollection
    poly_collection = PolyCollection(triangles, array=norm_areas,
                                     cmap='Reds',
                                     alpha=0.4, lw=1)

    # Plot
    ax.add_collection(poly_collection)
    ax.autoscale()
    ax.set_aspect('equal')


from scipy.stats import gaussian_kde

def plot_density_field(centers, ax, grid_size=100, bandwidth=0.1,
                       alpha=0.5, bounds: tuple=(0, 1, 0, 1)):
    """
    Plot a density field over the environment based on the given centers.
    
    Parameters:
    - centers: np.ndarray, shape (n_centers, 2)
        Array of node centers.
    - ax: matplotlib.axes.Axes
        The axes on which to plot the density field.
    - grid_size: int, optional
        The size of the grid for the density estimation.
    - bandwidth: float, optional
        The bandwidth for the kernel density estimation.
    """

    centers = centers[np.isfinite(centers[:, 0])]

    grid_size = int(np.floor(len(centers)*0.8))

    # Extract x and y coordinates of the centers
    x, y = centers[:, 0], centers[:, 1]

    # Create a grid over the environment
    xmin, xmax = bounds[0], bounds[1]
    ymin, ymax = bounds[2], bounds[3]
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, grid_size),
                         np.linspace(ymin, ymax, grid_size))
    
    # Perform kernel density estimation
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)

    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    print(zz.min())
    zz -= zz.min()
    # Plot the density field
    ax.imshow(zz, extent=[xmin, xmax, ymin, ymax],
              origin='lower', cmap='Greens', vmin=0, alpha=alpha)



""" main functions """


def main(args):

    N = 200
    Nj = 25**2

    params = {
        "N": N,
        "Nj": Nj,
        "tau": 10.0,
        "alpha": 0.2,
        "beta": 20.0,
        "lr": 0.012,
        "threshold": 0.03,
        "da_threshold": 0.9,
        "tau_ach": 2.,
        "eq_ach": 1.,
        "tau_da": 2.,
        "eq_da": 0.,
    }

    # make trajectory
    trajectory, whole_track, inputs, whole_track_layer, layer = make_trajectory(
                            plot=False,
                            Nj=Nj,
                            duration=args.duration,
                            is2d=True,
                            sigma=0.004)

    online = args.online
    if online:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        da = []
        ach = []
        umax = []

    animation_maker = AnimationMaker(fps=7, use_logger=True,
                                     path=ANIM_PATH)

    # train
    logger("training...")

    model = minPCNN(**params)
    centers = []
    umask0 = model._umask.copy()
    tper = 500
    for t, x in tqdm_enumerate(inputs):
        model.step(x=x.reshape(-1, 1))

        # make centers
        idxs = np.where(model._umask - umask0)[0]
        umask0 = model._umask.copy()
        if len(idxs) > 0:
            centers += [[trajectory[t, 0], trajectory[t, 1]]]
            model.set_centers(centers=np.array(centers))

        # online plot
        if online:
            da += [model._DA]
            ach += [model._ACh]
            umax += [model.u.max()]

            if t % tper == 0:

                model.clear_connections(epsilon=0.9)
                centers2, connections2, _, _ = make_centers(
                                model=model,
                                trajectory=trajectory,
                                whole_track=whole_track,
                                whole_track_layer=whole_track_layer,
                                plot=False,
                                show=False,
                                color='black',
                                verbose=False,
                                alpha=0.95,
                                ax=ax1)

                model.unfreeze()
                online_plot(t=t, tper=tper, model=model,
                            trajectory=trajectory, policy_eq=0,
                            policy_time=0, with_policy=False,
                            ax1=ax1, ax2=ax2, fig=fig,
                            umax=umax, da=da, ach=ach,
                            centers=centers2,
                            connections=connections2,
                            plot_trajectory=False,
                            show=False)

                animation_maker.add_frame(fig)

    if online:
        animation_maker.make_animation(
            name=f"roaming_{time.strftime('%H%M%S')}")
        logger(f"animation saved at {ANIM_PATH}")
        animation_maker.play_animation(return_Image=False)

    # --- PLOT
    # --- with old pc estimate
    _ = make_centers(model=model,
                     trajectory=trajectory,
                     whole_track=whole_track,
                     whole_track_layer=whole_track_layer,
                     plot=True, show=True)


def main_double(args):

    N = 80
    Nj = 12**2
    sigma = 0.008

    params = {
        "N": N,
        "Nj": Nj,
        "tau": 10.0,
        "alpha": 0.19,
        "beta": 20.0, # 20.0
        "lr": 0.022,
        "threshold": 0.05,#kwargs.get("threshold", 0.09),
        "ach_threshold": 0.9,#args.ach_threshold,
        "da_threshold": 0.8,
        "tau_ach": 100.,  # 2.
        "eq_ach": 1.,
        "tau_da": 40.,#args.tau,  # 2.
        "eq_da": 0.,
        "epsilon": 0.5,
        "rec_epsilon": 0.02,# kwargs.get("rec_epsilon", 0.1),
    }

    # params = {
    #     "N": N,
    #     "Nj": Nj,
    #     "tau": 10.0,
    #     "alpha": 0.2,
    #     "beta": 20.0,
    #     "lr": args.lr,
    #     "threshold": 0.1,
    #     "ach_threshold": args.ach_threshold,
    #     "da_threshold": 0.5,
    #     "tau_ach": 200.,  # 2.
    #     "eq_ach": 1.,
    #     "tau_da": args.tau,  # 2.
    #     "eq_da": 0.,
    #     "epsilon": 0.5,
    # }

    # make trajectory
    trajectory, whole_track, inputs, whole_track_layer, layer = make_trajectory(
                            plot=False,
                            Nj=Nj,
                            duration=args.duration,
                            is2d=True,
                            sigma=sigma)

    model = minPCNN(**params)
    if args.verbose:
        logger(f"%{model}")
        logger("training 1...")


    da = []
    ach = []
    dwd = []
    umax = []
    ux = []
    max_mem = 50
    tper_anim = 400
    if args.animate:
        # fig_anim, (ax_anim, ax_anim_2) = plt.subplots(1, 2)

        # make the layout of the plot as two axis of equal size
        fig_anim = plt.figure(figsize=(6, 3))
        gs = fig_anim.add_gridspec(1, 2)
        ax_anim = fig_anim.add_subplot(gs[0, 0])
        ax_anim_2 = fig_anim.add_subplot(gs[0, 1])

        animation_maker = AnimationMaker(fps=10,
                                         use_logger=True,
                                         path=ANIM_PATH)

    # train
    for t, x in tqdm_enumerate(inputs, disable=not args.verbose):
        model.step(x=x.reshape(-1, 1))
        # if t % 100:
        #     model.clear_connections()

        # record
        if t % tper_anim == 0 and args.animate:
            da += [model._DA]
            ach += [model._ACh]
            # logger.debug(f"{len(da)=}, {len(ach)=}")
            # logger.debug(f"{da=}, {ach=}")
            # dwd += [model.dwda.max()]
            umax += [model.u.max()]
            ux += [model.u.flatten().tolist()]
            if len(da) > max_mem:
                del da[0]
                del ach[0]
                # del dwd[0]
                del umax[0]
                del ux[0]

        if args.animate and t % tper_anim == 0:
            ax_anim.clear()
            # update figure
            ax_anim.plot(trajectory[:t, 0],
                    trajectory[:t, 1], 'r-', alpha=0.2, lw=1)
            # vis.plot_c(W=model._Wff.copy(),
            #            layer=layer,
            #            color="blue",
            #            k=6,
            #            max_dist=0.3,
            #            ax=ax_anim,
            #            show=False,
            #            title=f"t={t/1000:.1f}s [no reward]")
            centers1, _, connections1 = calc_nodes_edges(Wff=model._Wff,
                                                 Wrec=model.W_rec,
                                                 centers=layer.centers)

            ax_anim.scatter(centers1[:, 0], centers1[:, 1], c="blue",
                       s=20, marker='o', alpha=0.5)
            for i in range(centers1.shape[0]):
                for j in range(centers1.shape[0]):
                    if connections1[i, j] == 1:
                        ax_anim.plot([centers1[i, 0], centers1[j, 0]],
                                [centers1[i, 1], centers1[j, 1]],
                                '-', color="blue", alpha=0.5,
                                linewidth=0.5)

            ax_anim.set_title(f"t={t/1000:03.1f} [no reward]")
            ax_anim.set_xlim((0, 1.))
            ax_anim.set_ylim((0, 1.))
            ax_anim.set_xticks(())
            ax_anim.set_yticks(())
            # ax_anim.axis('off')

            # other ax
            # ACh
            ax_anim_2.clear()
            # ax_anim_2.axhline(model._da_threshold,
            #                   color='g', alpha=0.5,
            #                   linestyle='--', lw=1)
            ax_anim_2.plot(range(len(da)),
                     np.array(da),
                     'g-', label=f"DA={model._DA:.2f}",
                     alpha=0.4)
            # logger.debug(f"{ach}, {len(ach)}")
            ax_anim_2.plot(range(len(ach)),
                        np.array(ach),
                        'b-', label=f"ACh={model._ACh:.2f}",
                        alpha=0.4)
            ax_anim_2.set_xticks(())
            ax_anim_2.set_ylim(0, 1.1)
            ax_anim_2.set_xlim(0, max_mem)
            # ax_anim_2.set_ylabel("concentration")
            ax_anim_2.set_yticks((0., 1.), (0., 1.))
            ax_anim_2.set_xlabel("time")
            ax_anim_2.set_title(f"Neuromodulators")

            # aspect
            ax_anim_2.set_aspect('auto')
            ax_anim_2.legend(loc="lower right")

            animation_maker.add_frame(fig_anim)

    # --- PLOT
    # --- with old pc estimate
    if args.verbose:
        logger("plotting 1...")

    ax = None

    # model.clear_connections(epsilon=0.05)
    # centers1, connections1, nodes1, edges1 = make_centers(model=model,
    #                  trajectory=trajectory,
    #                  whole_track=whole_track,
    #                  whole_track_layer=whole_track_layer,
    #                  plot=ax is not None, show=False, color='grey',
    #                  verbose=args.verbose,
    #                  ax=ax)

    # delete centers at the origin
    centers1, _, connections1 = calc_nodes_edges(Wff=model._Wff,
                                             Wrec=model.W_rec,
                                             centers=layer.centers)

    # model.add_many_centers(centers=centers1)
    if args.verbose:
        logger(f"{model._umask.sum():.0f} place cells")

    # --- train 2 -----------------------------------------------
    # make trajectory
    trajectory, whole_track, inputs, whole_track_layer, layer = make_trajectory(
                            plot=False,
                            Nj=Nj,
                            duration=args.duration,
                            is2d=True,
                            sigma=0.004)

    # policy
    startime = 20
    trg = np.array([0.7, 0.7])
    policy = Policy(eq_da=1.,
                    trg=trg,
                    threshold=0.4,
                    startime=startime)

    online = args.online
    if online:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        ax1.scatter(trg[0], trg[1], c='g', s=20,
                    alpha=0.8, marker='x')

    da = []
    ach = []
    dwd = []
    umax = []
    ux = []
    max_mem = 50

    if args.verbose:
        logger(f"%{policy}")
        logger("training 2...")

    #
    umax = []
    ft = []
    umask0 = model._umask.copy()
    tper = 300

    input_duration = len(inputs)

    model.unfreeze()
    for t, x in tqdm_enumerate(inputs,
                               disable=not args.verbose):
        eq_da = policy(pos=trajectory[t, :], t=t)
        model(x=x, eq_da=eq_da)

        # - calc nodes
        idxs = np.where(model._umask - umask0)[0]
        umask0 = model._umask.copy()
        if len(idxs) > 0:
            ft += [[trajectory[t, 0], trajectory[t, 1]]]
            # model.add_center(center=[trajectory[t, 0],
            #                          trajectory[t, 1]],
            #                  idx=idxs[0])

        # record
        if t % tper_anim == 0 and args.animate:
            da += [model._DA]
            ach += [model._ACh]
            # logger.debug(f"{len(da)=}, {len(ach)=}")
            # logger.debug(f"{da=}, {ach=}")
            # dwd += [model.dwda.max()]
            umax += [model.u.max()]
            ux += [model.u.flatten().tolist()]
            if len(da) > max_mem:
                del da[0]
                del ach[0]
                # del dwd[0]
                del umax[0]
                del ux[0]

        # online plot
        if online:

            if t % tper == 0:
                # model.clear_connections(epsilon=0.05)
                # centers2, connections2, nodes2, edges2 = make_centers(model=model,
                #                 trajectory=trajectory,
                #                 whole_track=whole_track,
                #                 whole_track_layer=whole_track_layer,
                #                 plot=False, show=False, color='black',
                #                 verbose=False,
                #                 alpha=0.95,
                #                 ax=ax)

                model.unfreeze()

                online_plot(t=t, tper=tper, model=model,
                            trajectory=trajectory,
                            policy_eq=eq_da,
                            policy_time=startime,
                            with_policy=True,
                            ax1=ax1, ax2=ax2, fig=fig,
                            umax=ux, da=da, ach=ach,
                            dwd=dwd,
                            # centers=centers2,
                            # connections=connections2,
                            # plot_trajectory=True,
                            show=True)

        if args.animate and t % tper_anim == 0:

            # main ax
            ax_anim.clear()
            # update figure
            ax_anim.scatter(trg[0], trg[1], c='g', s=100,
                            alpha=0.8, marker='x')
            ax_anim.plot(trajectory[:t, 0],
                    trajectory[:t, 1], 'r-', alpha=0.2, lw=1)
            # vis.plot_c(W=model._Wff.copy(),
            #            layer=layer,
            #            color="black",
            #            k=5,
            #            max_dist=0.25,
            #            ax=ax_anim,
            #            show=False,
            #            title=f"t={t/1000:.1f}s [reward]")
            centers2, _, connections2 = calc_nodes_edges(Wff=model._Wff,
                                                     Wrec=model.W_rec,
                                                     centers=layer.centers)

            # plot centers
            ax_anim.scatter(centers2[:, 0], centers2[:, 1], c="blue",
                       s=20, marker='o', alpha=0.5)
            for i in range(centers2.shape[0]):
                for j in range(centers2.shape[0]):
                    if connections2[i, j] == 1:
                        ax_anim.plot([centers2[i, 0], centers2[j, 0]],
                                [centers2[i, 1], centers2[j, 1]],
                                '-', color="blue", alpha=0.5,
                                linewidth=0.5)

            ax_anim.set_title(f"t={t/1000:03.1f} [reward]")
            ax_anim.set_xlim((0, 1.))
            ax_anim.set_ylim((0, 1))
            ax_anim.set_xticks(())
            ax_anim.set_yticks(())
            # ax_anim.axis('off')

            # other ax
            # ACh
            ax_anim_2.clear()
            # ax_anim_2.axhline(model._da_threshold,
            #                   color='g', alpha=0.5,
            #                   linestyle='--', lw=1)
            ax_anim_2.plot(range(len(da)),
                     np.array(da),
                     'g-', label=f"DA={model._DA:.2f}",
                     alpha=0.4)
            # logger.debug(f"{ach}, {len(ach)}")
            ax_anim_2.plot(range(len(ach)),
                        np.array(ach),
                        'b-', label=f"ACh={model._ACh:.2f}",
                        alpha=0.4)
            ax_anim_2.set_ylim(0, 1.1)
            ax_anim_2.set_xlim(0, max_mem)
            ax_anim_2.set_xticks(())
            ax_anim_2.set_yticks((0., 1.), (0., 1.))
            # ax_anim_2.set_ylabel("concentration")
            ax_anim_2.set_xlabel("time")

            # aspect
            ax_anim_2.set_aspect('auto')
            ax_anim_2.legend(loc='lower right')
            ax_anim_2.set_title(f"Neuromodulators")

            animation_maker.add_frame(fig_anim)

    # --- PLOT 2
    # --- with old pc estimate
    if args.verbose:
        logger("plotting 2...")

    if online:
        plt.close()
        fig, ax = plt.subplots()
        fig.suptitle(f"$\eta=${args.lr} - $\\tau=${args.tau} - " + \
                     f"$\\theta_{{ach}}$={args.ach_threshold} - " + \
                     f"$\\theta_{{da}}$={params['da_threshold']}")

    if args.animate:
        animation_maker.make_animation(
            name=f"roaming_{time.strftime('%H%M%S')}")
        logger(f"animation saved at {ANIM_PATH}")
        animation_maker.play_animation(return_Image=False)

    # centers2, connections2, nodes2, edges2 = make_centers(
    #                 model=model,
    #                 trajectory=trajectory,
    #                 whole_track=whole_track,
    #                 whole_track_layer=whole_track_layer,
    #                 plot=False, show=False, color='black',
    #                 verbose=args.verbose,
    #                 alpha=0.95,
    #                 ax=ax)

    centers2, _, connections2 = calc_nodes_edges(Wff=model._Wff,
                                             Wrec=model.W_rec,
                                             centers=layer.centers)

    if args.verbose:
        logger(f"{model._umask.sum():.0f} place cells")

    # plt.close()

    """ plot """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                        figsize=(6, 3))
    fig.suptitle(f"$\eta=${args.lr} - $\\tau=${args.tau} - " + \
                 f"$\\theta_{{ach}}$={args.ach_threshold} - " + \
                 f"$\\theta_{{da}}$={params['da_threshold']}")

    #
    ax1.scatter(trg[0], trg[1], c='r', s=40, alpha=0.8,
               marker='x',
               label="target")
    vis.plot_graph(centers=centers1,
                   connectivity=connections1,
                   ax=ax1, alpha=1., color="black",
                   marker='o',
                   label="centers",
                   plot_connections=True,
                   plot_centers=True)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal', 'box')
    ax1.set_title(f"OLD {len(centers1)} PCs")

    #
    ax2.scatter(trg[0], trg[1], c='r', s=40, alpha=0.8,
               marker='x',
               label="target")

    vis.plot_graph(centers=centers2,
                   connectivity=connections2,
                   ax=ax2, alpha=0.8, color="green",
                   marker='o',
                   label="centers",
                   plot_connections=True,
                   plot_centers=True)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal', 'box')
    ax2.set_title(f"NEW - {len(centers2)} PCs")

    #
    ax3.imshow(model._Wff, cmap='viridis', vmin=0.,
               aspect='auto', interpolation='nearest')
    tot = np.where(model._Wff.sum(axis=1) > 0)[0].shape[0]
    ax3.set_title(f"$W_{{ff}}$ - {tot} active neurons")
    ax3.set_label("input PCs $j$")
    ax3.set_ylabel("local PCs $i$")

    if args.save:
        plt.savefig(f"media/online_{args.lr}_{args.tau}.png")
        plt.close()
    elif args.show:
        plt.show()


def main_alternative(args, model: object=None,
                     **kwargs) -> tuple:

    """ Main function to run the PCNN model training and visualization.

    Parameters:
    -----------
    args: argparse.Namespace
        Command-line arguments parsed by argparse. Expected attributes:
        duration: int
            Duration of the trajectory.
        online: bool
            Whether to plot online visualizations.
        lr: float
            Learning rate for the model.
        tau: float
            Time constant for the model.
        ach_threshold: float
            Threshold for acetylcholine.
    model: object, optional
        Pre-initialized model object. If None, a new model will be created using kwargs.
    **kwargs: dict, optional
        Additional keyword arguments for model initialization and trajectory creation.
        N: int
            Number of neurons.
        Nj: int
            Number of junctions.
        threshold: float
            Threshold value for the model.
        rec_epsilon: float
            Recurrent epsilon value.
        trajectory: np.ndarray
            Predefined trajectory.
        inputs: np.ndarray
            Input data for the model.
        layer: object
            Layer object containing centers.
        tper: int
            Time period for logging and plotting.
        verbose: bool
            Whether to print verbose logs.

    Returns:
    --------
    centers: np.ndarray
        Centers of the place cells.
    edges: np.ndarray
        Edges of the place cells.
    model: object
        Trained model object.
    """

    # make model if not provided
    sigma = kwargs.get("sigma", 0.008)
    bounds = (0, 1, 0, 1)
    if model is None:
        N = kwargs.get("N", 50)
        Nj = kwargs.get("Nj", 11**2)

        params = {
            "N": N,
            "Nj": Nj,
            "tau": 10.0,
            "alpha": 0.19,
            "beta": 20.0, # 20.0
            "lr": args.lr,
            "threshold": kwargs.get("threshold", 0.09),
            "ach_threshold": 0.5,#args.ach_threshold,
            "da_threshold": 0.7,
            "tau_ach": 200.,  # 2.
            "eq_ach": 1.,
            "tau_da": args.tau,  # 2.
            "eq_da": 0.,
            "epsilon": 0.000000001,
            "rec_epsilon": kwargs.get("rec_epsilon", 0.1),
        }

        model = minPCNN(**params)
        threshold = params["threshold"]
    else:
        N = model.N
        Nj = model.Nj
        threshold = model._threshold

    # make trajectory if not provided
    if kwargs.get("trajectory", None) is not None:
        trajectory = kwargs.get("trajectory")
        inputs = kwargs.get("inputs")
        layer = kwargs.get("layer")
    else:
        trajectory, _, inputs, _, layer = make_trajectory(
                                plot=False,
                                Nj=Nj,
                                duration=args.duration,
                                is2d=True,
                                sigma=sigma,
                                verbose=True,
                                make_full=False,
                                bounds=bounds)

    verbose = kwargs.get("verbose", True)
    online = args.online
    if online:
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        da = []
        ach = []
        umax = []

    # choose input type
    if kwargs.get("input_type", None) == "trajectory":
        inputs = trajectory
        logger.debug(f"Using trajectory as input data")

    # policy
    startime = 2
    trg = np.array([0.1, 0.1])
    policy = Policy(eq_da=1.,
                    trg=trg,
                    threshold=0.6,
                    startime=startime)

    # run
    if verbose:
        logger(f"[threshold={threshold}]")
        logger("training...")

    tper = kwargs.get("tper", 500)
    for t, x in tqdm_enumerate(inputs, disable=not verbose):
        # model.step(x=x.reshape(-1, 1))

        eq_da = policy(pos=trajectory[t, :], t=t)
        model(x=x, eq_da=eq_da)

        if t % tper == 0:

            # calc centers positions
            centers = mm.calc_centers_from_layer(wff=model._Wff,
                                                 centers=layer.centers)
            edges, connections = mm.make_edge_list(M=model.W_rec,
                                      centers=centers)
            # online plot
            if online:

                # plot
                ax1.clear()
                # ax2.clear()
                # ax3.clear()


                ax1.plot(centers[:, 0], centers[:, 1], 'bo', alpha=0.3, label="place cells")

                for e in edges:
                    ax1.plot(e[:, 0], e[:, 1], 'b-', alpha=0.4, lw=0.5)

                ax1.plot(trajectory[:t, 0], trajectory[:t, 1],
                         'r-', alpha=0.05, lw=2, label="trajectory")

                ax1.set_xlim(bounds[0], bounds[1])
                ax1.set_ylim(bounds[2], bounds[3])
                ax1.set_axis_off()
                ax1.set_aspect('equal', 'box')
                # ax1.legend(loc="lower right")
                # ax1.set_title("the geometry of a spatial representation")
                # ax1.set_title(f"t={t/1000:.1f}s | $N=${model._umask.sum():.0f}")

                # ax3.imshow(model.W_rec, cmap="hot", vmin=0,
                #            aspect='equal', interpolation='nearest')
                # ax3.set_title(f"model=$W_{{rec}}$ - {model.W_rec.max():.2f}")
                # ax3.set_xticks(())
                # ax3.set_yticks(())

                # ax2.imshow(model.u.copy().reshape(1, -1),
                #            cmap="Greys", vmin=0, vmax=0.5,
                #            aspect='auto', interpolation='nearest')
                # ax2.set_title(f"activation, max={model.u.max():.5f}")
                # ax2.set_xticks(())
                # ax2.set_yticks(())

                plt.pause(0.0001)


    # plot_density_field(centers, ax1, grid_size=None,
    #                    bandwidth=0.3,
    #                    alpha=0.3)


    if online:
        if str(input("save? [y/n] ")) == "y":
            name = f"online_{np.random.random():.3f}"
            fig.savefig(f"/Users/daniekru/Desktop/{name}.svg", format="svg", dpi=300)
            logger(f"saved as {name}.svg")
            plt.close()

    return centers, edges, model

    # time.sleep(2)
    # plt.close()


def super_alternative(args):

    """
    run many iterations and plot mean and variance of the 
    distribution of nodes and edges
    """

    N = 40
    Nj = 11**2
    sigma = 0.008

    num = 20
    num_avg = 2
    vars = np.around(np.linspace(0.02, 0.03, num), 7)
    var_name = "threshold"
    # vars = np.around(np.linspace(0.0, 0.0002, num), 8)
    # var_name = "rec_epsilon"
    th_label = vars[::2]

    C = np.zeros((num_avg, num))
    E = np.zeros((num_avg, num))

    for i in range(num_avg):
        logger(f"[{i+1}/{num_avg}]")

        # data
        trajectory, _, inputs, _, layer = make_trajectory(
                                plot=False,
                                Nj=25**2,
                                duration=args.duration,
                                is2d=True,
                                sigma=sigma,
                                verbose=True,
                                make_full=False)

        for j, v in tqdm_enumerate(vars):
            c, e, _ = main_alternative(args,
                                    threshold=v if var_name == "threshold" else 0.02198,
                                    rec_epsilon=v if var_name == "rec_epsilon" else 0.0001,
                                    trajectory=trajectory,
                                    inputs=inputs,
                                    layer=layer,
                                    N=N,
                                    Nj=Nj,
                                    verbose=False)
            c = [ci for ci in c if 1 > ci[0] > 0 and 1 > ci[1] > 0]
            C[i, j] = len(c)
            E[i, j] = len(e)

    #
    if args.online:
        plt.show()

    #
    C_mean = C.mean(axis=0)
    C_std = C.std(axis=0)
    E_mean = E.mean(axis=0)
    E_std = E.std(axis=0)


    # Plot statistics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(23, 7))

    # Centers
    ax1.plot(vars, C_mean, 'o-', color='black', label='Mean')
    ax1.fill_between(vars, C_mean - C_std, C_mean + C_std,
                     color='gray', alpha=0.3, label='Std Dev')
    ax1.set_title(f"Number of cells | {num_avg} averages")
    ax1.set_xlabel(f"{var_name}")
    ax1.set_ylabel("#cells")
    ax1.set_xticks(th_label)
    ax1.set_xticklabels(th_label)
    ax1.legend()

    # Edges
    ax2.plot(vars, E_mean, 'o-', color='black', label='Mean')
    ax2.fill_between(vars, E_mean - E_std, E_mean + E_std,
                     color='gray', alpha=0.3, label='Std Dev')
    ax2.set_title(f"Number of edges | {num_avg} averages")
    ax2.set_xlabel(f"{var_name}")
    ax2.set_ylabel("#edges")
    ax2.set_xticks(th_label)
    ax2.set_xticklabels(th_label)
    ax2.legend()

    plt.show()



""" misc """

class Settings:

    duration = 2
    seed = None
    online = False
    main = 0
    lr = 0.1
    tau = 2.
    ach_threshold = 0.5
    save = False
    show = False
    animate = False


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--online", action="store_true")
    # parser.add_argument("--double", action="store_true")
    parser.add_argument("--main", type=int, default=0,
                        help="0: main, 1: main_double, 2: main_alternative 3: super_alternative")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=20.)
    parser.add_argument("--ach_threshold", type=float,
                        default=0.4)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show", type=int, default=1)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.show = bool(int(args.show)) # convert to bool

    # set seed
    if args.seed is not None:
        np.random.seed(args.seed)

    #
    if args.main == 0:
        logger(f"[main]")
        main(args)
    elif args.main == 1:
        logger(f"[double]")
        main_double(args)
    elif args.main == 2:
        logger(f"[alternative]")
        main_alternative(args)
    elif args.main == 3:
        logger(f"[super alternative]")
        super_alternative(args)
    else:
        raise ValueError(f"Unknown main {args.main}")
    # if args.double:
    #     main_double(args)
    # else:
    #     main(args)

