import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import argparse

from tools.utils import clf, tqdm_enumerate, logger
import inputools.Trajectory as it
import pcnn_core as pcnn
import mod_core as mod

import os, sys
base_path = os.getcwd().split("PCNN")[0]+"PCNN/src/"
sys.path.append(base_path)
import utils
import simplerl.environments as ev




""" for testing """""


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

    trajectory, inputs, whole_track, whole_track_layer = utils.make_env(
        layer=layer, duration=duration, speed=0.1,
        dt=None, distance=None, dx=1e-2,
        plot=False,
        verbose=True,
        bounds=bounds,
        line_env=False,
        make_full=kwargs.get("make_full", False),
        dx_whole=5e-3)

    if plot:
        plt.figure(figsize=(3, 3))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'k-')
        plt.xticks(())
        plt.yticks(())
        plt.axis('off')
        plt.show()

    return trajectory, whole_track, inputs, whole_track_layer, layer


def train(inputs: np.ndarray, layer_centers: np.ndarray,
          params: dict):

    model = PCNN(**params)
    for x in inputs:
        model(x=x.reshape(-1, 1))

    info = {"centers": calc_centers_from_layer(wff=model._Wff,
                                   centers=layer_centers),
            "connectivity": model._Wrec.copy(),
            "count": len(model)}

    return info


def plot_network(centers, connectivity, ax):

    """
    plot the network
    """

    ax.plot(centers[:, 0], centers[:, 1], 'ko', markersize=2)
    for i in range(connectivity.shape[0]):
        for j in range(connectivity.shape[1]):
            if connectivity[i, j] > 0:
                ax.plot([centers[i, 0], centers[j, 0]],
                        [centers[i, 1], centers[j, 1]], 'k-',
                        alpha=0.2, lw=0.5)

    ax.axis('off')

    return ax


def simple_run(args):

    np.random.seed(0)
    duration = args.duration
    N = args.N
    Nj = 13**2

    logger(f"{duration}")
    logger(f"{N=}")
    logger(f"{Nj=}")

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.2,
        "beta": 20.0,
        "threshold": 0.1,
        "rep_thresold": 0.5,
        "rec_threshold": 0.1,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "eq_ach": 1.,
        "tau_ach": 2.,
        "ach_threshold": 0.9,
    }

    # make trajectory
    trajectory, _, _, _, _ = make_trajectory(
                            plot=False,
                            Nj=Nj,
                            duration=duration,
                            is2d=True,
                            sigma=0.01)

    # pc filter
    pclayer = pcnn.PClayer(n=13, sigma=0.01)
    logger.debug(f"{pclayer=}")
    params["xfilter"] = pclayer

    # model
    model = pcnn.PCNN(**params)

    # train
    fig, ax = plt.subplots(figsize=(6, 6))

    for t, x in tqdm_enumerate(trajectory):
        model(x=x.reshape(-1, 1))

        if t % 50 == 0:
            ax.clear()
            # ax.imshow(model._Wff, cmap='viridis', aspect='auto')
            ax.plot(trajectory[:t, 0], trajectory[:t, 1], 'r-',
                    lw=0.5, alpha=0.4)
            centers = pcnn.calc_centers_from_layer(wff=model._Wff,
                                              centers=model.xfilter.centers)
            plot_network(centers=centers,
                         connectivity=model._Wrec,
                         ax=ax)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"t={t} | {len(model)} neurons")
            plt.pause(0.005)


def experimentI():

    np.random.seed(0)
    duration = 10
    N = 80
    Nj = 13**2

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.2,
        "beta": 20.0,
        "threshold": 0.1,
        "rep_thresold": 0.5,
        "rec_threshold": 0.1,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "eq_ach": 1.,
        "tau_ach": 2.,
        "ach_threshold": 0.9,
    }

    # make trajectory
    trajectory, _, inputs, _, _ = make_trajectory(
                            plot=False,
                            Nj=Nj,
                            duration=duration,
                            is2d=True,
                            sigma=0.01)

    xlayer = PClayer(n=13, sigma=0.01)
    params["xfilter"] = xlayer

    D = 10
    var_name1 = "rec_threshold"
    var_values1 = np.linspace(0., 1., D)

    var_name2 = "threshold"
    var_values2 = np.linspace(0., 1., D)

    results = []
    avg_neighbors = np.zeros((D, D))
    info = []
    logger("training..")
    max_cell_count = 0
    for i, value1 in tqdm_enumerate(var_values1):
        for j, value2 in tqdm_enumerate(var_values2):
            params[var_name1] = value1
            params[var_name2] = value2
            info = train(inputs=trajectory, layer_centers=xlayer.centers,
                         params=params)
            results += [info]
            avg_neighbors[i, j] = info["connectivity"].sum() / info["count"]

            max_cell_count = max(max_cell_count, info["count"])

    logger.debug(f"max cell count: {max_cell_count}")
    logger("plotting..")

    # plot
    plt.figure(figsize=(9, 9))
    plt.imshow(np.flip(avg_neighbors, axis=0),
               cmap='viridis')
    plt.colorbar()
    plt.xlabel(var_name2)
    plt.ylabel(var_name1)
    xtickslab = [" "] * D
    xtickslab[0] = f"{var_values2[0]:.2f}"
    xtickslab[-1] = f"{var_values2[-1]:.2f}"
    plt.xticks(range(D), xtickslab)
    ytickslab = [" "] * D
    ytickslab[-1] = f"{var_values1[0]:.2f}"
    ytickslab[0] = f"{var_values1[-1]:.2f}"
    plt.yticks(range(D), ytickslab)
    plt.title("Average number of neighbors")

    plt.show()


def experimentII(args):

    np.random.seed(0)
    duration = args.duration
    N = args.N
    Nj = 13**2

    logger(f"{duration}")
    logger(f"{N=}")
    logger(f"{Nj=}")

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.2,
        "beta": 20.0,
        "threshold": 0.3,
        "rep_thresold": 0.8,
        "rec_threshold": 0.1,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "name": "PCNN"
    }

    # make trajectory
    trajectory, _, _, _, _ = make_trajectory(
                            plot=False,
                            Nj=Nj,
                            duration=duration,
                            is2d=True,
                            sigma=0.01)

    # pc filter
    pclayer = pcnn.PClayer(n=13, sigma=0.01)
    logger.debug(f"{pclayer=}")
    params["xfilter"] = pclayer

    # model
    model = pcnn.PCNN(**params)
    model_plotter = pcnn.PlotPCNN(model=model)
    modulators_list = [mod.BoundaryMod(N=N),
                       mod.Acetylcholine()]

    for modulator in modulators_list:
        logger.debug(f"{modulator} keys: {modulator.input_key}")

    modulators = mod.Modulators(modulators=modulators_list)

    exp_module = mod.ExperienceModule(pcnn=model,
                                      modulators=modulators_list)

    # train
    for t, x in tqdm_enumerate(trajectory):

        exp_module(x=x.reshape(-1, 1))
        modulators(u=exp_module.output[0],
                   position=x,
                   delta_update=exp_module.output[2],
                   collision=False)

        if t % 100 == 0:
            # exp_module.render()
            modulators.render()
            model_plotter.render(trajectory=trajectory[:t])
            plt.pause(0.005)


def experimentIII(args):

    # --- settings
    np.random.seed(0)
    duration = args.duration

    # --- brain
    N = args.N
    Nj = 13**2

    logger(f"{duration}")
    logger(f"{N=}")
    logger(f"{Nj=}")

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.1,
        "beta": 20.0,
        "threshold": 0.8,
        "rep_thresold": 0.8,
        "rec_threshold": 0.1,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "name": "PCNN"
    }

    # pc filter
    pclayer = pcnn.PClayer(n=13, sigma=0.01)
    logger.debug(f"{pclayer=}")
    params["xfilter"] = pclayer

    # model
    model = pcnn.PCNN(**params)
    model_plotter = pcnn.PlotPCNN(model=model,
                                  makefig=False)
    modulators_list = [mod.BoundaryMod(N=N),
                       mod.Acetylcholine()]

    for modulator in modulators_list:
        logger.debug(f"{modulator} keys: {modulator.input_key}")

    modulators = mod.Modulators(modulators=modulators_list)
    exp_module = mod.ExperienceModule(pcnn=model,
                                      pcnn_plotter=model_plotter,
                                      modulators=modulators_list)

    # --- agent & env
    env = ev.make_room(name="square", thickness=4.)
    agent = ev.Zombie(body=ev.AgentBody(room=env),
                      speed=0.01)

    fig, ax = plt.subplots(figsize=(5, 5))

    trajectory = []
    for t in range(duration):

        x, collision = agent()

        if collision:
            logger.debug(f">>> collision at t={t}")

        exp_module(x=agent.position.reshape(-1, 1))
        modulators(u=exp_module.output[0],
                   position=x,
                   delta_update=exp_module.output[2],
                   collision=collision)

        trajectory += [agent.position.tolist()]

        # -- plot
        if t % 5 == 0:

            ax.clear()
            exp_module.render(ax=ax,
                              trajectory=np.array(trajectory))
            modulators.render()
            env.draw(ax=ax)
            agent.render(ax=ax)

            ax.set_title(f"t={t}")
            fig.canvas.draw()

            plt.pause(0.01)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=str, default="simple")
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--N", type=int, default=80)


    args = parser.parse_args()

    if args.main == "simple":
        simple_run(args=args)
    elif args.main == "I":
        experimentI()
    elif args.main == "II":
        experimentII(args=args)
    elif args.main == "III":
        experimentIII(args=args)




