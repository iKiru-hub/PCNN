
""" from `mod_core` """

class LeakyVariable:

    def __init__(self, eq: float=0., tau: float=10,
                 name: str="leaky", ndim: int=1,
                 visualize: bool=False,
                 number: int=None,
                 max_record: int=100):

        """
        Parameters
        ----------
        eq : float
            Default 0.
        tau : float
            Default 10
        threshold : float
            Default 0.
        """

        self.name = name
        self.eq = np.array([eq]*ndim).reshape(-1, 1) if ndim > 1 else np.array([eq])
        self.ndim = ndim
        self._v = np.ones(1)*self.eq if ndim == 1 else np.ones((ndim, 1))*self.eq
        self.tau = tau
        self.record = []
        self._max_record = max_record
        self._visualize = False 
        self._number = number

        # figure configs
        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))

    def __repr__(self):
        return f"{self.name}(eq={self.eq}, tau={self.tau})"

    def __call__(self, x: float=0., eq: np.ndarray=None,
                 simulate: bool=False):

        if simulate:
            if eq is not None:
                self.eq = eq
            return self._v + (self.eq - self._v) / self.tau + x

        if eq is not None:
            self.eq = eq
        self._v += (self.eq - self._v) / self.tau + x
        self._v = np.maximum(0., self._v)
        # self.v = np.clip(self.v, -1, 1.)
        self.record += [self._v.tolist()]
        if len(self.record) > self._max_record:
            del self.record[0]

        return self._v

    def get_v(self):
        return self._v

    def reset(self):
        self._v = self.eq
        self.record = []

    def render(self):

        if not self._visualize:
            return

        self.ax.clear()
        self.ax.plot(range(len(self.record)), self.record)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.grid()
        self.ax.set_title(f"{self.name} |" +
            f" v={np.around(self._v, 2).tolist()}")

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/{self._number}.png")
            return
        self.fig.canvas.draw()



""" from `run_core` """


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


def experimentIV(args):

    # --- settings
    np.random.seed(args.seed)
    evc.set_seed(seed=args.seed)
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
        "alpha": 0.3,
        "beta": 20.0,
        "clip_min": 0.005,
        "threshold": 0.4,
        "rep_threshold": 0.7,
        "rec_threshold": 0.99,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "name": "PCNN"
    }

    # pc filter
    pclayer = pcnn.PClayer(n=13, sigma=0.01)
    logger.debug(f"{pclayer=}")
    params["xfilter"] = pclayer

    # pcnn
    model = pcnn.PCNN(**params)
    model_plotter = pcnn.PlotPCNN(model=model,
                                  visualize=True)
    modulators_dict = {"Bnd": mod.BoundaryMod(N=N,
                                              visualize=False),
                       "Ach": mod.Acetylcholine(visualize=False),
                       "ET": mod.EligibilityTrace(N=N,
                                                  visualize=False),
                       "dPos": mod.PositionTrace(visualize=False)}

    for _, modulator in modulators_dict.items():
        logger.debug(f"{modulator} keys: {modulator.input_key}")

    # other components
    modulators = mod.Modulators(modulators_dict=modulators_dict,
                                visualize=True)
    exp_module = mod.ExperienceModule(pcnn=model,
                                      pcnn_plotter=model_plotter,
                                      modulators=modulators,
                                      speed=0.01)
    brain = mod.Brain(exp_module=exp_module,
                      modulators=modulators)

    # --- agent & env
    agent = evc.AgentBody(brain=brain)

    # --- run
    evc.main(agent=agent,
             duration=duration)


def experimentV(args):

    # --- settings
    np.random.seed(args.seed)
    evc.set_seed(seed=args.seed)
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
        "alpha": 0.3,
        "beta": 20.0,
        "clip_min": 0.005,
        "threshold": 0.4,
        "rep_threshold": 0.9,
        "rec_threshold": 0.99,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "name": "PCNN"
    }

    # pc filter
    pclayer = pcnn.PClayer(n=13, sigma=0.01)
    logger.debug(f"{pclayer=}")
    params["xfilter"] = pclayer

    # pcnn
    model = pcnn.PCNN(**params)
    model_plotter = pcnn.PlotPCNN(model=model,
                                  visualize=True,
                                  number=0)
    modulators_dict = {"Bnd": mod.BoundaryMod(N=N,
                                              visualize=True,
                                              number=2),
                       "dPos": mod.PositionTrace(visualize=False)}

    for _, modulator in modulators_dict.items():
        logger.debug(f"{modulator} keys: {modulator.input_key}")

    # other components
    modulators = mod.Modulators(modulators_dict=modulators_dict,
                                visualize=True,
                                number=1)
    exp_module = mod.ExperienceModule(pcnn=model,
                                      pcnn_plotter=model_plotter,
                                      modulators=modulators,
                                      speed=0.009)
    brain = mod.Brain(exp_module=exp_module,
                      modulators=modulators)

    # --- agent & env
    agent = evc.AgentBody(brain=brain)

    # --- run
    evc.main(agent=agent,
             duration=duration)


