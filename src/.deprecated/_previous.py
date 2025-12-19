def _initialize(sim_settings: dict = sim_settings,
                agent_settings: dict = agent_settings,
                model_params: dict = model_params):

    # --- settings
    duration = sim_settings["max_duration"]
    rendering = sim_settings["rendering"]
    rendering_pcnn = sim_settings["rendering_pcnn"]
    trg_position = sim_settings["rw_position"]
    trg_radius = sim_settings["rw_radius"]
    SPEED = sim_settings["speed"]
    PLOT_INTERVAL = sim_settings["plot_interval"]
    ROOM = sim_settings["room"]
    BOUNDS = sim_settings["bounds"]
    RW_BOUNDS = sim_settings["rw_bounds"]
    USE_GAME = sim_settings["use_game"]

    _weights = [w for (k, w) in model_params.items() if "w" in k.lower()]
    _num_weights = len(_weights)

    exp_weights = np.array(_weights[:5])
    ach_weights = np.array(_weights[5:])

    logger.debug(f"{rendering_pcnn=}")

    logger(f"room: {ROOM}")
    logger(f"plot_interval: {PLOT_INTERVAL}")
    logger(f"{duration}")

    # --- brain

    N = agent_settings["N"]
    sigma = agent_settings["sigma"]
    trg_threshold = agent_settings["trg_threshold"]

    logger(f"{N=}")
    logger(f"{sigma=}")
    logger(f"{trg_threshold=}")

    xfilter = pclib.GridNetwork([
                pclib.GridLayer(N=25, sigma=0.03,
                                speed=0.09,
                            init_bounds=[-1., 1., -1., 1.],
                            boundary_type="square",
                                basis_type="square"),
                pclib.GridLayer(N=25, sigma=0.03,
                                speed=0.07,
                               init_bounds=[-1., 1., -1., 1.],
                                boundary_type="square",
                                basis_type="square"),
                pclib.GridLayer(N=25, sigma=0.03,
                                speed=0.05,
                               init_bounds=[-1., 1., -1., 1.],
                                boundary_type="square",
                                basis_type="square"),
                pclib.GridLayer(N=25, sigma=0.03,
                                speed=0.02,
                            init_bounds=[-1., 1., -1., 1.],
                                boundary_type="square",
                                basis_type="square")])
    Nj = len(xfilter)
    pcnn_ = pclib.PCNNgrid(N=N, Nj=Nj, gain=8.,
                            offset=1.3,
                        clip_min=0.09,
                        threshold=0.02,
                rep_threshold=0.2,
                        rec_threshold=0.1,
                        num_neighbors=8, trace_tau=0.1,
                        xfilter=xfilter, name="2D")


    # xfilter = pclib.GridHexNetwork([
    #             pclib.GridHexLayer(sigma=0.03, speed=0.1),
    #             pclib.GridHexLayer(sigma=0.05, speed=0.09),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.08),
    #             pclib.GridHexLayer(sigma=0.03, speed=0.07),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.06)])

    # gcn = pclib.GridHexNetwork([
    #             pclib.GridHexLayer(sigma=0.03, speed=0.01),
    #             pclib.GridHexLayer(sigma=0.05, speed=0.009),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.008),
    #             pclib.GridHexLayer(sigma=0.03, speed=0.007),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.006)])

    # pcnn_ = pclib.PCNNgridhex(N=N,
    #                           Nj=len(gcn),
    #                           gain=10.,
    #                           offset=1.5,
    #                           clip_min=0.001,
    #                           threshold=0.02,
    #                           rep_threshold=0.3,
    #                           rec_threshold=0.1,
    #                           num_neighbors=8,
    #                           trace_tau=0.1,
    #                           xfilter=gcn, name="2D")
 
    # plotter
    pcnn2D_plotter = utc.PlotPCNN(model=pcnn_,
                                  bounds=BOUNDS,
                                  visualize=rendering_pcnn,
                                  number=0)

    # --- circuits
    circuits_dict = {
                     "DA": mod.Dopamine(N=N,
                                        threshold=0.15,
                            visualize=rendering,
                            pcnn_plotter2d=pcnn2D_plotter,
                                        number=5,
                            fig_standalone=True),
                     "Bnd": mod.BoundaryMod(N=N,
                                threshold=model_params["bnd_threshold"],
                                            eta=0.2,
                                            tau=model_params["bnd_tau"],
                            pcnn_plotter2d=pcnn2D_plotter,
                                            visualize=rendering,
                                            number=6,
                            fig_standalone=True),
                     "dPos": mod.PositionTrace(visualize=False),
                     "Pop": mod.PopulationProgMax(N=N,
                                                  visualize=False,
                                                  number=None),
                     "Ftg": mod.FatigueMod(tau=300)}

    # --- other circuits
    # densitymod = pclib.DensityMod(weights=ach_weights,
    #                               theta=1.)
    densitymod = None

    # object
    circuits = mod.Circuits(circuits_dict=circuits_dict,
                            # other_circuits={"ACh": densitymod},
                            visualize=rendering,
                            number=4)

    # --- modules

    trg_module = mod.TargetModule(pcnn=pcnn_,
                                  circuits=circuits,
                                  speed=SPEED,
                                  threshold=trg_threshold,
                                  visualize=rendering,
                                  number=1)
    logger(trg_module)

    # [ bnd, dpos, pop, trg, smooth ]
    exp_module = mod.ExperienceModule(pcnn=pcnn_,
                                      pcnn_plotter=pcnn2D_plotter,
                                      trg_module=trg_module,
                                      circuits=circuits,
                                      weights=exp_weights,
                                      max_depth=model_params["max_depth"],
                                      action_delay=model_params["action_delay"],
                                      speed=SPEED,
                                      visualize=rendering,
                                      number=2,
                                      number2=3)

    brain = mod.Brain(exp_module=exp_module,
                      circuits=circuits,
                      pcnn2D=pcnn_,
                      densitymod=densitymod)
    # brain = mod.randBrain(speed=SPEED,
    #                      pcnn2D=pcnn_)

    pcnn2D_plotter.add_element(element=exp_module)

    # --- agent & reward

    if USE_GAME:

        # --- room
        room = games.make_room(name=ROOM, thickness=5.)
        room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                       room.bounds[1]+10, room.bounds[3]-10]

        # --- objects
        body = games.objects.AgentBody(position=sim_settings["init_position"],
                                       possible_positions=possible_positions,
                                       bounds=room_bounds)
        reward_obj = games.objects.RewardObj(position=trg_position,
                                     radius=trg_radius,
                                     fetching=sim_settings["rw_fetching"],
                                     bounds=room_bounds)
        logger(reward_obj)

        # --- env
        env = games.Environment(room=room,
                                agent=body,
                                reward_obj=reward_obj,
                                scale=1,
                                rw_event=sim_settings["rw_event"],
                                verbose=False,
                                visualize=sim_settings["render_game"])
        logger(env)
    else:
        # --- objects
        body = ev.AgentBody(position=sim_settings["init_position"],
                            bounds=BOUNDS)
        reward_obj = ev.RewardObj(position=trg_position,
                                  radius=trg_radius,
                                  fetching=sim_settings["rw_fetching"],
                                  bounds=sim_settings["rw_bounds"])
        logger(reward_obj)

        # --- env
        room = ev.make_room(name=ROOM, thickness=4.,
                            bounds=BOUNDS,
                            visualize=rendering)

        env = ev.Environment(room=room,
                             agent=body,
                             reward_obj=reward_obj,
                             rw_event=sim_settings["rw_event"])
        pcnn2D_plotter.add_element(element=env)

    velocity = np.zeros(2)
    observation = {
        "position": env.position,
        "collision": False,
        "reward": 0.
    }

    configuration = {
        "brain": brain,
        "env": env,
        "reward_obj": reward_obj,
        "pcnn2D_plotter": pcnn2D_plotter,
        "observation": observation
    }

    return configuration

def _make_brain(sim_settings: dict = sim_settings,
                agent_settings: dict = agent_settings,
                model_params: dict = model_params):

    # --- settings
    duration = sim_settings["max_duration"]
    rendering = sim_settings["rendering"]
    rendering_pcnn = sim_settings["rendering_pcnn"]
    trg_position = sim_settings["rw_position"]
    trg_radius = sim_settings["rw_radius"]
    SPEED = sim_settings["speed"]
    PLOT_INTERVAL = sim_settings["plot_interval"]
    ROOM = sim_settings["room"]
    BOUNDS = sim_settings["bounds"]
    RW_BOUNDS = sim_settings["rw_bounds"]
    USE_GAME = sim_settings["use_game"]

    _weights = [w for (k, w) in model_params.items() if "w" in k.lower()]
    _num_weights = len(_weights)

    exp_weights = np.array(_weights[:5])
    ach_weights = np.array(_weights[5:])

    logger.debug(f"{rendering_pcnn=}")

    logger(f"room: {ROOM}")
    logger(f"plot_interval: {PLOT_INTERVAL}")
    logger(f"{duration}")

    # --- brain

    N = agent_settings["N"]
    sigma = agent_settings["sigma"]
    trg_threshold = agent_settings["trg_threshold"]

    logger(f"{N=}")
    logger(f"{sigma=}")
    logger(f"{trg_threshold=}")

    # xfilter = pclib.GridNetwork([
    #             pclib.GridLayer(N=25, sigma=0.03,
    #                             speed=0.09,
    #                         init_bounds=[-1., 1., -1., 1.],
    #                         boundary_type="square",
    #                             basis_type="square"),
    #             pclib.GridLayer(N=25, sigma=0.03,
    #                             speed=0.07,
    #                            init_bounds=[-1., 1., -1., 1.],
    #                             boundary_type="square",
    #                             basis_type="square"),
    #             pclib.GridLayer(N=25, sigma=0.03,
    #                             speed=0.05,
    #                            init_bounds=[-1., 1., -1., 1.],
    #                             boundary_type="square",
    #                             basis_type="square"),
    #             pclib.GridLayer(N=25, sigma=0.03,
    #                             speed=0.02,
    #                         init_bounds=[-1., 1., -1., 1.],
    #                             boundary_type="square",
    #                             basis_type="square")])
    # Nj = len(xfilter)
    # pcnn_ = pclib.PCNNgrid(N=N, Nj=Nj, gain=8.,
    #                         offset=1.3,
    #                     clip_min=0.09,
    #                     threshold=0.02,
    #             rep_threshold=0.2,
    #                     rec_threshold=0.1,
    #                     num_neighbors=8, trace_tau=0.1,
    #                     xfilter=xfilter, name="2D")


    # xfilter = pclib.GridHexNetwork([
    #             pclib.GridHexLayer(sigma=0.03, speed=0.1),
    #             pclib.GridHexLayer(sigma=0.05, speed=0.09),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.08),
    #             pclib.GridHexLayer(sigma=0.03, speed=0.07),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.06)])

    # gcn = pclib.GridHexNetwork([
    #             pclib.GridHexLayer(sigma=0.03, speed=0.01),
    #             pclib.GridHexLayer(sigma=0.05, speed=0.009),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.008),
    #             pclib.GridHexLayer(sigma=0.03, speed=0.007),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.006)])

    # pcnn_ = pclib.PCNNgridhex(N=N,
    #                           Nj=len(gcn),
    #                           gain=10.,
    #                           offset=1.5,
    #                           clip_min=0.001,
    #                           threshold=0.02,
    #                           rep_threshold=0.3,
    #                           rec_threshold=0.1,
    #                           num_neighbors=8,
    #                           trace_tau=0.1,
    #                           xfilter=gcn, name="2D")

    # SquareGrid
    if True:
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
 
    # plotter
    pcnn2D_plotter = utc.PlotPCNN(model=pcnn_,
                                  bounds=BOUNDS,
                                  visualize=rendering_pcnn,
                                  number=0)

    # --- circuits
    circuits_dict = {
                     "DA": mod.Dopamine(N=N,
                                        threshold=0.15,
                            visualize=rendering,
                            pcnn_plotter2d=pcnn2D_plotter,
                                        number=5,
                            fig_standalone=True),
                     "Bnd": mod.BoundaryMod(N=N,
                                threshold=model_params["bnd_threshold"],
                                            eta=0.2,
                                            tau=model_params["bnd_tau"],
                            pcnn_plotter2d=pcnn2D_plotter,
                                            visualize=rendering,
                                            number=6,
                            fig_standalone=True),
                     "dPos": mod.PositionTrace(visualize=False),
                     "Pop": mod.PopulationProgMax(N=N,
                                                  visualize=False,
                                                  number=None),
                     "Ftg": mod.FatigueMod(tau=300)}

    # --- other circuits
    # densitymod = pclib.DensityMod(weights=ach_weights,
    #                               theta=1.)
    densitymod = None

    # object
    circuits = mod.Circuits(circuits_dict=circuits_dict,
                            # other_circuits={"ACh": densitymod},
                            visualize=rendering,
                            number=4)

    # --- modules

    trg_module = mod.TargetModule(pcnn=pcnn_,
                                  circuits=circuits,
                                  speed=SPEED,
                                  threshold=trg_threshold,
                                  visualize=rendering,
                                  number=1)
    logger(trg_module)

    # [ bnd, dpos, pop, trg, smooth ]
    exp_module = mod.ExperienceModule(pcnn=pcnn_,
                                      pcnn_plotter=pcnn2D_plotter,
                                      trg_module=trg_module,
                                      circuits=circuits,
                                      weights=exp_weights,
                                      max_depth=model_params["max_depth"],
                                      action_delay=model_params["action_delay"],
                                      speed=SPEED,
                                      visualize=rendering,
                                      number=2,
                                      number2=3)


    brain = mod.BrainHex(exp_module=exp_module,
                      circuits=circuits,
                      pcnn2D=pcnn_,
                      densitymod=densitymod)
    # brain = mod.randBrain(speed=SPEED,
    #                      pcnn2D=pcnn_)

    pcnn2D_plotter.add_element(element=exp_module)

    return {"brain": brain,
            "pcnn2D_plotter": pcnn2D_plotter} 

def _make_game(sim_settings: dict = sim_settings,
               agent_settings: dict = agent_settings,
               model_params: dict = model_params):

    # --- settings
    duration = sim_settings["max_duration"]
    rendering = sim_settings["rendering"]
    rendering_pcnn = sim_settings["rendering_pcnn"]
    trg_position = sim_settings["rw_position"]
    trg_radius = sim_settings["rw_radius"]
    SPEED = sim_settings["speed"]
    PLOT_INTERVAL = sim_settings["plot_interval"]
    ROOM = sim_settings["room"]
    BOUNDS = sim_settings["bounds"]
    RW_BOUNDS = sim_settings["rw_bounds"]
    USE_GAME = sim_settings["use_game"]

    if USE_GAME:

        # --- room
        room = games.make_room(name=ROOM, thickness=5.)
        room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                       room.bounds[1]+10, room.bounds[3]-10]

        # --- objects
        body = games.objects.AgentBody(
                    position=sim_settings["init_position"],
                    possible_positions=possible_positions,
                    bounds=room_bounds)
        reward_obj = games.objects.RewardObj(
                    position=trg_position,
                    radius=trg_radius,
                    fetching=sim_settings["rw_fetching"],
                    bounds=room_bounds)
        logger(reward_obj)

        # --- env
        env = games.Environment(room=room,
                                agent=body,
                                reward_obj=reward_obj,
                                scale=1,
                                rw_event=sim_settings["rw_event"],
                                verbose=False,
                                visualize=sim_settings["render_game"])
        logger(env)
    else:
        # --- objects
        body = ev.AgentBody(position=sim_settings["init_position"],
                            bounds=BOUNDS)
        reward_obj = ev.RewardObj(position=trg_position,
                                  radius=trg_radius,
                                  fetching=sim_settings["rw_fetching"],
                                  bounds=sim_settings["rw_bounds"])
        logger(reward_obj)

        # --- env
        room = ev.make_room(name=ROOM, thickness=4.,
                            bounds=BOUNDS,
                            visualize=rendering)

        env = ev.Environment(room=room,
                             agent=body,
                             reward_obj=reward_obj,
                             rw_event=sim_settings["rw_event"])
        pcnn2D_plotter.add_element(element=env)

    velocity = np.zeros(2)
    observation = {
        "position": env.position,
        "collision": 0.,
        "reward": 0.
    }

    return {"env": env,
            "reward_obj": reward_obj,
            "pcnn2D_plotter": pcnn2D_plotter,
            "observation": observation}


def main_game(sim_settings=sim_settings,
              agent_settings=agent_settings,
              model_params=model_params):

    """
    meant to be run standalone
    """

    # --- settings
    sim_settings["rendering"] = False
    sim_settings["init_position"] = np.array([120, 440])
    sim_settings["rw_position"] = np.array([130, 450])
    sim_settings["rw_radius"] = 30
    sim_settings["use_game"] = True
    sim_settings["render_game"] = True

    configuration = _initialize(
        sim_settings=sim_settings,
        agent_settings=agent_settings,
        model_params=model_params,
    )

    brain = configuration["brain"]
    env = configuration["env"]
    reward_obj = configuration["reward_obj"]
    observation = configuration["observation"]
    pcnn2D_plotter = configuration["pcnn2D_plotter"]

    duration = sim_settings["max_duration"]
    PLOT_INTERVAL = sim_settings["plot_interval"]

    # --- record
    reward_count = 0
    trajectory = [env.position.tolist()]
    velocity = np.zeros(2)

    # -- run
    games.run_game(env=env,
                   brain=brain,
                   pcnn_plotter=pcnn2D_plotter,
                   # element=brain.circuits,
                   fps=30)


def main_game_rand(room_name: str="Square.v0"):

    """
    meant to be run standalone
    """

    # --- settings
    sim_settings = {
        "bounds": np.array([0.05, 0.95,
                            0.05, 0.95]) * GAME_SCALE,
        "speed": 0.01,
        "init_position": np.array([0.5, 0.5]) * GAME_SCALE,
        "rw_fetching": "deterministic",
        "rw_event": "move reward",
        "rw_position": np.array([0.5, 0.8]) * GAME_SCALE,
        "rw_radius": 0.1 * GAME_SCALE,
        "rw_bounds": np.array([0.2, 0.8, 0.2, 0.8]) * GAME_SCALE,
        "plot_interval": 1,
        "rendering": True,
        "rendering_pcnn": True,
        "render_game": True,
        "room": "square",
        "use_game": False,
        "max_duration": None,
        "seed": None
    }

    # brain
    gcn = pclib.GridHexNetwork([
                pclib.GridHexLayer(sigma=0.04, speed=0.2),
                pclib.GridHexLayer(sigma=0.04, speed=0.1),
                pclib.GridHexLayer(sigma=0.04, speed=0.05),
                pclib.GridHexLayer(sigma=0.04, speed=0.025),
                pclib.GridHexLayer(sigma=0.04, speed=0.0125)])

    pcnn_ = pclib.PCNNgridhex(N=100,
                              Nj=len(gcn),
                              gain=10.,
                              offset=1.5,
                              clip_min=0.001,
                              threshold=0.05,
                              rep_threshold=0.2,
                              rec_threshold=0.1,
                              num_neighbors=8,
                              trace_tau=0.1,
                              xfilter=gcn, name="2D")
    pcnn2D_plotter = utc.PlotPCNN(model=pcnn_,
                    bounds=sim_settings["bounds"],
                    visualize=sim_settings["rendering_pcnn"],
                    number=0)
    brain = mod.randBrain(speed=1.0,
                          pcnn2D=pcnn_)

    # --- room
    room = games.make_room(name=room_name, thickness=5.)
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    # --- objects
    body = games.objects.AgentBody(
                position=sim_settings["init_position"],
                possible_positions=None,
                bounds=room_bounds)
    reward_obj = games.objects.RewardObj(
                position=sim_settings["rw_position"],
                radius=sim_settings["rw_radius"],
                fetching=sim_settings["rw_fetching"],
                bounds=room_bounds)
    logger(reward_obj)

    # --- env
    env = games.Environment(room=room,
                            agent=body,
                            reward_obj=reward_obj,
                            scale=1,
                            rw_event=sim_settings["rw_event"],
                            verbose=False,
                            visualize=sim_settings["render_game"])
    logger(env)

    observation = {
        "position": env.position,
        "collision": False,
        "reward": 0.
    }

    # --- record
    reward_count = 0
    trajectory = [env.position.tolist()]
    velocity = np.zeros(2)

    # -- run
    games.run_game(env=env,
                   brain=brain,
                   pcnn_plotter=pcnn2D_plotter,
                   plotter_int=30,
                   fps=50)


def main_game_rand_2(room_name: str="Square.v0"):

    """
    meant to be run standalone
    """

    # --- settings
    sim_settings = {
        "bounds": np.array([0.05, 0.95,
                            0.05, 0.95]) * GAME_SCALE,
        "speed": 0.05,
        "init_position": np.array([0.5, 0.5]) * GAME_SCALE,
        "rw_fetching": "deterministic",
        "rw_event": "move reward",
        "rw_position": np.array([0.5, 0.8]) * GAME_SCALE,
        "rw_radius": 0.1 * GAME_SCALE,
        "rw_bounds": np.array([0.2, 0.8, 0.2, 0.8]) * GAME_SCALE,
        "plot_interval": 1,
        "rendering": True,
        "rendering_pcnn": True,
        "render_game": True,
        "room": "square",
        "use_game": False,
        "max_duration": None,
        "seed": None
    }

    # brain
    # gcn = pclib.GridHexNetwork([
    #             pclib.GridHexLayer(sigma=0.04, speed=0.2),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.1),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.05),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.025),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.0125)])
    # pcnn_ = pclib.PCNNgridhex(N=N,
    #                           Nj=len(gcn),
    #                           gain=10.,
    #                           offset=1.5,
    #                           clip_min=0.001,
    #                           threshold=0.05,
    #                           rep_threshold=0.2,
    #                           rec_threshold=0.1,
    #                           num_neighbors=8,
    #                           trace_tau=0.1,
    #                           xfilter=gcn, name="2D")


    # gcn = pclib.GridNetwork([pclib.GridLayer(N=9, sigma=0.04, speed=0.1, init_bounds=[-1, 0, -1, 0],
    #                           boundary_type="square"),
    #            pclib.GridLayer(N=9, sigma=0.04, speed=0.1, init_bounds=[0, 1, -1, 0],
    #                boundary_type="square"),
    #            pclib.GridLayer(N=9, sigma=0.04, speed=0.1, init_bounds=[-1, 0, 0, 1],
    #                boundary_type="square"),
    #            pclib.GridLayer(N=9, sigma=0.04, speed=0.1, init_bounds=[0, 1, 0, 1],
    #                boundary_type="square"),
    #            pclib.GridLayer(N=9, sigma=0.04, speed=0.05, init_bounds=[-1, 0, -1, 0],
    #                           boundary_type="square"),
    #            pclib.GridLayer(N=9, sigma=0.04, speed=0.05, init_bounds=[0, 1, -1, 0],
    #               boundary_type="square"),
    #            pclib.GridLayer(N=9, sigma=0.04, speed=0.05, init_bounds=[-1, 0, 0, 1],
    #               boundary_type="square"),
    #            pclib.GridLayer(N=9, sigma=0.04, speed=0.05, init_bounds=[0, 1, 0, 1],
    #               boundary_type="square"),
    #            pclib.GridLayer(N=9, sigma=0.04, speed=0.025, init_bounds=[-1, 0, -1, 0],
    #                           boundary_type="square"),
    #            pclib.GridLayer(N=9, sigma=0.04, speed=0.025, init_bounds=[0, 1, -1, 0],
    #               boundary_type="square"),
    #            pclib.GridLayer(N=9, sigma=0.04, speed=0.025, init_bounds=[-1, 0, 0, 1],
    #               boundary_type="square"),
    #            pclib.GridLayer(N=9, sigma=0.04, speed=0.025, init_bounds=[0, 1, 0, 1],
    #                           boundary_type="square")])

    # pcnn_ = pclib.PCNNgrid(N=N, Nj=len(gcn), gain=7., offset=1.1,
    #                        clip_min=0.01,
    #                        threshold=0.4,
    #                        rep_threshold=0.5,
    #                        rec_threshold=0.1,
    #                        num_neighbors=8, trace_tau=0.1,
    #                        xfilter=gcn, name="2D")

    """ PCNN """
    N = 144

    # --- Square PCNN
    gcn = pclib.GridNetworkSq([pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[-1., 1., -1., 1.]),
                               pclib.GridLayerSq(sigma=0.04, speed=0.07, bounds=[-1., 1., -1., 1.]),
                               pclib.GridLayerSq(sigma=0.04, speed=0.03, bounds=[-1., 1., -1., 1.]),])

    # space = pclib.PCNNsq(N=N, Nj=len(gcn), gain=7., offset=1.1,
    #                        clip_min=0.01,
    #                        threshold=0.4,
    #                        rep_threshold=0.5,
    #                        rec_threshold=0.1,
    #                        num_neighbors=8, trace_tau=0.1,
    #                        xfilter=gcn, name="2D")

    # gcn = pclib.GridNetworkSq([
    #            pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[-1, 0, -1, 0]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[0, 1, -1, 0]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[-1, 0, 0, 1]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[0, 1, 0, 1]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.07, bounds=[-1, 0, -1, 0]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.07, bounds=[0, 1, -1, 0]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.07, bounds=[-1, 0, 0, 1]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.07, bounds=[0, 1, 0, 1]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.03, bounds=[-1, 0, -1, 0]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.03, bounds=[0, 1, -1, 0]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.03, bounds=[-1, 0, 0, 1]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.03, bounds=[0, 1, 0, 1]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.015, bounds=[-1, 0, -1, 0]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.015, bounds=[0, 1, -1, 0]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.015, bounds=[-1, 0, 0, 1]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.015, bounds=[0, 1, 0, 1]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.005, bounds=[-1, 0, -1, 0]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.005, bounds=[0, 1, -1, 0]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.005, bounds=[-1, 0, 0, 1]),
    #            pclib.GridLayerSq(sigma=0.04, speed=0.005, bounds=[0, 1, 0, 1])
    # ])

    space = pclib.PCNNsq(N=N, Nj=len(gcn), gain=7., offset=1.4,
                           clip_min=0.01,
                           threshold=0.5,
                           rep_threshold=0.85,
                           rec_threshold=0.1,
                           num_neighbors=8, trace_tau=0.1,
                           xfilter=gcn, name="2D")
 
    pcnn2d = PCNNplotter(space,
                         max_iter=100_000)

    # --- hard-coded PCNN
    # gcn = pclib.GridNetworkSq([pclib.GridLayerSq(sigma=0.04,
    #                                              speed=0.1,
    #                                              bounds=[-1., 1.-1/6])])
    # space = pclib.PCNNbase(N, len(gcn), 5, 0.1, 0.01, 0.7, 0.1,
    #                     0.1, GAME_SCALE, 0.1, gcn, 20, "pcnn")

    # """ remaining components """

    # da = pclib.BaseModulation(name="DA", size=N, min_v=0.01, lr=0.1,
    #                           offset=0.01, gain=50.0)
    # bnd = pclib.BaseModulation(name="BND", size=N, min_v=0.01,
    #                            tau=1, lr=0.2, offset=0.01,
    #                            gain=50.0)
    # circuit = pclib.Circuits(da, bnd)

    # wrec = space.get_wrec()
    # trgp = pclib.TargetProgram(0., wrec,
    #                            da, 20, 0.)

    # eval_net = pclib.OneLayerNetwork([-0.5, 1., -1.1, 1.])
    # expmd = pclib.ExperienceModule(sim_settings["speed"],
    #                                circuit,
    #                                trgp, space, eval_net)

    # brain = pclib.Brain(circuit, space, trgp, expmd)

    # ---
    # gcn = pclib.GridNetworkSq([pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[-1., 1.-1/6, -1., -1.])])
    # space = pclib.PCNNbase(N, len(gcn), 20, 0.9, 0.1, 0.7, 0.1, 0.1, 10, 0.1, gcn, 1, "pcnn")

    da = pclib.BaseModulation(name="DA", size=N, min_v=0.1, lr=0.1, tau=1, clip=0.05,
                              offset=0.1, gain=50.0)
    bnd = pclib.BaseModulation(name="BND", size=N, min_v=0.9, clip=0.1,
                               tau=1, lr=0.2, offset=0.9,
                               gain=50.0)
    circuit = pclib.Circuits(da, bnd)

    wrec = space.get_wrec()
    trgp = pclib.TargetProgram(0., wrec,
                               da, 20, 0.)

    eval_net = pclib.OneLayerNetwork([-1., 1., 1., 2.])
    expmd = pclib.ExperienceModule(speed=sim_settings["speed"],
                                   circuits=circuit,
                                   trgp=trgp, space=space, eval_network=eval_net,
                                   max_depth=20, action_delay=1.0)

    brain = pclib.Brain(circuit, space, trgp, expmd)

    pcnn2D_plotter = utc.PlotPCNN(model=space,
                    bounds=sim_settings["bounds"],
                    visualize=sim_settings["rendering_pcnn"],
                    number=0)
    # --- room
    room = games.make_room(name=room_name, thickness=5.)
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    # --- objects
    body = games.objects.AgentBody(
                position=sim_settings["init_position"],
                possible_positions=None,
                bounds=room_bounds)
    reward_obj = games.objects.RewardObj(
                position=sim_settings["rw_position"],
                radius=sim_settings["rw_radius"],
                fetching=sim_settings["rw_fetching"],
                bounds=room_bounds)
    logger(reward_obj)

    # --- env
    env = games.Environment(room=room,
                            agent=body,
                            reward_obj=reward_obj,
                            scale=GAME_SCALE,
                            rw_event=sim_settings["rw_event"],
                            verbose=False,
                            visualize=sim_settings["render_game"])
    logger(env)

    observation = {
        "position": env.position,
        "collision": False,
        "reward": 0.
    }

    # --- record
    reward_count = 0
    trajectory = [env.position.tolist()]
    velocity = np.zeros(2)

    # --- rendered
    class Renderer:
        def __init__(self, element, space, color="Greens"):
            self.element = element
            self.space = space
            self.color = color
            self.fig, self.ax = plt.subplots()

        def render(self):
            self.ax.clear()
            plt.scatter(*np.array(self.space.get_centers()).T,
                        c=self.element.get_weights(),
                        cmap=self.color, s=50)
            plt.pause(0.01)

    renderer_da = Renderer(element=da, space=space)


    # -- run
    games.run_game(env=env,
                   brain=brain,
                   pcnn_plotter=pcnn2D_plotter,
                   plotter_int=50,
                   pcnn2d=pcnn2d,
                   fps=50, renderer_da=renderer_da)


""" from utils """


""" VISUALIZATION """


class PlotPCNN:

    def __init__(self, model: object,
                 visualize: bool=True,
                 number: int=None,
                 edges: bool=True,
                 bounds: tuple=(0, 1, 0, 1),
                 cmap: str='viridis'):

        self._model = model
        self._number = number
        self._bounds = bounds
        self._elements = []
        self.visualize = visualize
        if visualize:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._fig2, self._ax2 = plt.subplots(
                3, 1, figsize=(6, 6))
        # else:
        #     self._fig, self._ax = None, None

    def add_element(self, element: object):
        assert hasattr(element, "render"), \
            "element must have a render method"

        self._elements += [element]

    def render(self, trajectory: np.ndarray=None,
               rollout: tuple=None,
               edges: bool=True, cmap: str='RdBu_r',
               ax=None, new_a: np.ndarray=None,
               alpha_nodes: float=0.1,
               alpha_edges: float=0.2,
               return_fig: bool=False,
               render_elements: bool=False,
               customize: bool=False,
               draw_fig: bool=False,
               title: str=None):

        new_ax = True

        if ax is None:
            # fig, ax = plt.subplots(figsize=(6, 6))
            fig, ax = self._fig, self._ax
            ax.clear()
            new_ax = draw_fig

        # new_a = new_a if new_a is not None else self._model.u

        # render other elements
        if render_elements:
            for element in self._elements:
                element.render(ax=ax)

        # --- trajectory
        if trajectory is not None and len(trajectory) > 0:
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-',
                    lw=0.5,
                    alpha=0.5 if new_a is not None else 0.9)
            # ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
            #             c='k', s=150, marker='x')

        # --- rollout
        if rollout is not None and len(rollout[0]) > 0:
            rollout_trj, rollout_vals = rollout
            ax.plot(rollout_trj[:, 0], rollout_trj[:, 1], 'b',
                    lw=1, alpha=0.5, linestyle='--')
            for i, val in enumerate(rollout_vals):
                ax.scatter(rollout_trj[i, 0], rollout_trj[i, 1],
                           facecolors='white', edgecolors='blue',
                           s=10*(2+val), alpha=0.7, marker='o')

        # --- network
        centers = self._model.get_centers()
        connectivity = self._model.get_wrec()

        ax.scatter(centers[:, 0],
                   centers[:, 1],
                   c=new_a if new_a is not None else None,
                   s=40,# cmap=cmap,
                   # vmin=0, vmax=0.04,
                   alpha=alpha_nodes)

        if edges and new_a is not None:
            for i in range(connectivity.shape[0]):
                for j in range(connectivity.shape[1]):
                    if connectivity[i, j] > 0:
                        ax.plot([centers[i, 0], centers[j, 0]],
                                [centers[i, 1], centers[j, 1]],
                                'k-',
                                alpha=alpha_edges,
                                lw=0.5)

        #
        # ax.axis('off')
        if customize:
            # ax.axis('off')
            ax.set_xlim(self._bounds[0], self._bounds[1])
            ax.set_ylim(self._bounds[2], self._bounds[3])
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_aspect('auto')

        if title is None:
            title = f"PCNN | N={len(self._model)}"
        ax.set_title(title, fontsize=14)

        if self._number is not None and not new_ax:
            try:
                fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            except Exception as e:
                logger.debug(f"{e=}")
                return
            plt.close()
            return

        if not new_ax:
            fig.canvas.draw()

        if draw_fig:
            fig.canvas.draw()

        # if ax == self._ax:
        #     self._fig.canvas.draw()

        # second axis
        self._ax2[0].clear()
        self._ax2[0].imshow(
            self._model.get_activation().reshape(1, -1),
            aspect="auto", cmap="plasma", alpha=0.8,
        vmin=0., vmax=0.4)
        self._ax2[0].set_title(f"PCNN | max={self._model.get_activation().max():.3f}")

        self._ax2[1].clear()
        self._ax2[1].imshow(
            self._model.get_activation_gcn().reshape(1, -1),
            aspect="auto", cmap="plasma", alpha=0.8,
        vmin=0., vmax=0.99)
        self._ax2[1].set_title("GCN")

        self._ax2[2].clear()
        self._ax2[2].imshow(
            self._model.get_wff(),
                aspect="auto", cmap="plasma", alpha=0.8)

        plt.pause(0.001)

        if return_fig:
            return fig


def make_surface(points: np.ndarray):

    """
    make a surface from a set of points
    """

    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)

    # --- PLOT
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

    # plot convex hull
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1],
                points[simplex, 2], 'r-')

    plt.show()


""" FUNCTIONS """


def cosine_similarity(M: np.ndarray):

    """
    calculate the cosine similarity
    """

    # normalized matrix dot product
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    M = np.where(np.isnan(M), 0., M)
    return (M @ M.T) * (1 - np.eye(M.shape[0]))


def cosine_similarity_vec(x: np.ndarray,
                         y: np.ndarray) -> np.ndarray:

    """
    calculate the cosine similarity
    """

    y = y.reshape(-1, 1)
    x = x.reshape(1, -1)

    norms = (np.linalg.norm(x) * np.linalg.norm(y))

    if norms == 0:
        return 0.

    z = (x @ y) / norms
    if np.isnan(z):
        return 0.
    return z.item()


@jit(nopython=True)
def generalized_sigmoid(x: np.ndarray,
                        alpha: float,
                        beta: float,
                        clip_min: float=0.,
                        gamma: float=1.
                        ) -> np.ndarray:

    """
    generalized sigmoid function and set values below
    a certain threshold to zero.

    Parameters
    ----------
    x : np.ndarray
        the input
    alpha : float
        the threshold
    beta : float
        the slope
    gamma : float
        the intensity (height).
        Default is 1.
    clip_min : float
        the minimum value to clip.
        Default is 0.

    Returns
    -------
    np.ndarray
        The output array.
    """

    x = gamma / (1.0 + np.exp(-beta * (x - alpha)))

    return np.where(x < clip_min, 0., x)


def calc_position_from_centers(a: np.ndarray,
                               centers: np.ndarray) -> np.ndarray:

    """
    calculate the position of the agent from the
    activations of the neurons in the layer
    """

    if a.sum() == 0:
        return np.array([np.nan, np.nan])

    return (centers * a.reshape(-1, 1)).sum(axis=0) / a.sum()



""" ANALYSIS """


def _multiple_simulations(N: int, simulator: object,
                          use_tqdm: bool=True):

    """
    run multiple simulations
    """

    # --- INITIALIZATION
    # define initial positions as N points on a grid
    # over a box [0, 1] x [0, 1]

    # approximate N to the nearest square number
    if np.sqrt(N) % 1 != 0:
        N = int(np.sqrt(N)) ** 2
        logger.warning(f"Approximated N to the" + \
            f" nearest square number: {N}")
    xg = np.linspace(0.1, 0.9, int(np.sqrt(N)))
    yg = np.linspace(0.1, 0.9, int(np.sqrt(N)))
    all_init_positions = np.array(np.meshgrid(xg, yg)).T.reshape(-1, 2)

    # --- SIMULATION
    def run(simulator: object):

        done = False
        while not done:
            done = simulator.update()

        return simulator.get_trajectory(), simulator.get_reward_visit()
        # return simulator.get_pcnn_graph()

    data = []
    for i in tqdm(range(N), disable=not use_tqdm):

        # data += [run(simulator)]
        simulator.reset(init_position=all_init_positions[i])
        data += [run(simulator)]

    return N, data


def analysis_0(simulator: object):

    """
    plot the start and end positions of the trajectory,
    GOAL: highlight how the agent stays within the reward area
    """

    # --- RUN
    done = False
    simulator.reset()
    duration = simulator.max_duration
    for _ in tqdm(range(duration)):
        simulator.update()

    # --- PLOT
    fig, ax = plt.subplots(figsize=(6, 6))

    trajectory = np.array(simulator.get_trajectory())

    # reward
    rw_position, rw_radius = simulator.get_reward_info()

    # reward area
    ax.add_patch(plt.Circle(rw_position, rw_radius,
                            color="green", alpha=0.1))

    # trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1],
               lw=0.5, alpha=0.7)

    # start and end
    ax.scatter(trajectory[0, 0], trajectory[0, 1],
               marker="o", color="white", s=40,
               edgecolor="red")
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
               marker="o", color="red", s=40,
               edgecolor="red")


    # ax[i].axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def analysis_I(N: int, simulator: object):

    """
    plot the start and end positions of the trajectory,
    GOAL: highlight how the agent stays within the reward area
    """

    # --- RUN
    N, data = _multiple_simulations(N, simulator)

    # --- PLOT
    if len(data) > 10:
        num_per_col = 10
    else:
        num_per_col = len(data)
    ncols = min((N, num_per_col))
    nrows = N // num_per_col

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(13, 5))

    # reward
    rw_position, rw_radius = simulator.get_reward_info()
    for i, ax in enumerate(axs.flatten()):

        if i < len(data) and data[i][1]:

            # reward area
            ax.add_patch(plt.Circle(rw_position, rw_radius,
                                    color="green", alpha=0.1))

            # trajectory
            trajectory = np.array(data[i][0])
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                       lw=0.5, alpha=0.7)

            # start and end
            ax.scatter(trajectory[0, 0], trajectory[0, 1],
                       marker="o", color="white", s=40,
                       edgecolor="red")
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                       marker="o", color="red", s=40,
                       edgecolor="red")

            # graph
            # nodes, edges = data[i]
            # ax.scatter(nodes[:, 0], nodes[:, 1], s=10)

            ax.set_title(f"{i=}")
            # ax[i].axis("off")
            ax.set_aspect("equal")
            ax.set_xlim(0., 1.)
            ax.set_ylim(0., 1.)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ax.axis("off")

    plt.tight_layout()
    plt.show()


def analysis_II(N: int, simulator: object):

    """
    GOAL: highlight the fraction of simulations whose last part
    of the trajectory is spend within the reward area, given
    different reward positions
    """

    assert int(np.sqrt(N)) ** 2 == N, "N must be a perfect square"

    # reward positions
    # all_rw_positions = [
    #     [0.1, 0.1], [0.2, 0.5],
    #     [0.1, 0.9], [0.5, 0.7],
    #     [0.9, 0.9], [0.7, 0.5],
    #     [0.9, 0.1], [0.5, 0.2],
    #     [0.5, 0.5]
    # ]

    all_rw_positions = [
        [0.2, 0.2], [0.2, 0.8],
        [0.8, 0.8], [0.8, 0.2],
        [0.2, 0.5], [0.5, 0.7],
        [0.7, 0.2], [0.5, 0.2],
        [0.5, 0.5], [-10., -10.]
    ]
    rw_radius = simulator.get_reward_info()[1]
    NUM_TRIALS = len(all_rw_positions)

    # run & plot
    fig, axs = plt.subplots(nrows=2, ncols=NUM_TRIALS, figsize=(13, 5))
    fig.suptitle("Reward reaching accuracy in different positions",
                 fontsize=16)
    for i in tqdm(range(NUM_TRIALS)):

        simulator.set_rw_position(rw_position=all_rw_positions[i])
        _, data = _multiple_simulations(N, simulator, use_tqdm=False)

        # process:
        # average residuals for the last 70% of the trajectory
        residuals = []
        avg_end_positions = []
        all_positions = []
        for trajectory, _ in data:

            # average position
            avg_pos = np.array(trajectory[int(0.7 * len(trajectory)):]).mean(axis=0)
            avg_end_positions += [avg_pos]
            all_positions += [avg_pos.tolist()]

            # check if the average position is within the reward area
            # if np.linalg.norm(avg_pos - all_rw_positions[i]) < rw_radius:
            #     num_within += 1
            rw_position_i = all_rw_positions[i] if i < NUM_TRIALS - 1 else np.array([0.5, 0.5])
            residuals += [np.linalg.norm(avg_pos - rw_position_i)]

        accuracy = 1. - np.array(residuals)
        accuracy = np.flip(np.sort(accuracy))[:int(0.9 * N)]
        variance = np.var(all_positions, axis=0).mean()

        # plot:
        # A) plot
        # axs[0, i].bar(0, accuracy, color="green", alpha=0.8)
        axs[0, i].plot(range(len(accuracy)), accuracy, color="green", alpha=0.8)

        # variance as a shaded area around the mean
        axs[0, i].fill_between(range(len(accuracy)),
                              np.mean(accuracy) - variance,
                              np.mean(accuracy) + variance,
                              color="red", alpha=0.1)
        axs[0, i].axhline(np.mean(accuracy), color="red", lw=2.)

        if i == NUM_TRIALS - 1:
            axs[0, i].set_title(f"[baseline]\n{np.mean(accuracy):.2f}")
        else:
            axs[0, i].set_title(f"{np.mean(accuracy):.2f}")
        # axs[0, i].axis("off")
        axs[0, i].set_ylim(0., 1.)
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([0, 1.])
        axs[0, i].set_yticklabels(["0", "1"])
        axs[0, i].grid(True)

        # B) scatter plot of the reward area and the average end positions
        axs[1, i].add_patch(plt.Circle(rw_position_i, rw_radius,
                                      color="green", alpha=0.2))
        avg_end_positions = np.array(avg_end_positions)
        axs[1, i].scatter(avg_end_positions[:, 0], avg_end_positions[:, 1],
                         color="red", s=5)
        axs[1, i].set_aspect("equal")
        axs[1, i].set_xlim(0., 1.)
        axs[1, i].set_ylim(0., 1.)
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

    # save fig
    # import time
    # timestamp = time.strftime("%H%M%S")
    # fig.savefig(f"reward_reaching_accuracy_{timestamp}.png")
    # print("Figure saved as 'reward_reaching_accuracy.png'")

    plt.tight_layout()
    plt.show()




""" from evolution """


def load_model_settings(idx: int=None, verbose: bool=True) -> tuple:

    """
    load the results from an evolutionary search and
    return it as `sim_settings`, `agent_settings`, and
    `model_params`
    """

    import json
    def log(msg: str):
        if verbose:
            print(msg)

    # list all files
    file_list = os.listdir(EVOPATH)
    if len(file_list) == 0:
        return None, None, None

    # order the files based on their name [numbe]
    file_list = sorted(file_list, key=lambda x: int(x.split("_")[0]))

    log(f">>> files: {file_list}")

    if idx is None:
        idx = len(file_list) - 1
    log(f">>> idx: {idx}")
    filename = file_list[idx]

    log(f">>> loading file: {filename}")

    # load the file
    with open(f"{EVOPATH}/{filename}", "r") as f:
        data = json.load(f)

    info = data["info"]
    model_params = data["genome"]

    log(f"#date: {info['date']}")
    log(f"#evolution: {info['evolution']}")
    log(f"#evolved: {info['evolved']}")
    log(f"#other: '{info['other']}'")
    log(f"#performance: {info['performance']}")

    sim_settings = info["data"]["sim_settings"]
    agent_settings = info["data"]["agent_settings"]

    log(f"#sim_settings: {sim_settings}")
    log(f"#agent_settings: {agent_settings}")
    log(f"#model_params: {model_params}")

    # make some lists into numpy arrays
    sim_settings["bounds"] = np.array(sim_settings["bounds"])
    sim_settings["rw_position"] = np.array(sim_settings["rw_position"])
    sim_settings["init_position"] = np.array(sim_settings["init_position"])

    evo_info = {
        "date": info["date"],
        "evolution": info["evolution"],
        "evolved": info["evolved"],
        "other": info["other"],
        "performance": info["performance"]
    }

    return info["data"]["sim_settings"], \
              info["data"]["agent_settings"], \
                model_params, evo_info


""" from game """


def run_game_2(env: Environment,
             brain: object,
             pcnn_plotter: object = None,
             element: object = None,
             fps: int = 30,
             pcnn2d: object = None,
             plotter_int: int = 100,
             **kwargs):

    clock = pygame.time.Clock()
    last_position = np.zeros(2)

    # [position, velocity, collision, reward, done, terminated]
    observation = [env.position, np.array([0., 0.]), 0., 0.,
                   False, False]
    # observation = {
    #     "position": env.position,
    #     "collision": 0.,
    #     "reward": 0.
    # }

    renderer_da = kwargs.get("renderer_da", None)

    expmd = brain.get_expmd()

    running = True
    while running:

        # Event handling
        if env.visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # step
        # velocity = brain(observation)# * SCREEN_WIDTH
        # obs = [(observation["position"]-last_position).astype(np.float32).reshape(-1, 1),
        #        observation["collision"],
        #        observation["reward"]]
        velocity = brain(observation[1],
                         observation[2],
                         observation[3],
                         observation[0])
        if not isinstance(velocity, np.ndarray):
            velocity = np.array(velocity)

        # logger.debug("plan: " + str(expmd.get_plan()[0]))

        observation = env(velocity=velocity)

        pcnn2d(brain.get_representation(),
               observation[0] / env.scale)

        # reset agent's brain
        if observation[4]:
            logger.info(">> Game reset <<")
            brain.reset(position=env.agent.position)

        # update observation
        # observation["position"] = next_observation[1]
        # observation["collision"] = next_observation[2]
        # observation["reward"] = next_observation[0]
        # last_position = observation[0]

        # render
        if env.visualize:
            # env.render()

            if env.t % plotter_int == 0:
                # if pcnn_plotter is not None:
                #     pcnn_plotter.render(np.array(
                #         env.agent.trajectory),# /\
                #         # env.scale,
                #         customize=True,
                #         draw_fig=True,
                #         render_elements=True, 
                #         alpha_nodes=0.5,
                #         alpha_edges=0.2)

                # if element is not None:
                #     # element.render_circuits()
                #     element.circuits["DA"].render_field()
                #     element.circuits["Bnd"].render_field()

                # #
                # if renderer_da:
                #     renderer_da.render()

                if pcnn2d:
                    pcnn2d.render()

            # clock.tick(FPS)

        # exit 1
        if observation[4]:
            running = False

    pygame.quit()



def main_simple_square(duration: int):

    """ settings """

    SPEED = 5.
    BOUNDS = [0., 100.]
    N = 30**2
    action_delay = 3

    """ initialization """
    gcn = pclib.GridNetworkSq([
           pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.08, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.07, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.05, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.03, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.005, bounds=[-1, 1, -1, 1])])

    space = pclib.PCNNsqv2(N=N, Nj=len(gcn), gain=10., offset=1.1,
           clip_min=0.01,
           threshold=0.1,
           rep_threshold=0.4,
           rec_threshold=15.0,
           num_neighbors=5,
           xfilter=gcn, name="2D")

    #
    da = pclib.BaseModulation(name="DA", size=N, lr=0.9, threshold=0., max_w=2.0,
                              tau_v=2.0, eq_v=0.0, min_v=0.01)
    bnd = pclib.BaseModulation(name="BND", size=N, lr=0.99, threshold=0., max_w=2.0,
                               tau_v=1.0, eq_v=0.0, min_v=0.01)
    circuit = pclib.Circuits(da, bnd)

    trgp = pclib.TargetProgram(space.get_connectivity(), space.get_centers(),
                               da.get_weights(), SPEED)

    expmd = pclib.ExperienceModule(speed=SPEED,
                                   circuits=circuit,
                                   space=space, weights=[0., 0., 0.],
                                   max_depth=15, action_delay=action_delay)
    brain = pclib.Brain(circuit, space, trgp, expmd)

    plan_ = []

    # ---
    s = [SPEED, SPEED]
    points = [[14., 14.5]]
    x, y = points[0]

    tra = []
    color = "Greys"
    collision = 0.
    reward = 0.
    rx, ry, rs = 85, 30000, 5
    rt = 0
    rdur = 100
    nb_rw = 0
    delay = 0
    tplot = 10
    offset = 14

    pref = points[0]

    trg_plan = np.zeros(N)

    _, axs = plt.subplots(2, 2, figsize=(8, 8))
    ax1, ax2, ax3, ax4 = axs.flatten()

    for t in range(duration):

        # update sim
        x += s[0]
        y += s[1]

        # collision
        if x <= (BOUNDS[0]) or x >= (BOUNDS[1]):

            s[0] *= -1
            x += s[0]*2.
            #color = "Reds"
            delay = 10
            collision = 1.
        elif y <= (BOUNDS[0]) or y >= (BOUNDS[1]):
            s[1] *= -1
            y += s[1]*2.
            #color = "Oranges"
            delay = 10
            collision = 1.
        else:
            collision = 0.
            if delay == 0:
                color = "Greys"
            else:
                delay -= 1

        # reward
        if rt > 0:
            rt = max((0, rt-1));
            trigger = False;
            reward = 0
        else: 
            trigger = True;

        dist = np.sqrt((x-rx)**2 + (y-ry)**2)
        if dist < rs:
            rt = rdur
            nb_rw += 1
            reward = 1.
        else:
            reward = 0

        # record
        points += [[x, y]]

        if expmd.new_plan:
            pref = points[-1]

        # fwd
        s = brain(s, collision, reward, trigger)

        # trg directive
        if brain.get_directive() == "trg":
            trg_idxs = brain.get_plan()
            trg_plan *= 0
            trg_plan[trg_idxs] = 1.

        # get the plan
        pos_plan = np.array(brain.get_plan_positions(points[-1]))
        score_plan = np.array(brain.get_plan_scores())

        # plot
        if t % tplot == 0:

            # === 1
            ax1.clear()
            ax1.scatter(rx+4, ry+4, alpha=0.9, color='green', s=210, marker="x")
            if brain.get_directive() == "trg":
                hcolor = "green"
            else:
                hcolor = "red"
                # ax1.plot(*pos_plan.T, "b-", alpha=0.3)
                ax1.scatter(*pos_plan.T, c=score_plan, cmap="RdYlGn", alpha=0.98, s=30,
                            edgecolors="black", linewidths=1.)

            ax1.scatter(points[-1][0], points[-1][1], alpha=0.9, color=hcolor,
                        s=100)

            ax1.set_xlim(BOUNDS[0]-10, BOUNDS[1]+10)
            ax1.set_ylim(BOUNDS[0]-10, BOUNDS[1]+10)
            ax1.set_xticks(())
            ax1.set_yticks(())
            ax1.set_title(f"Trajectory [{t}] | #PCs:{len(space)} [#R={nb_rw}]")

            # === 2
            ax2.clear()
            if brain.get_directive() == "trg":
                ax2.scatter(*space.get_centers().T+offset, color="blue", alpha=0.1)
                ax2.scatter(*space.get_centers().T+offset, c=trg_plan, cmap="Greens", alpha=0.6)
                ax2.plot(*np.array(points).T[:, -10:], "g-", alpha=0.9)
            else:
                ax2.scatter(*space.get_centers().T+offset, color="blue", alpha=0.3)
                ax2.plot(*np.array(points).T, "r-", alpha=0.3)

            ax2.scatter(points[-1][0], points[-1][1], alpha=0.9, color='red', s=10)

            ax2.set_xlim(BOUNDS[0]-10, BOUNDS[1]+10)
            ax2.set_ylim(BOUNDS[0]-10, BOUNDS[1]+10)
            ax2.set_xticks(())
            ax2.set_yticks(())
            ax2.set_title(f"Map | trg: {trgp.is_active()}, {trigger=}")

            # === 3
            ax3.clear()
            ax3.scatter(*space.get_centers().T, c=da.get_weights(), s=30, cmap="Greens", alpha=0.5,
                        vmin=0., vmax=0.3)

            ax3.set_xlim(-5, 50)
            ax3.set_ylim(-5, 50)
            ax3.set_xticks(())
            ax3.set_yticks(())
            ax3.set_title(f"DA representation | maxw={da.get_weights().max():.3f}")

            # == 4
            ax4.clear()
            ax4.scatter(*space.get_centers().T, c=bnd.get_weights(), s=30, cmap="Blues", alpha=0.5,
                        vmin=0., vmax=0.3)
            ax4.scatter(points[-1][0], points[-1][1], alpha=0.9, color='red', s=10)

            # ax4.set_xlim(BOUNDS[0], BOUNDS[1])
            # ax4.set_ylim(BOUNDS[0], BOUNDS[1])
            ax4.set_xlim(-20, 120)
            ax4.set_ylim(-20, 120)
            ax4.set_xticks(())
            ax4.set_yticks(())
            ax4.set_title(f"BND representation | maxw={bnd.get_weights().max():.3f}")

            plt.pause(0.001)

    plt.show()



def main_game_rand(room_name: str="Square.v0"):

    SCALE = 100.0
    brain = objects.RandomAgent(scale=SCALE)

    room = games.make_room(name=room_name)
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    agent = objects.AgentBody(position=np.array([110, 110]),
                              width=25, height=25,
                              bounds=room_bounds,
                              possible_positions=[
                                    np.array([110, 110]),
                                    np.array([110, 190]),
                                    np.array([190, 110]),
                                    np.array([190, 190])],
                              max_speed=4.0)
    reward_obj = objects.RewardObj(position=np.array([150, 150]),
                                bounds=room_bounds)

    env = games.Environment(room=room, agent=agent,
                            reward_obj=reward_obj,
                            rw_event="move both",
                            duration=args.duration,
                            scale=SCALE,
                            visualize=True)

    run_game(env, brain, fps=100)



def main_game(room_name: str="Square.v0", load: bool=False, duration: int=-1):

    """
    meant to be run standalone
    """


    if load:
        parameters = utils.load_parameters()
        logger.debug(parameters)
    else:
        parameters = fixed_params

    """ make model """

    # ===| space |===

    local_scale_fine = global_parameters["local_scale_fine"]
    local_scale_coarse = global_parameters["local_scale_coarse"]

    gcn = pclib.GridNetworkSq([
           pclib.GridLayerSq(sigma=0.04, speed=1.*local_scale_fine, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.8*local_scale_fine, bounds=[-1, 1, -1, 1]),
           # pclib.GridLayerSq(sigma=0.04, speed=0.7*local_scale_fine, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.5*local_scale_fine, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.3*local_scale_fine, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.1*local_scale_fine, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.05*local_scale_fine, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.01*local_scale_fine, bounds=[-1, 1, -1, 1])])

    space_fine = pclib.PCNN(N=global_parameters["N"],
                            Nj=len(gcn),
                            gain=parameters["gain_fine"],
                            offset=parameters["offset_fine"],
                            clip_min=0.01,
                            threshold=parameters["threshold_fine"],
                            rep_threshold=parameters["rep_threshold_fine"],
                            rec_threshold=parameters["rec_threshold_fine"],
                            min_rep_threshold=parameters["min_rep_threshold"],
                            xfilter=gcn,
                            name="fine")

    # gcn_coarse = pclib.GridNetworkSq([
    #        pclib.GridLayerSq(sigma=0.04, speed=1.*local_scale_coarse,
    #                          bounds=[-1, 1, -1, 1]),
    #        pclib.GridLayerSq(sigma=0.04, speed=0.8*local_scale_coarse,
    #                          bounds=[-1, 1, -1, 1]),
    #        pclib.GridLayerSq(sigma=0.04, speed=0.7*local_scale_coarse,
    #                          bounds=[-1, 1, -1, 1]),
    #        pclib.GridLayerSq(sigma=0.04, speed=0.5*local_scale_coarse,
    #                          bounds=[-1, 1, -1, 1]),
    #        pclib.GridLayerSq(sigma=0.04, speed=0.3*local_scale_coarse,
    #                          bounds=[-1, 1, -1, 1]),
    #        pclib.GridLayerSq(sigma=0.04, speed=0.05*local_scale_coarse,
    #                          bounds=[-1, 1, -1, 1])])


    space_coarse = pclib.PCNN(N=global_parameters["Nc"],
                             Nj=len(gcn),
                             gain=parameters["gain_coarse"],
                             offset=parameters["offset_coarse"],
                             clip_min=0.01,
                             threshold=parameters["threshold_coarse"],
                             rep_threshold=parameters["rep_threshold_coarse"],
                             rec_threshold=parameters["rec_threshold_coarse"],
                             min_rep_threshold=parameters["min_rep_threshold"],
                             xfilter=gcn,
                             name="coarse")

    # ===| modulation |===

    da = pclib.BaseModulation(name="DA", size=global_parameters["N"],
                              lr=parameters["lr_da"],
                              threshold=parameters["threshold_da"],
                              max_w=1.0,
                              tau_v=1.0,
                              eq_v=0.0, min_v=0.0)
    bnd = pclib.BaseModulation(name="BND", size=global_parameters["N"],
                               lr=parameters["lr_bnd"],
                               threshold=parameters["threshold_bnd"],
                               max_w=1.0,
                               tau_v=1.0, eq_v=0.0, min_v=0.0)
    ssry = pclib.StationarySensory(global_parameters["N"],
                                   parameters["tau_ssry"],
                                   parameters["threshold_ssry"],
                                   0.99)
    circuit = pclib.Circuits(da, bnd, parameters["threshold_circuit"])

    # ===| target program |===

    dpolicy = pclib.DensityPolicy(parameters["rwd_weight"],
                                  parameters["rwd_sigma"],
                                  parameters["col_weight"],
                                  parameters["col_sigma"])

    expmd = pclib.ExplorationModule(speed=global_parameters["speed"]*2.0,
                                    circuits=circuit,
                                    space_fine=space_fine,
                                    action_delay=parameters["action_delay"],
                                    edge_route_interval=parameters["edge_route_interval"],)
    brain = pclib.Brain(circuit, space_fine, space_coarse, expmd, ssry, dpolicy,
                        global_parameters["speed"],
                        global_parameters["speed"]*local_scale_fine/local_scale_coarse,
                        parameters["forced_duration"],
                        parameters["fine_tuning_min_duration"],
                        global_parameters["min_weight_value"])

    """ use brain 3 """

    brain = pclib.Brainv3(
                local_scale_fine=global_parameters["local_scale_fine"],
                local_scale_coarse=global_parameters["local_scale_coarse"],
                N=global_parameters["N"],
                Nc=global_parameters["Nc"],
                min_rep_threshold=parameters["min_rep_threshold"],
                rec_threshold_fine=parameters["rec_threshold_fine"],
                rec_threshold_coarse=parameters["rec_threshold_coarse"],
                speed=global_parameters["speed"],
                gain_fine=parameters["gain_fine"],
                offset_fine=parameters["offset_fine"],
                threshold_fine=parameters["threshold_fine"],
                rep_threshold_fine=parameters["rep_threshold_fine"],
                gain_coarse=parameters["gain_coarse"],
                offset_coarse=parameters["offset_coarse"],
                threshold_coarse=parameters["threshold_coarse"],
                rep_threshold_coarse=parameters["rep_threshold_coarse"],
                lr_da=parameters["lr_da"],
                threshold_da=parameters["threshold_da"],
                tau_v_da=parameters["tau_v_da"],
                lr_bnd=parameters["lr_bnd"],
                threshold_bnd=parameters["threshold_bnd"],
                tau_v_bnd=parameters["tau_v_bnd"],
                tau_ssry=parameters["tau_ssry"],
                threshold_ssry=parameters["threshold_ssry"],
                threshold_circuit=parameters["threshold_circuit"],
                rwd_weight=parameters["rwd_weight"],
                rwd_sigma=parameters["rwd_sigma"],
                col_weight=parameters["col_weight"],
                col_sigma=parameters["col_sigma"],
                action_delay=parameters["action_delay"],
                edge_route_interval=parameters["edge_route_interval"],
                forced_duration=parameters["forced_duration"],
                fine_tuning_min_duration=parameters["fine_tuning_min_duration"],
                min_weight_value=parameters["fine_tuning_min_duration"])

    """ make game environment """

    room = games.make_room(name=room_name,
                           thickness=game_settings["room_thickness"],
                           bounds=[0, 1, 0, 1])
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    # ===| objects |===

    rw_position_idx = np.random.randint(0, len(constants.POSSIBLE_POSITIONS))
    rw_position = constants.POSSIBLE_POSITIONS[rw_position_idx]
    agent_possible_positions = constants.POSSIBLE_POSITIONS.copy()
    del agent_possible_positions[rw_position_idx]
    agent_position = agent_possible_positions[np.random.randint(0,
                                                len(agent_possible_positions))]

    reward_obj = objects.RewardObj(
                # position=reward_settings["rw_position"],
                position=rw_position,
                possible_positions=constants.POSSIBLE_POSITIONS.copy(),
                radius=reward_settings["rw_radius"],
                sigma=reward_settings["rw_sigma"],
                fetching=reward_settings["rw_fetching"],
                value=reward_settings["rw_value"],
                bounds=room_bounds,
                delay=reward_settings["delay"],
                silent_duration=reward_settings["silent_duration"],
                fetching_duration=reward_settings["fetching_duration"],
                transparent=reward_settings["transparent"])

    body = objects.AgentBody(
                # position=agent_settings["init_position"],
                position=agent_position,
                speed=global_parameters["speed"],
                possible_positions=agent_possible_positions,
                bounds=agent_settings["agent_bounds"],
                room=room,
                color=(10, 10, 10))

    logger(reward_obj)

    duration = game_settings["max_duration"] if duration < 0 else duration

    # --- env
    env = games.Environment(room=room,
                            agent=body,
                            reward_obj=reward_obj,
                            duration=duration,
                            rw_event=game_settings["rw_event"],
                            verbose=False,
                            visualize=game_settings["rendering"])
    logger(env)


    """ run game """

    if game_settings["rendering"]:
        renderer = Renderer(brain=brain, colors=["Greens", "Blues"],
                            names=["DA", "BND"])
    else:
        renderer = None

    logger("[@simulations.py]")
    run_game(env=env,
             brain=brain,
             renderer=renderer,
             plot_interval=game_settings["plot_interval"],
             pause=-1)

    logger(f"rw_count={env.rw_count}")



class EnvironmentWrapper(gym.Env):

    def __init__(self, env, image_obs: bool = False, resize_to=(84, 84)):
        super(EnvironmentWrapper, self).__init__()
        self.env = env
        self.image_obs = image_obs
        self.resize_to = resize_to
        self.prev_position = self.env.position.copy()
        self.speed = 1.

        # --- Action space: Discrete (left, right, up, down)
        self.action_space = spaces.Discrete(4)  # 0=left, 1=right, 2=up, 3=down

        # --- Observation space: image or vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        if self.env.visualize:
            logger.debug("env rendering")

    def set_speed(self, speed: float):
        self.speed = speed

    def reset(self, **kwargs):
        self.env.reset()
        self.prev_position = self.env.position.copy()
        velocity = [0.0, 0.0]
        obs = self._get_obs(velocity)
        return obs, {}

    def step(self, action):
        # Map discrete actions to velocity vectors
        if action == 0:   # left
            velocity = np.array([-self.speed, 0.])
        elif action == 1: # right
            velocity = np.array([self.speed, 0.])
        elif action == 2: # up
            velocity = np.array([0., -self.speed])
        elif action == 3: # down
            velocity = np.array([0., self.speed])
        else:
            raise ValueError("Invalid action index")

        prev_position = self.env.position.copy()
        obs_tuple = self.env(velocity=velocity, brain=None)
        new_position = self.env.position.copy()

        actual_velocity = [new_position[0] - prev_position[0],
                           -(new_position[1] - prev_position[1])]
        self.env.render()
        obs = self._get_obs(actual_velocity)
        # reward = float(obs_tuple[2]) if obs_tuple[2] is not False else 0.0
        reward = obs_tuple[2] if obs_tuple[2] > 0. else -1
        done = bool(obs_tuple[3]) or reward > 0.

        return obs, reward, done, False, {}

    def _get_obs(self, velocity):

        return np.array([
            self.env.position[0] / 1000,
            self.env.position[1] / 1000,
            self.env.velocity[1] / 10,
            self.env.velocity[0] / 10,
            float(self.env._reward),
            float(self.env._collision),
        ], dtype=np.float32)

    def render(self, mode='human'):
        return self.env.render()

    @property
    def duration(self):
        return self.env.duration

    @property
    def count(self):
        return self.env.count

    @property
    def visualize(self):
        return self.env.visualize

    @property
    def t(self):
        return self.env.t

    @property
    def rw_count(self):
        return self.env.rw_count


class EnvironmentWrapperIMG(gym.Env):
    def __init__(self, env, resize_to=(84, 84)):
        super(EnvironmentWrapper, self).__init__()
        self.env = env
        self.resize_to = resize_to

        # --- Action space: Discrete (left, right, up, down)
        self.action_space = spaces.Discrete(4)  # 0=left, 1=right, 2=up, 3=down

        # --- Observation space: image (1 channel), reward, collision, velocity
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(1, self.resize_to[1], self.resize_to[0]), dtype=np.uint8),
            'reward': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'collision': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'velocity': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })

        if self.env.visualize:
            logger.debug("env rendering")

    def set_speed(self, speed: float):
        self.env.speed = speed

    def reset(self, **kwargs):
        self.env.reset()
        image_obs = self._get_image_obs()
        reward_obs = np.array([float(self.env._reward)], dtype=np.float32)
        collision_obs = np.array([float(self.env._collision)], dtype=np.float32)
        velocity_obs = np.array(self.env.velocity.copy(), dtype=np.float32)
        obs = {
            'image': image_obs,
            'reward': reward_obs,
            'collision': collision_obs,
            'velocity': velocity_obs
        }
        return obs, {}

    def step(self, action):
        # Map discrete actions to velocity vectors
        if action == 0:    # left
            velocity = np.array([-self.env.speed, 0.])
        elif action == 1: # right
            velocity = np.array([self.env.speed, 0.])
        elif action == 2: # up
            velocity = np.array([0., -self.env.speed])
        elif action == 3: # down
            velocity = np.array([0., self.env.speed])
        else:
            raise ValueError("Invalid action index")

        obs_tuple = self.env(velocity=velocity, brain=None)

        self.env.render()

        image_obs = self._get_image_obs()
        reward = float(obs_tuple[2]) if obs_tuple[2] is not False else 0.0
        reward_obs = np.array([reward], dtype=np.float32)
        collision = float(obs_tuple[3]) if obs_tuple[3] is not False else 0.0
        collision_obs = np.array([collision], dtype=np.float32)
        velocity_obs = np.array(self.env.velocity.copy(), dtype=np.float32)
        obs = {
            'image': image_obs,
            'reward': reward_obs,
            'collision': collision_obs,
            'velocity': velocity_obs
        }
        done = bool(obs_tuple[3]) or reward > 0

        reward = reward if reward > 0 else -1

        return obs, reward, done, False, {}

    def _get_image_obs(self):
        surface = self.env.screen
        if surface is None:
            raise RuntimeError("self.env.screen is not set.")

        # Convert screen to array (W, H, C)
        raw = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        frame = np.transpose(raw, (1, 0, 2))  # (W, H, C) -> (H, W, C)

        # Convert to grayscale (single channel)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Normalize to [0, 1] and then scale to [0, 255] as integers
        normalized_gray = (gray / 255.0 * 255).astype(np.uint8)

        # Resize the frame
        resized = cv2.resize(normalized_gray, self.resize_to, interpolation=cv2.INTER_AREA)

        # Reshape to (C, H, W) = (1, H, W)
        resized = resized.reshape(1, self.resize_to[1], self.resize_to[0])

        return resized

    def render(self, mode='human'):
        return self.env.render()

    @property
    def duration(self):
        return self.env.duration

    @property
    def count(self):
        return self.env.count

    @property
    def visualize(self):
        return self.env.visualize

    @property
    def t(self):
        return self.env.t

    @property
    def rw_count(self):
        return self.env.rw_count


def run_cartpole(brain: object,
                 renderer: object,
                 duration: int,
                 t_goal: int=10,
                 t_rendering: int=10,
                 record_flag: bool=False,
                 verbose_min: bool=True):

    # ===| setup |===

    clock = pygame.time.Clock()
    last_position = np.zeros(2)

    # objects
    agent = objects.CartPoler(brain=brain)
    env = gym.make("CartPole-v1", render_mode="human")

    # [obs, reward, done, done, terminated]
    # observation: [position, velocity, angle, angular velocity]
    obs = env.reset()[0]
    reward = 0
    action = 0

    # starting position: [position, angle]
    prev_position = [obs[0], obs[2]]
    agent.reset(new_position=prev_position)

    # init
    record = {"activity_fine": [],
              "activity_coarse": [],
              "scores": [],
              "trajectory": []}

    # ===| main loop |===
    score = 0
    eps_count = 0
    for t in tqdm(range(duration), desc="Game", leave=False,
                  disable=not verbose_min):

        # env step
        obs, reward, done, terminated, info = env.step(action)
        score += reward
        velocity = [(obs[0] - prev_position[0]) * 100,
                    (obs[2] - prev_position[1]) * 100]

        collision_ = float(done)
        reward_ = np.exp(-obs[2]**2 / 0.01)

        # brain step
        action = agent(velocity, reward_, collision_, t>=t_goal)
        prev_position = [obs[0], obs[2]]

        # -check: render
        if t > t_rendering:
            env.render()
            renderer.render()

        # -check: record
        if record_flag:
            record["activity_fine"] += [brain.get_representation_fine()]
            record["activity_coarse"] += [brain.get_representation_coarse()]
            record["trajectory"] += [env.position]

        # -check: done
        if done:
            obs = env.reset()[0]
            agent.reset(new_position=[obs[0], obs[2]])
            record["scores"] += [score]
            logger(f"[end episode] - score={score}")
            logger(f"new position: {obs[0]:.2f}, {obs[2]:.2f}")
            score = 0

        # -check: terminated
        if terminated:
            break

    record["num_eps"] = eps_count

    return record

