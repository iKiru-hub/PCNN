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

