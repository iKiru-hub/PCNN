import numpy as np
import matplotlib.pyplot as plt
import argparse, json
from tqdm import tqdm

# import mod_core as mod
import utils_core as utc
# import envs_core as ev
from pcnn_core import PCNNplotter

import game.envs as games

try:
    # import libs.pclib as pclib
    import core.build.pclib as pclib
except ImportError:
    import warnings
    warnings.warn("pclib [c++] not found, using python version")
    import libs.pclib1 as pclib


""" INITIALIZATION  """


CONFIGPATH = "dashboard/cache/configs.json"

logger = utc.setup_logger(name="RUN",
                          level=2,
                          is_debugging=True,
                          is_warning=False)


def edit_logger(level: int=-1,
                is_debugging: bool=True,
                is_warning: bool=False):
    global logger
    logger.set_level(level)
    logger.set_debugging(is_debugging)
    logger.set_warning(is_warning)


""" SETTINGS """

GAME_SCALE = games.SCREEN_WIDTH


sim_settings = {
    "bounds": np.array([0., 1., 0., 1.]) * GAME_SCALE,
    "speed": 7.0,
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

agent_settings = {
    "N": 200,
    "Nj": 13**2,
    "sigma": 0.03 * GAME_SCALE,
    "max_depth": 20,
    "trg_threshold": 0.0
}

possible_positions = np.array([
    [0.2, 0.2], [0.2, 0.8],
    [0.8, 0.2], [0.8, 0.8],
    [0.8, 0.3], [0.8, 0.8],
    [0.3, 0.8], [0.3, 0.8],
]) * GAME_SCALE

model_params = {
    "bnd_threshold": 0.2,
    "bnd_tau": 1,
    "threshold": 0.3,
    "rep_threshold": 0.5,
    "action_delay": 6,
    "max_depth": 4,

    "w1": -0.6, "w2": 0.0, "w3": -3.0, "w4": 0.4, "w5": 1.4, "w6": -1.8, "w7": 0.2, "w8": -2.0, "w9": 1.4, "w10": -1.6}

model_params_mlp = model_params | {
    "w6": 0.3,
    "w7": 0.2,
    "w8": 0.1,
    "w9": 0.1,
    "w10": 0.1,
    "w11": 0.1,
    "w12": 0.1,
}


""" RUN CLASSES """


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


def main_game_rand_3(room_name: str="Square.v0"):

    """
    meant to be run standalone
    """

    # --- settings
    sim_settings = {
        "bounds": np.array([0.05, 0.95,
                            0.05, 0.95]) * GAME_SCALE,
        "speed": 1.,
        "init_position": np.array([0.5, 0.5]) * GAME_SCALE,
        "rw_fetching": "deterministic",
        "rw_event": "nothing",
        "rw_position": np.array([0.9, 0.8]) * GAME_SCALE,
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

    """ PCNN """
    N = 30**2

    # --- Square PCNN
    gcn = pclib.GridNetworkSq([
           pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.08, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.07, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.05, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.03, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.005, bounds=[-1, 1, -1, 1])])

    space = pclib.PCNNsqv2(N=N, Nj=len(gcn), gain=10., offset=1.4,
           clip_min=0.01,
           threshold=0.3,
           rep_threshold=0.9,
           rec_threshold=4.,
           num_neighbors=5,
           xfilter=gcn, name="2D")
 
    da = pclib.BaseModulation(name="DA", size=N, lr=0.9, threshold=0., max_w=1.0,
                              tau_v=2.0, eq_v=0.0, min_v=0.001)
    bnd = pclib.BaseModulation(name="BND", size=N, lr=0.99, threshold=0., max_w=1.0,
                               tau_v=2.0, eq_v=0.0, min_v=0.001)
    circuit = pclib.Circuits(da, bnd)

    trgp = pclib.TargetProgram(space.get_connectivity(), space.get_centers(),
                               da.get_weights(), sim_settings["speed"])

    expmd = pclib.ExperienceModule(speed=sim_settings["speed"],
                                   circuits=circuit,
                                   space=space, weights=[0., 0., 1.],
                                   max_depth=15, action_delay=4)
    brain = pclib.Brain(circuit, space, trgp, expmd)

    # # ---
    # pcnn2d = PCNNplotter(space,
    #                      max_iter=100_000)
    # pcnn2D_plotter = utc.PlotPCNN(model=space,
    #                 bounds=sim_settings["bounds"],
    #                 visualize=sim_settings["rendering_pcnn"],
    #                 number=0)

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

    # --- rendered
    class Renderer:
        def __init__(self, elements=[da, bnd], space=space,
                     colors=["Greens", "Blues"],
                     names=["DA", "BND"]):

            self.elements = elements
            self.size = len(elements)
            self.space = space
            self.colors = colors
            self.names = names
            self.fig, self.axs = plt.subplots(1, self.size+1,
                                             figsize=((1+self.size)*4, 2))
            self.bounds = (-30, 30)

        def render(self):
            self.axs[0].clear()
            self.axs[0].scatter(*np.array(self.space.get_centers()).T,
                        color="blue", s=30, alpha=0.5)
            self.axs[0].scatter(*np.array(self.space.get_position()).T,
                                color="red", s=80, marker="o")
            for edge in self.space.make_edges():
                self.axs[0].plot((edge[0][0], edge[1][0]), (edge[0][1], edge[1][1]),
                            alpha=0.1, color="black")
            self.axs[0].set_xlim(self.bounds)
            self.axs[0].set_ylim(self.bounds)
            # equal aspect ratio
            self.axs[0].set_aspect('equal', adjustable='box')
            self.axs[0].set_title(f"Space | #PCs={len(self.space)}")

            for i in range(1, 1+self.size):
                self.axs[i].clear()
                self.axs[i].scatter(*np.array(self.space.get_centers()).T,
                            c=self.elements[i-1].get_weights(),
                            cmap=self.colors[i-1], alpha=0.5,
                            s=30, vmin=0., vmax=0.1)
                self.axs[i].set_xlim(self.bounds)
                self.axs[i].set_ylim(self.bounds)
                self.axs[i].set_title(self.names[i-1])
                self.axs[i].set_aspect('equal', adjustable='box')

            # plt.axis("equal")
            plt.pause(0.00001)


    renderer = Renderer()

    # -- run
    games.run_game(env=env,
                   brain=brain,
                   plotter_int=1,
                   fps=1, renderer=renderer)
    # games.run_game(env=env,
    #                brain=brain,
    #                pcnn_plotter=None,
    #                plotter_int=50,
    #                pcnn2d=None,
    #                fps=10, renderer_da=renderer_da)



""" simple game """


def main_simple_game(duration):

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




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--seed", type=int, default=-1,
                        help="random seed: -1 for random seed.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--room", type=str, default="Square.v0")

    args = parser.parse_args()

    #
    if args.seed > 0:
        logger.debug(f"seed: {args.seed}")
        np.random.seed(args.seed)

    # run
    sim_settings["seed"] = args.seed
    sim_settings["max_duration"] = args.duration
    sim_settings["rendering"] = not args.plot
    sim_settings["room"] = args.room
    sim_settings["use_game"] = True 

    agent_settings["N"] = args.N
    # main_game(sim_settings=sim_settings,
    #           agent_settings=agent_settings,
    #           model_params=model_params)

    # main_game_rand(room_name=args.room)
    # main_game_rand_2(room_name=args.room)
    # main_game_rand_3(room_name=args.room)
    main_simple_game(duration=args.duration)

