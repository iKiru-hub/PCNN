import numpy as np
import matplotlib.pyplot as plt
import argparse, json
from tqdm import tqdm

import mod_core as mod
import utils_core as utc
import envs_core as ev

import game.envs as games

try:
    import libs.pclib as pclib
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
    "action_delay": 7,
    "max_depth": 20,

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

    # xfilter = pclib.GridNetwork([
    #             pclib.GridLayer(N=25, sigma=0.05, speed=0.9, init_bounds=[-1., 1., -1., 1.],
    #                             boundary_type="square", basis_type="square"),
    #             pclib.GridLayer(N=25, sigma=0.2, speed=0.7, init_bounds=[-1., 1., -1., 1.],
    #                             boundary_type="square", basis_type="random_square"),
    #             pclib.GridLayer(N=25, sigma=0.03, speed=0.9, init_bounds=[-1., 1., -1., 1.],
    #                             boundary_type="hexagon", basis_type="random_circle")])
    # Nj = len(xfilter)
    # pcnn2D = pclib.PCNNgrid(N=N, Nj=Nj, gain=3., offset=1.,
    #                     clip_min=0.09,
    #                     threshold=model_params["threshold"],
    #                     rep_threshold=model_params["rep_threshold"],
    #                     rec_threshold=0.1,
    #                     num_neighbors=8, trace_tau=0.1,
    #                     xfilter=xfilter, name="2D")


    # xfilter = pclib.GridHexNetwork([
    #             pclib.GridHexLayer(sigma=0.03, speed=0.1),
    #             pclib.GridHexLayer(sigma=0.05, speed=0.09),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.08),
    #             pclib.GridHexLayer(sigma=0.03, speed=0.07),
    #             pclib.GridHexLayer(sigma=0.04, speed=0.06)])

    gcn = pclib.GridHexNetwork([
                pclib.GridHexLayer(sigma=0.04, speed=0.8),
                pclib.GridHexLayer(sigma=0.04, speed=0.72),
                pclib.GridHexLayer(sigma=0.04, speed=0.59),
                pclib.GridHexLayer(sigma=0.04, speed=0.46),
                pclib.GridHexLayer(sigma=0.04, speed=0.36),
                pclib.GridHexLayer(sigma=0.04, speed=0.11)])

    pcnn_ = pclib.PCNNgridhex(N=N,
                              Nj=len(gcn),
                              gain=8.,
                              offset=1.4,
                              clip_min=0.01,
                              threshold=0.5,
                              rep_threshold=0.9,
                              rec_threshold=0.1,
                              num_neighbors=8,
                              trace_tau=0.1,
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

    # brain = mod.Brain(exp_module=exp_module,
    #                   circuits=circuits,
    #                   pcnn2D=pcnn_,
    #                   densitymod=densitymod)
    brain = mod.randBrain(speed=SPEED,
                         pcnn2D=pcnn_)

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

    main_game_rand(room_name=args.room)

