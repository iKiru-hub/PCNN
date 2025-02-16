import numpy as np
import simulations as sim

import os
import json
import argparse

from game.constants import GAME_SCALE
import utils



reward_settings = {
    "rw_fetching": "deterministic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.05 * GAME_SCALE,
    "rw_sigma": 1.5 * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 10,
    "silent_duration": 3_000,
    "fetching_duration": 1,
    "transparent": False,
    "beta": 35.,
    "alpha": 0.06,
}


game_settings = {
    "plot_interval": 100,
    "rw_event": "move agent",
    "rendering": False,
    "rendering_pcnn": True,
    "max_duration": 10_000,
    "room_thickness": 10,
    "seed": None,
    "pause": -1,
    "verbose": True
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=-1)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--rendering", action="store_true")
    parser.add_argument("--interval", type=int, default=20,
                        help="plotting interval")
    parser.add_argument("--room", type=str, default="Square.v0",
                        help='room name: ["Square.v0", "Square.v1", "Square.v2",' + \
                         '"Hole.v0", "Flat.0000", "Flat.0001", "Flat.0010", "Flat.0011",' + \
                         '"Flat.0110", "Flat.1000", "Flat.1001", "Flat.1010",' + \
                         '"Flat.1011", "Flat.1110"] or `random`')

    args = parser.parse_args()


    """ setup """

    logger = utils.setup_logger(name="RUN",
                                level=2,
                                is_debugging=True,
                                is_warning=False)


    if args.duration > 0:
        game_settings["max_duration"] = args.duration

    if args.rendering:
        game_settings["rendering"] = True

    game_settings["plot_interval"] = args.interval

    reward_settings["transparent"] = args.transparent

    if args.load:
        parameters = utils.load_parameters()
        logger.debug(parameters.keys())
    else:
        parameters = sim.parameters

    """ run """

    logger("[@scratch.py]")
    out = sim.run_model(parameters=parameters,
                        global_parameters=sim.global_parameters,
                        agent_settings=sim.agent_settings,
                        reward_settings=reward_settings,
                        game_settings=game_settings,
                        room_name=args.room,
                        verbose=False, pause=game_settings["pause"])

    logger(f"rw_count={out}")
