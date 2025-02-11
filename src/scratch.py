import numpy as np
import simulations as sim
from game.constants import GAME_SCALE

import os
import json
import argparse


logger = sim.logger


def load_parameters():

    files = os.listdir("cache")
    for i, file in enumerate(files):
        logger(f"{i}: {file}")

    ans = input("Select file: ")
    idx = -1 if ans == "" else int(ans)

    with open(f"cache/{files[idx]}", "r") as f:
        run_data = json.load(f)

    return run_data["info"]["record_genome"]["0"]["genome"]



reward_settings = {
    "rw_fetching": "probabilistic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.05 * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 5,
    "silent_duration": 5_000,
    "transparent": True,
}


game_settings = {
    "plot_interval": 1,
    "rw_event": "move agent",
    "rendering": False,
    "rendering_pcnn": True,
    "max_duration": 8_000,
    "room_thickness": 10,
    "seed": None,
    "pause": -1
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=-1)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--rendering", action="store_true")
    parser.add_argument("--room", type=str, default="Square.v0",
                        help='room name: ["Square.v0", "Square.v1", "Square.v2",' + \
                         '"Hole.v0", "Flat.0000", "Flat.0001", "Flat.0010", "Flat.0011",' + \
                         '"Flat.0110", "Flat.1000", "Flat.1001", "Flat.1010",' + \
                         '"Flat.1011", "Flat.1110"] or `random`')

    args = parser.parse_args()


    """ setup """

    if args.duration > 0:
        game_settings["max_duration"] = args.duration

    if args.rendering:
        game_settings["rendering"] = True

    reward_settings["transparent"] = args.transparent

    if args.load:
        parameters = load_parameters()
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
