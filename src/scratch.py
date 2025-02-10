import numpy as np
import simulations as sim
from game.constants import GAME_SCALE


reward_settings = {
    "rw_fetching": "deterministic",
    "rw_value": "continuous",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.05 * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 50,
    "silent_duration": 5_000,
    "transparent": True,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "rendering": False,
    "rendering_pcnn": True,
    "max_duration": 8_000,
    "room_thickness": 30,
    "seed": None
}


if __name__ == "__main__":

    sim.logger("[@scratch.py]")
    out = sim.run_model(parameters=sim.parameters,
                        global_parameters=sim.global_parameters,
                        agent_settings=sim.agent_settings,
                        reward_settings=reward_settings,
                        game_settings=game_settings,
                        room_name="Square.v0",
                        verbose=False)

    sim.logger(f"rw_count={out}")
