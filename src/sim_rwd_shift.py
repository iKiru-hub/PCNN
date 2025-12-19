import numpy as np
import matplotlib.pyplot as plt
import argparse, json, os
from tqdm import tqdm
import pygame

import utils
import game.envs as games
import game.objects as objects
import game.constants as constants

try:
    # import libs.pclib as pclib
    import core.build.pclib as pclib
    import libs.pclib2 as pclib2
except ImportError:
    import warnings
    warnings.warn("pclib [c++] not found, using python version")
    import libs.pclib1 as pclib

logger = utils.setup_logger(__name__, level=2)


""" SETTINGS """

GAME_SCALE = games.SCREEN_WIDTH
ENV = "Square.v0"
# ENV = "Flat.0001"


reward_settings = {
    "rw_fetching": "deterministic",
    "rw_value": "discrete",
    "move_period": 5000,
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.05 * GAME_SCALE,
    "rw_sigma": 0.8,# * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 20,
    "silent_duration": 1_000,
    "fetching_duration": 2,
    "transparent": False,
    "beta": 40.,
    "alpha": 0.06,# * GAME_SCALE,
    "tau": 300,# * GAME_SCALE,
    "move_threshold": 8,# * GAME_SCALE,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "rendering": True,
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "max_duration": 20_000,
    "room_thickness": 30,
    "t_teleport": 1000,
    "limit_position_len": 1,
    "seed": None,
    "pause": -1,
    "verbose": False
}

global_parameters = {
    "local_scale": 0.02,
    # "local_scale_fine": 0.02,
    # "local_scale_coarse": 0.006,
    "N": 27**2,
    # "N": 42**2,
    # "Nc": 35**2,
    "use_sprites": False,
    "speed": 0.9,
    "min_weight_value": 0.5
}

parameters = {
        "gain": 25.0,
        "offset": 1.05,
        "threshold": 0.3,
        "rep_threshold": 0.9,
        "rec_threshold": 63,
        "tau_trace": 20,
        "remap_tag_frequency": 1,
        "min_rep_threshold": 0.95,

        "lr_da": 0.9,
        "lr_pred": 0.01,
        "threshold_da": 0.04,
        "tau_v_da": 3.0,

        "lr_bnd": 0.9,
        "threshold_bnd": 0.01,
        "tau_v_bnd": 3.0,

        "tau_ssry": 437.0,
        "threshold_ssry": 1.986,
        "threshold_circuit": 0.9,

        "rwd_weight": 0.7,
        "rwd_sigma": 80.,
        "rwd_threshold": 0.49,
        "col_weight": 0.,
        "col_sigma": 2.6,
        "col_threshold": 0.17,
        "rwd_field_mod": 1.0,
        "col_field_mod": 2.6,

        "action_delay": 120.0,
        "edge_route_interval": 10,
        "forced_duration": 19,
        "min_weight_value": 0.2
}

parameters_2 = {
      "gain": 102.4,
      "offset": 1.02,
      "threshold": 0.4,
      "rep_threshold": 0.999,
      "rec_threshold": 33,
      "tau_trace": 10,
      "remap_tag_frequency": 1,
      "num_neighbors": 4,
      "min_rep_threshold": 0.99,
      "lr_da": 0.9,
      "lr_pred": 0.05,
      "threshold_da": 0.05,
      "tau_v_da": 1.0,
      "lr_bnd": 0.9,
      "threshold_bnd": 0.01,
      "tau_v_bnd": 1.0,
      "tau_ssry": 437.0,
      "threshold_ssry": 1.986,
      "threshold_circuit": 0.9,
      "rwd_weight": -2.11,
      "rwd_sigma": 96.8,
      "rwd_threshold": 0.49,
      "col_weight": -0.83,
      "col_sigma": 26.1,
      "col_threshold": 0.07,
      "rwd_field_mod": 4.6,
      "col_field_mod": 5.4,
      "action_delay": 120.0,
      "edge_route_interval": 50,
      "forced_duration": 19,
      "min_weight_value": 0.1,
    "modulation_options": [True]*4
}


fixed_params = parameters.copy()


possible_positions = np.array([
    [0.25, 0.75], [0.75, 0.75],
    [0.25, 0.25], [0.75, 0.25]]) * GAME_SCALE


def run_game_sil(global_parameters: dict=global_parameters,
                 reward_settings: dict=reward_settings,
                 game_settings: dict=game_settings,
                 load_idx: int=-1,
                 load: bool=False):

    """
    meant to be run standalone
    """

    if load:
        load_idx = load_idx if load_idx > -1 else None
        # load_idx = None
        parameters = utils.load_parameters(idx=load_idx)
    else:
        parameters = fixed_params

    """ make model """

    remap_tag_frequency = parameters["remap_tag_frequency"] if "remap_tag_frequency" in parameters else 200
    remapping_flag = parameters["remapping_flag"] if "remapping_flag" in parameters else 0
    lr_pred = parameters["lr_pred"] if "lr_pred" in parameters else 0.2

    brain = pclib2.Brain(
                local_scale=global_parameters["local_scale"],
                N=global_parameters["N"],
                rec_threshold=parameters["rec_threshold"],
                speed=global_parameters["speed"],
                min_rep_threshold=parameters["min_rep_threshold"],
                gain=parameters["gain"],
                offset=parameters["offset"],
                threshold=parameters["threshold"],
                rep_threshold=parameters["rep_threshold"],
                tau_trace=parameters["tau_trace"],
                remap_tag_frequency=parameters["remap_tag_frequency"],
                lr_da=parameters["lr_da"],
                lr_pred=parameters["lr_pred"],
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
                rwd_threshold=parameters["rwd_threshold"],
                col_weight=parameters["col_weight"],
                col_sigma=parameters["col_sigma"],
                col_threshold=parameters["col_threshold"],
                rwd_field_mod=parameters["rwd_field_mod"],
                col_field_mod=parameters["col_field_mod"],
                action_delay=parameters["action_delay"],
                edge_route_interval=parameters["edge_route_interval"],
                forced_duration=parameters["forced_duration"],
                min_weight_value=parameters["min_weight_value"])

    """ make game environment """

    room = games.make_room(name=ENV,
                           thickness=game_settings["room_thickness"],
                           bounds=[0, 1, 0, 1])
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    # ===| objects |===

    possible_positions = room.get_room_positions()

    agent_possible_positions = possible_positions.copy()
    agent_position = room.sample_next_position()

    rw_tau = reward_settings["tau"] if "tau" in reward_settings else 400
    if "move_threshold" in reward_settings:
        rw_move_threshold = reward_settings["move_threshold"]
    else:
        rw_move_threshold = 2

    reward_obj = objects.RewardObj(
                position=possible_positions[0],
                possible_positions=possible_positions,
                radius=reward_settings["rw_radius"],
                sigma=reward_settings["rw_sigma"],
                fetching=reward_settings["rw_fetching"],
                value=reward_settings["rw_value"],
                bounds=room_bounds,
                delay=reward_settings["delay"],
                silent_duration=reward_settings["silent_duration"],
                fetching_duration=reward_settings["fetching_duration"],
                use_sprites=global_parameters["use_sprites"],
                tau=rw_tau,
                move_threshold=rw_move_threshold,
                move_period=reward_settings["move_period"],
                transparent=reward_settings["transparent"])

    body = objects.AgentBody(
                position=agent_position,
                speed=global_parameters["speed"],
                possible_positions=agent_possible_positions,
                bounds=game_settings["agent_bounds"],
                use_sprites=global_parameters["use_sprites"],
                limit_position_len=game_settings["limit_position_len"],
                room=room,
                color=(10, 10, 10))

    logger(reward_obj)

    duration = game_settings["max_duration"]
    verbose_min = True
    verbose = False
    record_flag = False
    pause = -1
    t_teleport=game_settings["t_teleport"]
    plot_interval=game_settings["plot_interval"]

    # --- env
    env = games.Environment(room=room,
                            agent=body,
                            reward_obj=reward_obj,
                            duration=duration,
                            rw_event=game_settings["rw_event"],
                            verbose=False,
                            visualize=False)
    logger(env)


    """ run game """

    logger("[@simulations.py]")

    # ===| setup |===
    last_position = np.zeros(2)

    # [position, velocity, collision, reward, done, terminated]
    observation = [[0., 0.], 0., 0., False, False]
    prev_position = env.position

    events = []

    # ===| main loop |===
    # running = True
    # while running:
    for _ in tqdm(range(env.duration), desc="Game", leave=False,
                  disable=not verbose_min):


        # -check: teleport
        if env.t % t_teleport == 0 and env.reward_obj.is_silent: # <=========================
            env._reset_agent_position(brain, True)


        # velocity
        v = [(env.position[0] - prev_position[0]),
             (-env.position[1] + prev_position[1])]

        # brain step
        try:
            velocity = brain(v,
                             observation[1],
                             observation[2],
                             env.reward_availability)
        except IndexError:
            logger.debug(f"IndexError: {len(observation)}")
            raise IndexError
        # velocity = np.around(velocity, 2)

        # store past position
        prev_position = env.position
        events += [[observation[1], observation[2]]]

        # env step
        observation = env(velocity=np.array([velocity[0], -velocity[1]]),
                          brain=brain)

        # -check: reset agent's brain
        if observation[3]:
            if verbose and verbose_min:
                logger.info(">> Game reset <<")
            break

    logger(f"rw_count={env.rw_count}")

    return events



def main_rep(reps: int, save: bool=True):

    events = []

    logger(f"starting {reps} repetitions")

    rbar = tqdm(range(reps))
    for i in rbar:
        rbar.set_description(f"rep={i} [{reps}]")
        run_events = run_game_sil()
        events += [run_events]

    logger("[done]")

    if save:

        name = utils.DATA_PATH + "/rwd_shift_data"
        num = len([f for f in os.listdir(utils.DATA_PATH) if "rwd_shift_data" in f])
        name = f"{name}_{num}.json"

        with open(name, 'w') as f:
            json.dump(events, f)

        logger(f"[rwd_shift_data data saved at {name}]")



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    # -- run

    main_rep(reps=args.reps,
             save=args.save)

