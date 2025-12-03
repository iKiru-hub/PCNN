import numpy as np
import matplotlib.pyplot as plt
import argparse, json
from tqdm import tqdm
import pygame
# import gymnasium as gym

import utils
import game.envs as games
import game.objects as objects
import game.constants as constants

try:
    # import libs.pclib as pclib
    # import core.build.pclib as pclib
    import libs.pclib2 as pclib2
except ImportError:
    import warnings
    warnings.warn("pclib [c++] not found, using python version")
    import libs.pclib1 as pclib

logger = utils.setup_logger(__name__, level=-2, is_debugging=False)


""" SETTINGS """

GAME_SCALE = games.SCREEN_WIDTH


reward_settings = {
    "rw_fetching": "deterministic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.05 * GAME_SCALE,
    "rw_sigma": 0.8,# * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 2,
    "silent_duration": 8_000,
    "fetching_duration": 2,
    "transparent": False,
    "beta": 40.,
    "alpha": 0.06,# * GAME_SCALE,
    "move_threshold": 4,# * GAME_SCALE,
    "rw_position_idx": 0,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "rendering": True,
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "max_duration": 50_000,
    "room_thickness": 30,
    "t_teleport": 3_000,
    "limit_position_len": -1,
    "start_position_idx": 1,
    "seed": None,
    "pause": -1,
    "verbose": True
}

global_parameters = {
    "local_scale": 0.015,
    "N": 32**2,
    "use_sprites": bool(1),
    "speed": 1.0,
    "min_weight_value": 0.5
}

parameters3 = {
        "gain": 20.0,
        "offset": 1.2,
        "threshold": 0.2,
        "rep_threshold": 0.99,
        "rec_threshold": 63,
        "tau_trace": 20,
        "remap_tag_frequency": 1,
        "min_rep_threshold": 0.98,

        "lr_da": 0.9,
        "lr_pred": 0.01,
        "threshold_da": 0.03,
        "tau_v_da": 2.0,

        "lr_bnd": 0.9,
        "threshold_bnd": 0.01,
        "tau_v_bnd": 3.0,

        "tau_ssry": 437.0,
        "threshold_ssry": 1.986,
        "threshold_circuit": 0.9,

        "rwd_weight": 1.,
        "rwd_sigma": 40.,
        "col_weight": 0.,
        "col_sigma": 2.6,
        "rwd_field_mod": 1.0,
        "col_field_mod": 2.6,

        "action_delay": 120.0,
        "edge_route_interval": 1,
        "forced_duration": 19,
        "min_weight_value": 0.2}

parameters4 = {
          "gain": 80.0,
          "offset": 1.0,
          "threshold": 0.1,
          "rep_threshold": 0.999,
          "rec_threshold": 70,
          "tau_trace": 20,
          "remap_tag_frequency": 1,
          "num_neighbors": 16,
          "min_rep_threshold": 0.99,

          "lr_da": 0.9,
          "lr_pred": 0.12,
          "threshold_da": 0.05,
          "tau_v_da": 1.0,
          "lr_bnd": 0.9,
          "threshold_bnd": 0.05,
          "tau_v_bnd": 2.0,

          "tau_ssry": 437.0,
          "threshold_ssry": 1.986,
          "threshold_circuit": 0.9,

          "rwd_weight": -2.68,
          "rwd_sigma": 67.7,
          "col_weight": 3.14,
          "col_sigma": 27.8,
          "rwd_field_mod": 3.0,
          "col_field_mod": 0.9,

          "action_delay": 100.0,
          "edge_route_interval": 100,
          "forced_duration": 19,
          "min_weight_value": 0.01
}

parameters = {
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
      "threshold_bnd": 0.1,
      "tau_v_bnd": 1.0,
      "tau_ssry": 437.0,
      "threshold_ssry": 1.986,
      "threshold_circuit": 0.9,
      "rwd_weight": -2.11,
      "rwd_sigma": 96.8,
      "rwd_threshold": 0.49,
      "col_weight": -0.53,
      "col_sigma": 16.1,
      "col_threshold": 0.37,
      "rwd_field_mod": 4.6,
      "col_field_mod": 4.4,
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


""" UTILITIES """



class Renderer:

    def __init__(self, brain: object, boundsx: tuple=(-430, 100),
                 boundsy: tuple=(-200, 330),
                 render_type: str="space"):

        self.brain = brain
        self.names = ["DA", "BND"]
        self.min_x = boundsx[0]
        self.max_x = boundsx[1]
        self.min_y = boundsy[0]
        self.boundsx = boundsx
        self.boundsy = boundsy
        # print(f"boundsx={self.boundsx}")
        # print(f"boundsy={self.boundsy}")
        self.render_type = render_type
        if render_type == "space0":
            self.fig, self.axs = plt.subplots(1, 3, figsize=(6, 6))
            self.fig.set_tight_layout(True)
        elif render_type == "space3":
            self.fig, self.axs = plt.subplots(figsize=(6, 6))
        else:
            self.fig, self.axs = plt.subplots(1, 1, figsize=(4, 4))
            self.fig.set_tight_layout(True)

    def __call__(self):

        if self.render_type == "space":
            self.render()
        elif self.render_type == "space0":
            self.render2()
        elif self.render_type == "space3":
            self.render3()
        elif self.render_type == 'none':
            pass
        else:
            raise ValueError("render_type not recognized")

    def render(self):

        self.axs.clear()
        gg = self.brain.get_gain()

        # -- pc
        self.axs.scatter(*np.array(self.brain.get_space_centers()).T,
                         color='grey',
                         alpha=0.1,
                         s=0.9*np.mean(gg))

        # -- BND
        bndw = self.brain.get_bnd_weights()
        bndidx = np.where(bndw>0.05)[0]
        self.axs.scatter(*np.array(self.brain.get_space_centers())[bndidx, :].T,
                         c=bndw[bndidx],
                         cmap="Blues", alpha=1.,
                         # s=np.where(bndw > 0.01, 80, 0),
                         s=0.9*gg.mean()*gg.mean()/gg[bndidx],
                         vmin=0., vmax=0.2)

        # -- DA
        daw = self.brain.get_da_weights()
        daidx = np.where(daw>0.05)[0]
        self.axs.scatter(*np.array(self.brain.get_space_centers())[daidx, :].T,
                         c=daw[daidx],
                         # s=np.where(daw > 0.01, 80, 0),
                         s=0.9*gg.mean()*gg.mean()/gg[daidx],
                         cmap="Greens", alpha=1.,
                         vmin=0., vmax=0.3)

        # -- plan
        plan_center = np.array(self.brain.get_space_centers())[self.brain.get_plan_idxs()]
        self.axs.plot(*plan_center.T, color="red", alpha=1., lw=2.)

        self.axs.set_title(f"N={len(self.brain)}")

        self.axs.set_xlim(self.boundsx)
        self.axs.set_ylim(self.boundsy)
        self.axs.set_xticks(())
        self.axs.set_yticks(())
        self.axs.set_aspect('equal', adjustable='box')
        for spine in self.axs.spines.values():
            spine.set_linewidth(5)
        plt.pause(0.00001)


    def render2(self, show: bool=False):

        # -- space plots --
        self.axs[0, 0].clear()
        self.axs[0, 1].clear()

        # -- goal-behaviour
        if self.brain.get_directive() == "trg" or \
           self.brain.get_directive() == "trg rw" or \
           self.brain.get_directive() == "trg ob":

            # -- fine space
            self.axs[0, 0].scatter(*np.array(self.brain.get_space_centers()).T,
                                   color="blue", s=10, alpha=0.1,
                                   label="fine-grained space")
            # goal
            self.axs[0, 1].scatter(*np.array(self.brain.get_space_centers()).T,
                                   c=self.brain.get_trg_representation(),
                                   s=100*self.brain.get_trg_representation(),
                                   marker="x",
                                   label="goal",
                                   cmap="Greens", alpha=0.7)


            # plan
            plan_center = np.array(self.brain.get_space_centers())[self.brain.get_plan_idxs()]
            self.axs[0, 0].plot(*plan_center.T,
                                color="grey", alpha=0.7, lw=4.,
                                label="plan")

            # -- coarse space
            self.axs[0, 1].scatter(*np.array(self.brain.get_space_coarse_centers()).T,
                                   color="brown", s=30, alpha=0.2,
                                   label="coarse-grained space")

        # -- explorative behaviour
        else:
            # -- fine space
            self.axs[0, 0].scatter(*np.array(self.brain.get_space_enters()).T,
                                   color="blue", s=10, alpha=0.7)

            for edge in self.brain.make_space_edges():
                self.axs[0, 0].plot((edge[0][0], edge[1][0]),
                                    (edge[0][1], edge[1][1]),
                                 alpha=0.3, lw=0.5, color="black")

        # -- fine space
        self.axs[0, 0].set_title(f"#PCs={self.brain.get_space_count()}")
        self.axs[0, 0].scatter(*np.array(self.brain.get_space_position()).T,
                               color="red", s=50, marker="v", alpha=0.8)
        self.axs[0, 0].set_xlim(self.boundsx)
        self.axs[0, 0].set_ylim(self.boundsy)
        self.axs[0, 0].set_aspect('equal', adjustable='box')
        self.axs[0, 0].grid(alpha=0.1)
        self.axs[0, 0].set_xticks(())
        self.axs[0, 0].set_yticks(())
        # self.axs[0, 0].legend()

        # -- BND
        self.axs[1, 0].clear()
        self.axs[1, 0].scatter(*np.array(self.brain.get_space_centers()).T,
                               c=self.brain.get_bnd_weights(),
                               cmap="Blues", alpha=0.8,
                               s=20, vmin=0., vmax=0.2,
                               label="BND")

        daw = self.brain.get_da_weights()
        self.axs[1, 0].scatter(*np.array(self.brain.get_space_centers()).T,
                               c=daw,
                               s=np.where(daw > 0.01, 30, 0),
                               cmap="Greens", alpha=0.8,
                               label="DA",
                               vmin=0., vmax=0.2)
        plan_center_coarse = np.array(self.brain.get_space_centers())[self.brain.get_plan_idxs()]
        self.axs[1, 0].plot(*plan_center_coarse.T,
                            color="grey", alpha=0.7, lw=4.,
                            label="plan")
        self.axs[1, 0].scatter(*np.array(self.brain.get_space_centers()).T,
                               c=self.brain.get_trg_representation(),
                               s=100*self.brain.get_trg_representation(),
                               marker="x",
                               label="goal",
                               cmap="Greens", alpha=0.7)

        self.axs[1, 0].set_xlim(self.boundsx)
        self.axs[1, 0].set_ylim(self.boundsy)
        self.axs[1, 0].set_xticks(())
        self.axs[1, 0].set_yticks(())
        self.axs[1, 0].set_title(f"BND")
        self.axs[1, 0].set_aspect('equal', adjustable='box')
        self.axs[1, 0].legend()

        # -- DA
        self.axs[1, 1].clear()
        daw = self.brain.get_da_weights()
        self.axs[1, 1].scatter(*np.array(self.brain.get_space_centers()).T,
                               c=daw,
                               s=np.where(daw > 0.01, 30, 1),
                               cmap="Greens", alpha=0.8,
                               vmin=0., vmax=0.2)
        self.axs[1, 1].set_xlim(self.boundsx)
        self.axs[1, 1].set_ylim(self.boundsy)
        self.axs[1, 1].set_xticks(())
        self.axs[1, 1].set_yticks(())
        self.axs[1, 1].set_title(f"DA")
        self.axs[1, 1].set_aspect('equal', adjustable='box')

        plt.pause(0.00001)

    def render3(self):

        # -- BND
        self.axs.clear()
        self.axs.scatter(*np.array(self.brain.get_space_centers()).T,
                               c=self.brain.get_bnd_weights(),
                               cmap="Blues", alpha=0.8,
                               s=40, vmin=0., vmax=0.5,
                               label="BND")

        daw = self.brain.get_da_weights()
        self.axs.scatter(*np.array(self.brain.get_space_centers()).T,
                               c=daw,
                               s=np.where(daw > 0.01, 40, 0),
                               cmap="Greens", alpha=0.8,
                               label="DA",
                               vmin=0., vmax=0.4)
        self.axs.scatter(*np.array(self.brain.get_space_centers()).T,
                               color="brown", s=20, alpha=0.2)
        plan_centers = np.array(self.brain.get_space_centers())[self.brain.get_plan_idxs()]
        self.axs.plot(*plan_centers.T, color="grey", alpha=0.7, lw=4.,
                            label="plan")
        self.axs.scatter(*np.array(self.brain.get_space_centers()).T,
                               c=self.brain.get_trg_representation(),
                               s=200*self.brain.get_trg_representation(),
                               marker="x",
                               label="goal",
                               cmap="Greens", alpha=0.7)

        self.axs.set_xlim(self.boundsx)
        self.axs.set_ylim(self.boundsy)
        self.axs.set_xticks(())
        self.axs.set_yticks(())
        self.axs.set_aspect('equal', adjustable='box')
        self.axs.set_title(f"N={len(self.brain.space)}")
        plt.pause(0.00001)


def setup_env(global_parameters: dict,
              reward_settings: dict,
              game_settings: dict,
              room_name: str,
              pause: int=-1, verbose: bool=True,
              record_flag: bool=False,
              limit_position_len: int=-1,
              agent_position_list: list=None,
              preferred_positions: list=None,
              verbose_min: bool=True):

    """ make game environment """

    if verbose and verbose_min:
        logger(f"room_name={room_name}")

    room = games.make_room(name=room_name,
                           thickness=game_settings["room_thickness"],
                           bounds=[0, 1, 0, 1])
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    # ===| objects |===

    possible_positions = room.get_room_positions()

    # reward position
    if reward_settings['rw_position_idx'] > -1:
        rw_position_idx = reward_settings['rw_position_idx']
    else:
        rw_position_idx = np.random.randint(0, len(possible_positions))

    rw_position = possible_positions[rw_position_idx]

    # agent position
    if 'agent_position_idx_list' in tuple(game_settings.keys()):
        agent_possible_positions = possible_positions[game_settings['agent_position_idx_list']]
    else:
        agent_possible_positions = possible_positions.copy()

    agent_position = room.sample_next_position()

    #
    rw_tau = reward_settings["tau"] if "tau" in reward_settings else 100

    reward_obj = objects.RewardObj(
                position=rw_position,
                # possible_positions=constants.POSSIBLE_POSITIONS.copy(),
                possible_positions=possible_positions,
                radius=reward_settings["rw_radius"],
                sigma=reward_settings["rw_sigma"],
                fetching=reward_settings["rw_fetching"],
                value=reward_settings["rw_value"],
                bounds=room_bounds,
                delay=reward_settings["delay"],
                use_sprites=global_parameters["use_sprites"],
                silent_duration=reward_settings["silent_duration"],
                tau=rw_tau,
                preferred_positions=preferred_positions,
                move_threshold=reward_settings["move_threshold"],
                transparent=reward_settings["transparent"])

    body = objects.AgentBody(
                position=agent_position,
                speed=global_parameters["speed"],
                possible_positions=possible_positions,
                use_sprites=global_parameters["use_sprites"],
                bounds=game_settings["agent_bounds"],
                room=room,
                limit_position_len=game_settings["limit_position_len"],
                color=(10, 10, 10))

    # --- env
    env = games.Environment(room=room,
                            agent=body,
                            reward_obj=reward_obj,
                            rw_event=game_settings["rw_event"],
                            verbose=False,
                            agent_position_list=agent_possible_positions,
                            duration=game_settings["max_duration"],
                            visualize=game_settings["rendering"])

    return env, reward_obj


def run_model(parameters: dict,
              global_parameters: dict,
              reward_settings: dict,
              game_settings: dict,
              room_name: str,
              pause: int=-1, verbose: bool=True,
              record_flag: bool=False,
              limit_position_len: int=-1,
              preferred_positions: list=None,
              verbose_min: bool=True) -> int:

    """
    meant to be run standalone
    """


    """ make model """

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


    """ setup """

    env, reward_obj = setup_env(global_parameters=global_parameters,
                    reward_settings=reward_settings,
                    game_settings=game_settings,
                    room_name=room_name,
                    pause=pause, verbose=verbose,
                    record_flag=record_flag,
                    limit_position_len=limit_position_len,
                    preferred_positions=preferred_positions,
                    verbose_min=verbose_min)

    if env.reward_obj.preferred_positions is not None:
        idx = np.random.choice(env.reward_obj.preferred_positions)
        env.reward_obj.set_position(
                env.room.sample_random_position(idx))

    if verbose_min:
        logger("[@simulations.py]")
    record = run_game(env=env,
             model=brain,
             renderer=None,
             plot_interval=game_settings["plot_interval"],
             t_teleport=game_settings["t_teleport"],
             pause=-1,
             record_flag=record_flag,
             verbose=verbose,
             verbose_min=verbose_min)

    if verbose_min:
        logger(f"rw_count={env.rw_count}")
        logger(f"rw_collisions={env.nb_collisions_from_rw}")

    # if record_flag:
    #     record["rw_count"] = env.rw_count
    #     return record


    info = {
        "env": env,
        "reward_obj": reward_obj,
        "brain": brain,
        "record": record,
        "collisions": env.nb_collisions,
        "collisions_from_rw": env.nb_collisions_from_rw
    }


    return env.rw_count, info


def run_game(env: games.Environment,
             model: object,
             renderer: object,
             plot_interval: int,
             t_teleport: int=100,
             pause: int=-1,
             record_flag: bool=False,
             verbose: bool=True,
             verbose_min: bool=True):

    # ===| setup |===
    clock = pygame.time.Clock()
    last_position = np.zeros(2)

    # [position, velocity, collision, reward, done, terminated]
    observation = [[0., 0.], 0., 0., False, False]
    prev_position = env.position

    record = {"activity": [],
              "trajectory": []}

    # ===| main loop |===
    for _ in tqdm(range(env.duration), desc="Game", leave=False,
                  disable=not verbose_min):

        # Event handling
        if env.visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # -check: teleport
        # if env.t % t_teleport == 0 and env.reward_obj.is_silent:
        if env.t % t_teleport == 0:# and (env.reward_obj.is_silent or env.reward_obj.count == 0):
            env._reset_agent_position(model, True)

        # velocity
        v = [(env.position[0] - prev_position[0]),
             (-env.position[1] + prev_position[1])]

        # model step
        try:
            velocity = model(v,
                             observation[1],
                             observation[2],
                             env.reward_availability)
        except IndexError:
            logger.debug(f"IndexError: {len(observation)}")
            raise IndexError

        # store past position
        prev_position = env.position

        # env step
        observation = env(velocity=np.array([velocity[0], -velocity[1]]),
                          brain=model)

        # -check: reset agent's brain
        if observation[3]:
            if verbose and verbose_min:
                logger.info(">> Game reset <<")
            break

        # -check: render
        if env.visualize:
            if env.t % plot_interval == 0:
                env.render()
        if renderer:
            if env.t % plot_interval == 0:
                renderer()

        # -check: record
        if record_flag:
            record["activity"] += [model.get_representation()]
            record["trajectory"] += [env.position]

        # -check: exit
        if observation[4]:
            if verbose and verbose_min:
                logger.debug(">> Game terminated <<")
            break

        # reward navigation time
        if model.is_reward_represented:
            env.set_rw_navigation_time()

        # pause
        if pause > 0:
            pygame.time.wait(pause)

    pygame.quit()

    return record



""" MAIN """


def main_game(global_parameters: dict=global_parameters,
              reward_settings: dict=reward_settings,
              game_settings: dict=game_settings,
              room_name: str="Square.v0", load: bool=False,
              load_idx: int=-1,
              render_type: str="space", duration: int=-1,
              render: bool=False):

    """
    meant to be run standalone
    """


    if load:
        load_idx = load_idx if load_idx > -1 else -1
        parameters = utils.load_parameters(idx=load_idx)
    else:
        parameters = fixed_params

    """ make model """

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

    room = games.make_room(name=room_name,
                           thickness=game_settings["room_thickness"],
                           bounds=[0, 1, 0, 1])
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    # ===| objects |===

    possible_positions = room.get_room_positions()

    rw_position_idx = np.random.randint(0, len(possible_positions))
    rw_position = possible_positions [rw_position_idx]
    agent_possible_positions = possible_positions.copy()
    agent_position = possible_positions[game_settings['start_position_idx']]

    rw_tau = reward_settings["tau"] if "tau" in reward_settings else 400
    if "move_threlshold" in reward_settings:
        rw_move_threshold = reward_settings["move_threshold"]
    else:
        rw_move_threshold = 2

    reward_obj = objects.RewardObj(
                # position=reward_settings["rw_position"],
                position=rw_position,
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
                transparent=reward_settings["transparent"])

    body = objects.AgentBody(
                # position=agent_settings["init_position"],
                position=agent_position,
                speed=global_parameters["speed"],
                possible_positions=agent_possible_positions,
                bounds=game_settings["agent_bounds"],
                use_sprites=global_parameters["use_sprites"],
                limit_position_len=game_settings["limit_position_len"],
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
                            visualize=render)
                            # visualize=game_settings["rendering"])
    logger(env)


    """ run game """

    if game_settings["rendering"]:
        renderer = Renderer(brain=brain, render_type=render_type)
    else:
        renderer = None

    logger("[@simulations.py]")
    run_game(env=env,
             model=brain,
             renderer=renderer,
             t_teleport=game_settings["t_teleport"],
             plot_interval=game_settings["plot_interval"],
             pause=-1)

    logger(f"rw_count={env.rw_count}")

    plt.show()


def main_cartpole(load: bool=False,
                  rendering: bool=False,
                  duration: int=-1):

    """
    meant to be run standalone
    """

    global_parameters = {
        "local_scale_fine": 0.02,
        "local_scale_coarse": 0.00006,
        "N": 10**2,
        "Nc": 8**2,
        "speed": 0.1,
    }


    if load:
        parameters = utils.load_parameters()
        # logger.debug(parameters)
        # parameters, session_config = utils.load_session()

        # global_parameters = session_config["global_parameters"]
        # reward_settings = session_config["reward_settings"]
        # agent_settings = session_config["agent_settings"]
        # game_settings = session_config["game_settings"]
        # game_settings["rendering"] = True
        # game_settings["plot_interval"] = 100
        # game_settings["rw_event"] = "move both"


    else:
        parameters = {

            "gain_fine": 15.,
            "offset_fine": 1.0,
            "threshold_fine": 0.3,
            "rep_threshold_fine": 0.7,
            "rec_threshold_fine": 30.,
            "tau_trace_fine": 20.0,

            "remap_tag_frequency": 1,
            "min_rep_threshold": 25,

            "gain_coarse": 15.,
            "offset_coarse": 1.1,
            "threshold_coarse": 0.3,
            "rep_threshold_coarse": 0.4,
            "rec_threshold_coarse": 70.,
            "tau_trace_coarse": 100.0,

            "lr_da": 0.99,
            "lr_pred": 0.2,
            "threshold_da": 0.04,
            "tau_v_da": 2.0,

            "lr_bnd": 0.6,
            "threshold_bnd": 0.3,
            "tau_v_bnd": 3.0,

            "tau_ssry": 100.,
            "threshold_ssry": 0.995,

            "threshold_circuit": 0.02,
            "remapping_flag": -7,

            "rwd_weight": 1.5,
            "rwd_sigma": 40.0,
            "col_weight": 0.5,
            "col_sigma": 25.0,

            "action_delay": 100.,
            "edge_route_interval": 3,

            "forced_duration": 100,
            "fine_tuning_min_duration": 10,
        }

    """ make model """

    remap_tag_frequency = parameters["remap_tag_frequency"] if "remap_tag_frequency" in parameters else 200
    remapping_flag = parameters["remapping_flag"] if "remapping_flag" in parameters else 0
    lr_pred = parameters["lr_pred"] if "lr_pred" in parameters else 0.2

    brain = pclib.Brain(
                local_scale_fine=global_parameters["local_scale_fine"],
                local_scale_coarse=global_parameters["local_scale_coarse"],
                N=global_parameters["N"],
                Nc=global_parameters["Nc"],
                min_rep_threshold=parameters["min_rep_threshold"],
                rec_threshold_fine=parameters["rec_threshold_fine"],
                rec_threshold_coarse=parameters["rec_threshold_coarse"],
                tau_trace_fine=parameters["tau_trace_fine"],
                speed=global_parameters["speed"],
                gain_fine=parameters["gain_fine"],
                offset_fine=parameters["offset_fine"],
                threshold_fine=parameters["threshold_fine"],
                rep_threshold_fine=parameters["rep_threshold_fine"],
                remap_tag_frequency=remap_tag_frequency,
                tau_trace_coarse=parameters["tau_trace_coarse"],
                gain_coarse=parameters["gain_coarse"],
                offset_coarse=parameters["offset_coarse"],
                threshold_coarse=parameters["threshold_coarse"],
                rep_threshold_coarse=parameters["rep_threshold_coarse"],
                lr_da=parameters["lr_da"],
                lr_pred=lr_pred,
                threshold_da=parameters["threshold_da"],
                tau_v_da=parameters["tau_v_da"],
                lr_bnd=parameters["lr_bnd"],
                threshold_bnd=parameters["threshold_bnd"],
                tau_v_bnd=parameters["tau_v_bnd"],
                tau_ssry=parameters["tau_ssry"],
                threshold_ssry=parameters["threshold_ssry"],
                threshold_circuit=parameters["threshold_circuit"],
                remapping_flag=remapping_flag,
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

    duration = game_settings["max_duration"] if duration < 0 else duration

    t_goal = 100
    t_rendering = 10 if rendering else 10000000000
    record_flag = False

    """ run game """

    if rendering:
        renderer = Renderer(brain=brain, boundsx=(-400.0, 400.0),
                            boundsy=(-400.0, 400.0))
    else:
        renderer = None

    logger("[@simulations.py]")
    record = run_cartpole(brain=brain,
                 renderer=renderer,
                 duration=duration,
                 t_goal=t_goal,
                 t_rendering=t_rendering,
                 record_flag=record_flag)

    logger(f"total reward={np.sum(record['scores'])}")

    plt.show()


def main_game_v2(room_name: str="Flat.1011", load: bool=False):

    if load:
        parameters = utils.load_parameters()
        logger.debug(parameters)

    run_model(parameters, global_parameters,
              agent_settings, reward_settings,
              game_settings, room_name, pause=game_settings["pause"],
              verbose=game_settings["verbose"])



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--seed", type=int, default=-1,
                        help="random seed: -1 for random seed.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--room", type=str, default="Square.v0",
                        help=f'room name: {constants.ROOMS} or `random`')
    parser.add_argument("--main", type=str, default="game",
                        help="[game, rand, simple]")
    parser.add_argument("--interval", type=int, default=20,
                        help="plotting interval")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_type", type=str, default="space",
                        help="options: 'space', 'space0', 'space3', 'none'")

    args = parser.parse_args()

    # --- seed
    if args.seed > 0:
        logger.debug(f"seed: {args.seed}")
        np.random.seed(args.seed)

    logger = utils.setup_logger(name="RUN",
                                level=2,
                                is_debugging=True,
                                is_warning=False)


    # --- settings
    game_settings["plot_interval"] = args.interval

    if args.room == "random":
        args.room = games.get_random_room()
        logger(f"random room: {args.room}")

    # --- run
    try:
        if args.main == "game":
            main_game(room_name=args.room, load=args.load,
                      duration=args.duration,
                      render_type=args.render_type,
                      load_idx=args.idx,
                      render=args.render)
        if args.main == "cartpole":
            main_cartpole(load=args.load,
                          duration=args.duration,
                          rendering=args.render)
        else:
            logger.error("main not found ...")
    except KeyboardInterrupt:
        logger.debug("Keyboard interrupt")
        # plt.show()
        plt.close()
        pygame.quit()

