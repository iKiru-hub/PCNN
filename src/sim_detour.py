import numpy as np
import matplotlib.pyplot as plt
import argparse, json
from tqdm import tqdm
import pygame
import gymnasium as gym

import utils
import game.envs as games
import game.objects as objects
import game.constants as constants

try:
    # import libs.pclib as pclib
    import core.build.pclib as pclib
except ImportError:
    import warnings
    warnings.warn("pclib [c++] not found, using python version")
    import libs.pclib1 as pclib

logger = utils.setup_logger(__name__, level=-1)


""" SETTINGS """

GAME_SCALE = games.SCREEN_WIDTH


reward_settings = {
    "rw_fetching": "probabilistic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.05 * GAME_SCALE,
    "rw_sigma": 0.8,# * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 200,
    "silent_duration": 12_000,
    "fetching_duration": 10,
    "transparent": False,
    "beta": 40.,
    "alpha": 0.06,# * GAME_SCALE,
    "tau": 300,# * GAME_SCALE,
    "move_threshold": 4000,# * GAME_SCALE,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "rendering": True,
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "max_duration": 25_000,
    "room_thickness": 30,
    "t_teleport": 1000,
    "limit_position_len": 1,
    "seed": None,
    "pause": -1,
    "verbose": True
}

global_parameters = {
    "local_scale_fine": 0.02,
    "local_scale_coarse": 0.006,
    "N": 42**2,
    "Nc": 35**2,
    "use_sprites": False,
    "speed": 0.7,
    "min_weight_value": 0.5
}

parameters = {

    "gain_fine": 15.,
    "offset_fine": 1.0,
    "threshold_fine": 0.3,
    "rep_threshold_fine": 0.9,
    "rec_threshold_fine": 45.,
    "tau_trace_fine": 30.0,

    "remap_tag_frequency": 1,
    "num_neighbors": 8,
    "min_rep_threshold": 34,

    "gain_coarse": 21.,
    "offset_coarse": 1.1,
    "threshold_coarse": 0.3,
    "rep_threshold_coarse": 0.6,
    "rec_threshold_coarse": 50.,
    "tau_trace_coarse": 50.0,

    "lr_da": 0.5,
    "lr_pred": 0.2,
    "threshold_da": 0.03,
    "tau_v_da": 20.0,

    "lr_bnd": 0.6,
    "threshold_bnd": 0.3,
    "tau_v_bnd": 3.0,

    "tau_ssry": 100.,
    "threshold_ssry": 0.995,

    "threshold_circuit": 0.9,
    "remapping_flag": 7,

    "rwd_weight": 15.0,
    "rwd_sigma": 80.0,
    "col_weight": 0.0,
    "col_sigma": 35.0,

    "rwd_field_mod_fine": 2.5,
    "rwd_field_mod_coarse": 4.5,
    "col_field_mod_fine": 1.5,
    "col_field_mod_coarse": 0.3,

    "action_delay": 80.,
    "edge_route_interval": 50000, # <<<<<<<<<<<<<<<<<<<

    "forced_duration": 100,
    "fine_tuning_min_duration": 10,
}


fixed_params = parameters.copy()


possible_positions = np.array([
    [0.25, 0.75], [0.75, 0.75],
    [0.25, 0.25], [0.75, 0.25]]) * GAME_SCALE


""" UTILITIES """



class Renderer:

    def __init__(self, brain: object, boundsx: tuple=(-110, 450),
                 boundsy: tuple=(-450, 110),
                 render_type: str="space"):

        self.brain = brain
        self.names = ["DA", "BND"]
        self.min_x = boundsx[0]
        self.max_x = boundsx[1]
        self.min_y = boundsy[0]
        self.boundsx = boundsx
        self.boundsy = boundsy
        print(f"boundsx={self.boundsx}")
        print(f"boundsy={self.boundsy}")
        self.render_type = render_type
        if render_type == "space0":
            self.fig, self.axs = plt.subplots(2, 2, figsize=(6, 6))
        else:
            self.fig, self.axs = plt.subplots(1, 1, figsize=(4, 4))
        self.fig.set_tight_layout(True)

    def __call__(self):

        if self.render_type == "space":
            self.render()
        elif self.render_type == "space0":
            self.render2()
        else:
            raise ValueError("render_type not recognized")

    def render(self):

        # -- BND
        self.axs.clear()
        self.axs.scatter(*np.array(self.brain.get_space_fine_centers()).T,
                               c=self.brain.get_bnd_weights(),
                               cmap="Blues", alpha=0.8,
                               s=40, vmin=0., vmax=0.5,
                               label="BND")

        daw = self.brain.get_da_weights()
        self.axs.scatter(*np.array(self.brain.get_space_fine_centers()).T,
                               c=daw,
                               s=np.where(daw > 0.01, 40, 0),
                               cmap="Greens", alpha=0.8,
                               label="DA",
                               vmin=0., vmax=0.4)
        self.axs.scatter(*np.array(self.brain.get_space_coarse_centers()).T,
                               color="brown", s=20, alpha=0.2)
        plan_center_coarse = np.array(self.brain.get_space_coarse_centers())[self.brain.get_plan_idxs_coarse()]
        self.axs.plot(*plan_center_coarse.T,
                            color="grey", alpha=0.7, lw=4.,
                            label="plan")
        self.axs.scatter(*np.array(self.brain.get_space_fine_centers()).T,
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
        # self.axs.legend(loc="upper right")
        plt.pause(0.00001)

        # if self.brain.get_directive() == "trg" or \
        #     self.brain.get_directive() == "trg rw" or len(plan_center_coarse) > 0:
        #     input(">>")

    def render2(self, show: bool=False):

        # -- space plots --
        self.axs[0, 0].clear()
        self.axs[0, 1].clear()

        # maxc = max((self.max_x, self.brain.get_space_fine_centers().max()))
        # self.boundsx = (self.min_x, maxc)
        # self.boundsy = (self.min_y, maxc)

        # -- goal-behaviour
        if self.brain.get_directive() == "trg" or \
           self.brain.get_directive() == "trg rw" or \
           self.brain.get_directive() == "trg ob":

            # -- fine space
            self.axs[0, 0].scatter(*np.array(self.brain.get_space_fine_centers()).T,
                                   color="blue", s=10, alpha=0.1,
                                   label="fine-grained space")
            # goal
            self.axs[0, 1].scatter(*np.array(self.brain.get_space_fine_centers()).T,
                                   c=self.brain.get_trg_representation(),
                                   s=100*self.brain.get_trg_representation(),
                                   marker="x",
                                   label="goal",
                                   cmap="Greens", alpha=0.7)
            # current position
            # self.axs[0, 0].scatter(*np.array(self.brain.get_space_fine_centers()).T,
            #                        c = self.brain.get_representation_fine(),
            #                        cmap="Greys",
            #                        s=20, alpha=0.3)

            # plan
            # plan = np.zeros(len(self.brain))
            # plan[self.brain.get_plan_idxs_fine()] = 1.
            plan_center_fine = np.array(self.brain.get_space_fine_centers())[self.brain.get_plan_idxs_fine()]
            self.axs[0, 0].plot(*plan_center_fine.T,
                                color="grey", alpha=0.7, lw=4.,
                                label="plan")
            # self.axs[0, 0].scatter(*np.array(self.brain.get_space_fine_centers()).T,
            #             c=plan, s=20*plan, cmap="Greens", alpha=0.7)

            # title
            # loc_ = np.array(self.brain.get_space_fine_centers()
            #                 )[self.brain.get_trg_idx()]
            # self.axs[0, 0].set_title(f"Space | trg_idx={self.brain.get_trg_idx()} " + \
            #     f" ({self.brain.get_trg_representation().max():.3f}, " + \
            #     f"{self.brain.get_trg_representation().argmax()}) " + \
            #     f" loc={loc_}")

            # -- coarse space
            self.axs[0, 1].scatter(*np.array(self.brain.get_space_coarse_centers()).T,
                                   color="brown", s=30, alpha=0.2,
                                   label="coarse-grained space")
            # current position
            # self.axs[0, 1].scatter(*np.array(self.space_coarse.get_centers()).T,
            #                        c = self.brain.get_representation(),
            #                        cmap="Greys",
            #                        s=30, alpha=0.3)
            # goal
            # self.axs[0, 1].scatter(*np.array(self.space_coarse.get_centers()).T,
            #                        c=self.brain.get_trg_representation(),
            #                        s=100*self.brain.get_trg_representation(),
            #                        cmap="Greens", alpha=0.7)

            # plan
            # plan = np.zeros(self.brain.get_space_coarse_size())
            # plan[self.brain.get_plan_idxs_coarse()] = 1.
            # self.axs[0, 1].scatter(*np.array(self.brain.get_space_coarse_centers()).T,
            #             c=plan, s=30*plan, cmap="Greens", alpha=0.7)
            plan_center_coarse = np.array(self.brain.get_space_coarse_centers())[self.brain.get_plan_idxs_coarse()]
            self.axs[0, 1].plot(*plan_center_coarse.T,
                                color="grey", alpha=0.7, lw=4.,
                                label="plan")

            # title
            # self.axs[0, 1].set_title(f"Coarse space") 

        # -- explorative behaviour
        else:
            # -- fine space
            self.axs[0, 0].scatter(*np.array(self.brain.get_space_fine_centers()).T,
                                   color="blue", s=10, alpha=0.7)
            # for edge in self.brain.make_space_fine_edges():
            #     self.axs[0, 0].plot((edge[0][0], edge[1][0]),
            #                         (edge[0][1], edge[1][1]),
            #                      alpha=0.1, lw=0.5, color="black")

            # -- coarse space
            for edge in self.brain.make_space_coarse_edges():
                self.axs[0, 1].plot((edge[0][0], edge[1][0]),
                                    (edge[0][1], edge[1][1]),
                                 alpha=0.3, lw=0.5, color="black")

            self.axs[0, 1].scatter(*np.array(self.brain.get_space_coarse_centers()).T,
                                   color="brown", s=20, alpha=0.7)

        # current representation
        # self.axs[0, 0].scatter(*np.array(self.brain.get_space_fine_centers()).T,
        #                        c = self.brain.get_representation_fine(), cmap="Greys",
        #                        s=40*self.brain.get_representation_fine(),
        #                        alpha=0.9)
        # self.axs[0, 1].scatter(*np.array(self.brain.get_space_coarse_centers()).T,
        #                        c=self.brain.get_representation_coarse(),
        #                        s=50*self.brain.get_representation_coarse(),
        #                        cmap="Greys", alpha=0.9)

        # plot edge representation
        # self.axs[0, 0].scatter(*np.array(self.space.get_centers()).T,
        #                        c=self.brain.get_edge_representation(),
        #                        cmap="Oranges", alpha=0.5)

        # -- fine space
        self.axs[0, 0].set_title(f"#PCs={self.brain.get_space_fine_count()}")
        # self.axs[0, 0].set_title(f"Fine-grained space")
        self.axs[0, 0].scatter(*np.array(self.brain.get_space_fine_position()).T,
                               color="red", s=50, marker="v", alpha=0.8)
        self.axs[0, 0].set_xlim(self.boundsx)
        self.axs[0, 0].set_ylim(self.boundsy)
        self.axs[0, 0].set_aspect('equal', adjustable='box')
        self.axs[0, 0].grid(alpha=0.1)
        self.axs[0, 0].set_xticks(())
        self.axs[0, 0].set_yticks(())
        # self.axs[0, 0].legend()

        # -- coarse space
        self.axs[0, 1].set_title(f"#PCs={self.brain.get_space_coarse_count()}")
        # self.axs[0, 1].set_title(f"Coarse-grained space")
        self.axs[0, 1].scatter(*np.array(self.brain.get_space_coarse_position()).T,
                               color="red", s=50, marker="v", alpha=0.8)
        self.axs[0, 1].set_xlim(self.boundsx)
        self.axs[0, 1].set_ylim(self.boundsy)
        self.axs[0, 1].set_xticks(())
        self.axs[0, 1].set_yticks(())
        self.axs[0, 1].set_aspect('equal', adjustable='box')
        self.axs[0, 1].grid(alpha=0.1)
        # self.axs[0, 1].legend()

        # -- BND
        self.axs[1, 0].clear()
        self.axs[1, 0].scatter(*np.array(self.brain.get_space_fine_centers()).T,
                               c=self.brain.get_bnd_weights(),
                               cmap="Blues", alpha=0.8,
                               s=20, vmin=0., vmax=0.2,
                               label="BND")

        daw = self.brain.get_da_weights()
        self.axs[1, 0].scatter(*np.array(self.brain.get_space_fine_centers()).T,
                               c=daw,
                               s=np.where(daw > 0.01, 30, 0),
                               cmap="Greens", alpha=0.8,
                               label="DA",
                               vmin=0., vmax=0.2)
        # self.axs[1, 0].scatter(*np.array(self.brain.get_space_coarse_centers()).T,
        #                        color="brown", s=20, alpha=0.2)
        plan_center_coarse = np.array(self.brain.get_space_coarse_centers())[self.brain.get_plan_idxs_coarse()]
        self.axs[1, 0].plot(*plan_center_coarse.T,
                            color="grey", alpha=0.7, lw=4.,
                            label="plan")
        self.axs[1, 0].scatter(*np.array(self.brain.get_space_fine_centers()).T,
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
        # self.axs[1, 0].scatter(*np.array(self.brain.get_space_fine_position()).T,
        #                            color="red", s=50, marker="v")
        self.axs[1, 0].legend()

        # -- DA
        self.axs[1, 1].clear()
        daw = self.brain.get_da_weights()
        self.axs[1, 1].scatter(*np.array(self.brain.get_space_fine_centers()).T,
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
        # self.axs[1, 1].scatter(*np.array(self.brain.get_space_fine_position()).T,
        #                            color="red", s=50, marker="v")

        plt.pause(0.00001)


def run_model(parameters: dict,
              global_parameters: dict,
              reward_settings: dict,
              game_settings: dict,
              room_name: str="Flat.1011",
              pause: int=-1, verbose: bool=True,
              record_flag: bool=False,
              limit_position_len: int=-1,
              preferred_positions: list=None,
              verbose_min: bool=True) -> int:

    """
    meant to be run standalone
    """

    remap_tag_frequency = parameters["remap_tag_frequency"] if "remap_tag_frequency" in parameters else 200
    remapping_flag = parameters["remapping_flag"] if "remapping_flag" in parameters else 1
    modulation_option = parameters["modulation_option"] if "modulation_option" in parameters else [True]*4
    lr_pred = parameters["lr_pred"] if "lr_pred" in parameters else 0.3

    """ make model """

    brain = pclib.Brain(
                local_scale_fine=global_parameters["local_scale_fine"],
                local_scale_coarse=global_parameters["local_scale_coarse"],
                N=global_parameters["N"],
                Nc=global_parameters["Nc"],
                min_rep_threshold=parameters["min_rep_threshold"],
                num_neighbors=parameters["num_neighbors"],
                rec_threshold_fine=parameters["rec_threshold_fine"],
                rec_threshold_coarse=parameters["rec_threshold_coarse"],
                speed=global_parameters["speed"],
                gain_fine=parameters["gain_fine"],
                offset_fine=parameters["offset_fine"],
                threshold_fine=parameters["threshold_fine"],
                rep_threshold_fine=parameters["rep_threshold_fine"],
                tau_trace_fine=parameters["tau_trace_fine"],
                remap_tag_frequency=parameters['remap_tag_frequency'],
                gain_coarse=parameters["gain_coarse"],
                offset_coarse=parameters["offset_coarse"],
                threshold_coarse=parameters["threshold_coarse"],
                rep_threshold_coarse=parameters["rep_threshold_coarse"],
                tau_trace_coarse=parameters["tau_trace_coarse"],
                lr_da=parameters["lr_da"],
                lr_pred=parameters['lr_pred'],
                threshold_da=parameters["threshold_da"],
                tau_v_da=parameters["tau_v_da"],
                lr_bnd=parameters["lr_bnd"],
                threshold_bnd=parameters["threshold_bnd"],
                tau_v_bnd=parameters["tau_v_bnd"],
                tau_ssry=parameters["tau_ssry"],
                threshold_ssry=parameters["threshold_ssry"],
                threshold_circuit=parameters["threshold_circuit"],
                remapping_flag=parameters['remapping_flag'],
                modulation_option=parameters['modulation_option'],
                rwd_weight=parameters["rwd_weight"],
                rwd_sigma=parameters["rwd_sigma"],
                col_weight=parameters["col_weight"],
                col_sigma=parameters["col_sigma"],
                rwd_field_mod_fine=parameters["rwd_field_mod_fine"],
                rwd_field_mod_coarse=parameters["rwd_field_mod_coarse"],
                col_field_mod_fine=parameters["col_field_mod_fine"],
                col_field_mod_coarse=parameters["col_field_mod_coarse"],
                action_delay=parameters["action_delay"],
                edge_route_interval=parameters["edge_route_interval"],
                forced_duration=parameters["forced_duration"],
                fine_tuning_min_duration=parameters["fine_tuning_min_duration"],
                min_weight_value=parameters["fine_tuning_min_duration"])

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
    print(f"{possible_positions=}")

    agent_possible_positions = possible_positions.copy()
    agent_position = room.sample_next_position()

    rw_tau = reward_settings["tau"] if "tau" in reward_settings else 100

    reward_obj = objects.RewardObj(
                position=possible_positions[0],
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
                            duration=game_settings["max_duration"],
                            visualize=game_settings["rendering"])

    """ run game """

    if env.reward_obj.preferred_positions is not None:
        idx = np.random.choice(env.reward_obj.preferred_positions)
        env.reward_obj.set_position(
                env.room.sample_random_position(idx))

    if verbose_min:
        logger("[@simulations.py]")
    record = run_game(env=env,
             brain=brain,
             renderer=None,
             plot_interval=game_settings["plot_interval"],
             t_teleport=game_settings["t_teleport"],
             pause=-1,
             record_flag=record_flag,
             verbose=verbose,
             verbose_min=verbose_min)

    if verbose_min:
        logger(f"rw_count={env.rw_count}")

    if record_flag:
        record["rw_count"] = env.rw_count
        return record

    info = {
        "env": env,
        "reward_obj": reward_obj,
        "brain": brain
    }

    return env.rw_count, info 


""" MAIN """


def main_game(global_parameters: dict=global_parameters,
              reward_settings: dict=reward_settings,
              t_room_change: int=10000,
              game_settings: dict=game_settings,
              load_idx: int=-1,
              room_name: str="Square.v0", load: bool=False,
              render_type: str="space", duration: int=-1):

    """
    meant to be run standalone
    """

    if load:
        load_idx = load_idx if load_idx > -1 else -1
        parameters = utils.load_parameters(idx=load_idx)
    else:
        parameters = fixed_params

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
                num_neighbors=parameters["num_neighbors"],
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
                rwd_field_mod_fine=parameters["rwd_field_mod_fine"],
                rwd_field_mod_coarse=parameters["rwd_field_mod_coarse"],
                col_field_mod_fine=parameters["col_field_mod_fine"],
                col_field_mod_coarse=parameters["col_field_mod_coarse"],
                action_delay=parameters["action_delay"],
                edge_route_interval=parameters["edge_route_interval"],
                forced_duration=parameters["forced_duration"],
                fine_tuning_min_duration=parameters["fine_tuning_min_duration"],
                min_weight_value=parameters["fine_tuning_min_duration"])

    """ make game environment """

    room = games.make_room(name="Square.v0",
                           thickness=game_settings["room_thickness"],
                           bounds=[0, 1, 0, 1])
    room_2 = games.make_room(name="Square.b",
                           thickness=game_settings["room_thickness"],
                           bounds=[0, 1, 0, 1])
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    # ===| objects |===

    possible_positions = room.get_room_positions()
    logger.debug(f"{possible_positions=}")

    agent_possible_positions = possible_positions.copy()
    agent_position = room.sample_next_position()

    rw_tau = reward_settings["tau"] if "tau" in reward_settings else 400
    if "move_threlshold" in reward_settings:
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

    duration = game_settings["max_duration"] if duration < 0 else duration
    verbose_min = False
    verbose = True
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
                            visualize=game_settings["rendering"])
    logger(env)


    """ run game """

    if game_settings["rendering"]:
        renderer = Renderer(brain=brain, render_type=render_type)
    else:
        renderer = None

    logger("[@simulations.py]")

    # ===| setup |===
    clock = pygame.time.Clock()
    last_position = np.zeros(2)

    # [position, velocity, collision, reward, done, terminated]
    observation = [[0., 0.], 0., 0., False, False]
    prev_position = env.position
    room_changed = False

    record = {"activity_fine": [],
              "activity_coarse": [],
              "trajectory": []}


    # ===| main loop |===
    # running = True
    # while running:
    for _ in tqdm(range(env.duration), desc="Game", leave=False,
                  disable=not verbose_min):

        # Event handling
        if env.visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # -check: teleport
        if env.t % t_teleport == 0 and env.reward_obj.is_silent: # <=================================
            env._reset_agent_position(brain, True)

        # -check: change room
        if env.t > t_room_change and not room_changed:
            env.room = room_2
            body.room = room_2
            logger(f"Room change -> {env.room}")
            room_changed = True

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

        # env step
        observation = env(velocity=np.array([velocity[0], -velocity[1]]),
                          brain=brain)

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
                    renderer()

        # -check: record
        if record_flag:
            record["activity_fine"] += [brain.get_representation_fine()]
            record["activity_coarse"] += [brain.get_representation_coarse()]
            record["trajectory"] += [env.position]

        # -check: exit
        if observation[4]:
            if verbose and verbose_min:
                logger.debug(">> Game terminated <<")
            break

        # pause
        if pause > 0:
            pygame.time.wait(pause)

    pygame.quit()

    logger(f"rw_count={env.rw_count}")

    plt.show()



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
    parser.add_argument("--tchange", type=int, default=1000000,
                        help="time to change room")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_type", type=str, default="space")

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
        main_game(room_name=args.room, load=args.load,
                  duration=args.duration,
                  t_room_change=args.tchange,
                  render_type=args.render_type,
                  load_idx=args.idx)
    except KeyboardInterrupt:
        logger.debug("Keyboard interrupt")
        plt.show()

