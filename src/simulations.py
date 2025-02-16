import numpy as np
import matplotlib.pyplot as plt
import argparse, json
from tqdm import tqdm
import pygame

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
    "rw_radius": 0.03 * GAME_SCALE,
    "rw_sigma": 0.75 * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 5,
    "silent_duration": 4_000,
    "fetching_duration": 1,
    "transparent": False,
    "beta": 35.,
    "alpha": 0.06,# * GAME_SCALE,
}

agent_settings = {
    # "speed": 0.7,
    "init_position": np.array([0.2, 0.2]) * GAME_SCALE,
    "agent_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move agent",
    "rendering": True,
    "rendering_pcnn": True,
    "max_duration": 8_000,
    "room_thickness": 30,
    "seed": None,
    "pause": -1,
    "verbose": True
}

model_params = {
    "N": 30**2,
    "bnd_threshold": 0.2,
    "bnd_tau": 1,
    "threshold": 0.3,
    "rep_threshold": 0.5,
    "action_delay": 6,
}

global_parameters = {
    "local_scale_fine": 0.015,
    "local_scale_coarse": 0.006,
    "N": 30**2,
    "Nc": 20**2,
    "rec_threshold_fine": 28.,
    "rec_threshold_coarse": 70.,
    "speed": 1.5,
    "min_weight_value": 0.5
}

parameters = {

    "gain_fine": 10.,
    "offset_fine": 1.1,
    "threshold_fine": 0.3,
    "rep_threshold_fine": 0.88,
    "min_rep_threshold": 0.95,

    "gain_coarse": 8.,
    "offset_coarse": 0.9,
    "threshold_coarse": 0.3,
    "rep_threshold_coarse": 0.8,

    "lr_da": 0.3,
    "threshold_da": 0.04,
    "tau_v_da": 1.0,

    "lr_bnd": 0.4,
    "threshold_bnd": 0.05,
    "tau_v_bnd": 2.0,

    "tau_ssry": 100.,
    "threshold_ssry": 0.998,

    "threshold_circuit": 0.1,

    "rwd_weight": 0.1,
    "rwd_sigma": 40.0,
    "col_weight": 0.0,
    "col_sigma": 2.0,

    "action_delay": 50.,
    "edge_route_interval": 50,

    "forced_duration": 100,
    "fine_tuning_min_duration": 10,
}

fixed_params = parameters.copy()


possible_positions = np.array([
    [0.25, 0.75], [0.75, 0.75],
    [0.25, 0.25], [0.75, 0.25]]) * GAME_SCALE


""" UTILITIES """


class Renderer:

    def __init__(self, elements, space_fine, space_coarse,
                 brain, colors, names):

        self.elements = elements
        self.size = len(elements)
        self.space_fine = space_fine
        self.space_coarse = space_coarse
        self.brain = brain
        self.colors = colors
        self.names = names
        self.fig, self.axs = plt.subplots(2, 2, figsize=(6, 6))
        self.fig.set_tight_layout(True)
        self.boundsx = (-600, 600)
        self.boundsy = (-600, 600)

    def render(self):

        # -- space plots --
        self.axs[0, 0].clear()
        self.axs[0, 1].clear()

        # -- goal-behaviour
        if self.brain.get_directive() == "trg" or \
           self.brain.get_directive() == "trg rw" or \
           self.brain.get_directive() == "trg ob":

            # space
            self.axs[0, 0].scatter(*np.array(self.space_fine.get_centers()).T,
                                   color="blue", s=20, alpha=0.1)
            # current position
            self.axs[0, 0].scatter(*np.array(self.space_fine.get_centers()).T,
                                   c = self.brain.get_representation(),
                                   cmap="Greys",
                                   s=20, alpha=0.3)
            # goal
            self.axs[0, 0].scatter(*np.array(self.space_fine.get_centers()).T,
                                   c=self.brain.get_trg_representation(),
                                   s=100*self.brain.get_trg_representation(),
                                   cmap="Greens", alpha=0.7)

            # plan
            plan = np.zeros(len(self.brain))
            plan[self.brain.get_plan_idxs_fine()] = 1.
            self.axs[0, 0].scatter(*np.array(self.space_fine.get_centers()).T,
                        c=plan, s=40*plan, cmap="Greens", alpha=0.7)

            # title
            loc_ = np.array(self.space_fine.get_centers())[self.brain.get_trg_idx()]
            self.axs[0, 0].set_title(f"Space | trg_idx={self.brain.get_trg_idx()} " + \
                f" ({self.brain.get_trg_representation().max():.3f}, " + \
                f"{self.brain.get_trg_representation().argmax()}) " + \
                f" loc={loc_}")

            # space
            self.axs[0, 1].scatter(*np.array(self.space_coarse.get_centers()).T,
                                   color="blue", s=50, alpha=0.1)
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
            plan = np.zeros(self.space_coarse.get_size())
            plan[self.brain.get_plan_idxs_coarse()] = 1.
            self.axs[0, 1].scatter(*np.array(self.space_coarse.get_centers()).T,
                        c=plan, s=50*plan, cmap="Greens", alpha=0.7)

            # title
            # loc_ = np.array(self.space.get_centers())[self.brain.get_trg_idx()]
            self.axs[0, 1].set_title(f"Coarse space") #| trg_idx={self.brain.get_trg_idx()} " + \
                # f" ({self.brain.get_trg_representation().max():.3f}, " + \
                # f"{self.brain.get_trg_representation().argmax()}) " + \
                # f" loc={loc_}")

        else:
            # fine space
            self.axs[0, 0].scatter(*np.array(self.space_fine.get_centers()).T,
                                   color="blue", s=20, alpha=0.4)

            # coarse space
            # for edge in self.space_coarse.make_edges():
            #     self.axs[0, 1].plot((edge[0][0], edge[1][0]),
            #                         (edge[0][1], edge[1][1]),
            #                      alpha=0.1, color="black")
            self.axs[0, 1].scatter(*np.array(self.space_coarse.get_centers()).T,
                                   color="blue", s=40, alpha=0.4)

            # for edge in self.space.make_edges():
            #     self.axs[0, 0].plot((edge[0][0], edge[1][0]),
            #                         (edge[0][1], edge[1][1]),
            #                      alpha=0.1, color="black")

        # current representation
        self.axs[0, 0].scatter(*np.array(self.space_fine.get_centers()).T,
                               c = self.brain.get_representation(), cmap="Greys",
                               s=40*self.brain.get_representation(),
                               alpha=0.9)
        self.axs[0, 1].scatter(*np.array(self.space_coarse.get_centers()).T,
                               c=self.brain.get_representation_coarse(),
                               s=50*self.brain.get_representation_coarse(),
                               cmap="Greys", alpha=0.9)

        # plot edge representation
        # self.axs[0, 0].scatter(*np.array(self.space.get_centers()).T,
        #                        c=self.brain.get_edge_representation(),
        #                        cmap="Oranges", alpha=0.5)

        # fine
        # self.axs[0, 0].set_title(f"#PCs={len(self.space)} | " + \
            # f"dir:{self.brain.get_directive()}")
        self.axs[0, 0].set_title(f"#PCs={len(self.space_fine)}")
        self.axs[0, 0].scatter(*np.array(self.space_fine.get_position()).T,
                               color="red", s=50, marker="v", alpha=0.8)
        self.axs[0, 0].set_xlim(self.boundsx)
        self.axs[0, 0].set_ylim(self.boundsy)
        self.axs[0, 0].set_aspect('equal', adjustable='box')
        self.axs[0, 0].grid(alpha=0.1)
        self.axs[0, 0].set_xticks(())
        self.axs[0, 0].set_yticks(())

        # coarse
        self.axs[0, 1].set_title(f"#PCs={len(self.space_coarse)}")
        self.axs[0, 1].scatter(*np.array(self.space_coarse.get_position()).T,
                               color="red", s=50, marker="v", alpha=0.8)
        self.axs[0, 1].set_xlim(self.boundsx)
        self.axs[0, 1].set_ylim(self.boundsy)
        self.axs[0, 1].set_xticks(())
        self.axs[0, 1].set_yticks(())
        self.axs[0, 1].set_aspect('equal', adjustable='box')
        self.axs[0, 1].grid(alpha=0.1)

        self.axs[1, 0].clear()
        self.axs[1, 0].scatter(*np.array(self.space_fine.get_centers()).T,
                               c=self.elements[1].get_weights(),
                               cmap=self.colors[1], alpha=0.5,
                               s=30, vmin=0., vmax=0.1)
        self.axs[1, 0].set_xlim(self.boundsx)
        self.axs[1, 0].set_ylim(self.boundsy)
        self.axs[1, 0].set_xticks(())
        self.axs[1, 0].set_yticks(())
        self.axs[1, 0].set_title(f"BND")# [{self.brain.get_leaky_v()[1]:.2f}]")
        self.axs[1, 0].set_aspect('equal', adjustable='box')
        self.axs[1, 0].scatter(*np.array(self.space_fine.get_position()).T,
                                   color="red", s=50, marker="v")

        self.axs[1, 1].clear()
        daw = self.elements[0].get_weights()
        self.axs[1, 1].scatter(*np.array(self.space_fine.get_centers()).T,
                               c=self.elements[0].get_weights(),
                               s=np.where(daw > 0.01, 30, 1),
                               cmap=self.colors[0], alpha=0.8,
                               vmin=0.)
        self.axs[1, 1].set_xlim(self.boundsx)
        self.axs[1, 1].set_ylim(self.boundsy)
        self.axs[1, 1].set_xticks(())
        self.axs[1, 1].set_yticks(())
        self.axs[1, 1].set_title(f"DA")# [{self.brain.get_leaky_v()[0]:.2f}]")# + \
                    # f" | {self.elements[0].get_weights().max():.2f}")
        self.axs[1, 1].set_aspect('equal', adjustable='box')
        self.axs[1, 1].scatter(*np.array(self.space_fine.get_position()).T,
                                   color="red", s=50, marker="v")
            # if i == 1:
            # self.axs[i].set_title(f"")
            # elif i == 2:
                # text = f"g:{self.space.get_gain():.3f} "
                # text += f"r:{self.space.get_rep():.3f} "
                # text += f"t:{self.space.get_threshold():.3f}"
                # self.axs[i].set_title(text)

        # plt.axis("equal")
        plt.pause(0.00001)


def run_game(env: games.Environment,
             brain: object,
             renderer: object,
             plot_interval: int,
             pause: int=-1,
             verbose: bool=True,
             verbose_min: bool=True):

    # ===| setup |===
    clock = pygame.time.Clock()
    last_position = np.zeros(2)

    # [position, velocity, collision, reward, done, terminated]
    observation = [[0., 0.], 0., 0., False, False]
    prev_position = env.position

    # ===| main loop |===
    running = True
    while running:

        # Event handling
        if env.visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # -reward
        if observation[2] and verbose and verbose_min:
            logger.debug(f">> Reward << [{observation[2]}]")
            # input()

        # -collision
        if observation[2] and verbose and verbose_min:
            logger.debug(f">!!< Collision << [{observation[2]}]")
            # input()

        # velocity
        v = [(env.position[0] - prev_position[0]),
             (-env.position[1] + prev_position[1])]

        # brain step
        velocity = brain(v,
                         observation[1],
                         observation[2],
                         env.reward_availability)
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
            running = False
            # brain.reset(position=env.agent.position)

        # -check: render
        if env.visualize:
            if env.t % plot_interval == 0:
                env.render()
                if renderer:
                    renderer.render()
        # else:
        #     if env.t % plot_interval == 0 and verbose:
        #         logger(f"{env.t/env.duration*100:.1f}%")

        # -check: exit
        if observation[4]:
            running = False
            if verbose and verbose_min:
                logger.debug(">> Game terminated <<")

        # pause
        if pause > 0:
            pygame.time.wait(pause)

    pygame.quit()


def run_model(parameters: dict, global_parameters: dict,
              agent_settings: dict, reward_settings: dict,
              game_settings: dict, room_name: str="Flat.1011",
              pause: int=-1, verbose: bool=True,
              verbose_min: bool=True) -> int:

    """
    meant to be run standalone
    """

    """ make model """

    brain = pclib.Brainv2(
                local_scale_fine=global_parameters["local_scale_fine"],
                local_scale_coarse=global_parameters["local_scale_coarse"],
                N=global_parameters["N"],
                Nc=global_parameters["Nc"],
                min_rep_threshold=parameters["min_rep_threshold"],
                rec_threshold_fine=global_parameters["rec_threshold_fine"],
                rec_threshold_coarse=global_parameters["rec_threshold_coarse"],
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

    if verbose and verbose_min:
        logger(f"room_name={room_name}")

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
                position=rw_position,
                possible_positions=constants.POSSIBLE_POSITIONS.copy(),
                radius=reward_settings["rw_radius"],
                sigma=reward_settings["rw_sigma"],
                fetching=reward_settings["rw_fetching"],
                value=reward_settings["rw_value"],
                bounds=room_bounds,
                delay=reward_settings["delay"],
                silent_duration=reward_settings["silent_duration"],
                transparent=reward_settings["transparent"])

    body = objects.AgentBody(
                position=agent_position,
                speed=global_parameters["speed"],
                possible_positions=agent_possible_positions,
                bounds=agent_settings["agent_bounds"],
                room=room,
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

    # if game_settings["rendering"]:
    #     renderer = Renderer(elements=[brain.get_da(), brain.get_bnd()],
    #                         space=brain.get_space_fine(),
    #                         space_coarse=brain.get_space_coarse(),
    #                         brain=brain, colors=["Greens", "Blues"],
    #                         names=["DA", "BND"])
    # else:
    #     renderer = None

    if verbose_min:
        logger("[@simulations.py]")
    run_game(env=env,
             brain=brain,
             renderer=None,
             plot_interval=game_settings["plot_interval"],
             pause=-1,
             verbose=verbose,
             verbose_min=verbose_min)

    if verbose_min:
        logger(f"rw_count={env.rw_count}")

    return env.rw_count


""" MAIN """


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
           pclib.GridLayerSq(sigma=0.04, speed=0.7*local_scale_fine, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.5*local_scale_fine, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.3*local_scale_fine, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.05*local_scale_fine, bounds=[-1, 1, -1, 1])])

    space_fine = pclib.PCNN(N=global_parameters["N"],
                            Nj=len(gcn),
                            gain=parameters["gain_fine"],
                            offset=parameters["offset_fine"],
                            clip_min=0.01,
                            threshold=parameters["threshold_fine"],
                            rep_threshold=parameters["rep_threshold_fine"],
                            rec_threshold=global_parameters["rec_threshold_fine"],
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
                             rec_threshold=global_parameters["rec_threshold_coarse"],
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
        renderer = Renderer(elements=[da, bnd], space_fine=space_fine,
                            space_coarse=space_coarse,
                            brain=brain, colors=["Greens", "Blues"],
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
    if args.main == "game":
        main_game(room_name=args.room, load=args.load, duration=args.duration)
    elif args.main == "rand":
        main_game_rand(room_name=args.room)
    elif args.main == "simple":
        main_simple_square(duration=args.duration)
    else:
        logger.error("main not found ...")

