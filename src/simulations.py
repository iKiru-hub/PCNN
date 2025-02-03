import numpy as np
import matplotlib.pyplot as plt
import argparse, json
from tqdm import tqdm
import pygame

import utils
import game.envs as games
import game.objects as objects

try:
    # import libs.pclib as pclib
    import core.build.pclib as pclib
except ImportError:
    import warnings
    warnings.warn("pclib [c++] not found, using python version")
    import libs.pclib1 as pclib


logger = utils.setup_logger(name="RUN",
                           level=2,
                           is_debugging=True,
                           is_warning=False)


""" SETTINGS """

GAME_SCALE = games.SCREEN_WIDTH


reward_settings = {
    "rw_fetching": "deterministic",
    "rw_position": np.array([0.5, 0.5]) * GAME_SCALE,
    "rw_radius": 0.04 * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 2,
}

agent_settings = {
    "speed": 1.5,
    "init_position": np.array([0.4, 0.5]) * GAME_SCALE,
    "agent_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
}

game_settings = {
    "plot_interval": 1,
    "rw_event": "move agent",
    "rendering": True,
    "rendering_pcnn": True,
    "max_duration": None,
    "seed": None
}

model_params = {
    "N": 30**2,
    "bnd_threshold": 0.2,
    "bnd_tau": 1,
    "threshold": 0.3,
    "rep_threshold": 0.5,
    "action_delay": 6,
}

possible_positions = np.array([
    [0.2, 0.2], [0.2, 0.8],
    [0.8, 0.8], [0.8, 0.2],
]) * GAME_SCALE


""" UTILITIES """


class Renderer:

    def __init__(self, elements, space,
                 brain, colors, names):

        self.elements = elements
        self.size = len(elements)
        self.space = space
        self.brain = brain
        self.colors = colors
        self.names = names
        self.fig, self.axs = plt.subplots(1, self.size+1,
                                         figsize=((1+self.size)*4, 2))
        self.bounds = (-30, 40)

    def render(self):
        self.axs[0].clear()

        if self.brain.get_directive() == "trg":
            self.axs[0].scatter(*np.array(self.space.get_centers()).T,
                        c=self.brain.get_trg_representation(), s=100,
                                cmap="Greens", alpha=0.5)
            self.axs[0].set_title(f"Space | trg_idx={self.brain.get_trg_idx()} " + \
                f" ({self.brain.get_trg_representation().max():.3f}, " + \
                f"{self.brain.get_trg_representation().argmax()})")

        else:
            self.axs[0].scatter(*np.array(self.space.get_centers()).T,
                        color="blue", s=30, alpha=0.4)
            for edge in self.space.make_edges():
                self.axs[0].plot((edge[0][0], edge[1][0]), (edge[0][1], edge[1][1]),
                                 alpha=0.1, color="black")
            self.axs[0].set_title(f"Space | #PCs={len(self.space)}")

        self.axs[0].scatter(*np.array(self.space.get_position()).T,
                            color="red", s=80, marker="o", alpha=0.8)
        self.axs[0].set_xlim(self.bounds)
        self.axs[0].set_ylim(self.bounds)
        # equal aspect ratio
        self.axs[0].set_aspect('equal', adjustable='box')

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
            self.axs[i].scatter(*np.array(self.space.get_position()).T,
                                color="red", s=80, marker="o")
            if i == 1:
                self.axs[1].set_title(f"directive={self.brain.get_directive()}")

        # plt.axis("equal")
        plt.pause(0.00001)


def run_game(env: games.Environment,
             brain: object,
             renderer: object,
             plot_interval: int,
             pause: int=-1):

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

        # brain step
        velocity = brain([(env.position[0] - prev_position[0]) / env.scale,
                          (env.position[1] - prev_position[1]) / env.scale],
                         observation[1],
                         observation[2],
                         env.reward_availability)

        # store past position
        prev_position = env.position

        # env step
        observation = env(velocity=np.array(velocity), brain=brain)

        # -check: reset agent's brain
        if observation[3]:
            logger.info(">> Game reset <<")
            brain.reset(position=env.agent.position)

        # -check: render
        if env.visualize:
            if env.t % plot_interval == 0:
                env.render()
                if renderer:
                    renderer.render()

        # -check: exit
        if observation[4]:
            running = False
            logger.debug(">> Game terminated <<")

        # pause
        if pause > 0:
            pygame.time.wait(pause)

    pygame.quit()



""" MAIN """


def main_game_rand(room_name: str="Square.v0"):

    SCALE = 100.0
    brain = objects.RandomAgent(scale=SCALE)

    room = games.make_room(name=room_name)
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    agent = objects.AgentBody(position=np.array([110, 110]),
                              width=25, height=25,
                              bounds=room_bounds,
                              possible_positions=[
                                    np.array([110, 110]),
                                    np.array([110, 190]),
                                    np.array([190, 110]),
                                    np.array([190, 190])],
                              max_speed=4.0)
    reward_obj = objects.RewardObj(position=np.array([150, 150]),
                                bounds=room_bounds)

    env = games.Environment(room=room, agent=agent,
                            reward_obj=reward_obj,
                            rw_event="move both",
                            duration=args.duration,
                            scale=SCALE,
                            visualize=True)

    run_game(env, brain, fps=100)


def main_game(room_name: str="Square.v0"):

    """
    meant to be run standalone
    """

    """ make model """

    # ===| space |===

    gcn = pclib.GridNetworkSq([
           pclib.GridLayerSq(sigma=0.04, speed=0.1, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.08, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.07, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.05, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.03, bounds=[-1, 1, -1, 1]),
           pclib.GridLayerSq(sigma=0.04, speed=0.005, bounds=[-1, 1, -1, 1])])

    space = pclib.PCNNsqv2(N=model_params["N"],
                           Nj=len(gcn),
                           gain=10.,
                           offset=1.3,
                           clip_min=0.01,
                           threshold=0.4,
                           rep_threshold=0.92,
                           rec_threshold=4.,
                           num_neighbors=4,
                           xfilter=gcn,
                           name="2D")

    # ===| modulation |===

    da = pclib.BaseModulation(name="DA", size=model_params["N"],
                              lr=0.2, threshold=0.03, max_w=1.0,
                              tau_v=1.0, eq_v=0.0, min_v=0.0)
    bnd = pclib.BaseModulation(name="BND", size=model_params["N"],
                               lr=0.3, threshold=0.01, max_w=1.0,
                               tau_v=1.0, eq_v=0.0, min_v=0.0)
    memrepr = pclib.MemoryRepresentation(model_params["N"], 2.0, 0.1)
    memact = pclib.MemoryAction(3.0)
    circuit = pclib.Circuits(da, bnd, memrepr, memact)

    # ===| target program |===

    # trgp = pclib.TargetProgram(space.get_connectivity(), space.get_centers(),
    #                            da.get_weights(), agent_settings["speed"])

    expmd = pclib.ExperienceModule(speed=agent_settings["speed"],
                                   circuits=circuit,
                                   space=space,
                                   weights=[-1., 0., -0.1, -1., -1.],
                                   action_delay=5)
    brain = pclib.Brain(circuit, space, expmd, agent_settings["speed"],
                        5)


    """ make game environment """

    room = games.make_room(name=room_name, thickness=20.)
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    # ===| objects |===

    body = objects.AgentBody(
                position=agent_settings["init_position"],
                possible_positions=possible_positions,
                bounds=agent_settings["agent_bounds"],
                room=room,
                color=(10, 10, 10))
    reward_obj = objects.RewardObj(
                position=reward_settings["rw_position"],
                radius=reward_settings["rw_radius"],
                fetching=reward_settings["rw_fetching"],
                bounds=room_bounds,
                delay=reward_settings["delay"])
    logger(reward_obj)

    # --- env
    env = games.Environment(room=room,
                            agent=body,
                            reward_obj=reward_obj,
                            scale=GAME_SCALE,
                            rw_event=game_settings["rw_event"],
                            verbose=False,
                            visualize=game_settings["rendering"])
    logger(env)


    """ run game """

    renderer = Renderer(elements=[da, bnd], space=space,
                        brain=brain, colors=["Greens", "Blues"],
                        names=["DA", "BND"])

    run_game(env=env,
             brain=brain,
             renderer=renderer,
             plot_interval=5,
             pause=-1)


def main_simple_square(duration: int):

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
    parser.add_argument("--seed", type=int, default=-1,
                        help="random seed: -1 for random seed.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--room", type=str, default="Square.v0",
                        help='room name: ["Square.v0", "Square.v1", "Square.v2",' + \
                         '"Hole.v0", "Flat.0000", "Flat.0001", "Flat.0010", "Flat.0011",' + \
                         '"Flat.0110", "Flat.1000", "Flat.1001", "Flat.1010",' + \
                         '"Flat.1011", "Flat.1110"]')
    parser.add_argument("--main", type=str, default="game",
                        help="[game, rand, simple]")

    args = parser.parse_args()

    # --- seed
    if args.seed > 0:
        logger.debug(f"seed: {args.seed}")
        np.random.seed(args.seed)

    # --- run
    if args.main == "game":
        main_game(room_name=args.room)
    elif args.main == "rand":
        main_game_rand(room_name=args.room)
    elif args.main == "simple":
        main_simple_square(duration=args.duration)
    else:
        logger.error("main not found ...")

