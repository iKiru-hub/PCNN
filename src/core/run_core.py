import numpy as np
import matplotlib.pyplot as plt
import argparse, json
from tqdm import tqdm

import mod_core as mod
import utils_core as utc
import envs_core as ev

import pclib


""" INITIALIZATION  """


CONFIGPATH = "dashboard/cache/configs.json"

logger = utc.setup_logger(name="RUN",
                          level=0,
                          is_debugging=True,
                          is_warning=False)


def edit_logger(level: int=-1,
                is_debugging: bool=True,
                is_warning: bool=False):
    global logger
    logger.set_level(level)
    logger.set_debugging(is_debugging)
    logger.set_warning(is_warning)


def write_configs(num_figs: int,
                  circuits: object,
                  t: int,
                  trg_module: object,
                  observation: dict=None,
                  other: dict={}):

    info = {
        "num_figs": num_figs,
        "circuits": circuits.names,
        "t": t,
        "position": np.around(observation["position"], 3).tolist(),
    }

    info = info | other

    with open(CONFIGPATH, 'w') as f:
        json.dump(info, f)



""" CLASSES """

sim_settings = {
    "bounds": np.array([0., 1., 0., 1.]),
    "speed": 0.03,
    "init_position": np.array([0.8, 0.2]),
    "rw_fetching": "probabilistic",
    "rw_behaviour": "static",
    "rw_position": np.array([0.5, 0.8]),
    "rw_radius": 0.1,
    "plot_interval": 8,
    "rendering": True,
    "room": "square",
    "max_duration": None,
    "seed": None
}

agent_settings = {
    "N": 80,
    "Nj": 13**2,
    "sigma": 0.04,
    "max_depth": 10
}

model_params = {
    "threshold": 0.5,
    "rep_threshold": 0.5,
    "w1": -1.,  # bnd
    "w2": 0.2,  # dpos
    "w3": -0.5,  # pop
    "w4": 1.,  # trg
    "w5": 0.4,  # smooth
}


class Simulation:

    """
    very similar to main(), but it is meant to be run
    with the streamlit dashboard
    """

    def __init__(self, sim_settings: dict = sim_settings,
                 agent_settings: dict = agent_settings,
                 model_params: dict = model_params,
                 pcnn2D: pclib.PCNN = None):

        # --- SETTINGS
        self.sim_settings = sim_settings
        self.agent_settings = agent_settings

        seed = sim_settings["seed"]
        SPEED = sim_settings["speed"]
        init_position = sim_settings["init_position"]
        rw_position = sim_settings["rw_position"]
        rw_radius = sim_settings["rw_radius"]
        rendering = sim_settings["rendering"]

        self.plot_interval = sim_settings["plot_interval"]
        self.rendering = rendering
        self.max_duration = sim_settings["max_duration"]
        self.info = {}

        # --- MODEL SETTINGS
        N = agent_settings["N"]
        Nj = agent_settings["Nj"]
        exp_weights = np.array([w for (k, w) in model_params.items()
                                if "w" in k.lower()])

        logger(f"{N=}")
        logger(f"{Nj=}")

        # ---
        self.bounds = sim_settings["bounds"]

        if pcnn2D is None:
            sigma = agent_settings["sigma"]
            xfilter = pclib.PCLayer(int(np.sqrt(Nj)),
                                    sigma, self.bounds)

            # definition
            pcnn2D = pclib.PCNN(N=N, Nj=Nj, gain=3., offset=1.,
                                clip_min=0.09,
                                threshold=model_params["threshold"],
                                rep_threshold=model_params["rep_threshold"],
                                rec_threshold=0.01,
                                num_neighbors=8, trace_tau=0.1,
                                xfilter=xfilter, name="2D")

        pcnn2D_plotter = utc.PlotPCNN(model=pcnn2D,
                                      bounds=self.bounds,
                                      visualize=rendering)

        # --- CIRCUITS
        circuits_dict = {"Bnd": mod.BoundaryMod(N=N,
                                                threshold=0.1,
                                                eta=0.5,
                                                tau=1.,
                                                visualize=rendering),
                         "DA": mod.Dopamine(N=N,
                                            threshold=0.15,
                                            visualize=rendering),
                         "dPos": mod.PositionTrace(visualize=False),
                         "Pop": mod.PopulationProgMax(N=N,
                                                      visualize=False,
                                                      number=None),
                         "Ftg": mod.FatigueMod(tau=300)}

        circuits = mod.Circuits(circuits_dict=circuits_dict,
                                visualize=rendering)

        # --- MODULES
        trg_module = mod.TargetModule(pcnn=pcnn2D,
                                      circuits=circuits,
                                      speed=SPEED,
                                      threshold=0.01,
                                      visualize=rendering)

        # [ bnd, dpos, pop, trg ]
        exp_module = mod.ExperienceModule(
                            pcnn=pcnn2D,
                            pcnn_plotter=pcnn2D_plotter,
                            trg_module=trg_module,
                            circuits=circuits,
                            weights=exp_weights,
                            max_depth=agent_settings["max_depth"],
                            speed=SPEED,
                            visualize=False,
                            visualize_action=rendering)
        self.agent = mod.Brain(exp_module=exp_module,
                               pcnn2D=pcnn2D,
                               circuits=circuits)

        # --- agent & env
        self.env = ev.make_room(name=sim_settings["room"],
                                thickness=4.,
                                visualize=rendering)
        self.env = ev.AgentBody(room=self.env,
                                position=init_position)
        self.reward_obj = ev.RewardObj(position=rw_position,
                                       radius=rw_radius,
                                       fetching=sim_settings["rw_fetching"],
                                       behaviour=sim_settings["rw_behaviour"],
                                       bounds=self.bounds)
        self.velocity = np.zeros(2)
        self.observation = {
            "u": np.zeros(N),
            "position": self.env.position,
            "velocity": self.velocity,
            "delta_update": 0.,
            "collision": False,
            "reward": 0.
        }

        # --- RECORD
        self.agent.record["trajectory"] += [
                    self.env.position.tolist()]
        self.t = -1

        # --- visutalization
        if rendering:
            self.figures = self.agent.render(return_fig=True)
            self.fig_a, self.ax_a = plt.subplots(figsize=(4, 4))

    def update(self) -> list:

        """
        update the simulation and return the figures
        """

        self.t += 1

        # --- env
        position, collision, truncated = self.env(
                        velocity=self.velocity)
        reward = self.reward_obj(position=position)

        # --- observation
        self.observation["position"] = position
        self.observation["collision"] = collision
        self.observation["reward"] = reward

        # --- agent
        self.velocity = self.agent(
                    observation=self.observation)
        # --- plot
        if self.rendering:
            self._render()
            return [fig if fig is not None else plt.figure() for fig in self.figures]

        # --- exit
        if self.max_duration is not None and self.t >= self.max_duration:
            logger.warning(f"truncated at t={self.t}")
            return True

    def step(self): return self.update()

    def output(self): pass

    def get_trajectory(self):
        return self.agent.record["trajectory"]

    def get_pcnn_graph(self):

        centers = self.agent.exp_module.pcnn.get_centers()
        connectivity = self.agent.exp_module.pcnn.get_wrec()

        return centers, connectivity

    def get_reward_visit(self):
        return self.agent.circuits.circuits["DA"].weights.sum() > 0.

    def get_reward_info(self):
        return self.sim_settings["rw_position"], \
                    self.sim_settings["rw_radius"]

    def get_reward_count(self):
        return self.reward_obj.get_count()

    def get_pcnn2D(self):
        return self.agent.exp_module.pcnn

    def _render(self):

        if self.t % self.plot_interval == 0:

            # env
            self.ax_a.clear()
            self.env.render(ax=self.ax_a, velocity=self.velocity)
            self.ax_a.axis("off")
            self.ax_a.set_title(f"t={self.t} | v={np.around(self.velocity, 3)} ")
            self.ax_a.set_xlim(self.bounds[:2])
            self.ax_a.set_ylim(self.bounds[2:])

            self.figures = []
            for f in [self.fig_a] + self.agent.render(return_fig=True):
                if f is not None:
                    self.figures.append(f)

    def set_rw_position(self, rw_position: np.ndarray):
        self.sim_settings["rw_position"] = rw_position

    def reset(self, seed: int=None, init_position: np.ndarray=None):

        self.sim_settings["seed"] = seed
        self.sim_settings["init_position"] = init_position
        self.reward_obj.reset()

        self.__init__(sim_settings=self.sim_settings,
                      agent_settings=self.agent_settings)
        logger(f"%% reset [seed={seed}] %%")


def main(sim_settings=sim_settings,
         agent_settings=agent_settings,
         model_params=model_params,
         plot: bool=False):

    """
    meant to be run standalone
    """

    # --- settings
    duration = sim_settings["max_duration"]
    trg_position = sim_settings["rw_position"]
    trg_radius = sim_settings["rw_radius"]
    SPEED = sim_settings["speed"]
    PLOT_INTERVAL = sim_settings["plot_interval"]
    ROOM = sim_settings["room"]
    BOUNDS = sim_settings["bounds"]

    exp_weights = np.array([w for (k, w) in model_params.items()
                            if "w" in k.lower()])

    other_info = {}
    logger(f"room: {ROOM}")
    logger(f"plot_interval: {PLOT_INTERVAL}")
    logger(f"{duration}")

    # --- brain
    N = agent_settings["N"]
    Nj = agent_settings["Nj"]
    sigma = agent_settings["sigma"]

    logger(f"{N=}")
    logger(f"{Nj=}")

    xfilter = pclib.PCLayer(int(np.sqrt(Nj)), sigma, BOUNDS)

    # definition
    pcnn2D = pclib.PCNN(N=N, Nj=Nj, gain=3., offset=1.,
                        clip_min=0.09,
                        threshold=model_params["threshold"],
                        rep_threshold=model_params["rep_threshold"],
                        rec_threshold=0.01,
                        num_neighbors=8, trace_tau=0.1,
                        xfilter=xfilter, name="2D")

    # plotter
    pcnn2D_plotter = utc.PlotPCNN(model=pcnn2D,
                                  bounds=BOUNDS,
                                  visualize=True,
                                  number=0)

    # --- circuits
    circuits_dict = {"Bnd": mod.BoundaryMod(N=N,
                                            threshold=0.15,
                                            eta=0.1,
                                            tau=1.,
                                            visualize=True,
                                            number=5),
                     "DA": mod.Dopamine(N=N,
                                        threshold=0.15,
                                        visualize=True,
                                        number=4),
                     "dPos": mod.PositionTrace(visualize=False),
                     "Pop": mod.PopulationProgMax(N=N,
                                                  visualize=False,
                                                  number=None),
                     "Ftg": mod.FatigueMod(tau=300)}

    # object
    circuits = mod.Circuits(circuits_dict=circuits_dict,
                            visualize=True,
                            number=3)

    # --- modules

    trg_module = mod.TargetModule(pcnn=pcnn2D,
                                  circuits=circuits,
                                  speed=SPEED,
                                  threshold=0.1,
                                  visualize=True,
                                  number=1)
    other_info["Trg_thr"] = trg_module.threshold

    # [ bnd, dpos, pop, trg, smooth ]
    exp_module = mod.ExperienceModule(pcnn=pcnn2D,
                                      pcnn_plotter=pcnn2D_plotter,
                                      trg_module=trg_module,
                                      circuits=circuits,
                                      weights=exp_weights,
                                      action_delay=10,
                                      max_depth=7,
                                      speed=SPEED,
                                      visualize=True,
                                      number=2,
                                      number2=6,
                                      visualize_action=True)
    agent = mod.Brain(exp_module=exp_module,
                      circuits=circuits,
                      pcnn2D=pcnn2D)

    # --- agent & env
    env = ev.make_room(name=ROOM, thickness=4.,
                       bounds=BOUNDS,
                       visualize=True)
    pcnn2D_plotter.add_element(element=env)

    env = ev.AgentBody(room=env,
                       position=sim_settings["init_position"])
    reward_obj = ev.RewardObj(position=trg_position,
                              radius=trg_radius,
                              fetching=sim_settings["rw_fetching"],
                              behaviour=sim_settings["rw_behaviour"],
                              bounds=BOUNDS)
    logger(reward_obj)
    pcnn2D_plotter.add_element(element=reward_obj)

    velocity = np.zeros(2)
    observation = {
        "position": env.position,
        "collision": False,
        "reward": 0.
    }

    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))

    reward_count = 0

    trajectory = [env.position.tolist()]
    for t in range(duration):

        if t % 1000 == 0:
            write_configs(num_figs=8,
                          circuits=circuits,
                          t=t,
                          trg_module=trg_module,
                          observation=observation,
                          other=other_info)

        # --- env
        position, collision, truncated = env(
                    velocity=velocity)
        reward = reward_obj(position=position)
        trajectory += [position.tolist()]
        reward_count += reward

        # --- observation
        observation["position"] = position
        observation["collision"] = collision
        observation["reward"] = reward

        # --- agent
        velocity = agent(observation=observation)

        # --- exit
        if truncated:
            plot_update(fig=fig, ax=ax,
                        agent=agent,
                        env=env, trajectory=trajectory,
                        t=t, velocity=velocity)
            logger.warning(f"truncated at t={t}")
            input()
            break

        # if reward_count > 1:
        #     logger("Reward count reached!", level=0)
        #     break

        # --- plot
        if t % PLOT_INTERVAL == 0:
            if not plot:
                agent.render(use_trajectory=True,
                             alpha_nodes=0.1,
                             alpha_edges=0.06)

            if plot:
                plot_update(fig=fig, ax=ax,
                            agent=agent,
                            env=env,
                            reward_obj=reward_obj,
                            trajectory=trajectory,
                            t=t, velocity=velocity)

    return agent


def plot_update(fig, ax, agent, env, reward_obj,
                trajectory, t, velocity):

    ax.clear()

    #
    env.render(ax=ax)
    reward_obj.render(ax=ax)
    # agent.render(use_trajectory=True,
    #              alpha_nodes=0.2,
    #              alpha_edges=0.2)

    #
    # ax.set_title(f"t={t} | v={np.around(velocity, 3)} " + \
    #     f"p={np.around(env.position, 3)}")
    fig.canvas.draw()

    plt.pause(0.001)


def run_analysis(N: int=5, duration: int=1000):

    mod.edit_logger(is_debugging=False)
    edit_logger(level=0)

    # settings
    rw_position = np.random.uniform(0.1, 0.9, 2)
    sim_settings["rw_position"] = rw_position
    sim_settings["max_duration"] = duration
    sim_settings["rendering"] = False
    agent_settings["N"] = 200

    # object
    simulator = Simulation(sim_settings=sim_settings,
                           agent_settings=agent_settings)

    # utc.analysis_0(simulator=simulator)
    # utc.analysis_I(N=N, simulator=simulator)
    utc.analysis_II(N=N, simulator=simulator)


def simple_run(sim_settings: dict,
               agent_settings: dict,
               model_params: dict,
               render: bool=True):

    """
    plot the start and end positions of the trajectory,
    GOAL: highlight how the agent stays within the reward area
    """

    # --- make simulator
    sim_settings["rendering"] = False
    simulator = Simulation(sim_settings=sim_settings,
                           agent_settings=agent_settings,
                           model_params=model_params)

    # --- RUN
    done = False
    simulator.reset()
    duration = simulator.max_duration
    for _ in tqdm(range(duration)):
        simulator.update()

    # --- PLOT
    if not render:
        return simulator

    fig, ax = plt.subplots(figsize=(6, 6))

    trajectory = np.array(simulator.get_trajectory())

    # reward
    rw_position, rw_radius = simulator.get_reward_info()

    # reward area
    ax.add_patch(plt.Circle(rw_position, rw_radius,
                            color="green", alpha=0.1))

    # trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1],
               lw=0.5, alpha=0.7)

    # start and end
    ax.scatter(trajectory[0, 0], trajectory[0, 1],
               marker="o", color="white", s=40,
               edgecolor="red")
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
               marker="o", color="red", s=40,
               edgecolor="red")


    # ax[i].axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def loaded_run(idx: int=None,
             render: bool=True):

    """
    load the settings from a recorded evolutionary search
    """

    sim_settings, agent_settings, model_params = utc.load_model_settings(idx=idx,
                                                                         verbose=True)

    # --- make simulator
    sim_settings["rendering"] = False
    simulator = Simulation(sim_settings=sim_settings,
                           agent_settings=agent_settings,
                           model_params=model_params)

    # --- RUN
    done = False
    fig, ax = plt.subplots(figsize=(6, 6))

    while True:

        simulator.reset()
        # duration = simulator.max_duration
        duration = 1000
        for _ in tqdm(range(duration)):
            simulator.update()

        # --- PLOT
        if not render:
            return simulator

        trajectory = np.array(simulator.get_trajectory())

        # reward
        rw_position, rw_radius = simulator.get_reward_info()

        ax.clear()

        # reward area
        ax.add_patch(plt.Circle(rw_position, rw_radius,
                                color="green", alpha=0.1))

        # trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                   lw=0.5, alpha=0.7)

        # start and end
        ax.scatter(trajectory[0, 0], trajectory[0, 1],
                   marker="o", color="white", s=40,
                   edgecolor="red")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                   marker="o", color="red", s=40,
                   edgecolor="red")


        # ax[i].axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(0., 1.)
        ax.set_ylim(0., 1.)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.pause(0.001)

    plt.show()


def loop_main(sim_settings: dict,
              agent_settings: dict,
              load: bool=False,
              renew: bool=False):

    if load:
        sim_settings, agent_settings, model_params = utc.load_model_settings(idx=0,
                                                                       verbose=True)

        if sim_settings is None:
            load = False
    else:
        model_params = {
            "threshold": 0.5,
            "rep_threshold": 0.5,
            "w1": -1.,  # bnd
            "w2": 0.2,  # dpos
            "w3": -0.5,  # pop
            "w4": 1.0,  # trg
            "w5": 0.3,  # smooth
        }

    count = 0
    while True:

        logger(f"[round {count}]", level=0)

        if not load:
            sim_settings["seed"] = np.random.randint(0, 1000)
            sim_settings["init_position"] = np.random.uniform(0.1, 0.9, 2)
            sim_settings["rw_fetching"] = "deterministic"
            sim_settings["rw_position"] = np.random.uniform(0.1, 0.9, 2)
            sim_settings["rw_radius"] = 0.1
            sim_settings["plot_interval"] = 5
            sim_settings["speed"] = 0.04

            agent_settings["max_depth"] = 20
            agent_settings["exp_weights"] = np.array([-1., 0.2, -1., 0.2, 0.4]),

        main(sim_settings=sim_settings,
             agent_settings=agent_settings,
             model_params=model_params)

        count += 1

#

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=str, default="main")
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--seed", type=int, default=-1,
                        help="random seed: -1 for random seed.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--load", action="store_true")

    args = parser.parse_args()

    #
    if args.seed > 0:
        logger.debug(f"seed: {args.seed}")
        np.random.seed(args.seed)

    # mod.edit_logger(is_debugging=False)
    # edit_logger(level=0)

    # run
    if args.main == "main":
        sim_settings["seed"] = args.seed
        sim_settings["max_duration"] = args.duration
        sim_settings["rendering"] = not args.plot
        agent_settings["N"] = args.N
        main()

    elif args.main == "loop":
        sim_settings["seed"] = args.seed
        sim_settings["max_duration"] = args.duration
        sim_settings["rendering"] = not args.plot
        agent_settings["N"] = args.N
        loop_main(sim_settings=sim_settings,
                  agent_settings=agent_settings,
                  load=args.load)

    elif args.main == "analysis":
        run_analysis(N=args.N,
                     duration=args.duration)

    elif args.main == "simple":
        sim_settings["seed"] = args.seed
        sim_settings["max_duration"] = args.duration
        agent_settings["N"] = args.N
        simple_run(sim_settings=sim_settings,
                   agent_settings=agent_settings,
                   model_params=model_params,
                   render=True)

    elif args.main == "loaded":
        loaded_run(idx=args.N,
                   render=True,
                   load=args.load)


