import numpy as np
import matplotlib.pyplot as plt
import argparse, json

import mod_core as mod
import utils_core as utc
import envs_core as ev

import pclib


""" INITIALIZATION  """


CONFIGPATH = "dashboard/cache/configs.json"

logger = utc.setup_logger(name="RUN",
                          level=1,
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


BOUNDS = np.array([0., 1., 0., 1.])


""" CLASSES """

sim_settings = {
    "speed": 0.03,
    "init_position": np.array([0.8, 0.2]),
    "rw_position": np.array([0.5, 0.8]),
    "rw_radius": 0.1,
    "plot_interval": 1,
    "rendering": True,
    "room": "square",
    "max_duration": None,
    "seed": None
}

agent_settings = {
    "N": 80,
    "Nj": 14**2,
    "exp_weights": np.array([-0.2, 0.2, -1., 0.2, 0.4]),
    "max_depth": 7
}


class Simulation:

    """
    very similar to main(), but it is meant to be run
    with the streamlit dashboard
    """

    def __init__(self, N: int=80,
                 sim_settings: dict = sim_settings,
                 agent_settings: dict = agent_settings):

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
        exp_weights = agent_settings["exp_weights"]

        logger(f"{N=}")
        logger(f"{Nj=}")

        # ---
        sigma = 0.05
        self.bounds = BOUNDS
        xfilter = pclib.PCLayer(int(np.sqrt(Nj)),
                                sigma, self.bounds)

        # definition
        pcnn2D = pclib.PCNN(N=N, Nj=Nj, gain=3., offset=1.5,
                          clip_min=0.09, threshold=0.5,
                          rep_threshold=0.5, rec_threshold=0.01,
                          num_neighbors=8, trace_tau=0.1,
                          xfilter=xfilter, name="2D")

        pcnn2D_plotter = utc.PlotPCNN(model=pcnn2D,
                                      bounds=self.bounds,
                                      visualize=rendering)

        # --- CIRCUITS
        circuits_dict = {"Bnd": mod.BoundaryMod(N=N,
                                                threshold=0.2,
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
                            action_delay=10,
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
                                       radius=rw_radius)
        self.velocity = np.zeros(2)
        self.observation = {
            "u": np.zeros(N),
            "position": self.env.position,
            "velocity": self.velocity,
            "delta_update": 0.,
            "collision": False,
            "reward": 0.
        }
        self.output = {
            "u": np.zeros(N),
            "velocity": self.velocity,
            "delta_update": 0.,
            "action_idx": None,
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

        self.__init__(sim_settings=self.sim_settings,
                      agent_settings=self.agent_settings)
        logger(f"%% reset [seed={seed}] %%")


def main(args):

    """
    meant to be run standalone
    """

    # --- settings
    duration = args.duration
    trg_position = np.array([0.5, 0.8])
    trg_radius = 0.1
    SPEED = 0.03
    PLOT_INTERVAL = 5
    ROOM = "flat"

    other_info = {}
    logger(f"room: {ROOM}")
    logger(f"plot_interval: {PLOT_INTERVAL}")
    logger(f"{duration}")

    # --- brain
    N = args.N
    Nj = 14**2
    sigma = 0.03
    bounds = BOUNDS

    logger(f"{N=}")
    logger(f"{Nj=}")

    xfilter = pclib.PCLayer(int(np.sqrt(Nj)), sigma, bounds)

    # definition
    pcnn2D = pclib.PCNN(N=N, Nj=Nj, gain=3., offset=1.,
                      clip_min=0.09, threshold=0.3,
                      rep_threshold=0.9, rec_threshold=0.01,
                      num_neighbors=8, trace_tau=0.1,
                      xfilter=xfilter, name="2D")

    # plotter
    pcnn2D_plotter = utc.PlotPCNN(model=pcnn2D,
                                  bounds=bounds,
                                  visualize=True,
                                  number=0)

    # --- circuits
    circuits_dict = {"Bnd": mod.BoundaryMod(N=N,
                                            threshold=0.2,
                                            eta=0.5,
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

    # for _, circuit in circuits_dict.items():
    #     logger.debug(f"{circuit} keys: {circuit.input_key}")

    # object
    circuits = mod.Circuits(circuits_dict=circuits_dict,
                            visualize=True,
                            number=3)

    # --- modules

    trg_module = mod.TargetModule(pcnn=pcnn2D,
                                  circuits=circuits,
                                  speed=SPEED,
                                  threshold=0.2,
                                  visualize=True,
                                  number=1)
    other_info["Trg_thr"] = trg_module.threshold

    # [ bnd, dpos, pop, trg, smooth ]
    weights = np.array([-0.2, 0.2, -1., 0.2, 0.4])
    exp_module = mod.ExperienceModule(pcnn=pcnn2D,
                                      pcnn_plotter=pcnn2D_plotter,
                                      trg_module=trg_module,
                                      circuits=circuits,
                                      weights=weights,
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
                       position=np.array([0.8, 0.2]))
    reward_obj = ev.RewardObj(position=trg_position,
                              radius=trg_radius)
    pcnn2D_plotter.add_element(element=reward_obj)

    velocity = np.zeros(2)
    observation = {
        # "u": np.zeros(N),
        "position": env.position,
        # "velocity": velocity,
        # "delta_update": 0.,
        "collision": False,
        "reward": 0.
    }
    output = {
        "u": np.zeros(N),
        "velocity": velocity,
        "delta_update": 0.,
        "action_idx": None,
    }

    if args.plot:
        fig, ax = plt.subplots(figsize=(5, 5))

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

        # --- plot
        if t % PLOT_INTERVAL == 0:
            if not args.plot:
                agent.render(use_trajectory=True,
                             alpha_nodes=0.1,
                             alpha_edges=0.06)

            if args.plot:
                plot_update(fig=fig, ax=ax,
                            agent=agent,
                            env=env,
                            reward_obj=reward_obj,
                            trajectory=trajectory,
                            t=t, velocity=velocity)


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
    agent_settings["N"] = N

    # object
    simulator = Simulation(sim_settings=sim_settings,
                           agent_settings=agent_settings)

    # utc.analysis_I(N=N, simulator=simulator)
    utc.analysis_II(N=N, simulator=simulator)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=str, default="main")
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--seed", type=int, default=-1,
                        help="random seed: -1 for random seed.")
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    #
    if args.seed > 0:
        # args.seed = np.random.randint(0, 10000)
        logger.debug(f"seed: {args.seed}")

        np.random.seed(args.seed)
    # mod.set_seed(seed=args.seed)

    # run
    if args.main == "main":
        main(args)

    elif args.main == "analysis":
        run_analysis(N=args.N,
                     duration=args.duration)



