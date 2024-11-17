import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import argparse

from tools.utils import clf, tqdm_enumerate, logger
import inputools.Trajectory as it
import pcnn_core as pcore
import mod_core as mod
import utils_core as utc
import envs_core as ev

import os, sys, json
base_path = os.getcwd().split("PCNN")[0]+"PCNN/src/"
sys.path.append(base_path)
import utils
# import simplerl.environments as ev

# import matplotlib
# matplotlib.use("TkAgg")


import pclib


""" INITIALIZATION  """

CONFIGPATH = "dashboard/media/configs.json"

logger = utc.setup_logger(name="RUN",
                          level=-1,
                          is_debugging=False,
                          is_warning=False)


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
        "velocity": np.around(observation["velocity"], 3).tolist(),
        "trg_pos": np.around(trg_module.output["trg_position"], 3).tolist(),
        "trg_velocity": np.around(trg_module.output["velocity"], 3).tolist()
    }

    info = info | other

    with open(CONFIGPATH, 'w') as f:
        json.dump(info, f)

    logger(f"configs written to {CONFIGPATH}")
    logger(f"{info}")

BOUNDS = np.array([0., 1., 0., 1.])

""" CLASSES """


class Simulation:

    """
    very similar to main(), but it is meant to be run
    with the streamlit dashboard
    """

    def __init__(self, N: int=80, seed: int=None,
                 plot_interval: int=1,
                 rendering: bool=True,
                 max_duration: int=None,
                 init_position: np.ndarray=None,
                 rw_position: np.ndarray=None,
                 rw_radius: float=0.2,):

        # --- SETTINGS
        SPEED = 0.03
        init_position = np.array([0.8, 0.2]) if init_position is None else init_position
        rw_position = np.array([0.8, 0.8]) if rw_position is None else rw_position

        self.plot_interval = plot_interval
        self.rendering = rendering
        self.max_duration = max_duration
        self.info = {}
        self.init_config = {"N": N,
                            "seed": seed,
                            "plot_interval": plot_interval,
                            "rendering": rendering,
                            "max_duration": max_duration,
                            "init_position": init_position,
                            "rw_position": rw_position,
                            "rw_radius": rw_radius}

        # --- PCNN
        N = 80
        Nj = 13**2

        logger(f"{N=}")
        logger(f"{Nj=}")

        sigma = 0.05
        # self.bounds = np.array([0., 1., 0., 1.])
        self.bounds = BOUNDS
        xfilter = pclib.PCLayer(int(np.sqrt(Nj)), sigma, self.bounds)

        # definition
        pcnn2D = pclib.PCNN(N=N, Nj=Nj, gain=3., offset=1.5,
                          clip_min=0.09, threshold=0.5,
                          rep_threshold=0.5, rec_threshold=0.01,
                          num_neighbors=8, trace_tau=0.1,
                          xfilter=xfilter, name="2D")

        model_plotter = pcore.PlotPCNN(model=pcnn2D,
                                      bounds=self.bounds,
                                      visualize=rendering)

        # --- CIRCUITS
        circuits_dict = {"Bnd": mod.BoundaryMod(N=N,
                                                threshold=0.02,
                                                visualize=rendering,
                                                score_weight=5.),
                         "DA": mod.Dopamine(N=N,
                                            visualize=rendering,
                                            score_weight=1.),
                         "dPos": mod.PositionTrace(visualize=False,
                                                   score_weight=2.),
                         "Pop": mod.PopulationProgMax(N=N,
                                                      visualize=False,
                                                      number=None),
                         "Ftg": mod.FatigueMod()}

        for _, circuit in circuits_dict.items():
            logger.debug(f"{circuit} keys: {circuit.input_key}")

        circuits = mod.Circuits(circuits_dict=circuits_dict,
                                visualize=rendering)

        # --- MODULES
        trg_module = mod.TargetModule(pcnn=pcnn2D,
                                      circuits=circuits,
                                      speed=SPEED,
                                      threshold=0.01,
                                      visualize=rendering)

        # [ bnd, dpos, pop, trg ]
        weights = np.array([-1., 0., -2., 1., 1.5])
        exp_module = mod.ExperienceModule(pcnn=pcnn2D,
                                          pcnn_plotter=model_plotter,
                                          trg_module=trg_module,
                                          circuits=circuits,
                                          weights=weights,
                                          action_delay=10,
                                          speed=SPEED,
                                          visualize=False,
                                          visualize_action=rendering)
        self.agent = mod.Brain(exp_module=exp_module,
                               pcnn2D=pcnn2D,
                               circuits=circuits)

        # --- agent & env
        self.env = ev.make_room(name="square", thickness=4.,
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
        self.agent.record["trajectory"] += [self.env.position.tolist()]
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
        position, collision, truncated = self.env(velocity=self.velocity)
        reward = self.reward_obj(position=position)

        # --- observation
        self.observation["u"] = self.agent.exp_module.fwd_pcnn(
            x=position.reshape(-1, 1)).flatten()
        self.observation["position"] = position
        self.observation["velocity"] = self.velocity
        self.observation["collision"] = collision
        self.observation["reward"] = reward
        self.observation["delta_update"] = self.agent.observation_int['delta_update']
        self.observation["action_idx"] = self.agent.observation_int['action_idx']

        if collision:
            logger.debug(f">>> collision at t={self.t}")

        if reward > 0:
            logger.debug(f">>> reward at t={self.t}")

        # --- agent
        self.velocity = self.agent(observation=self.observation)

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
        return self.init_config["rw_position"], self.init_config["rw_radius"]

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
        self.init_config["rw_position"] = rw_position

    def reset(self, seed: int=None, init_position: np.ndarray=None):

        self.init_config["seed"] = seed
        self.init_config["init_position"] = init_position

        self.__init__(**self.init_config)
        logger(f"%% reset [seed={seed}] %%")


def main(args):

    """
    meant to be run standalone
    """

    # --- settings
    duration = args.duration
    SPEED = 0.07
    PLOT_INTERVAL = 5
    other_info = {}

    # --- brain
    N = args.N
    Nj = 13**2

    logger(f"{duration}")
    logger(f"{N=}")
    logger(f"{Nj=}")

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.17,
        "beta": 35.0,
        "clip_min": 0.005,
        "threshold": 0.3,
        "rep_threshold": 0.8,
        "rec_threshold": 0.7,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "name": "PCNN"
    }

    # pc filter
    # pclayer = pcnn.PClayer(n=13, sigma=0.01)
    # logger.debug(f"{pclayer=}")
    # params["xfilter"] = pclayer
    # model = pcnn.PCNN(**params)

    # ---
    sigma = 0.05
    # bounds = np.array([0., 1., 0., 1.])
    bounds = BOUNDS
    xfilter = pclib.PCLayer(int(np.sqrt(Nj)), sigma, bounds)

    # definition
    pcnn2D = pclib.PCNN(N=N, Nj=Nj, gain=3., offset=1.5,
                      clip_min=0.09, threshold=0.5,
                      rep_threshold=0.5, rec_threshold=0.01,
                      num_neighbors=8, trace_tau=0.1,
                      xfilter=xfilter, name="2D")
    # ---

    # pcnn
    model_plotter = pcore.PlotPCNN(model=pcnn2D,
                                  bounds=bounds,
                                  visualize=True,
                                  number=0)

    # --- circuits
    circuits_dict = {"Bnd": mod.BoundaryMod(N=N,
                                            threshold=0.02,
                                            visualize=True,
                                            score_weight=5.,
                                        number=5),
                     "DA": mod.Dopamine(N=N,
                                        visualize=True,
                                        score_weight=1.,
                                        number=4),
                     "dPos": mod.PositionTrace(visualize=False,
                                               score_weight=2.),
                     "Pop": mod.PopulationProgMax(N=N,
                                                  visualize=False,
                                                  number=None),
                     "Ftg": mod.FatigueMod()}

    for _, circuit in circuits_dict.items():
        logger.debug(f"{circuit} keys: {circuit.input_key}")

    # object
    # modulators = mod.Modulators(modulators_dict=modulators_dict,
    #                             visualize=True,
    #                             number=3)

    circuits = mod.Circuits(circuits_dict=circuits_dict,
                            visualize=True,
                            number=3)

    # --- modules

    trg_module = mod.TargetModule(pcnn=pcnn2D,
                                  circuits=circuits,
                                  speed=SPEED,
                                  threshold=0.5,
                                  visualize=True,
                                  number=1)
    other_info["Trg_thr"] = trg_module.threshold


    # weight_policy = mod.WeightsPolicy(circuits_dict=circuits_dict,
    #                                    trg_module=trg_module,
    #                                    visualize=True,
    #                                    number=6)
    # logger(f"{weight_policy}")

    # exp_module = mod.ExperienceModule(pcnn=model,
    #                                   pcnn_plotter=model_plotter,
    #                                   trg_module=trg_module,
    #                                   weight_policy=weight_policy,
    #                                   circuits=circuits,
    #                                   speed=SPEED,
    #                                   max_depth=20,
    #                                   visualize=False,
    #                                   number=2,
    #                                   visualize_action=True)

    # [ bnd, dpos, pop, trg, smooth ]
    weights = np.array([-2., 0.0, -2., 0.9, 0.1])
    exp_module = mod.ExperienceModule(pcnn=pcnn2D,
                                       pcnn_plotter=model_plotter,
                                       trg_module=trg_module,
                                       circuits=circuits,
                                       weights=weights,
                                       action_delay=10,
                                       max_depth=20,
                                       speed=SPEED,
                                       visualize=False,
                                       number=2,
                                       visualize_action=True)
    agent = mod.Brain(exp_module=exp_module,
                      circuits=circuits,
                      pcnn2D=pcnn2D,
                      number=None)

    # --- agent & env
    env = ev.make_room(name="square", thickness=4.,
                       bounds=BOUNDS,
                       visualize=True)
    env = ev.AgentBody(room=env,
                       position=np.array([0.8, 0.2]))
    reward_obj = ev.RewardObj(position=np.array([0.8, 0.8]),
                       radius=0.15)
    velocity = np.zeros(2)
    observation = {
        "u": np.zeros(N),
        "position": env.position,
        "velocity": velocity,
        "delta_update": 0.,
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
        position, collision, truncated = env(velocity=velocity)
        reward = reward_obj(position=position)
        trajectory += [position.tolist()]

        # --- observation
        # observation["u"] = agent.exp_module.fwd_pcnn(
        #     x=position.reshape(-1, 1)).flatten()
        observation["position"] = position
        observation["velocity"] = velocity
        observation["collision"] = collision
        observation["reward"] = reward
        observation["delta_update"] = agent.state['delta_update']
        observation["action_idx"] = agent.state['action_idx']

        # observation["delta_update"] = agent.observation_int['delta_update']
        # observation["action_idx"] = agent.observation_int['action_idx']

        # if collision:
        #     logger.debug(f">>> collision at t={t}")

        # if reward > 0:
        #     logger.debug(f">>> reward at t={t}")

        # --- agent
        velocity = agent(observation=observation)
        # agent.routines(wall_vectors=env._room.wall_vectors)

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
                             alpha_nodes=0.2,
                             alpha_edges=0.2)

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

    rw_position = np.random.uniform(0.1, 0.9, 2)

    simulator = Simulation(max_duration=duration,
                           rendering=False,
                           rw_position=rw_position,
                           plot_interval=1)

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



