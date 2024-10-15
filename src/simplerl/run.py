import numpy as np
import environments as ev
from stable_baselines3 import PPO, A2C, TD3, DDPG, DQN

import pcnn_wrapper as pw

import sys, os
exp_path = os.getcwd().split("PCNN")[0] + "PCNN"
sys.path.append(exp_path)
import src.spatial_cells as sc




logger = ev.logger



def demo_room_run(args):

    # --- make objects
    room = ev.make_room(name=args.room)
    logger(room)

    agent = ev.Agent(room=room,
                     speed=0.015,
                     color="red")
    logger(agent)

    # --- run
    ev.run(room=room, agent=agent, duration=args.duration,
           fpi=10, fpause=0.05)


def dummy_run(args):

    # --- instantiate the environment ---
    # PCNN model
    pcnn_model = pw.make_default_model()
    if not args.new_pc:
        pcnn_model.load_params(path=f"{path}/pcnn_param.json")

    # new Policy
    # policy = pw.Policy(isgoal=True,
    #                    speed=0.0075,
    #                    min_trg_distance=0.01)

    policy = pw.MinimalPolicy(num_lambdas=3,
                              speed=0.002,
                              speed_max=0.0075)
    pcnn_model.set_policy(policy=policy)

    # room
    room = ev.make_room(name=args.room)
    logger(room)

    # body
    body = ev.AgentBody(room=room,
                        position=np.array([0.6, 0.3]),
                        radius=0.03,
                        color="red")
    logger(body)

    # environment
    env = ev.CampoVerdeEnv(model=pcnn_model,
                           agent_body=body,
                           room=room,
                           bounds=(0, 1, 0, 1),
                           min_obs_activity=0.01,
                           isreward=True,
                           objective="target",
                           max_steps=args.duration,
                           max_wall_hits=np.inf,
                           is_policy_first=True)
    logger(env)
    logger(f"objectives: `{env.objective}` [rewarded={env.isreward}]")

    agent = ev.DummyAgent(action_size=env.action_size)
    logger("[dummy agent loaded]")
    logger(agent)

    # --- run an episode ---
    _ = ev.run_episode(agent=agent,
                       env=env,
                       render=True,
                       tper=2,
                       render_kind=1)


def train_agent(args):


    # --- instantiate the environment ---
    # PCNN model
    pcnn_model = pw.make_default_model()
    if not args.new_pc:
        pcnn_model.load_params(path="{}".format(ev.PCNN_PATH + "/pcnn_param.json"))

    # new Policy
    if args.policy == "weighted":
        policy = pw.MinimalPolicy(num_lambdas=3,
                                  speed=0.0007,
                                  min_trg_distance=0.01)
    elif args.policy == "forward":
        policy = pw.VelocityPolicy(speed=0.0007)
    pcnn_model.set_policy(policy=policy)

    # room
    room = ev.make_room(name=args.room)
    logger(room)

    # body
    body = ev.AgentBody(room=room,
                        position=np.array([0.6, 0.3]),
                        radius=0.03,
                        color="red")
    logger(body)

    # spatial cells
    gc = [
        sc.GridCellLayer(nx=5, ny=5, sigma=0.05),
        sc.GridCellLayer(nx=5, ny=5, sigma=0.05,
                      offset=np.array([0., 0.13])),
        sc.GridCellLayer(nx=4, ny=4, sigma=0.1)
    ]

    spatial_layers = sc.LayerGroup(layers=gc)

    logger(f"duration: {args.duration}")
    logger(f"epochs: {args.epochs}")

    # environment
    env = ev.CampoVerdeEnv(model=pcnn_model,
                           agent_body=body,
                           room=room,
                           spatial_layers=spatial_layers,
                           bounds=(0, 1, 0, 1),
                           min_obs_activity=0.01,
                           isreward=True,
                           objective=args.objective,
                           max_steps=args.duration,
                           max_wall_hits=np.inf,
                           is_policy_first=True,
                           random_start=True,
                           random_target=True,
                           inner_step_duration=args.dilation,
                           policy_type=args.policy)
    logger(env)
    logger(f"{env.description=}")

    logger(f"{ev.OBS_SIZE=}")

    # --- train the agent ---
    # Train the agent for 10,000 timesteps
    if not args.load and args.agent != "dummy":
        # Create the PPO agent
        if args.agent == "ppo":
            agent = PPO("MlpPolicy", env, verbose=1)
        elif args.agent == "a2c":
            agent = A2C("MlpPolicy", env, verbose=1)
        elif args.agent == "td3":
            agent = TD3("MlpPolicy", env, verbose=1)
        elif args.agent == "ddpg":
            agent = DDPG("MlpPolicy", env, verbose=1)
        elif args.agent == "dqn":
            agent = DQN("MlpPolicy", env, verbose=1)
        else:
            raise ValueError("agent not found")

        logger(agent)

        if args.train:
            logger("training started [save={}]".format(args.save))
            agent.learn(total_timesteps=args.epochs,
                        log_interval=1_000,
                        progress_bar=True)
            logger.info("[{} training done]".format(args.agent))

            # save the model
            if args.save:
                filename = "{}/{}_{}".format(ev.SAVEPATH, args.agent, env.objective)
                agent.save("{}".format(filename))
                logger.info("[{} model saved]".format(filename))

    # custom dummy agent
    elif args.agent == "dummy":
        agent = DummyAgent(action_size=env.action_size)
        logger("[dummy agent loaded]")
        logger(agent)

    # load the model
    else:
        filename = "{}/{}_{}".format(ev.SAVEPATH, args.agent, env.objective)
        if args.agent == "ppo":
            agent = PPO.load("{}".format(filename))
        elif args.agent == "a2c":
            agent = A2C.load("{}".format(filename))
        elif args.agent == "td3":
            agent = TD3.load("{}".format(filename))
        elif args.agent == "ddpg":
            agent = DDPG.load("{}".format(filename))
        elif args.agent == "dqn":
            agent = DQN.load("{}".format(filename))
        else:
            raise ValueError("agent not found")

        logger.info("[{} model loaded]".format(args.agent))

    # --- run an episode ---
    if args.test:
        logger("testing agent...")
        _ = ev.run_episode(agent=agent,
                           env=env,
                           render=True,
                           render_kind=1,
                           tper=2)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=100_000,
                        help="number of epochs")
    parser.add_argument("--duration", type=int, default=1_000,
                        help="episode duration in steps")
    parser.add_argument("--dilation", type=int, default=10,
                        help="timesteps per episode step")
    parser.add_argument("--agent", type=str, default="ppo",
                        help="agent type [ppo, a2c, td3, dummy]")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--room", type=str, default="square",
                        help="type of room")
    parser.add_argument("--policy", type=str, default="weighted",
                        help="type of policy [weighted, forward]")
    parser.add_argument("--run", type=str, default="demo",
                        help="type of run: 'demo', 'dummy', 'train'")
    parser.add_argument("--objective", type=str, default="target",
                        help="type of objective: 'target', 'explore'")
    parser.add_argument("--new_pc", action="store_true",
                        help="use an untrained PCNN model")
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)


    # --- type of run
    if args.run == "demo":
        demo_room_run(args=args)

    elif args.run == "dummy":
        dummy_run(args=args)

    elif args.run == "train":
        train_agent(args=args)

    else:
        raise NameError(f"'{args_run}' is not a valid run")



