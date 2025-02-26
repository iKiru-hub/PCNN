import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym



def play_cartpole():
    env = gym.make("CartPole-v1", render_mode="human")
    env.reset()
    for _ in range(1000):
        env.render()
        observation, reward, done, terminated, info = env.step(env.action_space.sample())
        print(observation)

        if done:
            env.reset()
        if terminated:
            print("Terminated")
            break
    env.close()



class CartPoler:

    def __init__(self, brain: object, renderer: object=None):

        self.position = np.zeros(2)
        self.brain = brain
        self.renderer = renderer

    def __call__(self, velocity: np.ndarray, reward: float,
                 collision: float, goal_flag: bool) -> int:

        self.position += velocity
        out_velocity = self.brain(velocity, reward, collision, goal_flag)

        return int(out_velocity[0] < 0)

    def render(self):
        if self.renderer is not None:
            self.renderer.render()

    def reset(self, new_position: np.ndarray):
        self.brain.reset()
        self.position = new_position



def run_cartpole(brain: object,
                 renderer: object,
                 duration: int,
                 t_goal: int=10,
                 t_rendering: int=10,
                 record_flag: bool=False,
                 verbose_min: bool=True):

    # ===| setup |===

    clock = pygame.time.Clock()
    last_position = np.zeros(2)

    # objects
    agent = CartPoler(brain=brain)
    env = gym.make("CartPole-v1", render_mode="human")

    # observation: [position, velocity, angle, angular velocity]
    obs = env.reset()

    # [obs, reward, done, done, terminated]
    env_out = (obs, 0., False, False, {})

    # starting position: [position, angle]
    prev_position = [obs[0], obs[2]]
    agent.reset(new_position=prev_position)

    # init
    record = {"activity_fine": [],
              "activity_coarse": [],
              "scores": [],
              "trajectory": []}

    # ===| main loop |===
    score = 0
    eps_count = 0
    for t in tqdm(range(duration), desc="Game", leave=False,
                  disable=not verbose_min):

        # brain step
        action = agent([obs[0], [obs[2]],
                       reward, 0.0, t>=t_goal)

        # env step
        obs, reward, done, terminated, info = env.step(action)
        score += reward

        # -check: render
        if t > t_rendering:
            env.render()

        # -check: record
        if record_flag:
            record["activity_fine"] += [brain.get_representation_fine()]
            record["activity_coarse"] += [brain.get_representation_coarse()]
            record["trajectory"] += [env.position]

        # -check: done
        if done:
            obs = env.reset()
            agent.reset(new_position=[obs[0], obs[2]])
            record["scores"] += score
            score = 0

        # -check: terminated
        if terminated:
            break

    record["num_eps"] = eps_count

    return record

if __name__ == "__main__":
    play_cartpole()

