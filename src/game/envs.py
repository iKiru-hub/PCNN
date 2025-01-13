import pygame
import numpy as np
from typing import Tuple, List
from matplotlib.pyplot import pause

import os, sys
sys.path.append(os.path.join(os.getcwd().split("PCNN")[0], "PCNN/src"))
from game.constants import *
import game.objects as objects
from game.objects import logger

# try:
#     from constants import *
#     import objects
# except ImportError:
#     from game.constants import *
#     from game import objects




""" Environment components """


class Wall:

    def __init__(self, x: int, y: int,
                 width: int, height: int,
                 thickness: int = 5,
                 color: Tuple[int, int, int] = BLACK):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.thickness = thickness

        # Store edges for line segment collision detection
        self.edges = [
            # Top edge
            ((x, y), (x + width, y)),
            # Right edge
            ((x + width, y), (x + width, y + height)),
            # Bottom edge
            ((x, y + height), (x + width, y + height)),
            # Left edge
            ((x, y), (x, y + height))
        ]

    def line_segment_intersection(self, p1: Tuple[float, float],
                                p2: Tuple[float, float]) -> bool:
        """Check if a line segment intersects with any wall edge."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        for edge_start, edge_end in self.edges:
            if intersect(p1, p2, edge_start, edge_end):
                return True
        return False

    def render(self, screen: pygame.Surface):

        """
        plot the `self.rect` such as the y-axis
        is inverted
        """

        pygame.draw.rect(screen, self.color,
                         self.rect, self.thickness)



        # pygame.draw.rect(screen, self.color,
        #                 self.rect, self.thickness)


class Room:

    def __init__(self, walls_bounds: List[Wall],
                 walls_extra: List[Wall] = [],
                 bounds: Tuple[int, int, int, int] = None,
                 bounce_coeff: float = 1.0,
                 moving_wall: bool = False,
                 name: str = "Square.v0"):

        self.name = name
        self.walls_extra = walls_extra
        self.walls = walls_extra + walls_bounds

        if bounds is None:
            bounds = [OFFSET, OFFSET,
                      SCREEN_WIDTH-2*OFFSET,
                      SCREEN_HEIGHT-2*OFFSET]
        self.bounds = bounds
        self.bounce_coeff = bounce_coeff
        self.moving_wall = moving_wall
        self.t = 0

    def __str__(self):
        return f"Room({self.name})"

    def is_out_of_bounds(self, x: int, y: int) -> bool:
        """
        Check if a point is out of bounds.
        """
        return (x < self.bounds[0] or x > self.bounds[2] - 1 or
                y < self.bounds[1] or y > self.bounds[3] - 1)

    def __call__(self, x: float, y: float,
                 velocity: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Check if a point is out of bounds and update velocity accordingly.

        Parameters
        ----------
        x : float
            x-coordinate of the point.
        y : float
            y-coordinate of the point.
        velocity : np.ndarray
            Current velocity of the point.
        """
        # First check bounds
        if self.is_out_of_bounds(x, y):
            x, y = self._check_bounds(x, y)
            velocity *= -self.bounce_coeff
            return velocity, True, (x, y)

        # Then check wall collisions with continuous detection
        velocity, collision, new_pos = self.check_collision(
                            x, y, velocity)
        if collision:
            velocity *= self.bounce_coeff

        self.t += 1
        return velocity, collision, new_pos

    def _check_bounds(self, x: float, y: float) -> Tuple[float, float]:
        margin = self.safety_margin
        x = max(self.bounds[0] + margin,
               min(x, self.bounds[2] - margin))
        y = max(self.bounds[1] + margin,
               min(y, self.bounds[3] - margin))
        return x, y

    def check_collision(self, x: float, y: float,
                       velocity: np.ndarray) -> Tuple[np.ndarray, bool, Tuple[float, float]]:
        if np.all(velocity == 0):
            return velocity, False, (x, y)

        # Current position and next position
        p1 = (x, y)
        p2 = (x + velocity[0], y + velocity[1])

        collision = False
        new_pos = (x, y)

        for wall in self.walls:
            # Check if movement line intersects with any wall edge
            if wall.line_segment_intersection(p1, p2):
                # Determine which axis to reflect
                next_x = x + velocity[0]
                next_y = y + velocity[1]

                # Test x movement
                if wall.rect.collidepoint(next_x, y):
                    velocity[0] *= -1
                    collision = True

                # Test y movement
                if wall.rect.collidepoint(x, next_y):
                    velocity[1] *= -1
                    collision = True

                if collision:
                    # Move to safe position
                    new_x = x
                    new_y = y
                    if velocity[0] != 0:
                        new_x = x + (velocity[0] * 0.1)  # Move a small amount in the new direction
                    if velocity[1] != 0:
                        new_y = y + (velocity[1] * 0.1)
                    new_pos = (new_x, new_y)
                    logger.debug(f"Collision detected: pos={new_pos}," + \
                        f" vel={velocity}")
                    break

        return velocity, collision, new_pos

    def add_wall(self, wall: Wall):
        self.walls.append(wall)
        self.num_walls += 1
        self.wall_vectors = self._make_wall_vectors()

    def move_wall(self):

        if not self.moving_wall:
            return

        if len(self.walls_extra) == 0:
            return

        if len(self.walls) == 0:
            idx = 0
        else:
            idx = np.random.randint(len(self.walls_extra))

        new_wall = Wall(x=np.random.randint(0, SCREEN_WIDTH),
                        y=np.random.randint(0, SCREEN_HEIGHT),
                        width=np.random.randint(100, 200),
                        height=np.random.randint(100, 200),
                        thickness=5,
                        color=BLACK)
        self.walls_extra[idx] = new_wall
        self.walls[idx] = new_wall

    def render(self, screen: pygame.Surface):
        for wall in self.walls:
            wall.render(screen)


def make_room(name: str="square", thickness: float=5.,
              bounds: list=[0, 1, 0, 1],
              moving_wall: bool=False,
              visualize: bool=False):

    walls_bounds = [
        Wall(OFFSET, OFFSET,
             SCREEN_WIDTH-2*OFFSET+thickness, thickness),  # Top
        Wall(OFFSET, SCREEN_HEIGHT-OFFSET-thickness,
             SCREEN_WIDTH-2*OFFSET+thickness, thickness),  # Bottom
        Wall(OFFSET, OFFSET, thickness,
             SCREEN_HEIGHT-2*OFFSET),  # Left
        Wall(SCREEN_WIDTH-OFFSET, OFFSET,
             thickness, SCREEN_HEIGHT-2*OFFSET),  # Right
    ]

    walls_extra = []

    if name == "Square.v1":
        walls_extra += [
            Wall(OFFSET, SCREEN_HEIGHT//2, SCREEN_WIDTH//2, thickness)
        ]
    elif name == "Square.v2":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, SCREEN_HEIGHT//2-OFFSET,
                 thickness, SCREEN_HEIGHT//2),
            Wall(2*SCREEN_WIDTH//3, OFFSET,
                    thickness, SCREEN_HEIGHT//2)
        ]
    elif name == "Hole.v0":
        walls_extra += [
            Wall(SCREEN_WIDTH//2.5-OFFSET,
                 SCREEN_HEIGHT//2.5-OFFSET,
                 SCREEN_WIDTH//3, SCREEN_HEIGHT//3),
        ]
    elif name == "Flat.0000":
        walls_extra += [
            Wall(OFFSET, SCREEN_HEIGHT//3,
                 2*SCREEN_WIDTH//3-OFFSET, thickness),
        ]
    elif name == "Flat.0001":
        walls_extra += [
            Wall(OFFSET, 2*SCREEN_HEIGHT//3,
                 2*SCREEN_WIDTH//3-OFFSET, thickness),
        ]
    elif name == "Flat.0010":
        walls_extra += [
            Wall(OFFSET, SCREEN_HEIGHT//3,
                 2*SCREEN_WIDTH//3-OFFSET, thickness),
            Wall(OFFSET, 2*SCREEN_HEIGHT//3,
                 2*SCREEN_WIDTH//3-OFFSET, thickness),
        ]
    elif name == "Flat.0011":
        walls_extra += [
            Wall(OFFSET, SCREEN_HEIGHT//3,
                 2*SCREEN_WIDTH//3-OFFSET, thickness),
            Wall(SCREEN_WIDTH//3-OFFSET, 2*SCREEN_HEIGHT//3,
                 2*SCREEN_WIDTH//3-OFFSET, thickness),
        ]
    elif name == "Flat.0110":
        walls_extra += [
            Wall(SCREEN_WIDTH//3-OFFSET, SCREEN_HEIGHT//3,
                 2*SCREEN_WIDTH//3-OFFSET, thickness),
            Wall(OFFSET, 2*SCREEN_HEIGHT//3,
                 2*SCREEN_WIDTH//3-OFFSET, thickness),
        ]
    elif name == "Flat.1000":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
        ]
    elif name == "Flat.1001":
        walls_extra += [
            Wall(2*SCREEN_WIDTH//3, OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
        ]
    elif name == "Flat.1010":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
            Wall(2*SCREEN_WIDTH//3, OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
        ]
    elif name == "Flat.1011":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
            Wall(2*SCREEN_WIDTH//3, SCREEN_HEIGHT//3-OFFSET, 
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
        ]
    elif name == "Flat.1110":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, SCREEN_HEIGHT//3-OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
            Wall(2*SCREEN_WIDTH//3, OFFSET, 
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
        ]
    else:
        name = "Square.v0"

    room = Room(walls_bounds=walls_bounds,
                walls_extra=walls_extra,
                moving_wall=moving_wall,
                name=name)

    return room


ROOM_LIST = ["Square.v0", "Square.v1", "Square.v2",
             "Hole.v0", "Flat.0000", "Flat.0001",
             "Flat.0010", "Flat.0011", "Flat.0110",
             "Flat.1000", "Flat.1001", "Flat.1010",
             "Flat.1011", "Flat.1110"]


def render_room(name: str):

    room = make_room(name=name)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    room.render(screen)
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


""" Environment class """


class Environment:

    def __init__(self, room: Room, agent: objects.AgentBody,
                 reward_obj: objects.RewardObj,
                 rw_event: str = "nothing",
                 duration: int=np.inf,
                 scale: float=1.0,
                 verbose: bool=False,
                 visualize: bool=False):

        # Environment components
        self.room = room
        self.agent = agent
        self.reward_obj = reward_obj

        # Reward event
        self.rw_event = rw_event
        self.duration = duration
        self.scale = scale
        self.t = 0

        # rendering
        self.visualize = visualize
        self.verbose = verbose
        if visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Simple Pygame Game")

    def __str__(self):
        return f"Environment({self.room}, verbose={self.verbose})"

    def __call__(self, velocity: np.ndarray) -> \
        Tuple[float, np.ndarray, bool, bool]:

        # scale velocity
        velocity *= self.scale

        # Update agent position with improved collision handling
        _, collision = self.agent(velocity, self.room)

        # Check reward collisions
        reward = self.reward_obj(self.agent.rect.center)
        terminated = False
        if reward:
            terminated = self._reward_event()

        self.t += 1

        if self.verbose:
            if reward:
                logger.info(f"[t={self.t}] +reward")
            if collision:
                logger.info(f"[t={self.t}] -collision")

        # position = self.agent.position.copy() / self.scale

        return self.agent.position.copy(), velocity, reward, float(collision), float(self.t >= self.duration), terminated

    def _reward_event(self):

        """
        logic for when the reward is collected
        """

        if self.rw_event == "move reward":
            self.reward_obj.set_position()
            return False
        elif self.rw_event == "move agent":
            self.agent.set_position()
            self.agent.reset()
            return True
        elif self.rw_event == "move both":
            self.agent.set_position()
            self.reward_obj.set_position()
            return True
        elif self.rw_event == "nothing":
            return False
        else:
            raise ValueError(f"Unknown reward " + \
                f"event: {self.rw_event}")

    @property
    def position(self):
        return self.agent.position.copy()

    def render(self, **kargs):

        self.screen.fill(WHITE)

        # Render game objects
        self.room.render(self.screen)
        self.agent.render(self.screen)
        self.reward_obj.render(self.screen)

        # write text
        font = pygame.font.Font(None, 36)
        score_text = font.render(f't: {self.t}', True, BLACK)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()


def run_game(env: Environment,
             brain: object,
             pcnn_plotter: object = None,
             element: object = None,
             fps: int = 30,
             plotter_int: int = 100):

    clock = pygame.time.Clock()
    last_position = np.zeros(2)

    # [position, velocity, collision, reward, done, terminated]
    observation = [env.position, np.array([0., 0.]), 0., 0.,
                   False, False]
    # observation = {
    #     "position": env.position,
    #     "collision": 0.,
    #     "reward": 0.
    # }

    running = True
    while running:

        # Event handling
        if env.visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # step
        # velocity = brain(observation)# * SCREEN_WIDTH
        # obs = [(observation["position"]-last_position).astype(np.float32).reshape(-1, 1),
        #        observation["collision"],
        #        observation["reward"]]
        velocity = brain(observation[1],
                         observation[2],
                         observation[3],
                         observation[0])
        if not isinstance(velocity, np.ndarray):
            velocity = np.array(velocity)

        observation = env(velocity=velocity)

        # reset agent's brain
        if observation[4]:
            logger.info(">> Game reset <<")
            brain.reset(position=env.agent.position)

        # update observation
        # observation["position"] = next_observation[1]
        # observation["collision"] = next_observation[2]
        # observation["reward"] = next_observation[0]
        # last_position = observation[0]

        # render
        if env.visualize:
            env.render()

            if env.t % plotter_int == 0:
                if pcnn_plotter is not None:
                    pcnn_plotter.render(np.array(
                        env.agent.trajectory),# /\
                        # env.scale,
                        customize=True,
                        draw_fig=True,
                        render_elements=True, 
                        alpha_nodes=0.5,
                        alpha_edges=0.2)

                if element is not None:
                    # element.render_circuits()
                    element.circuits["DA"].render_field()
                    element.circuits["Bnd"].render_field()

                pause(0.001)

            clock.tick(FPS)

        # exit 1
        if observation[4]:
            running = False

    pygame.quit()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=1000)
    parser.add_argument("--visual", action="store_true")
    args = parser.parse_args()


    SCALE = 100.0
    brain = objects.RandomAgent(scale=SCALE)

    room = make_room(name="Square.v0")
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    agent = objects.AgentBody(position=np.array([110, 110]),
                              width=25, height=25,
                              bounds=room_bounds,
                              possible_positions=[
                                    np.array([110, 110]),
                                    np.array([110, 190]),
                                    np.array([190, 110]),
                                    np.array([190, 190])
                              ],
                              max_speed=4.0)
    reward_obj = objects.RewardObj(position=np.array([150, 150]),
                                bounds=room_bounds)

    env = Environment(room=room, agent=agent,
                      reward_obj=reward_obj,
                      rw_event="move both",
                      duration=args.duration,
                      scale=SCALE,
                      visualize=args.visual)

    run_game(env, brain, fps=100)


