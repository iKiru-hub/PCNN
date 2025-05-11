import pygame
import numpy as np
from typing import Tuple, List

import argparse
import os, sys
sys.path.append(os.path.join(os.getcwd().split("PCNN")[0], "PCNN/src"))

import gym
from gym import spaces

from game.constants import *
from utils import setup_logger

logger = setup_logger(name='ENV', level=-2, is_debugging=True)
# import game.objects as objects
# from game.objects import logger



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

                # self.color = PURPLE
                return True

        # self.color = BLACK
        return False

    def has_collided(self, collision: bool):

        self.color = PURPLE if collision else BLACK

    def render(self, screen: pygame.Surface):

        """
        plot the `self.rect` such as the y-axis
        is inverted
        """

        pygame.draw.rect(screen, self.color,
                         self.rect)#, self.thickness)

        # pygame.draw.rect(screen, self.color,
        #                 self.rect, self.thickness)


class Room:

    def __init__(self, walls_bounds: List[Wall],
                 walls_extra: List[Wall] = [],
                 bounds: Tuple[int, int, int, int] = None,
                 bounce_coeff: float = 1.0,
                 moving_wall: bool = False,
                 room_positions: np.ndarray = None,
                 name: str = "Square.v0"):

        self.name = name
        self.walls_extra = walls_extra
        self.walls = walls_extra + walls_bounds
        self.room_positions = room_positions

        if bounds is None:
            bounds = [OFFSET, OFFSET,
                      SCREEN_WIDTH-1*OFFSET,
                      SCREEN_HEIGHT-1*OFFSET]
        self.bounds = bounds
        self.bounce_coeff = bounce_coeff
        self.moving_wall = moving_wall
        self.t = 0
        self.counter = -1

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
        x = max(self.bounds[0] + margin, min(x, self.bounds[2] - margin))
        y = max(self.bounds[1] + margin, min(y, self.bounds[3] - margin))
        return x, y

    def check_collision(self, x: float, y: float,
                       velocity: np.ndarray) -> tuple:
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
                    if velocity[0] != 0:  # Move a small amount in the new direction
                        new_x = x + (velocity[0] * 0.1)
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

    def set_room_positions(self, room_positions: list):
        assert isinstance(room_positions, list), "shall be list"
        self.room_positions = np.array(room_positions)

    def get_room_positions(self) -> Tuple[float, float]:
        return self.room_positions.copy()

    def sample_next_position(self) -> Tuple[float, float]:
        self.counter += 1
        pos = self.room_positions[self.counter % len(self.room_positions)]
        return pos

    def sample_random_position(self, limit: int=None) -> Tuple[float, float]:
        if limit is not None:
            return self.room_positions[limit]  # as an idx
        return self.room_positions[np.random.randint(len(self.room_positions))]

    def render(self, screen: pygame.Surface):
        for wall in self.walls:
            wall.render(screen)


def make_room(name: str="square", thickness: float=10.,
              bounds: list=[0, 1, 0, 1],
              moving_wall: bool=False,
              visualize: bool=False):

    walls_bounds = [
        Wall(OFFSET, 2*OFFSET-thickness,
             SCREEN_WIDTH-2*OFFSET+thickness, thickness),  # Top
        Wall(OFFSET, SCREEN_HEIGHT-1*OFFSET,
             SCREEN_WIDTH-2*OFFSET+thickness, thickness),  # Bottom
        Wall(OFFSET, 2*OFFSET, thickness,
             SCREEN_HEIGHT-3*OFFSET),  # Left
        Wall(SCREEN_WIDTH-1*OFFSET, 2*OFFSET,
             thickness, SCREEN_HEIGHT-3*OFFSET),  # Right
    ]

    walls_extra = []

    if name == "Square.v1":
        walls_extra += [
            Wall(OFFSET, SCREEN_HEIGHT//2, SCREEN_WIDTH//2, thickness)
        ]
        room_positions = [
            [0.5, 0.5], [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
            [0.4, 0.4], [0.6, 0.6], [0.4, 0.6], [0.6, 0.4]
        ]
    elif name == "Square.v2":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, SCREEN_HEIGHT//2-OFFSET,
                 thickness, SCREEN_HEIGHT//2),
            Wall(2*SCREEN_WIDTH//3, OFFSET,
                    thickness, SCREEN_HEIGHT//2)
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
        ]
    elif name == "Hole.v0":
        walls_extra += [
            Wall(SCREEN_WIDTH//2.4-OFFSET,
                 SCREEN_HEIGHT//2.6,
                 SCREEN_WIDTH//3, SCREEN_HEIGHT//3),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
        ]
    elif name == "Flat.0000":
        walls_extra += [
            Wall(OFFSET, SCREEN_HEIGHT//2,
                 2*SCREEN_WIDTH//3-2*OFFSET, thickness),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
        ]
    elif name == "Flat.0001":
        walls_extra += [
            Wall(OFFSET, 2*SCREEN_HEIGHT//3,
                 2*SCREEN_WIDTH//3-OFFSET, thickness),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
            [0.5, 0.5], [0.6, 0.6]
        ]
    elif name == "Flat.0010":
        walls_extra += [
            Wall(OFFSET, SCREEN_HEIGHT//3+OFFSET,
                 2*SCREEN_WIDTH//3-OFFSET, thickness),
            Wall(OFFSET, 2*SCREEN_HEIGHT//3,
                 2*SCREEN_WIDTH//3-OFFSET, thickness),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
            [0.25, 0.5], [0.75, 0.5]
        ]
    elif name == "Flat.0011":
        walls_extra += [
            Wall(OFFSET, SCREEN_HEIGHT//3+OFFSET,
                 2*SCREEN_WIDTH//3-2*OFFSET, thickness),
            Wall(SCREEN_WIDTH//3+1*OFFSET, 2*SCREEN_HEIGHT//3,
                 2*SCREEN_WIDTH//3-2*OFFSET, thickness),
        ]
        room_positions = [
            [0.25, 0.25], [0.8, 0.8], [0.5, 0.5], [0.25, 0.5], [0.75, 0.5]
        ]
    elif name == "Flat.0100":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, SCREEN_HEIGHT//2-2*OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
            Wall(SCREEN_WIDTH//3, SCREEN_HEIGHT//2+OFFSET,
                 2*SCREEN_WIDTH//3-3*OFFSET, thickness),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
            [0.5, 0.25], [0.5, 0.75]
        ]
    elif name == "Flat.0101":
        walls_extra += [
            Wall(SCREEN_WIDTH//3+OFFSET, SCREEN_HEIGHT//3+OFFSET,
                 2*SCREEN_WIDTH//3-2*OFFSET, thickness),
            Wall(OFFSET, 2*SCREEN_HEIGHT//3,
                 2*SCREEN_WIDTH//3-OFFSET, thickness),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
            [0.5, 0.5], [0.25, 0.5], [0.75, 0.5]
        ]
    elif name == "Flat.0110":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, SCREEN_HEIGHT//3-2*OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
            Wall(SCREEN_WIDTH//3, SCREEN_HEIGHT//2+OFFSET,
                 2*SCREEN_WIDTH//3-3*OFFSET, thickness),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
            [0.75, 0.25]
        ]
    elif name == "Flat.0111":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, SCREEN_HEIGHT//3-4*OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-0*OFFSET),
            Wall(SCREEN_WIDTH//3, SCREEN_HEIGHT//2+2*OFFSET,
                 2*SCREEN_WIDTH//3-4*OFFSET, thickness),
            Wall(2*SCREEN_WIDTH//3-OFFSET, SCREEN_HEIGHT//3+1*OFFSET,
                 2*SCREEN_WIDTH//3-5*OFFSET, thickness),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
            [0.5, 0.5], [0.25, 0.5], [0.75, 0.5]
        ]
    elif name == "Flat.1000":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, 2*OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-2*OFFSET),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
            [0.5, 0.5], [0.6, 0.6], [0.75, 0.4]
        ]
    elif name == "Flat.1001":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, 2*OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-2*OFFSET),
            Wall(3*SCREEN_WIDTH//5, SCREEN_HEIGHT//2+1*OFFSET,
                 thickness, SCREEN_HEIGHT//2-2*OFFSET),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
            [0.5, 0.5], [0.25, 0.5]
        ]
    elif name == "Flat.1010":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, 2*OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-2*OFFSET),
            Wall(2*SCREEN_WIDTH//3, 2*OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-2*OFFSET),
        ]
        room_positions = [
            [0.2, 0.2], [0.755, 0.755], [0.2, 0.755], [0.755, 0.2],
            [0.5, 0.5], [0.5, 0.755]
        ]
    elif name == "Flat.1011":
        walls_extra += [
            Wall(OFFSET, SCREEN_HEIGHT//2+OFFSET,
                 2*SCREEN_WIDTH//3-3*OFFSET, thickness),
            Wall(2*SCREEN_WIDTH//3, SCREEN_HEIGHT//3+thickness,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75],
            [0.5, 0.5]
        ]
    elif name == "Flat.1100":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, SCREEN_HEIGHT//3-2*OFFSET,
                 thickness, SCREEN_HEIGHT//3-2*OFFSET),
            Wall(OFFSET, SCREEN_HEIGHT//2+OFFSET,
                 2*SCREEN_WIDTH//3-4*OFFSET, thickness),
            Wall(2*SCREEN_WIDTH//3, SCREEN_HEIGHT//3+thickness,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
        ]
        room_positions = [
            [0.2, 0.2], [0.8, 0.8], [0.2, 0.8],
        ]
    elif name == "Flat.1101":
        walls_extra += [
            Wall(4*OFFSET, SCREEN_HEIGHT//2+0*OFFSET,
                 1*SCREEN_WIDTH//3-0*OFFSET, thickness),
            Wall(SCREEN_WIDTH//3, 4*OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-3*OFFSET),
            Wall(2*SCREEN_WIDTH//3, SCREEN_HEIGHT//3+thickness,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
        ]
        room_positions = [
            [0.25, 0.25], [0.8, 0.8], [0.25, 0.8], [0.8, 0.25],
            [0.8, 0.4], [0.25, 0.8]
        ]
    elif name == "Flat.1110":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, SCREEN_HEIGHT//3-2*OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
            Wall(2*SCREEN_WIDTH//3, 2*OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
            [0.5, 0.75]
        ]
    elif name == "Flat.1111":
        walls_extra += [
            Wall(SCREEN_WIDTH//3, SCREEN_HEIGHT//3-2*OFFSET,
                 thickness, 2*SCREEN_HEIGHT//3-OFFSET),
            Wall(SCREEN_WIDTH//2+0.5*OFFSET, SCREEN_HEIGHT//3+OFFSET,
                 2*SCREEN_WIDTH//3-3.5*OFFSET, thickness),
        ]
        room_positions = [
            [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25],
            [0.75, 0.5]
        ]
    elif name == "Hallway.00":

        height = 5

        walls_bounds = [
            Wall(OFFSET, height*OFFSET-thickness,
                 SCREEN_WIDTH-2*OFFSET+thickness, thickness),  # Top
            Wall(OFFSET, SCREEN_HEIGHT-height*OFFSET,
                 SCREEN_WIDTH-2*OFFSET+thickness, thickness),  # Bottom
            Wall(OFFSET, height*OFFSET, thickness,
                 SCREEN_HEIGHT-(height*2)*OFFSET),  # Left
            Wall(SCREEN_WIDTH-1*OFFSET, height*OFFSET,
                 thickness, SCREEN_HEIGHT-(height*2)*OFFSET),  # Right
        ]
        room_positions = [
            [0.25, 0.5], [0.5, 0.5], [0.75, 0.5]
        ]
    elif name == "Square.b":
        room_positions = [[0.8, 0.6], [0.8, 0.6]]
        walls_bounds += [Wall(5*SCREEN_WIDTH//8, SCREEN_HEIGHT//2-1*OFFSET,
                 thickness, SCREEN_HEIGHT//2-2*OFFSET)]
    else:
        name = "Square.v0"
        room_positions = [[0.5, 0.5]] + [[0.8, 0.6]] + \
                [[0.2806364523971874, 0.2435772280597547],
                 [0.84758117279920714, 0.8512782166456011],
                 [0.621934808306112, 0.34311630349521194],
                 [0.29821968919291714, 0.5418489965017503],
                 [0.4599671145340115, 0.6945489730728632],
                 [0.39576820293662573, 0.29145453982651204],
                 [0.6186957334336476, 0.4516116142178095]]

            # np.random.uniform(0.25, 0.755, (40, 2)).tolist()

    room = Room(walls_bounds=walls_bounds,
                walls_extra=walls_extra,
                moving_wall=moving_wall,
                room_positions=np.array(room_positions) * GAME_SCALE,
                name=name)

    return room


def render_room(name: str, pos_idx: int=-1):

    room = make_room(name=name, thickness=30.)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    room.render(screen)
    pygame.display.flip()

    print(f"\nRoom: {name}")

    if pos_idx > -1:
        pos = room.room_positions[pos_idx]
    else:
        pos = None

    print(f"{pos=}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        room.render(screen)
        if pos is not None:
            pygame.draw.circle(screen, (25, 155, 10),
                              (pos[0], pos[1]), 10)
        pygame.display.flip()


def get_random_room() -> str:
    return np.random.choice(ROOMS)



""" Environment class """


class Environment:

    def __init__(self, room: Room, agent: object,
                 reward_obj: object,
                 agent_position_list: list=None,
                 rw_event: str = "nothing",
                 duration: int=np.inf,
                 # scale: float=1.0,
                 verbose: bool=False,
                 visualize: bool=False):

        # Environment components
        self.room = room
        self.agent = agent
        self.reward_obj = reward_obj
        self._agent_position_list = agent_position_list

        # Reward event
        self.rw_event = rw_event
        self.duration = duration
        # self.scale = scale# * 0.02
        self.rw_count = 0
        self.t = 0
        self.velocity = np.zeros(2)
        self.trajectory = []
        self.trajectory_set = [[]]
        self.traj_color = (255, 0, 0)
        self._collision = 0
        self._reward = 0

        self.rw_time = 0
        self.time_flag = None

        # rendering
        self.visualize = visualize
        self.verbose = verbose
        if visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Simple Pygame Game")
            logger.debug("%env rendering")

        if self.agent.limit_position_len is not None:
            logger.debug(f"Agent fixed position: {self.agent.limit_position_len}")

        logger.debug(f"Reward event: {rw_event}")

        logger.debug(self)

    def __str__(self):
        return f"Environment({self.room}, duration={self.duration}, " + \
               f"verbose={self.verbose})"

    def __call__(self, velocity: np.ndarray, brain: object) -> \
        Tuple[float, np.ndarray, bool, bool]:

        # scale velocity
        # velocity *= self.scale
        self.t += 1

        if (self.t == self.rw_time): #> self.reward_obj.fetching_duration:
            # print(f"[ENV] [t={self.t} = {self.rw_time}")
            terminated = self._reward_event(brain=None)
            # self.trajectory_set += [self.trajectory]
            # self.trajectory_set += [[]]
            # self.trajectory = []
            # print("trajectory_set: ", len(self.trajectory_set))

        # Update agent position with improved collision handling
        velocity, collision = self.agent(velocity)
        # self.velocity = np.around(velocity, 3)

        # Check reward collisions
        reward = self.reward_obj(self.agent.rect.center)
        terminated = False
        if reward and self.rw_time <= self.t:
            self.rw_time = self.t + self.reward_obj.fetching_duration
            # print(f"[ENV] [t={self.t}] +reward [{self.rw_time=}, {self.t=}]")
            self.rw_count += 1

        # if self.verbose:
            # if reward:
            #     logger.info(f"[t={self.t}] +reward")
            # if collision:
            #     logger.info(f"[t={self.t}] -collision")

        # position = self.agent.position.copy() / self.scale
        self.trajectory.append([self.agent.position[0]+8,
                                self.agent.position[1]+8])
        self.trajectory_set[-1].append([self.agent.position[0]+8,
                                        self.agent.position[1]+8])
        self._collision = float(collision)
        self._reward = float(reward)

        # return self.agent.position.copy(), velocity, reward, float(collision), float(self.t >= self.duration), terminated
        return self.agent.position, float(collision), self._reward > 0., float(self.t >= self.duration), terminated

    def _reward_event(self, brain: object):

        """
        logic for when the reward is collected
        """

        if self.rw_event == "move reward":
            self.reward_obj.set_position()
            return False
        elif self.rw_event == "move agent":
            self._reset_agent_position(brain)
            return False
        elif self.rw_event == "move both":
            self._reset_agent_position(brain)
            if self.reward_obj.is_ready_to_move:
                if self.reward_obj.preferred_positions is not None:
                    idx = np.random.choice(self.reward_obj.preferred_positions)
                    self.reward_obj.set_position(
                            self.room.sample_random_position(idx))
                else:
                    self.reward_obj.set_position(self.room.sample_next_position())
            return False
        elif self.rw_event == "nothing" or self.rw_event == 'none':
            return False
        elif self.rw_event == "terminate":
            return True
        else:
            raise ValueError(f"Unknown reward " + \
                f"event: {self.rw_event}")

    def _reset_agent_position(self, brain: object=None,
                              exploration: bool = False):

        if brain is not None:
            brain.reset()

        prev_position = self.agent.position.copy()

        if exploration:
            self.agent.set_position(self.room.sample_next_position())
        else:
            if self._agent_position_list is not None:
                self.agent.set_position(np.random.choice(self._agent_position_list))
            elif self.agent.limit_position_len > -1:
                self.agent.set_position(
                    self.room.get_room_positions()[
                            self.agent.limit_position_len])
                logger.debug(f"reset position to {self.agent.position}")
            else:
                self.agent.set_position(self.room.sample_next_position())

        displacement = [(self.agent.position[0] - prev_position[0]),
                        (-self.agent.position[1] + prev_position[1])]
        self.trajectory_set += [[]]
        self.trajectory = []

    @property
    def position(self):
        return self.agent.position.copy()

    @property
    def reward_availability(self):
        return self.reward_obj.available

    def set_time_flag(self):
        self.time_flag = len(self.trajectory_set)-1

    def render(self, **kargs):

        self.screen.fill(WHITE)

        # Render game objects
        self.room.render(self.screen)

        # plot trajectory
        # if len(self.trajectory) > 1:
        #     pygame.draw.lines(self.screen, (*self.traj_color, 0.1),
        #                       False, self.trajectory, 1)

        if len(self.trajectory_set) > 0:
            # print(self.t, self.trajectory_set)
            for i, traj in enumerate(self.trajectory_set):
                # print(traj)
                # print(*np.array(traj).T)
                if len(traj) > 1:
                    pygame.draw.lines(self.screen, (*self.traj_color,
                                0.5/(len(self.trajectory_set)-i+1)),
                                        False, traj, 1)
            # pygame.draw.lines(self.screen, (*self.traj_color, 0.1),
            #                   False, self.trajectory, 1)

        # write text
        font = pygame.font.Font(None, 36)
        text = f"#R={self.reward_obj.count} | "
        text += f"t: {self.t:04d}"
        # text += f"-R={self.reward_availability} | "
        # text += f"C={self._collision} |"
        # text += f"v={np.around(self.velocity, 2)}"
        score_text = font.render(text, True, BLACK)

        self.reward_obj.render(self.screen)
        self.agent.render(self.screen)

        self.screen.blit(score_text, (200, 5))
        pygame.display.flip()

    def reset(self):

        self.rw_count = 0
        self.t = 0
        self.velocity = np.zeros(2)
        self.trajectory = []
        self.trajectory_set = [[]]
        self.traj_color = (255, 0, 0)
        self._collision = 0
        self._reward = 0
        self._reset_agent_position()

        self.rw_time = 0
        self.time_flag = None


class EnvironmentWrapper(gym.Env):
    def __init__(self, env: Environment):
        super(EnvironmentWrapper, self).__init__()
        self.env = env
        self.prev_position = self.env.position.copy()

        self.speed = 1.

        # Define action and observation space
        # Action: 2D velocity in range [-1, 1]
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(2,), dtype=np.float32)

        # Observation: velocity_x, velocity_y, agent_y_velocity,
        # collision_flag, reward_available_flag
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        if self.env.visualize:
            logger.debug("env rendering")

    def set_speed(self, speed: float):
        self.speed = speed

    def reset(self):
        self.env.reset()
        self.prev_position = self.env.position.copy()
        velocity = [0.0, 0.0]
        obs = self._get_obs(velocity)

        return obs

    def step(self, action):
        # Clip actions just in case
        action = np.clip(action, -1., 1.) * self.speed 

        # Get previous position
        prev_position = self.env.position.copy()

        # Your env does not use brain during training
        obs_tuple = self.env(velocity=np.array([action[0], -action[1]]), brain=None)

        # Compute velocity (displacement)
        new_position = self.env.position.copy()
        velocity = [new_position[0] - prev_position[0],
                    -(new_position[1] - prev_position[1])]

        # Construct observation
        obs = self._get_obs(velocity)

        # Reward
        reward = float(obs_tuple[2]) if obs_tuple[2] is not False else 0.0

        # Done flag from your environment (based on time)
        done = bool(obs_tuple[3])# or reward > 0)

        return obs, reward, done, {}

    def render(self, mode='human'):
        self.env.render()

    @property
    def duration(self):
        return self.env.duration

    @property
    def count(self):
        return self.env.count

    @property
    def visualize(self):
        return self.env.visualize

    @property
    def t(self):
        return self.env.t

    def _get_obs(self, velocity):
        return np.array([
            self.env.position[0],
            self.env.position[1],
            float(self.env._reward),           # from your env
            float(self.env._collision),            # from your env
            float(np.sqrt(velocity[0]**2 + velocity[1]**2))    # from your env
        ], dtype=np.float32)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--room", type=str, default="Square.v0",
                        help='room name: ["Square.v0", "Square.v1", "Square.v2",' + \
                         '"Hole.v0", "Flat.0000", "Flat.0001", "Flat.0010", "Flat.0011",' + \
                         '"Flat.0110", "Flat.1000", "Flat.1001", "Flat.1010",' + \
                         '"Flat.1011", "Flat.1110"] or `random`')
    parser.add_argument("--pos", type=int, default=-1,
                        help="position idx")
    args = parser.parse_args()


    render_room(name=args.room, pos_idx=args.pos)

