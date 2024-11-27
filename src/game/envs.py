import pygame
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

from constants import *
import objects



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
        pygame.draw.rect(screen, self.color,
                        self.rect, self.thickness)


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
                    logger.debug(f"Collision detected: pos={new_pos}, vel={velocity}")
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


def make_room(name: str="square", thickness: float=20.,
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
            Wall(SCREEN_WIDTH//2, SCREEN_HEIGHT//2-OFFSET,
                 thickness, SCREEN_HEIGHT//2),
        ]
    elif name == "flat":
        raise NotImplementedError
    elif name == "flat2":
        raise NotImplementedError
    elif name == "Hole.v0":
        walls_extra += [
            Wall(SCREEN_WIDTH//2.5-OFFSET, SCREEN_HEIGHT//2.5-OFFSET,
                 SCREEN_WIDTH//3, SCREEN_HEIGHT//3),
        ]
    else:
        name = "Square.v0"

    room = Room(walls_bounds=walls_bounds,
                walls_extra=walls_extra,
                moving_wall=moving_wall,
                name=name)

    return room


""" Environment class """


class Environment:

    def __init__(self, room: Room, agent: objects.AgentBody,
                 reward_obj: objects.Reward,
                 rw_event: str = "nothing",
                 duration: int=np.inf,
                 visualize: bool=False):

        # Environment components
        self.room = room
        self.agent = agent
        self.reward_obj = reward_obj

        # Reward event
        self.rw_event = rw_event
        self.duration = duration
        self.t = 0

        # rendering
        self.visualize = visualize
        if visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Simple Pygame Game")

    def __call__(self, velocity: np.ndarray) -> Tuple[float, np.ndarray, bool, bool]:

        # # Update agent position with improved collision handling
        _, collision = self.agent(velocity,
                                              self.room)

        # Check reward collisions
        reward = self.reward_obj(self.agent.rect.center)
        if reward:
            self._reward_event()

        self.t += 1

        return reward, self.agent.position.copy(), collision, self.t >= self.duration

    def _reward_event(self):

        """
        logic for when the reward is collected
        """

        if self.rw_event == "move reward":
            self.reward_obj.set_position()
        elif self.rw_event == "move agent":
            self.agent.set_position()
            self.agent.reset()
        elif self.rw_event == "move both":
            self.agent.set_position()
            self.reward_obj.set_position()
        elif self.rw_event == "nothing":
            pass
        else:
            raise ValueError(f"Unknown reward " + \
                f"event: {self.rw_event}")

    @property
    def position(self):
        return self.agent.position.copy()

    def render(self):

        if not self.visualize:
            return

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



def run_env(env: Environment,
            brain: object,
            fps: int = 30):

    clock = pygame.time.Clock()

    running = True
    while running:

        # Event handling
        if env.visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # update
        velocity = brain()

        obs = env(velocity=velocity)

        reward, position, collision, done = obs

        env.render()

        if env.visualize:
            clock.tick(FPS)

        if done:
            running = False

    pygame.quit()



if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=1000)
    parser.add_argument("--visual", action="store_true")
    args = parser.parse_args()



    brain = objects.RandomAgent()
    room = make_room(name="Square.v0")
    room_bounds = [room.bounds[0], room.bounds[2],
                   room.bounds[1], room.bounds[3]]
    agent = objects.AgentBody(110, 110,
                              width=25, height=25,
                              bounds=room_bounds,
                              max_speed=4.0)
    reward_obj = objects.Reward(x=150, y=150,
                                bounds=room_bounds)

    env = Environment(room=room, agent=agent,
                      reward_obj=reward_obj,
                      rw_event="move both",
                      duration=args.duration,
                      visualize=args.visual)

    run_env(env, brain, fps=100)


