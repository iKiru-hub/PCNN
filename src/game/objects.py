# objects.py
import pygame
import numpy as np
from typing import Tuple, List
from game.constants import *
import os, sys


sys.path.append(os.path.join(os.getcwd().split("PCNN")[0], "PCNN/src"))
from game.constants import *
from utils_core import setup_logger

# try:
#     from constants import *
# except ImportError:
    # from game.constants import *

logger = setup_logger("GAME", level=2)


class AgentBody:

    def __init__(self, position: np.ndarray,
                 width: int = 20, height: int = 20,
                 color: Tuple[int, int, int] = (255, 0, 0),
                 bounds: Tuple[int, int, int, int] = (0, SCREEN_WIDTH, 0, SCREEN_HEIGHT),
                 max_speed: float = 5.0,
                 random_brain=None):

        self.bounds = bounds
        if np.all(position != None):
            self.position = position
        else:
            self.set_position()

        self.rect = pygame.Rect(self.position[0],
                                self.position[1], width, height)
        self.color = color
        self.initial_pos = tuple(self.position)
        self.radius = max((width, height))
        # self.max_speed = max_speed
        self.random_brain = random_brain
        self.trajectory = []

    def __str__(self) -> str:
        return f"AgentBody{tuple(self.position.tolist())}"

    def __call__(self, velocity: np.ndarray,
                 room: object) -> Tuple[float, float]:
        """
        Move with very small steps, checking collisions at each step

        Parameters
        ----------
        velocity : np.ndarray
            The velocity vector to move the agent
        room : object
            The room object containing the walls
        """

        if self.random_brain is not None:
            velocity = self.random_brain()

        # Clamp velocity to max speed
        # speed = np.linalg.norm(velocity)
        # if speed > self.max_speed:
        #     velocity = velocity * (self.max_speed /\
        #         speed)

        # Use very small steps for movement
        step_velocity = velocity / NUM_STEPS

        for _ in range(NUM_STEPS):
            # Store previous position
            prev_pos = self.position.copy()

            # Try to move
            next_pos = self.position + step_velocity

            # Check for collisions at the new position
            self.rect.x = int(next_pos[0])
            self.rect.y = int(next_pos[1])

            collision = False

            # First check boundaries
            if room.is_out_of_bounds(self.rect.x,
                                     self.rect.y):
                collision = True
            else:
                # Then check wall collisions
                for wall in room.walls:
                    if self.rect.colliderect(wall.rect):
                        collision = True
                        break

            if collision:
                # Revert to previous position
                self.position = prev_pos
                self.rect.x = int(prev_pos[0])
                self.rect.y = int(prev_pos[1])

                # Try moving in x direction only
                test_pos = prev_pos.copy()
                test_pos[0] += step_velocity[0]
                self.rect.x = int(test_pos[0])
                x_ok = not any(self.rect.colliderect(
                    wall.rect) for wall in room.walls)

                # Try moving in y direction only
                test_pos = prev_pos.copy()
                test_pos[1] += step_velocity[1]
                self.rect.y = int(test_pos[1])
                y_ok = not any(self.rect.colliderect(
                    wall.rect) for wall in room.walls)

                # Apply valid movements
                if x_ok:
                    self.position[0] += step_velocity[0]
                if y_ok:
                    self.position[1] += step_velocity[1]

                # Update rect to final position
                self.rect.x = int(self.position[0])
                self.rect.y = int(self.position[1])

                # Modify velocity based on collision
                if not x_ok:
                    velocity[0] *= -1
                if not y_ok:
                    velocity[1] *= -1

                return velocity, True
            else:
                self.position = next_pos

        self.trajectory.append(self.position.tolist())

        return velocity, False

    def set_position(self, position: np.ndarray=None):

        if position is None:
            position = np.array([
                np.random.uniform(self.bounds[0],
                                  self.bounds[1]),
                np.random.uniform(self.bounds[2],
                                  self.bounds[3])
            ])
        self.position = position

    def render(self, screen: pygame.Surface):
        pygame.draw.rect(screen, self.color, self.rect)

    def reset(self):
        self.position = np.array(self.initial_pos, dtype=float)
        self.rect.x, self.rect.y = self.initial_pos
        self.trajectory = []


class RewardObj:

    def __init__(self, position: np.ndarray,
                 radius: int = 10,
                 bounds: Tuple[int, int, int, int] = (0, SCREEN_WIDTH, 0, SCREEN_HEIGHT),
                 fetching: str="probabilistic",
                 color: Tuple[int, int, int] = (25, 255, 0)):

        self._bounds = bounds
        if np.all(position != None):
            self.position = position
        else:
            self.set_position()

        self.x = self.position[0]
        self.y = self.position[1]
        self.radius = radius
        self.color = color
        self.collected = False
        self._fetching = fetching

    def __str__(self) -> str:
        return f"Reward({self.x}, {self.y})"

    def __call__(self, agent_pos: Tuple[int, int]) -> bool:

        distance = np.sqrt((self.x - agent_pos[0])**2 +
                      (self.y - agent_pos[1])**2)

        if distance < self.radius * 2:

            if self._fetching == "deterministic":
                self.collected = 1.0
            else:
                p = np.exp(-distance / (2 * self.radius**2))
                self.collected = np.random.binomial(1, p)
        else:
            self.collected = 0.

        return self.collected

    def set_position(self, position: np.ndarray=None):

        if position is None:
            self.x = np.random.uniform(self._bounds[0],
                                       self._bounds[1])
            self.y = np.random.uniform(self._bounds[2],
                                       self._bounds[3])
        self.position = np.array([self.x, self.y])

    def render(self, screen: pygame.Surface):
        pygame.draw.circle(screen, self.color,
                         (self.x, self.y), self.radius)

    def reset(self):
        self.set_position()
        self.collected = False


class RandomAgent:

    def __init__(self, change_interval: int = 30,
                 max_speed: float = 4.0,
                 scale: float = 1.0):
        self.change_interval = change_interval
        self.steps = 0
        self.current_velocity = np.zeros(2)
        self.max_speed = max_speed

        self.scale = scale

    def __call__(self, *args) -> np.ndarray:

        if self.steps % self.change_interval == 0:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(1.0, self.max_speed)
            self.current_velocity = np.array([
                np.cos(angle) * speed,
                np.sin(angle) * speed
            ])
        self.steps += 1
        return self.current_velocity / self.scale



