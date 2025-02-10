# objects.py
import pygame
import numpy as np
from typing import Tuple, List
import os, sys

from game.constants import *


class AgentBody:

    def __init__(self, position: np.ndarray,
                 room: object,
                 speed = 1.0,
                 width: int = 20, height: int = 20,
                 color: Tuple[int, int, int] = (255, 0, 0),
                 bounds: Tuple[int, int, int, int] = (0, SCREEN_WIDTH, 0, SCREEN_HEIGHT),
                 possible_positions: List[Tuple[int, int]] = None,
                 random_brain=None):

        self.bounds = bounds
        self.room = room
        if np.all(position != None):
            self.position = position
        else:
            self.set_position()

        self.rect = pygame.Rect(self.position[0],
                                self.position[1], width, height)
        self.color = color
        self.speed = speed
        self.initial_pos = tuple(self.position)
        self.radius = max((width, height))
        self.random_brain = random_brain
        self.trajectory = [self.position.tolist() if isinstance(self.position, np.ndarray) else self.position]
        self._possible_positions = possible_positions

    def __str__(self) -> str:
        return f"AgentBody{tuple(self.position.tolist())}"

    def __call__(self, velocity: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Move the agent and handle bouncing collisions with walls

        Parameters
        ----------
        velocity : np.ndarray
            The velocity vector to move the agent
        room : object
            The room object containing the walls

        Returns
        -------
        Tuple[np.ndarray, bool]
            The potentially modified velocity vector and whether a collision occurred
        """
        # Store previous position
        prev_pos = self.position.copy()

        # Try to move
        next_pos = self.position + velocity
        self.rect.x = int(next_pos[0])
        self.rect.y = int(next_pos[1])

        # Check for collisions
        collision = False

        # First check boundaries
        if self.room.is_out_of_bounds(self.rect.x, self.rect.y):
            collision = True
        else:
            # Then check wall collisions
            for wall in self.room.walls:
                if self.rect.colliderect(wall.rect):
                    collision = True
                    break

        if collision:
            # Revert to previous position
            self.position = prev_pos
            self.rect.x = int(prev_pos[0])
            self.rect.y = int(prev_pos[1])

            # Simple bounce: reverse velocity components
            bounce_factor = 0.8  # Adjust this to control bounce strength
            velocity *= -bounce_factor

            return velocity, True
        else:
            # No collision, update position
            self.position = next_pos
            return velocity, False

    def set_position(self, position: np.ndarray=None):

        if np.all(position == None):
            if np.all(self._possible_positions) != None:
                position = self._possible_positions[
                    np.random.randint(len(self._possible_positions))
            ]
            else:
                # raise ValueError("No possible positions provided")
                position = np.array([
                    np.random.uniform(self.bounds[0],
                                      self.bounds[1]),
                    np.random.uniform(self.bounds[2],
                                      self.bounds[3])
                ])
        self.position = position

    def render(self, screen: pygame.Surface):
        pygame.draw.rect(screen, self.color, self.rect)

    def reset(self, starting_position: bool=False):
        if starting_position:
            self.position = np.array(self.initial_pos, dtype=float)
        else:
            self.set_position()
        self.rect.x, self.rect.y = self.position
        self.trajectory = []


class RewardObj:

    def __init__(self, position: np.ndarray,
                 radius: int = 10,
                 bounds: Tuple[int, int, int, int] = \
                    (100, SCREEN_WIDTH-100,
                     100, SCREEN_HEIGHT-100),
                 fetching: str="probabilistic",
                 transparent: bool = False,
                 value: str="binary",
                 silent_duration: int = 10,
                 color: Tuple[int, int, int] = (25, 255, 0),
                 delay: int = 10):

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
        self._transparent = transparent
        self.reward_value = value
        self.silent_duration = silent_duration
        self.count = 0

        self.available = True
        self.t = 0
        self.t_collected = 0
        self.delay = delay

    def __str__(self) -> str:
        return f"Reward({self.x}, {self.y}, silent={self.silent_duration}, " + \
            f"transparency={self._transparent})"

    def __call__(self, agent_pos: Tuple[int, int]) -> bool:

        distance = np.sqrt((self.x - agent_pos[0])**2 +
                      (self.y - agent_pos[1])**2)

        # if distance < self.radius and self.available and \
        if distance < self.radius and \
            self.t > self.silent_duration:
            # print(f"[RW] collected | distance:{distance} | t:{self.t}" + \
            #     " | silent:{self.silent_duration} av:{self.available}")

            if self._fetching == "deterministic":
                if self.reward_value == "continuous":
                    self.collected = np.exp(-distance / self.radius)
                else:
                    self.collected = 1.0

            elif self._fetching == "probabilistic":
                p = np.exp(-distance / (2 * self.radius**2))
                self.collected = np.random.binomial(1, p)
                if self.reward_value == "continuous":
                    self.collected *= p
        else:
            self.collected = 0.

        if self.collected:
            self.count += 1
            self.t_collected = self.t

        self.available = ((self.t - self.t_collected) > self.delay) and \
                (self.t > self.silent_duration) and \
                (not self._transparent)

        self.t += 1

        return self.collected

    def set_position(self, position: np.ndarray=None):

        if np.all(position == None):
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

    def reset(self, **kwargs):
        pass


