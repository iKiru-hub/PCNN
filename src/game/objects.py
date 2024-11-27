# objects.py
import pygame
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

from constants import *



class AgentBody:

    def __init__(self, x: int, y: int,
                 width: int = 20, height: int = 20,
                 color: Tuple[int, int, int] = (255, 0, 0),
                 max_speed: float = 5.0,
                 random_brain=None):

        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.initial_pos = (x, y)
        self.width = width
        self.height = height
        # self.max_speed = max_speed
        self.position = np.array([float(x), float(y)])
        self.random_brain = random_brain

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

        return velocity, False

    def render(self, screen: pygame.Surface):
        pygame.draw.rect(screen, self.color, self.rect)

    def reset(self):
        self.position = np.array(self.initial_pos, dtype=float)
        self.rect.x, self.rect.y = self.initial_pos


class RandomAgent:

    def __init__(self, change_interval: int = 30,
                 max_speed: float = 4.0):
        self.change_interval = change_interval
        self.steps = 0
        self.current_velocity = np.zeros(2)
        self.max_speed = max_speed

    def __call__(self) -> np.ndarray:
        if self.steps % self.change_interval == 0:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(1.0, self.max_speed)
            self.current_velocity = np.array([
                np.cos(angle) * speed,
                np.sin(angle) * speed
            ])
        self.steps += 1
        return self.current_velocity


class Reward:

    def __init__(self, x: int, y: int,
                 radius: int = 10,
                 color: Tuple[int, int, int] = (25, 255, 0)):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.collected = False

    def __str__(self) -> str:
        return f"Reward({self.x}, {self.y})"

    def __call__(self, agent_pos: Tuple[int, int]) -> bool:
        # if self.collected:
        #     return False

        dist = np.sqrt((self.x - agent_pos[0])**2 +
                      (self.y - agent_pos[1])**2)
        if dist < self.radius * 2:
            self.collected = True
            # self.x = -10
            # Randomize position
            self.x = np.random.randint(200, SCREEN_WIDTH-200)
            self.y = np.random.randint(200, SCREEN_HEIGHT-200)
            return True
        return False

    def reset(self):
        self.collected = False

    def render(self, screen: pygame.Surface):
        pygame.draw.circle(screen, self.color,
                         (self.x, self.y), self.radius)



