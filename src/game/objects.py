# objects.py
import pygame
import numpy as np
from typing import Tuple, List
import os, sys

from game.constants import *


# absolute path to the sprites folder
sprite_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites")


class AgentBody:

    def __init__(self, position: np.ndarray,
                 room: object,
                 speed = 1.0,
                 width: int = 20, height: int = 20,
                 color: Tuple[int, int, int] = (255, 0, 0),
                 bounds: Tuple[int, int, int, int] = (0,
                                    SCREEN_WIDTH, 0, SCREEN_HEIGHT),
                 possible_positions: List[Tuple[int, int]] = None,
                 random_brain=None,
                 limit_position_len: int=-1,
                 use_sprites: bool = True):

        self.bounds = bounds
        self.room = room
        if np.all(position != None):
            self.position = position
        else:
            self.set_position()

        self.rect = pygame.Rect(self.position[0],
                                self.position[1], width, height)
        self.color = color
        self.use_sprites = use_sprites
        self.limit_position_len = limit_position_len
        self.speed = speed
        self.initial_pos = tuple(self.position)
        self.radius = max((width, height))
        self.random_brain = random_brain
        self.trajectory = [self.position.tolist() if isinstance(
                    self.position, np.ndarray) else self.position]
        self._possible_positions = possible_positions

        # Load sprites
        self.sprites = {
            "stand": pygame.image.load(f"{sprite_folder}/standeye.png"),
            "up": pygame.image.load(f"{sprite_folder}/upeye.png"),
            "down": pygame.image.load(f"{sprite_folder}/downeye.png"),
            "left": pygame.image.load(f"{sprite_folder}/lefteye.png"),
            "right": pygame.image.load(f"{sprite_folder}/righteye.png")
        }

        # Resize all sprites to fit the agent size
        for key in self.sprites:
            self.sprites[key] = pygame.transform.scale(
                            self.sprites[key], (2.*width,
                                                2.*height))

        # Default sprite
        self.current_sprite = self.sprites["stand"]

    def __str__(self) -> str:
        return f"AgentBody{tuple(self.position.tolist())}"

    def update_sprite(self, velocity: np.ndarray):
        """Updates the sprite based on the direction of movement."""
        if np.linalg.norm(velocity) < 1e-2:  # Small movement = standing
            self.current_sprite = self.sprites["stand"]
        else:
            angle = np.arctan2(velocity[1], velocity[0])

            if -np.pi / 4 <= angle < np.pi / 4:
                self.current_sprite = self.sprites["right"]
            elif np.pi / 4 <= angle < 3 * np.pi / 4:
                self.current_sprite = self.sprites["down"]
            elif -3 * np.pi / 4 <= angle < -np.pi / 4:
                self.current_sprite = self.sprites["up"]
            else:
                self.current_sprite = self.sprites["left"]

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
                    wall.has_collided(True)
                    break
                wall.has_collided(False)


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
            self.update_sprite(velocity)  # Change sprite
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

        if self.use_sprites:
            screen.blit(self.current_sprite, (self.rect.x, self.rect.y))
        else:
            pygame.draw.rect(screen, self.color, self.rect)

    def reset(self, starting_position: bool=False):
        if starting_position:
            self.position = np.array(self.initial_pos, dtype=float)
        else:
            self.set_position()
        self.rect.x, self.rect.y = self.position
        self.trajectory = []


class CartPoler:

    def __init__(self, brain: object, renderer: object=None):

        self.position = np.zeros(2)
        self.brain = brain
        self.renderer = renderer
        self.trajectories = []

    def __call__(self, velocity: np.ndarray, reward: float,
                 collision: float, goal_flag: bool) -> int:

        self.position += velocity
        # print(velocity, reward, collision, goal_flag)
        out_velocity = self.brain(velocity, reward, collision, goal_flag)

        return int(out_velocity[0] < 0)

    def render(self):
        if self.renderer is not None:
            self.renderer.render()

    def reset(self, new_position: np.ndarray):
        self.brain.reset()
        self.position = new_position


class RewardObj:

    def __init__(self, position: np.ndarray,
                 radius: int = 10,
                 sigma: int = 10,
                 tau: int = 10,
                 move_threshold: int = 10,
                 bounds: Tuple[int, int, int, int] = \
                    (100, SCREEN_WIDTH-100,
                     100, SCREEN_HEIGHT-100),
                 fetching: str="probabilistic",
                 possible_positions: List[Tuple[int, int]] = None,
                 transparent: bool = False,
                 value: str="binary",
                 silent_duration: int = 10,
                 color: Tuple[int, int, int] = (25, 255, 0),
                 delay: int = 10,
                 fetching_duration = 2,
                 use_sprites: bool = True,
                 **kwargs):

        self._bounds = bounds
        if np.all(position != None):
            self.position = position
        else:
            self.set_position()

        self.x = self.position[0]
        self.y = self.position[1]
        self._possible_positions = possible_positions
        self.radius = radius
        self.sigma = sigma
        self.tau = tau
        self.move_threshold = move_threshold
        self.v = 0
        self.beta = kwargs.get("beta", 10)
        self.alpha = kwargs.get("alpha", 0.6)
        self.color = color
        self.collected = False
        self._fetching = fetching
        self._transparent = transparent
        self.reward_value = value
        self.silent_duration = silent_duration
        self.fetching_duration = fetching_duration
        self.count = 0
        self.use_sprites = use_sprites

        self.available = True
        self.t = 0
        self.t_collected = 0
        self.delay = delay

        self.sprites = {
           "taken": pygame.image.load(f"{sprite_folder}/reward_free.png"),
           "free": pygame.image.load(f"{sprite_folder}/reward.png"),
           "locked": pygame.image.load(f"{sprite_folder}/reward_locked.png"),
        }

        # Resize all sprites to fit the agent size
        for key in self.sprites:
            self.sprites[key] = pygame.transform.scale(
                            self.sprites[key], (self.radius,
                                                self.radius))

        self.current_sprite = self.sprites["free"] 

    def __str__(self) -> str:
        return f"Reward({self.x}, {self.y}, " + \
            f"silent={self.silent_duration}, " + \
            f"{self._fetching}, " + \
            f"transparency={self._transparent})"

    def _probability_function(self, distance: float) -> float:

        # p = 1 / (1 + np.exp(-self.beta * ( np.exp(-distance**2 / \
        #     self.sigma) - self.alpha)))
        # p = np.where(p < 0.02, 0, p)
        return self.sigma if distance < self.radius else 0.0

    def _update_sprite(self):

        """ Updates the sprite based on the direction of movement. """

        if self.t - self.t_collected < 30 or self.collected==1:
            self.current_sprite = self.sprites["taken"]
        elif not self.available:
            self.current_sprite = self.sprites["locked"]
        else:
            self.current_sprite = self.sprites["free"]

    def __call__(self, agent_pos: Tuple[int, int]) -> bool:

        # decay
        self.v += -self.v / self.tau

        # distance from the agent
        distance = np.sqrt((self.x - agent_pos[0])**2 +
                      (self.y - agent_pos[1])**2)

        # if distance < self.radius and self.available and \
        if distance < self.radius: #and \
            # print(f"[Rw] {distance} ({self.radius})")

            if self.available:

                if self._fetching == "deterministic":
                    if self.reward_value == "continuous":
                        self.collected = np.exp(-distance / \
                            self.radius)
                    else:
                        self.collected = 1.0

                elif self._fetching == "probabilistic":
                    p = self._probability_function(distance)
                    self.collected = np.random.binomial(1, p)
                    if self.reward_value == "continuous":
                        self.collected *= p
        else:
            self.collected = 0.


        if self.collected:
            self.count += 1
            self.t_collected = self.t
            self.v += 1

        self.available = ((self.t - self.t_collected) > self.delay) and \
                (self.t > self.silent_duration) and \
                (not self._transparent)

        self.t += 1

        if self.use_sprites:
            self._update_sprite()

        return self.collected

    def set_position(self, position: np.ndarray=None):

        if np.all(position == None):
            if np.all(self._possible_positions) != None:
                position = self._possible_positions[
                    np.random.randint(len(self._possible_positions))
            ]
            else:
                self.x = np.random.uniform(self._bounds[0],
                                           self._bounds[1])
                self.y = np.random.uniform(self._bounds[2],
                                           self._bounds[3])
        else:
            self.x = position[0]
            self.y = position[1]

        self.position = np.array([self.x, self.y])

    @property
    def is_ready_to_move(self) -> bool:
        # out = self.v > self.move_threshold
        out = (self.count % self.move_threshold) == 0
        return out

    @property
    def is_silent(self) -> bool:
        return self.t < self.silent_duration

    def render(self, screen: pygame.Surface):

        if self.use_sprites:
            screen.blit(self.current_sprite, (self.x, self.y))
        else:
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


