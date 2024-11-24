# objects.py
import pygame
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class AgentBody:
    def __init__(self, x: int, y: int, 
                 width: int = 20, height: int = 20, 
                 color: Tuple[int, int, int] = (255, 0, 0)):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.initial_pos = (x, y)
        self.width = width
        self.height = height
        
    def reset(self):
        self.rect.x, self.rect.y = self.initial_pos
        
    def __call__(self, velocity: np.ndarray) -> Tuple[int, int]:
        self.rect.x += velocity[0]
        self.rect.y += velocity[1]
        return self.rect.x, self.rect.y
    
    def render(self, screen: pygame.Surface):
        pygame.draw.rect(screen, self.color, self.rect)

class Reward:
    def __init__(self, x: int, y: int, 
                 radius: int = 10,
                 color: Tuple[int, int, int] = (255, 255, 0)):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.collected = False
        
    def __call__(self, agent_pos: Tuple[int, int]) -> bool:
        if self.collected:
            return False
            
        dist = np.sqrt((self.x - agent_pos[0])**2 + 
                      (self.y - agent_pos[1])**2)
        if dist < self.radius * 2:
            self.collected = True
            return True
        return False
    
    def reset(self):
        self.collected = False
        
    def render(self, screen: pygame.Surface):
        if not self.collected:
            pygame.draw.circle(screen, self.color, 
                             (self.x, self.y), self.radius)

# envs.py
import pygame
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

BLACK = (0, 0, 0)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class Wall:
    def __init__(self, x: int, y: int,
                 width: int, height: int,
                 thickness: int = 5,
                 color: Tuple[int, int, int] = BLACK):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.thickness = thickness
        # center of the base
        x_hortog_vector = np.array([
            [x + width//2, y],
            [x + width//2, y + height]
        ])
        y_hortog_vector = np.array([
            [x, y + height//2],
            [x + width, y + height//2]
        ])
        self._wall_vectors = np.stack([x_hortog_vector,
                                     y_hortog_vector])
                                     
    def render(self, screen: pygame.Surface):
        pygame.draw.rect(screen, self.color,
                        self.rect, self.thickness)

class Room:
    def __init__(self, walls: List[Wall] = None,
                 bounds: Tuple[int, int, int, int] = (0, 0, 
                                                     SCREEN_WIDTH,
                                                     SCREEN_HEIGHT),
                 bounce_coeff: float = 1.0,
                 name: str = "Room"):
        self.walls = walls if walls else []
        self.bounds = bounds
        self.num_walls = len(self.walls)
        self.bounce_coeff = bounce_coeff
        self.name = name
        self.wall_vectors = self._make_wall_vectors()
        
    def _make_wall_vectors(self) -> np.ndarray:
        if not self.walls:
            return np.array([])
        return np.stack([wall._wall_vectors for wall in self.walls])
    
    def __call__(self, x: int, y: int, 
                 velocity: np.ndarray) -> Tuple[np.ndarray, bool]:
        if self.is_out_of_bounds(x, y):
            x, y = self._check_bounds(x, y)
            velocity *= -self.bounce_coeff
            return velocity, True
            
        velocity, collision = self.check_collision(x, y, velocity)
        if collision:
            velocity *= self.bounce_coeff
        return velocity, collision
    
    def _check_bounds(self, x: int, y: int) -> Tuple[int, int]:
        x = max(self.bounds[0], 
               min(x, self.bounds[2]))
        y = max(self.bounds[1], 
               min(y, self.bounds[3]))
        return x, y
    
    def is_out_of_bounds(self, x: int, y: int) -> bool:
        return (x < self.bounds[0] or 
                x > self.bounds[2] or
                y < self.bounds[1] or 
                y > self.bounds[3])
    
    def check_collision(self, x: int, y: int,
                       velocity: np.ndarray) -> Tuple[np.ndarray, bool]:
        collision = False
        for wall in self.walls:
            if wall.rect.collidepoint(x + velocity[0], y):
                velocity[0] *= -1
                collision = True
                logger.debug(f"Collision x {velocity[0]}")
            if wall.rect.collidepoint(x, y + velocity[1]):
                velocity[1] *= -1
                collision = True
                logger.debug(f"Collision y {velocity[1]}")
        return velocity, collision
    
    def add_wall(self, wall: Wall):
        self.walls.append(wall)
        self.num_walls += 1
        self.wall_vectors = self._make_wall_vectors()
        
    def render(self, screen: pygame.Surface):
        for wall in self.walls:
            wall.render(screen)

# main.py
import pygame
import numpy as np
import logging
from objects import AgentBody, Reward
from envs import Wall, Room

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Simple Pygame Game")
        self.clock = pygame.time.Clock()
        
        # Create game objects
        self.setup_game()
        
    def setup_game(self):
        # Create walls for the room
        walls = [
            Wall(100, 100, 600, 20),  # Top
            Wall(100, 480, 600, 20),  # Bottom
            Wall(100, 100, 20, 400),  # Left
            Wall(680, 100, 20, 400),  # Right
        ]
        
        # Create room
        self.room = Room(walls)
        
        # Create agent
        self.agent = AgentBody(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
        
        # Create rewards
        self.rewards = [
            Reward(200, 200),
            Reward(600, 400),
            Reward(400, 300)
        ]
        
        # Game state
        self.velocity = np.array([0.0, 0.0])
        self.score = 0
        
    def handle_input(self):
        keys = pygame.key.get_pressed()
        
        # Update velocity based on input
        acceleration = 0.5
        drag = 0.95
        
        if keys[pygame.K_LEFT]:
            self.velocity[0] -= acceleration
        if keys[pygame.K_RIGHT]:
            self.velocity[0] += acceleration
        if keys[pygame.K_UP]:
            self.velocity[1] -= acceleration
        if keys[pygame.K_DOWN]:
            self.velocity[1] += acceleration
            
        # Apply drag
        self.velocity *= drag
        
    def update(self):
        # Update agent position
        x, y = self.agent(self.velocity)
        
        # Check room collisions
        self.velocity, collision = self.room(x, y, self.velocity)
        
        # Check reward collisions
        for reward in self.rewards:
            if reward(self.agent.rect.center):
                self.score += 1
                logger.info(f"Score: {self.score}")
                
    def render(self):
        self.screen.fill(WHITE)
        
        # Render game objects
        self.room.render(self.screen)
        self.agent.render(self.screen)
        for reward in self.rewards:
            reward.render(self.screen)
            
        # Render score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, BLACK)
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        
    def run(self):
        running = True
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            self.handle_input()
            self.update()
            self.render()
            self.clock.tick(FPS)
            
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()
