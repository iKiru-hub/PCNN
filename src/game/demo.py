import pygame
import numpy as np
import logging
from objects import AgentBody, Reward, RandomAgent
from envs import make_room
from constants import *

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Game:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Simple Pygame Game")
        self.clock = pygame.time.Clock()

        # Create game objects
        self.setup_game()

    def setup_game(self):

        self.room = make_room(name="Hole.v0")

        # Create agent with slower max speed
        pos = np.array([SCREEN_WIDTH//2, SCREEN_HEIGHT//2])
        self.agent = AgentBody(position=pos,
                               width=25, height=25,
                               max_speed=4.0)

        self.agent_list = []
        for _ in range(10):
            pos = np.array([np.random.randint(200, SCREEN_WIDTH-200),
                   np.random.randint(200, SCREEN_HEIGHT-200)])
            self.agent_list += [AgentBody(pos, max_speed=4.0,
                    random_brain=RandomAgent(
                        change_interval=30, max_speed=3.0),
                color=tuple(random.randint(0, 255) for _ in range(3)))]

        # Create rewards
        self.rewards = [
            RewardObj(position=np.array([110, 130])),
            RewardObj(position=np.array([200, 400])),
            RewardObj(position=np.array([500, 300])),
        ]


        # Game state
        self.velocity = np.array([0.0, 0.0])
        self.score = 0
        self.t = 0

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

        # # Update agent position with improved collision handling
        self.velocity, collision = self.agent(self.velocity,
                                              self.room)

        # # Check reward collisions
        for reward in self.rewards:
            if reward(self.agent.rect.center):
                self.score += 1
                logger.info(f"Score: {self.score}")

        # Update agent list
        for agent in self.agent_list:
            agent(None, self.room)
            for reward in self.rewards:
                reward(agent.rect.center)

        self.t += 1

        if self.t % 100 == 0:
            self.room.move_wall()

    def render(self):
        self.screen.fill(WHITE)

        # Render game objects
        self.room.render(self.screen)
        self.agent.render(self.screen)
        for reward in self.rewards:
            reward.render(self.screen)

        for agent in self.agent_list:
            agent.render(self.screen)

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
