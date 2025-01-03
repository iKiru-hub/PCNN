import pygame
import numpy as np
import logging
from objects import AgentBody, RewardObj, RandomAgent
from envs import make_room, ROOM_LIST
from constants import *

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class Game:

    def __init__(self, tot: int = 10):

        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH,
                                               SCREEN_HEIGHT))
        pygame.display.set_caption("Simple Pygame Game")
        self.clock = pygame.time.Clock()

        # Create game objects
        self.setup_game(total_agents=tot)

    def setup_game(self, total_agents: int):

        self.room_name = np.random.choice(ROOM_LIST)
        # self.room = make_room(name="Hole.v0")
        self.room = make_room(name=self.room_name)

        # Create agent with slower max speed
        pos = np.array([SCREEN_WIDTH//2, SCREEN_HEIGHT//2])
        self.agent = AgentBody(position=pos,
                               width=25, height=25,
                               max_speed=4.0)

        self.agent_list = []
        self.scores = np.zeros(total_agents)
        self.names = []
        for _ in range(total_agents):
            pos = np.array([np.random.randint(200, SCREEN_WIDTH-200),
                   np.random.randint(200, SCREEN_HEIGHT-200)])
            self.agent_list += [AgentBody(pos, max_speed=4.0,
                    random_brain=RandomAgent(
                        change_interval=30, max_speed=3.0),
                color=tuple(random.randint(0, 255) for _ in range(3)))]
            self.names += ["".join([np.random.choice([
                'red', 'car', 'dog', 'cat', 'bird', 'fish',
                'thr', 'bob', 'jim', 'tim', 'sam', 'tom',
                'jane', 'sue', 'liz', 'lou', 'ann', 'eve',
                'lucy', 'mary', 'sara', 'jill', 'june',
            ]), str(np.random.randint(0, 99))])]

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
                reward.set_position()

        # Update agent list
        for i, agent in enumerate(self.agent_list):
            agent(None, self.room)
            for reward in self.rewards:
                if reward(agent.rect.center):
                    reward.set_position()
                    self.scores[i] += 1

        self.t += 1

        if self.t % 100 == 0:
            self.room_name = np.random.choice(ROOM_LIST)
            self.room = make_room(name=self.room_name)


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
        name = f"[{self.room_name}] | "
        name += f"user: {self.score} | "
        name += f"{self.names[self.scores.argmax()]}:"
        name += f" {self.scores.max()}"
        score_text = font.render(name, True, BLACK)
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


class GameSingle:

    def __init__(self, room_name: str):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH,
                                               SCREEN_HEIGHT))
        pygame.display.set_caption("Simple Pygame Game")
        self.clock = pygame.time.Clock()

        # Create game objects
        self.setup_game(room_name)

    def setup_game(self, room_name: str):

        # self.room = make_room(name="Hole.v0")
        self.room = make_room(name=room_name)

        # Create agent with slower max speed
        pos = np.array([SCREEN_WIDTH//2, SCREEN_HEIGHT//2])
        self.agent = AgentBody(pos, max_speed=4.0,
                            random_brain=RandomAgent(
                        change_interval=30, max_speed=3.0),
                            color=tuple(
                    random.randint(0, 255) for _ in range(3)))

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

        self.agent(None, self.room)
        for reward in self.rewards:
            reward(self.agent.rect.center)

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

        self.agent.render(self.screen)

        # Render score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}',
                                 True, BLACK)
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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", type=str, default="Flat.a")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--total", type=int, default=10)

    args = parser.parse_args()

    if args.single:
        game = GameSingle(args.room)
    else:
        game = Game(tot=args.total)

    game.run()
