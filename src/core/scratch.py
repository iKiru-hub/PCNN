import pygame
import math

# Initialize Pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Define screen size
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

class Wall:
    def __init__(self, x, y, width, height, thickness=5):
        self.rect = pygame.Rect(x, y, width, height)
        self.thickness = thickness

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect, self.thickness)

class Room:
    def __init__(self):
        self.walls = []

    def add_wall(self, wall):
        self.walls.append(wall)

    def draw(self, screen):
        for wall in self.walls:
            wall.draw(screen)

class Agent:
    def __init__(self, x, y, radius=10):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = BLUE
        self.velocity = pygame.math.Vector2(2, 3)  # Start with an initial velocity

    def update(self, room):
        # Update position based on velocity
        self.x += self.velocity.x
        self.y += self.velocity.y

        # Check for collision with walls and bounce
        for wall in room.walls:
            if wall.rect.collidepoint(self.x + self.velocity.x, self.y):
                self.velocity.x *= -1  # Bounce horizontally
            if wall.rect.collidepoint(self.x, self.y + self.velocity.y):
                self.velocity.y *= -1  # Bounce vertically

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

# Main loop
def main():
    clock = pygame.time.Clock()
    room = Room()

    # Define walls (left, right, top, bottom)
    room.add_wall(Wall(0, 0, SCREEN_WIDTH, 5))            # Top wall
    room.add_wall(Wall(0, SCREEN_HEIGHT - 5, SCREEN_WIDTH, 5))  # Bottom wall
    room.add_wall(Wall(0, 0, 5, SCREEN_HEIGHT))           # Left wall
    room.add_wall(Wall(SCREEN_WIDTH - 5, 0, 5, SCREEN_HEIGHT))  # Right wall
    room.add_wall(Wall(2*SCREEN_WIDTH//3, 0, 5, 400))  # Right wall
    room.add_wall(Wall(SCREEN_WIDTH//3, 0, 5, 400))  # Right wall

    # add wall in the middle
    # room.add_wall(Wall(SCREEN_WIDTH // 2 - 50,
    #                    SCREEN_HEIGHT // 2 - 50,
    #                    100, 100))

    agent = Agent(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

    running = True
    while running:
        screen.fill((0, 0, 0))  # Clear screen

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        agent.update(room)
        room.draw(screen)
        agent.draw(screen)

        pygame.display.flip()
        clock.tick(600)

    pygame.quit()

if __name__ == "__main__":
    main()

