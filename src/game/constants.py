
import random

""" screen constants """

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
OFFSET = 50
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLOR1 = tuple(random.randint(0, 255) for _ in range(3))
COLOR2 = tuple(random.randint(0, 255) for _ in range(3))
COLOR3 = tuple(random.randint(0, 255) for _ in range(3))

PURPLE = (255, 0, 255)

""" game constants """

# Increase for more precision, decrease
# for better performance
NUM_STEPS = 10
GAME_SCALE = SCREEN_WIDTH


""" rooms """

ROOMS = ["Square.v0", "Square.v1", "Square.v2",
         "Hole.v0", "Flat.0000", "Flat.0001",
         "Flat.0010", "Flat.0011", "Flat.0110",
         "Flat.1000", "Flat.1001", "Flat.1010",
         "Flat.1011", "Flat.1110"]
