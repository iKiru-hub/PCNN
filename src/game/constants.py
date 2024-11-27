
import random

""" screen constants """

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
OFFSET = 50
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLOR1 = tuple(random.randint(0, 255) for _ in range(3))
COLOR2 = tuple(random.randint(0, 255) for _ in range(3))
COLOR3 = tuple(random.randint(0, 255) for _ in range(3))

""" game constants """

# Increase for more precision, decrease
# for better performance
NUM_STEPS = 10
