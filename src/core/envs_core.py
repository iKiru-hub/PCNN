import pygame
import numpy as np
from tools.utils import logger
import graphics as grph


""" CONSTANTS """

SEED = None

# Define colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)

# MAIN SCREEN
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = WHITE

# ANALSYS SCREEN
SCREEN_ANALYSIS_WIDTH = 300
SCREEN_ANALYSIS_HEIGHT = 300
ANALYSIS_BACKGROUND_COLOR = WHITE



""" CLASSES """

class Wall:
    def __init__(self, x: int, y: int,
                 width: int, height: int,
                 thickness: int=5,
                 color: tuple=BLACK):

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

    def render(self, screen: object):
        pygame.draw.rect(screen, self.color,
                         self.rect, self.thickness)


class Room:
    def __init__(self, walls: list=[],
                 bounds: tuple=(0, 0, SCREEN_WIDTH,
                                SCREEN_HEIGHT),
                 bounce_coeff: float=1.0,
                 name: str="Room"):
        self.walls = walls
        self.num_walls = len(walls)
        self.bounce_coeff = bounce_coeff
        self.wall_vectors = self._make_wall_vectors()

    def __str__(self):
        return f"Room(#walls{self.num_walls})"

    def _make_wall_vectors(self):
        wall_vectors = []
        for wall in self.walls:
            wall_vectors.append(wall._wall_vectors)
        return np.stack(wall_vectors)

    def _check_bounds(self, x: int, y: int):
        x = max(0, min(x, SCREEN_WIDTH))
        y = max(0, min(y, SCREEN_HEIGHT))
        return x, y

    def is_out_of_bounds(self, x: int, y: int):
        ans = x < 0 or x > SCREEN_WIDTH or \
               y < 0 or y > SCREEN_HEIGHT
        if ans:
            logger.error(f"Out of bounds: {x}, {y}")

        return ans

    def check_collision(self, x: int, y: int,
                        velocity: np.array
                        ) -> bool:
        collision = False
        for wall in self.walls:
            if wall.rect.collidepoint(x + velocity[0], y):
                velocity[0] *= - 1
                collision = True
            if wall.rect.collidepoint(x, y + velocity[1]):
                velocity[1] *= - 1
                collision = True

        return velocity, collision

    def add_wall(self, wall: object):
        self.walls.append(wall)
        self.num_walls += 1
        self.wall_vectors = self._make_wall_vectors()

    def render(self, screen: object):
        for wall in self.walls:
            wall.render(screen)


class AgentBody:

    """
    rigid body
    """

    def __init__(self, brain: object=None,
                 x: int=None, y: int=None, radius=10,
                 color: str=ORANGE, render_freq=100):

        if x is None or y is None:
            self.x = np.random.randint(10, SCREEN_WIDTH-10)
            self.y = np.random.randint(10, SCREEN_HEIGHT-10)
        else:
            self.x = x
            self.y = y

        logger.debug(f"AgentBody({self.x}, {self.y})")

        self.radius = radius
        self.color = color
        self.velocity = np.array([2, 3])
        self.collision = False

        self.brain = brain

        self.render_freq = render_freq
        self.t = 0

        # Example usage:
        self.win = grph.GraphWin("Bar Plot", 600, 500)

    def __str__(self):
        return f"AgentBody({self.x}, {self.y})"

    def __call__(self, room: Room):

        self.t += 1

        # Update position based on velocity
        new_x = self.x + 2*self.velocity[0]
        new_y = self.y + 2*self.velocity[1]

        # Check for collision with walls and bounce
        self.velocity, collision = room.check_collision(
                                x=self.x, y=self.y,
                                velocity=self.velocity)

        if collision:
            logger(f">| Collision {self.velocity}")
            new_x = self.x + self.velocity[0] * room.bounce_coeff
            new_y = self.y + self.velocity[1] * room.bounce_coeff

        if not room.is_out_of_bounds(new_x, new_y):
            self.x = new_x
            self.y = new_y

        # brain step
        if self.brain is not None:
            observation = {
                "position": np.array(self._scale_to_01(
                        self.x, self.y)),
                "velocity": self.velocity,
                "collision": collision
            }

            self.velocity = self.brain(observation=observation)
            self.velocity = self._scale_to_screen(self.velocity)
            self.brain.routines(wall_vectors=room.wall_vectors)

    def _scale_to_01(self, x: int, y: int):

        x = x / SCREEN_WIDTH
        y = y / SCREEN_HEIGHT

        return x, y

    def _scale_to_screen(self, position: np.array):

        position[0] = position[0] * SCREEN_WIDTH
        position[1] = position[1] * SCREEN_HEIGHT

        return position

    def render(self, screen: object):

        pygame.draw.circle(screen, self.color,
                           (int(self.x), int(self.y)),
                           self.radius)

        logger(f"position: {self.x:04.0f}, {self.y:04.0f}")

        if self.t % self.render_freq == 0:
            return

        if self.brain is not None:
            data, names = self.brain.render_values

            data = np.where(np.isnan(data), 0, data)
            # self.brain.render()
            # render_bar_plot(data=data, names=names,
            #                 screen=screen,
            #                 bar_color=(0, 0, 255),
            #                 bg_color=(255, 255, 255),
            #                 label_color=(0, 0, 0),
            #                 font_size=24,
            #                 margin=10)

            draw_bar_plot(self.win, data, names)
            # self.win.getMouse()  # Wait for mouse click to close



""" other functions """


def set_seed(seed: int=None):
    if seed is not None:
        np.random.seed(seed)
        logger(f"seed set to {seed}")


def setup_room(name: str=None, thickness: int=5,
               bounce_coeff: float=1.0):

    sq_walls = [
        Wall(0, 0, SCREEN_WIDTH, thickness),
        Wall(0, SCREEN_HEIGHT - thickness,
             SCREEN_WIDTH, thickness),
        Wall(0, 0, thickness, SCREEN_HEIGHT),
        Wall(SCREEN_WIDTH - thickness, 0,
             thickness, SCREEN_HEIGHT),
    ]

    room = Room(walls=sq_walls,
                bounce_coeff=bounce_coeff)

    if name == "square" or name is None:
        pass

    elif name == "room_theeth":
        room.add_wall(Wall(2*SCREEN_WIDTH//3, 0, 5, 400))
        room.add_wall(Wall(SCREEN_WIDTH//3, 0, 5, 400))

    else:
        raise ValueError("Invalid room name")

    return room


class Randy:

    def __call__(self, observation: dict):

        # Random action
        action = observation["velocity"] * (1 + \
            np.random.uniform(-0.01, 0.01, 2))
        return action

    def render(self):
        pass


def render_bar_plot(data: list,
                    names: list,
                    screen: object, **kwargs):

    # Extract optional parameters or set defaults
    bar_color = kwargs.get('bar_color', (0, 0, 255))
    bg_color = kwargs.get('bg_color', (255, 255, 255))
    label_color = kwargs.get('label_color', (0, 0, 0))
    font_size = kwargs.get('font_size', 24)
    margin = kwargs.get('margin', 10)

    # Calculate bar dimensions
    bar_width = (SCREEN_ANALYSIS_WIDTH - 2 * margin) // len(data)
    max_value = max(data)

    # Draw bars
    for i, value in enumerate(data):
        value = max(value, 1e-6)
        if np.isnan(value):
            value = 0
        bar_height = int((value / max_value) * (
            SCREEN_ANALYSIS_HEIGHT - 2 * margin))
        x = SCREEN_ANALYSIS_HEIGHT - (margin + i * bar_width)
        y = SCREEN_ANALYSIS_HEIGHT - margin - bar_height
        pygame.draw.rect(screen, bar_color,
                         (x, y,
                          bar_width - margin, bar_height))

        # Draw labels
        font = pygame.font.Font(None, font_size)
        label = font.render(names[i], True, label_color)
        screen.blit(label, (x + (bar_width - \
            margin) // 2 - label.get_width() // 2,
                            SCREEN_ANALYSIS_HEIGHT - margin + 5))



def draw_bar_plot(win, values, names, bar_width=40, gap=20):

    """
    Draws a bar plot on the given Graphics window.

    Parameters:
    win        -- The Graphics window to draw on.
    values     -- List of values (heights of the bars).
    names      -- List of names corresponding to each bar.
    bar_width  -- Width of each bar (default 40).
    gap        -- Gap between each bar (default 20).
    """
    # Check that values and names have the same length
    if len(values) != len(names):
        raise ValueError("Length of values and names must be the same.")

    # Calculate scaling and max height
    max_value = 1.
    plot_height = 400  # height of the plot area
    plot_width = (bar_width + gap) * len(values)  # width of the plot area
    scale = 200

    # Set up the window size
    win.setCoords(0, 0, plot_width, plot_height)

    # clear the window
    win.delete("all")

    # set background color
    win.setBackground("white")

    # Draw bars
    base_height = 100
    for i, value in enumerate(values):
        bar_height = value * scale
        x_left = i * (bar_width + gap)
        x_right = x_left + bar_width
        bar = grph.Rectangle(grph.Point(x_left, base_height),
                             grph.Point(x_right, bar_height + base_height))
        bar.setFill("blue")
        bar.draw(win)

        # Draw value label on top of each bar
        # value_text = grph.Text(grph.Point((x_left + x_right) / 2, bar_height + 10), str(value))
        # value_text.draw(win)

        # Draw name label below each bar
        name_text = grph.Text(grph.Point((x_left + x_right) / 2, 15),
                              f"{names[i]}\n{value:.2f}")
        name_text.draw(win)


""" initialize """



# Main loop
def main(agent: AgentBody, room_name: str=None,
         duration: int=100000):

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH,
                                      SCREEN_HEIGHT))
    # screen_analysis = pygame.display.set_mode((
    #     SCREEN_ANALYSIS_WIDTH, SCREEN_ANALYSIS_HEIGHT))
    clock = pygame.time.Clock()

    room = setup_room(name=room_name,
                      bounce_coeff=2.0)
    pygame.display.set_caption(f"{room}     {agent}")


    logger()
    logger("<Starting simulation>")

    running = True
    t = 0
    for t in range(duration):
        screen.fill(BACKGROUND_COLOR)  # Clear screen

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- logic
        agent(room=room)

        # --- render
        room.render(screen=screen)
        agent.render(screen=screen)

        # - #
        pygame.display.flip()
        clock.tick(400)

    logger()
    logger(f"<Simulation ended at t={t}>")
    pygame.quit()





if __name__ == "__main__":

    agent = AgentBody(brain=Randy())
    main(agent=agent, room_name="room_theeth")

