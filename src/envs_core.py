import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numba import jit
import argparse

from utils_core import setup_logger


logger = setup_logger(name="ENVS",
                      level=1,
                      is_debugging=True,
                      is_warning=True)

def set_logger_level(level: int):
    logger.set_level(level)
    logger(f"Logger level set to {level}", level=-1)


""" ENVIRONMENT logic """


class Wall:
    def __init__(self, point: np.ndarray,
                 orientation: str, length: float = 1., **kwargs):
        self.point = np.array(point)
        self.length = length
        self.thickness = kwargs.get('thickness', 1.)
        self.orientation = orientation
        self._wall_angle = 0. if orientation == "horizontal" else np.pi/2
        self._wall_vector = np.array([
            self.point,
            self.point + np.array([np.cos(self._wall_angle) * \
                length, np.sin(self._wall_angle) * length])
        ])
        self._color = kwargs.get('color', 'black')
        self._bounce_coefficient = kwargs.get('bounce_coefficient', 1.)

    def vector_collide(self, vector: np.ndarray) -> bool:
        """ check if a given [velocity] vector intersects with the wall """

        return segments_intersect(self._wall_vector[0], self._wall_vector[1],
                                  vector[0], vector[1])

    def collide(self, position: np.ndarray, velocity: np.ndarray,
                radius: float) -> tuple:
        """Check if a circle collides with the wall and return the new velocity and collision angle"""

        # adjust radius of the object with the wall thickness
        # radius += self.thickness

        # Calculate the nearest point on the wall to the circle's center
        wall_vector = self._wall_vector[1] - self._wall_vector[0]
        wall_length = np.linalg.norm(wall_vector)
        wall_unit = wall_vector / wall_length

        relative_position = position - self._wall_vector[0]
        projection = np.dot(relative_position, wall_unit)
        projection = np.clip(projection, 0, wall_length)

        nearest_point = self._wall_vector[0] + projection * wall_unit

        # Check if the circle intersects with the wall
        distance_to_wall = np.linalg.norm(position - nearest_point)
        if distance_to_wall > radius:
            return None, None

        # Calculate reflection
        normal = (position - nearest_point) / distance_to_wall
        reflection = velocity - 2 * np.dot(velocity, normal) * normal
        new_velocity = reflection * self._bounce_coefficient

        # Calculate angle
        angle = np.arctan2(new_velocity[1], new_velocity[0])

        return new_velocity, angle

    def render(self, ax: plt.Axes,
               alpha: float=1.):

        ax.plot([self._wall_vector[0][0],
                 self._wall_vector[1][0]],
                [self._wall_vector[0][1],
                 self._wall_vector[1][1]],
                color=self._color, alpha=alpha,
                lw=self.thickness)


class Room:

    def __init__(self, walls: list, **kwargs):

        self.walls = walls
        # self.bounds = kwargs.get("bounds", [0, 1, 0, 1])

        wdx = 0.05
        self.bounds = kwargs.get("bounds", [wdx, 1.-wdx,
                                            wdx, 1.-wdx])
        self.name = kwargs.get("name", "Base")

        self.nb_collisions = 0
        self.wall_vectors = np.stack([wall._wall_vector for wall in self.walls])
        self.visualize = kwargs.get("visualize", False)
        if self.visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 4))

    def __repr__(self):
        return "Room.{}(#walls{})".format(self.name, len(self.walls))

    def check_bounds(self, position: np.ndarray,
                     radius: float,
                     velocity: np.ndarray=None):

        beyond = False
        if position[0] < self.bounds[0] + radius or \
            position[0] > self.bounds[1] - radius:
            if velocity is not None:
                velocity[0] = -velocity[0]
            beyond = True
        if position[1] < self.bounds[2] + radius or \
            position[1] > self.bounds[3] - radius:
            if velocity is not None:
                velocity[1] = -velocity[1]
            beyond = True

        return beyond, velocity

    def handle_collision(self, position: np.ndarray,
                         velocity: np.ndarray,
                         radius: float,
                         stop: bool=False) -> np.ndarray:

        beyond, new_velocity = self.check_bounds(
            position=position, radius=radius,
            velocity=velocity)

        if beyond:
            # logger.error(f"OUT OF THE BORDERS")
            self._self_check(position, velocity, radius,
                             stop)
            return new_velocity, None, True

        for wall in self.walls:
            new_velocity, angle = wall.collide(position,
                                               velocity,
                                               radius)
            if new_velocity is not None:
                self._self_check(position, velocity, radius,
                                 stop)
                return new_velocity, angle, True

        self._self_check(position, velocity, radius, stop)
        return velocity, None, False

    def _self_check(self, position, velocity, radius,
                    stop: bool=False):

        if stop: return

        beyond, new_velocity = self.check_bounds(
            position=position+velocity, radius=0.,
            velocity=velocity)

        # if beyond:
        #     logger.error(f"DOOMED TO COLLIDE")
            # logger.error(f"[p:{np.around(position, 3)}, v:{np.around(velocity, 3)}]")

    def handle_vector_collision(self, vector: np.ndarray):
        for wall in self.walls:
            if wall.vector_collide(vector):
                return True
        return False

    def render(self, ax: plt.Axes=None,
               alpha: float=1.,
               returning: bool=False):

        if not self.visualize:
            return

        ax_provided = ax is not None
        if ax is None:
            ax = self.ax
            ax.clear()

        ax.set_xlim(self.bounds[0],
                    self.bounds[1])
        ax.set_ylim(self.bounds[2],
                    self.bounds[3])
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.axis('off')
        ax.set_aspect('equal')
        for wall in self.walls:
            wall.render(ax=ax, alpha=alpha)

        if not ax_provided:
            self.fig.canvas.draw()

        if returning:
            return ax, self.fig


    def reset(self):
        self.nb_collisions = 0


def make_room(name: str="square", thickness: float=1.,
              bounds: list=[0, 1, 0, 1],
              visualize: bool=False):

    walls = [Wall([bounds[0], bounds[2]],
                  orientation="horizontal",
                  length=bounds[1]-bounds[0],
                 thickness=thickness),
            Wall([bounds[0], bounds[0]],
                 orientation="vertical",
                 length=bounds[3]-bounds[2],
                 thickness=thickness),
            Wall([bounds[0], bounds[3]],
                 orientation="horizontal",
                 length=bounds[1]-bounds[0],
                 thickness=thickness),
            Wall([bounds[1], bounds[2]],
                 orientation="vertical",
                 length=bounds[3]-bounds[2],
                 thickness=thickness),
    ]

    if name == "square":
        name = "square"
    elif name == "square1":
        walls += [
            Wall([0, bounds[1]],
                 orientation="horizontal",
                 length=0.5, thickness=thickness),
        ]
        name = "square1"
        room = Room(walls=walls, name="Square1",
                    bounds=bounds,
                    visualize=visualize)
    elif name == "square2":
        walls += [
            Wall([0.5, 0.], orientation="vertical",
                 length=0.5, thickness=thickness),
            # Wall([0.5, 0.5], orientation="horizontal",
            #      length=0.5, thickness=thickness),
        ]
        name = "square2"
    elif name == "flat":
        walls += [
            Wall([0., 0.33], orientation="horizontal",
                    length=0.6, thickness=thickness),
            Wall([0., 0.66], orientation="horizontal",
                    length=0.6, thickness=thickness),
        ]
        name = "flat"
    elif name == "flat2":
        walls += [
            Wall([0., 0.33], orientation="horizontal",
                    length=0.6, thickness=thickness),
            Wall([0., 0.66], orientation="horizontal",
                    length=0.6, thickness=thickness),
            Wall([0.6, 0.], orientation="vertical",
                    length=0.115, thickness=thickness),
            Wall([0.6, 0.215], orientation="vertical",
                    length=0.215, thickness=thickness),
            Wall([0.6, 0.555], orientation="vertical",
                    length=0.215, thickness=thickness),
            Wall([0.6, 0.875], orientation="vertical",
                    length=0.115, thickness=thickness),
        ]
        name = "flat2"
    elif name == "hole1":
        walls += [
            Wall([0.33, 0.33], orientation="horizontal",
                    length=0.33, thickness=thickness),
            Wall([0.33, 0.33], orientation="vertical",
                    length=0.33, thickness=thickness),
            Wall([0.66, 0.33], orientation="vertical",
                    length=0.33, thickness=thickness),
            Wall([0.33, 0.66], orientation="horizontal",
                    length=0.33, thickness=thickness),
        ]
        name = "hole1"
    else:
        raise NameError("'{}' is not a room".format(name))

    room = Room(walls=walls, name=name,
                bounds=bounds,
                visualize=visualize)

    return room



""" AGENT """


class AgentBody:

    def __init__(self, room: Room,
                 position: np.ndarray = None,
                 **kwargs):
        self.radius = kwargs.get("radius", 0.05)
        self.position = position if position is not None else self._random_position()
        self.prev_position = self.position.copy()
        self.velocity = np.zeros(2).astype(float)
        self.color = kwargs.get("color", "red")
        self._room = room
        self.verbose = kwargs.get("verbose", False)
        self.bounce_coefficient = kwargs.get("bounce_coefficient", 0.5)

        self.visualize = kwargs.get("visualize", False)

    def _random_position(self):
        return np.random.rand(2)

    def __call__(self, velocity: np.ndarray):

        self.velocity = velocity
        self.velocity, collision = self._handle_collisions()

        # update position
        # + considering a possible collision
        self.prev_position = self.position.copy()
        self.position += self.velocity * (1 + \
                            self.bounce_coefficient * \
                            1 * collision)

        # if collision:
        #     logger.debug(f"new velocity: {np.around(self.velocity, 3)}")
        #     logger.debug(f"new position: {np.around(self.position, 3)}")

        truncated = not self._room.check_bounds(
                                position=self.position,
                                radius=self.radius*0.2)

        return self.position.copy(), collision, truncated

    def _handle_collisions(self) -> tuple:
        new_velocity, _, collision = self._room.handle_collision(
            self.position, self.velocity, self.radius)
        if collision:
            self.velocity = new_velocity
            # Move the agent slightly after collision to prevent sticking
            self._room.nb_collisions += 1

            if self.verbose:
                logger.debug("%collision detected%")

        return new_velocity, collision

    def set_position(self, position: np.ndarray):
        self.position = position

    def render(self, ax: plt.Axes=None,
               velocity: np.ndarray=None):

        """
        Plot the agent in the room

        Parameters
        ----------
        ax : plt.Axes
            axis to plot the agent.
            Default is None.
        velocity : np.ndarray
            velocity vector to plot the agent's movement.
            Default is None.
        """

        if not self.visualize:
            return

        ax, fig = self._room.render(ax=ax, returning=ax is None)

        ax.add_patch(Circle(self.prev_position, self.radius,
                            fc=self.color, ec='black'))

        displacement = self.position - self.prev_position
        ax.arrow(self.prev_position[0], self.prev_position[1],
                 displacement[0], displacement[1],
                 head_width=0.02, head_length=0.02,
                 fc='black', ec='black')

        if fig is not None:
            fig.canvas.draw()


class Zombie:

    def __init__(self, body: AgentBody,
                 speed: float = 0.1,
                 visualize: bool = False):

        self.body = body
        self.speed = speed

        self.p = 0.5
        self.velocity = np.zeros(2)
        self.visualize = visualize
        self.position = self.body.position

    def __call__(self, **kwargs):

        self.p += (0.2 - self.p) * 0.02
        self.p = np.clip(self.p, 0.01, 0.99)

        if np.random.binomial(1, self.p):
            angle = np.random.uniform(0, 2*np.pi)
            self.velocity = self.speed * np.array([np.cos(angle),
                                              np.sin(angle)])
            self.p *= 0.2

        self.position, collision, _ = self.body(velocity=self.velocity)

        if collision:
            self.velocity = -self.velocity

        return self.velocity, collision

    def render(self, ax: plt.Axes=None):

        if not self.visualize:
            return

        self.body.render(velocity=self.velocity)


class RewardObj:

    def __init__(self, position: np.ndarray,
                 radius: float=0.05,
                 fetching: str="probabilistic",
                 behaviour: str="static",
                 bounds: list=[0, 1, 0, 1]):

        self._position = position
        self._radius = radius
        self._fetching = fetching
        self._behaviour = behaviour
        self._bounds = bounds
        self._count = 0

    def __repr__(self):
        return f"Reward({self._fetching}, {self._behaviour})"

    def __call__(self, position: np.ndarray):

        """
        assuming a box of size 1x1
        """

        # distance = np.linalg.norm(position - self._position)
        distance = ((position - self._position)**2).sum()
        p = np.exp(-distance / (2 * self._radius**2))
        result = 0.
        if distance < self._radius:
            if self._fetching == "deterministic":
                result = 1.0
            else:
                result = np.random.binomial(1, p)

        if result:
            self._count += 1
            if self._behaviour == "dynamic":
                self.reset()

        return result

    def get_count(self):
        return self._count

    def reset(self, new_position: np.ndarray=None):
        if new_position is None:
            new_position = np.array([
                np.random.uniform(self._bounds[0], self._bounds[1]),
                np.random.uniform(self._bounds[2], self._bounds[3])
            ])
        self._position = new_position

    def render(self, ax: plt.Axes, alpha: float=0.25):

        ax.add_patch(Circle(self._position, self._radius,
                            fc="green", ec='black',
                            alpha=alpha))



""" UTILS """

@jit(nopython=True)
def action_to_angle(value: float) -> float:

    """
    map an action value (float in [-1, 1])
    to and angle in radians (float in [0, 2*pi])
    """

    return (value + 1) * np.pi

@jit(nopython=True)
def two_lines_intersection(p1, p2, p3, p4):
    """
    Check if two lines intersect, where
    (p1, p2) is the first line and (p3, p4) is the second line.
    Returns the point of intersection if the lines intersect, None otherwise.
    """
    (x11, y11), (x12, y12) = p1, p2
    (x21, y21), (x22, y22) = p3, p4

    # Calculate the direction of the lines
    dx1 = x12 - x11
    dy1 = y12 - y11
    dx2 = x22 - x21
    dy2 = y22 - y21

    # Calculate the determinant
    determinant = dx1 * dy2 - dy1 * dx2

    # If the determinant is zero, the lines are parallel or coincident
    if determinant == 0:
        return None

    # Calculate the intersection point
    t1 = ((x21 - x11) * dy2 - (y21 - y11) * dx2) / determinant
    t2 = ((x21 - x11) * dy1 - (y21 - y11) * dx1) / determinant

    # Check if the intersection point lies on both line segments
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        x = x11 + t1 * dx1
        y = y11 + t1 * dy1
        return np.array([x, y])
    else:
        return None

@jit(nopython=True)
def calc_angle(vector: np.ndarray) -> float:

        if vector[1, 0] - vector[0, 0] == 0:
            if vector[1, 1] - vector[0, 1] > 0:
                return np.pi/2
            return 3*np.pi/2

        angle = np.arctan(
            (vector[1, 1] - vector[0, 1]) / \
                (vector[1, 0] - vector[0, 0])
        )

        if angle < 0:
            angle += np.pi

        return angle

@jit(nopython=True)
def calc_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors with the same origin.

    Parameters
    ----------
    v1 : np.ndarray
        The first vector of shape (2, 2).
    v2 : np.ndarray
        The second vector of shape (2, 2).

    Returns
    -------
    float
        The angle between the two vectors in radians.
    """
    # Extract direction vectors
    dir_v1 = v1[1] - v1[0]
    dir_v2 = v2[1] - v2[0]
    
    # Calculate the dot product of the direction vectors
    dot_product = np.dot(dir_v1, dir_v2)
    
    # Calculate the magnitudes of the direction vectors
    norm_v1 = np.linalg.norm(dir_v1)
    norm_v2 = np.linalg.norm(dir_v2)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Ensure the cosine value is within the valid range [-1, 1] to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle = np.arccos(cos_theta)
    
    return angle

@jit(nopython=True)
def vector_norm(vector: np.ndarray):
    return np.sqrt((vector[0, 0] - vector[1, 0])**2 +
                   (vector[0, 1] - vector[1, 1])**2)

@jit(nopython=True)
def reflect_point(point1: np.ndarray,
                  point2: np.ndarray) -> tuple:
    """
    reflect a point (point1) wrt an origin (point2)
    """
    return (point2[0] + (point2[0] - point1[0]),
            point2[1] + (point2[1] - point1[1]))

@jit(nopython=True)
def reflect_vector(vector1: np.ndarray,
                   vector2: np.ndarray,
                   momentum_c: float=1.) -> np.ndarray:

    # Normalize vector2 to get the direction vector
    vector2_dir = vector2[1] - vector2[0]
    vector2_dir /= np.linalg.norm(vector2_dir)

    # Calculate the projection of vector1[1] onto vector2_dir
    projection_length = np.dot(vector1[1] - vector1[0], vector2_dir)
    projection = projection_length * vector2_dir

    # Calculate the reflection point
    reflection_point = 2 * projection - (vector1[1] - vector1[0])
    reflection_point = vector2[0] - momentum_c * reflection_point

    # Create the reflected vector
    reflected_vector = np.stack((vector2[0],
                                 reflection_point))


    return reflected_vector

@jit(nopython=True)
def check_nan_move(move: np.ndarray) -> bool:
    if move is None:
        return False
    return np.isnan(move).any()

@jit(nopython=True)
def line_intersection(p1, p2, p3, p4):

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denom == 0:
        return None  # Lines are parallel

    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom

    return (px, py)

@jit(nopython=True)
def is_point_on_segment(p, segment_start, segment_end):
    x, y = p
    x1, y1 = segment_start
    x2, y2 = segment_end

    # Check if the point is within the bounding box of the segment
    if (min(x1, x2) <= x <= max(x1, x2) and
        min(y1, y2) <= y <= max(y1, y2)):
        return True
    return False

@jit(nopython=True)
def segments_intersect(p1, p2, p3, p4):
    intersection = line_intersection(p1, p2, p3, p4)
    if intersection is None:
        return False  # Parallel or coincident segments

    # Check if the intersection point is on both segments
    return (is_point_on_segment(intersection, p1, p2) and
            is_point_on_segment(intersection, p3, p4))



""" MAIN """

if __name__ == "__main__":

    # --- ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", type=str, default="square",
                        help="room to use: 'square', 'square1'," + \
                        "'square2', 'flat', 'flat2', 'hole1'")
    parser.add_argument("--demo", action="store_true",
                        help="run the demo with a zombie agent")

    args = parser.parse_args()

    # --- INITIALIZATION

    room = make_room(name=args.room, thickness=2.,
                     bounds=[0, 2, 0, 2],
                     visualize=True)
    env = AgentBody(room=room,
                    position=np.random.uniform(0.1, 0.9, 2),
                    radius=0.05, speed=0.01, color="red",
                    visualize=True)
    env = Zombie(body=env, speed=0.03, visualize=True)

    # --- RUN
    if args.demo:

        logger("Running the demo", level=1)

        while True:

            env()
            room.render()
            env.render()
            plt.pause(0.001)

    else:
        env.render()
        plt.show()


