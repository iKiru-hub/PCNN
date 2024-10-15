import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numba import jit
from abc import ABC, abstractmethod


class Wall:

    def __init__(self, p1: np.ndarray, p2: np.ndarray,
                 **kwargs):

        self.p1 = np.array(p1) if not isinstance(p1, np.ndarray) else p1 
        self.p2 = np.array(p2) if not isinstance(p2, np.ndarray) else p2

        self._color = kwargs.get('color', 'black')
        self._bounce_coefficient = kwargs.get('bounce_coefficient', 1.)

        self._wall_angle = calc_angle(vector=np.stack((self.p1, self.p2)))

    def __repr__(self):
        return f"Wall({self.p1}, {self.p2})"

    def _wall_vector(self, point: np.ndarray) -> np.ndarray:
        return np.stack((point, self.p2))

    def collide(self, velocity_vector: np.ndarray, radius: float) -> bool:
        """
        Check if a given velocity with a certain radius collides
        with the wall.

        Parameters
        ----------
        velocity_vector : np.ndarray
            The velocity of the object, as [[x1, y1], [x2, y2]]
        radius : float
            The radius of the object.

        Returns
        -------
        bool
            True if the object collides with the wall, False otherwise.
        """

        # check if the velocity is intersects the wall
        # aka: given two lines (4 points) check if they intersect
        intersection_point = two_lines_intersection(p1=self.p1,
                                                    p2=self.p2,
                                                    p3=velocity_vector[0],
                                        p4=velocity_vector[1])

        # print(velocity_vector, radius, self.p1, self.p2)
        # angle = calc_angle(vector=velocity_vector)
        # angle = self.adjust_angle(angle=angle)
        # print(f"calculated angle: {angle:.3f}°")

        # --- bounce ---
        if intersection_point is None:
            return None, None

        print(f">>> collision! [{self.p1}, {self.p2}] at {np.around(intersection_point, 4)}")

        # calculate the vector from the object position (first point 
        # of the velocity vector) to the intersection point
        intersection_vector = np.stack((velocity_vector[0],
                                        intersection_point))
        wall_vector = self._wall_vector(intersection_point)

        print(f"intersection_vector: {np.around(intersection_vector.flatten(), 3)}")
        print(f"wall_vector        : {np.around(wall_vector.flatten(), 3)}")

        # reflect vecto+radiusr
        reflected_vector = reflect_vector(vector1=wall_vector,
                            vector2=intersection_vector,
                            momentum_c=self._bounce_coefficient)

        print(">>> reflect ", np.around(reflected_vector.flatten(), 3))
        print(">>> input v ", np.around(velocity_vector.flatten(), 3))

        vangle = calc_angle(vector=velocity_vector)
        vangle = self.adjust_angle(angle=vangle, sign=np.sign(intersection_point[0]-velocity_vector[1, 0]))
        print(f"calculated angle: {vangle:.3f}°")

        # angle
        # angle = calc_angle(vector=reflected_vector)
        angle = calc_two_vector_angle(reflected_vector,
                                      wall_vector).item()

        print(f"relative angle: {angle:.3f}° [{self._wall_angle:.3f}°]")
        angle = self.adjust_angle(angle=angle, sign=np.sign(-intersection_point[0]+reflected_vector[0, 0]))

        print(f"reflected angle: {angle:.3f}°")

        input()
        # print(f"{reflected_vector[1]=}")

        return reflected_vector, angle

    def adjust_angle(self, angle: float, sign: 1) -> float:

        """
        correct the angle wrt to the wall
        """

        angle = sign * (self._wall_angle - angle)

        if angle < 0:
            angle = 2 * np.pi + angle

        return angle

    def draw(self, ax: plt.Axes, alpha: float=1.):
        ax.plot([self.p1[0], self.p2[0]],
                [self.p1[1], self.p2[1]], color=self._color, alpha=alpha)



class Room:
    """
    the abstract class for a room
    """
    def __init__(self, walls: list, **kwargs):

        self.walls = walls
        self.bounds = kwargs.get("bounds", (0, 1, 0, 1))
        self.bound_x = (self.bounds[0], self.bounds[1])
        self.bound_y = (self.bounds[2], self.bounds[3])

    def add_wall(self, wall: Wall):
        self.walls.append(wall)

    def handle_collision(self, velocity_vector: np.ndarray,
                         radius: float=0.) -> np.ndarray:

        """
        calculates whether an objects collides with a wall and returns a new position

        Parameters
        ----------
        position : np.ndarray
            The position of the object.
        velocity_vector : np.ndarray
            The velocity of the object as (tail, head)
        radius : float
            The radius of the object.
            Default is 0.

        Returns
        -------
        np.ndarray
            The new position of the object.
        """


        for wall in self.walls:
            collision_point, angle = wall.collide(
                velocity_vector=velocity_vector,
                radius=radius)

            # return at the first collision
            if collision_point is not None:
                print(f"angle from room: {angle:.3f}°")
                return collision_point, angle, True

        # no collision
        return velocity_vector[1], None, False

    def draw(self, ax: plt.Axes):
        for wall in self.walls:
            wall.draw(ax=ax)


class Reward:
    def __init__(self, position, value):
        self.position = np.array(position)
        self.value = value
        self.collected = False

    def check_collection(self, agent_pos, agent_radius):
        if not self.collected and np.linalg.norm(self.position - agent_pos) <= agent_radius:
            self.collected = True
            return self.value
        return 0


class AgentBody3:

    def __init__(self, x: float=None, y: float=None,
                 radius: float=0.1,
                 **kwargs):

        self.bounds = kwargs.get("bounds", (0, 1, 0, 1))
        self.x = np.random.uniform(self.bounds[0],
                                   self.bounds[1])
        self.y = np.random.uniform(self.bounds[2],
                                   self.bounds[3])
        self.color = kwargs.get("color", "black")
        self.radius = radius
        self.direction = 0 # angle in radians
        self.speed = 0.
        self.velocity_vector = np.stack((self.position(), self.position()))
        self.t = 0

    def _update_direction(self):
        dw = self.trg_direction - self.direction
        self.direction += dw * 0.1

        if dw < 0.05:
            self.trg_direction = random.uniform(0, 2*np.pi)

    def _update_speed(self):
        self.speed *= 1 + 7 * np.sin(self.t / self.speed_freq)

    def position(self):
        return np.array([self.x, self.y])

    def draw(self):
        pygame.draw.circle(self.screen, self.color,
                           (self.x, self.y), self.radius)

        # write the uid
        font = pygame.font.Font(None, 15)
        text = font.render(self.uid, True, (0, 0, 0))
        self.screen.blit(text, (self.x - 20, self.y -25))

    def move(self, direction: float, speed: float):

        x = self.x + speed * np.cos(direction) 
        y = self.y + speed * np.sin(direction) 

        self.velocity_vector = np.array([
            [self.x, self.y], [x, y]
        ])

        self.x = x
        self.y = y
        self.direction = direction
        self.speed = speed

        self.t += 1

    def bounce(self) -> float:

        # undow the last move
        self.y -= self.speed * np.sin(self.direction)
        self.x -= self.speed * np.cos(self.direction)

        # change direction
        self.direction = (self.direction + np.pi) % (2*np.pi)

        # new move in the opposite direction
        self.y += self.speed * np.sin(self.direction)
        self.x += self.speed * np.cos(self.direction)

        return self.direction

    def _check_collision(self, x: float, y: float) -> bool:
        distance = np.sqrt((self.x - x)**2 + \
            (self.y - y)**2)
        if distance < self.radius:
            return True
        else:
            return False

    def handle_collisions(self, objects: list):

        for obj in objects:
            if hasattr(obj, 'collide'):
                collision_point = obj.collide(self.velocity_vector,
                                              radius=self.radius)
                if collision_point is not None:
                    direction = self.bounce()
                    print(f"collision: {np.around(collision_point, 3)}, new={direction=:.2f}")
                    return direction

        return None

    def draw(self, ax: plt.Axes, alpha: float=1.):
        ax.add_patch(Circle(self.position(),
                            self.radius, fc=self.color, ec='black'))




class AgentBody:

    def __init__(self, position: np.ndarray=None, **kwargs):

        self.radius = 0.05# kwargs.get("radius", 0.1)
        self.bounds = kwargs.get("bounds", (0., 1., 0., 1.))
        self.speed = kwargs.get("speed", 0.1)
        self.position = np.array([np.random.uniform(self.bounds[0],
                                                    self.bounds[1]),
                                  np.random.uniform(self.bounds[2],
                                                    self.bounds[3])])
        self.velocity_vector = np.stack((self.position,
                                         self.position))
        self.velocity = np.zeros(2).astype(float)
        self.direction = 0.
        self.color = kwargs.get("color", "red")

    def __repr__(self):
        return f"Body(radius={self.radius})"

    def bounce(self):

        # undow the last move
        self.position -= self.speed * np.array([
            np.sin(self.direction),
            np.cos(self.direction)
        ])

        # change direction
        self.direction = (self.direction + np.pi) % (2*np.pi)
        self.trg_direction = random.uniform(0, 2*np.pi)

        # new move in the opposite direction
        self.y += self.speed * np.sin(self.direction)
        self.x += self.speed * np.cos(self.direction)

    def move(self, angle: float, speed: float):

        print(f"move angle: {angle:.3f}°")

        # --- velocity vector ---
        # calculate the new position of the object according to the
        # position and velocity
        self.velocity_vector = np.stack((self.position,
                    np.array([self.position[0] + np.cos(angle) * speed,
                    self.position[1] + np.sin(angle) * speed])))

        # self.velocity = self.velocity_vector[1] - self.velocity_vector[0]
        self.position = self.velocity_vector[1]
        print(f"used velocity: {np.around(self.velocity_vector.flatten(), 3)}")

        if self.position[0] < 0. or self.position[0] > 1. or \
            self.position[1] < 0. or self.position[1] > 1.:
            input("out of bounds!")

    def update_velocity_vector(self, angle: float, speed: float):
        print(f"> (velocity update) - position: {np.around(self.position, 3)}")
        self.velocity_vector = np.stack((self.position,
                    np.array([self.position[0] + np.cos(angle) * speed,
                    self.position[1] + np.sin(angle) * speed])))

        angle = calc_angle(self.velocity_vector)

        print(f"> (velocity update) - angle={angle:.3f}° v={np.around(self.velocity_vector.flatten(), 3)}")

    def handle_collisions(self, room: object) -> bool:
        new_velocity_vector, angle, collision = room.handle_collision(
            velocity_vector=self.velocity_vector,
            radius=self.radius)
        if angle is not None:
            print(f"angle from collisions: {angle:.3f}°")
            self.velocity_vector = new_velocity_vector
            self.position = self.velocity_vector[1]
            print(f"(collision), velocity vector: {np.around(self.velocity_vector.flatten(), 3)}")
        return new_velocity_vector, angle, collision

    def draw(self, ax: plt.Axes, alpha: float=1.):
        ax.add_patch(Circle(self.position,
                            self.radius, fc=self.color, ec='black'))
        ax.plot([self.velocity_vector[0, 0], self.velocity_vector[1, 0]],
                [self.velocity_vector[0, 1], self.velocity_vector[1, 1]],
                color='black', alpha=alpha)


class Agent:

    def __init__(self, position: np.ndarray=None, speed: float=0.1,
                 **kwargs):

        self.body = AgentBody(position=position,
                              radius=kwargs.get("radius", 0.05))
        self.speed = speed

        self.angle = np.pi/3
        self._has_collide = False
        self._t = 0
        self._last_bounce = 0

    def __repr__(self):
        return f"Agent({self.body})"

    def call2(self, room: object):

        # make choice : change after 10tu from the last collision
        # if self._last_bounce == 10:
        #     self.angle = np.random.uniform(0, 2*np.pi)

        self.body.move(direction=self.angle, speed=self.speed)
        # direction = self.body.handle_collisions(objects=room.walls)
        if direction is not None:
            self._has_collide = True
            # print("previous ", self.angle)
            self.angle = direction
            self._last_bounce = -1
            # print("current ", self.body.direction)

        self._t += 1
        self._last_bounce += 1

    def __call__(self, room: object):

        self.body.move(angle=self.angle, speed=self.speed)

        _, angle, self._has_collide = self.body.handle_collisions(room=room)
        if self._has_collide:
            self.angle = angle
            print(f"new angle after collision: {angle:.3f}°")
            # self.body.update_velocity_vector(angle=angle,
            #                                  speed=self.speed)

        print(f"position: {np.around(self.body.position, 3)}")

    def draw(self, ax: plt.Axes):
        self.body.draw(ax=ax)




""" main """

def draw_env(t: int, room: Room, agent: Agent,
             ax: plt.Axes, **kwargs):

    if t % kwargs.get("fpi", 10) != 0:
        return

    ax.clear()
    room.draw(ax=ax)
    agent.draw(ax=ax)
    ax.set_xlim((room.bounds[0], room.bounds[1]))
    ax.set_ylim((room.bounds[2], room.bounds[3]))
    ax.set_title(kwargs.get("title", ""))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.grid()
    plt.pause(kwargs.get("fpause", 0.01))


def run(room: Room, agent: Agent, duration: int=100,
        **kwargs):


    # plot
    _, ax = plt.subplots()
    title = ""

    for t in range(duration):

        title = f"{t=}"
        draw_env(t=t, ax=ax, room=room, agent=agent, title=title,
                 fpi=kwargs.get("fpi", 10),
                 fpause=kwargs.get("fpause", 0.01))

        agent(room=room)

    print("done")


""" utils """


# @jit
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


@jit
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

def calc_two_vector_angle(v1, v2) -> float:

    return np.arccos(
        (v1.reshape(1, -1) @ v2.reshape(-1, 1)) /
            (np.linalg.norm(v1) * np.linalg.norm(v2))
    )

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


# @jit(nopython=True)
# def reflect_vector(vector1: np.ndarray,
#                    vector2: np.ndarray,
#                    momentum: float=1.) -> np.ndarray:
#     """
#     reflect a vector wrt another
#     """

#     # Normalize vector2 to get the direction vector
#     vector2_dir = vector2[1] - vector2[0]
#     vector2_dir /= np.linalg.norm(vector2_dir)

#     # Calculate the projection of vector1[1] onto vector2_dir
#     projection_length = np.dot(vector1[1] - vector1[0], vector2_dir)
#     projection = projection_length * vector2_dir

#     # Calculate the reflection point
#     reflection_point = 2 * projection - (vector1[1] - vector1[0])
#     reflection_point = vector2[0] - momentum * reflection_point

#     # Create the reflected vector
#     reflected_vector = np.stack((vector2[0],
#                                  reflection_point))

#     return reflected_vector

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




if __name__ == "__main__":

    np.random.seed(0)


    #
    walls = (
        Wall([0, 0], [0, 1]),
        Wall([0, 1], [1, 1]),
        Wall([1, 1], [1, 0]),
        Wall([1, 0], [0, 0])
    )

    room = Room(walls=walls)
    print(room)

    agent = Agent(position=np.array([0.5, 0.5]),
                  radius=0.04,
                  speed=0.01)

    print(agent)


    run(room=room, agent=agent, duration=10_000,
        fpi=10, fpause=0.1)






