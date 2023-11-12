"""
This module contains functions for stimulation of the network.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product as iterprod
try:
    from tools.utils import logger, tqdm_enumerate
except ModuleNotFoundError:
    class Logger:
        def info(self, *args):
            print(*args)
    logger = Logger()



####


class AnimalTrajectory:

    def __init__(self,  dt: float, **kwargs):

        """
        Initialize the Animal object with given parameters.

        Parameters
        ----------
        dt : float
            Time step in ms.
        kwargs : dict
            prob_turn : float
                Probability of turning, default 0.0.
            prob_speed : float
                Probability of changing speed, default 0.0.
            prob_rest : float
                Probability of resting, default 0.1.
            day_cycle : bool
                Day/Night cycle influence, default True.
        """

        self.position = np.array([0.5, 0.5])
        self.speed = 0
        self.direction = np.array([1, 0])
        self.target_position = np.array([0.5, 0.5])
        self.distance = 0
        self.prob_turn = kwargs.get('prob_turn', 0.0)
        self.prob_speed = kwargs.get('prob_speed', 0.0)
        self.dt = dt

        self.prob_rest = kwargs.get('prob_rest', 0.1) / dt
        self.day_cycle = kwargs.get('day_cycle', True)
        self.time_of_day = 0  # Representing the time of day in arbitrary units

    def _calc_speed(self):

        """
        Calculate the speed of the animal based on the distance to the target.
        """

        # gaussian distance to target
        self.distance = np.linalg.norm(self.target_position - self.position)
        if self.distance < 0.05:
            self.speed = np.exp(-self.distance ** 2 / 0.02)

        elif np.random.rand() < self.prob_speed:
            self.speed = np.random.uniform(0.05, 0.15)

    def _behave(self):

        """
        Define the behavior of the animal based on the time of day.
        """


        # Introduce rest periods
        if np.random.rand() < self.prob_rest:
            self.speed = 0  # Rest
        else:
            self._calc_speed()  # Use existing speed calculation

        # Day/Night cycle influence
        if self.day_cycle:
            if 6 <= self.time_of_day <= 18:  # Daytime
                self.speed *= 1.5
            else:  # Nighttime
                self.speed *= 0.5

    def _change_target(self):

        """
        Change the target position of the animal.
        """

        # turn with probability prob_turn
        if np.random.rand() < self.prob_turn or self.distance < 0.002:
            self.target_position *= np.random.uniform(0.8, 1.2, size=2)
            self.target_position = np.clip(self.target_position, 0.05, 0.95)

    def _change_direction(self):

        """
        Change the direction of the animal.
        """

        # Gradual turns instead of instant turns
        turn_angle = np.random.uniform(-np.pi/8, np.pi/8)  # Limit the turn angle
        rotation_matrix = np.array([[np.cos(turn_angle), -np.sin(turn_angle)],
                                    [np.sin(turn_angle), np.cos(turn_angle)]])
        self.direction = np.dot(rotation_matrix, self.direction)
        self.direction /= np.linalg.norm(self.direction)  # Normalize direction

    def _whole_walk(self, dx: int=0.1) -> np.ndarray:

        """
        Generate a whole walk in the grid.

        Parameters
        ----------
        dx : int
            Step size. Default: 0.1

        Returns
        -------
        trajectory : numpy.ndarray
            Trajectory of the random walk.
        """

        track_slice = np.arange(0, 1, dx)

        return np.stack(np.meshgrid(track_slice, track_slice),
                        axis=-1).reshape(-1, 2)


    def step(self):

        """
        Update the position of the animal.
        """

        self._change_target()
        self._behave()
        self._change_direction()

        # Update position with direction and speed
        self.position += self.direction * self.speed / self.dt

        # Update the time of day
        self.time_of_day = (self.time_of_day + 1) % 24

    def make_trajectory(self, duration: int, whole: bool=False) -> np.ndarray:

        """
        Create a trajectory of the animal.

        Parameters
        ----------
        duration : int
            Duration of the trajectory (s).
        whole : bool
            Whether to generate a whole walk or not.
            Default: False

        Returns
        -------
        array-like : trajectory
            trajectory of the animal.
        """

        if whole:
            return self._whole_walk()

        trajectory = np.zeros((duration, 2))
        for t in range(duration):
            trajectory[t] = self.position
            self.step()

        # constrain trajectory to the track
        trajectory[:, 0] = (trajectory[:, 0].max() - trajectory[:, 0]) / (trajectory[:, 0].max() - trajectory[:, 0].min())
        trajectory[:, 1] = (trajectory[:, 1].max() - trajectory[:, 1]) / (trajectory[:, 1].max() - trajectory[:, 1].min())

        return trajectory


class InputLayer:

    """
    Input layer of the network, made of neurons that will encode a
    trajectory in the grid according to their tuning function.
    """

    def __init__(self, N: int, bounds: tuple=None, 
                 kind: str='place', mode: str='rate',
                 **kwargs):

        """
        Input layer, made of neurons that will encode a 
        trajectory in the grid.

        Parameters
        ----------
        N : int
            number of neurons in the layer.
        bounds : tuple
            (x_min, x_max, y_min, y_max) bounds of the grid.
        kind : str
            kind of tuning function. 
            Options: 'place', 'grid'.
            Default: 'place'.
        mode : str
            mode of the tuning function.
            Options: 'rate', 'spike'.
            Default: 'rate'.
        **kwargs : dict
            sigma : float
                Standard deviation of the tuning function.
                Default: 1
            max_rate : float
                Maximum spike rate of the neurons in the layer.
            min_rate : float
                Minimum spike rate of the neurons in the layer.
        """

        self.N = N
        self.n = int(np.sqrt(N))
        self.bounds = (0, 1, 0, 1) if bounds is None else bounds
        self.kind = kind
        self.mode = mode
        self.sigma = kwargs.get('sigma', 1)
        self.max_rate = kwargs.get('max_rate', 100)
        self.min_rate = kwargs.get('min_rate', 0)

        self.centers = self._make_centers()
        
        self.activation = np.zeros(self.N)

    def __repr__(self):

        return f"InputLayer(N={self.N}, kind={self.kind}, sigma={self.sigma})"

    def _make_centers(self) -> np.ndarray:

        """
        Make the tuning function for the neurons in the layer.

        Returns
        -------
        centers : numpy.ndarray
            centers for the neurons in the layer.
        """

        if self.kind == 'place':
            return self._make_place_fields()
        elif self.kind == 'grid':
            pass
        
    def _make_place_fields(self) -> np.ndarray:

        """
        Make the tuning function for the neurons in the layer.

        Returns
        -------
        centers : numpy.ndarray
            centers for the neurons in the layer.
        """

        x_min, x_max, y_min, y_max = self.bounds

        # Define the centers of the tuning functions
        x_centers = np.linspace(x_min, x_max, 
                                self.n, endpoint=False)
        y_centers = np.linspace(y_min, y_max,
                                self.n, endpoint=False)

        # Make the tuning function
        centers = np.zeros((self.N, 2))
        for i in range(self.N):
            centers[i] = (x_centers[i // self.n], 
                          y_centers[i % self.n])

        return centers

    def step(self, position: np.ndarray) -> np.ndarray:

        """
        Step function of the input layer.

        Parameters
        ----------
        position : numpy.ndarray
            (x, y) coordinates of the current position.

        Returns
        -------
        activation : numpy.ndarray
            Activation of the neurons in the layer.
        """

        self.activation = np.exp(-np.linalg.norm(
            position - self.centers, axis=1)**2 / self.sigma)

        if self.mode == 'rate':
            self.activation *= self.max_rate 

        elif self.mode == 'spike':
            self.activation = np.random.binomial(
                1, (self.activation*self.max_rate).clip(
                    self.min_rate, self.max_rate)/1000,
                size=self.N
            )

        return self.activation

    def parse_trajectory(self, trajectory: np.ndarray, 
                         **kwargs) -> np.ndarray:

        """
        Parse a trajectory into the input layer.

        Parameters
        ----------
        trajectory : numpy.ndarray
            Trajectory to parse into the input layer.
        **kwargs : dict
            timestep : float
                Time step of the trajectory.
                Default: 1.

        Returns
        -------
        activations : numpy.ndarray
            Activations of the neurons in the layer.
        """

        timestep = kwargs.get('timestep', 1)
        activations = np.zeros((len(trajectory)*timestep, 
                                self.N))

        for t in range(len(trajectory)):
            for i in range(timestep):
                activations[t*timestep+i] = self.step(
                    trajectory[t])

        return activations




####


class InputLayer2:

    """
    Input layer of the network, made of neurons that will encode a
    trajectory in the grid according to their tuning function.
    """

    def __init__(self, N: int, bounds: tuple, kind: str='place', **kwargs):

        """
        Input layer, made of neurons that will encode a 
        trajectory in the grid.

        Parameters
        ----------
        N : int
            number of neurons in the layer.
        bounds : tuple
            (x_min, x_max, y_min, y_max) bounds of the grid.
        kind : str
            kind of tuning function. 
            Options: 'place', 'grid'.
            Default: 'place'.
        **kwargs : dict
            sigma : float
                Standard deviation of the tuning function.
                Default: 1
        """

        self.N = N
        self.n = int(np.sqrt(N))
        self.bounds = bounds
        self.kind = kind
        self.sigma = kwargs.get('sigma', 1)

        self.centers = self._make_centers(bounds)
        
        self.activation = np.zeros(self.N)

    def __repr__(self):

        return f"InputLayer(N={self.N}, kind={self.kind}, sigma={self.sigma})"

    def _make_centers(self, bounds: tuple) -> np.ndarray:

        """
        Make the tuning function for the neurons in the layer.

        Parameters
        ----------
        bounds : tuple
            (x_min, x_max, y_min, y_max) bounds of the grid.

        Returns
        -------
        centers : numpy.ndarray
            centers for the neurons in the layer.
        """

        if self.kind == 'place':
            return self._make_place_fields(bounds=bounds)
        
    def _make_place_fields(self, bounds: tuple) -> np.ndarray:

        """
        Make the tuning function for the neurons in the layer.

        Parameters
        ----------
        bounds : tuple
            (x_min, x_max, y_min, y_max) bounds of the grid.

        Returns
        -------
        centers : numpy.ndarray
            centers for the neurons in the layer.
        """

        x_min, x_max, y_min, y_max = bounds

        # Define the centers of the tuning functions
        x_centers = np.linspace(x_min+1, x_max, 
                                self.n, endpoint=False)
        y_centers = np.linspace(y_min+1, y_max,
                                self.n, endpoint=False)

        # Make the tuning function
        centers = np.zeros((self.N, 2))
        for i in range(self.N):
            centers[i] = (x_centers[i // self.n], 
                         y_centers[i % self.n])

        return centers

    def step(self, position: np.ndarray) -> np.ndarray:

        """
        Step function of the input layer.

        Parameters
        ----------
        position : numpy.ndarray
            (x, y) coordinates of the current position.

        Returns
        -------
        activation : numpy.ndarray
            Activation of the neurons in the layer.
        """

        # self.activation *= 0

        # for i in range(self.N):
        #     self.activation[i] = np.exp(-np.linalg.norm(
        #         position - self.centers[i]) / self.sigma)

        self.activation = np.exp(-np.linalg.norm(
            position - self.centers, axis=1) / self.sigma)

        return self.activation


def generate_walk_trajectory(steps: int,
                             layer: object,
                             verbose: bool=False,
                             origin: tuple=None) -> tuple:

    """
    Generate a continuous forward-only walk trajectory in an N x N grid.

    Parameters
    ----------
    steps : int
        Number of steps to take.
    layer : object
        Input layer.
    verbose : bool
        Whether to print the trajectory.
        Default: False
    origin : tuple
        Starting position of the walk.
        Default: center of the grid

    Returns
    -------
    trajectory : list of tuple
        List of (x, y) coordinates of the trajectory.
    activations : list of numpy.ndarray
        List of neuron activations over time.
    """

    N = layer.N
    _, xsize, _, ysize = layer.bounds

    # Start from the center of the grid
    if origin is None:
        position = (xsize // 2, ysize // 2)  
    else:
        position = origin

    if verbose:
        logger.info(f"Start position: {position}")

    trajectory = np.zeros((steps, 2))
    activations = np.zeros((steps, N))

    # Define the previous move to avoid staying in the
    # same spot or moving back
    prev_move = None

    for t in range(steps):

        position = np.array(position)

        trajectory[t] = position
        activations[t] = layer.step(position)

        # Define possible moves and remove the opposite of the
        # previous move to avoid going back or staying
        possible_moves = ['up', 'down', 'left', 'right']
        if prev_move:
            opposite_moves = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
            # Remove the opposite move of the previous one
            possible_moves.remove(opposite_moves[prev_move])

        # Choose a random direction from the remaining possible moves
        move = np.random.choice(possible_moves)
        prev_move = move  # Update the previous move


        # UP
        done = False

        while not done:
            if move == 'up' and position[1] < ysize-1:  # legal
                position = (position[0], position[1] + 1)
                done = True
            elif move == 'up' and position[1] >= ysize: # illegal
                possible_moves = [s for s in possible_moves if s != 'up']
                if verbose:
                    logger.info(f"- !up, new: {possible_moves}")
        
            # Down
            if move == 'down' and position[1] > 0:  # legal
                position = (position[0], position[1] - 1)
                done = True
            elif move == 'down' and position[1] <= 0: # illegal
                possible_moves = [s for s in possible_moves if s != 'down']
                if verbose:
                    logger.info(f"- !down, new: {possible_moves}")
        
            # Left
            if move == 'right' and position[0] < xsize-1:  # legal
                position = (position[0]+1, position[1])
                done = True
            elif move == 'right' and position[0] >= xsize: # illegal
                possible_moves = [s for s in possible_moves if s != 'right']
                if verbose:
                    logger.info(f"- !up, right: {possible_moves}")
                
            # Right
            if move == 'left' and position[0] > 0:  # legal
                position = (position[0]-1, position[1])
                done = True
            elif move == 'left' and position[0] <= 0: # illegal
                possible_moves = [s for s in possible_moves if s != 'left']
                if verbose:
                    logger.info(f"- !left, new: {possible_moves}")
        
            # last draw
            if not done:
                if verbose:
                    logger.info(f"-- new choices: {possible_moves}")
                # Choose a random direction from the remaining possible moves
                move = np.random.choice(possible_moves)
                if move == prev_move:
                    if verbose:
                        logger.info(f"{move}, {prev_move}, {possible_moves}")
                prev_move = move  # Update the previous move

        if verbose:
            logger.info(f"move {move}, {position}")
            
    return trajectory, activations


def visualize_trajectory_and_activations(trajectory: np.ndarray, 
                                         activations: np.ndarray,
                                         bounds: tuple):

    """
    Visualize the trajectory and neuron activations.

    Parameters
    ----------
    trajectory : numpy.ndarray
        Trajectory of the random walk.
    activations : numpy.ndarray
        Neuron activations over time.
    bounds : tuple
        (x_min, x_max, y_min, y_max) bounds of the grid.
    """

    _, xsize, _, ysize = bounds

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot trajectory
    axes[0].plot(*zip(*trajectory), marker='o')
    axes[0].set_xlim(-0.5, xsize-0.5)
    axes[0].set_ylim(-0.5, ysize-0.5)
    axes[0].set_title('Random Walk Trajectory')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True)

    # Plot activations
    activation_map = np.array(activations).T  # correct orientation
    axes[1].imshow(activation_map, cmap='Greys', aspect='auto')
    axes[1].set_title('Neuron Activations Over Time')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('Neuron Index')

    plt.tight_layout()
    plt.show()


def make_spiking_trajectory(Nj: int, activations: np.ndarray,
                            stm_duration: int, pause_time=100,
                            binary: bool=False,
                            **kwargs) -> np.ndarray:

    """
    Make a spiking trajectory from neuron activations.

    Parameters
    ----------
    Nj : int
        Number of neurons.
    activations : numpy.ndarray
        Neuron activations over time.
    stm_duration : int
        Duration of the stimulation in time steps.
    pause_time : int
        Pause time between stimulations in time steps.
        Default: 100
    binary : bool
        Whether to make a binary spiking trajectory.
        Default: False
    **kwargs : dict
        base_rate : float
            Base firing rate.
            Default: 10
        stm_rate : float
            Stimulation firing rate.
            Default: 300
    
    Returns
    -------
    spiking_trajectory : numpy.ndarray
        Spiking trajectory.
    """

    base_rate = kwargs.get('base_rate', 10)
    stm_rate = kwargs.get('stm_rate', 300)

    T = len(activations) * stm_duration + len(activations) * pause_time

    # input spikes
    Sj = np.random.binomial(1, base_rate/1000, size=(T, Nj, 1))

    if binary:
        for i, t in enumerate(range(0, T, stm_duration + pause_time)):
            Sj[t: t + stm_duration, activations[i].argmax()] = np.random.binomial(
                1, stm_rate/1000, size=(stm_duration, 1))
    else:
        for i, t in enumerate(range(0, T, stm_duration + pause_time)):
            for ti in range(stm_duration):
                Sj[t: t + ti, :, 0] = np.random.binomial(
                    1, stm_rate * activations[i].reshape(-1) / 1000, size=Nj)

    logger.info(f"2D stimulus ready: {Sj.shape}")

    return Sj


def make_dataset(n_samples: int, N: int, steps: int, stm_duration: int, 
                 pause_time: int, **kwargs) -> list:

    """
    Make a dataset of spiking trajectories.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    N : int
        Size of the grid (N x N).
    steps : int
        Number of steps to take.
    stm_duration : int
        Duration of the stimulation in time steps.
    pause_time : int
        Pause time between stimulations in time steps.
    **kwargs : dict
        base_rate : float
            Base firing rate.
            Default: 10
        stm_rate : float
            Stimulation firing rate.
            Default: 300

    Returns
    -------
    dataset : list of numpy.ndarray
        Dataset of spiking trajectories.
    """

    dataset = []

    for _ in range(n_samples):

        trajectory, activations = generate_walk_trajectory(N, steps)
        Sj = make_spiking_trajectory(Nj=N**2, activations=activations,
                                     stm_duration=stimulus_duration,
                                     pause_time=pause_time, **kwargs)
        dataset.append(Sj)

    return dataset


def make_whole_walk(layer: object, dx: float=1) -> tuple:

    """
    generate a whole walk in the grid

    Parameters
    ----------
    layer : object
        Input layer.
    dx : float
        Step size.
        Default: 1

    Returns
    -------
    trajectory : numpy.ndarray
        Trajectory of the random walk.
    activations : numpy.ndarray
        Neuron activations over time.
    """

    N = layer.N
    _, xsize, _, ysize = layer.bounds

    # Start from the center of the grid
    position = (0, 0)

    # the axis with stepsize dx
    xaxis = np.arange(0, xsize, dx)
    yaxis = np.arange(0, ysize, dx)

    # all possible points in the space
    all_positions = np.array([*iterprod(xaxis, yaxis)])
    all_activations = np.zeros((len(all_positions), N))

    # activations of the layer in all positions
    for i, position in enumerate(all_positions):
        all_activations[i] = layer.step(position)

    return all_positions, all_activations


def get_network_tuning(model: object, layer: object, dx: float=1,
                       stm_duration: int=1000, pause_time=100,
                       binary: bool=False, **kwargs) -> np.ndarray:

    """
    Given a model and an input layer, get the tuning of the network.

    Parameters
    ----------
    model : object
        The model.
    layer : object
        The input layer.
    dx : float
        Step size.
        Default: 1
    stm_duration : int
        Duration of the stimulation in time steps.
    pause_time : int
        Pause time between stimulations in time steps.
        Default: 100
    binary : bool
        Whether to make a binary spiking trajectory.
        Default: False
    **kwargs : dict
        base_rate : float
            Base firing rate.
            Default: 10
        stm_rate : float
            Stimulation firing rate.
            Default: 300

    Returns
    -------
    network_tuning : numpy.ndarray
        Network tuning.
    """

    # bounds 
    _, xsize, _, ysize = layer.bounds
    xlen = np.arange(0, xsize, dx).__len__()
    ylen = np.arange(0, ysize, dx).__len__()

    # get the whole walk
    all_positions, all_activations = make_whole_walk(layer, dx)

    #
    activations = np.zeros((len(all_positions), model.N))

    # get the activations of the network in all positions
    for i, input_activation in tqdm_enumerate(all_activations):
        local_act = np.zeros(model.N)
        for _ in range(stm_duration):
            model.step(input_activation.reshape(-1, 1))
            local_act += model.s.reshape(-1)
           
        activations[i] = local_act / stm_duration

    return activations.reshape(xlen, ylen, model.N)




