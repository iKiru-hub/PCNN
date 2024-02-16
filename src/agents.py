import numpy as np 
import matplotlib.pyplot as plt 
from itertools import product as iterprod
from scipy.signal import correlate2d, find_peaks
import warnings
try:
    from tools.utils import logger, tqdm_enumerate
except ModuleNotFoundError:
    warnings.warn('`tools.utils` not found, using fake logger. Some functions may not work')
    class Logger:

        print('Logger not found, using fake logger')

        def info(self, msg: str):
            print(msg)

        def debug(self, msg: str):
            print(msg)

    logger = Logger()
try:
    import inputools.Trajectory as it
except ModuleNotFoundError:
    warnings.warn('`inputools.Trajectory` not found, some functions may not work')






class AgentPC:

    """
    an agent using an hard-coded place cell layer for representing
    the environment. The agent can move in the environment by making 
    discrete steps in chosen direction.
    """

    def __init__(self, Npc: int, Nh: int, **kwargs):

        """
        Parameters
        ----------
        Npc : int
            number of place cells
        Nh : int
            number of hidden units
        **kwargs : dict
            sigma_pc : float
                standard deviation of the place cells.
                Default is 0.1
            step_size : float
                step size of the agent.
                Default is 0.01
            lr : float
                learning rate of the agent. 
                Default is 0.01
            start_position : np.ndarray
                starting position of the agent.
                Default is np.array([0, 0])
            activation : str
                activation function of the hidden layer.
                Default is None
        """

        # parameters
        self.Npc = Npc
        self.Nh = Nh
        self._step_size = kwargs.get('step_size', 0.01)
        self._lr = kwargs.get('lr', 0.01)

        # components 
        self.layer_pc = it.PlaceLayer(N=Npc,
                                          sigma=kwargs.get('sigma_pc', 0.1))
        self.Wh = np.random.normal(0, 1/np.sqrt(Npc), size=(Nh, Npc))
        self.bh = np.random.normal(0, 1/np.sqrt(Npc), size=(Nh, 1))
        self.Wout = np.random.normal(0, 1/np.sqrt(Nh), size=(Npc, Nh))
        self.bout = np.random.normal(0, 1/np.sqrt(Nh), size=(Npc, 1))

        # activation function
        if kwargs.get('activation', None) is None:
            self._activation = lambda x: x
            self._activation_prime = lambda x: 1
        elif kwargs.get('activation') == 'relu':
            self._activation = lambda x: np.maximum(0, x)
            self._activation_prime = lambda x: np.where(x > 0, 1, 0)
        elif kwargs.get('activation') == 'sigmoid':
            self._activation = lambda x: 1 / (1 + np.exp(-x))
            self._activation_prime = lambda x: self._activation(x) * \
                (1 - self._activation(x))
        elif kwargs.get('activation') == 'tanh':
            self._activation = lambda x: np.tanh(x)
            self._activation_prime = lambda x: 1 - np.tanh(x)**2
        else:
            raise NotImplementedError(
                f"Activation {kwargs.get('activation')} not implemented.")


        # variables
        self.current_position = kwargs.get('start_position', np.zeros((2, 1))).reshape(2, 1)
        self.previous_position = None
        self.pc_post_z = None 
        self.pc_post_a = None
        self.current_pc = None
        self.goal_pc = None
        self.input_pc = None
        self.zh = None
        self.ah = None

        self.grad_out = None
        self.grad_h = None

    def __repr__(self) -> str:

        return f'AgentPC(Npc={self.Npc}, Nh={self.Nh})'

    def __call__(self, goal_position: np.ndarray) -> np.ndarray:

        """
        Parameters
        ----------
        goal_position : np.ndarray
            target position of the agent

        Returns
        -------
        np.ndarray
            action to take
        """

        # determine the PC encoding of the current position
        self.current_pc = self.layer_pc.step(position=self.current_position)

        # determine the PC encoding of the goal position
        self.goal_pc = self.layer_pc.step(position=goal_position)

        # determine the hidden layer activation
        self.input_pc = self.current_pc + self.goal_pc
        self.zh = self.Wh @ self.input_pc + self.bh
        self.ah = self._activation(self.zh)

        # determine the second hidden layer activation
        self.pc_post_z = self.Wout @ self.ah + self.bout
        self.pc_post_a = self._activation(self.pc_post_z)

        # # determine the action/angle to take
        # action = self._calc_angle(p1=self.layer_pc.centers[np.argmax(self.current_pc)],
        #                           p2=self.layer_pc.centers[np.argmax(self.pc_post_a)])

        # # update the next position
        # self.previous_position = self.current_position.copy()
        # self.current_position += self._step_size * np.array([[np.cos(action)],
        #                                                      [np.sin(action)]])

        # calculate the current position in xy
        x, y = self.layer_pc.centers[np.argmax(self.pc_post_a)]
        self.current_position = np.array([[x], [y]])

        return x, y

    @staticmethod
    def _calc_angle(p1: np.ndarray, p2: np.ndarray) -> float:

        """
        Parameters
        ----------
        p1 : np.ndarray
            position 1
        p2 : np.ndarray
            position 2

        Returns
        -------
        float
            angle between the two positions
        """

        # calculate the angle between the two positions
        x1, y1 = p1
        x2, y2 = p2

        dx = x2 - x1
        dy = y2 - y1

        theta = 2*np.pi - np.arctan2(dy, dx) + np.pi/2

        return theta % (2 * np.pi)

    def update(self, target_position: np.ndarray) -> float:

        """
        Parameters
        ----------
        position_error : np.ndarray
            error in the position

        Returns
        -------
        float
            error in the position
        """

        # ----| Error
        # calculate the target activation of the place cell layer
        # by selecting the place cell closest to the target position
        target_pc = self.layer_pc.step(position=target_position)
        position_error = target_pc - self.pc_post_a
        # logger.debug(f'position_error: {position_error}')

        # ----| Gradient

        # calculate the gradient in the output weights
        delta_out = self._activation_prime(self.pc_post_z) * position_error
        self.grad_out = delta_out @ self.ah.T
        # logger.debug(f'grad_out: {grad_out}')

        # calculate the gradient in the hidden layer weights
        delta_h = (self.Wout.T @ delta_out) * self._activation_prime(self.zh)
        self.grad_h = delta_h @ self.input_pc.T
        # logger.debug(f'grad_h: {grad_h}')

        # ----| Update
        self.Wout -= self._lr * self.grad_out
        self.bout -= self._lr * delta_out
        self.Wh -= self._lr * self.grad_h
        self.bh -= self._lr * delta_h

        return (position_error**2).sum()

    def set_position(self, position: np.ndarray) -> None:

        """
        set the position of the agent
        """

        self.current_position = position

