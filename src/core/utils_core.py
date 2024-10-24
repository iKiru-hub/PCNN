import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod







class OnlineFigure(ABC):

    """
    an object responsible for plotting online data
    in different figures
    """

    def __init__(self):

        self.fig, self.ax = plt.subplots()

    @abstractmethod
    def update(self):
        pass
