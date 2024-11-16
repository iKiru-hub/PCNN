import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

def sigmoid_like(x, slope=10, threshold=0.5):
    """
    A sigmoid-like function that maps [0, 1] to [0, 1].
    Parameters:
    - x: Input value(s) (numpy array or scalar).
    - slope: Controls the steepness of the function.
    - threshold: The x-value where the midpoint occurs.
    """
    return (np.exp(slope * (x - threshold)) - 1) / (np.exp(slope * (x - threshold)) + 1) * 0.5 + 0.5

def piecewise_linear(x, slope=10, threshold=0.5):
    """
    A piecewise linear function mapping [0, 1] to [0, 1].
    Parameters:
    - x: Input value(s) (numpy array or scalar).
    - slope: Controls the steepness.
    - threshold: The x-value where the function has a midpoint.
    """
    y = slope * (x - threshold)
    y = np.clip(y, 0, 1)  # Ensure the output is within [0, 1]
    return y

def plot_function(slope=10, threshold=0.5, mode='sigmoid_like'):
    """
    Plots the activation function.
    Parameters:
    - slope: Controls the steepness of the function.
    - threshold: The x-value where the midpoint occurs.
    - mode: Select 'sigmoid_like' or 'piecewise_linear'.
    """
    x = np.linspace(0, 1, 500)
    if mode == 'sigmoid_like':
        y = sigmoid_like(x, slope, threshold)
    elif mode == 'piecewise_linear':
        y = piecewise_linear(x, slope, threshold)
    else:
        raise ValueError("Invalid mode. Choose 'sigmoid_like' or 'piecewise_linear'.")
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f'{mode} (slope={slope}, threshold={threshold})', color='blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(1, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
    plt.title(f'{mode.capitalize()} Function')
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.legend()
    plt.grid()
    plt.show()

# Interactive widget to explore the function
interact(plot_function, 
         slope=widgets.FloatSlider(value=10, min=0.1, max=50, step=0.5, description='Slope'), 
         threshold=widgets.FloatSlider(value=0.5, min=0, max=1, step=0.01, description='Threshold'),
         mode=widgets.Dropdown(options=['sigmoid_like', 'piecewise_linear'], value='sigmoid_like', description='Mode'))
