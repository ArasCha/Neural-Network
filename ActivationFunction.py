import numpy as np

def tanh(x: float):
    return np.tanh(x)

def sigmoid(x: float):
    return 1/(1+np.exp(-x))

def unit_step(x: float):
    return 0 if x<0 else 1 if x>0 else 0.5
