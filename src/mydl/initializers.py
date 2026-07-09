import numpy as np

def zeros(shape):
    return np.zeros(shape)

def ones(shape):
    return np.ones(shape)

def random(shape, scale):
    return np.random.randn(shape) * scale


def xavier(shape):
    n_prev = shape[1]

    return np.random.randn(*shape) * np.sqrt(1 / n_prev)


def he(shape):
    n_prev = shape[1]

    return np.random.randn(*shape) * np.sqrt(2 / n_prev)
