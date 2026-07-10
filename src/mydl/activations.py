import numpy as np 

# sigmoid function for non linear output 
def sigmoid(Z):
    """
    Compute the sigmoid activation.

    Parameters
    ----------
    z : np.ndarray or float
        Input of shape (no of neuron in layer , no of training examples)

    Returns
    -------
    np.ndarray or float of size (no of neuron in lyaer , no of training examples )
        Sigmoid activation with the same shape as z.
    """
    
    return 1 / (1+np.exp(-Z))


    
def relu(Z):
    """
    Parameters
    -----------
    z: np.ndarray or float , input values
        (no of neuron in lyaer , no of training examples )
        
    Returns
    -------
    np.ndarray
        (no of neuron in lyaer , no of training examples )
    """
    
    return np.maximum(0,Z)



def tanh(Z):
    """
    Parameters
    ----------
    Z: np.ndarray 
        (no of neuron in lyaer , no of training examples )
    
    Returns
    --------
    np.ndarray of size ()
        (no of neuron in lyaer , no of training examples )
    """
    
    exp_p = np.exp(Z)
    exp_n = np.exp(-Z)
    return (exp_p - exp_n) / (exp_p + exp_n)


def sigmoid_backward(dA, cache):
    """
    Backward propagation for the sigmoid activation.

    Parameters
    ----------
    dA : np.ndarray
        Gradient of the loss with respect to the activation output.

    cache : np.ndarray
        Cached pre-activation values (Z) from the forward pass.

    Returns
    -------
    np.ndarray
        Gradient of the loss with respect to Z.
    """
    
    Z = cache
    A = sigmoid(Z)
    dZ = dA * A * (1-A)
    
    return dZ


def relu_backward(dA, cache):  
    """
    Backward propagation for the ReLU activation.

    Parameters
    ----------
    dA : np.ndarray
        Gradient of the loss with respect to the activation output.

    cache : np.ndarray
        Cached pre-activation values (Z) from the forward pass.

    Returns
    -------
    np.ndarray
        Gradient of the loss with respect to Z.
    """
    
    Z = cache
    dZ = dA.copy()
    dZ[Z <= 0 ] = 0
    
    return dZ


def tanh_backward(dA, cache):
    """
    Backward propagation for the tanh activation.

    Parameters
    ----------
    dA : np.ndarray
        Gradient of the loss with respect to the activation output.

    cache : np.ndarray
        Cached pre-activation values (Z) from the forward pass.

    Returns
    -------
    np.ndarray
        Gradient of the loss with respect to Z.
    """
    
    Z = cache
    A = tanh(Z)
    dZ = dA * (1-A**2)
    
    return dZ