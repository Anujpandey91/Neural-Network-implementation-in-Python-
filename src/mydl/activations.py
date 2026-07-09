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