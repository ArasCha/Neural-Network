import numpy as np



def logloss(A: np.ndarray, Y: np.ndarray) -> float:
    """
    Returns the value of the error
    """
    assert A.shape == Y.shape, "A and Y matrixes must have the same size"
    return -(1/len(Y)) * np.sum( [y * np.log(a) + (1-y) * np.log(1-a) for a,y in zip(A,Y)] )

def logloss_derivative_weights(X: np.ndarray, A: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Returns ∂L/∂W
    """
    assert A.shape == Y.shape, "A and Y matrixes must have the same size"
    assert A.shape[0] == X.shape[0], "A and Z must have the same number of lines"
    return (1/len(A)) * np.transpose(X).dot(A - Y)

def logloss_derivative_bias(A: np.ndarray, Y: np.ndarray) -> float:
    """
    Returns ∂L/∂b
    """
    assert A.shape == Y.shape, "A and Y matrixes must have the same size"
    return (1/len(A)) * np.sum(A - Y)