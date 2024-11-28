import numpy as np
def  normal_arr(shape, mean=0, std_dev=1):
    """Generate an array of normally distributed random numbers.

    Args:
        shape (tuple of int): the desired shape of the output array 
        mean (int, optional): the mean  of the normal distribution. Defaults to 0.
        std_dev (int, optional): the standard deviation of the normal distribution. Defaults to 1.

    Returns:
        np.ndarray: an array of normally distributed random numbers withthe specified shape, mean, and standard deviation
    """
    return np.random.normal(loc=mean, scale=std_dev, size=shape)

def test_normal_arr():
    shape = (500,500)
    mean = 10
    std_dev = 2 

    array = normal_arr(shape, mean, std_dev)

    assert array.shape == shape

    calculated_mean = np.mean(array)
    assert abs(calculated_mean - mean) <0.1

    calculated_std_dev = np.std(array)
    assert abs(calculated_std_dev - std_dev) < 0.1

def test_cramer():
    A = np.array([[2, -1, 3,],
                  [1, 1, 1]
                  [3, -2, 4]])
    B = np.array([5, 6, 8])

    excepted = np.array([1, 2, 3])

    solution = cramer(A, B)

    assert np.allclose(solution, excepted)