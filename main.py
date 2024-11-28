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

def cramer(coeff_matrix, constants):
    """Solves a sytem of linear equations using Creamer's rule

    Args:
        coeff_matrix (numpy.ndarry): A square matrix representinf the coefficients of the linear sysystem
        constants (numpy.ndarray): A 1D array representing the constants

    Raises:
        ValueError: if the determinat of the coefficient matriz is zero

    Returns:
        numpy.ndarray: A 1D array containing the values of the variables
    """
    det_A = np.linalg.det(coeff_matrix)
    if np.isclose(det_A, 0):
        raise ValueError("Has no unique solution.")
        
    n = coeff_matrix.shape[1]
    solutions = np.zeros(n)
    
    for i in range(n):
        mod_matrix = coeff_matrix.copy()
        mod_matrix[:, i] = constants
        det_modified = np.linalg.det(mod_matrix)
        solutions[i] = det_modified / det_A
    
    return solutions


def test_cramer():
 
    A = np.array([[2, -1, 3],
                  [1, 1, 1],
                  [3, -2, 4]])
    
    B = np.array([5, 6, 8])

    expected_solution = np.linalg.solve(A, B)

    solution = cramer(A, B)

    assert np.allclose(solution, expected_solution)



def test_generate_even_odd():
    shape = (8, 8)
    even_indexes, odd_indexes, array = generate_even_odd(shape, low=0, high=20)

    for idx in even_indexes:
        assert array[idx[0], idx[1]] % 2 == 0

    for idx in odd_indexes:
        assert array[idx[0], idx[1]] % 2 != 0



