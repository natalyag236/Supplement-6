import numpy as np
def  normal_arr(shape, mean=0, std_dev=1):
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