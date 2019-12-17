import numpy as np
from functools import partial


l2_norm = partial(np.linalg.norm, ord=2, axis=-1)


def fuzz_min_sum(x, y):
    """ Calculate fuzzy_min and sum over vectors

    >>> fuzz_min_sum(np.array([3, 4, 5]), np.array([1, 7, 2]))
    7
    """
    assert len(x) == len(y), "length of input vectors do not match"
    return sum(np.minimum(x, y))


def normalized_euc_dist(x, y, coded):
    assert len(x) == len(y), "length of input vectors do not match"
    if coded:
        data_point = x[:len(x)//2]
        complemented = np.ones(len(y[len(y)//2:])) - y[len(y)//2:]
        center_point = (y[:len(y)//2] + complemented) / 2
        return 1 - np.sqrt(sum((data_point - center_point) * (data_point - center_point))) / np.sqrt(len(x) // 2)
    return 1 - np.sqrt(sum((x - y) * (x - y)) / np.sqrt(len(x)))


def chi_square_dist(x, y):
    pass


def complement_coding(vector):
    """ Extend the given vector with complemented vector

    >>> vector = np.ones(5)
    >>> complement_coding(vector)
    array([ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.])
    """
    complemented = np.ones(len(vector)) - vector
    return np.append(vector, complemented)


def make_cluster_data():
    # TO DO: fixed mean, cov, num_cluster => parameterize!
    x, y = np.array([]), np.array([])
    mean = [[0.3, 0.2], [0.2, 0.7], [0.5, 0.5], [0.8, 0.4], [0.7, 0.8]]
    cov = [[0.001, 0], [0, 0.001]]
    for i in range(len(mean)):
        x_temp, y_temp = np.random.multivariate_normal(mean[i], cov, 30).T
        x = np.append(x, x_temp)
        y = np.append(y, y_temp)
    return x, y


def make_2d_seq_data(seq_len):
    x, y = make_cluster_data()
    indices = np.random.choice(len(x)-1, size=seq_len, replace=False)
    return [[np.array([x[idx]]), np.array([y[idx]])] for idx in indices]


def synthetic_data():
    x = np.arange(0.1, 1, 0.1)
    y = np.append(np.arange(0.1, 0.6, 0.1), np.arange(0.4, 0, -0.1))
    return x, y
