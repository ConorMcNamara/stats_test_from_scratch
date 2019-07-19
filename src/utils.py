from numbers import Number
from math import sqrt, factorial
import numpy as np


def _standard_error(std, n):
    """Calculates the standard error given the standard deviation and length of data

    Parameters
    ----------
    std: number
        The standard deviation of our data
    n: int
        The length of our data

    Return
    ------
    The standard error, or our standard deviation divided by the square root of n.
    """
    assert isinstance(std, Number), "Cannot calculate standard error with std as type {}".format(type(std))
    assert isinstance(n, int), "Cannot calculate standard error with n as type {}".format(type(n))
    assert n > 0, "Cannot have non-positive length"
    return std / sqrt(n)


def _hypergeom_distribution(a, b, c, d):
    """Calculates the hyper-geometric distribution for a given a, b, c and d

    Parameters
    ----------
    a: int
        The top left corner value of our 2x2 matrix
    b: int
        The top right corner value of our 2x2 matrix
    c: int
        The bottom left corner of our 2x2 matrix
    d: int
        The bottom right corner of our 2x2 matrix

    Return
    ------
    The hyper-geometric distribution given a, b, c and d"""
    assert isinstance(a, int), "Cannot compute factorial with type {}".format(type(a))
    assert isinstance(b, int), "Cannot compute factorial with type {}".format(type(b))
    assert isinstance(c, int), "Cannot compute factorial with type {}".format(type(c))
    assert isinstance(d, int), "Cannot compute factorial with type {}".format(type(d))
    return (factorial(a + b) * factorial(c + d) * factorial(a + c) * factorial(b + d)) / \
           (factorial(a) * factorial(b) * factorial(c) * factorial(d) * factorial(a + b + c + d))


def _check_table(table, only_count=False):
    """Performs checks on our table to ensure that it is suitable for our statistical tests

    Parameters
    ----------
    table: list or numpy array
        The dataset we are applying our checks on
    only_count: bool
        Whether or not this dataset involves counts of instances

    Return
    ------
    table: numpy array
        The dataset, convered to a numpy array
    """
    if isinstance(table, list):
        table = np.array(table)
    elif isinstance(table, (np.ndarray, np.generic)):
        pass
    else:
        raise Exception("Data type {} is not supported".format(type(table)))
    if only_count:
        assert np.issubdtype(table.dtype, np.integer), "Cannot perform statistical test with non-integer counts"
    else:
        assert np.issubdtype(table.dtype, Number), "Cannot perform statistical test with non-numeric values"
    if only_count:
        assert np.all(table > 0), "Cannot have negative counts"
    return table


def _sse(sum_data, square_data, n_data):
    sum_data, n_data = _check_table(sum_data, False), _check_table(n_data, False)
    cm = np.power(np.sum(sum_data), 2) / sum(n_data)
    sst = np.sum(square_data) - cm
    ssr = np.sum(np.power(sum_data, 2) / n_data) - cm
    sse = sst - ssr
    return sse
