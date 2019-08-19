from numbers import Number
from math import sqrt, factorial
import numpy as np
from scipy.stats import binom


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
    if not isinstance(std, Number):
        raise TypeError("Cannot calculate standard error with standard deviation of type {}".format(type(n)))
    if not isinstance(n, int):
        raise TypeError("Cannot calculate standard error with n of type{}".format(type(n)))
    if n <= 0:
        raise ValueError("Cannot calculate standard error with n less than or equal to zero")
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
    The hyper-geometric distribution given a, b, c and d
    """
    if isinstance(a, int) and isinstance(b, int) and isinstance(c, int) and isinstance(d, int):
        pass
    elif not isinstance(a, np.integer) or not isinstance(b, np.integer) or not isinstance(c, np.integer) or not isinstance(d, np.integer):
        raise TypeError("Cannot compute factorials for non-integer values")
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
        The dataset, converted to a numpy array
    """
    if isinstance(table, list):
        table = np.array(table)
    elif isinstance(table, (np.ndarray, np.generic)):
        pass
    else:
        raise TypeError("Data type {} is not supported".format(type(table)))
    if only_count:
        if not np.issubdtype(table.dtype, np.integer):
            raise TypeError("Cannot perform statistical test with non-integer counts")
    else:
        if np.issubdtype(table.dtype, np.integer):
            pass
        elif np.issubdtype(table.dtype, np.float):
            pass
        else:
            raise TypeError("Cannot perform statistical test with non-numeric values")
    if only_count:
        if not np.all(table > 0):
            raise ValueError("Cannot have negative counts")
    return table


def _sse(sum_data, square_data, n_data):
    """Calculates the sum of squares for the errors

    Parameters
    ----------
    sum_data: list or numpy array
        An array containing the sum of each group of data.
    square_data: list or numpy array
        An array containing the sum of the squared differences of each group of data
    n_data: list or numpy array
        An array containing the length of each group of data

    Return
    ------
    sse: float
        The sum of squares of the errors
    """
    sum_data, square_data, n_data = _check_table(sum_data, False), _check_table(square_data, False), _check_table(n_data, False)
    if not np.all(square_data >= 0):
        raise ValueError("Cannot have negative square of numbers")
    if not np.all(n_data > 0):
        raise ValueError("Cannot have negative lengths")
    cm = np.power(np.sum(sum_data), 2) / sum(n_data)
    sst = np.sum(square_data) - cm
    ssr = np.sum(np.power(sum_data, 2) / n_data) - cm
    sse = sst - ssr
    return sse


def _right_extreme(n_instances, n_total, prob):
    counter = np.arange(n_instances, n_total + 1)
    p = np.sum(binom.pmf(counter, n_total, prob))
    return p


def _left_extreme(n_instances, n_total, prob):
    counter = np.arange(n_instances+1)
    p = np.sum(binom.pmf(counter, n_total, prob))
    return p
