from numbers import Number
from math import sqrt, factorial
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from scipy.stats import binom


def _standard_error(std: float, n: int) -> float:
    """Calculates the standard error given the standard deviation and length of data

    Parameters
    ----------
    std: float
        The standard deviation of our data
    n: int
        The length of our data

    Returns
    -------
    The standard error, or our standard deviation divided by the square root of n.
    """
    if not isinstance(std, Number):
        raise TypeError(
            "Cannot calculate standard error with standard deviation of type {}".format(
                type(n)
            )
        )
    if not isinstance(n, int):
        raise TypeError(
            "Cannot calculate standard error with n of type{}".format(type(n))
        )
    if n <= 0:
        raise ValueError(
            "Cannot calculate standard error with n less than or equal to zero"
        )
    return std / sqrt(n)


def _hypergeom_distribution(a: int, b: int, c: int, d: int) -> float:
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

    Returns
    -------
    The hyper-geometric distribution given a, b, c and d
    """
    if (
        isinstance(a, int)
        and isinstance(b, int)
        and isinstance(c, int)
        and isinstance(d, int)
    ):
        pass
    elif (
        not isinstance(a, np.integer)
        or not isinstance(b, np.integer)
        or not isinstance(c, np.integer)
        or not isinstance(d, np.integer)
    ):
        raise TypeError("Cannot compute factorials for non-integer values")
    return (
        factorial(a + b) * factorial(c + d) * factorial(a + c) * factorial(b + d)
    ) / (
        factorial(a)
        * factorial(b)
        * factorial(c)
        * factorial(d)
        * factorial(a + b + c + d)
    )


def _check_table(
    table: Union[Sequence, np.ndarray, pd.Series, pd.DataFrame, pd.SparseDtype],
    only_count: bool = False,
) -> np.ndarray:
    """Performs checks on our table to ensure that it is suitable for our statistical tests

    Parameters
    ----------
    table: list or numpy array
        The dataset we are applying our checks on
    only_count: bool, default=False
        Whether or not this dataset involves counts of instances

    Returns
    -------
    table: numpy array
        The dataset, converted to a numpy array
    """
    if isinstance(table, list):
        table = np.array([np.array(xi) for xi in table])
    elif isinstance(table, (np.ndarray, np.generic)):
        pass
    elif isinstance(
        table,
        (pd.Series, pd.DataFrame, pd.SparseDataFrame, pd.SparseSeries, pd.SparseArray),
    ):
        table = np.array(table)
    else:
        raise TypeError("Data type {} is not supported".format(type(table)))
    if only_count:
        for tab in table:
            if not np.issubdtype(tab.dtype, np.integer):
                raise TypeError(
                    "Cannot perform statistical test with non-integer counts"
                )
            if not np.all(tab >= 0):
                raise ValueError("Cannot have negative counts")
    else:
        if np.issubdtype(table.dtype, np.integer):
            pass
        elif np.issubdtype(table.dtype, np.dtype(float).type):
            pass
        else:
            raise TypeError("Cannot perform statistical test with non-numeric values")
    return table


def _sse(
    sum_data: Union[Sequence, np.ndarray],
    square_data: Union[Sequence, np.ndarray],
    n_data: Union[Sequence, np.ndarray],
) -> float:
    """Calculates the sum of squares for the errors

    Parameters
    ----------
    sum_data: list or numpy array
        An array containing the sum of each group of data.
    square_data: list or numpy array
        An array containing the sum of the squared differences of each group of data
    n_data: list or numpy array
        An array containing the length of each group of data

    Returns
    -------
    sse: float
        The sum of squares of the errors
    """
    sum_data, square_data, n_data = (
        _check_table(sum_data, False),
        _check_table(square_data, False),
        _check_table(n_data, False),
    )
    if not np.all(square_data >= 0):
        raise ValueError("Cannot have negative square of numbers")
    if not np.all(n_data > 0):
        raise ValueError("Cannot have negative lengths")
    cm = np.power(np.sum(sum_data), 2) / sum(n_data)
    sst = np.sum(square_data) - cm
    ssr = np.sum(np.power(sum_data, 2) / n_data) - cm
    sse = sst - ssr
    return sse


def _right_extreme(n_instances: int, n_total: int, prob: float) -> float:
    """Used for a binomial problem. Calculates the exact likelihood of finding observations as and more extreme
    than our observed value

    Parameters
    ----------
    n_instances: int
        The number of observed successes in our binomial problem
    n_total: int
        The total number of observations
    prob: float, 0<=p<=1
        The expected probability of success

    Returns
    -------
    p: float
        The exact likelihood that we would find observations more extreme than our observed number of
        success
    """
    counter = np.arange(n_instances, n_total + 1)
    p = np.sum(binom.pmf(counter, n_total, prob))
    return p


def _left_extreme(n_instances: int, n_total: int, prob: float) -> float:
    """Used for a binomial problem. Calculates the exact likelihood of finding observations as and less extreme
    than our observed value

    Parameters
    ----------
    n_instances: int
        The number of observed successes in our binomial problem
    n_total: int
        The total number of observations
    prob: float, 0<=p<=1
        The expected probability of success

    Returns
    -------
    p: float
        The exact likelihood that we would find observations less extreme than our observed number of
        success
    """
    counter = np.arange(n_instances + 1)
    p = np.sum(binom.pmf(counter, n_total, prob))
    return p


def _skew(data: Union[Sequence, np.ndarray]) -> float:
    """Calculates the skew (third moment) of the data

    Parameters
    ----------
    data: list or numpy array
        Contains all measured observations that we want to evaluate the skew on

    Returns
    -------
    skew: float
        Our measure of the asymmetry of the probability distribution of a real-valued random variable about its mean
    """
    x_bar = np.mean(data)
    n = len(data)
    mu_3 = np.sum(np.power(data - x_bar, 3)) / n
    sigma_3 = np.sum(np.power(np.var(data), 1.5))
    skew = mu_3 / sigma_3
    return skew


def _kurtosis(data: Union[Sequence, np.ndarray]) -> float:
    """Calculates the kurtosis (fourth moment) of the data

    Parameters
    ----------
    data: list or numpy array
        Contains all measured observations that we want to evaluate the kurtosis on

    Returns
    -------
    kurtosis: float
        The sharpness of the peak of a frequency-distribution curve in our data
    """
    x_bar = np.mean(data)
    n = len(data)
    mu_4 = np.sum(np.power(data - x_bar, 4)) / n
    sigma_4 = np.sum(np.power(np.var(data), 2))
    kurtosis = mu_4 / sigma_4
    return kurtosis


def _autocorr(
    data: Union[Sequence, np.ndarray], lags: Union[Sequence, np.ndarray]
) -> np.ndarray:
    """Calculates the autocorrelation for a given time series dataset given a set amount of lags

    Parameters
    ----------
    data: list or numpy array
        Observations from a time series dataset
    lags: list or numpy array
        The length of our time lags that we wish to calculate the autocorrelation with

    Returns
    -------
    corr: numpy array
        The autocorrelation for a dataset for each given lag
    """
    mean = np.mean(data)
    var = np.var(data)
    xp = data - mean
    corr = [
        1.0 if lag == 0 else np.sum(xp[lag:] * xp[:-lag]) / len(data) / var
        for lag in lags
    ]
    return np.array(corr)


def _rle(
    arr: Union[Sequence, np.ndarray]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Similar to R rle function, runs length encoding for a binary sequence.

     Parameters
     ----------
     arr: list or numpy array
        Our observed binary sequence

    Returns
    -------
    z: numpy array
        The length of each run/sequence
    p: numpy array
        The starting position of each run/sequence
    ia[i]: numpy array
        Whether the run/sequence belonged to our False or True condition
    """
    ia = np.array(arr)  # force numpy
    n = len(ia)
    if n == 0:
        return None, None, None
    else:
        y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return z, p, ia[i]
