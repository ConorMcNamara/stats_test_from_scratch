from scipy.stats import t, rankdata
import numpy as np
from StatsTest.utils import _check_table
from math import sqrt


def pearson_test(x, y):
    """Found in scipy.stats as pearsonr
    Used to evaluate the pearson correlation between X and Y.

    Parameters
    ----------
    x: list or numpy array, 1-D
        Our "X" variable for determining the strength of our pearson correlation with
    y: list or numpy array, 1-D
        Our "Y" variable for determining the strength of our pearson correlation with

    Returns
    -------
    rho: float, -1 <= rho <= 1
        Our measure of pearson correlation between x and y
    p: float, 0 <= p <= 1
        How significant our observed pearson correlation is
    """
    x, y = _check_table(x, only_count=False), _check_table(y, only_count=False)
    if len(x) != len(y):
        raise ValueError("Cannot calculate correlation with datasets of different lengths")
    n = len(x)
    rho = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (sqrt(n * np.sum(np.power(x, 2)) - pow(np.sum(x), 2)) *
                                                       sqrt(n * np.sum(np.power(y, 2)) - pow(np.sum(y), 2)))
    t_stat = rho * sqrt((n - 2) / (1 - pow(rho, 2)))
    p = 2 * (1 - t.cdf(abs(t_stat), n - 2))
    return rho, p


def spearman_test(x, y):
    """Found in scipy.stats as spearmanr
    Used to evaluate the correlation between the ranks of "X" and "Y", that is, if there exists a
    monotonic relationship between X and Y.

    Parameters
    ----------
    x: list or numpy array, 1-D
        Our "X" variable for determining the strength of monotonic correlation with
    y: list or numpy array, 1-D
        Our "Y" variable for determining the strength of monotonic correlation with

    Returns
    -------
    rho: float, -1 <= t_stat <= 1
        Our measure of monotonic correlation between x and y
    p: float, 0 <= p <= 1
        How significant our observed monotonic correlation is
    """
    x, y = _check_table(x, only_count=False), _check_table(y, only_count=False)
    if len(x) != len(y):
        raise ValueError("Cannot calculate correlation with datasets of different lengths")
    df = len(x) - 2
    rank_x, rank_y = rankdata(x), rankdata(y)
    std_x, std_y = np.std(rank_x, ddof=1), np.std(rank_y, ddof=1)
    cov = np.cov(rank_x, rank_y)[0][1]
    rho = cov / (std_x * std_y)
    t_stat = rho * sqrt(df / (1 - pow(rho, 2)))
    p = 2 * (1 - t.cdf(abs(t_stat), df))
    return rho, p


def kendall_tau_test(x, y):
    """

    Parameters
    ----------
    x: list or numpy array, 1-D
        Our "X" variable for determining the strength of monotonic correlation with
    y: list or numpy array, 1-D
        Our "Y" variable for determining the strength of monotonic correlation with

    Returns
    -------
    rho: float, -1 <= t_stat <= 1
        Our measure of monotonic correlation between x and y
    p: float, 0 <= p <= 1
        How significant our observed monotonic correlation is
    """
    x, y = _check_table(x, only_count=False), _check_table(y, only_count=False)
    if len(x) != len(y):
        raise ValueError("Cannot calculate correlation with datasets of different lengths")
    n = len(x)
    perm = np.argsort(y)
    x, y = x[perm], y[perm]
    x = np.argsort(x, kind="mergesort")
