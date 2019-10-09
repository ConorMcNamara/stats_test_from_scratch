from scipy.stats import t, rankdata, norm
import numpy as np
from StatsTest.utils import _check_table
from math import sqrt, factorial


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


def kendall_tau_test(x, y, method='hypothesis'):
    """Found in scipy.stats as kendalltau
    Used to evaluate if two ordinal variables are correlated to one another.

    Parameters
    ----------
    x: list or numpy array, 1-D
        Our "X" ordinal variable
    y: list or numpy array, 1-D
        Our "Y" ordinal variable
    method: str, {hypothesis, significance, exact}, default is hypothesis
        Whether we want to run a hypothesis test, a significance test or an exact test.

    Returns
    -------
    tau: float, -1 <= tau <= 1
        Our measure of ordinal correlation, where +1 indicates that the two variables have identical rank while -1 indicates
        that the two variables have fully different ranks
    p: float, 0 <= p <= 1
        The likelihood that we would see this exhibited difference
    """
    x, y = _check_table(x, only_count=True), _check_table(y, only_count=True)
    if len(x) != len(y):
        raise ValueError("Cannot calculate correlation with datasets of different lengths")
    n = len(x)
    denom = n * (n - 1) / 2
    if method.casefold() not in ['hypothesis', 'significance', 'exact']:
        raise ValueError("Cannot determine type of test for Kendall Tau")

    def find_concordant_pairs(x, y):
        concordant, discordant = 0, 0
        unique_x, counts_x = np.unique(x, return_counts=True)
        unique_y, counts_y = np.unique(y, return_counts=True)
        t, u = counts_x[counts_x != 1], counts_y[counts_y != 1]
        for i in np.arange(len(x) - 1):
            x_data, y_data = x[i + 1:], y[i + 1:]
            x_val, y_val = x[i], y[i]
            concordant += len(np.intersect1d(np.where(x_val < x_data)[0], np.where(y_val < y_data)[0]))
            discordant += len(np.intersect1d(np.where(x_val != x_data)[0], np.where(y_val > y_data)[0]))
        return concordant, discordant, t, u

    x, y = x[np.argsort(x)], y[np.argsort(x)]
    concordant, discordant, t, u = find_concordant_pairs(x, y)
    n1 = np.sum(t * (t - 1) / 2)
    n2 = np.sum(u * (u - 1) / 2)
    tau = (concordant - discordant) / sqrt((denom - n1) * (denom - n2))
    if method.casefold() == 'hypothesis':
        v = 2 * (2 * n + 5) / (9 * n * (n - 1))
        p = 2 * (1 - norm.cdf(abs(tau) / sqrt(v)))
    elif method.casefold() == 'exact':
        if len(t) != 0 or len(u) != 0:
            raise AttributeError("Cannot run exact test when ties are present")
        n_choose = (n * (n - 1) // 2)
        c = min(discordant, n_choose - discordant)
        if n > 171:
            p = 0.0
        elif c == 0:
            p = 2.0 / factorial(n)
        elif c == 1:
            p = 2.0 / factorial(n - 1)
        elif n == 1:
            p = 1.0
        elif n == 2:
            p = 1.0
        else:
            new_vals = [1, 1] + [0] * (c - 1)
            for j in range(3, n + 1):
                old = new_vals[:]
                for k in range(1, min(j, c + 1)):
                    new_vals[k] += new_vals[k - 1]
                for k in range(j, c + 1):
                    new_vals[k] += new_vals[k - 1] - old[k - j]
            p = 2.0 * np.sum(new_vals) / factorial(n)
    else:
        v_0 = n * (n - 1) * (2 * n + 5)
        v_t, v_u = np.sum(t * (t - 1) * (2 * t + 5)), np.sum(u * (u - 1) * (2 * u + 5))
        v_1 = np.sum(t * (t - 1)) * np.sum(u * (u - 1)) / (2 * n * (n - 1))
        v_2 = np.sum(t * (t - 1) * (t - 2)) * np.sum(u * (u - 1) * (u - 2)) / (9 * n * (n - 1) * (n - 2))
        v = (v_0 - v_t - v_u) / 18 + v_1 + v_2
        p = 2 * (1 - norm.cdf(abs(concordant - discordant) / sqrt(v)))
    return tau, p


def point_biserial_correlation_test(x, y):
    """Found in scipy.stats as pointbiserialr

    x: list or numpy array, 1-D
        Our observations. These are expected to be continuous.
    y: list or numpy array, 1-D
        Our groupings variable, or masked array. Must only have two variables and be the same length as x

    Returns
    -------
    rho: float
        The measure of correlation between our two groups
    p: float
        The likelihood that our two groups would be correlated if both were derived from a t (if point) distribution
    """
    x = _check_table(x, only_count=False)
    y = _check_table(y, only_count=True)
    if len(x) != len(y):
        raise ValueError("X and Y must be of the same length")
    if len(np.unique(y)) != 2:
        raise AttributeError("Need to have two groupings for biseral correlation")
    group_0, group_1 = x[y == np.unique(y)[0]], x[y == np.unique(y)[1]]
    mu_1, mu_0 = np.mean(group_1), np.mean(group_0)
    n, n_1, n_0 = len(x), len(group_1), len(group_0)
    s = np.std(x, ddof=1)
    rho = ((mu_1 - mu_0) / s) * sqrt(n_1 * n_0 / (n * (n - 1)))
    t_val = rho * sqrt((n - 2) / (1 - pow(rho, 2)))
    p = 2 * (1 - t.cdf(abs(t_val), n - 2))
    return rho, p


def rank_biserial_correlation_test(x, y):
    """Not found in scipy.stats or statsmodels

    x: list or numpy array, 1-D
        Our observations. These are expected to be ordinal
    y: list or numpy array, 1-D
        Our groupings variable, or masked array. Must only have two variables and be the same length as x

    Returns
    -------
    rho: float
        The measure of correlation between our two groups
    p: float
        The likelihood that our two groups would be correlated if both were derived from a normal distribution
    """
    x, y = _check_table(x, only_count=True), _check_table(y, only_count=True)
    if len(x) != len(y):
        raise ValueError("X and Y must be of the same length")
    if len(np.unique(y)) != 2:
        raise AttributeError("Need to have two groupings for biseral correlation")
    group_0, group_1 = x[y == np.unique(y)[0]], x[y == np.unique(y)[1]]
    mu_1, mu_0 = np.mean(group_1), np.mean(group_0)
    n, n_1, n_0 = len(x), len(group_1), len(group_0)
    s = sqrt(n_1 * n_0 * (n + 1) / 12)
    rho = 2 * ((mu_1 - mu_0) / (n_1 + n_0))
    u_min = min((1 + rho) * n_1 * n_0 / 2, (1 - rho) * n_1 * n_0 / 2)
    mu = n_1 * n_0 / 2
    z = (u_min - mu) / s
    p = 2 * (1 - norm.cdf(abs(z)))
    return rho, p