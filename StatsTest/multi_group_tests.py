import numpy as np
from StatsTest.utils import _check_table, _sse
from scipy.stats import f, chi2
from statsmodels.stats.libqsturng import psturng
from math import sqrt


def levene_test(*args):
    """Found in scipy.stats as levene(center='mean')

    Parameters
    ----------
    args: list or numpy arrays
        The observed measurements for each group, organized into lists or numpy array

    Return
    ------
    w: float
        The W statistic
    p: float
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Need at least two groups to perform a Levene Test")
    n_i, z_bar, all_z_ij, z_bar_condensed = [], [], [], []
    for obs in args:
        obs = _check_table(obs, False)
        n_i = np.append(n_i, len(obs))
        z_ij = abs(obs - np.mean(obs))
        all_z_ij = np.append(all_z_ij, z_ij)
        z_bar = np.append(z_bar, np.repeat(np.mean(z_ij), len(obs)))
        z_bar_condensed = np.append(z_bar_condensed, np.mean(z_ij))
    scalar = (np.sum(n_i) - k) / (k - 1)
    w = scalar * np.sum(n_i * np.power(z_bar_condensed - np.mean(z_bar), 2)) / np.sum(np.power(all_z_ij - z_bar, 2))
    p = 1 - f.cdf(w, k - 1, np.sum(n_i) - k)
    return w, p


def brown_forsythe_test(*args):
    """Found in scipy.stats as levene(center='median')

    Parameters
    ----------
    args: list or numpy arrays
        The observed measurements for each group, organized into lists or numpy array

    Return
    ------
    w: float
        The W statistic
    p: float
        The likelihood that our observed differences occur due to chance
    """
    k = len(args)
    if k < 2:
        raise AttributeError( "Need at least two groups to perform a Brown-Forsythe Test")
    n_i, z_bar, all_z_ij, z_bar_condensed = [], [], [], []
    for obs in args:
        obs = _check_table(obs, False)
        n_i = np.append(n_i, len(obs))
        z_ij = abs(obs - np.median(obs))
        all_z_ij = np.append(all_z_ij, z_ij)
        z_bar = np.append(z_bar, np.repeat(np.mean(z_ij), len(obs)))
        z_bar_condensed = np.append(z_bar_condensed, np.mean(z_ij))
    scalar = (np.sum(n_i) - k) / (k - 1)
    w = scalar * np.sum(n_i * np.power(z_bar_condensed - np.mean(z_bar), 2)) / np.sum(np.power(all_z_ij - z_bar, 2))
    p = 1 - f.cdf(w, k - 1, np.sum(n_i) - k)
    return w, p


def one_way_f_test(*args):
    """Found in scipy.stats as f_oneway

    Parameters
    ----------
    args: list or numpy arrays
        The observed measurements for each group, organized into lists or numpy array

    Return
    ------
    f_statistics: float
        The F statistic
    p: float
        The likelihood that our observed differences occur due to chance
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Need at least two groups to perform a one-way F Test")
    n_i, y_bar, all_y_ij, y_bar_condensed = [], [], [], []
    all_y_ij = np.hstack(args)
    for obs in args:
        obs = _check_table(obs, False)
        n_i = np.append(n_i, len(obs))
        obs_mean = np.mean(obs)
        y_bar_condensed = np.append(y_bar_condensed, obs_mean)
        y_bar = np.append(y_bar, np.repeat(obs_mean, len(obs)))
    explained_variance = np.sum(n_i * np.power(y_bar_condensed - np.mean(all_y_ij), 2) / (k - 1))
    unexplained_variance = np.sum(np.power(all_y_ij - y_bar, 2) / (np.sum(n_i) - k))
    f_statistic = explained_variance / unexplained_variance
    p = 1 - f.cdf(f_statistic, k - 1, np.sum(n_i) - k)
    return f_statistic, p


def bartlett_test(*args):
    """Found in scipy.stats as bartlett

    Parameters
    ----------
    args: list or numpy arrays
        The observed measurements for each group, organized into lists or numpy array

    Return
    ------
    X: float
        The Chi statistic
    p: float
        The likelihood that our observed differences occur due to chance
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Need at least two groups to perform the Bartlett Test")
    n_i, var_i = [], []
    for obs in args:
        obs = _check_table(obs)
        n_i = np.append(n_i, len(obs))
        var_i = np.append(var_i, np.var(obs, ddof=1))
    pooled_variance = np.sum((n_i - 1) * var_i) / (np.sum(n_i) - k)
    top = (np.sum(n_i) - k) * np.log(pooled_variance) - np.sum((n_i - 1) * np.log(var_i))
    bottom = 1 + (1 / (3 * (k - 1))) * (np.sum(1 / (n_i - 1)) - (1 / (np.sum(n_i) - k)))
    X = top / bottom
    p = 1 - chi2.cdf(X, k - 1)
    return X, p


def tukey_range_test(*args):
    """Found in statsmodels as pairwise_tukeyhsd

    Parameters
    ----------
    args: list or numpy arrays
        The observed measurements for each group, organized into lists or numpy arrays

    Return
    ------
    results: list
        A list of lists containing 3 attributes:
            1) The groups being compared
            2) The Q Statistic
            3) p, or the likelihood our observed differences are due to chance
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Need at least two groups to perform Tukey Range Test")
    mean_i, groups, n_i, sum_data, square_data, results = [], [], [], [], [], []
    i = 0
    for obs in args:
        obs = _check_table(obs, False)
        mean_i = np.append(mean_i, np.mean(obs))
        groups = np.append(groups, i)
        n_i = np.append(n_i, len(obs))
        sum_data = np.append(sum_data, np.sum(obs))
        square_data = np.append(square_data, np.power(obs, 2))
        i += 1
    df = sum(n_i) - k
    sse = _sse(sum_data, square_data, n_i)
    for group in np.unique(groups):
        group = int(group)
        for next_group in range(group+1, len(np.unique(groups))):
            mean_a, mean_b = mean_i[group], mean_i[next_group]
            n_a, n_b = n_i[group], n_i[next_group]
            difference = abs(mean_a - mean_b)
            std_group = sqrt(sse / df / min(n_a, n_b))
            q = difference / std_group
            p = psturng(q, k, df)
            results.append(['group {} - group {}'.format(group, next_group), q, p])
    return results


def cochran_test(*args):
    """Found in statsmodels as chochrans_q

    Parameters
    ----------
    args: list or numpy arrays
        The observed measurements for each group, organized into lists or numpy arrays

    Return
    ------
    T: float
        Our T statistic
    p: float
        The likelihood that our observed differences are due to chance
    """
    k = len(args)
    if k < 3:
        raise AttributeError("Need at least 3 groups to perform Cochran's Q Test")
    if len(np.unique(args)) > 2:
        raise AttributeError("Cochran's Q Test only works with binary variables")
    df = k - 1
    N = np.mean(args)
    all_data = np.vstack(_check_table(args, False)).T
    col_sum, row_sum = np.sum(all_data, axis=1), np.sum(all_data, axis=0)
    scalar = k * (k - 1)
    T = scalar * np.sum(np.power(row_sum - (N / k)), 2) / np.sum(col_sum - (k - col_sum))
    p = 1 - chi2.cdf(T, df)
    return T, p