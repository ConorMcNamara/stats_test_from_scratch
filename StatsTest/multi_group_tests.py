import numpy as np
from StatsTest.utils import _check_table, _sse
from scipy.stats import f, chi2
from statsmodels.stats.libqsturng import psturng
from math import sqrt


def levene_test(*args):
    """Found in scipy.stats as levene(center='mean')
    Used to determine if a variable/observation in multiple groups has equal variances across all groups.

    Parameters
    ----------
    args: list or numpy arrays
        The observed variable/observations for each group, organized into lists or numpy array

    Return
    ------
    w: float
        The W statistic, our measure of difference in variability, which is approximately F-distributed.
    p: float, 0 <= p <= 1
        The probability that our observed differences in variances could occur due to random sampling from a population
        of equal variance.
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
    Used instead of general levene test if we believe our data to be non-normal.

    Parameters
    ----------
    args: list or numpy arrays
        The observed variable/observations for each group, organized into lists or numpy array

    Return
    ------
    w: float
        The W statistic, our measure of difference in variability, which is approximately F-distributed.
    p: float, 0 <= p <= 1
        The probability that our observed differences in variances could occur due to random sampling from a population
        of equal variance.
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Need at least two groups to perform a Brown-Forsythe Test")
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
    Used to measure if multiple normal populations have the same mean. Note that this test is very sensitive to
    non-normal data, meaning that it should not be used unless we can verify that the data is normally distributed.

    Parameters
    ----------
    args: list or numpy arrays
        The observed measurements for each group, organized into lists or numpy array

    Return
    ------
    f_statistics: float
        The F statistic, or a measure of the ratio of data explained by the mean versus that unexplained by the mean
    p: float, 0 <= p <= 1
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
    This test is used to determine if multiple samples are from a population of equal variances. Note that this test
    is much more sensitive to data that is non-normal compared to Levene or Brown-Forsythe.

    Parameters
    ----------
    args: list or numpy arrays
        The observed measurements for each group, organized into lists or numpy array

    Return
    ------
    X: float
        The Chi statistic, or a measure of the observed difference in variances
    p: float, 0 <= p <= 1
        The probability that our observed differences in variances could occur due to random sampling from a population
        of equal variance.
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
    This test compares all possible pairs of means and determines if there are any differences in these pairs.

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


# def hartley_test(*args):
#     """Not found in either scipy or statsmodels
#     Used to determine if the variances between multiple groups are equal. Note that this test is very sensitive to data
#     that is non-normal and should only be used in instances where we can verify that all data points are normally
#     distributed.
#
#     Parameters
#     ----------
#     args: list or numpy arrays
#         The observed measurements for each group, organized into lists or numpy arrays
#
#     Return
#     ------
#     f_statistic: float
#         Our f_statistic, or the ratio of group with the highest variance with the group with the lowest variance
#     p: float
#         The likelihood that this ratio could occur from randomly sampling a population of equal variances.
#     """
#     k = len(args)
#     if k < 2:
#         raise AttributeError("Need at least two groups to perform Hartley's Test")
#     lengths = np.unique([len(arg) for arg in args])
#     if len(lengths) != 1:
#         raise AttributeError("Hartley's Test requires that all groups have the same number of observations")
#     variances = [np.var(arg, ddof=1) for arg in args]
#     max_var, min_var = np.max(variances), np.min(variances)
#     df = lengths[0] - 1
#     f_statistic = max_var / min_var
#     p = 1 - f.cdf(f_statistic, df, k)
#     return f_statistic, p


def cochran_q_test(*args):
    """Found in statsmodels as chochrans_q
    Used to determine if k treatments in a 2 way randomized block design have identical effects. Note that this test
    requires that there be only two variables encoded: a 1 for success and a 0 for failure.

    Parameters
    ----------
    args: list or numpy arrays
        Each array corresponds to all observations from a single treatment. That is, each array corresponds to a
        column in our table

    Return
    ------
    T: float
        Our T statistic
    p: float, 0 <= p <= 1
        The likelihood that our observed differences are due to chance
    """
    k = len(args)
    if k < 3:
        raise AttributeError("Cannot run Cochran's Q Test with less than 3 treatments")
    if len(np.unique(args)) > 2:
        raise AttributeError("Cochran's Q Test only works with binary variables")
    df = k - 1
    N = np.sum(args)
    all_data = np.vstack(args).T
    row_sum, col_sum = np.sum(all_data, axis=1), np.sum(all_data, axis=0)
    scalar = k * (k - 1)
    T = scalar * np.sum(np.power(col_sum - (N / k), 2)) / np.sum(row_sum * (k - row_sum))
    p = 1 - chi2.cdf(T, df)
    return T, p