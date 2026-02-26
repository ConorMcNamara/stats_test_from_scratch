from math import sqrt

import numpy as np
from scipy.stats import chi2, f, norm

from StatsTest.utils import _check_table


def levene_test(*args) -> tuple[float, float]:
    """Found in scipy.stats as levene(center='mean')

    Used to determine if a variable/observation in multiple groups has equal variances across all groups. In short, does
    each group have equal variance?

    Parameters
    ----------
    args : list or numpy arrays, 1-D
        The observed variable/observations for each group, organized into lists or numpy array

    Returns
    -------
    w : float
        The W statistic, our measure of difference in variability, which is approximately F-distributed.
    p : float, 0 <= p <= 1
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


def brown_forsythe_test(*args) -> tuple[float, float]:
    """Found in scipy.stats as levene(center='median')

    Used instead of general levene test if we believe our data to be non-normal.

    Parameters
    ----------
    args : list or numpy arrays, 1-D
        The observed variable/observations for each group, organized into lists or numpy array

    Returns
    -------
    w : float
        The W statistic, our measure of difference in variability, which is approximately F-distributed.
    p : float, 0 <= p <= 1
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


def one_way_f_test(*args) -> tuple[float, float]:
    """Found in scipy.stats as f_oneway

    Used to measure if multiple normal populations have the same mean. Note that this test is very sensitive to
    non-normal data, meaning that it should not be used unless we can verify that the data is normally distributed.

    Parameters
    ----------
    args : list or numpy arrays, 1-D
        The observed measurements for each group, organized into lists or numpy array

    Return
    ------
    f_statistics: float
        The F statistic, or a measure of the ratio of data explained by the mean versus that unexplained by the mean
    p: float, 0 <= p <= 1
        The likelihood that our observed ratio would occur, in a population with the same mean, due to chance
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


def bartlett_test(*args) -> tuple[float, float]:
    """Found in scipy.stats as bartlett

    This test is used to determine if multiple samples are from a population of equal variances. Note that this test
    is much more sensitive to data that is non-normal compared to Levene or Brown-Forsythe.

    Parameters
    ----------
    args : list or numpy arrays, 1-D
        The observed measurements for each group, organized into lists or numpy array

    Returns
    -------
    X : float
        The Chi statistic, or a measure of the observed difference in variances
    p : float, 0 <= p <= 1
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


def cochran_q_test(*args) -> tuple[float, float]:
    """Found in statsmodels as chochrans_q

    Used to determine if k treatments in a 2 way randomized block design have identical effects. Note that this test
    requires that there be only two variables encoded: 1 for success and 0 for failure.

    Parameters
    ----------
    args : list or numpy arrays, 1-D
        Each array corresponds to all observations from a single treatment. That is, each array corresponds to a
        column in our table (Treatment_k), if we were to look at https://en.wikipedia.org/wiki/Cochran%27s_Q_test

    Returns
    -------
    T : float
        Our T statistic
    p : float, 0 <= p <= 1
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


def jonckheere_trend_test(*args, **kwargs) -> tuple[float, float]:
    """This test is not found in scipy or statsmodels

    This test is used to determine if the population medians for each group have an a priori ordering.
    Note that the default alternative hypothesis is that median_1 <= median_2 <= median_3 <= ... <= median_k, with at
    least one strict inequality.

    Parameters
    ----------
    args : list or numpy array, 1-D
        List or numpy arrays, where each array constitutes a population/group, and within that group are their responses.
        For example, based on the numeric example found here: https://en.wikipedia.org/wiki/Jonckheere%27s_trend_test,
        the first array would be the measurements found in "contacted" and the second array would the measurements found
        in "bumped" and the third array would be the measurements found in "smashed"
    kwargs : str
        Our alternative hypothesis. The two options are "greater" and "less", indicating the a priori ordering. Default
        is less

    Returns
    -------
    z_statistic : float
        A measure of the difference in trends for the median of each group
    p : float, 0 <= p <= 1
        The likelihood that our trend could be found if each group were randomly sampled from a population with the same
        medians.
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Cannot run Jonckheere Test with less than 2 groups")
    u = [len(arg) for arg in args]
    if len(np.unique(u)) != 1:
        raise AttributeError("Jonckheere Test requires that each group have the same number of observations")
    if "alternative" in kwargs:
        alternative = kwargs.get("alternative")
        if not isinstance(alternative, str):
            raise TypeError("Cannot have alternative hypothesis with non-string value")
        if alternative.casefold() not in ["greater", "less"]:
            raise ValueError("Cannot discern alternative hypothesis")
    else:
        alternative = "less"
    all_data = np.vstack([sorted(arg) for arg in args]).T
    if alternative.casefold() == "greater":
        all_data = np.flip(all_data, axis=1)
    t = np.unique(all_data, return_counts=True)[1]
    n = all_data.shape[0] * k
    p, q = 0, 0
    for col in range(k - 1):
        for row in range(all_data.shape[0]):
            val = all_data[row, col]
            right_side = [False] * (col + 1) + [True] * (k - col - 1)
            right_data = np.compress(right_side, all_data, axis=1)
            p += len(np.where(right_data > val)[0])
            q += len(np.where(right_data < val)[0])
    s = p - q
    sum_t_2, sum_t_3 = np.sum(np.power(t, 2)), np.sum(np.power(t, 3))
    sum_u_2, sum_u_3 = np.sum(np.power(u, 2)), np.sum(np.power(u, 3))
    part_one = (2 * (pow(n, 3) - sum_t_3 - sum_u_3) + 3 * (pow(n, 2) - sum_t_2 - sum_u_2) + 5 * n) / 18
    part_two = (sum_t_3 - 3 * sum_t_2 + 2 * n) * (sum_u_3 - 3 * sum_u_2 + 2 * n) / (9 * n * (n - 1) * (n - 2))
    part_three = (sum_t_2 - n) * (sum_u_2 - n) / (2 * n * (n - 1))
    var_s = part_one + part_two + part_three
    z_statistic = s / sqrt(var_s)
    p = 1 - norm.cdf(z_statistic)
    return z_statistic, p


def mood_median_test(*args, **kwargs) -> tuple[float, float]:
    """Found in scipy.stats as median_test

    This test is used to determine if two or more samples/observations come from a population with the same median.

    Parameters
    ----------
    args : list or numpy arrays, 1-D
       List or numpy arrays, where each array constitutes a number of observations in a population/group.
    kwargs : str
        "alternative": Our alternative hypothesis. The three options are "greater", "less" or "two-sided', used to determine
        whether we expect our data to favor being greater, less than or different from the median.
        Default is two-sided.
        "handle_med": How we handle the median value. The three options are "greater", "less' or "ignore". If greater,
        median value is added to values above median. If less, median value is added to values below median. If ignore,
        median value is not added at all. Default is "less".

    Returns
    -------
    X : float
        Our Chi Statistic measuring the difference of our groups compared to the median
    p : float, 0 <= p <= 1
        The likelihood that our observed differences in medians are due to chance
    """
    if len(args) < 2:
        raise AttributeError("Cannot run Median Test with less than 2 groups")
    all_data = np.concatenate(args)
    med = np.median(all_data)
    if "alternative" in kwargs:
        alternative = kwargs.get("alternative").casefold()
        if alternative not in ["greater", "less", "two-sided"]:
            raise ValueError("Cannot discern alternative hypothesis")
    else:
        alternative = "two-sided"
    if "handle_med" in kwargs:
        handle_med = kwargs.get("handle_med").casefold()
        if handle_med not in ["greater", "less", "ignore"]:
            raise ValueError("Cannot discern how to handle median value")
    else:
        handle_med = "less"
    above_med, below_med = [], []
    # To-do: see if I can simplify this logic by using vectorized functions and eliminate the for-loop
    if handle_med == "less":
        for arg in args:
            arg = _check_table(arg, only_count=False)
            above_med.append(np.sum(arg > med))
            below_med.append(np.sum(arg <= med))
    elif handle_med == "greater":
        for arg in args:
            arg = _check_table(arg, only_count=False)
            above_med.append(np.sum(arg >= med))
            below_med.append(np.sum(arg < med))
    else:
        for arg in args:
            arg = _check_table(arg, only_count=False)
            above_med.append(np.sum(arg > med))
            below_med.append(np.sum(arg < med))
    cont_table = np.vstack([above_med, below_med])
    row_sum, col_sum = np.sum(cont_table, axis=1), np.sum(cont_table, axis=0)
    expected = np.matmul(np.transpose(row_sum[np.newaxis]), col_sum[np.newaxis]) / np.sum(row_sum)
    X = np.sum(pow(cont_table - expected, 2) / expected)
    df = len(args) - 1
    if alternative == "two-sided":
        p = 2 * (1 - chi2.cdf(X, df))
    elif alternative == "less":
        p = 1 - chi2.cdf(X, df)
    else:
        p = chi2.cdf(X, df)
    return X, p
