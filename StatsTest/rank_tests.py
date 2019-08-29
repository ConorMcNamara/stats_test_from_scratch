from StatsTest.utils import _check_table
from scipy.stats import rankdata, norm, chi2
import numpy as np
from math import sqrt
from scipy.stats.statlib import gscale


def two_sample_mann_whitney_test(data_1, data_2, alternative='two-sided'):
    """This test can be found in scipy.stats as mannwhitneyu
    Used when we want to test whether or not the distribution of two ordinal response variables are equal or not,
    assuming that each sample is independent of one another.

    Parameters
    ----------
    data_1: list or numpy array, 1-D
        The observed sample for ordinal response variable 1
    data_2: list or numpy array, 1-D
        The observed sample for ordinal response variable 2
    alternative: str, default is two-sided
        Our alternative hypothesis. It can be two-sided, greater or less

    Return
    ------
    u: float
        The U statistic for our observed differences in the two ordinal responses
    p: float, 0 <= p <= 1
        The likelihood that the observed differences are due to chance
    """
    if alternative.casefold() not in ['two-sided', 'greater', 'less']:
        raise ValueError("Cannot determine method for alternative hypothesis")
    data_1, data_2 = _check_table(data_1, False), _check_table(data_2, False)
    combined_data = rankdata(np.concatenate([data_1, data_2]))
    combined_data_len = len(combined_data)
    data_1_len, data_2_len = len(data_1), len(data_2)
    data_1_rank = np.sum(combined_data[:len(data_1)])
    data_2_rank = np.sum(combined_data[len(data_1):])
    u1 = data_1_rank - ((data_1_len * (data_1_len + 1)) / 2)
    u2 = data_2_rank - ((data_2_len * (data_2_len + 1)) / 2)
    u_mean = (u1 + u2) / 2
    if alternative.casefold() == 'two-sided':
        u = np.min([u1, u2])
    elif alternative.casefold() == 'greater':
        u = u1
    else:
        u = u2
    T = np.unique(u, return_counts=True)[1]
    sum_T = np.sum(np.power(T, 3) - T) / (combined_data_len * (combined_data_len - 1))
    u_sd = sqrt((data_1_len * data_2_len / 12) * (combined_data_len + 1 - sum_T))
    z_score = (u - u_mean) / u_sd
    if alternative.casefold() == 'two-sided':
        p = 2 * (1 - norm.cdf(abs(z_score)))
    elif alternative.casefold() == 'greater':
        p = 1 - norm.cdf(z_score)
    else:
        p = norm.cdf(z_score)
    return u, p


def two_sample_wilcoxon_test(data_1, data_2, alternative='two-sided', handle_zero='wilcox'):
    """This test can be found in scipy.stats as wilcoxon
    Used when we want to compare two related or paired samples, or repeated measurements, and see if their population
    mean ranks differ. Also used when we cannot assume that the samples are normally distributed.

    Parameters
    ----------
    data_1: list or numpy array, 1-D
        The first sample or repeated measure
    data_2: list or numpy array, 1-D
        The second sample or repeated measure
    alternative: str, default is two-sided
        Our alternative hypothesis. It can be two-sided, greater or less
    handle_zero: str, default is wilcox
        How we treat differences of zero. It can be either wilcox (ignore) or pratt

    Return
    ------
    w_value: float
        The W statistic for our observed differences in mean ranks
    p: float, 0 <= p <= 1
        The likelihood that the observed mean rank differences would be found in two datasets sampled from the same
        population
    """
    if alternative.casefold() not in ['two-sided', 'greater', 'less']:
        raise ValueError("Cannot determine method for alternative hypothesis")
    if handle_zero.casefold() not in ['wilcox', 'pratt']:
        raise ValueError("Cannot determine how to handle differences of zero")
    if len(data_1) != len(data_2):
        raise AttributeError("Cannot perform signed wilcoxon test on unpaired data")
    data_1, data_2 = _check_table(data_1, False), _check_table(data_2, False)
    diff = data_1 - data_2
    if handle_zero.casefold() == 'wilcox':
        assert np.sum(diff == 0) != len(data_1), "Cannot perform wilcoxon test when all differences are zero"
        diff = np.compress(np.not_equal(diff, 0), diff)
    n = len(diff)
    abs_diff, sign_diff = np.abs(diff), np.sign(diff)
    rank = rankdata(abs_diff)
    if handle_zero.casefold() == "pratt":
        zero_ranks = np.not_equal(abs_diff, 0)
        sign_diff, rank = np.compress(zero_ranks, sign_diff), np.compress(zero_ranks, rank)
    w_value = np.sum(sign_diff * rank)
    std = sqrt(n * (n + 1) * (2 * n + 1) / 6)
    z_score = w_value / std
    if alternative.casefold() == 'two-sided':
        p = 2 * (1 - norm.cdf(abs(z_score)))
    elif alternative.casefold() == 'greater':
        p = 1 - norm.cdf(z_score)
    else:
        p = norm.cdf(z_score)
    return w_value, p


def friedman_test(*args):
    """This can be found in scipy.stats as friedmanchisquare
    Used to detect the differences in treatments across multiple test attempts. For example:
    Suppose there are n wine judges each rate k different wines. Are any of the k wines ranked consistently higher or
    lower than the others?

    Parameters
    ----------
    args: list or numpy array, 1-D
        An array containing the observations for each treatment (In the example above, each array would represent one
        of the judges ratings for every k wine).

    Return
    ------
    q: float
        Our Q statistic, or a measure of the difference between our expected result and the observed outcomes
    p: float, 0 <= p <= 1
        The likelihood that our observed differences between each treatment is due to chance
    """
    k = len(args)
    if k < 3:
        raise AttributeError("Friedman Test not appropriate for {} levels".format(k))
    df = k - 1
    all_data = np.vstack(args).T
    n = len(all_data)
    rank = np.apply_along_axis(rankdata, 1, all_data)
    r_bar = np.mean(rank, axis=0)
    scalar = (12 * n) / (k * (k + 1))
    q = scalar * np.sum(np.power(r_bar - ((k + 1) / 2), 2))
    p = 1 - chi2.cdf(q, df)
    return q, p


def page_trend_test(*args, **kwargs):
    """Not found in either scipy or statsmodels
    Used to evaluate whether or not there is a monotonic trend within each treatment/condition. Note that the default
    alternative hypothesis is that treatment_1 >= treatment_2 >= treatment_3 >= .... >= treatment_n, with at least
    one strict inequality.

    Parameters
    ----------
    args: list or numpy array, 1-D
        Here, each list/array represents a treatment/condition, where within each condition are the results of each
        subject in response to said treatment.
    kwargs: str
        Used to determine whether or not we are evaluating a monotonically increasing trend or a monotonically decreasing
        trend.
        Options for 'alternative' are greater [increasing trend] or less [decreasing trend]. Default is greater

    Return
    ------
    l: float
        Our measure of the strength of the trend, based on our alternative hypothesis
    p: float, 0 <= p <= 1
        The likelihood that our observed trend would happen if we were to randomly sample from a population of equal
        tendencies/trends.
    """
    n_conditions = len(args)
    if n_conditions < 3:
        raise AttributeError("Page Test not appropriate for {} levels".format(n_conditions))
    lengths = [len(arg) for arg in args]
    if len(np.unique(lengths)) != 1:
        raise AttributeError("Page Test requires that each level have the same number of observations")
    if "alternative" in kwargs:
        alternative = kwargs.get("alternative")
        if not isinstance(alternative, str):
            raise ValueError("Cannot have alternative hypothesis with non-string value")
        if alternative.casefold() not in ['greater', 'less']:
            raise ValueError("Cannot discern alternative hypothesis")
    else:
        alternative = "greater"
    k_subjects = lengths[0]
    if alternative.casefold() == 'greater':
        n_rank = np.arange(n_conditions, 0, -1)
    else:
        n_rank = np.arange(n_conditions) + 1
    rank_data = []
    for i in range(k_subjects):
        rank_data.append(rankdata([arg[i] for arg in args]))
    x_i = np.sum(rank_data, axis=0)
    L = np.sum(n_rank * x_i)
    top = pow(12 * L - (3 * k_subjects * n_conditions * pow(n_conditions + 1, 2)), 2)
    bottom = k_subjects * pow(n_conditions, 2) * (pow(n_conditions, 2) - 1) * (n_conditions + 1)
    x = top / bottom
    p = (1 - chi2.cdf(x, 1)) / 2
    return L, p


def kruskal_wallis_test(*args):
    """Found in scipy.stats as kruskal
    This test is used to determine whether or not two or more samples originate from the same distribution.
    Note that this requires the samples to be independent of one another, and that it only tells us if there is a
    difference, not where the difference(s) occur.

    Parameters
    ----------
    args: list or numpy arrays, 1-D
        Each list/array represents a group or a sample, and within that array contains the measurements for said group
        or array

    Returns
    -------
    H: float
        The statistic measuring the difference in distribution
    p: float, 0 <= p <= 1
        The likelihood that our observed differences could occur if they were randomly sampled from
        a population with the same distribution
    """
    g = len(args)
    if g < 2:
        raise AttributeError("Cannot run Kruskal-Wallis Test with less than 2 groups")
    rank_data = rankdata(args)
    r_bar = np.mean(rank_data)
    n_i = [len(arg) for arg in args]
    n = np.sum(n_i)
    rank_data_split = np.split(rank_data, np.cumsum(n_i)[0:len(n_i) - 1])
    rank_data_s_mean = np.mean(rank_data_split, axis=1)
    top = np.sum(n_i * np.power(rank_data_s_mean - r_bar, 2))
    bottom = np.sum(np.power(rank_data_split - r_bar, 2))
    H = (n - 1) * top / bottom
    p = 1 - chi2.cdf(H, g - 1)
    return H, p


def fligner_kileen_test(*args, **kwargs):
    """Found in scipy.stats as fligner
    Used to test the homogeneity of group variances, making no assumptions of the data distribution beforehand

    Parameters
    ----------
    args: list or numpy array, 1-D
        Each list represents all observations in a group that we wish to test their variances on
    kwargs: str
        Whether we are measuring our residuals as distance from the mean or median. Default is center='median'

    Returns
    -------
    x: float
        The test statistic measuring the differences in variances between all groups
    p: float, 0 <= p <= 1
        The likelihood that our differences would be observed if all groups were drawn from the same population
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Cannot perform Fligner-Kileen Test with less than 2 groups")
    if "center" in kwargs:
        center = kwargs.get("center").casefold()
        if center not in ['median', 'mean']:
            raise ValueError("Cannot discern how to center the data")
    else:
        center = 'median'
    if center == 'median':
        m_i = [np.median(arg) for arg in args]
    else:
        m_i = [np.mean(arg) for arg in args]
    n_i = [len(arg) for arg in args]
    n = np.sum(n_i)
    resids = np.abs([args[i] - m_i[i] for i in range(k)])
    all_resids = np.hstack([resid for resid in resids])
    rank_all_resids = rankdata(all_resids)
    normalized_rank = norm.ppf(rank_all_resids / (2 * (n + 1)) + 0.5)
    normalized_split = np.split(normalized_rank, np.cumsum(n_i)[0:len(n_i) - 1])
    var_nr = np.var(normalized_rank, ddof=1)
    x_bar = np.mean(normalized_rank)
    x_j = np.mean(normalized_split, axis=1)
    x = np.sum(n_i * np.power(x_j - x_bar, 2)) / var_nr
    p = 1 - chi2.cdf(x, k - 1)
    return x, p


def ansari_bradley_test(data_1, data_2, alternative='two-sided'):
    """Found in scipy.stats as ansari
    Used to measure the level of dispersion (the distance from the median) between two datasets. Note that this is
    based off of the assumptions that the two datasets have the same median.

    Parameters
    ----------
    data_1: list or numpy array, 1-D
        A list or array containing all observations from our first dataset
    data_2: list or numpy array, 1-D
        A list or array containing all observations from our second dataset
    alternative: str, default is two-sided
        Our alternative hypothesis. It can be two-sided, less or greater

    Returns
    -------
    ab: float
        The sum of ranks of our first dataset, or our test statistic
    p: float, 0 <= p <= 1
        The likelihood that our observed  dispersion would be likely if the two datasets were sampled from the same
        population
    """
    data_1, data_2 = _check_table(data_1, only_count=False), _check_table(data_2, only_count=False)
    if alternative.casefold() not in ['two-sided', 'greater', 'less']:
        raise ValueError("Cannot determine method for alternative hypothesis")
    n, m = len(data_1), len(data_2)
    all_data = np.concatenate([data_1, data_2])
    n_obs = n + m
    if len(np.unique(all_data)) != n_obs:
        ties = True
    else:
        ties = False
    rank_data = rankdata(all_data)
    re_rank = - rank_data[rank_data > np.median(rank_data)] + np.max(rank_data) + 1
    np.place(rank_data, rank_data > np.median(rank_data), re_rank)
    ab = np.sum(rank_data[:n])
    # exact p-value
    if n < 55 and m < 55 and not ties:
        a_start, a_1, i_fault = gscale(n,m)
        ind = ab - a_start
        total = np.sum(a_1)
        if ind < len(a_1) / 2:
            top_index = int(np.ceil(ind))
            if ind == top_index:
                if alternative.casefold() == 'two-sided':
                    p = 2 * np.sum(a_1[:top_index + 1]) / total
                elif alternative.casefold() == 'less':
                    p = np.sum(a_1[:top_index + 1]) / total
                else:
                    p = np.sum(a_1[top_index + 2:]) / total
            else:
                if alternative.casefold() == 'two-sided':
                    p = 2 * np.sum(a_1[:top_index]) / total
                elif alternative.casefold() == 'less':
                    p = np.sum(a_1[:top_index]) / total
                else:
                    p = np.sum(a_1[top_index + 1:]) / total
        else:
            bottom_index = int(np.floor(ind))
            if ind == bottom_index:
                if alternative.casefold() == "two-sided":
                    p = 2 * np.sum(a_1[bottom_index:]) / total
                elif alternative.casefold() == 'less':
                    p = np.sum(a_1[bottom_index:]) / total
                else:
                    p = np.sum(a_1[:bottom_index + 1]) / total
            else:
                if alternative.casefold() == 'two-sided':
                    p = 2 * np.sum(a_1[bottom_index + 1:]) / total
                elif alternative.casefold() == 'less':
                    p = np.sum(a_1[bottom_index + 1:]) / total
                else:
                    p = np.sum(a_1[:bottom_index + 2]) / total
    else:
        # even
        if n_obs % 2 == 0:
            mu_c = n * (n_obs + 2) / 4
            if ties:
                var_c = m * n * (16 * np.sum(np.power(rank_data, 2)) - n_obs * pow(n_obs + 2, 2)) / (16 * n_obs * (n_obs - 1))
            else:
                var_c = m * n * (n_obs + 2) * (n_obs - 2) / 48 / (n_obs - 1)
        # odd
        else:
            mu_c = n * pow(n_obs + 1, 2) / 4 / n_obs
            if ties:
                var_c = m * n * (16 * n_obs * np.sum(np.power(rank_data, 2)) - pow(n_obs + 1, 4)) / (16 * pow(n_obs, 2) * (n_obs - 1))
            else:
                var_c = m * n * (n_obs + 1) * (3 + pow(n_obs, 2)) / (48 * pow(n_obs, 2))
        z = (ab - mu_c) / sqrt(var_c)
        if alternative.casefold() == 'two-sided':
            p = 2 * (1 - norm.cdf(abs(z)))
        elif alternative.casefold() == 'greater':
            p = 1 - norm.cdf(z)
        else:
            p = norm.cdf(z)
    return ab, p


def mood_test(data_1, data_2, alternative='two-sided'):
    """Found in scipy.stats as mood
    Used to measure the level of dispersion of the ranks of the two datasets.

    Parameters
    ----------
    data_1: list or numpy array, 1-D
        A list or array containing all observations from our first dataset
    data_2: list or numpy array, 1-D
        A list or array containing all observations from our second dataset
    alternative: str, default is two-sided
        Our alternative hypothesis. It can be two-sided, less or greater

    Returns
    -------
    z: float
        Our test statistic that measures the degree of normality of the rank dispersions
    p: float, 0 <= p <= 1
        The likelihood that our rank dispersion would occur from two datasets drawn from the same
        distribution
    """
    data_1, data_2 = _check_table(data_1, only_count=False), _check_table(data_2, only_count=False)
    if alternative.casefold() not in ['two-sided', 'greater', 'less']:
        raise ValueError("Cannot determine method for alternative hypothesis")
    len_1, len_2 = len(data_1), len(data_2)
    n_obs = len_1 + len_2
    if n_obs < 3:
        raise AttributeError("Not enough observations to perform mood dispertion test")
    all_data = np.concatenate([data_1, data_2])
    rank_data = rankdata(all_data)
    r_1 = rank_data[:len_1]
    m = np.sum(np.power(r_1 - (n_obs + 1) / 2, 2))
    mu_m = len_1 * (pow(n_obs, 2) - 1) / 12
    var_m = len_1 * len_2 * (n_obs + 1) * (n_obs + 2) * (n_obs - 2) / 180
    z = (m - mu_m) / sqrt(var_m)
    if alternative.casefold() == 'two-sided':
        if z > 0:
            p = 2 * (1 - norm.cdf(z))
        else:
            p = 2 * norm.cdf(z)
    elif alternative.casefold() == 'greater':
        p = 1 - norm.cdf(z)
    else:
        p = norm.cdf(z)
    return z, p
