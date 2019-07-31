from StatsTest.utils import _check_table
from scipy.stats import rankdata, norm, chi2
import numpy as np
from math import sqrt


def two_sample_mann_whitney_test(data_1, data_2, alternative='two-sided'):
    """This test can be found in scipy.stats as mannwhitneyu
    Used when we want to test whether or not the distribution of two ordinal response variables are equal or not,
    assuming that each sample is independent of one another.

    Parameters
    ----------
    data_1: list or numpy array
        The observed sample for ordinal response variable 1
    data_2: list or numpy array
        The observed sample for ordinal response variable 2
    alternative: str, default is two-sided
        Our alternative hypothesis. It can be two-sided, greater or less

    Return
    ------
    u: number
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
    data_1: list or numpy array
        The first sample or repeated measure
    data_2: list or numpy array
        The second sample or repeated measure
    alternative: str, default is two-sided
        Our alternative hypothesis. It can be two-sided, greater or less
    handle_zero: str, default is wilcox
        How we treat differences of zero. It can be either wilcox (ignore) or pratt

    Return
    ------
    w_value: number
        The W statistic for our observed differences
    p: float, 0 <= p <= 1
        The likelihood that the observed mean rank differences are due to chance
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
    args: list or numpy array
        An array containing the observations for each treatment (In the example above, each array would represent one
        of the judges ratings for every k wine).

    Return
    ------
    q: float
        Our Q statistic, or a measure of the difference between our expected result and the observed outcomes
    p: float
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


# def page_test(*args):
#     k = len(args)
#     if k < 3:
#         raise AttributeError("Page Test not appropriate for {} levels".format(k))




