from src.utils import _check_table
from scipy.stats import rankdata, norm
import numpy as np
from math import sqrt


def two_sample_mann_whitney_test(data_1, data_2, alternative='two-sided'):
    """This test can be found in scipy.stats as mannwhitneyu

    Parameters
    ----------
    data_1: list or numpy array
        The observed dataset we are comparing to data_2
    data_2: list or numpy array
        The observed dataset we are comparing to data_1
    alternative: str
        Our alternative hypothesis. It can be two-sided, greater or less

    Return
    ------
    u: number
        The U statistic for our observed differences
    p: float
        The likelihood that the observed differences are due to chance
    """
    assert alternative.casefold() in ['two-sided', 'greater', 'less'], \
        "Cannot determine method for alternative hypothesis"
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


#To-do add Pratt handling of zero-ranked data
def two_sample_wilcoxon_test(data_1, data_2, alternative='two-sided', handle_zero='wilcox'):
    """This test can be found in scipy.stats as wilcoxon

    Parameters
    ----------
    data_1: list or numpy array
        The observed dataset we are comparing to data_2
    data_2: list or numpy array
        The observed dataset we are comparing to data_1
    alternative: str
        Our alternative hypothesis. It can be two-sided, greater or less
    handle_zero: str
        How we treat differences of zero. It can be either wilcox (ignore) or pratt

    Return
    ------
    w_value: number
        The W statistic for our observed differences
    p: float
        The likelihood that the observed differences are due to chance
    """
    assert alternative.casefold() in ['two-sided', 'greater', 'less'], \
        "Cannot determine method for alternative hypothesis"
    assert handle_zero.casefold() in ['wilcox', 'pratt'], "Cannot determine how to handle differences of zero"
    assert len(data_1) == len(data_2), "Cannot perform signed wilcoxon test on unpaired data"
    data_1, data_2 = _check_table(data_1, False), _check_table(data_2, False)
    diff = data_1 - data_2
    if handle_zero.casefold() == 'wilcox':
        assert np.sum(diff == 0) != len(data_1), "Cannot perform wilcoxon test when all differences are zero"
        diff = np.compress(np.not_equal(diff, 0), diff)
    n = len(diff)
    abs_diff = np.abs(diff)
    sign_diff = np.sign(diff)
    rank = rankdata(abs_diff)
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