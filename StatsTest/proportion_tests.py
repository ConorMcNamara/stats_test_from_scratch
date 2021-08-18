from math import sqrt
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from scipy.stats import norm, binom, chi2

from StatsTest.utils import _check_table, _left_extreme, _right_extreme


def one_sample_proportion_z_test(
    sample_data: Union[Sequence, np.ndarray],
    pop_mean: float,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Found in statsmodels as proportions_ztest

    Used when comparing whether our observed proportion mean is difference to the population mean, assuming that the
    proportion mean is normally distributed.

    Parameters
    ----------
    sample_data: list or numpy array, must be binary, 1-D
        An array containing all observations, marked as a 0 for failure and a 1 for success
    pop_mean: float
        Our expected proportion of success
    alternative: {'two-sided', 'greater', 'less}
        Our alternative hypothesis. It can be two-sided, less or greater

    Returns
    -------
    z_score: float
        Our z-statistic to analyze the likelihood that our observed difference is due to chance
    p: float, 0 <= p <= 1
        The probability that the observed proportion differs from our population proportion, assuming a normal
        distribution, due to chance
    """
    if not isinstance(pop_mean, float):
        raise TypeError("Population mean is not of float type")
    if pop_mean > 1 or pop_mean < 0:
        raise ValueError("Population mean must be between 0 and 1")
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine method for alternative hypothesis")
    sample_data = _check_table(sample_data)
    if not np.array_equal(sample_data, sample_data.astype(bool)):
        raise AttributeError("Cannot perform a proportion test on non-binary data")
    if (
        len(np.where(sample_data == 1)[0]) < 10
        or len(np.where(sample_data == 0)[0]) < 10
    ):
        raise AttributeError(
            "Too few instances of success or failure to run proportion test"
        )
    p = np.mean(sample_data)
    q = 1 - p
    n = len(sample_data)
    std = sqrt((p * q) / n)
    z_score = (p - pop_mean) / std
    if alternative.casefold() == "two-sided":
        p = 2 * (1 - norm.cdf(abs(z_score)))
    elif alternative.casefold() == "greater":
        p = 1 - norm.cdf(z_score)
    else:
        p = norm.cdf(z_score)
    return z_score, p


def two_sample_proportion_z_test(
    data_1: Union[Sequence, np.ndarray],
    data_2: Union[Sequence, np.ndarray],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Found in statsmodels as proportions_ztest

    Used when we are comparing whether or not two proportion means are the same, given that both of them come from a
    normal distribution.

    Parameters
    ----------
    data_1: list or numpy array, must be binary, 1-D
        An array containing all observations, marked as a 0 for failure and a 1 for success, that we are comparing to
        data_2
    data_2: list or numpy array, must be binary, 1-D
        An array containing all observations, marked as a 0 for failure and a 1 for success, that we are comparing to
        data_1
    alternative: {'two-sided', 'greater', 'less'}
        Our alternative hypothesis. It can be two-sided, less or greater

    Returns
    -------
    z_score: float
        Our z-statistic to analyze the likelihood that our observed difference is due to chance
    p: float, 0 <= p <= 1
        The probability that the differences between two samples, assuming a normal distribution, is due to chance
    """
    data_1, data_2 = _check_table(data_1), _check_table(data_2)
    if not np.array_equal(data_1, data_1.astype(bool)):
        raise AttributeError(
            "Cannot perform a proportion test on non-binary data for data_1"
        )
    if not np.array_equal(data_2, data_2.astype(bool)):
        raise AttributeError(
            "Cannot perform a proportion test on non-binary data for data_2"
        )
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine method for alternative hypothesis")
    n_1, n_2 = len(data_1), len(data_2)
    p_1, p_2 = np.mean(data_1), np.mean(data_2)
    p = (p_1 * n_1 + p_2 * n_2) / (n_1 + n_2)
    q = 1 - p
    se = sqrt((p * q) * ((1 / n_1) + (1 / n_2)))
    z_score = (p_1 - p_2) / se
    if alternative.casefold() == "two-sided":
        p = 2 * (1 - norm.cdf(abs(z_score)))
    elif alternative.casefold() == "greater":
        p = 1 - norm.cdf(z_score)
    else:
        p = norm.cdf(z_score)
    return z_score, p


def binomial_test(
    success: Union[Sequence, np.ndarray, int],
    failure: Union[Sequence, np.ndarray, int],
    alternative: str = "two-sided",
    success_prob: Optional[float] = None,
) -> float:
    """The binomial test can be found in scipy.stats as binom_test.

    Used to determine if the likelihood that our observed measurements could occur given that we know the prior probability.
    For example, if we rolled a die 100 times and a 6 appeared 40 times, our test would measure the likelihood this
    could happen to a fair die, given that the percentage of rolling a 6 is 1/6.

    Parameters
    ----------
    success: list or numpy array, 1-D,  or int
        If int, the number of successes. If list, then it is the count of all successes.
    failure: list or numpy array, 1-D, or int
        If int, the number of failures. If list, then it is the count of all failures.
    alternative: {'two-sided', 'greater', 'less'}
        Our alternative hypothesis. It can be two-sided, less or greater
    success_prob: float, default=None
        The probability of success. If None is given, then probability of success is assumed to be
        length of data_1 / (length of data_1 + length of data_2)

    Returns
    -------
    p: float, 0 <= p <= 1
        The likelihood that our observed measurement would occur under a binomial distribution, given the success
        probability.
    """
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine method for alternative hypothesis")
    if isinstance(success, int) and isinstance(failure, int):
        num_success, num_failure = success, failure
    else:
        success, failure = _check_table(success), _check_table(failure)
        num_success, num_failure = len(success), len(failure)
    total = num_success + num_failure
    if not success_prob:
        success_prob = num_success / total
    elif not isinstance(success_prob, float):
        raise TypeError("Probability of success needs to be a decimal value")
    if success_prob > 1 or success_prob < 0:
        raise ValueError(
            "Cannot calculate probability of success, needs to be between 0 and 1"
        )
    if alternative.casefold() == "greater":
        p = _right_extreme(num_success, total, success_prob)
    elif alternative.casefold() == "less":
        p = _left_extreme(num_success, total, success_prob)
    else:
        expected = int(success_prob * total)
        observed_p = binom.pmf(num_success, total, success_prob)
        if num_success < expected:
            vals = binom.pmf(np.arange(expected, total + 1), total, success_prob)
            num_small = vals <= observed_p
            small_pmf = np.sum(np.compress(num_small, vals))
            p = _left_extreme(num_success, total, success_prob) + small_pmf
        else:
            vals = binom.pmf(np.arange(expected + 1), total, success_prob)
            num_small = vals <= observed_p
            small_pmf = np.sum(np.compress(num_small, vals))
            p = small_pmf + _right_extreme(num_success, total, success_prob)
    return p


def chi_square_proportion_test(
    success_prob: Union[Sequence, np.ndarray],
    n_total: Union[Sequence, np.ndarray],
    expected: Optional[Union[Sequence, np.ndarray]] = None,
) -> Tuple[float, float]:
    """Not found in either statsmodels or scipy.stats

    Used when we are given proportions of success (as well as total participants) instead of
    numbers of success.

    Parameters
    ----------
    success_prob: list or numpy array, 1-D
        A list containing the percentage of success for each successive group. Needs to be the same size
        as n_total and expected
    n_total: list or numpy array, 1-D
        A list containing the total count of each successive group. Needs to be the same size as success_prob and
        expected
    expected: list or numpy array, 1-D
        If None, then expected is the weighted average of success_prob
        Else, a list containing the expected probabilities of each success group. Needs to be the same size as success_prob
        and n_total

    Returns
    -------
    X: float
        Our Chi measure of the difference between our observed and expected results
    p: float, 0 <= p <= 1
        The likelihood that we would observe these differences if each group was sampled from the same population
    """
    success_prob, n_total = _check_table(success_prob, only_count=False), _check_table(
        n_total, only_count=True
    )
    if len(success_prob) != len(n_total):
        raise ValueError("Success probability and N Total are not of same length")
    if expected is None:
        expected = np.sum(success_prob * n_total) / np.sum(n_total)
    else:
        expected = _check_table(expected, only_count=False)
        if len(expected) != len(success_prob):
            raise ValueError("Expected and Success probability are not of same length")
        if not np.all(expected < 1):
            raise ValueError("Cannot have percentage of expected greater than 1")
        elif not np.all(expected >= 0):
            raise ValueError("Cannot have negative percentage of expected")
    if not np.all(success_prob < 1):
        raise ValueError("Cannot have percentage of success greater than 1")
    elif not np.all(success_prob >= 0):
        raise ValueError("Cannot have negative percentage of success")
    n_success = success_prob * n_total
    n_failure = n_total - n_success
    n_expected_success = expected * n_total
    n_expected_failure = (1 - expected) * n_total
    df = len(n_total) - 1
    X = np.sum(
        np.power(n_success - n_expected_success, 2) / n_expected_success
    ) + np.sum(np.power(n_failure - n_expected_failure, 2) / n_expected_failure)
    p = 1 - chi2.cdf(X, df)
    return X, p


def g_proportion_test(
    success_prob: Union[Sequence, np.ndarray],
    n_total: Union[Sequence, np.ndarray],
    expected: Optional[Union[Sequence, np.ndarray]] = None,
) -> Tuple[float, float]:
    """Not found in either statsmodels or scipy.stats

    Used when we are given proportions of success (as well as total participants) instead of
    numbers of success

    Parameters
    ----------
    success_prob: list or numpy array, 1-D
        A list containing the percentage of success for each successive group. Needs to be the same size
        as n_total and expected
    n_total: list or numpy array, 1-D
        A list containing the total count of each successive group. Needs to be the same size as success_prob and
        expected
    expected: (optional) list or numpy array, 1-D, default=None
        If None, then expected is the weighted average of success_prob
        Else, a list containing the expected probabilities of each success group. Needs to be the same size as success_prob
        and n_total.

    Returns
    -------
    g: float
        Our measure of the difference between our observed and expected results
    p: float, 0 <= p <= 1
        The likelihood that we would observe these differences if each group was sampled from the same population
    """
    success_prob, n_total = _check_table(success_prob, only_count=False), _check_table(
        n_total, only_count=True
    )
    if len(success_prob) != len(n_total):
        raise ValueError("Success probability and N Total are not of same length")
    if expected is None:
        expected = np.sum(success_prob * n_total) / np.sum(n_total)
    else:
        expected = _check_table(expected, only_count=False)
        if len(expected) != len(success_prob):
            raise ValueError("Expected and Success probability are not of same length")
        if not np.all(expected < 1):
            raise ValueError("Cannot have percentage of expected greater than 1")
        elif not np.all(expected >= 0):
            raise ValueError("Cannot have negative percentage of expected")
    if not np.all(success_prob < 1):
        raise ValueError("Cannot have percentage of success greater than 1")
    elif not np.all(success_prob >= 0):
        raise ValueError("Cannot have negative percentage of success")
    n_success = success_prob * n_total
    n_failure = n_total - n_success
    n_expected_success = expected * n_total
    n_expected_failure = (1 - expected) * n_total
    df = len(n_total) - 1
    g = 2 * (
        np.sum(n_success * np.log(n_success / n_expected_success))
        + np.sum(n_failure * np.log(n_failure / n_expected_failure))
    )
    p = 1 - chi2.cdf(g, df)
    return g, p
