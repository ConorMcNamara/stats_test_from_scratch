import warnings

from math import sqrt, factorial
from numbers import Number
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from scipy.stats import t, norm, f
from scipy.special import factorial as st_factorial

from StatsTest.utils import (
    _standard_error,
    _check_table,
    _right_extreme,
    _left_extreme,
    _rle,
)


def one_sample_z_test(
    sample_data: Union[Sequence[Sequence], np.ndarray],
    pop_mean: float,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """This test can be found in statsmodels as ztest

    Determines the likelihood that our sample mean differs from our population mean, assuming that the data follows a
    normal distribution.

    Parameters
    ----------
    sample_data : list or numpy array, 1-D
        Our observational data
    pop_mean : float
        The mean of our population, or what we expect the mean of our sample data to be
    alternative : {'two-sided', 'greater', 'less'}, default=two-sided
        Our alternative hypothesis

    Returns
    -------
    z_score : float
        The Z-score of our data
    p : float, 0 <= p <= 1
        The likelihood that our observed data differs from our population mean, assuming a normal distribution, due to
        chance
    """
    if not isinstance(pop_mean, Number):
        raise TypeError("Population mean is not of numeric type")
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine method for alternative hypothesis")
    if len(sample_data) < 30:
        raise AttributeError("Too few observations for z-test to be reliable, use t-test instead")
    sample_data = _check_table(sample_data, False)
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)
    z_score = sample_mean - pop_mean / _standard_error(sample_std, len(sample_data))
    if alternative.casefold() == "two-sided":
        p = 2 * (1 - norm.cdf(abs(z_score)))
    elif alternative.casefold() == "greater":
        p = 1 - norm.cdf(z_score)
    else:
        p = norm.cdf(z_score)
    return z_score, p


def two_sample_z_test(
    data_1: Union[Sequence[Sequence], np.ndarray],
    data_2: Union[Sequence[Sequence], np.ndarray],
    alternative: str = "two-sided",
) -> Tuple[Number, float]:
    """This test can be found in statsmodels as ztest_ind

    Determines the likelihood that the distribution of two data points is significantly different, assuming that both
    data points are derived from a normal distribution.

    Parameters
    ----------
    data_1 : list or numpy array, 1-D
        The observed dataset we are comparing to data_2
    data_2 : list or numpy array, 1-D
        The observed dataset we are comparing to data_1
    alternative : {'two-sided', 'greater', 'less'}
        Our alternative hypothesis

    Returns
    -------
    z_score : number
        The Z-score of our observed differences
    p : float, 0 <= p <= 1
        The likelihood that the observed differences from data_1 to data_2 are due to chance
    """
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine method for alternative hypothesis")
    if len(data_1) < 30 or len(data_2) < 30:
        raise AttributeError("Too few observations for z-test to be reliable, use t-test instead")
    data_1, data_2 = _check_table(data_1, False), _check_table(data_2, False)
    data_1_mean, data_2_mean = np.mean(data_1), np.mean(data_2)
    data_1_std, data_2_std = np.std(data_1, ddof=1), np.std(data_2, ddof=1)
    z_score = (data_1_mean - data_2_mean) / sqrt(
        _standard_error(data_1_std, len(data_1)) + _standard_error(data_2_std, len(data_2))
    )
    if alternative.casefold() == "two-sided":
        p = 2 * (1 - norm.cdf(abs(z_score)))
    elif alternative.casefold() == "greater":
        p = 1 - norm.cdf(z_score)
    else:
        p = norm.cdf(z_score)
    return z_score, p


def one_sample_t_test(
    sample_data: Union[Sequence[Sequence], np.ndarray],
    pop_mean: float,
    alternative: str = "two-sided",
) -> Tuple[Number, float]:
    """This test can be found in scipy.stats as ttest_1samp

    Used when we want to compare our sample mean to that of an expected population mean, and while we assume that the
    data follows a normal distribution, our sample size is too small to reliably use the z-test.

    Parameters
    ----------
    sample_data : list or numpy array, 1-D
        The observed dataset we are comparing to the population mean
    pop_mean : float
        The mean of our population, or what we expect the mean of our sample data to be
    alternative : {'two-sided', 'greater', 'less'}
        Our alternative hypothesis

    Returns
    -------
    t_value : number
        The t statistic for the differences between the sample mean and population
    p : float, 0 <= p <= 1
        The likelihood that our observed data differs from the population mean due to chance
    """
    if not isinstance(pop_mean, Number):
        raise TypeError("Population mean is not of numeric type")
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine method for alternative hypothesis")
    sample_data = _check_table(sample_data, False)
    sample_mean = np.mean(sample_data)
    n_observations = len(sample_data)
    df = n_observations - 1
    sample_std = np.std(sample_data, ddof=1)
    t_value = (sample_mean - pop_mean) / (sample_std / sqrt(n_observations))
    p = 1.0 - t.cdf(abs(t_value), df)
    if alternative.casefold() == "two_sided":
        p *= 2
    elif alternative.casefold() == "less":
        p = 1 - p
    else:
        pass
    return t_value, p


def two_sample_t_test(
    data_1: Union[Sequence[Sequence], np.ndarray],
    data_2: Union[Sequence[Sequence], np.ndarray],
    alternative: str = "two-sided",
    paired: bool = False,
) -> Tuple[Number, float]:
    """This test can be found in scipy.stats as either ttest_rel or ttest_ind

    Used when we want to compare the distributions of two samples, and while we assume that they both follow a normal
    distribution, their sample size is too small to reliably use a z-test.

    Parameters
    ----------
    data_1 : list or numpy array, 1-D
        The observed dataset we are comparing to data_2
    data_2 : list or numpy array, 1-D
        The observed dataset we are comparing to data_1
    alternative : str, {two-sided, greater, less}, default=two-sided
        Our alternative hypothesis
    paired : bool, default=False
        Whether or not data_1 and data_2 are paired observations

    Returns
    -------
    t_value : number
        The t statistic for the difference between our datasets
    p : float, 0 <= p <= 1
        The likelihood that the observed differences are due to chance
    """
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine method for alternative hypothesis")
    data_1, data_2 = _check_table(data_1, False), _check_table(data_2, False)
    data_1_mean, data_2_mean = np.mean(data_1), np.mean(data_2)
    if paired:
        """This test can be found in scipy.stats as ttest_rel"""
        if len(data_1) != len(data_2):
            raise AttributeError("The data types are not paired")
        n = len(data_1)
        df = n - 1
        squared_difference = np.sum(np.power(data_1 - data_2, 2))
        difference = np.sum(data_1 - data_2)
        std = sqrt((squared_difference - (np.power(difference, 2) / n)) / df)
        standard_error_difference = _standard_error(std, n)

    else:
        # We perform the Welch T-Test due to assumption that variances are not equal
        """This test can be found in scipy.stats as ttest_ind"""
        data_1_var, data_2_var = np.var(data_1, ddof=1), np.var(data_2, ddof=1)
        data_1_n, data_2_n = len(data_1), len(data_2)
        df = np.power((data_1_var / data_1_n) + (data_2_var / data_2_n), 2) / (
            (np.power(data_1_var, 2) / (np.power(data_1_n, 2) * data_1_n - 1))
            + (np.power(data_2_var, 2) / (np.power(data_2_n, 2) * data_2_n - 1))
        )
        standard_error_difference = sqrt((data_1_var / data_1_n) + (data_2_var / data_2_n))
    t_value = (data_1_mean - data_2_mean) / standard_error_difference
    p = 1.0 - t.cdf(abs(t_value), df)
    if alternative.casefold() == "two-sided":
        p *= 2
    elif alternative.casefold() == "less":
        p = 1 - p
    else:
        pass
    return t_value, p


def trimmed_means_test(
    data_1: Union[Sequence[Sequence], np.ndarray],
    data_2: Union[Sequence[Sequence], np.ndarray],
    p: int = 10,
    alternative: str = "two-sided",
) -> Tuple[Number, float]:
    """Not found in scipy.stats or statsmodels.
    Used when we wish to perform a two-sample t-test, but suspect that the data is being heavily influenced by outliers,
    i.e., cannot assume normality.

    Parameters
    ----------
    data_1 : list or numpy array, 1-D
        The observed dataset we are comparing to data_2
    data_2 : list or numpy array, 1-D
        The observed dataset we are comparing to data_1
    p : float, 0 <= p <= 100, default=10
        The percentage of data we wish to drop from each sample
    alternative : {'two-sided', 'greater', 'less'}
        Our alternative hypothesis

    Returns
    -------
    t_value : number
        The t statistic for the difference between our datasets
    p : float, 0 <= p <= 1
        The likelihood that the observed differences are due to chance
    """
    if p < 0 or p > 100:
        raise ValueError("Percentage trimmed needs to be between 0 and 100")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine method for alternative hypothesis")
    data_1, data_2 = _check_table(data_1), _check_table(data_2)
    sort_data_1, sort_data_2 = np.sort(data_1), np.sort(data_2)
    n_1, n_2 = len(data_1) * p // 200, len(data_2) * p // 200
    trim_data_1, trim_data_2 = (
        sort_data_1[n_1: len(sort_data_1) - n_1],
        sort_data_2[n_2: len(sort_data_2) - n_2],
    )
    n_x, n_y = len(data_1), len(data_2)
    m_x, m_y = len(trim_data_1), len(trim_data_2)
    winsor_values_1, winsor_values_2 = (
        np.append(trim_data_1[0] * n_1, trim_data_1[-1] * n_1),
        np.append(trim_data_2[0] * n_2, trim_data_2[-1] * n_2),
    )
    winsor_data_1, winsor_data_2 = np.append(trim_data_1, winsor_values_1), np.append(trim_data_2, winsor_values_2)
    s_x, s_y = np.var(winsor_data_1, ddof=1), np.var(winsor_data_2, ddof=1)
    x_bar, y_bar = np.mean(trim_data_1), np.mean(trim_data_2)
    pooled_var = ((n_x - 1) * s_x + (n_y - 1) * s_y) / ((m_x - 1) + (m_y - 1))
    t_value = (x_bar - y_bar) / np.sqrt(pooled_var * ((1 / m_x) + (1 / m_y)))
    df = m_x + m_y - 2
    p = 1.0 - t.cdf(abs(t_value), df)
    if alternative.casefold() == "two-sided":
        p *= 2
    elif alternative.casefold() == "less":
        p = 1 - p
    else:
        pass
    return t_value, p


def yeun_welch_test(
    data_1: Union[Sequence[Sequence], np.ndarray],
    data_2: Union[Sequence[Sequence], np.ndarray],
    p=10,
    alternative: str = "two-sided",
) -> Tuple[Number, float]:
    """Not found in scipy.stats or statsmodels.

    Used when we wish to perform a two-sample t-test, but cannot assume normality or equality of variances.

    Parameters
    ----------
    data_1 : list or numpy array, 1-D
        The observed dataset we are comparing to data_2
    data_2 : list or numpy array, 1-D
        The observed dataset we are comparing to data_1
    p : float, 0 <= p <= 100, default=10
        The percentage of data we wish to drop from each sample
    alternative : str, {two-sided, greater, less}, default=two-sided
        Our alternative hypothesis

    Returns
    -------
    t_value : number
        The t statistic for the difference between our datasets
    p : float, 0 <= p <= 1
        The likelihood that the observed differences are due to chance
    """
    if p < 0 or p > 100:
        raise ValueError("Percentage trimmed needs to be between 0 and 100")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine method for alternative hypothesis")
    data_1, data_2 = _check_table(data_1), _check_table(data_2)
    sort_data_1, sort_data_2 = np.sort(data_1), np.sort(data_2)
    n_1, n_2 = len(data_1) * p // 200, len(data_2) * p // 200
    trim_data_1, trim_data_2 = (
        sort_data_1[n_1: len(sort_data_1) - n_1],
        sort_data_2[n_2: len(sort_data_2) - n_2],
    )
    n_x, n_y = len(data_1), len(data_2)
    m_x, m_y = len(trim_data_1), len(trim_data_2)
    winsor_values_1, winsor_values_2 = (
        np.append(trim_data_1[0] * n_1, trim_data_1[-1] * n_1),
        np.append(trim_data_2[0] * n_2, trim_data_2[-1] * n_2),
    )
    winsor_data_1, winsor_data_2 = np.append(trim_data_1, winsor_values_1), np.append(trim_data_2, winsor_values_2)
    s_x, s_y = np.var(winsor_data_1, ddof=1), np.var(winsor_data_2, ddof=1)
    x_bar, y_bar = np.mean(trim_data_1), np.mean(trim_data_2)
    d_x, d_y = (n_x - 1) * s_x / (m_x * (m_x - 1)), (n_y - 1) * s_y / (m_y * (m_y - 1))
    df = pow(d_x + d_y, 2) / (pow(d_x, 2) / (m_x - 1) + pow(d_y, 2) / (m_y - 1))
    t_value = (x_bar - y_bar) / sqrt(d_x + d_y)
    p = 1 - t.cdf(t_value, df // 1)
    if alternative.casefold() == "two-sided":
        p *= 2
    elif alternative.casefold() == "less":
        p = 1 - p
    else:
        pass
    return t_value, p


def two_sample_f_test(
    data_1: Union[Sequence[Sequence], np.ndarray],
    data_2: Union[Sequence[Sequence], np.ndarray],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """No method in scipy or statsmodels to immediately calculate this.

    Used to determine if two populations/samples have the same variance. Note that, due to this being a ratio between
    data_1 and data_2, a large p-value is just as significant as a small p-value. Also note that this test is extremely
    sensitive to data that is non-normal, so only use this test if the samples have been verified to come from a normal
    distribution.

    Parameters
    ----------
    data_1 : list or numpy array, 1-D
        The observed measurements for our first sample
    data_2 : list or numpy array, 1-D
        The observed measurements for our second sample
    alternative : {'two-sided', 'greater', 'less'}
        Our alternative hypothesis

    Returns
    -------
    f_statistic : float
        A ratio of the variance of data_1 to data_2
    p : float, 0 <= p <= 1
        The likelihood that this ratio could occur from two two samples with equal variances, due to chance
    """
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine method for alternative hypothesis")
    data_1, data_2 = _check_table(data_1), _check_table(data_2)
    df_1, df_2 = len(data_1) - 1, len(data_2) - 1
    var_1, var_2 = np.var(data_1, ddof=1), np.var(data_2, ddof=1)
    f_statistic = var_1 / var_2
    if alternative.casefold() == "two-sided":
        p = 2 * (1 - f.cdf(f_statistic, df_1, df_2))
    elif alternative.casefold() == "greater":
        p = 1 - f.cdf(f_statistic, df_1, df_2)
    else:
        p = f.cdf(f_statistic, df_1, df_2)
    return f_statistic, p


def binomial_sign_test(
    data_1: Union[Sequence[Sequence], np.ndarray],
    data_2: Union[Sequence[Sequence], np.ndarray],
    alternative: str = "two-sided",
    success_prob: float = 0.5,
) -> float:
    """Found in scipy as sign_test

    Used to determine whether or not the measured differences between two groups (X and Y) is
    significantly greater and/or less than each other. For instance, we might use this to determine if the weight loss
    for users who followed a certain diet is significant or not.

    Parameters
    ----------
    data_1 : list or numpy array, 1-D
        A list of all observations for group X.
    data_2 : list or numpy array, 1-D
        A list of all observations for group Y.
    alternative : {'two-sided', 'greater', 'less'}
        Our alternative hypothesis
    success_prob : float, 0 <= success_prob <= 1
        The probability of success. default=0.5

    Returns
    -------
    p : float, 0 <= p <= 1
        The probability that our observed differences would happen under a binomial distribution, assuming the given
        success probability.
    """
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine method for alternative hypothesis")
    if len(data_1) != len(data_2):
        raise AttributeError("The two data sets are not paired data sets")
    if not isinstance(success_prob, float):
        raise TypeError("Probability of success needs to be a decimal value")
    if success_prob > 1 or success_prob < 0:
        raise ValueError("Cannot calculate probability of success, needs to be between 0 and 1")
    data_1, data_2 = _check_table(data_1), _check_table(data_2)
    diff = data_1 - data_2
    pos_diff, neg_diff = np.sum(diff > 0), np.sum(diff < 0)
    total = pos_diff + neg_diff
    if alternative.casefold() == "greater":
        p = _right_extreme(pos_diff, total, success_prob)
    elif alternative.casefold() == "less":
        p = _left_extreme(pos_diff, total, success_prob)
    else:
        p = _left_extreme(neg_diff, total, success_prob) + _right_extreme(pos_diff, total, success_prob)
    return p


def wald_wolfowitz_test(
    data_1: Union[Sequence[Sequence], np.ndarray],
    expected: Optional[Union[Sequence[Sequence], np.ndarray]] = None,
    cutoff: str = "median",
):
    """Found in statsmodels as runstest_1samp

    Used to determine if the elements of a dataset/sequence are mutually independent

    Parameters
    ----------
    data_1 : list or numpy array, 1-D
        Our dataset that we are checking for mutual independence
    expected : list or numpy array, 1-D, default=None
        Contains the expected results from a given function. For example, if we expect our data to follow pow(x, 2), it
        would follow something like [1, 4, 9, 16, 25, ....]
    cutoff : {'median', 'mean'}
        If expected is None, then our cutoff point for what we regard as greater or less than.

    Returns
    -------
    x : float
        Our measure of mutual independence for each data point
    p : float, 0 <= p <= 1
        How likely we would observe this amount of mutal dependence assuming our data was derived from a mutually
        independent population
    """
    data_1 = np.array(data_1)

    if expected is None:
        if cutoff.casefold() not in ["median", "mean"]:
            raise ValueError("Cannot determine cutoff point")
        if cutoff.casefold() == "median":
            midpoint = np.median(data_1)
        else:
            midpoint = np.mean(data_1)
        plus_minus = data_1 >= midpoint
    else:
        expected = _check_table(expected)
        if len(expected) != len(data_1):
            raise AttributeError("Cannot perform Wald-Wolfowitz with unequal array lengths")
        plus_minus = np.greater_equal(data_1, expected)
    runs, _, loc = _rle(plus_minus)
    n_runs = len(runs)
    runs_length = np.sum(runs)
    run_pos, run_neg = runs[loc], runs[~loc]
    n_plus, n_minus = np.sum(run_pos), np.sum(run_neg)
    mu = (2 * n_plus * n_minus) / runs_length + 1
    var = 2 * n_plus * n_minus * (2 * n_plus * n_minus - runs_length) / (pow(runs_length, 2) * (runs_length - 1))
    z = (n_runs - mu) / sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def trinomial_test(
    data_1: Union[Sequence[Sequence], np.ndarray],
    data_2: Union[Sequence[Sequence], np.ndarray],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Not found in scipy.stats or statsmodels

    Used on paired-data when the sign test loses power, that is, when there exists instances of "zero observations" or
    differences of zero between the paired-data.

    Parameters
    ----------
    data_1 : list or numpy array, 1-D
        The observed measurements for our first sample
    data_2 : list or numpy array, 1-D
        The observed measurements for our first sample
        alternative: {'two-sided', 'greater', 'less'}
        Our alternative hypothesis

    Returns
    -------
    d : int
        The number of positive instances minus the number of negative instances
    p : float, 0 <= p <= 1
        The likelihood that we would observe these sign differences due to random chance
    """
    data_1, data_2 = _check_table(data_1), _check_table(data_2)
    if len(data_1) != len(data_2):
        raise AttributeError("Cannot perform Trinomial Test on unpaired data")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine alternative hypothesis")
    n = len(data_1)
    diffs = data_1 - data_2
    pos_diff, neg_diff, zero_diff = (
        np.sum(diffs > 0),
        np.sum(diffs < 0),
        np.sum(diffs == 0),
    )
    p_0 = zero_diff / n
    probs = []

    def calculate_probs(n, z, k, p_0):
        return np.sum(
            factorial(n)
            / (st_factorial(n - z - 2 * k) * st_factorial(k + z) * st_factorial(k))
            * np.power(p_0, n - z - (2 * k))
            * np.power((1 - p_0) / 2, z + 2 * k)
        )

    for z in range(n + 1):
        k = np.arange(0, (n - z) // 2 + 1)
        probs.append(calculate_probs(n, z, k, p_0))
    d = pos_diff - neg_diff
    if alternative.casefold() == "two-sided":
        p = np.sum(probs[abs(d):]) * 2
    elif alternative.casefold() == "greater":
        p = np.sum(probs[abs(d):])
    else:
        p = np.sum(probs[: abs(d)])
    return d, p


def fligner_policello_test(
    data_1: Union[Sequence[Sequence], np.ndarray],
    data_2: Union[Sequence[Sequence], np.ndarray],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Not found in either scipy.stats or statsmodels.

    Used to determine whether the population medians corresponding to two independent samples are equal.

    Parameters
    ----------
    data_1 : list or numpy array, 1-D
        The observed measurements for our first sample
    data_2 : list or numpy array, 1-D
        The observed measurements for our first sample
    alternative : {'two-sided', 'greater', 'less'}
        Our alternative hypothesis

    Returns
    -------
    z : float
        The z-score of our observed median differences
    p : float, 0 <= p <= 1
        The likelihood that we would observe these differences due to chance
    """
    data_1, data_2 = _check_table(data_1), _check_table(data_2)
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine alternative hypothesis")
    m, n = len(data_1), len(data_2)
    if m < 12 or n < 12:
        warnings.warn("Datasets may be too small for accurate approximation of p")

    def compare_points(x, y):
        z = x - y[:, None]
        z = np.where(z > 0, 1, z)
        z = np.where(z == 0, 0.5, z)
        z = np.where(z < 0, 0, z)
        return np.sum(z, axis=0)

    n_x, n_y = compare_points(data_1, data_2), compare_points(data_2, data_1)
    Nx, Ny = np.sum(n_x), np.sum(n_y)
    m_x, m_y = np.mean(n_x), np.mean(n_y)
    ss_x, ss_y = np.sum(np.power(n_x - m_x, 2)), np.sum(np.power(n_y - m_y, 2))
    z = (Ny - Nx) / (2 * np.sqrt(ss_x + ss_y - (m_x * m_y)))
    if alternative.casefold() == "two-sided":
        p = 2 * (1 - norm.cdf(abs(z)))
    elif alternative.casefold() == "greater":
        p = 1 - norm.cdf(z)
    else:
        p = norm.cdf(z)
    return z, p
