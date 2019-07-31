import numpy as np
from numbers import Number
from math import sqrt
from scipy.stats import t, norm, f
from StatsTest.utils import _standard_error, _check_table


def one_sample_z_test(sample_data, pop_mean, alternative='two-sided'):
    """This test can be found in statsmodels as ztest
    Determines the likelihood that our sample mean differs from our population mean, assuming that the data follows a
    normal distribution.

    Parameters
    ----------
    sample_data: list or numpy array
        Our observational data
    pop_mean: number
        The mean of our population, or what we expect the mean of our sample data to be
    alternative: str, default is two-sided
        What our alternative hypothesis is. It can be two-sided, less or greater

    Return
    ------
    z_score: number
        The Z-score of our data
    p: float, 0 <= p <= 1
        The likelihood that our observed data differs from our population mean, assuming a normal distribution, due to
        chance
    """
    if not isinstance(pop_mean, Number):
        raise TypeError("Population mean is not of numeric type")
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ['two-sided', 'greater', 'less']:
        raise ValueError("Cannot determine method for alternative hypothesis")
    if len(sample_data) < 30:
        raise AttributeError("Too few observations for z-test to be reliable, use t-test instead")
    sample_data = _check_table(sample_data, False)
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)
    z_score = sample_mean - pop_mean / _standard_error(sample_std, len(sample_data))
    if alternative.casefold() == 'two-sided':
        p = 2 * (1 - norm.cdf(abs(z_score)))
    elif alternative.casefold() == 'greater':
        p = 1 - norm.cdf(z_score)
    else:
        p = norm.cdf(z_score)
    return z_score, p


def two_sample_z_test(data_1, data_2, alternative='two-sided'):
    """This test can be found in statsmodels as ztest_ind
    Determines the likelihood that the distribution of two data points is significantly different, assuming that both
    data points are derived from a normal distribution.

    Parameters
    ----------
    data_1: list or numpy array
        The observed dataset we are comparing to data_2
    data_2: list or numpy array
        The observed dataset we are comparing to data_1
    alternative: str, default is two-sided
        What our alternative hypothesis is. It can be two-sided, less or greater

    Return
    ------
    z_score: number
        The Z-score of our observed differences
    p: float, 0 <= p <= 1
        The likelihood that the observed differences from data_1 to data_2 are due to chance
    """
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ['two-sided', 'greater', 'less']:
        raise ValueError("Cannot determine method for alternative hypothesis")
    if len(data_1) < 30 or len(data_2) < 30:
        raise AttributeError("Too few observations for z-test to be reliable, use t-test instead")
    data_1, data_2 = _check_table(data_1, False), _check_table(data_2, False)
    data_1_mean, data_2_mean = np.mean(data_1), np.mean(data_2)
    data_1_std, data_2_std = np.std(data_1, ddof=1), np.std(data_2, ddof=1)
    z_score = (data_1_mean - data_2_mean) / sqrt(_standard_error(data_1_std, len(data_1)) + _standard_error(data_2_std,
                                                                                                            len(data_2)))
    if alternative.casefold() == 'two-sided':
        p = 2 * (1 - norm.cdf(abs(z_score)))
    elif alternative.casefold() == 'greater':
        p = 1 - norm.cdf(z_score)
    else:
        p = norm.cdf(z_score)
    return z_score, p


def one_sample_t_test(sample_data, pop_mean, alternative='two-sided'):
    """This test can be found in scipy.stats as ttest_1samp
    Used when we want to compare our sample mean to that of an expected population mean, and while we assume that the
    data follows a normal distribution, our sample size is too small to reliably use the z-test.

    Parameters
    ----------
    sample_data: list or numpy array
        The observed dataset we are comparing to the population mean
    pop_mean: number
        The mean of our population, or what we expect the mean of our sample data to be
    alternative: str, default is two-sided
        What our alternative hypothesis is. It can be two-sided, less or greater

    Return
    ------
    t_value: number
        The t statistic for the differences between the sample mean and population
    p: float, 0 <= p <= 1
        The likelihood that our observed data differs from the population mean due to chance
    """
    if not isinstance(pop_mean, Number):
        raise TypeError("Population mean is not of numeric type")
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ['two-sided', 'greater', 'less']:
        raise ValueError("Cannot determine method for alternative hypothesis")
    sample_data = _check_table(sample_data, False)
    sample_mean = np.mean(sample_data)
    n_observations = len(sample_data)
    df = n_observations - 1
    sample_std = np.std(sample_data, ddof=1)
    t_value = (sample_mean - pop_mean) / (sample_std / sqrt(n_observations))
    p = (1.0 - t.cdf(abs(t_value), df))
    if alternative.casefold() == 'two_sided':
        p *= 2
    return t_value, p


def two_sample_t_test(data_1, data_2, alternative='two_sided', paired=False):
    """This test can be found in scipy.stats as either ttest_rel or ttest_ind
    Used when we want to compare the distributions of two samples, and while we assume that they both follow a normal
    distribution, their sample size is too small to reliably use a z-test.

    Parameters
    ----------
    data_1: list or numpy array
        The observed dataset we are comparing to data_2
    data_2: list or numpy array
        The observed dataset we are comparing to data_1
    alternative: str, default is two-sided
        Our alternative hypothesis. It can be two-sided, less or greater
    paired: bool, default is False
        Whether or not data_1 and data_2 are paired observations

    Return
    ------
    t_value: number
        The t statistic for the difference between our datasets
    p: float, 0 <= p <= 1
        The likelihood that the observed differences are due to chance
    """
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ['two-sided', 'greater', 'less']:
        raise ValueError("Cannot determine method for alternative hypothesis")
    data_1, data_2 = _check_table(data_1, False), _check_table(data_2, False)
    data_1_mean, data_2_mean = np.mean(data_1), np.mean(data_2)
    if paired:
        """This test can be found in scipy.stats as ttest_rel"""
        assert len(data_1) == len(data_2), "The data types are not paired"
        n = len(data_1)
        df = n - 1
        squared_difference = sum((data_1 - data_2) ** 2)
        difference = sum(data_1 - data_2)
        std = sqrt((squared_difference - (difference**2 / n)) / df)
        standard_error_difference = _standard_error(std, n)

    else:
        # We perform the Welch T-Test due to assumption that variances are not equal
        """This test can be found in scipy.stats as ttest_ind"""
        data_1_var, data_2_var = np.var(data_1, ddof=1), np.var(data_2, ddof=1)
        data_1_n, data_2_n = len(data_1), len(data_2)
        df = np.power((data_1_var / data_1_n) + (data_2_var / data_2_n), 2) /\
             ((np.power(data_1_var, 2) / (np.power(data_1_n, 2) * data_1_n - 1)) +
              (np.power(data_2_var, 2) / (np.power(data_2_n, 2) * data_2_n - 1)))
        standard_error_difference = sqrt((data_1_var / data_1_n) + (data_2_var / data_2_n))
    t_value = (data_1_mean - data_2_mean) / standard_error_difference
    p = (1.0 - t.cdf(abs(t_value), df))
    if alternative.casefold() == 'two-sided':
        p *= 2
    return t_value, p


def one_sample_proportion_z_test(sample_data, pop_mean, alternative='two-sided'):
    """Found in statsmodels as proportions_ztest
    Used when comparing whether our observed proportion mean is difference to the population mean, assuming that the
    proportion mean is normally distributed.

    Parameters
    ----------
    sample_data: list or numpy array, must be binary
        An array containing all observations, marked as a 0 for failure and a 1 for success
    pop_mean: float
        Our expected proportion of success
    alternative: str, default is two-sided
        Our alternative hypothesis. It can be two-sided, less or greater

    Return
    ------
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
    if alternative.casefold() not in ['two-sided', 'greater', 'less']:
        raise ValueError("Cannot determine method for alternative hypothesis")
    sample_data = _check_table(sample_data)
    if not np.array_equal(sample_data, sample_data.astype(bool)):
        raise AttributeError("Cannot perform a proportion test on non-binary data")
    if len(np.where(sample_data == 1)[0]) < 10 or len(np.where(sample_data == 0)[0]) < 10:
        raise AttributeError("Too few instances of success or failure to run proportion test")
    p = np.mean(sample_data)
    q = 1 - p
    n = len(sample_data)
    std = sqrt((p * q) / n)
    z_score = (p - pop_mean) / std
    if alternative.casefold() == 'two-sided':
        p = 2 * (1 - norm.cdf(abs(z_score)))
    elif alternative.casefold() == 'greater':
        p = 1 - norm.cdf(z_score)
    else:
        p = norm.cdf(z_score)
    return z_score, p


def two_sample_proportion_z_test(data_1, data_2, alternative='two-sided'):
    """Found in statsmodels as proportions_ztest
    Used when we are comparing whether or not two proportion means are the same, given that both of them come from a
    normal distribution.

    Parameters
    ----------
    data_1: list or numpy array, must be binary
        An array containing all observations, marked as a 0 for failure and a 1 for success, that we are comparing to
        data_2
    data_2: list or numpy array, must be binary
        An array containing all observations, marked as a 0 for failure and a 1 for success, that we are comparing to
        data_1
    alternative: str, default is two-sided
        Our alternative hypothesis. It can be two-sided, less or greater

    Return
    ------
    z_score: float
        Our z-statistic to analyze the likelihood that our observed difference is due to chance
    p: float, 0 <= p <= 1
        The probability that the differences between two samples, assuming a normal distribution, is due to chance
    """
    data_1, data_2 = _check_table(data_1), _check_table(data_2)
    if not np.array_equal(data_1, data_1.astype(bool)):
        raise AttributeError("Cannot perform a proportion test on non-binary data for data_1")
    if not np.array_equal(data_2, data_2.astype(bool)):
        raise AttributeError("Cannot perform a proportion test on non-binary data for data_2")
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ['two-sided', 'greater', 'less']:
        raise ValueError("Cannot determine method for alternative hypothesis")
    n_1, n_2 = len(data_1), len(data_2)
    p_1, p_2 = np.mean(data_1), np.mean(data_2)
    p = (p_1 * n_1 + p_2 * n_2) / (n_1 + n_2)
    q = 1 - p
    se = sqrt((p * q) * ((1 / n_1) + (1 / n_2)))
    z_score = (p_1 - p_2) / se
    if alternative.casefold() == 'two-sided':
        p = 2 * (1 - norm.cdf(abs(z_score)))
    elif alternative.casefold() == 'greater':
        p = 1 - norm.cdf(z_score)
    else:
        p = norm.cdf(z_score)
    return z_score, p

# To-do: add unit tests for two_sample_f_test
def two_sample_f_test(data_1, data_2, alternative='two-sided'):
    """No method in scipy or statsmodels to immediately calculate this.
    Used to determine if two populations/samples have the same variance. Note that, due to this being a ratio between
    data_1 and data_2, a large p-value is just as significant as a small p-value. Also note that this test is extremely
    sensitive to data that is non-normal, so only use this test if the samples have been verified to come from a normal
    distribution.

    Parameters
    ----------
    data_1: list or numpy array
        The observed measurements for our first sample
    data_2: list or numpy array
        The observed measurements for our second sample
    alternative: str, default is two-sided
        Our alternative hypothesis. It can be two-sided, less or greater

    Return
    ------
    f_statistic: float
        A ratio of the variance of data_1 to data_2
    p: float
        The likelihood that this ratio could occur from two two samples with equal variances, due to chance
    """
    data_1, data_2 = _check_table(data_1), _check_table(data_2)
    df_1, df_2 = len(data_1) - 1, len(data_2) - 1
    var_1, var_2 = np.var(data_1, ddof=1), np.var(data_2, ddof=1)
    f_statistic = var_1 / var_2
    if alternative.casefold() == 'two-sided':
        p = 2 * (1 - f.cdf(abs(f_statistic), df_1, df_2))
    elif alternative.casefold() == 'greater':
        p = 1 - norm.cdf(f_statistic, df_1, df_2)
    else:
        p = norm.cdf(f_statistic, df_1, df_2)
    return f_statistic, p
