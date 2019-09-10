from StatsTest.utils import _check_table, _skew, _kurtosis, _autocorr
from scipy.stats import chi2, norm
from scipy.stats import statlib
import numpy as np
from math import sqrt, log, asinh


def shapiro_wilk_test(data):
    """Found in scipy.stats as shapiro
    Used to determine if a sample comes from a normally distributed population

    Parameters
    ----------
    data: list or numpy array, 1-D
        Our sample data

    Returns
    -------
    w: float
        Our test statistic for measuring the degree of normality in our data
    p: float, 0 <= p <= 1
        The likelihood that we would observe our data from a normally distributed population
    """
    data = _check_table(data, only_count=False)
    n = len(data)
    if n < 3:
        raise AttributeError("Cannot run Shapiro-Wilks Test with less than 3 datapoints")
    zeroes = np.zeros(n // 2)
    data = np.sort(data)
    a, w, p, ifault = statlib.swilk(data, zeroes, 0)
    return w, p


def chi_goodness_of_fit_test(observed, expected=None):
    """Found in scipy.stats as chisquare
    Used when we cannot divide the data cleanly into a contingency table or when we have actual expected results to
    compare to.

    Parameters
    ----------
    observed: list or numpy array, 1-D
        Our observed data points
    expected: (Optional) list or numpy array, 1-D
        What we expected the results to be. If none given, then we expect all data points to be equally likely

    Returns
    -------
    X: float
        The Chi statistic, or the sum of squared differences between observed and expected
    p: float, 0 <= p <= 1
        The likelihood that our observed differences, given the amount of data, can be attributed to chance
    """
    observed = _check_table(observed, False)
    if not expected:
        expected = np.repeat(np.mean(observed), len(observed))
    else:
        expected = _check_table(expected)
    df = len(observed) - 1
    X = np.sum(np.power(observed - expected, 2) / expected)
    p = 1 - chi2.cdf(X, df)
    return X, p


def g_goodness_of_fit_test(observed, expected=None):
    """Found in scipy.stats as power_divergence(lambda_="log-likelihood")
    Similar to chi_goodness_of_fit_test, used when we cannot divide the data cleanly into a contingency table or when we
    have actual expected results to compare to.

    Parameters
    ----------
    observed: list or numpy array, 1-D
        Our observed data
    expected: (Optional) list or numpy array
        What we expected the results to be. If none given, then we expect all data points to be equally likely

    Returns
    -------
    g: float
        The G statistic, or the likelihood ratio of the difference between observed and expected
    p: float, 0 <= p <= 1
        The likelihood that our observed differences are due to chance
    """
    observed = _check_table(observed, False)
    if not expected:
        expected = np.repeat(np.mean(observed), len(observed))
    else:
        expected = _check_table(expected)
    df = len(observed) - 1
    g = 2 * np.sum(observed * np.log(observed / expected))
    p = 1 - chi2.cdf(g, df)
    return g, p


def jarque_bera_test(data):
    """Found in statsmodels as jarque_bera
    This test is used to evaluate whether the skew and kurtosis of said data follows that of a normal distribution

    Parameters
    ----------
    data: list or numpy array, 1-D
        An array containing all observations from our data

    Returns
    -------
    jb: float
        Our test statistic, tells us how close our data is to matching a normal distribution. The closer to 0, the more
        normal the data is
    p_value: float, 0 <= p <= 1
        The likelihood that we would see this skew and kurtosis from a normal distribution due to chance
    """
    data = _check_table(data, only_count=False)
    n = len(data)
    skew = _skew(data)
    kurtosis = _kurtosis(data)
    jb = (n / 6) * (pow(skew, 2) + (pow(kurtosis - 3, 2) / 4))
    p_value = 1 - chi2.cdf(jb, 2)
    return jb, p_value


def ljung_box_test(data, num_lags=None):
    """Found in statsmodels as acorr_ljung(boxpierce=False)
    Used to determine if any group of autocorrelations in a time series dataset are different from zero

    Parameters
    ----------
    data: list or numpy array, 1-D
        The time series dataset we are performing our test on
    num_lags: int or list, default is none
        If int, the maximum number of time lags
        If list, then the series of time lags we are performing
        If none, then use np.arange(1, 10)

    Returns
    -------
    q: float
        The Ljung-Box statistic, or our measure of autocorrelations differing from zero
    p: float
        The likelihood that our observed autocorrelations would differ from zero due to chance
    """
    if num_lags is None:
        h_lags = np.arange(1, 11)
    elif isinstance(num_lags, int):
        h_lags = np.arange(1, num_lags + 1)
    elif isinstance(num_lags, list) or isinstance(num_lags, (np.ndarray, np.generic)):
        num_lags = _check_table(num_lags, only_count=False)
        h_lags = num_lags
    else:
        raise ValueError("Cannot discern number of lags")
    h = np.max(h_lags)
    n = len(data)
    n_repeat = np.repeat(n, h)
    box_sum = np.sum(pow(_autocorr(data, h_lags), 2) / (n_repeat - h_lags))
    q = n * (n + 2) * box_sum
    p = 1 - chi2.cdf(q, h)
    return q, p


def box_pierce_test(data, num_lags=None):
    """Found in statsmodels as acorr_ljung(boxpierce=True)
    Used to determine if any group of autocorrelations in a time series dataset are different from zero

    Parameters
    ----------
    data: list or numpy array, 1-D
        The time series dataset we are performing our test on
    num_lags: int or list, default is none
        If int, the maximum number of time lags
        If list, then the series of time lags we are performing
        If none, then use np.arange(1, 11)

    Returns
    -------
    q: float
        The Box-Pierce statistic, or our measure of autocorrelations differing from zero
    p: float, 0 <= p <= 1
        The likelihood that our observed autocorrelations would differ from zero due to chance
    """
    if num_lags is None:
        h_lags = np.arange(1, 11)
    elif isinstance(num_lags, int):
        h_lags = np.arange(1, num_lags + 1)
    elif isinstance(num_lags, list) or isinstance(num_lags, (np.ndarray, np.generic)):
        h_lags = _check_table(num_lags, only_count=False)
    else:
        raise ValueError("Cannot discern number of lags")
    h = np.max(h_lags)
    n = len(data)
    q = n * np.sum(pow(_autocorr(data, h_lags), 2))
    p = 1 - chi2.cdf(q, h)
    return q, p


def skew_test(data):
    """Found in scipy.stats as skewtest.
    Used to determine the likelihood that our sample dataset comes from a normal distribution based on its skewness.

    Parameters
    ----------
    data: list or numpy array, 1-D
        Contains all observations from our sample to measure departure from normality

    Returns
    -------
    z: float
        Our test statistic, or the measure of difference of our skewness compared to a normal distribution
    p: float, 0 <= p <= 1
        The likelihood that we would see the observed differences in skewness from a normal population due
        to chance
    """
    data = _check_table(data, only_count=False)
    if len(data) < 8:
        raise AttributeError("Skew Test is not reliable on datasets with less than 8 observations")
    n = len(data)
    skew = _skew(data)
    y2 = (36 * (n - 7) * (pow(n, 2) + 2 * n - 5)) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    u2 = 6 * (n - 2) / ((n + 1) * (n + 3))
    w2 = sqrt(2 * y2 + 4) - 1
    delta = 1 / sqrt(log(sqrt(w2)))
    alpha_2 = 2 / (w2 - 1)
    z = delta * asinh(skew / sqrt(alpha_2 * u2))
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def kurtosis_test(data):
    """Found in scipy.stats as kurtosistest.
    Used to determine the likelihood that our sample dataset comes from a normal distribution based on its kurtosis.

    Parameters
    ----------
    data: list or numpy array, 1-D
        Contains all observations from our sample to measure departure from normality

    Returns
    -------
    z: float
        Our test statistic, or the measure of difference of our kurtosis compared to a normal distribution
    p: float, 0 <= p <= 1
        The likelihood that we would see the observed differences in kurtosis from a normal population due
        to chance
    """
    data = _check_table(data, only_count=False)
    if len(data) < 20:
        raise AttributeError("Kurtosis Test is not reliable on datasets with less than 20 observations")
    n = len(data)
    kurtosis = _kurtosis(data) - 3
    mean_kurt = - 6 / (n + 1)
    var_kurt = 24 * n * (n - 2) * (n - 3) / (pow(n + 1, 2) * (n + 3) * (n + 5))
    skew_kurt = (6 * (pow(n, 2) - 5 * n + 2) / ((n + 7) * (n + 9))) * sqrt(6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    a = 6 + ((8 / skew_kurt) * (2 / skew_kurt + sqrt(1 + 4 / pow(skew_kurt, 2))))
    z_top = 1 - 2 / a
    z_bottom = 1 + ((kurtosis - mean_kurt) / sqrt(var_kurt)) * sqrt(2 / (a - 4))
    z = sqrt(9 * a / 2) * (1 - 2 / (9 * a) - np.sign(z_bottom) * np.power(z_top / abs(z_bottom), 1 / 3.0))
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def k_squared_test(data):
    """Found in scipy.stats as normaltest
    Used to determine the likelihood that our sample dataset comes from a normal distribution based on its
    skewness and kurtosis.

    Parameters
    ----------
    data: list or numpy array, 1-D
        Contains all observations from our sample to measure departure from normality

    Returns
    -------
    k2: float
        Our test statistic, or the measure of difference of our skewness and kurtosis compared to a normal distribution
    p: float, 0 <= p <= 1
        The likelihood that we would see the observed differences in skewness and kurtosis from a normal population due
        to chance
    """
    z1, _ = skew_test(data)
    z2, _ = kurtosis_test(data)
    k2 = pow(z1, 2) + pow(z2, 2)
    p = 1 - chi2.cdf(k2, 2)
    return k2, p