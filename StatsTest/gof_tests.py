from math import sqrt, log, asinh
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from scipy.stats import chi2, norm, statlib

from StatsTest.utils import _check_table, _skew, _kurtosis, _autocorr


def shapiro_wilk_test(data: Union[Sequence, np.ndarray]) -> Tuple[float, float]:
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
        raise AttributeError(
            "Cannot run Shapiro-Wilks Test with less than 3 datapoints"
        )
    zeroes = np.zeros(n // 2)
    data = np.sort(data)
    a, w, p, ifault = statlib.swilk(data, zeroes, 0)
    return w, p


def chi_goodness_of_fit_test(
    observed: Union[Sequence, np.ndarray],
    expected: Optional[Union[Sequence, np.ndarray]] = None,
) -> Tuple[float, float]:
    """Found in scipy.stats as chisquare

    Used when we cannot divide the data cleanly into a contingency table or when we have actual expected results to
    compare to.

    Parameters
    ----------
    observed: list or numpy array, 1-D
        Our observed data points
    expected: list or numpy array, 1-D, default=None
        What we expected the results to be. If None given, then we expect all data points to be equally likely

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


def g_goodness_of_fit_test(
    observed: Union[Sequence, np.ndarray],
    expected: Optional[Union[Sequence, np.ndarray]] = None,
) -> Tuple[float, float]:
    """Found in scipy.stats as power_divergence(lambda_="log-likelihood")

    Similar to chi_goodness_of_fit_test, used when we cannot divide the data cleanly into a contingency table or when we
    have actual expected results to compare to.

    Parameters
    ----------
    observed: list or numpy array, 1-D
        Our observed data
    expected: list or numpy array, default=None
        What we expected the results to be. If None given, then we expect all data points to be equally likely

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


def jarque_bera_test(data: Union[Sequence, np.ndarray]) -> Tuple[float, float]:
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


def ljung_box_test(
    data: Union[Sequence, np.ndarray],
    num_lags: Optional[Union[int, Sequence, np.ndarray]] = None,
) -> Tuple[float, float]:
    """Found in statsmodels as acorr_ljung(boxpierce=False)

    Used to determine if any group of autocorrelations in a time series dataset are different from zero

    Parameters
    ----------
    data: list or numpy array, 1-D
        The time series dataset we are performing our test on
    num_lags: int or list, default=None
        If int, the maximum number of time lags
        If list, then the series of time lags we are performing
        If None, then use np.arange(1, 10)

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


def box_pierce_test(
    data: Union[Sequence, np.ndarray],
    num_lags: Optional[Union[int, Sequence, np.ndarray]] = None,
) -> Tuple[int, int]:
    """Found in statsmodels as acorr_ljung(boxpierce=True)

    Used to determine if any group of autocorrelations in a time series dataset are different from zero

    Parameters
    ----------
    data: list or numpy array, 1-D
        The time series dataset we are performing our test on
    num_lags: int or list, default=None
        If int, the maximum number of time lags
        If list, then the series of time lags we are performing
        If None, then use np.arange(1, 11)

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


def skew_test(data: Union[Sequence, np.ndarray]) -> Tuple[float, float]:
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
        raise AttributeError(
            "Skew Test is not reliable on datasets with less than 8 observations"
        )
    n = len(data)
    skew = _skew(data)
    y2 = (36 * (n - 7) * (pow(n, 2) + 2 * n - 5)) / (
        (n - 2) * (n + 5) * (n + 7) * (n + 9)
    )
    u2 = 6 * (n - 2) / ((n + 1) * (n + 3))
    w2 = sqrt(2 * y2 + 4) - 1
    delta = 1 / sqrt(log(sqrt(w2)))
    alpha_2 = 2 / (w2 - 1)
    z = delta * asinh(skew / sqrt(alpha_2 * u2))
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def kurtosis_test(data: Union[Sequence, np.ndarray]) -> Tuple[float, float]:
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
        raise AttributeError(
            "Kurtosis Test is not reliable on datasets with less than 20 observations"
        )
    n = len(data)
    kurtosis = _kurtosis(data) - 3
    mean_kurt = -6 / (n + 1)
    var_kurt = 24 * n * (n - 2) * (n - 3) / (pow(n + 1, 2) * (n + 3) * (n + 5))
    skew_kurt = (6 * (pow(n, 2) - 5 * n + 2) / ((n + 7) * (n + 9))) * sqrt(
        6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3))
    )
    a = 6 + ((8 / skew_kurt) * (2 / skew_kurt + sqrt(1 + 4 / pow(skew_kurt, 2))))
    z_top = 1 - 2 / a
    z_bottom = 1 + ((kurtosis - mean_kurt) / sqrt(var_kurt)) * sqrt(2 / (a - 4))
    z = sqrt(9 * a / 2) * (
        1 - 2 / (9 * a) - np.sign(z_bottom) * np.power(z_top / abs(z_bottom), 1 / 3.0)
    )
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def k_squared_test(data: Union[Sequence, np.ndarray]) -> Tuple[float, float]:
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


def lilliefors_test(
    data: Union[Sequence, np.ndarray], alpha: float = 0.05
) -> Tuple[float, bool]:
    """Found in statsmodels as lilliefors.

    Used to determine if the data follows a normal distribution

    Parameters
    ----------
    data: list or numpy array, 1-D
        Contains all observations from our sample to measure if it follows  a normal distribution
    alpha: {0.01, 0.05, 0.10, 0.15, 0.20}, default=0.05
        Our alpha level for determining level of significant difference

    Returns
    -------
    d_max: float
        The maximum difference between our actual and expected distribution values
    bool: Whether our data follows a normal distribution or not
    """

    def f(x):
        return (0.83 + x) / sqrt(x) - 0.01

    data = _check_table(data)
    n = len(data)
    if n < 4:
        raise AttributeError(
            "Cannot perform Lilliefors Test on less than 4 observations"
        )
    if alpha not in [0.01, 0.05, 0.10, 0.15, 0.20]:
        raise ValueError("Cannot determine alpha level for Lilleifors Test")
    index = n - 4 if n <= 50 else 47

    q_01 = [
        0.4129,
        0.3959,
        0.3728,
        0.3504,
        0.3331,
        0.3162,
        0.3037,
        0.2905,
        0.2812,
        0.2714,
        0.2627,
        0.2545,
        0.2477,
        0.2408,
        0.2345,
        0.2285,
        0.2226,
        0.2190,
        0.2141,
        0.2090,
        0.2053,
        0.2010,
        0.1985,
        0.1941,
        0.1911,
        0.1886,
        0.1848,
        0.1820,
        0.1798,
        0.1770,
        0.1747,
        0.1720,
        0.1695,
        0.1677,
        0.1653,
        0.1634,
        0.1616,
        0.1599,
        0.1573,
        0.1556,
        0.1542,
        0.1525,
        0.1512,
        0.1499,
        0.1476,
        0.1463,
        0.1457,
        1.035 / f(n),
    ]
    q_05 = [
        0.3754,
        0.3427,
        0.3245,
        0.3041,
        0.2875,
        0.2744,
        0.2616,
        0.2506,
        0.2426,
        0.2337,
        0.2257,
        0.2196,
        0.2128,
        0.2071,
        0.2018,
        0.1965,
        0.1920,
        0.1881,
        0.1840,
        0.1798,
        0.1766,
        0.1726,
        0.1699,
        0.1665,
        0.1641,
        0.1614,
        0.1590,
        0.1559,
        0.1542,
        0.1518,
        0.1497,
        0.1478,
        0.1454,
        0.1436,
        0.1421,
        0.1402,
        0.1386,
        0.1373,
        0.1353,
        0.1339,
        0.1322,
        0.1309,
        0.1293,
        0.1282,
        0.1269,
        0.1256,
        0.1246,
        0.895 / f(n),
    ]
    q_10 = [
        0.3456,
        0.3118,
        0.2982,
        0.2802,
        0.2649,
        0.2522,
        0.2410,
        0.2306,
        0.2228,
        0.2147,
        0.2077,
        0.2016,
        0.1956,
        0.1902,
        0.1852,
        0.1803,
        0.1764,
        0.1726,
        0.1690,
        0.1650,
        0.1619,
        0.1589,
        0.1562,
        0.1533,
        0.1509,
        0.1483,
        0.1460,
        0.1432,
        0.1415,
        0.1392,
        0.1373,
        0.1356,
        0.1336,
        0.1320,
        0.1303,
        0.1288,
        0.1275,
        0.1258,
        0.1244,
        0.1228,
        0.1216,
        0.1204,
        0.1189,
        0.1180,
        0.1165,
        0.1153,
        0.1142,
        0.819 / f(n),
    ]
    q_15 = [
        0.3216,
        0.3027,
        0.2816,
        0.2641,
        0.2502,
        0.2382,
        0.2273,
        0.2179,
        0.2101,
        0.2025,
        0.1959,
        0.1899,
        0.1843,
        0.1794,
        0.1747,
        0.1700,
        0.1666,
        0.1629,
        0.1592,
        0.1555,
        0.1527,
        0.1498,
        0.1472,
        0.1448,
        0.1423,
        0.1398,
        0.1378,
        0.1353,
        0.1336,
        0.1314,
        0.1295,
        0.1278,
        0.1260,
        0.1245,
        0.1230,
        0.1214,
        0.1204,
        0.1186,
        0.1172,
        0.1159,
        0.1148,
        0.1134,
        0.1123,
        0.1113,
        0.1098,
        0.1089,
        0.1079,
        0.775 / f(n),
    ]
    q_20 = [
        0.3207,
        0.2893,
        0.2694,
        0.2521,
        0.2387,
        0.2273,
        0.2171,
        0.2080,
        0.2004,
        0.1932,
        0.1869,
        0.1811,
        0.1758,
        0.1711,
        0.1666,
        0.1624,
        0.1589,
        0.1553,
        0.1517,
        0.1484,
        0.1458,
        0.1429,
        0.1406,
        0.1381,
        0.1358,
        0.1334,
        0.1315,
        0.1291,
        0.1274,
        0.1254,
        0.1236,
        0.1220,
        0.1203,
        0.1188,
        0.1174,
        0.1159,
        0.1147,
        0.1131,
        0.1119,
        0.1106,
        0.1095,
        0.1083,
        0.1071,
        0.1062,
        0.1047,
        0.1040,
        0.1030,
        0.741 / f(n),
    ]
    if alpha == 0.01:
        d_x = q_01[index]
    elif alpha == 0.05:
        d_x = q_05[index]
    elif alpha == 0.10:
        d_x = q_10[index]
    elif alpha == 0.15:
        d_x = q_15[index]
    else:
        d_x = q_20[index]
    z_table = (data - np.mean(data)) / np.std(data, ddof=1)
    expected = norm.cdf(z_table)
    actual = np.cumsum(np.ones(len(data)) / len(data))
    diff = np.abs(actual - expected)
    d_max = np.max(diff)
    return d_max, d_max < d_x
