from math import sqrt
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from scipy.stats import t, median_abs_deviation, chi2
from scipy.special import erfc

from StatsTest.utils import _check_table


def tukey_fence_test(
    data: Union[Sequence, np.ndarray], coef: float = 1.5
) -> np.ndarray:
    """Not found in either scipy.stats or statsmodels

    Used to determine outliers in a normally distributed dataset.

    Parameters
    ----------
    data: list or numpy array, 1-D
        The data we are analyzing for outliers
    coef: float, default=1.5
        The coefficent we are multiplying IQR by to determine outliers

    Returns
    -------
    A numpy array containing all datapoints that are either more extreme than q1 - iqr * coef or q3 + iqr * coef
    """
    data = _check_table(data, only_count=False)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + iqr * coef
    lower_bound = q1 - iqr * coef
    return np.hstack([data[data < lower_bound], data[data > upper_bound]])


def grubbs_test(
    data: Union[Sequence, np.ndarray],
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> Optional[Union[float, int]]:
    """Not found in either scipy.stats or statsmodels

    Used to determine if there exists one outlier in the dataset based on their dispersion from the mean.
    Note that this assumes that the data is normally distributed.

    Parameters
    ----------
    data: list or numpy array, 1-D
        The sample dataset we are evaluating for outliers
    alternative: {'two-sided', 'greater', 'less'}
        Whether we are evaluating only minimum values, maximum values or both
    alpha: float, default=0.05
        Our alpha level for determining significant difference

    Returns
    -------
    If there is an outlier, returns the outlier. Else, returns None
    """
    if not isinstance(alternative, str):
        raise TypeError("Alternative Hypothesis is not of string type")
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
        raise ValueError("Cannot determine method for alternative hypothesis")
    if not isinstance(alpha, float):
        raise TypeError("Cannot discern alpha level for Grubb's test")
    if alpha > 1 or alpha < 0:
        raise ValueError("Alpha level must be within 0 and 1")
    data = _check_table(data, only_count=False)
    y_bar, s, n = np.mean(data), np.std(data, ddof=1), len(data)
    if alternative.casefold() == "less":
        return_val = np.min(data)
        val = y_bar - return_val
        t_value = t.isf(alpha / (2 * n), n - 2)
    elif alternative.casefold() == "greater":
        return_val = np.max(data)
        val = return_val - y_bar
        t_value = t.isf(alpha / (2 * n), n - 2)
    else:
        val = np.max([y_bar - np.min(data), np.max(data) - y_bar])
        if val == y_bar - np.min(data):
            return_val = np.min(data)
        else:
            return_val = np.max(data)
        t_value = t.isf(alpha / n, n - 2)
    g = val / s
    rejection_stat = ((n - 1) / sqrt(n)) * sqrt(
        pow(t_value, 2) / (n - 2 + pow(t_value, 2))
    )
    if g > rejection_stat:
        return return_val
    else:
        return None


def extreme_studentized_deviate_test(
    data: Union[Sequence, np.ndarray], num_outliers: int = 1, alpha: float = 0.05
) -> Tuple[int, List]:
    """Not found in either scipy.stats or statsmodels

    Used when we think there are at most k outliers, as other tests such as Grubbs or Tietjen-Moore rely on there existing
    exactly k number of outliers. Note that this assumes the data is normally distributed.

    Parameters
    ----------
    data: list or numpy array, 1-D
        The data we are analyzing for outliers
    num_outliers: int, default=1
        The maximum number of outliers we are checking for
    alpha: float, default=0.05
        The level of significance for determining outliers

    Returns
    -------
    max_outliers: int
        The maximum number of outliers that out test found to exist
    outliers: list
        The outliers corresponding to num_outliers
    """
    data = _check_table(data, only_count=False)
    if not isinstance(num_outliers, int):
        raise TypeError("Number of outliers must be an integer")
    if num_outliers < 0:
        raise ValueError("Cannot test for negative amount of outliers")
    r = np.zeros(num_outliers)
    if alpha >= 1 or alpha <= 0:
        raise ValueError("Alpha level must be within 0 and 1")
    outliers = []
    data_copy = np.copy(data)
    n = len(data)
    if num_outliers > n:
        raise ValueError(
            "Cannot have number of outliers greater than number of observations"
        )
    for i in range(1, num_outliers + 1):
        y_bar = np.mean(data_copy)
        s = np.std(data_copy, ddof=1)
        abs_resids = np.abs(data_copy - y_bar)
        r_i = np.max(abs_resids) / s
        p = 1 - (alpha / (2 * (n - i + 1)))
        lambda_i = ((n - i) * t.isf(p, n - i - 1)) / sqrt(
            (n - i - 1 + pow(t.isf(p, n - i - 1), 2)) * (n - i + 1)
        )
        r[i - 1] = r_i > abs(lambda_i)
        outliers.append(data_copy[np.argsort(abs_resids)][-1:][0])
        data_copy = data_copy[np.argsort(abs_resids)][:-1]
    max_outliers = np.max(np.where(r == 1)[0]) + 1
    return max_outliers, outliers[:max_outliers]


def tietjen_moore_test(
    data: Union[Sequence, np.ndarray],
    num_outliers: int = 1,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> Optional[List]:
    """Not found in either scipy.stats or statsmodels

    An extension of Grubbs where it is used to determine if there exists exactly k outliers in the dataset based on their
    dispersion from the mean. Note that this assumes that the data is normally distributed.

    Parameters
    ---------
    data: list or numpy array
        The dataset we are evaluating for outliers
    num_outliers: int
        The number of outliers we suspect to exist in the dataset
    alternative: {'two-sided', 'greater', 'less'}
        Whether we are looking at outliers that maximal, minimal or a mixture of both
    alpha: float
        Our level of confidence for rejecting the null that no outliers exist

    Returns
    -------
    If we found k outliers, returns the k outliers we detected. Else, returns None
    """
    data = _check_table(data, only_count=False)
    if alpha >= 1 or alpha <= 0:
        raise ValueError("Alpha level must be within 0 and 1")
    if not isinstance(num_outliers, int):
        raise TypeError("Number of outliers must be an integer")
    if num_outliers < 0:
        raise ValueError("Cannot test for negative amount of outliers")
    n = len(data)
    sort_data = np.sort(data)
    if num_outliers > n:
        raise ValueError(
            "Cannot have number of outliers greater than number of observations"
        )

    def teitjen(data, num_outliers=1, alternative="two-sided", simulation=True):
        y_bar = np.mean(data)
        if alternative.casefold() == "greater":
            data_large = sort_data[:-num_outliers]
            outliers = sort_data[len(data) - num_outliers :]
            y_bar_large = np.mean(data_large)
            l = np.sum(np.power(data_large - y_bar_large, 2)) / np.sum(
                np.power(data - y_bar, 2)
            )
        elif alternative.casefold() == "less":
            data_small = sort_data[num_outliers:]
            outliers = sort_data[:num_outliers]
            y_bar_small = np.mean(data_small)
            l = np.sum(np.power(data_small - y_bar_small, 2)) / np.sum(
                np.power(data - y_bar, 2)
            )
        else:
            abs_resids = np.abs(data - y_bar)
            z = data[np.argsort(abs_resids)]
            outliers = z[len(data) - num_outliers :]
            z_large = z[:-num_outliers]
            z_bar = np.mean(z_large)
            l = np.sum(np.power(z_large - z_bar, 2)) / np.sum(np.power(z - y_bar, 2))
        if simulation:
            return l, outliers
        else:
            return l

    l, outliers = teitjen(data, num_outliers, alternative, simulation=True)
    E_norm = np.random.normal(size=(10000, n))
    tietjen_E = np.apply_along_axis(
        teitjen,
        1,
        E_norm,
        num_outliers=num_outliers,
        alternative=alternative,
        simulation=False,
    )
    critical_value = np.percentile(tietjen_E, alpha * 100)
    if l < critical_value:
        return outliers
    else:
        return None


def chauvenet_test(data: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Not found in either scipy.stats or statsmodels.

    Based off of the Chauvenet criterion, which is that any data is an outlier if its error function is less than
    1 / (2 * number of entries).

    Parameters
    ----------
    data: list or numpy array
        The data we are evaluating for outliers

    Returns
    -------
    A numpy array that contains all observations deemed an outlier by the Chauvenet Criterion.
    """
    data = _check_table(data, only_count=False)
    mean, std, n = np.mean(data), np.std(data, ddof=1), len(data)
    z = np.abs(data - mean) / std
    criterion = 1 / (2 * n)
    prob = erfc(z)
    return data[prob < criterion]


def peirce_test(
    observed: Union[Sequence, np.ndarray],
    expected: Union[Sequence, np.ndarray],
    num_outliers: int = 1,
    num_coef: int = 1,
) -> np.ndarray:
    """Not found in either scipy.stats or statsmodels

    Parameters
    ----------
    observed: list or numpy array
        Our observed observations
    expected: list or numpy array
        Our expected observations, or what the model outputted for "Y"
    num_outliers: int, default=1
        The number of outliers we are trying to identify.
    num_coef: int, default=1
        The number of regression variables we are thinking of including

    Returns
    -------
    An array, containing all values that we found to be an outlier according to Peirce's criteria.
    """
    if not isinstance(num_outliers, int):
        raise TypeError("Number of outliers needs to be an integer")
    if num_outliers < 0:
        raise ValueError("Number of outliers has to be a positive value")
    if not isinstance(num_coef, int):
        raise TypeError("Number of regression coefficients needs to be an integer")
    if num_coef < 0:
        raise ValueError("Number of regression coefficients has to be a positive value")
    observed, expected = _check_table(observed), _check_table(expected)
    if len(observed) != len(expected):
        raise ValueError("Length of observed and expected need to be the same")
    n = len(observed)
    if num_outliers > n:
        raise ValueError(
            "Cannot have number of outliers greater than number of observations"
        )
    if num_coef > n:
        raise Warning(
            "Number of regressor variables is greater than number of observations"
        )
    q = (
        pow(num_outliers, num_outliers / n)
        * pow(n - num_outliers, (n - num_outliers) / n)
        / n
    )
    r_new, r_old = 1.0, 0.0
    while abs(r_new - r_old) > (n * 2.0e-16):
        ldiv = pow(r_new, num_outliers) if pow(r_new, num_outliers) != 0 else 1.0e-6
        lambda1 = pow(q, n) / pow(ldiv, 1 / (n - num_coef))
        x2 = 1 + (n - num_coef - num_outliers) / (
            num_outliers * (1.0 - pow(lambda1, 2))
        )
        if x2 < 0:
            x2 = 0.0
            r_old = r_new
        else:
            r_old = r_new
            r_new = np.exp((x2 - 1) / 2.0) * erfc(np.sqrt(x2 / 2))
    mean_squared_error = np.sum(np.power(observed - expected, 2)) / n
    threshold = x2 * mean_squared_error
    return observed[np.power(observed - expected, 2) > threshold]


def dixon_q_test(data: Union[Sequence, np.ndarray], alpha: float = 0.01) -> np.ndarray:
    """Not found in either scipy.stats or statsmodels

    Parameters
    ----------
    data: list or numpy array, 1-D
        Our observations we are analyzing to check for outliers
    alpha: float, options are 0.01, 0.05 and 0.1
        Our alpha level for checking critical values.

    Returns
    -------
    All values that Dixon's Q Test found to be outliers
    """
    data = _check_table(data)
    n = len(data)
    if n < 3:
        raise AttributeError("Cannot run Dixon's Q Test with less than 3 observations")
    if n > 30:
        raise AttributeError(
            "Too many observations, cannot determine critical score for Q test"
        )
    if alpha not in [0.01, 0.05, 0.1]:
        raise ValueError("Cannot determine alpha level for critical value")
    sort_data = np.sort(data)
    gap = np.hstack([np.abs(np.diff(sort_data)), sort_data[n - 1] - sort_data[n - 2]])
    range_ = np.ptp(sort_data)
    q = gap / range_
    q90 = [
        0.941,
        0.765,
        0.642,
        0.56,
        0.507,
        0.468,
        0.437,
        0.412,
        0.392,
        0.376,
        0.361,
        0.349,
        0.338,
        0.329,
        0.32,
        0.313,
        0.306,
        0.3,
        0.295,
        0.29,
        0.285,
        0.281,
        0.277,
        0.273,
        0.269,
        0.266,
        0.263,
        0.26,
    ]

    q95 = [
        0.97,
        0.829,
        0.71,
        0.625,
        0.568,
        0.526,
        0.493,
        0.466,
        0.444,
        0.426,
        0.41,
        0.396,
        0.384,
        0.374,
        0.365,
        0.356,
        0.349,
        0.342,
        0.337,
        0.331,
        0.326,
        0.321,
        0.317,
        0.312,
        0.308,
        0.305,
        0.301,
        0.29,
    ]

    q99 = [
        0.994,
        0.926,
        0.821,
        0.74,
        0.68,
        0.634,
        0.598,
        0.568,
        0.542,
        0.522,
        0.503,
        0.488,
        0.475,
        0.463,
        0.452,
        0.442,
        0.433,
        0.425,
        0.418,
        0.411,
        0.404,
        0.399,
        0.393,
        0.388,
        0.384,
        0.38,
        0.376,
        0.372,
    ]
    if alpha == 0.01:
        return sort_data[q > q99[n - 3]]
    elif alpha == 0.05:
        return sort_data[q > q95[n - 3]]
    else:
        return sort_data[q > q90[n - 3]]


def thompson_tau_test(data: Union[Sequence, np.ndarray], alpha: float = 0.05) -> List:
    """Not found in either scipy.stats or statsmodels

    Uses the Thompson-Tau criteria to iteratively identify outliers until no more exist.

    Parameters
    ----------
    data: list or numpy array, 1-D
        Our dataset we are evaluating for outliers
    alpha: float, default=0.05
        Our level of significance for detecting outliers

    Returns
    -------
    outliers_list: list
        A list containing all datapoints that we found to be an outlier by Thompson-Tau's criteria
    """
    data = _check_table(data, only_count=False)
    if alpha < 0 or alpha > 1:
        raise ValueError("Cannot have alpha level greater than 1 or less than 0")
    outlier_exist, outlier_table = True, []
    data_copy = np.copy(data)
    while outlier_exist:
        n, mu, s = len(data_copy), np.mean(data_copy), np.std(data_copy, ddof=1)
        ab_resid = np.abs(data_copy - mu) / s
        rejection = (
            t.isf(alpha / 2, n - 2)
            * (n - 1)
            / (sqrt(n) * sqrt(n - 2 + pow(t.isf(alpha / 2, n - 2), 2)))
        )
        is_outlier = ab_resid > rejection
        if np.sum(is_outlier) != 0:
            outlier_table.append(data_copy[np.argsort(ab_resid)][-1:][0])
            data_copy = data_copy[np.argsort(ab_resid)][:-1]
        else:
            outlier_exist = False
    return outlier_table


def mad_median_test(data: Union[Sequence, np.ndarray], alpha: float = 0.05) -> List:
    """Not found in either scipy.stats or statsmodels

    Uses the median absolute deviation rule as a method of outlier detection.

    Parameters
    ----------
    data: list or numpy array
        Our dataset we are evaluating for outliers
    alpha: float, default=0.05
        Our level of confidence for detecting outliers

    Returns
    -------
    A list containing all datapoints that we found to be an outlier by the MAD rule
    """
    data = _check_table(data, only_count=False)
    if alpha < 0 or alpha > 1:
        raise ValueError("Cannot have alpha level greater than 1 or less than 0")
    median = np.median(data)
    mad = median_abs_deviation(data)
    mad_med_obs = np.abs(data - median) / mad
    return data[mad_med_obs > sqrt(chi2.ppf(1 - (alpha / 2.0), 1))]
