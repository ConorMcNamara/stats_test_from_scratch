import numpy as np
from scipy.stats import chi2, binom
from src.utils import _check_table, _hypergeom_distribution


def chi_squared_test(cont_table):
    """Found in scipy.stats as chi2_contingency

    Parameters
    ----------
    cont_table: list or numpy array
        A contingency table to perform our chi_squared test

    Return
    ------
    X: float
        The Chi statistic
    p: float
        The likelihood that our observed differences are due to chance
    """
    cont_table = _check_table(cont_table, True)
    df = (cont_table.shape[0] - 1) * (cont_table.shape[1] - 1)
    X = 0
    col_sum, row_sum = np.sum(cont_table, axis=0), np.sum(cont_table, axis=1)
    for row in range(cont_table.shape[0]):
        for col in range(cont_table.shape[1]):
            expected = col_sum[col] * row_sum[row] / np.sum(row_sum)
            X += pow((cont_table[row, col] - expected), 2) / expected
    p = 1 - chi2.cdf(X, df)
    return X, p


def g_test(cont_table):
    """Found in scipy.stats as chi2_contingency(lambda_="log-likelihood")

    Parameters
    ----------
    cont_table: list or numpy array
        A contingency table to perform our G Test

    Return
    ------
    g: float
        The G statistic
    p: float
        The likelihood that our observed differences are due to chance
    """
    cont_table = _check_table(cont_table, True)
    df = (cont_table.shape[0] - 1) * (cont_table.shape[1] - 1)
    g = 0
    col_sum, row_sum = np.sum(cont_table, axis=0), np.sum(cont_table, axis=1)
    for row in range(cont_table.shape[0]):
        for col in range(cont_table.shape[1]):
            expected = col_sum[col] * row_sum[row] / np.sum(row_sum)
            g += cont_table[row, col] * np.log(cont_table[row, col] / expected)
    g *= 2
    p = 1 - chi2.cdf(g, df)
    return g, p


def chi_goodness_of_fit_test(observed, expected=None):
    """Found in scipy.stats as chisquare

    Parameters
    ----------
    observed: list or numpy array
        Our observed data
    expected: (Optional) list or numpy array
        What we expected the results to be. If none given, then we expect all data points to be equally likely

    Return
    ------
    X: float
        The Chi statistic
    p: float
        The likelihood that our observed differences are due to chance
    """
    observed = _check_table(observed, False)
    if not expected:
        expected = np.repeat(np.mean(observed), len(observed))
    else:
        expected = _check_table(expected)
    df = len(observed) - 1
    X = np.sum(np.power(observed - expected, 2) / expected)
    p = 1 - chi2.cdf(X, df)
    return X, df


def g_goodness_of_fit_test(observed, expected=None):
    """Found in scipy.stats as power_divergence(lambda_="log-likelihood")

    Parameters
    ----------
    observed: list or numpy array
        Our observed data
    expected: (Optional) list or numpy array
        What we expected the results to be. If none given, then we expect all data points to be equally likely

    Return
    ------
    g: float
        The G statistic
    p: float
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
    return g, df


def fisher_test(cont_table, alternative='two-sided'):
    """Found in scipy.stats as fisher_exact

    Parameters
    ----------
    cont_table: list or numpy array
        A contingency table to perform our fisher test
    alternative: str
         What our alternative hypothesis is. It can be two-sided, less or greater

    Return
    ------
    p: float
        The probability that our observed differences are due to chance
    """
    assert alternative.casefold() in ['two-sided', 'greater', 'less'], \
        "Cannot determine method for alternative hypothesis"
    cont_table = _check_table(cont_table, True)
    assert cont_table.shape == (2, 2), \
        "Fisher's Exact Test is meant for a 2x2 contingency table, use Freeman-Halton Test for {}x{} table".format(
            cont_table.shape[0], cont_table.shape[1])
    a, b, c, d = cont_table[0, 0], cont_table[0, 1], cont_table[1, 0], cont_table[1, 1]
    p = 0
    if alternative.casefold() in ['two-sided', 'less']:
        num_steps = min(a, d) + 1
        for i in range(num_steps):
            p += _hypergeom_distribution(a, b, c, d)
            a -= 1
            b += 1
            c += 1
            d -= 1
        if alternative.casefold() == 'two-sided':
            p *= 2
    else:
        num_steps = min(b, c) + 1
        for i in range(num_steps):
            p += _hypergeom_distribution(a, b, c, d)
            a += 1
            b -= 1
            c -= 1
            d += 1
    return p


#def cmh_test(cont_table, alternative='two-sided'):
   # """Found in statsmodels as Stratified Table"""


def mcnemar_test(cont_table):
    """Found in statsmodels as mcnemar

    Parameters
    ----------
    cont_table: list or numpy array

    Return
    ------
    chi_squared: float
        Our chi-squared statistic
    p: float
        The probabiltiy that our observed differences were due to chance"""
    cont_table = _check_table(cont_table, True)
    assert cont_table.shape == (2, 2), \
        "McNemar's Test is meant for a 2x2 contingency table, use cmh_test for {}x{} table".format(
            cont_table.shape[0], cont_table.shape[1])
    b, c = cont_table[0, 1], cont_table[1, 0]
    if b + c > 25:
        chi_squared = pow(abs(b - c) - 1, 2) / (b + c)
        p = 1 - chi2.cdf(chi_squared, 1)
    else:
        chi_squared = min(b, c)
        p = 2 * binom.cdf(chi_squared, b + c, 0.5) - binom.pmf(binom.ppf(0.99, b + c, 0.5), b + c, 0.5)
    return chi_squared, p
