import numpy as np
from math import factorial
from scipy.stats import chi2, binom


def _hypergeom_distribution(a, b, c, d):
    return (factorial(a + b) * factorial(c + d) * factorial(a + c) * factorial(b + d)) / \
           (factorial(a) * factorial(b) * factorial(c) * factorial(d) * factorial(a + b + c + d))


def _check_table(table):
    if isinstance(table, list):
        assert all(isinstance(item, int) for row in table for item in row), \
            "Cannot do perform statistical test with with non-integer counts"
    elif isinstance(table, (np.ndarray, np.generic)):
        assert np.issubdtype(table.dtype, np.integer), "Cannot perform statistical test with non-integer counts"
    table = np.array(table)
    assert np.all(table > 0), "Cannot have negative counts"
    return table


def chi_squared_test(cont_table):
    """Found in scipy as chi2_contingency"""
    cont_table = _check_table(cont_table)
    df = (cont_table.shape[0] - 1) * (cont_table.shape[1] - 1)
    X = 0
    col_sum, row_sum = np.sum(cont_table, axis=0), np.sum(cont_table, axis=1)
    for row in range(cont_table.shape[0]):
        for col in range(cont_table.shape[1]):
            expected = col_sum[col] * row_sum[row] / np.sum(row_sum)
            X += pow((cont_table[row, col] - expected), 2) / expected
    p = chi2.sf(X, df)
    return X, p


def goodness_of_fit_test(observed, expected=None):
    """Found in scipy as chisquare"""
    observed = _check_table(observed)
    if not expected:
        expected = np.array([np.mean(observed)] * len(observed))
    else:
        expected = _check_table(expected)
    df = len(observed) - 1
    X = np.sum(np.power(observed - expected, 2) / expected)
    p = chi2.sf(X, df)
    return X, df


def fisher_test(cont_table, alternative='two-sided'):
    """Found in scipy as fisher_exact"""
    assert alternative.casefold() in ['two-sided', 'greater', 'less'], \
        "Cannot determine method for alternative hypothesis"
    cont_table = _check_table(cont_table)
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
    """Found in statsmodels as mcnemar"""
    cont_table = _check_table(cont_table)
    assert cont_table.shape == (2, 2), \
        "McNemar's Test is meant for a 2x2 contingency table, use cmh_test for {}x{} table".format(
            cont_table.shape[0], cont_table.shape[1])
    b, c = cont_table[0, 1], cont_table[1, 0]
    if b + c > 25:
        chi_squared = pow(abs(b - c) - 1, 2) / (b + c)
        p = chi2.sf(chi_squared, 1)
    else:
        chi_squared = min(b, c)
        p = 2 * binom.cdf(chi_squared, b+c, 0.5) - binom.pmf(binom.ppf(0.99, b + c, 0.5), b + c, 0.5)
    return chi_squared, p
