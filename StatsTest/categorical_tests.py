import numpy as np
from scipy.stats import chi2, binom
from StatsTest.utils import _check_table, _hypergeom_distribution


def chi_squared_test(cont_table):
    """Found in scipy.stats as chi2_contingency.
    Determines the difference between what we expect the count of a group to be versus what what was observed in our
    contingency table. Assuming our data follows a chi distribution (i.e., observations are independent), if the observed
    variances are found to be very high given the number of observations, then we reject our null hypothesis and
    conclude that this difference could not occur due to chance.

    Parameters
    ----------
    cont_table: list or numpy array
        A contingency table containing 2 counts of 2, or 4 counts total. As an example of expected output, refer to a
        confusion matrix for predicting a binary variable.

    Return
    ------
    X: float
        The Chi test statistic, or the variance of the difference of our observed results versus expected results.
    p: float, 0 <= p <= 1
        The likelihood that we would observe our X value given the number of observations we had.
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
    A likelihood ratio test used for determine if the difference between our observed results and expected results in
    our contingency table are likely to happen due to chance.

    Parameters
    ----------
    cont_table: list or numpy array
        A contingency table containing 2 counts of 2, or 4 counts total. As an example of expected output, refer to a
        confusion matrix for predicting a binary variable.

    Return
    ------
    g: float
        The G statistic, or the likelihood ratio of the difference between observed and expected
    p: float, 0 <= p <= 1
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
    Used when we cannot divide the data cleanly into a contingency table or when we have actual expected results to
    compare to.

    Parameters
    ----------
    observed: list or numpy array
        Our observed data
    expected: (Optional) list or numpy array
        What we expected the results to be. If none given, then we expect all data points to be equally likely

    Return
    ------
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
    observed: list or numpy array
        Our observed data
    expected: (Optional) list or numpy array
        What we expected the results to be. If none given, then we expect all data points to be equally likely

    Return
    ------
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


def fisher_test(cont_table, alternative='two-sided'):
    """Found in scipy.stats as fisher_exact
    Used to determine the exact likelihood that we would observe a measurement in our 2x2 contingency table that
    is just as extreme, if not moreso, than our observed results.

    Parameters
    ----------
    cont_table: list or numpy array
        A 2x2 contingency table
    alternative: str, default is two-sided
         What our alternative hypothesis is. It can be two-sided, less or greater

    Return
    ------
    p: float, 0 <= p <= 1
        The exact likelihood of finding a more extreme measurement than our observed data
    """
    if alternative.casefold() not in ['two-sided', 'greater', 'less']:
        raise ValueError("Cannot determine method for alternative hypothesis")
    cont_table = _check_table(cont_table, True)
    if cont_table.shape != (2, 2):
        raise AttributeError("Fisher's Exact Test is meant for a 2x2 contingency table")
    a, b, c, d = cont_table[0, 0], cont_table[0, 1], cont_table[1, 0], cont_table[1, 1]
    p = _hypergeom_distribution(a, b, c, d)

    # left side
    def left_side(a, b, c, d):
        num_steps = min(a, d)
        p_val = []
        for i in range(num_steps):
            a -= 1
            b += 1
            c += 1
            d -= 1
            p_val.append(_hypergeom_distribution(a, b, c, d))
        return p_val

    # right side
    def right_side(a, b, c, d):
        num_steps = min(b, c)
        p_val = []
        for i in range(num_steps):
            a += 1
            b -= 1
            c -= 1
            d += 1
            p_val.append(_hypergeom_distribution(a, b, c, d))
        return p_val

    if alternative.casefold() == 'greater':
        right_p_val = right_side(a, b, c, d)
        return p + np.sum(right_p_val)
    elif alternative.casefold() == 'less':
        left_p_val = left_side(a, b, c, d)
        return p + np.sum(left_p_val)
    else:
        left_p_val, right_p_val = left_side(a, b, c, d), right_side(a, b, c, d)
        all_p = right_p_val + left_p_val
        return p + np.sum([i for i in all_p if i <= p])


def mcnemar_test(cont_table):
    """Found in statsmodels as mcnemar
    Used when we have paired nominal data that is organized in a 2x2 contingency table. It is used to test the
    assumption that the marginal column and row probabilities are equal, i.e., that the probability that b and c
    are equivalent.

    Parameters
    ----------
    cont_table: list or numpy array
        A 2x2 contingency table

    Return
    ------
    chi_squared: float
        Our Chi statistic, or the sum of differences between b and c
    p: float, 0 <= p <= 1
        The probability that b and c aren't equivalent due to chance
    """
    cont_table = _check_table(cont_table, True)
    if cont_table.shape != (2, 2):
        raise AttributeError("McNemar's Test is meant for a 2x2 contingency table")
    b, c = cont_table[0, 1], cont_table[1, 0]
    if b + c > 25:
        chi_squared = pow(abs(b - c) - 1, 2) / (b + c)
        p = 1 - chi2.cdf(chi_squared, 1)
    else:
        chi_squared = min(b, c)
        p = 2 * binom.cdf(chi_squared, b + c, 0.5) - binom.pmf(binom.ppf(0.99, b + c, 0.5), b + c, 0.5)
    return chi_squared, p


def cmh_test(*args):
    """Found in statsmodels as Stratified Table.test_null_odds()
    Used when we want to evaluate the association between a binary predictor/treatment and a binary outcome variable
    with data that is stratified or matched. Our null hypothesis is that the common-odds ratio is 1 while our
    alternative is that the common odds ratio is not equal to one (i.e., a two-tailed test). This test is an
    extension of the mcnemar test to handle any arbitrary strata size.

    Parameters
    ----------
    args: list or numpy array
        A group of 2x2 contingency tables, where each group represents a strata

    Returns
    -------
    epsilon: float
        Our test statistic, used to evaluate the likelihood that all strata have the same common odds ratio
    p: float
        The likelihood that our common odds ratio would not equal one if we were to randomly sample strata from the same
        population
    """
    a, row_sum, col_sum, total, n_i, m_i = [], [], [], [], [], []
    for arg in args:
        arg = _check_table(arg)
        if arg.shape != (2, 2):
            raise AttributeError("CMH Test is meant for 2x2 contingency tables")
        a = np.append(a, arg[0, 0])
        row_sum = np.append(row_sum, np.sum(arg, axis=1))
        col_sum = np.append(col_sum, np.sum(arg, axis=0))
        total = np.append(total, np.sum(np.sum(arg, axis=0)))
        n_i, m_i = np.append(n_i, np.sum(arg, axis=1)[0]), np.append(m_i, np.sum(arg, axis=0)[0])
    top = pow(abs(np.sum(a - (n_i * m_i / total))), 2)
    bottom = np.sum((n_i * (total - n_i) * m_i * (total - m_i)) / (np.power(total, 2) * (total - 1)))
    epsilon = top / bottom
    p = 1 - chi2.cdf(epsilon, 1)
    return epsilon, p
