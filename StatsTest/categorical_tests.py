from typing import Sequence, Tuple, Union

import numpy as np

from scipy.stats import chi2, binom

from StatsTest.utils import _check_table, _hypergeom_distribution


def chi_squared_test(cont_table: Union[Sequence[Sequence], np.ndarray]) -> Tuple[float, float]:
    """Found in scipy.stats as chi2_contingency.

    Determines the difference between what we expect the count of a group to be versus what was observed in our
    contingency table. Assuming our data follows a chi distribution (i.e., observations are independent), if the observed
    variances are found to be very high given the number of observations, then we reject our null hypothesis and
    conclude that this difference could not occur due to chance.

    Parameters
    ----------
    cont_table : list or numpy array, 2 x 2
        A contingency table containing 2 counts of 2, or 4 counts total. As an example of expected output, refer to a
        confusion matrix for predicting a binary variable.

    Returns
    -------
    X : float
        The Chi test statistic, or the variance of the difference of our observed results versus expected results.
    p : float, 0 <= p <= 1
        The likelihood that we would observe our X value given the number of observations we had.
    """
    cont_table = _check_table(cont_table, only_count=True)
    df = (cont_table.shape[0] - 1) * (cont_table.shape[1] - 1)
    row_sum, col_sum = np.sum(cont_table, axis=1), np.sum(cont_table, axis=0)
    expected = np.matmul(np.transpose(row_sum[np.newaxis]), col_sum[np.newaxis]) / np.sum(row_sum)
    X = np.sum(pow(cont_table - expected, 2) / expected)
    p = 1 - chi2.cdf(X, df)
    return X, p


def g_test(cont_table: Union[Sequence[Sequence], np.ndarray]) -> Tuple[float, float]:
    """Found in scipy.stats as chi2_contingency(lambda_="log-likelihood")

    A likelihood ratio test used for determine if the difference between our observed results and expected results in
    our contingency table are likely to happen due to chance.

    Parameters
    ----------
    cont_table : list or numpy array, 2 x 2
        A contingency table containing 2 counts of 2, or 4 counts total. As an example of expected output, refer to a
        confusion matrix for predicting a binary variable.

    Returns
    -------
    g : float
        The G statistic, or the likelihood ratio of the difference between observed and expected
    p : float, 0 <= p <= 1
        The likelihood that our observed differences are due to chance
    """
    cont_table = _check_table(cont_table, True)
    df = (cont_table.shape[0] - 1) * (cont_table.shape[1] - 1)
    row_sum, col_sum = np.sum(cont_table, axis=1), np.sum(cont_table, axis=0)
    expected = np.matmul(np.transpose(row_sum[np.newaxis]), col_sum[np.newaxis]) / np.sum(row_sum)
    g = 2 * np.sum(cont_table * np.log(cont_table / expected))
    p = 1 - chi2.cdf(g, df)
    return g, p


def fisher_test(cont_table: Union[Sequence[Sequence], np.ndarray], alternative: str = "two-sided") -> float:
    """Found in scipy.stats as fisher_exact

    Used to determine the exact likelihood that we would observe a measurement in our 2x2 contingency table that
    is just as extreme, if not moreso, than our observed results.

    Parameters
    ----------
    cont_table : list or numpy array, 2 x 2
        A 2x2 contingency table
    alternative : str, {two-sided, greater, less}, default=two-sided
        Our alternative hypothesis

    Returns
    -------
    p : float, 0 <= p <= 1
        The exact likelihood of finding a more extreme measurement than our observed data
    """
    if alternative.casefold() not in ["two-sided", "greater", "less"]:
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

    if alternative.casefold() == "greater":
        right_p_val = right_side(a, b, c, d)
        return p + np.sum(right_p_val)
    elif alternative.casefold() == "less":
        left_p_val = left_side(a, b, c, d)
        return p + np.sum(left_p_val)
    else:
        left_p_val, right_p_val = left_side(a, b, c, d), right_side(a, b, c, d)
        all_p = right_p_val + left_p_val
        return p + np.sum([i for i in all_p if i <= p])


def mcnemar_test(cont_table: Union[Sequence[Sequence], np.ndarray]) -> Tuple[float, float]:
    """Found in statsmodels as mcnemar

    Used when we have paired nominal data that is organized in a 2x2 contingency table. It is used to test the
    assumption that the marginal column and row probabilities are equal, i.e., that the probability that b and c
    are equivalent.

    Parameters
    ----------
    cont_table : list or numpy array, 2 x 2
        A 2x2 contingency table

    Returns
    -------
    chi_squared : float
        Our Chi statistic, or the sum of differences between b and c
    p : float, 0 <= p <= 1
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


def cmh_test(tables: Union[Sequence[Sequence], np.ndarray]) -> Tuple[float, float]:
    """Found in statsmodels as Stratified Table.test_null_odds()

    Used when we want to evaluate the association between a binary predictor/treatment and a binary outcome variable
    with data that is stratified or matched. Our null hypothesis is that the common-odds ratio is 1 while our
    alternative is that the common odds ratio is not equal to one (i.e., a two-tailed test). This test is an
    extension of the mcnemar test to handle any arbitrary strata size.

    Parameters
    ----------
    tables : list or numpy array, 2 x 2
        A group of 2x2 contingency tables, where each group represents a strata

    Returns
    -------
    epsilon : float
        Our test statistic, used to evaluate the likelihood that all strata have the same common odds ratio
    p : float, 0 <= p <= 1
        The likelihood that our common odds ratio would not equal one if we were to randomly sample strata from the same
        population
    """
    if len(tables) < 2:
        raise AttributeError("Cannot perform CMH Test on less than 2 groups")
    a, row_sum, col_sum, total, n_i, m_i = [], [], [], [], [], []
    for table in tables:
        table = _check_table(table)
        if table.shape != (2, 2):
            raise AttributeError("CMH Test is meant for 2x2 contingency tables")
        a = np.append(a, table[0, 0])
        n, m = np.sum(table, axis=1), np.sum(table, axis=0)
        row_sum = np.append(row_sum, n)
        col_sum = np.append(col_sum, m)
        total = np.append(total, np.sum(n))
        n_i, m_i = np.append(n_i, n[0]), np.append(m_i, m[0])
    top = pow(abs(np.sum(a - (n_i * m_i / total))), 2)
    bottom = np.sum((n_i * (total - n_i) * m_i * (total - m_i)) / (np.power(total, 2) * (total - 1)))
    epsilon = top / bottom
    p = 1 - chi2.cdf(epsilon, 1)
    return epsilon, p


def woolf_test(tables: Union[Sequence[Sequence], np.ndarray]) -> Tuple[float, float]:
    """Not found in either scipy or statsmodels

    Used to test the homogeneity of the odds ratio of each contingency table. Unlike Breslow-Day, compares
    the actual results to the expected Mantel-Haenzel odds ratio for each strata.

    Parameters
    ----------
    tables : list or numpy array, 2 x 2
        A group of 2x2 contingency tables, where each group represents a strata

    Returns
    -------
    epsilon : float
        Our test statistic, used to evaluate the likelihood that all strata have the same common odds ratio
    p : float, 0 <= p <= 1
        The likelihood that our common odds ratio would not be equivalent if we were to randomly sample strata from the
        same population
    """
    k = len(tables)
    if k < 2:
        raise AttributeError("Cannot perform Woolf Test on less than two groups")
    or_i, w_i = [], []
    for table in tables:
        table = _check_table(table, only_count=True)
        if table.shape != (2, 2):
            raise AttributeError("Woolf Test is meant for 2x2 contingency table")
        a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
        or_i = np.append(or_i, np.log(a * d / (b * c)))
        w_i = np.append(w_i, np.power((1 / a) + (1 / b) + (1 / c) + (1 / d), -1))
    or_bar = np.sum(w_i * or_i) / np.sum(w_i)
    x = np.sum(w_i * np.power(or_i - or_bar, 2))
    df = k - 1
    p = 1 - chi2.cdf(x, df)
    return x, p


def breslow_day_test(tables: Union[Sequence[Sequence], np.ndarray]) -> Tuple[float, float]:
    """Found in statsmodels as StratifiedTable.test_equal_odds()

    Computes the likelihood that the odds ratio for each strata is the same, by comparing the first
    row and column with its expected pooled ratio amount
    For solving the quadratic, set A / (n_i - A) / (m_i1 - A) / (mi2 - n_i + A) equal to the pooled ratio, and then solve
    for zero.
    (1) A * (m_i2 - n_i1 + A) = ratio * (n_i - A) * (m_i1 - A)
    (2) A * m_i2 - A * n_i2 + A^2 = ratio * n_i * m_i1  - ratio * n_i * A - ratio * m_i1 * A - ratio * A^2
    (3) A^2 - ratio * A^2 + A * m_i2 - A * n_i + A * ratio * n_i + A * ratio * m_i1 - ratio * n_i * m_i1 = 0
    (4) A^2(1 - ratio) + A(m_i2 - n_i + ratio * n_i + ratio * m_i1) - x * n_i * m_i1 = 0
    From there, you can solve for the quadratic

    Parameters
    ----------
    tables : list or numpy array, 2 x 2
        A group of 2x2 contingency tables, where each group represents a strata

    Returns
    -------
    x : float
        Our test statistic, used to evaluate the likelihood that all strata have the same common odds ratio
    p : float, 0 <= p <= 1
        The likelihood that our common odds ratio would not be equivalent if we were to randomly generate a from the
        pooled odds ratio
    """
    k = len(tables)
    if k < 2:
        raise AttributeError("Cannot perform Breslow-Day Test for less than 2 groups")
    a_i, bc, ad, m_i1, m_i2, n_i = [], [], [], [], [], []
    for table in tables:
        table = _check_table(table, only_count=True)
        if table.shape != (2, 2):
            raise AttributeError("Breslow-Day Test is meant for 2x2 contingency table")
        a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
        a_i = np.append(a_i, a)
        ad = np.append(ad, a * d)
        bc = np.append(bc, b * c)
        m_i1 = np.append(m_i1, np.sum(table, axis=1)[0])
        m_i2 = np.append(m_i2, np.sum(table, axis=1)[1])
        n_i = np.append(n_i, np.sum(table, axis=0)[0])
    odds = np.sum(ad / (m_i1 + m_i2)) / np.sum(bc / (m_i1 + m_i2))

    def solve_quadratic(a, b, c):
        return (-b + np.sqrt(np.power(b, 2) - 4 * a * c)) / (2 * a)

    A = solve_quadratic(1 - odds, (m_i2 - n_i + (odds * n_i) + (odds * m_i1)), -odds * n_i * m_i1)
    B, C, D = m_i1 - A, n_i - A, m_i2 - n_i + A
    var_i = np.power((1 / A) + (1 / B) + (1 / C) + (1 / D), -1)
    x = np.sum(np.power(a_i - A, 2) / var_i)
    p = 1 - chi2.cdf(x, k - 1)
    return x, p


def bowker_test(cont_table: Union[Sequence[Sequence], np.ndarray]) -> Tuple[float, float]:
    """Found in statsmodels as TableSymmetry or as bowker_symmetry

    Used to test if a given square table is symmetric about the main diagonal

    Parameters
    ----------
    cont_table : list or numpy array, n x n
        A nxn contingency table

    Returns
    -------
    x : float
        Our Chi statistic, oor a measure of symmetry for our contingency table
    p : float, 0 <= p <= 1
        The probability that our table isn't symmetric due to chance
    """
    cont_table = _check_table(cont_table, only_count=True)
    n1, n2 = np.shape(cont_table)
    if n1 != n2:
        raise AttributeError("Contingency Table needs to be of a square shape")
    upper_diagonal = np.triu_indices(n1, 1)
    # lower_diagonal = np.tril_indices(n1, -1) The issue with this code is that it doesn't maintain the exact order
    # of a lower triangular matrix compared to np.triu_indices, which we need for our test statistic
    upper_triangle = cont_table[upper_diagonal]
    lower_triangle = cont_table.T[upper_diagonal]
    x = np.sum(np.power(lower_triangle - upper_triangle, 2) / (upper_triangle + lower_triangle))
    df = n1 * (n1 - 1) / 2
    p = 1 - chi2.cdf(x, df)
    return x, p
