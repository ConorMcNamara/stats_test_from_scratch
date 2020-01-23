import numpy as np
from StatsTest.utils import _check_table, _sse
from scipy.stats import f, chi2, norm
from statsmodels.stats.libqsturng import psturng
from math import sqrt
from itertools import chain


def levene_test(*args):
    """Found in scipy.stats as levene(center='mean')
    Used to determine if a variable/observation in multiple groups has equal variances across all groups. In short, does
    each group have equal variance?

    Parameters
    ----------
    args: list or numpy arrays, 1-D
        The observed variable/observations for each group, organized into lists or numpy array

    Return
    ------
    w: float
        The W statistic, our measure of difference in variability, which is approximately F-distributed.
    p: float, 0 <= p <= 1
        The probability that our observed differences in variances could occur due to random sampling from a population
        of equal variance.
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Need at least two groups to perform a Levene Test")
    n_i, z_bar, all_z_ij, z_bar_condensed = [], [], [], []
    for obs in args:
        obs = _check_table(obs, False)
        n_i = np.append(n_i, len(obs))
        z_ij = abs(obs - np.mean(obs))
        all_z_ij = np.append(all_z_ij, z_ij)
        z_bar = np.append(z_bar, np.repeat(np.mean(z_ij), len(obs)))
        z_bar_condensed = np.append(z_bar_condensed, np.mean(z_ij))
    scalar = (np.sum(n_i) - k) / (k - 1)
    w = scalar * np.sum(n_i * np.power(z_bar_condensed - np.mean(z_bar), 2)) / np.sum(np.power(all_z_ij - z_bar, 2))
    p = 1 - f.cdf(w, k - 1, np.sum(n_i) - k)
    return w, p


def brown_forsythe_test(*args):
    """Found in scipy.stats as levene(center='median')
    Used instead of general levene test if we believe our data to be non-normal.

    Parameters
    ----------
    args: list or numpy arrays, 1-D
        The observed variable/observations for each group, organized into lists or numpy array

    Return
    ------
    w: float
        The W statistic, our measure of difference in variability, which is approximately F-distributed.
    p: float, 0 <= p <= 1
        The probability that our observed differences in variances could occur due to random sampling from a population
        of equal variance.
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Need at least two groups to perform a Brown-Forsythe Test")
    n_i, z_bar, all_z_ij, z_bar_condensed = [], [], [], []
    for obs in args:
        obs = _check_table(obs, False)
        n_i = np.append(n_i, len(obs))
        z_ij = abs(obs - np.median(obs))
        all_z_ij = np.append(all_z_ij, z_ij)
        z_bar = np.append(z_bar, np.repeat(np.mean(z_ij), len(obs)))
        z_bar_condensed = np.append(z_bar_condensed, np.mean(z_ij))
    scalar = (np.sum(n_i) - k) / (k - 1)
    w = scalar * np.sum(n_i * np.power(z_bar_condensed - np.mean(z_bar), 2)) / np.sum(np.power(all_z_ij - z_bar, 2))
    p = 1 - f.cdf(w, k - 1, np.sum(n_i) - k)
    return w, p


def one_way_f_test(*args):
    """Found in scipy.stats as f_oneway
    Used to measure if multiple normal populations have the same mean. Note that this test is very sensitive to
    non-normal data, meaning that it should not be used unless we can verify that the data is normally distributed.

    Parameters
    ----------
    args: list or numpy arrays, 1-D
        The observed measurements for each group, organized into lists or numpy array

    Return
    ------
    f_statistics: float
        The F statistic, or a measure of the ratio of data explained by the mean versus that unexplained by the mean
    p: float, 0 <= p <= 1
        The likelihood that our observed ratio would occur, in a population with the same mean, due to chance
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Need at least two groups to perform a one-way F Test")
    n_i, y_bar, all_y_ij, y_bar_condensed = [], [], [], []
    all_y_ij = np.hstack(args)
    for obs in args:
        obs = _check_table(obs, False)
        n_i = np.append(n_i, len(obs))
        obs_mean = np.mean(obs)
        y_bar_condensed = np.append(y_bar_condensed, obs_mean)
        y_bar = np.append(y_bar, np.repeat(obs_mean, len(obs)))
    explained_variance = np.sum(n_i * np.power(y_bar_condensed - np.mean(all_y_ij), 2) / (k - 1))
    unexplained_variance = np.sum(np.power(all_y_ij - y_bar, 2) / (np.sum(n_i) - k))
    f_statistic = explained_variance / unexplained_variance
    p = 1 - f.cdf(f_statistic, k - 1, np.sum(n_i) - k)
    return f_statistic, p


def bartlett_test(*args):
    """Found in scipy.stats as bartlett
    This test is used to determine if multiple samples are from a population of equal variances. Note that this test
    is much more sensitive to data that is non-normal compared to Levene or Brown-Forsythe.

    Parameters
    ----------
    args: list or numpy arrays, 1-D
        The observed measurements for each group, organized into lists or numpy array

    Return
    ------
    X: float
        The Chi statistic, or a measure of the observed difference in variances
    p: float, 0 <= p <= 1
        The probability that our observed differences in variances could occur due to random sampling from a population
        of equal variance.
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Need at least two groups to perform the Bartlett Test")
    n_i, var_i = [], []
    for obs in args:
        obs = _check_table(obs)
        n_i = np.append(n_i, len(obs))
        var_i = np.append(var_i, np.var(obs, ddof=1))
    pooled_variance = np.sum((n_i - 1) * var_i) / (np.sum(n_i) - k)
    top = (np.sum(n_i) - k) * np.log(pooled_variance) - np.sum((n_i - 1) * np.log(var_i))
    bottom = 1 + (1 / (3 * (k - 1))) * (np.sum(1 / (n_i - 1)) - (1 / (np.sum(n_i) - k)))
    X = top / bottom
    p = 1 - chi2.cdf(X, k - 1)
    return X, p


def tukey_range_test(*args):
    """Found in statsmodels as pairwise_tukeyhsd
    This test compares all possible pairs of means and determines if there are any differences in these pairs.

    Parameters
    ----------
    args: list or numpy arrays, 1-D
        The observed measurements for each group, organized into lists or numpy arrays

    Return
    ------
    results: list
        A list of lists containing 3 attributes:
            1) The groups being compared
            2) The Q Statistic
            3) p, or the likelihood our observed differences are due to chance
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Need at least two groups to perform Tukey Range Test")
    mean_i, groups, n_i, sum_data, square_data, results = [], [], [], [], [], []
    i = 0
    for obs in args:
        obs = _check_table(obs, False)
        mean_i = np.append(mean_i, np.mean(obs))
        groups = np.append(groups, i)
        n_i = np.append(n_i, len(obs))
        sum_data = np.append(sum_data, np.sum(obs))
        square_data = np.append(square_data, np.power(obs, 2))
        i += 1
    df = sum(n_i) - k
    sse = _sse(sum_data, square_data, n_i)
    for group in np.unique(groups):
        group = int(group)
        for next_group in range(group + 1, len(np.unique(groups))):
            mean_a, mean_b = mean_i[group], mean_i[next_group]
            n_a, n_b = n_i[group], n_i[next_group]
            difference = abs(mean_a - mean_b)
            std_group = sqrt(sse / df / min(n_a, n_b))
            q = difference / std_group
            p = psturng(q, k, df)
            results.append(['group {} - group {}'.format(group, next_group), q, p])
    return results


def cochran_q_test(*args):
    """Found in statsmodels as chochrans_q
    Used to determine if k treatments in a 2 way randomized block design have identical effects. Note that this test
    requires that there be only two variables encoded: 1 for success and 0 for failure.

    Parameters
    ----------
    args: list or numpy arrays, 1-D
        Each array corresponds to all observations from a single treatment. That is, each array corresponds to a
        column in our table (Treatment_k), if we were to look at https://en.wikipedia.org/wiki/Cochran%27s_Q_test

    Return
    ------
    T: float
        Our T statistic
    p: float, 0 <= p <= 1
        The likelihood that our observed differences are due to chance
    """
    k = len(args)
    if k < 3:
        raise AttributeError("Cannot run Cochran's Q Test with less than 3 treatments")
    if len(np.unique(args)) > 2:
        raise AttributeError("Cochran's Q Test only works with binary variables")
    df = k - 1
    N = np.sum(args)
    all_data = np.vstack(args).T
    row_sum, col_sum = np.sum(all_data, axis=1), np.sum(all_data, axis=0)
    scalar = k * (k - 1)
    T = scalar * np.sum(np.power(col_sum - (N / k), 2)) / np.sum(row_sum * (k - row_sum))
    p = 1 - chi2.cdf(T, df)
    return T, p


def jonckheere_trend_test(*args, **kwargs):
    """This test is not found in scipy or statsmodels
    This test is used to determine if the population medians for each groups have an a priori ordering.
    Note that the default alternative hypothesis is that median_1 <= median_2 <= median_3 <= ... <= median_k, with at
    least one strict inequality.

    Parameters
    ----------
    args: list or numpy array, 1-D
        List or numpy arrays, where each array constitutes a population/group, and within that group are their responses.
        For example, based on the numeric example found here: https://en.wikipedia.org/wiki/Jonckheere%27s_trend_test,
        the first array would be the measurements found in "contacted" and the second array would the measurements found
        in "bumped" and the third array would be the measurements found in "smashed"
    kwargs: str
        Our alternative hypothesis. The two options are "greater" and "less", indicating the a priori ordering. Default
        is less

    Return
    ------
    z_statistic: float
        A measure of the difference in trends for the median of each group
    p: float, 0 <= p <= 1
        The likelihood that our trend could be found if each group were randomly sampled from a population with the same
        medians.
    """
    k = len(args)
    if k < 2:
        raise AttributeError("Cannot run Jonckheere Test with less than 2 groups")
    u = [len(arg) for arg in args]
    if len(np.unique(u)) != 1:
        raise AttributeError("Jonckheere Test requires that each group have the same number of observations")
    if "alternative" in kwargs:
        alternative = kwargs.get("alternative")
        if not isinstance(alternative, str):
            raise TypeError("Cannot have alternative hypothesis with non-string value")
        if alternative.casefold() not in ['greater', 'less']:
            raise ValueError("Cannot discern alternative hypothesis")
    else:
        alternative = "less"
    all_data = np.vstack([sorted(arg) for arg in args]).T
    if alternative.casefold() == "greater":
        all_data = np.flip(all_data, axis=1)
    t = np.unique(all_data, return_counts=True)[1]
    n = all_data.shape[0] * k
    p, q = 0, 0
    for col in range(k - 1):
        for row in range(all_data.shape[0]):
            val = all_data[row, col]
            right_side = [False] * (col + 1) + [True] * (k - col - 1)
            right_data = np.compress(right_side, all_data, axis=1)
            p += len(np.where(right_data > val)[0])
            q += len(np.where(right_data < val)[0])
    s = p - q
    sum_t_2, sum_t_3 = np.sum(np.power(t, 2)), np.sum(np.power(t, 3))
    sum_u_2, sum_u_3 = np.sum(np.power(u, 2)), np.sum(np.power(u, 3))
    part_one = (2 * (pow(n, 3) - sum_t_3 - sum_u_3) + 3 * (pow(n, 2) - sum_t_2 - sum_u_2) + 5 * n) / 18
    part_two = (sum_t_3 - 3 * sum_t_2 + 2 * n) * (sum_u_3 - 3 * sum_u_2 + 2 * n) / (9 * n * (n - 1) * (n - 2))
    part_three = (sum_t_2 - n) * (sum_u_2 - n) / (2 * n * (n - 1))
    var_s = part_one + part_two + part_three
    z_statistic = s / sqrt(var_s)
    p = 1 - norm.cdf(z_statistic)
    return z_statistic, p


def mood_median_test(*args, **kwargs):
    """Found in scipy.stats as median_test
    This test is used to determine if two or more samples/observations come from a population with the same median.

    Parameters
    ----------
    args: list or numpy arrays, 1-D
       List or numpy arrays, where each array constitutes a number of observations in a population/group.
    kwargs: str
        "alternative": Our alternative hypothesis. The three options are "greater", "less" or "two-sided', used to determine
        whether or not we expect our data to favor being greater, less than or different from the median.
        Default is two-sided.

        "handle_med": How we handle the median value. The three options are "greater", "less' or "ignore". If greater,
        median value is added to values above median. If less, median value is added to values below median. If ignore,
        median value is not added at all. Default is "less".

    Returns
    -------
    X: float
        Our Chi Statistic measuring the difference of our groups compared to the median
    p: float, 0 <= p <= 1
        The likelihood that our observed differences in medians are due to chance
    """
    if len(args) < 2:
        raise AttributeError("Cannot run Median Test with less than 2 groups")
    all_data = np.concatenate(args)
    med = np.median(all_data)
    if "alternative" in kwargs:
        alternative = kwargs.get("alternative").casefold()
        if alternative not in ['greater', 'less', 'two-sided']:
            raise ValueError("Cannot discern alternative hypothesis")
    else:
        alternative = "two-sided"
    if "handle_med" in kwargs:
        handle_med = kwargs.get("handle_med").casefold()
        if handle_med not in ['greater', 'less', 'ignore']:
            raise ValueError("Cannot discern how to handle median value")
    else:
        handle_med = "less"
    above_med, below_med = [], []
    # To-do: see if I can simplify this logic by using vectorized functions and eliminate the for-loop
    if handle_med == "less":
        for arg in args:
            arg = _check_table(arg, only_count=False)
            above_med.append(np.sum(arg > med))
            below_med.append(np.sum(arg <= med))
    elif handle_med == "greater":
        for arg in args:
            arg = _check_table(arg, only_count=False)
            above_med.append(np.sum(arg >= med))
            below_med.append(np.sum(arg < med))
    else:
        for arg in args:
            arg = _check_table(arg, only_count=False)
            above_med.append(np.sum(arg > med))
            below_med.append(np.sum(arg < med))
    cont_table = np.vstack([above_med, below_med])
    row_sum, col_sum = np.sum(cont_table, axis=1), np.sum(cont_table, axis=0)
    expected = np.matmul(np.transpose(row_sum[np.newaxis]), col_sum[np.newaxis]) / np.sum(row_sum)
    X = np.sum(pow(cont_table - expected, 2) / expected)
    df = len(args) - 1
    if alternative == "two-sided":
        p = 2 * (1 - chi2.cdf(X, df))
    elif alternative == "less":
        p = 1 - chi2.cdf(X, df)
    else:
        p = chi2.cdf(X, df)
    return X, p


def dunnett_test(control, alpha=0.05, *args):
    """Not found in either scipy or statsmodels
    This test is used to compare the means of several groups to a control and determine which groups are significant

    Parameters
    ----------
    control: list or numpy array
        A list or array containing data pertaining to our control group
    alpha: float, {0.01, 0.05, 0.10}, default is 0.05
        Our alpha level for determining level of significant difference
    args: list or numpy arrays
        List or numpy arrays, where each array corresponds to data for a treatment group

    Returns
    -------
    A list of booleans, with each bool corresponding to whether the treatment group is a significant pair with the control group
    """
    k = len(args) + 1
    if k <= 2:
        raise AttributeError("Cannot run Dunnett Test with less than two groups")
    if alpha not in [0.01, 0.05, 0.10]:
        raise ValueError("Alpha level not currently supported")
    n = len(control)
    df = k * (n - 1)
    len_data = np.append(n, [len(arg) for arg in args])
    mean_data = np.append(np.mean(control), np.mean(args, axis=1))
    all_data = np.append(control, args)
    grand_mean = np.mean(all_data)
    ssb = np.sum(len_data * np.power(mean_data - grand_mean, 2))
    sst = np.sum(np.power(all_data - grand_mean, 2))
    msw = (sst - ssb) / df
    se = sqrt(msw * 2 / n)
    if df <= 20:
        index = df - 1
    elif 20 < df <= 30:
        index = 20
    elif 30 < df <= 40:
        index = 21
    elif 40 < df <= 60:
        index = 22
    elif 60 < df <= 80:
        index = 23
    elif 80 < df <= 120:
        index = 24
    else:
        index = 25
    col = k - 2 if k < 20 else 19
    q_01 = [[63.657, 86.959, 100.280, 109.360, 116.153, 121.535, 129.720, 132.966, 135.358, 138.358, 1440.6454, 142.721, 1444.621, 146.371, 147.991, 149.499, 150.908, 152.231, 153.76],
            [9.925, 12.388, 13.826, 14.825, 15.582, 16.189, 16.692, 17.121, 17.494, 17.823, 18.117, 18.383, 18.624, 18.846, 19.051, 19.241, 19.418, 19.583, 19.739, 19.886],
            [5.841, 6.974, 7.639, 8.104, 8.460, 8.746, 8.985, 9.189, 9.367, 9.525, 9.666, 9.794, 9.910, 10.017, 10.116, 10.208, 10.294, 10.375, 10.450, 10.522],
            [4.604, 5.364, 5.809, 6.121, 6.361, 6.554, 6.716, 6.854, 6.975, 7.082, 7.179, 7.266, 7.346, 7.419, 7.487, 7.550, 7.609, 7.664, 7.716, 7.765],
            [4.032, 4.627, 4.975, 5.219, 5.406, 5.557, 5.683, 5.792, 5.887, 5.971, 6.047, 6.116, 6.178, 6.236, 6.289, 6.339, 6.286, 6.429, 6.470, 6.509],
            [3.707, 4.212, 4.506, 4.711, 4.869, 4.997, 5.104, 5.196, 5.276, 5.347, 5.411, 5.469, 5.523, 5.572, 5.617, 5.659, 5.699, 5.736, 5.770, 5.803],
            [3.499, 3.948, 4.208, 4.389, 4.529, 4.642, 4.736, 4.817, 4.888, 4.951, 5.008, 5.059, 5.106, 5.150, 5.190, 5.227, 5.262, 5.295, 5.236, 5.355],
            [3.355, 3.766, 4.002, 4.168, 4.295, 4.397, 4.483, 4.557, 4.621, 4.679, 4.730, 4.777, 4.820, 4.859, 4.896, 4.930, 4.961, 4.991, 5.019, 5.046],
            [3.250, 3.633, 3.853, 4.006, 4.124, 4.219, 4.299, 4.367, 4.427, 4.480, 4.528, 4.571, 4.611, 4.647, 4.681, 4.713, 4.742, 4.770, 4.796, 4.821],
            [3.169, 3.531, 3.739, 3.883, 3.994, 4.084, 4.159, 4.223, 4.279, 4.329, 4.374, 4.415, 4.452, 4.487, 4.519, 4.548, 4.576, 4.602, 4.627, 4.650],
            [3.106, 3.452, 3.649, 3.787, 3.892, 3.978, 4.049, 4.110, 4.164, 4.211, 4.254, 4.293, 4.328, 4.361, 4.391, 4.419, 4.446, 4.470, 4.494, 4.516],
            [3.055, 3.387, 3.577, 3.709, 3.811, 3.892, 3.960, 4.019, 4.070, 4.116, 4.157, 4.194, 4.228, 4.259, 4.288, 4.315, 4.340, 4.364, 4.387, 4.408],
            [3.012, 3.335, 3.518, 3.646, 3.743, 3.822, 3.888, 3.944, 3.994, 4.038, 4.077, 4.113, 4.146, 4.176, 4.204, 4.230, 4.254, 4.277, 4.299, 4.319],
            [2.977, 3.290, 3.468, 3.592, 3.687, 3.763, 3.827, 3.882, 3.930, 3.972, 4.011, 4.045, 4.077, 4.106, 4.133, 4.158, 4.182, 4.204, 4.225, 4.245],
            [2.947, 3.253, 3.426, 3.547, 3.639, 3.713, 3.776, 3.829, 3.875, 3.917, 3.954, 3.988, 4.019, 4.047, 4.073, 4.098, 4.121, 4.142, 4.163, 4.182],
            [2.921, 3.220, 3.390, 3.508, 3.598, 3.671, 3.731, 3.783, 3.829, 3.869, 3.905, 3.938, 3.968, 3.996, 4.022, 4.046, 4.068, 4.089, 4.109, 4.128],
            [2.898, 3.192, 3.359, 3.474, 3.563, 3.634, 3.693, 3.744, 3.788, 3.828, 3.863, 3.896, 3.925, 3.952, 3.977, 4.001, 4.023, 4.043, 4.062, 4.081],
            [2.878, 3.168, 3.331, 3.445, 3.531, 3.601, 3.659, 3.709, 3.753, 3.792, 3.826, 3.858, 3.887, 3.914, 3.938, 3.961, 3.983, 4.003, 4.022, 4.040],
            [2.861, 3.146, 3.307, 3.419, 3.504, 3.572, 3.360, 3.679, 3.722, 3.760, 3.794, 3.825, 3.853, 3.879, 3.904, 3.926, 3.947, 3.967, 3.986, 4.003],
            [2.845, 3.127, 3.285, 3.395, 3.479, 3.547, 3.603, 3.651, 3.694, 3.731, 3.765, 3.795, 3.823, 3.849, 3.873, 3.895, 3.916, 3.935, 3.954, 3.971],
            [2.750, 3.009, 3.154, 3.254, 3.330, 3.391, 3.442, 3.486, 4.524, 4.558, 3.589, 3.616, 3.641, 3.665, 3.686, 3.706, 3.725, 3.743, 3.759, 3.775],
            [2.704, 2.952, 3.091, 3.186, 3.259, 3.317, 3.366, 3.408, 3.444, 3.476, 3.505, 3.531, 3.555, 3.577, 3.598, 3.617, 3.634, 3.651, 3.667, 3.682],
            [2.660, 2.898, 3.030, 3.121, 3.190, 3.246, 3.292, 3.332, 3.366, 3.397, 3.424, 3.449, 3.472, 3.493, 3.512, 3.530, 3.547, 3.563, 3.578, 3.592],
            [2.617, 2.845, 2.972, 3.059, 3.124, 3.177, 3.221, 3.259, 3.291, 3.320, 3.346, 3.370, 3.392, 3.411, 3.430, 3.447, 3.463, 3.478, 3.492, 3.505],
            [2.576, 2.794, 2.915, 2.998, 3.060, 3.110, 3.152, 3.188, 3.219, 3.246, 3.271, 3.293, 3.314, 3.333, 3.350, 3.366, 3.381, 3.395, 3.409, 3.421]]

    q_05 = [[12.706, 17.369, 21.850, 23.209, 24.285, 25.171, 25.922, 26.570, 27.141, 27.649, 28.106, 28.521, 28.901, 29.251, 29.575, 29.876, 30.158, 30.422, 30.671],
            [4.303, 5.418, 6.065, 6.513, 6.852, 7.124, 7.349, 7.540, 7.707, 7.853, 7.895, 8.103, 8.211, 8.310, 8.401, 8.485, 8.564, 8.638, 8.707, 8.773],
            [3.182, 3.867, 4.263, 4.538, 4.748, 4.916, 5.056, 5.176, 5.280, 5.372, 5.455, 5.529, 5.597, 5.660, 5.717, 5.771, 5.821, 5.868, 5.912, 5.953],
            [2.776, 3.310, 3.618, 3.832, 3.994, 4.125, 4.235, 4.328, 4.410, 4.482, 4.546, 4.605, 4.658, 4.707, 4.752, 4.794, 4.834, 4.870, 4.905, 4.938],
            [2.571, 3.030, 3.293, 3.476, 3.615, 3.727, 3.821, 3.900, 3.970, 4.032, 4.087, 4.137, 4.183, 4.225, 4.264, 4.300, 4.334, 4.366, 4.395, 4.424],
            [2.447, 2.863, 3.099, 3.263, 3.388, 3.489, 3.573, 3.644, 3.707, 3.763, 3.812, 3.857, 3.898, 3.936, 3.971, 4.004, 4.034, 4.063, 4.089, 4.115],
            [2.365, 2.752, 2.971, 3.123, 3.238, 3.331, 3.408, 3.475, 3.533, 3.584, 3.630, 3.671, 3.709, 3.744, 3.777, 3.807, 3.835, 3.861, 3.886, 3.909],
            [2.306, 2.673, 2.880, 3.023, 3.131, 3.219, 3.292, 3.354, 3.408, 3.457, 3.500, 3.539, 3.575, 3.608, 3.638, 3.666, 3.693, 3.718, 3.741, 3.783],
            [2.262, 2.614, 2.812, 2.948, 3.052, 3.135, 3.205, 3.264, 3.316, 3.362, 3.403, 3.440, 3.474, 3.506, 3.535, 3.562, 3.587, 3.610, 3.633, 3.654],
            [2.228, 2.568, 2.759, 2.890, 2.990, 3.070, 3.137, 3.194, 3.244, 3.288, 3.328, 3.364, 3.396, 3.427, 3.454, 3.480, 3.504, 3.527, 3.549, 3.569],
            [2.201, 2.532, 2.717, 2.845, 2.941, 3.019, 3.084, 3.139, 3.187, 3.230, 3.268, 3.303, 3.334, 3.363, 3.390, 3.415, 3.439, 3.461, 3.481, 3.501],
            [2.179, 2.502, 2.683, 2.807, 2.901, 2.977, 3.040, 3.094, 3.140, 3.182, 3.219, 3.253, 3.284, 3.312, 3.338, 3.363, 3.385, 3.407, 3.427, 3.446],
            [2.160, 2.478, 2.654, 2.776, 2.868, 2.942, 3.003, 3.056, 3.102, 3.142, 3.179, 3.212, 3.242, 3.269, 3.295, 3.319, 3.341, 3.362, 3.381, 3.400],
            [2.145, 2.457, 2.631, 2.750, 2.840, 2.912, 2.973, 3.024, 3.069, 3.109, 3.144, 3.177, 3.206, 3.233, 3.258, 3.282, 3.303, 3.324, 3.343, 3.361],
            [2.131, 2.439, 2.610, 2.727, 2.816, 2.887, 2.946, 2.997, 3.041, 3.080, 3.115, 3.147, 3.176, 3.202, 3.227, 3.250, 3.271, 3.291, 3.310, 3.328],
            [2.120, 2.424, 2.592, 2.708, 2.795, 2.865, 2.924, 2.974, 3.017, 3.056, 3.090, 3.121, 3.150, 3.176, 3.200, 3.222, 3.243, 3.263, 3.282, 3.299],
            [2.110, 2.410, 2.577, 2.691, 2.777, 2.846, 2.904, 2.953, 2.996, 3.034, 3.068, 3.099, 3.127, 3.152, 3.176, 3.199, 3.219, 3.329, 3.257, 3.274],
            [2.101, 2.399, 2.563, 2.676, 2.761, 2.830, 2.887, 2.935, 2.977, 3.015, 3.048, 3.079, 3.107, 3.132, 3.156, 3.177, 3.198, 3.217, 3.235, 3.252],
            [2.093, 2.388, 2.551, 2.663, 2.747, 2.815, 2.871, 2.919, 2.961, 2.998, 3.031, 3.061, 3.089, 3.114, 3.137, 3.159, 3.179, 3.198, 3.216, 3.233],
            [2.086, 2.379, 2.540, 2.651, 2.735, 2.802, 2.857, 2.905, 2.946, 2.983, 3.016, 3.045, 3.073, 3.098, 3.121, 3.142, 3.162, 3.181, 3.198, 3.215],
            [2.042, 2.321, 2.474, 2.578, 2.657, 2.720, 2.772, 2.817, 2.856, 2.890, 2.921, 2.949, 2.974, 2.997, 3.019, 3.039, 3.058, 3.075, 3.092, 3.107],
            [2.021, 2.293, 2.441, 2.543, 2.619, 2.680, 2.731, 2.774, 2.812, 2.845, 2.875, 2.902, 2.926, 2.949, 2.970, 2.989, 3.007, 3.024, 3.040, 3.055],
            [2.000, 2.265, 2.410, 2.508, 2.582, 2.642, 2.691, 2.733, 2.769, 2.801, 2.830, 2.856, 2.880, 2.901, 2.922, 2.940, 2.958, 2.974, 2.989, 3.004],
            [1.990, 2.252, 2.394, 2.491, 2.564, 2.623, 2.671, 2.712, 2.748, 2.780, 2.808, 2.833, 2.857, 2.878, 2.898, 2.916, 2.933, 2.950, 2.965, 2.979],
            [1.980, 2.238, 2.379, 2.475, 2.547, 2.604, 2.651, 2.692, 2.727, 2.758, 2.786, 2.811, 2.834, 2.855, 2.875, 2.893, 2.910, 2.925, 2.940, 2.954],
            [1.960, 2.212, 2.349, 2.442, 2.511, 2.567, 2.613, 2.652, 2.686, 2.716, 2.743, 2.767, 2.790, 2.810, 2.829, 2.846, 2.862, 2.878, 2.892, 2.905]]

    q_10 = [[6.314, 8.650, 9.983, 10.891, 11.570, 12.108, 12.551, 12.926, 13.250, 13.535, 13.788, 14.017, 14.224, 14.414, 14.589, 14.750, 14.901, 15.042, 15.174, 15.298],
            [2.920, 3.721, 4.182, 4.500, 4.740, 4.932, 5.091, 5.226, 5.344, 5.447, 5.540, 5.623, 5.699, 5.768, 5.833, 5.892, 5.948, 6.000, 6.048, 6.094],
            [2.353, 2.912, 3.232, 3.453, 3.621, 3.755, 3.866, 3.961, 4.044, 4.117, 4.182, 4.242, 4.295, 4.345, 4.390, 4.432, 4.472, 4.509, 4.543, 4.576],
            [2.132, 2.598, 2.863, 3.046, 3.185, 3.296, 3.389, 3.468, 3.536, 3.597, 3.652, 3.701, 3.746, 3.787, 3.825, 3.860, 3.893, 3.924, 3.953, 3.980],
            [2.015, 2.433, 2.669, 2.832, 2.956, 3.055, 3.137, 3.207, 3.268, 3.322, 3.371, 3.415, 3.455, 3.491, 3.525, 3.557, 3.586, 3.614, 3.640, 3.664],
            [1.943, 2.332, 2.551, 2.701, 2.815, 2.906, 2.982, 3.047, 3.103, 3.153, 3.198, 3.238, 3.275, 3.309, 3.340, 3.369, 3.396, 3.422, 3.446, 3.468],
            [1.895, 2.264, 2.470, 2.612, 2.720, 2.806, 2.877, 2.938, 2.991, 3.038, 3.080, 3.119, 3.153, 3.185, 3.215, 3.242, 3.268, 3.292, 3.314, 3.336],
            [1.860, 2.215, 2.413, 2.548, 2.651, 2.733, 2.802, 2.860, 2.911, 2.956, 2.996, 3.032, 3.065, 3.096, 3.124, 3.150, 3.175, 3.198, 3.219, 3.239],
            [1.833, 2.178, 2.369, 2.500, 2.599, 2.679, 2.745, 2.801, 2.850, 2.893, 2.932, 2.967, 2.999, 3.028, 3.055, 3.081, 3.104, 3.126, 3.147, 3.167],
            [1.812, 2.149, 2.335, 2.463, 2.559, 2.636, 2.700, 2.755, 2.802, 2.844, 2.882, 2.916, 2.947, 2.975, 3.002, 3.026, 3.049, 3.070, 3.091, 3.110],
            [1.796, 2.216, 2.308, 2.433, 2.527, 2.602, 2.664, 2.718, 2.764, 2.805, 2.842, 2.875, 2.905, 2.933, 2.959, 2.982, 3.005, 3.026, 3.045, 3.064],
            [1.782, 2.107, 2.286, 2.408, 2.500, 2.574, 2.635, 2.687, 2.733, 2.773, 2.809, 2.841, 2.871, 2.898, 2.923, 2.947, 2.968, 2.989, 3.008, 3.026],
            [1.771, 2.091, 2.267, 2.387, 2.478, 2.550, 2.611, 2.662, 2.706, 2.746, 2.781, 2.813, 2.842, 2.869, 2.894, 2.917, 2.938, 2.958, 2.977, 2.995],
            [1.761, 2.078, 2.252, 2.370, 2.459, 2.531, 2.590, 2.640, 2.684, 2.723, 2.758, 2.789, 2.818, 2.844, 2.868, 2.891, 2.912, 2.932, 2.951, 2.968],
            [1.753, 2.066, 2.238, 2.355, 2.443, 2.514, 2.572, 2.622, 2.665, 2.703, 2.738, 2.769, 2.797, 2.823, 2.847, 2.869, 2.890, 2.910, 2.928, 2.945],
            [1.746, 2.056, 2.226, 2.342, 2.429, 2.499, 2.557, 2.606, 2.649, 2.687, 2.720, 2.751, 2.779, 2.805, 2.828, 2.850, 2.871, 2.890, 2.908, 2.925],
            [1.740, 2.048, 2.216, 2.331, 2.417, 2.486, 2.543, 2.592, 2.634, 2.672, 2.705, 2.735, 2.763, 2.788, 2.812, 2.834, 2.854, 2.873, 2.891, 2.908],
            [1.734, 2.040, 2.207, 2.321, 2.406, 2.475, 2.531, 2.579, 2.621, 2.659, 2.692, 2.722, 2.749, 2.774, 2.797, 2.819, 2.839, 2.858, 2.876, 2.892],
            [1.729, 2.033, 2.199, 2.312, 2.397, 2.464, 2.521, 2.568, 2.610, 2.647, 2.680, 2.710, 2.737, 2.762, 2.785, 2.806, 2.826, 2.845, 2.862, 2.879],
            [1.725, 2.027, 2.192, 2.304, 2.388, 2.455, 2.511, 2.559, 2.600, 2.636, 2.669, 2.699, 2.726, 2.750, 2.773, 2.794, 2.814, 2.833, 2.850, 2.866],
            [1.697, 1.989, 2.147, 2.254, 2.335, 2.399, 2.452, 2.497, 2.537, 2.572, 2.603, 2.631, 2.656, 2.680, 2.701, 2.722, 2.740, 2.758, 2.775, 2.790],
            [1.684, 1.970, 2.125, 2.230, 2.309, 2.372, 2.424, 2.468, 2.506, 2.540, 2.570, 2.598, 2.623, 2.645, 2.667, 2.686, 2.704, 2.722, 2.738, 2.753],
            [1.671, 1.952, 2.104, 2.206, 2.283, 2.345, 2.395, 2.438, 2.476, 2.509, 2.538, 2.565, 2.589, 2.612, 2.632, 2.651, 2.669, 2.686, 2.701, 2.716],
            [1.664, 1.943, 2.093, 2.195, 2.271, 2.331, 2.381, 2.424, 2.461, 2.494, 2.523, 2.549, 2.573, 2.595, 2.615, 2.634, 2.652, 2.668, 2.864, 2.698],
            [1.658, 1.934, 2.083, 2.183, 2.258, 2.318, 2.368, 2.410, 2.446, 2.478, 2.507, 2.533, 2.557, 2.578, 2.598, 2.617, 2.634, 2.651, 2.666, 2.680],
            [1.645, 1.916, 2.062, 2.160, 2.234, 2.292, 2.340, 2.381, 2.417, 2.448, 2.476, 2.502, 2.525, 2.546, 2.565, 2.583, 2.600, 2.616, 2.631, 2.645]]
    if alpha == 0.01:
        t = np.array(q_01)[index][col]
    elif alpha == 0.05:
        t = np.array(q_05)[index][col]
    elif alpha == 0.1:
        t = np.array(q_10)[index][col]
    a = t * se
    group_diffs = np.abs(np.mean(args, axis=1) - np.mean(control)) > a
    return group_diffs


def duncan_multiple_range_test(alpha=0.05, *args):
    """Not found in either scipy or statsmodels
    This test is used to compare the means of several groups and determine which groups are significant

    Parameters
    ----------
    alpha: float, {0.01, 0.05, 0.10}, default is 0.05
        Our alpha level for determining level of significant difference
    args: list or numpy arrays
        List or numpy arrays, where each array corresponds to data for a treatment group

    Returns
    -------
    A list of tuples, with each tuple corresponding to a group that was found to be significantly different in means to
    the other.
    """
    k = len(args)
    if k <= 1:
        raise AttributeError("Cannot run Duncan Multi-Range Test with less than two groups")
    if alpha not in [0.01, 0.05, 0.10]:
        raise ValueError("Alpha level not currently supported")
    len_data = [len(arg) for arg in args]
    n = np.max(len_data)
    df = k * (n - 1)
    means = np.mean(args, axis=1)
    rank = np.argsort(means)
    big_to_small = means[rank[::-1]]
    small_to_big = means[rank]
    if df <= 20:
        index = df - 1
    elif 20 < df <= 30:
        index = 20
    elif 30 < df <= 40:
        index = 21
    elif 40 < df <= 60:
        index = 22
    elif 60 < df <= 120:
        index = 23
    else:
        index = 24
    if k <= 20:
        col = k - 2
    elif 20 < k <= 40:
        col = (k // 2) + 8
    else:
        col = (k // 10) + 24
    q_10 = [np.repeat(8.929, 35), np.repeat(4.130, 35), np.repeat(3.328, 35), np.repeat(3.015, 35),
            np.append([2.850, 2.934, 2.964], np.repeat(2.970, 32)),
            np.append([2.748, 2.846, 2.890, 2.908], np.repeat(2.911, 31)),
            np.append([2.680, 2.785, 2.838, 2.864, 2.876], np.repeat(2.878, 30)),
            np.append([2.630, 2.742, 2.800, 2.832, 2.849, 2.857], np.repeat(2.858, 29)),
            np.append([2.592, 2.708, 2.771, 2.808, 2.829, 2.840, 2.845], np.repeat(2.847, 28)),
            np.append([2.563, 2.682, 2.748, 2.788, 2.813, 2.827, 2.835], np.repeat(2.839, 28)),
            np.append([2.540, 2.660, 2.730, 2.772, 2.799, 2.817, 2.827, 2.833], np.repeat(2.835, 27)),
            np.append([2.521, 2.643, 2.714, 2.759, 2.789, 2.808, 2.821, 2.828, 2.832], np.repeat(2.833, 26)),
            np.append([2.505, 2.628, 2.701, 2.748, 2.779, 2.800, 2.815, 2.824, 2.829], np.repeat(2.832, 26)),
            np.append([2.491, 2.616, 2.690, 2.739, 2.771, 2.794, 2.810, 2.820, 2.827, 2.831, 2.832],
                      np.repeat(2.833, 24)),
            np.append([2.479, 2.605, 2.681, 2.731, 2.765, 2.789, 2.805, 2.817, 2.825, 2.830, 2.833],
                      np.repeat(2.834, 24)),
            np.append([2.469, 2.596, 2.673, 2.723, 2.759, 2.784, 2.802, 2.815, 2.824, 2.829, 2.833, 2.835],
                      np.repeat(2.936, 23)),
            np.append([2.460, 2.588, 2.665, 2.717, 2.753, 2.780, 2.798, 2.812, 2.822, 2.829, 2.834, 2.836],
                      np.repeat(2.838, 23)),
            np.append([2.452, 2.580, 2.659, 2.712, 2.749, 2.776, 2.796, 2.810, 2.821, 2.828, 2.834, 2.838],
                      np.repeat(2.840, 23)),
            np.append([2.445, 2.574, 2.653, 2.707, 2.745, 2.773, 2.793, 2.808, 2.820, 2.828, 2.834, 2.839, 2.841],
                      np.repeat(2.842, 22)),
            np.append([2.439, 2.568, 2.648, 2.702, 2.741, 2.770, 2.791, 2.807, 2.819, 2.828, 2.834, 2.839, 2.843],
                      np.repeat(2.845, 22)),
            np.append([2.400, 2.532, 2.615, 2.674, 2.717, 2.750, 2.776, 2.796, 2.813, 2.826, 2.837, 2.846, 2.853, 2.859,
                       2.863, 2.867, 2.869, 2.871], np.repeat(2.873, 17)),
            np.append([2.381, 2.514, 2.600, 2.660, 2.705, 2.741, 2.769, 2.791, 2.810, 2.825, 2.838, 2.849, 2.858, 2.866,
                       2.873, 2.878, 2.833, 2.887, 2.890, 2.894, 2897], np.repeat(2.898, 14)),
            np.append([2.363, 2.497, 2.584, 2.646, 2.694, 2.731, 2.761, 2.786, 2.807, 2.825, 2.839, 2.853, 2.864, 2.874,
                       2.883, 2.890, 2.897, 2.903, 2.908, 2.916, 2.923, 2.927, 2.931, 2.933], np.repeat(2.935, 11)),
            np.append([2.344, 2.479, 2.568, 2.632, 2.682, 2.722, 2.754, 2.781, 2.804, 2.824, 2.842, 2.857, 2.871, 2.883,
                       2.893, 2.903, 2.912, 2.920, 2.928, 2.940, 2.951, 2.960, 2.967, 2.974, 2.979, 2.984, 2.988, 2.991,
                       2.994], np.repeat(3.001, 6)),
            np.array([2.326, 2.462, 2.552, 2.619, 2.670, 2.712, 2.746, 2.776, 2.801, 2.824, 2.844, 2.861, 2.877, 2.892,
                      2.905, 2.918, 2.929, 2.939, 2.949, 2.966, 2.982, 2.995, 3.008, 3.019, 3.029, 3.038, 3.047, 3.054,
                      3.062, 3.091, 3.113, 3.129, 3.143, 3.154, 3.163])]
    q_05 = [np.repeat(17.97, 35), np.repeat(6.085, 35),
            np.append([4.501], np.repeat(4.516, 34)),
            np.append([3.927, 4.013], np.repeat(4.033, 33)),
            np.append([3.635, 3.749, 3.797], np.repeat(3.814, 32)),
            np.append([3.461, 3.587, 3.649, 3.680, 3.694], np.repeat(3.697, 30)),
            np.append([3.344, 3.477, 3.548, 3.558, 3.611, 3.622], np.repeat(3.626, 29)),
            np.append([3.261, 3.399, 3.475, 3.521, 3.549, 3.566, 3.575], np.repeat(3.579, 28)),
            np.append([3.199, 3.339, 3.420, 3.470, 3.502, 3.523, 3.536, 3.544], np.repeat(3.547, 27)),
            np.append([3.151, 3.293, 3.376, 3.430, 3.465, 3.489, 3.505, 3.516, 3.522, 3.525], np.repeat(3.526, 25)),
            np.append([3.113, 3.256, 3.342, 3.397, 3.435, 3.462, 3.480, 3.493, 3.501, 3.506, 3.509],
                      np.repeat(3.510, 24)),
            np.append([3.082, 3.225, 3.313, 3.370, 3.410, 3.439, 3.459, 3.474, 3.484, 3.491, 3.496, 3.498],
                      np.repeat(3.498, 24)),
            np.append([3.055, 3.200, 3.289, 3.348, 3.389, 3.419, 3.442, 3.458, 3.470, 3.478, 3.484, 3.488],
                      np.repeat(3.490, 24)),
            np.append([3.033, 3.178, 3.268, 3.329, 3.372, 3.403, 3.426, 3.444, 3.457, 3.467, 3.474, 3.479, 3.482],
                      np.repeat(3.484, 23)),
            np.append([3.014, 3.160, 3.250, 3.312, 3.356, 3.389, 3.413, 3.432, 3.446, 3.457, 3.465, 3.471, 3.476, 3.478,
                       3.480], np.repeat(3.481, 21)),
            np.append([2.998, 3.114, 3.235, 3.298, 3.343, 3.376, 3.402, 3.422, 3.437, 3.449, 3.458, 3.465, 3.470, 3.473,
                       3.477], np.repeat(3.478, 21)),
            np.append([2.984, 3.130, 3.222, 3.285, 3.331, 3.366, 3.392, 3.412, 3.429, 3.441, 3.451, 3.459, 3.465, 3.469,
                       3.473, 3.475], np.repeat(3.476, 20)),
            np.append([2.971, 3.118, 3.210, 3.274, 3.321, 3.356, 3.383, 3.405, 3.421, 3.425, 3.445, 3.454, 3.460, 3.465,
                       3.470, 3.472], np.repeat(3.474, 20)),
            np.append([2.960, 3.107, 3.199, 3.264, 3.311, 3.347, 3.375, 3.397, 3.415, 3.429, 3.440, 3.449, 3.456, 3.462,
                       3.467, 3.470, 3.472, 3.473], np.repeat(3.474, 18)),
            np.append([2.950, 3.097, 3.190, 3.255, 3.303, 3.339, 3.368, 3.391, 3.409, 3.424, 3.436, 3.445, 3.453, 3.459,
                       3.464, 3.467, 3.470, 3.472, 3.473], np.repeat(3.474, 16)),
            np.append([2.888, 3.035, 3.131, 3.199, 3.250, 3.290, 3.322, 3.349, 3.371, 3.389, 3.405, 3.418, 3.430, 3.439,
                       3.477, 3.454, 3.460, 3.466, 3.470, 3.477, 3.481, 3.484], np.repeat(3.486, 13)),
            np.append([2.858, 3.006, 3.102, 3.171, 3.224, 3.266, 3.300, 3.328, 3.352, 3.373, 3.390, 3.405, 3.418, 3.429,
                       3.439, 3.448, 3.456, 3.463, 3.469, 3.479, 3.486, 3.492, 3.497, 3.500, 3.503], np.repeat(3.504, 10)),
            np.append([2.829, 2.976, 3.073, 3.143, 3.198, 3.241, 3.277, 3.307, 3.333, 3.355, 3.374, 3.391, 3.406, 3.419,
                       3.431, 3.442, 3.451, 3.460, 3.467, 3.481, 3.492, 3.501, 3.509, 3.515, 3.521, 3.525, 3.529, 3.531,
                       3.534], np.repeat(3.527, 6)),
            np.append([2.800, 2.947, 3.045, 3.116, 3.172, 3.217, 3.254, 3.287, 3.314, 3.337, 3.359, 3.377, 3.394, 3.409,
                       3.423, 3.435, 3.446, 3.457, 3.466, 3.483, 3.498, 3.511, 3.522, 3.532, 3.541, 3.548, 3.555, 3.561,
                       3.566, 3.585, 3.596, 3.600], np.repeat(3.601, 3)),
            np.array([2.772, 2.918, 3.017, 3.089, 3.146, 3.193, 3.232, 3.265, 3.294, 3.320, 3.343, 3.363, 3.382, 3.399,
                       3.414, 3.428, 3.442, 3.454, 3.466, 3.486, 3.505, 3.522, 3.536, 3.550, 3.562, 3.574, 3.584, 3.594,
                       3.603, 3.640, 3.668, 3.690, 3.708, 3.722, 3.735])]
    q_01 = [np.repeat(90.03,  35), np.repeat(14.04, 35),
            np.append([8.261], np.repeat(8.321, 34)),
            np.append([6.512, 6.677, 6.740], np.repeat(6.756, 32)),
            np.append([5.702, 5.893, 5.989, 6.040, 6.065], np.repeat(6.074, 30)),
            np.append([5.243, 5.439, 5.549, 5.614, 5.655, 5.680, 5.694, 5.701], np.repeat(5.703, 27)),
            np.append([4.949, 5.145, 5.260, 5.334, 5.383, 5.416, 5.439, 5.454, 5.464, 5.470], np.repeat(5.472, 25)),
            np.append([4.596, 4.787, 4.906, 4.986, 5.043, 5.086, 5.118, 5.142, 5.160, 5.174, 5.185, 5.193, 5.199, 5.203,
                       5.205], np.repeat(5.206, 20)),
            np.append([4.482, 4.671, 4.790, 4.871, 4.931, 4.975, 5.010, 5.037, 5.058, 5.074, 5.088, 5.098, 5.106, 5.112,
                       5.117, 5.120, 5.122], np.repeat(5.124, 18)),
            np.append([4.392, 4.579, 4.697, 4.780, 4.841, 4.887, 4.924, 4.952, 4.975, 4.994, 5.009, 5.021, 5.031, 5.039,
                       5.045, 5.050, 5.054, 5.057, 5.059], np.repeat(5.061, 16)),
            np.append([4.320, 4.504, 4.622, 4.706, 4.767, 4.815, 4.852, 4.883, 4.907, 4.927, 4.944, 4.958, 4.969, 4.978,
                       4.986, 4.993, 4.998, 5.002, 5.006, 5.010], np.repeat(5.011, 15)),
            np.append([4.260, 4.442, 4.560, 4.644, 4.706, 4.755, 4.793, 4.824, 4.850, 4.872, 4.889, 4.904, 4.917, 4.928,
                       4.937, 4.944, 4.950, 4.956, 4.960, 4.966, 4.970], np.repeat(4.972, 14)),
            np.append([4.210, 4.391, 4.508, 4.591, 4.654, 4.704, 4.743, 4.775, 4.802, 4.824, 4.843, 4.859, 4.872, 4.884,
                       4.894, 4.902, 4.910, 4.916, 4.921, 4.929, 4.935, 4.938], np.repeat(4.940, 13)),
            np.append([4.168, 4.347, 4.463, 4.547, 4.610, 4.660, 4.700, 4.733, 4.760, 4.783, 4.803, 4.820, 4.834, 4.846,
                       4.857, 4.866, 4.874, 4.881, 4.887, 4.897, 4.904, 4.909, 4.912], np.repeat(4.914, 12)),
            np.append([4.131, 4.309, 4.425, 4.509, 4.572, 4.622, 4.663, 4.696, 4.724, 4.748, 4.768, 4.786, 4.800, 4.813,
                       4.825, 4.835, 4.844, 4.851, 4.858, 4.869, 4.877, 4.883, 4.887, 4.890], np.repeat(4.892, 11)),
            np.append([4.099, 4.275, 4.391, 4.475, 4.539, 4.589, 4.630, 4.664, 4.693, 4.717, 4.738, 4.756, 4.771, 4.785,
                       4.797, 4.807, 4.816, 4.824, 4.832, 4.844, 4.853, 4.860, 4.865, 4.869, 4.872, 4.873],
                      np.repeat(4.874, 9)),
            np.append([4.071, 4.246, 4.362, 4.445, 4.509, 4.560, 4.601, 4.635, 4.664, 4.689, 4.711, 4.729, 4.745, 4.759,
                       4.772, 4.783, 4.792, 4.801, 4.808, 4.821, 4.832, 4.839, 4.846, 4.850, 4.854, 4.856, 4.857],
                      np.repeat(4.858, 8)),
            np.append([4.046, 4.220, 4.335, 4.419, 4.483, 4.534, 4.575, 4.610, 4.639, 4.665, 4.686, 4.705, 4.722, 4.736,
                       4.749, 4.761, 4.771, 4.780, 4.788, 4.802, 4.812, 4.821, 4.828, 4.833, 4.838, 4.841, 4.843, 4.844],
                      np.repeat(4.845, 7)),
            np.append([4.024, 4.197, 4.312, 4.395, 4.459, 4.510, 4.552, 4.587, 4.617, 4.642, 4.664, 4.684, 4.701, 4.716,
                       4.729, 4.741, 4.751, 4.761, 4.769, 4.784, 4.795, 4.805, 4.813, 4.818, 4.823, 4.827, 4.830, 4.832],
                       np.repeat(4.833, 7)),
            np.append([3.889, 4.056, 4.168, 4.250, 4.314, 4.366, 4.409, 4.445, 4.477, 4.504, 4.528, 4.550, 4.569, 4.586,
                       4.601, 4.615, 4.628, 4.640, 4.650, 4.669, 4.685, 4.699, 4.711, 4.721, 4.730, 4.738, 4.744, 4.750,
                       4.755, 4.772], np.repeat(4.777, 5)),
            np.append([3.825, 3.988, 4.098, 4.180, 4.244, 4.296, 4.339, 4.376, 4.408, 4.436, 4.461, 4.483, 4.503, 4.521,
                       4.537, 4.553, 4.566, 4.579, 4.591, 4.611, 4.630, 4.645, 4.659, 4.671, 4.682, 4.692, 4.700, 4.708,
                       4.715, 4.740, 4.754, 4.761], np.repeat(4.764, 3)),
            np.array([3.762, 3.922, 4.031, 4.111, 4.174, 4.226, 4.270, 4.307, 4.340, 4.368, 4.394, 4.417, 4.438, 4.456,
                       4.474, 4.490, 4.504, 4.518, 4.530, 4.553, 4.573, 4.591, 4.607, 4.620, 4.633, 4.645, 4.655, 4.665,
                       4.673, 4.707, 4.730, 4.745, 4.755, 4.761, 4.765]),
            np.array([3.702, 3.858, 3.965, 4.044, 4.107, 4.158, 4.202, 4.239, 4.272, 4.301, 4.327, 4.351, 4.372, 4.392,
                       4.410, 4.426, 4.442, 4.456, 4.469, 4.494, 4.516, 4.535, 4.552, 4.568, 4.583, 4.596, 4.609, 4.619,
                       4.630, 4.673, 4.703, 4.727, 4.745, 4.759, 4.770]),
            np.array([3.643, 3.796, 3.900, 3.978, 4.040, 4.091, 4.135, 4.172, 4.205, 4.235, 4.261, 4.285, 4.307, 4.327,
                       4.345, 4.363, 4.379, 4.394, 4.408, 4.434, 4.457, 4.478, 4.497, 4.514, 4.530, 4.545, 4.559, 4.572,
                       4.584, 4.635, 4.675, 4.707, 4.734, 4.756, 4.776])]
    mean_data = np.mean(args, axis=1)
    grand_mean = np.mean(args)
    ssb = np.sum(len_data * np.power(mean_data - grand_mean, 2))
    sst = np.sum(np.power(args - grand_mean, 2))
    msw = (sst - ssb) / df
    se = sqrt(msw * 2 / n)
    all_sig_diffs = []
    for idx, val in enumerate(big_to_small):
        other_vals = small_to_big[:-(idx + 1)]
        val_rank = rank[-(idx + 1)]
        other_val_rank = rank[:-(idx + 1)]
        if len(other_vals) == 0:
            break
        if alpha == 0.10:
            q = np.array(q_10)[index][np.arange(col - idx + 1)]
        elif alpha == 0.05:
            q = np.array(q_05)[index][np.arange(col - idx + 1)]
        else:
            q = np.array(q_10)[index][np.arange(col - idx + 1)]
        r = se * q
        combinations = [(val_rank, o_val_rank) for o_val_rank in other_val_rank]
        diffs = val - other_vals
        all_sig_diffs.append(np.array(combinations)[diffs > r])
    return list(chain(*all_sig_diffs))