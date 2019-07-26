from StatsTest.rank_tests import *
import pytest
import unittest
from scipy.stats import mannwhitneyu, wilcoxon
import numpy as np


class TestRankTest(unittest.TestCase):

    def test_twoSampleMannWhitneyTest_wrongAlternative_Error(self):
        sample_data = [10, 20, 30]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            two_sample_mann_whitney_test(sample_data, sample_data, alternative="more")

    def test_twoSampleWilcoxon_wrongAlternative_Error(self):
        sample_data = [1, 2, 3]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            two_sample_wilcoxon_test(sample_data, sample_data, alternative='more')

    def test_twoSampleWilcoxon_wrongHandleZero_Error(self):
        sample_data = [1, 2, 3]
        with pytest.raises(ValueError, match="Cannot determine how to handle differences of zero"):
            two_sample_wilcoxon_test(sample_data, sample_data, handle_zero="blarg")

    def test_twoSampleWilcoxon_unequalLength_Error(self):
        sample_data1 = [1, 2, 3]
        sample_data2 = [1, 2]
        with pytest.raises(AttributeError, match="Cannot perform signed wilcoxon test on unpaired data"):
            two_sample_wilcoxon_test(sample_data1, sample_data2)

    def test_friedmanTest_kLessThree_Error(self):
        sample_data1 = [1, 2, 3]
        sample_data2 = [1, 2, 3]
        with pytest.raises(AttributeError, match="Friedman Test not appropriate for 2 levels"):
            friedman_test(sample_data1, sample_data2)

    def test_twoSampleMannWhitneyTest_pResult(self):
        sample_data = np.repeat([1, 2, 3], 10)
        u_1, p_1 = two_sample_mann_whitney_test(sample_data, sample_data, alternative='two-sided')
        u_2, p_2 = mannwhitneyu(sample_data, sample_data, use_continuity=False, alternative='two-sided')
        assert pytest.approx(0.0) == p_1 - p_2

    def test_twoSampleMannWhitneyTest_uResult(self):
        sample_data = np.repeat([1, 2, 3], 10)
        u_1, p_1 = two_sample_mann_whitney_test(sample_data, sample_data, alternative='two-sided')
        u_2, p_2 = mannwhitneyu(sample_data, sample_data, alternative='two-sided')
        assert pytest.approx(0.0, 0.01) == u_1 - u_2

    def test_twoSampleWilcoxonTest_pResult(self):
        x1 = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
        x2 = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]
        w, p = two_sample_wilcoxon_test(x1, x2, alternative='two-sided', handle_zero='wilcox')
        w2, p2 = wilcoxon(x1, x2)
        assert pytest.approx(p2, 0.01) == p

    def test_twoSampleWilcoxonTest_wResult(self):
        x1 = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
        x2 = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]
        w, p = two_sample_wilcoxon_test(x1, x2, alternative='two-sided', handle_zero='wilcox')
        assert pytest.approx(9) == w

    def test_friedmanTest_pResult(self):
        x1 = [27, 2, 4, 18, 7, 9]
        x2 = [20, 8, 14, 36, 21, 22]
        x3 = [34, 31, 3, 23, 30, 6]
        x, p = friedman_test(x1, x2, x3)
        assert pytest.approx(0.311, 0.01) == p

    def test_friedmanTest_xResult(self):
        x1 = [27, 2, 4, 18, 7, 9]
        x2 = [20, 8, 14, 36, 21, 22]
        x3 = [34, 31, 3, 23, 30, 6]
        x, p = friedman_test(x1, x2, x3)
        assert pytest.approx(2.333, 0.01) == x


if __name__ == '__main__':
    unittest.main()