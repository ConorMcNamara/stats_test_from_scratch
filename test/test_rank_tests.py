from StatsTest.rank_tests import *
import pytest
import unittest
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, friedmanchisquare
import numpy as np


class TestRankTest(unittest.TestCase):

    # Mann-Whitney U Test

    def test_twoSampleMannWhitneyTest_wrongAlternative_Error(self):
        sample_data = [10, 20, 30]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            two_sample_mann_whitney_test(sample_data, sample_data, alternative="more")

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

    # Wilcoxon Test

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

    def test_twoSampleWilcoxonTest_Wilcox_pResult(self):
        x1 = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
        x2 = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]
        w, p = two_sample_wilcoxon_test(x1, x2, alternative='two-sided', handle_zero='wilcox')
        w2, p2 = wilcoxon(x1, x2)
        assert pytest.approx(p2, 0.001) == p

    def test_twoSampleWilcoxonTest_Wilcox_wResult(self):
        x1 = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
        x2 = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]
        w, p = two_sample_wilcoxon_test(x1, x2, alternative='two-sided', handle_zero='wilcox')
        assert pytest.approx(9) == w

    def test_twoSampleWilcoxonTest_Pratt_pResult(self):
        x1 = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
        x2 = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]
        w, p = two_sample_wilcoxon_test(x1, x2, alternative='two-sided', handle_zero='pratt')
        w2, p2 = wilcoxon(x1, x2, zero_method="pratt")
        assert pytest.approx(p2, 0.01) == p

    def test_twoSampleWilcoxonTest_Pratt_wResult(self):
        x1 = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
        x2 = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]
        w, p = two_sample_wilcoxon_test(x1, x2, alternative='two-sided', handle_zero='pratt')
        assert pytest.approx(10) == w

    # Friedman Test

    def test_friedmanTest_kLessThree_Error(self):
        sample_data1 = [1, 2, 3]
        sample_data2 = [1, 2, 3]
        with pytest.raises(AttributeError, match="Friedman Test not appropriate for 2 levels"):
            friedman_test(sample_data1, sample_data2)

    def test_friedmanTest_pResult(self):
        x1 = [27, 2, 4, 18, 7, 9]
        x2 = [20, 8, 14, 36, 21, 22]
        x3 = [34, 31, 3, 23, 30, 6]
        x, p = friedman_test(x1, x2, x3)
        x2, p2 = friedmanchisquare(x1, x2, x3)
        assert pytest.approx(p2) == p

    def test_friedmanTest_xResult(self):
        x1 = [27, 2, 4, 18, 7, 9]
        x2 = [20, 8, 14, 36, 21, 22]
        x3 = [34, 31, 3, 23, 30, 6]
        x, p = friedman_test(x1, x2, x3)
        x2, p2 = friedmanchisquare(x1, x2, x3)
        assert pytest.approx(x2) == x

    # Page's Trend Test

    def test_pageTrendTest_nLessThree_Error(self):
        sample_data1 = [1, 2, 3]
        sample_data2 = [1, 2, 3]
        with pytest.raises(AttributeError, match="Page Test not appropriate for 2 levels"):
            page_trend_test(sample_data1, sample_data2)

    def test_pageTrendTest_unequalLength_Error(self):
        sample_data1 = [1, 2, 3]
        sample_data2 = [1, 2, 3]
        sample_data3 = [1, 2]
        with pytest.raises(AttributeError, match="Page Test requires that each level have the same number of observations"):
            page_trend_test(sample_data1, sample_data2, sample_data3)

    def test_pageTrendTest_alternativeInt_Error(self):
        sample_data1 = [1, 2, 3]
        sample_data2 = [1, 2, 3]
        sample_data3 = [1, 2, 3]
        with pytest.raises(ValueError, match="Cannot have alternative hypothesis with non-string value"):
            page_trend_test(sample_data1, sample_data2, sample_data3, alternative=10)

    def test_pageTrendTest_alternativeTwoSided_Error(self):
        sample_data1 = [1, 2, 3]
        sample_data2 = [1, 2, 3]
        sample_data3 = [1, 2, 3]
        with pytest.raises(ValueError, match="Cannot discern alternative hypothesis"):
            page_trend_test(sample_data1, sample_data2, sample_data3, alternative="two-sided")

    def test_pageTrendTest_pResult(self):
        data1, data2, data3, data4, data5 = [100, 200, 121, 90], [90, 150, 130, 75], [105, 80, 75, 76], [70, 50, 20, 54], [5, 10, 25, 32]
        l, p = page_trend_test(data1, data2, data3, data4, data5)
        assert pytest.approx(0.00033692926567685522) == p

    def test_pageTrendTest_lResult(self):
        data1, data2, data3, data4, data5 = [100, 200, 121, 90], [90, 150, 130, 75], [105, 80, 75, 76], [70, 50, 20, 54], [5, 10, 25, 32]
        l, p = page_trend_test(data1, data2, data3, data4, data5)
        assert pytest.approx(214) == l

    # Kruskal-Wallis Test

    def test_kruskalWallisTest_gLessTwo_Error(self):
        sample_data = [1, 2, 3]
        with pytest.raises(AttributeError, match="Cannot run Kruskal-Wallis Test with less than 2 groups"):
            kruskal_wallis_test(sample_data)

    def test_kruskalWallis_pResult(self):
        x1 = [27, 2, 4, 18, 7, 9]
        x2 = [20, 8, 14, 36, 21, 22]
        x3 = [34, 31, 3, 23, 30, 6]
        h, p = kruskal_wallis_test(x1, x2, x3)
        h2, p2 = kruskal(x1, x2, x3)
        assert pytest.approx(p2) == p

    def test_kruskalWallis_hResult(self):
        x1 = [27, 2, 4, 18, 7, 9]
        x2 = [20, 8, 14, 36, 21, 22]
        x3 = [34, 31, 3, 23, 30, 6]
        h, p = kruskal_wallis_test(x1, x2, x3)
        h2, p2 = kruskal(x1, x2, x3)
        assert pytest.approx(h) == h2


if __name__ == '__main__':
    unittest.main()