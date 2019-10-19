from StatsTest.rank_tests import *
import pytest
import unittest
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, friedmanchisquare, fligner, ansari, mood
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

    # Quade Test

    def test_quadeTest_kLessThree_Error(self):
        sample_data_1 = [1, 2, 3]
        sample_data_2 = [2, 4, 6]
        with pytest.raises(AttributeError, match="Quade Test not appropriate for 2 levels"):
            quade_test(sample_data_1, sample_data_2)

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

    # Fligner-Kileen Test

    def test_flignerKileenTest_kLessTwo_Error(self):
        sample_data = [1, 2, 3]
        with pytest.raises(AttributeError, match='Cannot perform Fligner-Kileen Test with less than 2 groups'):
            fligner_kileen_test(sample_data)

    def test_flignerKileenTest_centerWrong_Error(self):
        sample_data = [1, 2, 3]
        with pytest.raises(ValueError, match='Cannot discern how to center the data'):
            fligner_kileen_test(sample_data, sample_data, center="center")

    def test_flignerKileenTest_pResult(self):
        data_1 = [51, 87, 50, 48, 79, 61, 53, 54]
        data_2 = [82, 91, 92, 80, 52, 79, 73, 74]
        data_4 = [85, 80, 65, 71, 67, 51, 63, 93]
        data_3 = [79, 84, 74, 98, 63, 83, 85, 58]
        x1, p1 = fligner_kileen_test(data_1, data_2, data_3, data_4, center='median')
        x2, p2 = fligner(data_1, data_2, data_3, data_4, center='median')
        assert pytest.approx(p2) == p1

    def test_flignerKileenTest_xResult(self):
        data_1 = [51, 87, 50, 48, 79, 61, 53, 54]
        data_2 = [82, 91, 92, 80, 52, 79, 73, 74]
        data_4 = [85, 80, 65, 71, 67, 51, 63, 93]
        data_3 = [79, 84, 74, 98, 63, 83, 85, 58]
        x1, p1 = fligner_kileen_test(data_1, data_2, data_3, data_4, center='median')
        x2, p2 = fligner(data_1, data_2, data_3, data_4, center='median')
        assert pytest.approx(x2) == x1

    # Ansari-Bradley Test

    def test_ansariBradleyTest_alternativeWrong(self):
        data_1 = np.random.normal(0, 100, 100)
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            ansari_bradley_test(data_1, data_1, alternative="moar")

    def test_ansariBradleyTest_exact_pResult(self):
        data_1 = [-63, 18, 84, 160, 33, -82, 49, 74, 58, -31, 151]
        data_2 = [78, -124, -443, 225, -9, -3, 189, 164, 119, 184]
        x1, p1 = ansari_bradley_test(data_1, data_2, alternative="two-sided")
        x2, p2 = ansari(data_1, data_2)
        assert pytest.approx(p2) == p1

    def test_ansariBradleyTest_exact_xResult(self):
        data_1 = [-63, 18, 84, 160, 33, -82, 49, 74, 58, -31, 151]
        data_2 = [78, -124, -443, 225, -9, -3, 189, 164, 119, 184]
        x1, p1 = ansari_bradley_test(data_1, data_2, alternative="two-sided")
        x2, p2 = ansari(data_1, data_2)
        assert pytest.approx(x2) == x1

    def test_ansariBradleyTest_approxEven_pResult(self):
        data_1 = np.arange(0, 101)
        data_2 = np.arange(50, 151)
        x1, p1 = ansari_bradley_test(data_1, data_2, alternative="two-sided")
        x2, p2 = ansari(data_1, data_2)
        assert pytest.approx(p2) == p1

    def test_ansariBradleyTest_approxEven_xResult(self):
        data_1 = np.arange(0, 101)
        data_2 = np.arange(50, 151)
        x1, p1 = ansari_bradley_test(data_1, data_2, alternative="two-sided")
        x2, p2 = ansari(data_1, data_2)
        assert pytest.approx(x2) == x1

    def test_ansariBradleyTest_approxOdd_pResult(self):
        data_1 = np.arange(1, 101)
        data_2 = np.arange(50, 151)
        x1, p1 = ansari_bradley_test(data_1, data_2, alternative="two-sided")
        x2, p2 = ansari(data_1, data_2)
        assert pytest.approx(p2) == p1

    def test_ansariBradleyTest_approxOdd_xResult(self):
        data_1 = np.arange(1, 101)
        data_2 = np.arange(50, 151)
        x1, p1 = ansari_bradley_test(data_1, data_2, alternative="two-sided")
        x2, p2 = ansari(data_1, data_2)
        assert pytest.approx(x2) == x1

    # Mood Test

    def test_moodTest_alternativeWrong_Error(self):
        data_1 = np.arange(1, 100)
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            mood_test(data_1, data_1, alternative='moar')

    def test_moodTest_nLessThree_Error(self):
        data_1 = [1]
        with pytest.raises(AttributeError, match="Not enough observations to perform mood dispertion test"):
            mood_test(data_1, data_1)

    def test_moodTest_pResult(self):
        data_1 = np.random.randint(0, 100, 1000)
        data_2 = np.random.normal(0, 100, 1000)
        z1, p1 = mood_test(data_1, data_2)
        z2, p2 = mood(data_1, data_2)
        assert pytest.approx(p2) == p1

    def test_moodTest_zResult(self):
        data_1 = np.random.randint(0, 100, 1000)
        data_2 = np.random.normal(0, 100, 1000)
        z1, p1 = mood_test(data_1, data_2)
        z2, p2 = mood(data_1, data_2)
        assert pytest.approx(z2) == z1

    # Cucconi Test
    def test_cucconiTest_howWrong_Error(self):
        data_1 = np.random.randint(0, 100, 50)
        data_2 = np.random.randint(0, 10, 50)
        with pytest.raises(ValueError, match="Cannot identify method for calculating p-value"):
            cucconi_test(data_1, data_2, how='moar')

    # Lepage Test


if __name__ == '__main__':
    unittest.main()