import unittest
import pytest
from StatsTest.multi_group_tests import *
from scipy.stats import levene, f_oneway, bartlett, median_test
from numpy.random import randint
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.libqsturng import psturng


class TestMultiGroupTests(unittest.TestCase):

    # Levene Test

    def test_leveneTest_kLessTwo_Error(self):
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform a Levene Test"):
            levene_test(sample_data)

    def test_leveneTest_pResult(self):
        data_1 = randint(0, 100, 10)
        data_2 = randint(500, 550, 10)
        data_3 = randint(0, 10, 10)
        data_4 = randint(0, 50, 10)
        w1, p1 = levene_test(data_1, data_2, data_3, data_4)
        w2, p2 = levene(data_1, data_2, data_3, data_4, center='mean')
        assert pytest.approx(p2) == p1

    def test_leveneTest_wResult(self):
        data_1 = randint(0, 100, 10)
        data_2 = randint(500, 550, 10)
        data_3 = randint(0, 10, 10)
        data_4 = randint(0, 50, 10)
        w1, p1 = levene_test(data_1, data_2, data_3, data_4)
        w2, p2 = levene(data_1, data_2, data_3, data_4, center='mean')
        assert pytest.approx(w2) == w1

    # Brown-Forsythe Test

    def test_brownForsytheTest_kLessTwo_Error(self):
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform a Brown-Forsythe Test"):
            brown_forsythe_test(sample_data)

    def test_brownForsytheTest_pResult(self):
        data_1 = randint(0, 100, 10)
        data_2 = randint(500, 550, 10)
        data_3 = randint(0, 10, 10)
        data_4 = randint(0, 50, 10)
        w1, p1 = brown_forsythe_test(data_1, data_2, data_3, data_4)
        w2, p2 = levene(data_1, data_2, data_3, data_4, center='median')
        assert pytest.approx(p2) == p1

    def test_brownForsytheTest_wResult(self):
        data_1 = randint(0, 100, 10)
        data_2 = randint(500, 550, 10)
        data_3 = randint(0, 10, 10)
        data_4 = randint(0, 50, 10)
        w1, p1 = brown_forsythe_test(data_1, data_2, data_3, data_4)
        w2, p2 = levene(data_1, data_2, data_3, data_4, center='median')
        assert pytest.approx(w2) == w1

    # One Way F Test

    def test_oneWayFTest_kLessTwo_Error(self):
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform a one-way F Test"):
            one_way_f_test(sample_data)

    def test_oneWayFTest_pResult(self):
        data_1 = randint(0, 100, 10)
        data_2 = randint(500, 550, 10)
        data_3 = randint(0, 10, 10)
        data_4 = randint(0, 50, 10)
        f1, p1 = one_way_f_test(data_1, data_2, data_3, data_4)
        f2, p2 = f_oneway(data_1, data_2, data_3, data_4)
        assert pytest.approx(p2) == p1

    def test_oneWayFTest_fResult(self):
        data_1 = randint(0, 100, 10)
        data_2 = randint(500, 550, 10)
        data_3 = randint(0, 10, 10)
        data_4 = randint(0, 50, 10)
        f1, p1 = one_way_f_test(data_1, data_2, data_3, data_4)
        f2, p2 = f_oneway(data_1, data_2, data_3, data_4)
        assert pytest.approx(f2) == f1

    # Bartlett Test

    def test_bartlettTest_kLessTwo_Error(self):
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform the Bartlett Test"):
            bartlett_test(sample_data)

    def test_bartlettTest_pResult(self):
        data_1 = randint(0, 100, 10)
        data_2 = randint(500, 550, 10)
        data_3 = randint(0, 10, 10)
        data_4 = randint(0, 50, 10)
        x1, p1 = bartlett_test(data_1, data_2, data_3, data_4)
        x2, p2 = bartlett(data_1, data_2, data_3, data_4)
        assert pytest.approx(p2) == p1

    def test_barlettTest_xResult(self):
        data_1 = randint(0, 100, 10)
        data_2 = randint(500, 550, 10)
        data_3 = randint(0, 10, 10)
        data_4 = randint(0, 50, 10)
        x1, p1 = bartlett_test(data_1, data_2, data_3, data_4)
        x2, p2 = bartlett(data_1, data_2, data_3, data_4)
        assert pytest.approx(x2) == x1

    # Tukey Range Test

    def test_tukeyRangeTest_kLessTwo_Error(self):
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform Tukey Range Test"):
            tukey_range_test(sample_data)

    def test_tukeyRangeTest_pResult(self):
        x1, x2, x3 = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]
        results = tukey_range_test(x1, x2, x3)
        model = pairwise_tukeyhsd(x1+x2+x3, groups=[0]*5 + [1]*5 + [2]*5)
        p_vals = psturng(np.abs(model.meandiffs / model.std_pairs), len(model.groupsunique), model.df_total)
        for i in range(3):
            assert pytest.approx(p_vals[i]) == results[i][2]

    # Cochran Test

    def test_cochranTest_kLessThree_Error(self):
        sample_data = [0, 1, 0, 1]
        with pytest.raises(AttributeError, match="Cannot run Cochran's Q Test with less than 3 treatments"):
            cochran_q_test(sample_data, sample_data)

    def test_cochranTest_nonBinary_Error(self):
        sample_data = [1, 2, 3, 4]
        sample_data2 = [0, 1, 0, 1]
        with pytest.raises(AttributeError, match="Cochran's Q Test only works with binary variables"):
            cochran_q_test(sample_data, sample_data2, sample_data2)

    def test_cochranTest_pResult(self):
        x1, x2, x3, x4 = [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0]
        t1, p1 = cochran_q_test(x1, x2, x3, x4)
        t2, p2, df = cochrans_q(np.vstack([x1, x2, x3, x4]).T, return_object=False)
        assert pytest.approx(p1) == p2

    def test_cochranTest_tResult(self):
        x1, x2, x3, x4 = [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0]
        t1, p1 = cochran_q_test(x1, x2, x3, x4)
        t2, p2, df = cochrans_q(np.vstack([x1, x2, x3, x4]).T, return_object=False)
        assert pytest.approx(t1) == t2

    # Jonckheere Trend Test

    def test_jonckheereTest_uLessTwo_Error(self):
        sample_data = [1, 2, 3]
        with pytest.raises(AttributeError, match="Cannot run Jonckheere Test with less than 2 groups"):
            jonckheere_trend_test(sample_data)

    def test_jonckheereTest_unevenSampleSize_Error(self):
        sample_data1 = [1, 2, 3]
        sample_data2 = [3, 4]
        with pytest.raises(AttributeError, match="Jonckheere Test requires that each group have the same number of observations"):
            jonckheere_trend_test(sample_data1, sample_data2)

    def test_jonckheereTest_alternativeInt_Error(self):
        sample_data = [1, 2, 3]
        with pytest.raises(TypeError, match="Cannot have alternative hypothesis with non-string value"):
            jonckheere_trend_test(sample_data, sample_data, alternative=10)

    def test_jonckheereTest_alternativeTwoSided_Error(self):
        sample_data = [1, 2, 3]
        with pytest.raises(ValueError, match="Cannot discern alternative hypothesis"):
            jonckheere_trend_test(sample_data, sample_data, alternative="two-sided")

    def test_jonckheereTest_pResult(self):
        data_1, data_2, data_3 = [10, 12, 14, 16], [12, 18, 20, 22], [20, 25, 27, 30]
        z, p = jonckheere_trend_test(data_1, data_2, data_3)
        assert pytest.approx(0.0016454416431436192) == p

    def test_jonckheereTest_zResult(self):
        data_1, data_2, data_3 = [10, 12, 14, 16], [12, 18, 20, 22], [20, 25, 27, 30]
        z, p = jonckheere_trend_test(data_1, data_2, data_3)
        assert pytest.approx(2.939, 0.001) == z

    # Median Test

    def test_medianTest_kLessTwo_Error(self):
        data_1 = [100, 200, 300]
        with pytest.raises(AttributeError, match="Cannot run Median Test with less than 2 groups"):
            mood_median_test(data_1)

    def test_medianTest_alternativeNotString_Error(self):
        data_1, data_2 = [100, 200, 300], [300, 400, 500]
        with pytest.raises(AttributeError):
            mood_median_test(data_1, data_2, alternative=10)

    def test_medianTest_alternativeWrong_Error(self):
        data_1, data_2 = [100, 200, 300], [300, 400, 500]
        with pytest.raises(ValueError, match="Cannot discern alternative hypothesis"):
            mood_median_test(data_1, data_2, alternative="moar")

    def test_medianTest_handleMedNotString_Error(self):
        data_1, data_2 = [100, 200, 300], [300, 400, 500]
        with pytest.raises(AttributeError):
            mood_median_test(data_1, data_2, handle_med=10)

    def test_medianTest_handleMedWrong_Error(self):
        data_1, data_2 = [100, 200, 300], [300, 400, 500]
        with pytest.raises(ValueError, match="Cannot discern how to handle median value"):
            mood_median_test(data_1, data_2, handle_med="moar")

    def test_medianTest_pResult(self):
        g1 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
        g2 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
        g3 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]
        x1, p1 = mood_median_test(g1, g2, g3, alternative='less')
        x2, p2, med, tbl = median_test(g1, g2, g3)
        assert pytest.approx(p2) == p1

    def test_medianTest_xResult(self):
        g1 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
        g2 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
        g3 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]
        x1, p1 = mood_median_test(g1, g2, g3, alternative='less')
        x2, p2, med, tbl = median_test(g1, g2, g3)
        assert pytest.approx(x2) == x1

    ## Dunnett Test

    def test_dunnetTest_kLessTwo_Error(self):
        control = [10, 10, 10]
        d1 = [10, 20, 30]
        with pytest.raises(AttributeError, match="Cannot run Dunnett Test with less than two groups"):
            dunnett_test(control, 0.05, d1)

    def test_dunnettTest_alphaWrong_Error(self):
        control = [10, 10, 10]
        d1 = [10, 20, 30]
        with pytest.raises(ValueError, match="Alpha level not currently supported"):
            dunnett_test(control, 0.2, d1, d1)

    def test_dunnettTest_results(self):
        control = [55, 47, 48]
        p1 = [55, 64, 64]
        p2 = [55, 49, 52]
        p3 = [50, 44, 41]
        np.testing.assert_array_equal(dunnett_test(control, 0.05, p1, p2, p3), [True, False, False])

    def test_dunnettTest_moreResults(self):
        control = [51, 87, 50, 48, 79, 61, 53, 54]
        p1 = [82, 91, 92, 80, 52, 79, 73, 74]
        p2 = [79, 84, 74, 98, 63, 83, 85, 58]
        p3 = [85, 80, 65, 71, 67, 51, 63, 93]
        np.testing.assert_array_equal(dunnett_test(control, 0.05, p1, p2, p3), [True, True, False])

    # Duncan's New Multi-Range Test
    def test_duncanMultiRangeTest_kLessTwo_Error(self):
        d1 = [10, 20, 30]
        with pytest.raises(AttributeError, match="Cannot run Duncan Multi-Range Test with less than two groups"):
            duncan_multiple_range_test(0.05, d1)

    def test_duncanMultiRangeTest_alphaWrong_Error(self):
        d1 = [10, 20, 30]
        with pytest.raises(ValueError, match="Alpha level not currently supported"):
            duncan_multiple_range_test(0.2, d1, d1)

    def test_duncanMultiRangeTest_results(self):
        data_1, data_2, data_3 = [9, 14, 11], [20, 19, 23], [39, 38, 41]
        np.testing.assert_array_equal(duncan_multiple_range_test(0.05, data_1, data_2, data_3),
                                      [np.array([2, 0]), np.array([2, 1]), np.array([1, 0])])

    def test_duncanMultiRangeTest_moreResults(self):
        data_1, data_2, data_3, data_4, data_5 = [10, 10, 10, 10, 9], [15, 15, 15, 15, 17], [20, 20, 20, 20, 8], [22, 22, 22, 22, 20], [10, 10, 10, 10, 14]
        np.testing.assert_array_equal(duncan_multiple_range_test(0.05, data_1, data_2, data_3, data_4, data_5),
                                      [np.array([3, 0]), np.array([3, 4]), np.array([3, 1]), np.array([2, 0]), np.array([2, 4]), np.array([1, 0])])


if __name__ == '__main__':
    unittest.main()