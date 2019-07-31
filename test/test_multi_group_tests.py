import unittest
import pytest
from StatsTest.multi_group_tests import *
from scipy.stats import levene, f_oneway, bartlett
from numpy.random import randint
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.libqsturng import psturng


class TestMultiGroupTests(unittest.TestCase):

    def test_leveneTest_kLessTwo_Error(self):
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform a Levene Test"):
            levene_test(sample_data)

    def test_brownForsytheTest_kLessTwo_Error(self):
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform a Brown-Forsythe Test"):
            brown_forsythe_test(sample_data)

    def test_oneWayFTest_kLessTwo_Error(self):
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform a one-way F Test"):
            one_way_f_test(sample_data)

    def test_bartlettTest_kLessTwo_Error(self):
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform the Bartlett Test"):
            bartlett_test(sample_data)

    def test_tukeyRangeTest_kLessTwo_Error(self):
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform Tukey Range Test"):
            tukey_range_test(sample_data)

    def test_cochranTest_kLessThree_Error(self):
        sample_data = [0, 1, 0, 1]
        with pytest.raises(AttributeError, match="Cannot run Cochran's Q Test with less than 3 treatments"):
            cochran_q_test(sample_data, sample_data)

    def test_cochranTest_nonBinary_Error(self):
        sample_data = [1, 2, 3, 4]
        sample_data2 = [0, 1, 0, 1]
        with pytest.raises(AttributeError, match="Cochran's Q Test only works with binary variables"):
            cochran_q_test(sample_data, sample_data2, sample_data2)

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

    def test_tukeyRangeTest_pResult(self):
        x1 = [1, 2, 3, 4, 5]
        x2 = [6, 7, 8, 9, 10]
        x3 = [11, 12, 13, 14, 15]
        results = tukey_range_test(x1, x2, x3)
        model = pairwise_tukeyhsd(x1+x2+x3, groups=[0]*5 + [1]*5 + [2]*5)
        p_vals = psturng(np.abs(model.meandiffs / model.std_pairs), len(model.groupsunique), model.df_total)
        for i in range(3):
            assert pytest.approx(p_vals[i]) == results[i][2]

    def test_cochranTest_pResult(self):
        x1 = [1, 0, 1, 0, 1]
        x2 = [1, 1, 1, 1, 1]
        x3 = [0, 0, 0, 0, 0]
        x4 = [0, 1, 0, 1, 0]
        t1, p1 = cochran_q_test(x1, x2, x3, x4)
        t2, p2, df = cochrans_q(np.vstack([x1, x2, x3, x4]).T, return_object=False)
        assert pytest.approx(p1) == p2

    def test_cochranTest_tResult(self):
        x1 = [1, 0, 1, 0, 1]
        x2 = [1, 1, 1, 1, 1]
        x3 = [0, 0, 0, 0, 0]
        x4 = [0, 1, 0, 1, 0]
        t1, p1 = cochran_q_test(x1, x2, x3, x4)
        t2, p2, df = cochrans_q(np.vstack([x1, x2, x3, x4]).T, return_object=False)
        assert pytest.approx(t1) == t2


if __name__ == '__main__':
    unittest.main()