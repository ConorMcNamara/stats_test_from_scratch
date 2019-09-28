import pytest
import unittest
from StatsTest.correlation_tests import *
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, pointbiserialr


class TestCorrelationTests(unittest.TestCase):

    # Pearson Test

    def test_pearsonTest_wrongLength_Error(self):
        x = [1, 2, 3, 4]
        y = [1, 2, 3]
        with pytest.raises(ValueError, match="Cannot calculate correlation with datasets of different lengths"):
            pearson_test(x, y)

    def test_pearsonTest_pResult(self):
        x = np.random.normal(0, 100, 1000)
        y = np.random.normal(100, 50, 1000)
        r1, p1 = pearson_test(x, y)
        r2, p2 = pearsonr(x, y)
        assert pytest.approx(p2) == p1

    def test_pearsonTest_rhoResult(self):
        x = np.random.normal(0, 100, 1000)
        y = np.random.normal(100, 50, 1000)
        r1, p1 = pearson_test(x, y)
        r2, p2 = pearsonr(x, y)
        assert pytest.approx(r2) == r1

    # Spearman Rank Test

    def test_spearmanTest_wrongLength_Error(self):
        x = [1, 2, 3, 4]
        y = [1, 2, 3]
        with pytest.raises(ValueError, match="Cannot calculate correlation with datasets of different lengths"):
            spearman_test(x, y)

    def test_spearmanTest_pResult(self):
        x = np.random.normal(0, 100, 1000)
        y = np.random.normal(100, 50, 1000)
        r1, p1 = spearman_test(x, y)
        r2, p2 = spearmanr(x, y)
        assert pytest.approx(p2) == p1

    def test_spearmanTest_rhoResult(self):
        x = np.random.normal(0, 100, 1000)
        y = np.random.normal(100, 50, 1000)
        r1, p1 = spearman_test(x, y)
        r2, p2 = spearmanr(x, y)
        assert pytest.approx(r2) == r1

    # Kendall Tau Test

    def test_kendallTau_unevenLengths_Error(self):
        x = np.random.randint(0, 100, 20)
        y = np.random.randint(0, 100, 15)
        with pytest.raises(ValueError, match="Cannot calculate correlation with datasets of different lengths"):
            kendall_tau_test(x, y)

    def test_kendallTau_wrongMethod_Error(self):
        x = np.random.randint(0, 100, 20)
        with pytest.raises(ValueError, match="Cannot determine type of test for Kendall Tau"):
            kendall_tau_test(x, x, method="moar")

    def test_kendallTau_exactTies_Error(self):
        x1 = [12, 2, 1, 12, 2]
        x2 = [1, 4, 7, 1, 0]
        with pytest.raises(AttributeError, match="Cannot run exact test when ties are present"):
            kendall_tau_test(x1, x2, method='exact')

    def test_kendallTau_exact_pResult(self):
        x = [4, 10, 3, 1, 9, 2, 6, 7, 8, 5]
        y = [5, 8, 6, 2, 10, 3, 9, 4, 7, 1]
        t1, p1 = kendall_tau_test(x, y, method='exact')
        t2, p2 = kendalltau(x, y, method='exact')
        assert pytest.approx(p2) == p1

    def test_kendallTau_exact_tResult(self):
        x = [4, 10, 3, 1, 9, 2, 6, 7, 8, 5]
        y = [5, 8, 6, 2, 10, 3, 9, 4, 7, 1]
        t1, p1 = kendall_tau_test(x, y, method='exact')
        t2, p2 = kendalltau(x, y, method='exact')
        assert pytest.approx(t2) == t1

    def test_kendallTau_ties_pResult(self):
        x1 = [12, 2, 1, 12, 2]
        x2 = [1, 4, 7, 1, 0]
        t1, p1 = kendall_tau_test(x1, x2, method='significance')
        t2, p2 = kendalltau(x1, x2, method='asymptotic')
        assert pytest.approx(p2) == p1

    def test_kendallTau_ties_tResult(self):
        x1 = [12, 2, 1, 12, 2]
        x2 = [1, 4, 7, 1, 0]
        t1, p1 = kendall_tau_test(x1, x2, method='significance')
        t2, p2 = kendalltau(x1, x2, method='asymptotic')
        assert pytest.approx(t2) == t1

    # Point Biserial Test

    def test_BiserialCorrelation_pointWrong_Error(self):
        a = np.array([0, 0, 0, 1, 1, 1, 1])
        b = np.arange(7)
        with pytest.raises(ValueError, match="Cannot discern method for biserial correlation test"):
            biserial_correlation_test(b, a, "moar")

    def test_BiserialCorrelation_tooManyGroups_Error(self):
        a = np.array([0, 1, 2, 0, 1, 2])
        b = np.arange(6)
        with pytest.raises(AttributeError, match="Need to have two groupings for biseral correlation"):
            biserial_correlation_test(b, a, "point")

    def test_BiserialCorrelationPoint_pResult(self):
        a = np.array([0, 0, 0, 1, 1, 1, 1])
        b = np.arange(7)
        r1, p1 = biserial_correlation_test(b, a, 'point')
        r2, p2 = pointbiserialr(a, b)
        assert pytest.approx(p2) == p1

    def test_BiserialCorrelationPoint_rResult(self):
        a = np.array([0, 0, 0, 1, 1, 1, 1])
        b = np.arange(7)
        r1, p1 = biserial_correlation_test(b, a, 'point')
        r2, p2 = pointbiserialr(a, b)
        assert pytest.approx(r2) == r1


if __name__ == '__main__':
    unittest.main()