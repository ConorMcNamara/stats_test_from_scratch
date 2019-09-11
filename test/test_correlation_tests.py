import pytest
import unittest
from StatsTest.correlation_tests import *
import numpy as np
from scipy.stats import pearsonr, spearmanr


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


if __name__ == '__main__':
    unittest.main()