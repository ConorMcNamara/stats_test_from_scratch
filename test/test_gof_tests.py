from StatsTest.gof_tests import *
import unittest
import pytest
from scipy.stats import chisquare, power_divergence
from statsmodels.stats.stattools import jarque_bera


class TestGOFTests(unittest.TestCase):

    # Chi Square Test for Goodness of Fit

    def test_chiGoodnessOfFit_pResult(self):
        observed = [10, 20, 30, 40]
        expected = [20, 20, 20, 20]
        x1, p1 = chi_goodness_of_fit_test(observed, expected)
        x2, p2 = chisquare(observed, expected)
        assert pytest.approx(p2) == p1

    def test_chiGoodnessOfFit_xResult(self):
        observed = [10, 20, 30, 40]
        expected = [20, 20, 20, 20]
        x1, p1 = chi_goodness_of_fit_test(observed, expected)
        x2, p2 = chisquare(observed, expected)
        assert pytest.approx(x2) == x1

    # G Test for Goodness of Fit

    def test_gGoodnessOfFit_pResult(self):
        observed = [10, 20, 30, 40]
        expected = [20, 20, 20, 20]
        x1, p1 = g_goodness_of_fit_test(observed, expected)
        x2, p2 = power_divergence(observed, expected, lambda_='log-likelihood')
        assert pytest.approx(p2) == p1

    def test_gGoodnessOfFit_xResult(self):
        observed = [10, 20, 30, 40]
        expected = [20, 20, 20, 20]
        x1, p1 = g_goodness_of_fit_test(observed, expected)
        x2, p2 = power_divergence(observed, expected, lambda_='log-likelihood')
        assert pytest.approx(x2) == x1

    # Jarque-Bera Test

    def test_jarqueBera_pResult(self):
        data = np.random.normal(0, 100, 1000)
        j1, p1 = jarque_bera_test(data)
        j2, p2, skew, kurtosis = jarque_bera(data)
        assert pytest.approx(p2) == p1

    def test_jarqueBera_jResult(self):
        data = np.random.normal(0, 100, 1000)
        j1, p1 = jarque_bera_test(data)
        j2, p2, skew, kurtosis = jarque_bera(data)
        assert pytest.approx(j2) == j1


if __name__ == '__main__':
    unittest.main()