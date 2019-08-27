from StatsTest.gof_tests import *
import unittest
import pytest
from scipy.stats import chisquare, power_divergence, normaltest, kurtosistest, skewtest
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox


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

    # Ljung-Box Test
    def test_ljungBoxTest_lagsWrong_Error(self):
        data = np.random.normal(0, 100, 1000)
        with pytest.raises(ValueError, match="Cannot discern number of lags"):
            ljung_box_test(data, num_lags="s")

    def test_ljungBoxTest_pResult(self):
        data = np.random.normal(0, 100, 1000)
        num_lags = np.arange(1, 11)
        q1, p1 = ljung_box_test(data, num_lags=num_lags)
        q2, p2 = acorr_ljungbox(data, num_lags)
        assert pytest.approx(p2[len(p2) - 1]) == p1

    def test_ljungBoxTest_qResult(self):
        data = np.random.normal(0, 100, 1000)
        num_lags = np.arange(1, 6)
        q1, p1 = ljung_box_test(data, num_lags=num_lags)
        q2, p2 = acorr_ljungbox(data, num_lags)
        assert pytest.approx(q2[len(q2) - 1]) == q1

    # Box-Pierce Test
    def test_boxPierceTest_lagsWrong_Error(self):
        data = np.random.normal(0, 100, 1000)
        with pytest.raises(ValueError, match="Cannot discern number of lags"):
            box_pierce_test(data, num_lags="s")

    def test_boxPierceTest_pResult(self):
        data = np.random.normal(0, 100, 1000)
        num_lags = np.arange(1, 11)
        q1, p1 = box_pierce_test(data, num_lags=num_lags)
        _1, _2, q2, p2 = acorr_ljungbox(data, num_lags, boxpierce=True)
        assert pytest.approx(p2[len(p2) - 1]) == p1

    def test_boxPierceTest_qResult(self):
        data = np.random.normal(0, 100, 1000)
        num_lags = np.arange(1, 6)
        q1, p1 = box_pierce_test(data, num_lags=num_lags)
        _1, _2, q2, p2 = acorr_ljungbox(data, num_lags, boxpierce=True)
        assert pytest.approx(q2[len(q2) - 1]) == q1

    # Skew Test

    def test_skewTest_nLess8_Error(self):
        data = np.random.normal(0, 10, 7)
        with pytest.raises(AttributeError, match="Skew Test is not reliable on datasets with less than 8 observations"):
            skew_test(data)

    def test_skewTest_pResult(self):
        data = np.random.normal(0, 100, 1000)
        z1, p1 = skew_test(data)
        z2, p2 = skewtest(data)
        assert pytest.approx(p2) == p1

    def test_skewTest_zResult(self):
        data = np.random.normal(0, 100, 1000)
        z1, p1 = skew_test(data)
        z2, p2 = skewtest(data)
        assert pytest.approx(z2) == z1

    # Kurtosis Test

    def test_kurtosisTest_nLess20_Error(self):
        data = np.random.normal(0, 10, 19)
        with pytest.raises(AttributeError, match='Kurtosis Test is not reliable on datasets with less than 20 observations'):
            kurtosis_test(data)

    def test_kurtosisTest_pResult(self):
        data = np.random.normal(0, 100, 1000)
        z1, p1 = kurtosis_test(data)
        z2, p2 = kurtosistest(data)
        assert pytest.approx(p2) == p1

    def test_kurtosisTest_zResult(self):
        data = np.random.normal(0, 100, 1000)
        z1, p1 = kurtosis_test(data)
        z2, p2 = kurtosistest(data)
        assert pytest.approx(p2) == p1

    # K-Squared Test

    def test_k2Test_pResult(self):
        data = np.random.normal(0, 100, 1000)
        z1, p1 = k_squared_test(data)
        z2, p2 = normaltest(data)
        assert pytest.approx(p2) == p1

    def test_k2Test_zResult(self):
        data = np.random.normal(0, 100, 1000)
        z1, p1 = k_squared_test(data)
        z2, p2 = normaltest(data)
        assert pytest.approx(p2) == p1


if __name__ == '__main__':
    unittest.main()