import numpy as np
import pytest
from scipy.stats import chisquare, kurtosistest, normaltest, power_divergence, shapiro, skewtest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

from StatsTest.gof_tests import (
    box_pierce_test,
    chi_goodness_of_fit_test,
    g_goodness_of_fit_test,
    jarque_bera_test,
    k_squared_test,
    kurtosis_test,
    lilliefors_test,
    ljung_box_test,
    shapiro_wilk_test,
    skew_test,
)


class TestGOFTests:
    # Shapiro-Wilk Test

    def test_shapiroWilk_nLess3(self) -> None:
        data = [1, 2]
        with pytest.raises(AttributeError, match="Cannot run Shapiro-Wilks Test with less than 3 datapoints"):
            shapiro_wilk_test(data)

    def test_shapiroWilk_result(self) -> None:
        data = np.random.randint(0, 50, 100)
        w1, p1 = shapiro_wilk_test(data)
        w2, p2 = shapiro(data)
        assert pytest.approx(p2) == p1
        assert pytest.approx(w2) == w1

    # Chi Square Test for Goodness of Fit

    def test_chiGoodnessOfFit_result(self) -> None:
        observed = [10, 20, 30, 40]
        expected = [20, 20, 20, 20]
        x1, p1 = chi_goodness_of_fit_test(observed, expected)
        x2, p2 = chisquare(observed, expected)
        assert pytest.approx(p2) == p1
        assert pytest.approx(x2) == x1

    # G Test for Goodness of Fit

    def test_gGoodnessOfFit_result(self) -> None:
        observed = [10, 20, 30, 40, 10]
        expected = [20, 20, 20, 20, 20]
        x1, p1 = g_goodness_of_fit_test(observed, expected)
        x2, p2 = power_divergence(observed, expected, lambda_="log-likelihood")
        assert pytest.approx(p2) == p1
        assert pytest.approx(x2) == x1

    # Jarque-Bera Test

    def test_jarqueBera_result(self) -> None:
        data = np.random.normal(0, 100, 1000)
        j1, p1 = jarque_bera_test(data)
        j2, p2, skew, kurtosis = jarque_bera(data)
        assert pytest.approx(p2) == p1
        assert pytest.approx(j2) == j1

    # Ljung-Box Test

    def test_ljungBoxTest_lagsWrong_Error(self) -> None:
        data = np.random.normal(0, 100, 1000)
        with pytest.raises(ValueError, match="Cannot discern number of lags"):
            ljung_box_test(data, num_lags="s")

    def test_ljungBoxTest_result(self) -> None:
        data = np.random.normal(0, 100, 1000)
        num_lags = np.arange(1, 11)
        q1, p1 = ljung_box_test(data, num_lags=num_lags)
        q2, p2 = acorr_ljungbox(data, num_lags)
        assert pytest.approx(p2[len(p2) - 1]) == p1
        assert pytest.approx(q2[len(q2) - 1]) == q1

    # Box-Pierce Test

    def test_boxPierceTest_lagsWrong_Error(self) -> None:
        data = np.random.normal(0, 100, 1000)
        with pytest.raises(ValueError, match="Cannot discern number of lags"):
            box_pierce_test(data, num_lags="s")

    def test_boxPierceTest_result(self) -> None:
        data = np.random.normal(0, 100, 1000)
        num_lags = np.arange(1, 11)
        q1, p1 = box_pierce_test(data, num_lags=num_lags)
        _1, _2, q2, p2 = acorr_ljungbox(data, num_lags, boxpierce=True)
        assert pytest.approx(p2[len(p2) - 1]) == p1
        assert pytest.approx(q2[len(q2) - 1]) == q1

    # Skew Test

    def test_skewTest_nLess8_Error(self) -> None:
        data = np.random.normal(0, 10, 7)
        with pytest.raises(AttributeError, match="Skew Test is not reliable on datasets with less than 8 observations"):
            skew_test(data)

    def test_skewTest_result(self) -> None:
        data = np.random.normal(0, 100, 1000)
        z1, p1 = skew_test(data)
        z2, p2 = skewtest(data)
        assert pytest.approx(p2) == p1
        assert pytest.approx(z2) == z1

    # Kurtosis Test

    def test_kurtosisTest_nLess20_Error(self) -> None:
        data = np.random.normal(0, 10, 19)
        with pytest.raises(
            AttributeError, match="Kurtosis Test is not reliable on datasets with less than 20 observations"
        ):
            kurtosis_test(data)

    def test_kurtosisTest_result(self) -> None:
        data = np.random.normal(0, 100, 1000)
        z1, p1 = kurtosis_test(data)
        z2, p2 = kurtosistest(data)
        assert pytest.approx(p2) == p1
        assert pytest.approx(z1) == z1

    # K-Squared Test

    def test_k2Test_result(self) -> None:
        data = np.random.normal(0, 100, 1000)
        z1, p1 = k_squared_test(data)
        z2, p2 = normaltest(data)
        assert pytest.approx(p2) == p1
        assert pytest.approx(z2) == z1

    # Lilliefors Test

    def test_lilliefors_nLessFour_Error(self) -> None:
        data = [1, 2, 3]
        with pytest.raises(AttributeError, match="Cannot perform Lilliefors Test on less than 4 observations"):
            lilliefors_test(data, alpha=0.05)

    def test_lilliefors_alphaWrong_Error(self) -> None:
        data = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Cannot determine alpha level for Lilleifors Test"):
            lilliefors_test(data, alpha=0.5)

    def test_lilliefors_result(self) -> None:
        data = [1.2, 1.6, 1.8, 1.9, 1.9, 2.0, 2.2, 2.6, 3.0, 3.5, 4.0, 4.8, 5.6, 6.6, 7.6]
        d, result = lilliefors_test(data, 0.05)
        assert pytest.approx(0.1875, 0.001) == d
        assert result is True


if __name__ == "__main__":
    pytest.main()
