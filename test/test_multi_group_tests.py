import numpy as np
import pytest
from scipy.stats import levene, f_oneway, bartlett, median_test
from statsmodels.stats.contingency_tables import cochrans_q

from StatsTest.multi_group_tests import (
    levene_test,
    brown_forsythe_test,
    one_way_f_test,
    bartlett_test,
    cochran_q_test,
    jonckheere_trend_test,
    mood_median_test,
)


class TestMultiGroupTests:
    # Levene Test

    def test_leveneTest_kLessTwo_Error(self) -> None:
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform a Levene Test"):
            levene_test(sample_data)

    def test_leveneTest_result(self) -> None:
        data_1 = np.random.randint(0, 100, 10)
        data_2 = np.random.randint(500, 550, 10)
        data_3 = np.random.randint(0, 10, 10)
        data_4 = np.random.randint(0, 50, 10)
        w1, p1 = levene_test(data_1, data_2, data_3, data_4)
        w2, p2 = levene(data_1, data_2, data_3, data_4, center="mean")
        assert pytest.approx(p2) == p1
        assert pytest.approx(w2) == w1

    # Brown-Forsythe Test

    def test_brownForsytheTest_kLessTwo_Error(self) -> None:
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform a Brown-Forsythe Test"):
            brown_forsythe_test(sample_data)

    def test_brownForsytheTest_result(self) -> None:
        data_1 = np.random.randint(0, 100, 10)
        data_2 = np.random.randint(500, 550, 10)
        data_3 = np.random.randint(0, 10, 10)
        data_4 = np.random.randint(0, 50, 10)
        w1, p1 = brown_forsythe_test(data_1, data_2, data_3, data_4)
        w2, p2 = levene(data_1, data_2, data_3, data_4, center="median")
        assert pytest.approx(p2) == p1
        assert pytest.approx(w2) == w1

    # One Way F Test

    def test_oneWayFTest_kLessTwo_Error(self) -> None:
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform a one-way F Test"):
            one_way_f_test(sample_data)

    def test_oneWayFTest_pResult(self) -> None:
        data_1 = np.random.randint(0, 100, 10)
        data_2 = np.random.randint(500, 550, 10)
        data_3 = np.random.randint(0, 10, 10)
        data_4 = np.random.randint(0, 50, 10)
        f1, p1 = one_way_f_test(data_1, data_2, data_3, data_4)
        f2, p2 = f_oneway(data_1, data_2, data_3, data_4)
        assert pytest.approx(p2) == p1
        assert pytest.approx(f2) == f1

    # Bartlett Test

    def test_bartlettTest_kLessTwo_Error(self) -> None:
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform the Bartlett Test"):
            bartlett_test(sample_data)

    def test_bartlettTest_pResult(self) -> None:
        data_1 = np.random.randint(0, 100, 10)
        data_2 = np.random.randint(500, 550, 10)
        data_3 = np.random.randint(0, 10, 10)
        data_4 = np.random.randint(0, 50, 10)
        x1, p1 = bartlett_test(data_1, data_2, data_3, data_4)
        x2, p2 = bartlett(data_1, data_2, data_3, data_4)
        assert pytest.approx(p2) == p1
        assert pytest.approx(x2) == x1

    # Cochran Test

    def test_cochranTest_kLessThree_Error(self) -> None:
        sample_data = [0, 1, 0, 1]
        with pytest.raises(AttributeError, match="Cannot run Cochran's Q Test with less than 3 treatments"):
            cochran_q_test(sample_data, sample_data)

    def test_cochranTest_nonBinary_Error(self) -> None:
        sample_data = [1, 2, 3, 4]
        sample_data2 = [0, 1, 0, 1]
        with pytest.raises(AttributeError, match="Cochran's Q Test only works with binary variables"):
            cochran_q_test(sample_data, sample_data2, sample_data2)

    def test_cochranTest_pResult(self) -> None:
        x1, x2, x3, x4 = [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0]
        t1, p1 = cochran_q_test(x1, x2, x3, x4)
        t2, p2, df = cochrans_q(np.vstack([x1, x2, x3, x4]).T, return_object=False)
        assert pytest.approx(p1) == p2
        assert pytest.approx(t1) == t2

    # Jonckheere Trend Test

    def test_jonckheereTest_uLessTwo_Error(self) -> None:
        sample_data = [1, 2, 3]
        with pytest.raises(AttributeError, match="Cannot run Jonckheere Test with less than 2 groups"):
            jonckheere_trend_test(sample_data)

    def test_jonckheereTest_unevenSampleSize_Error(self) -> None:
        sample_data1 = [1, 2, 3]
        sample_data2 = [3, 4]
        with pytest.raises(
            AttributeError, match="Jonckheere Test requires that each group have the same number of observations"
        ):
            jonckheere_trend_test(sample_data1, sample_data2)

    def test_jonckheereTest_alternativeInt_Error(self) -> None:
        sample_data = [1, 2, 3]
        with pytest.raises(TypeError, match="Cannot have alternative hypothesis with non-string value"):
            jonckheere_trend_test(sample_data, sample_data, alternative=10)

    def test_jonckheereTest_alternativeTwoSided_Error(self) -> None:
        sample_data = [1, 2, 3]
        with pytest.raises(ValueError, match="Cannot discern alternative hypothesis"):
            jonckheere_trend_test(sample_data, sample_data, alternative="two-sided")

    def test_jonckheereTest_result(self) -> None:
        data_1, data_2, data_3 = [10, 12, 14, 16], [12, 18, 20, 22], [20, 25, 27, 30]
        z, p = jonckheere_trend_test(data_1, data_2, data_3)
        assert pytest.approx(0.0016454416431436192) == p
        assert pytest.approx(2.939, 0.001) == z

    # Median Test

    def test_medianTest_kLessTwo_Error(self) -> None:
        data_1 = [100, 200, 300]
        with pytest.raises(AttributeError, match="Cannot run Median Test with less than 2 groups"):
            mood_median_test(data_1)

    def test_medianTest_alternativeNotString_Error(self) -> None:
        data_1, data_2 = [100, 200, 300], [300, 400, 500]
        with pytest.raises(AttributeError):
            mood_median_test(data_1, data_2, alternative=10)

    def test_medianTest_alternativeWrong_Error(self) -> None:
        data_1, data_2 = [100, 200, 300], [300, 400, 500]
        with pytest.raises(ValueError, match="Cannot discern alternative hypothesis"):
            mood_median_test(data_1, data_2, alternative="moar")

    def test_medianTest_handleMedNotString_Error(self) -> None:
        data_1, data_2 = [100, 200, 300], [300, 400, 500]
        with pytest.raises(AttributeError):
            mood_median_test(data_1, data_2, handle_med=10)

    def test_medianTest_handleMedWrong_Error(self) -> None:
        data_1, data_2 = [100, 200, 300], [300, 400, 500]
        with pytest.raises(ValueError, match="Cannot discern how to handle median value"):
            mood_median_test(data_1, data_2, handle_med="moar")

    def test_medianTest_result(self) -> None:
        g1 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
        g2 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
        g3 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]
        x1, p1 = mood_median_test(g1, g2, g3, alternative="less")
        x2, p2, med, tbl = median_test(g1, g2, g3)
        assert pytest.approx(p2) == p1
        assert pytest.approx(x2) == x1


if __name__ == "__main__":
    pytest.main()
