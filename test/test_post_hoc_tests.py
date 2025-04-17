import numpy as np
import pytest
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from StatsTest.post_hoc_tests import tukey_range_test, dunnett_test, duncan_multiple_range_test


class TestPostHocTest:

    # Tukey Range Test

    def test_tukeyRangeTest_kLessTwo_Error(self) -> None:
        sample_data = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Need at least two groups to perform Tukey Range Test"):
            tukey_range_test(sample_data)

    def test_tukeyRangeTest_pResult(self) -> None:
        x1, x2, x3 = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]
        results = tukey_range_test(x1, x2, x3)
        model = pairwise_tukeyhsd(x1 + x2 + x3, groups=[0] * 5 + [1] * 5 + [2] * 5)
        p_vals = psturng(np.abs(model.meandiffs / model.std_pairs), len(model.groupsunique), model.df_total)
        for i in range(3):
            assert pytest.approx(p_vals[i]) == results[i][2]

    # Dunnett Test

    def test_dunnetTest_kLessTwo_Error(self) -> None:
        control = [10, 10, 10]
        d1 = [10, 20, 30]
        with pytest.raises(AttributeError, match="Cannot run Dunnett Test with less than two groups"):
            dunnett_test(control, 0.05, d1)

    def test_dunnettTest_alphaWrong_Error(self) -> None:
        control = [10, 10, 10]
        d1 = [10, 20, 30]
        with pytest.raises(ValueError, match="Alpha level not currently supported"):
            dunnett_test(control, 0.2, d1, d1)

    def test_dunnettTest_results(self) -> None:
        control = [55, 47, 48]
        p1 = [55, 64, 64]
        p2 = [55, 49, 52]
        p3 = [50, 44, 41]
        np.testing.assert_array_equal(dunnett_test(control, 0.05, p1, p2, p3), [True, False, False])

    def test_dunnettTest_moreResults(self) -> None:
        control = [51, 87, 50, 48, 79, 61, 53, 54]
        p1 = [82, 91, 92, 80, 52, 79, 73, 74]
        p2 = [79, 84, 74, 98, 63, 83, 85, 58]
        p3 = [85, 80, 65, 71, 67, 51, 63, 93]
        np.testing.assert_array_equal(dunnett_test(control, 0.05, p1, p2, p3), [True, True, False])

    # Duncan's New Multi-Range Test

    def test_duncanMultiRangeTest_kLessTwo_Error(self) -> None:
        d1 = [10, 20, 30]
        with pytest.raises(AttributeError, match="Cannot run Duncan Multi-Range Test with less than two groups"):
            duncan_multiple_range_test(0.05, d1)

    def test_duncanMultiRangeTest_alphaWrong_Error(self) -> None:
        d1 = [10, 20, 30]
        with pytest.raises(ValueError, match="Alpha level not currently supported"):
            duncan_multiple_range_test(0.2, d1, d1)

    def test_duncanMultiRangeTest_results(self) -> None:
        data_1, data_2, data_3 = [9, 14, 11], [20, 19, 23], [39, 38, 41]
        np.testing.assert_array_equal(duncan_multiple_range_test(0.05, data_1, data_2, data_3),
                                      [np.array([2, 0]), np.array([2, 1]), np.array([1, 0])])

    def test_duncanMultiRangeTest_moreResults(self) -> None:
        data_1, data_2, data_3, data_4, data_5 = [10, 10, 10, 10, 9], [15, 15, 15, 15, 17], [20, 20, 20, 20, 8], [
            22, 22, 22, 22, 20], [10, 10, 10, 10, 14]
        np.testing.assert_array_equal(duncan_multiple_range_test(0.05, data_1, data_2, data_3, data_4, data_5),
                                      [np.array([3, 0]), np.array([3, 4]), np.array([3, 1]), np.array([2, 0]),
                                       np.array([2, 4]), np.array([1, 0])])



if __name__ == '__main__':
    pytest.main()
