import pytest
from scipy.stats import binomtest

from StatsTest.proportion_tests import (
    binomial_test,
    chi_square_proportion_test,
    g_proportion_test,
    one_sample_proportion_z_test,
    two_sample_proportion_z_test,
)


class TestProportionTest:
    # One Sample Proportion Test

    def test_OneSampleZProp_popNotFloat_Error(self) -> None:
        sample_data = [1, 0, 1, 0, 1, 0]
        pop_mean = 1
        with pytest.raises(TypeError, match="Population mean is not of float type"):
            one_sample_proportion_z_test(sample_data, pop_mean)

    def test_OneSampleZProp_popLessZero_Error(self) -> None:
        sample_data = [1, 0, 1, 0, 1, 0]
        pop_mean = -0.5
        with pytest.raises(ValueError, match="Population mean must be between 0 and 1"):
            one_sample_proportion_z_test(sample_data, pop_mean)

    def test_OneSampleZProp_popGreaterOne_Error(self) -> None:
        sample_data = [1, 0, 1, 0, 1, 0]
        pop_mean = 1.5
        with pytest.raises(ValueError, match="Population mean must be between 0 and 1"):
            one_sample_proportion_z_test(sample_data, pop_mean)

    def test_OneSampleZProp_nonBinary_Error(self) -> None:
        sample_data = [1, 2, 3, 4]
        pop_mean = 0.5
        with pytest.raises(AttributeError, match="Cannot perform a proportion test on non-binary data"):
            one_sample_proportion_z_test(sample_data, pop_mean)

    def test_OneSampleZProp_tooFewObs_Error(self) -> None:
        sample_data = [1, 0, 1, 0, 1, 0]
        pop_mean = 0.5
        with pytest.raises(AttributeError, match="Too few instances of success or failure to run proportion test"):
            one_sample_proportion_z_test(sample_data, pop_mean)

    def test_OneSampleZProp_result(self) -> None:
        sample_data = [0, 1] * 15
        pop_mean = 0.5
        z_val, p_val = one_sample_proportion_z_test(sample_data, pop_mean)
        assert pytest.approx(1.0, 0.01) == p_val
        assert pytest.approx(0, 0.01) == z_val

    # Two Sample Proportion Test

    def test_TwoSampleZProp_nonBinaryD1_Error(self) -> None:
        sample_data1 = [0, 1, 2]
        sample_data2 = [0, 1, 0, 1]
        with pytest.raises(AttributeError, match="Cannot perform a proportion test on non-binary data for data_1"):
            two_sample_proportion_z_test(sample_data1, sample_data2)

    def test_TwoSampleZProp_nonBinaryD2_Error(self) -> None:
        sample_data1 = [0, 1, 0, 1]
        sample_data2 = [0, 1, 2]
        with pytest.raises(AttributeError, match="Cannot perform a proportion test on non-binary data for data_2"):
            two_sample_proportion_z_test(sample_data1, sample_data2)

    def test_TwoSampleZProp_result(self) -> None:
        sample_data = [0, 1] * 15
        z_val, p_val = two_sample_proportion_z_test(sample_data, sample_data)
        assert pytest.approx(1.0, 0.01) == p_val
        assert pytest.approx(0.0, 0.01) == z_val

    # Binomial Test

    def test_BinomialTest_alternativeNotString_Error(self) -> None:
        n_success, n_failure, success_prob = 50, 100, 0.5
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            binomial_test(n_success, n_failure, alternative=10, success_prob=success_prob)

    def test_BinomialTest_alternativeWrong_Error(self) -> None:
        n_success, n_failure, success_prob = 50, 100, 0.5
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            binomial_test(n_success, n_failure, alternative="moar", success_prob=success_prob)

    def test_BinomialTest_successProbNotFloat_Error(self) -> None:
        n_success, n_failure, success_prob = 50, 100, "0.5"
        with pytest.raises(TypeError, match="Probability of success needs to be a decimal value"):
            binomial_test(n_success, n_failure, success_prob=success_prob)

    def test_BinomialTest_successProbGreaterThanOne_Error(self) -> None:
        n_success, n_failure, success_prob = 50, 100, 1.5
        with pytest.raises(ValueError, match="Cannot calculate probability of success, needs to be between 0 and 1"):
            binomial_test(n_success, n_failure, success_prob=success_prob)

    def test_BinomialTest_successProbLessThanZero_Error(self) -> None:
        n_success, n_failure, success_prob = 50, 100, -0.5
        with pytest.raises(ValueError, match="Cannot calculate probability of success, needs to be between 0 and 1"):
            binomial_test(n_success, n_failure, success_prob=success_prob)

    def test_BinomialTest_rightSide_pResult(self) -> None:
        n_success, n_failure, success_prob = 50, 100, 0.66
        p1 = binomial_test(n_success, n_failure, success_prob=success_prob)
        p2 = binomtest(n_success, n_success + n_failure, success_prob)
        assert pytest.approx(p2) == p1

    def test_BinomialTest_leftSide_pResult(self) -> None:
        n_success, n_failure, success_prob = 50, 100, 0.25
        p1 = binomial_test(n_success, n_failure, success_prob=success_prob)
        p2 = binomtest(n_success, n_success + n_failure, success_prob)
        assert pytest.approx(p2) == p1

    # Chi Square Proportion Test

    def test_chiSquareProportionTest_unequalTotalLength_Error(self) -> None:
        prob = [0.2, 0.4, 0.6]
        n_total = [100, 100]
        with pytest.raises(ValueError, match="Success probability and N Total are not of same length"):
            chi_square_proportion_test(prob, n_total)

    def test_chiSquareProportionsTest_unequalExpectedLength_Error(self) -> None:
        prob = [0.2, 0.4, 0.6]
        n_total = [100, 100, 100]
        expected = [0.3, 0.5]
        with pytest.raises(ValueError, match="Expected and Success probability are not of same length"):
            chi_square_proportion_test(prob, n_total, expected)

    def test_chiSquareProportionsTest_expectedProbGreaterOne_Error(self) -> None:
        prob = [0.2, 0.4, 0.6]
        n_total = [100, 100, 100]
        expected = [0.3, 0.5, 1.2]
        with pytest.raises(ValueError, match="Cannot have percentage of expected greater than 1"):
            chi_square_proportion_test(prob, n_total, expected)

    def test_chiSquareProportionsTest_expectedProbLessZero_Error(self) -> None:
        prob = [0.2, 0.4, 0.6]
        n_total = [100, 100, 100]
        expected = [0.3, 0.5, -0.2]
        with pytest.raises(ValueError, match="Cannot have negative percentage of expected"):
            chi_square_proportion_test(prob, n_total, expected)

    def test_chiSquareProportionsTest_probGreaterOne_Error(self) -> None:
        prob = [0.2, 0.4, 1.2]
        n_total = [100, 100, 100]
        expected = [0.3, 0.5, 0.7]
        with pytest.raises(ValueError, match="Cannot have percentage of success greater than 1"):
            chi_square_proportion_test(prob, n_total, expected)

    def test_chiSquareProportionsTest_probLessZero_Error(self) -> None:
        prob = [0.2, 0.4, -0.2]
        n_total = [100, 100, 100]
        expected = [0.3, 0.5, 0.7]
        with pytest.raises(ValueError, match="Cannot have negative percentage of success"):
            chi_square_proportion_test(prob, n_total, expected)

    def test_chiSquareProportionTest_result(self) -> None:
        prob = [0.41025641025, 0.53773584605]
        n_total = [156, 212]
        x1, p1 = chi_square_proportion_test(prob, n_total)
        assert pytest.approx(0.01559, 0.001) == p1
        assert pytest.approx(5.8481, 0.00001) == x1

    # G Proportion Test

    def test_gSquareProportionTest_unequalTotalLength_Error(self) -> None:
        prob = [0.2, 0.4, 0.6]
        n_total = [100, 100]
        with pytest.raises(ValueError, match="Success probability and N Total are not of same length"):
            g_proportion_test(prob, n_total)

    def test_gSquareProportionsTest_unequalExpectedLength_Error(self) -> None:
        prob = [0.2, 0.4, 0.6]
        n_total = [100, 100, 100]
        expected = [0.3, 0.5]
        with pytest.raises(ValueError, match="Expected and Success probability are not of same length"):
            g_proportion_test(prob, n_total, expected)

    def test_gSquareProportionsTest_expectedProbGreaterOne_Error(self) -> None:
        prob = [0.2, 0.4, 0.6]
        n_total = [100, 100, 100]
        expected = [0.3, 0.5, 1.2]
        with pytest.raises(ValueError, match="Cannot have percentage of expected greater than 1"):
            g_proportion_test(prob, n_total, expected)

    def test_gSquareProportionsTest_expectedProbLessZero_Error(self) -> None:
        prob = [0.2, 0.4, 0.6]
        n_total = [100, 100, 100]
        expected = [0.3, 0.5, -0.2]
        with pytest.raises(ValueError, match="Cannot have negative percentage of expected"):
            g_proportion_test(prob, n_total, expected)

    def test_gSquareProportionsTest_probGreaterOne_Error(self) -> None:
        prob = [0.2, 0.4, 1.2]
        n_total = [100, 100, 100]
        expected = [0.3, 0.5, 0.7]
        with pytest.raises(ValueError, match="Cannot have percentage of success greater than 1"):
            g_proportion_test(prob, n_total, expected)

    def test_gSquareProportionsTest_probLessZero_Error(self) -> None:
        prob = [0.2, 0.4, -0.2]
        n_total = [100, 100, 100]
        expected = [0.3, 0.5, 0.7]
        with pytest.raises(ValueError, match="Cannot have negative percentage of success"):
            g_proportion_test(prob, n_total, expected)

    def test_gSquareProportionTest_result(self) -> None:
        prob = [0.41025641025, 0.53773584605]
        n_total = [156, 212]
        x1, p1 = g_proportion_test(prob, n_total)
        assert pytest.approx(0.0154, 0.001) == p1
        assert pytest.approx(5.8703, 0.00001) == x1


if __name__ == "__main__":
    pytest.main()
