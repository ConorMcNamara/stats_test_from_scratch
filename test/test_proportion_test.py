import unittest
import pytest
from StatsTest.proportion_test import *
from scipy.stats import binom_test


class TestProportionTest(unittest.TestCase):

    # One Sample Proportion Test

    def test_OneSampleZProp_popNotFloat_Error(self):
        sample_data = [1, 0, 1, 0, 1, 0]
        pop_mean = 1
        with pytest.raises(TypeError, match="Population mean is not of float type"):
            one_sample_proportion_z_test(sample_data, pop_mean)

    def test_OneSampleZProp_popLessZero_Error(self):
        sample_data = [1, 0, 1, 0, 1, 0]
        pop_mean = -0.5
        with pytest.raises(ValueError, match="Population mean must be between 0 and 1"):
            one_sample_proportion_z_test(sample_data, pop_mean)

    def test_OneSampleZProp_popGreaterOne_Error(self):
        sample_data = [1, 0, 1, 0, 1, 0]
        pop_mean = 1.5
        with pytest.raises(ValueError, match="Population mean must be between 0 and 1"):
            one_sample_proportion_z_test(sample_data, pop_mean)

    def test_OneSampleZProp_nonBinary_Error(self):
        sample_data = [1, 2, 3, 4]
        pop_mean = 0.5
        with pytest.raises(AttributeError, match='Cannot perform a proportion test on non-binary data'):
            one_sample_proportion_z_test(sample_data, pop_mean)

    def test_OneSampleZProp_tooFewObs_Error(self):
        sample_data = [1, 0, 1, 0, 1, 0]
        pop_mean = 0.5
        with pytest.raises(AttributeError, match="Too few instances of success or failure to run proportion test"):
            one_sample_proportion_z_test(sample_data, pop_mean)

    def test_OneSampleZProp_pResult(self):
        sample_data = [0, 1] * 15
        pop_mean = 0.5
        assert pytest.approx(1.0, 0.01) == one_sample_proportion_z_test(sample_data, pop_mean)[1]

    def test_OneSampleZProp_zResult(self):
        sample_data = [0, 1] * 15
        pop_mean = 0.5
        assert pytest.approx(0.0, 0.01) == one_sample_proportion_z_test(sample_data, pop_mean)[0]

    # Two Sample Proportion Test

    def test_TwoSampleZProp_nonBinaryD1_Error(self):
        sample_data1 = [0, 1, 2]
        sample_data2 = [0, 1, 0, 1]
        with pytest.raises(AttributeError, match="Cannot perform a proportion test on non-binary data for data_1"):
            two_sample_proportion_z_test(sample_data1, sample_data2)

    def test_TwoSampleZProp_nonBinaryD2_Error(self):
        sample_data1 = [0, 1, 0, 1]
        sample_data2 = [0, 1, 2]
        with pytest.raises(AttributeError, match="Cannot perform a proportion test on non-binary data for data_2"):
            two_sample_proportion_z_test(sample_data1, sample_data2)

    def test_TwoSampleZProp_pResult(self):
        sample_data = [0, 1] * 15
        assert pytest.approx(1.0, 0.01) == two_sample_proportion_z_test(sample_data, sample_data)[1]

    def test_TwoSampleZProp_zResult(self):
        sample_data = [0, 1] * 15
        assert pytest.approx(0.0, 0.01) == two_sample_proportion_z_test(sample_data, sample_data)[0]

    # Binomial Test

    def test_BinomialTest_alternativeNotString_Error(self):
        n_success, n_failure, success_prob = 50, 100, 0.5
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            binomial_test(n_success, n_failure, alternative=10, success_prob=success_prob)

    def test_BinomialTest_alternativeWrong_Error(self):
        n_success, n_failure, success_prob = 50, 100, 0.5
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            binomial_test(n_success, n_failure, alternative='moar', success_prob=success_prob)

    def test_BinomialTest_successProbNotFloat_Error(self):
        n_success, n_failure, success_prob = 50, 100, "0.5"
        with pytest.raises(TypeError, match="Probability of success needs to be a decimal value"):
            binomial_test(n_success, n_failure, success_prob=success_prob)

    def test_BinomialTest_successProbGreaterThanOne_Error(self):
        n_success, n_failure, success_prob = 50, 100, 1.5
        with pytest.raises(ValueError, match="Cannot calculate probability of success, needs to be between 0 and 1"):
            binomial_test(n_success, n_failure, success_prob=success_prob)

    def test_BinomialTest_successProbLessThanZero_Error(self):
        n_success, n_failure, success_prob = 50, 100, -0.5
        with pytest.raises(ValueError, match="Cannot calculate probability of success, needs to be between 0 and 1"):
            binomial_test(n_success, n_failure, success_prob=success_prob)

    def test_BinomialTest_rightSide_pResult(self):
        n_success, n_failure, success_prob = 50, 100, 0.66
        p1 = binomial_test(n_success, n_failure, success_prob=success_prob)
        p2 = binom_test(n_success, n_success + n_failure, success_prob)
        assert pytest.approx(p2) == p1

    def test_BinomialTest_leftSide_pResult(self):
        n_success, n_failure, success_prob = 50, 100, 0.25
        p1 = binomial_test(n_success, n_failure, success_prob=success_prob)
        p2 = binom_test(n_success, n_success + n_failure, success_prob)
        assert pytest.approx(p2) == p1
