from StatsTest.sample_tests import *
import pytest
import unittest
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp
from statsmodels.sandbox.stats.runs import runstest_1samp


class TestSampleTest(unittest.TestCase):

    # One Sample Z Test

    def test_OneSampleZTest_popMeanString_Error(self):
        sample_data = [10, 20, 30]
        pop_mean = 's'
        with pytest.raises(TypeError, match="Population mean is not of numeric type"):
            one_sample_z_test(sample_data, pop_mean)

    def test_OneSampleZTest_popMeanList_Error(self):
        sample_data = [10, 20, 30]
        pop_mean = [10, 20, 30]
        with pytest.raises(TypeError, match="Population mean is not of numeric type"):
            one_sample_z_test(sample_data, pop_mean)

    def test_OneSampleZTest_popMeanDict_Error(self):
        sample_data = [10, 20, 30]
        pop_mean = {'s', 10}
        with pytest.raises(TypeError, match="Population mean is not of numeric type"):
            one_sample_z_test(sample_data, pop_mean)

    def test_OneSampleZTest_alternativeInt_Error(self):
        sample_data = [10, 20, 30]
        pop_mean = 10
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            one_sample_z_test(sample_data, pop_mean, alternative=10)

    def test_OneSampleZTest_alternativeWrong_Error(self):
        sample_data = [10, 20, 30]
        pop_mean = 10
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            one_sample_z_test(sample_data, pop_mean, alternative='higher')

    def test_OneSampleZTest_tooFewObservations_Error(self):
        sample_data = [10, 20, 30]
        pop_mean = 10
        with pytest.raises(AttributeError, match="Too few observations for z-test to be reliable, use t-test instead"):
            one_sample_z_test(sample_data, pop_mean)

    def test_OneSampleZTest_pResult(self):
        sample_data = np.arange(30)
        pop_mean = 5
        assert pytest.approx(0.0, 0.01) == one_sample_z_test(sample_data, pop_mean)[1]

    def test_OneSampleZTest_zResult(self):
        sample_data = np.arange(30)
        pop_mean = 5
        assert pytest.approx(11.389, 0.01) == one_sample_z_test(sample_data, pop_mean)[0]

    # Two Sample Z Test

    def test_TwoSampleZTest_alternativeInt_Error(self):
        sample_data = [10, 20, 30]
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            two_sample_z_test(sample_data, sample_data, alternative=10)

    def test_TwoSampleZTest_alternativeWrong_Error(self):
        sample_data = [10, 20, 30]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            two_sample_z_test(sample_data, sample_data, alternative='more')

    def test_TwoSampleZTest_tooFewObservations_Error(self):
        sample_data = [10, 20, 30]
        with pytest.raises(AttributeError, match="Too few observations for z-test to be reliable, use t-test instead"):
            two_sample_z_test(sample_data, sample_data)

    def test_TwoSampleZTest_pResult(self):
        sample_data = np.arange(30)
        assert pytest.approx(1.0, 0.01) == two_sample_z_test(sample_data, sample_data)[1]

    def test_TwoSampleZTest_zResult(self):
        sample_data = np.arange(30)
        assert pytest.approx(0.0, 0.01) == two_sample_z_test(sample_data, sample_data)[0]

    # One Sample T Test

    def test_OneSampleTTest_popMeanString_Error(self):
        sample_data = [10, 20, 30]
        pop_mean = 's'
        with pytest.raises(TypeError, match="Population mean is not of numeric type"):
            one_sample_t_test(sample_data, pop_mean)

    def test_OneSampleTTest_popMeanList_Error(self):
        sample_data = [10, 20, 30]
        pop_mean = [10, 20, 30]
        with pytest.raises(TypeError, match="Population mean is not of numeric type"):
            one_sample_t_test(sample_data, pop_mean)

    def test_OneSampleTTest_popMeanDict_Error(self):
        sample_data = [10, 20, 30]
        pop_mean = {'s', 10}
        with pytest.raises(TypeError, match="Population mean is not of numeric type"):
            one_sample_t_test(sample_data, pop_mean)

    def test_OneSampleTTest_alternativeInt_Error(self):
        sample_data = [10, 20, 30]
        pop_mean = 10
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            one_sample_t_test(sample_data, pop_mean, alternative=10)

    def test_OneSampleTTest_alternativeWrong_Error(self):
        sample_data = [10, 20, 30]
        pop_mean = 10
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            one_sample_t_test(sample_data, pop_mean, alternative='higher')

    def test_OneSampleTTest_pResult(self):
        sample_data = np.random.normal(50, 25, 1000)
        pop_mean = 10
        t1, p1 = one_sample_t_test(sample_data, pop_mean)
        t2, p2 = ttest_1samp(sample_data, pop_mean)
        assert pytest.approx(p2) == p1

    def test_OneSampleTTest_tResult(self):
        sample_data = np.random.normal(25, 10, 1000)
        pop_mean = 10
        t1, p1 = one_sample_t_test(sample_data, pop_mean)
        t2, p2 = ttest_1samp(sample_data, pop_mean)
        assert pytest.approx(t2) == t1

    # Two Sample T Test

    def test_TwoSampleTTest_alternativeInt_Error(self):
        sample_data = [10, 20, 30]
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            two_sample_t_test(sample_data, sample_data, alternative=10)

    def test_TwoSampleTTest_alternativeWrong_Error(self):
        sample_data = [10, 20, 30]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            two_sample_t_test(sample_data, sample_data, alternative='more')

    def test_TwoSampleTTest_notPaired_Error(self):
        data_1 = [10, 20, 30]
        data_2 = [5, 10]
        with pytest.raises(AttributeError, match="The data types are not paired"):
            two_sample_t_test(data_1, data_2, paired=True)

    def test_TwoSampleTTest_paired_pResult(self):
        data_1 = np.random.normal(10, 50, 1000)
        data_2 = np.random.normal(100, 5, 1000)
        t1, p1 = two_sample_t_test(data_1, data_2, paired=True)
        t2, p2 = ttest_rel(data_1, data_2)
        assert pytest.approx(p2) == p1

    def test_TwoSampleTTest_paired_tResult(self):
        data_1 = np.random.normal(20, 30, 1000)
        data_2 = np.random.normal(50, 25, 1000)
        t1, p1 = two_sample_t_test(data_1, data_2, paired=True)
        t2, p2 = ttest_rel(data_1, data_2)
        assert pytest.approx(t2) == t1

    def test_TwoSampleTTest_pResult(self):
        data_1 = np.random.normal(500, 30, 1000)
        data_2 = np.random.normal(50, 25, 500)
        t1, p1 = two_sample_t_test(data_1, data_2)
        t2, p2 = ttest_ind(data_1, data_2, equal_var=False)
        assert pytest.approx(p2) == p1

    def test_TwoSampleTTest_tResult(self):
        data_1 = np.random.normal(20, 30, 1000)
        data_2 = np.random.normal(65, 37, 250)
        t1, p1 = two_sample_t_test(data_1, data_2)
        t2, p2 = ttest_ind(data_1, data_2, equal_var=False)
        assert pytest.approx(t2) == t1

    # Two Sample F Test

    def test_twoSampleFTest_alternativeNotString_Error(self):
        data_1 = [100, 200, 300]
        data_2 = [10, 20, 30]
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            two_sample_f_test(data_1, data_2, alternative=10)

    def test_twoSampleFTest_alternativeWrong_Error(self):
        data_1 = [100, 200, 300]
        data_2 = [10, 20]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            two_sample_f_test(data_1, data_2, alternative='moar')

    def test_twoSampleFTest_pResult(self):
        data_1 = [560, 530, 570, 490, 510, 550, 550, 530]
        data_2 = [600, 590, 590, 630, 610, 630]
        f, p = two_sample_f_test(data_1, data_2)
        assert pytest.approx(p, 0.0001) == 0.4263

    def test_twoSampleFTest_fResult(self):
        data_1 = [560, 530, 570, 490, 510, 550, 550, 530]
        data_2 = [600, 590, 590, 630, 610, 630]
        f, p = two_sample_f_test(data_1, data_2)
        assert pytest.approx(f, 0.0001) == 2.1163

    # Binomial Sign Test

    def test_BinomialSignTest_alternativeNotString_Error(self):
        n_success, n_failure, success_prob = [10, 20], [30, 40], 0.5
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            binomial_sign_test(n_success, n_failure, alternative=10, success_prob=success_prob)

    def test_BinomialSignTest_alternativeWrong_Error(self):
        n_success, n_failure, success_prob = [10, 20], [30, 40], 0.5
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            binomial_sign_test(n_success, n_failure, alternative='moar', success_prob=success_prob)

    def test_BinomialSignTest_successProbNotFloat_Error(self):
        n_success, n_failure, success_prob = [10, 20], [30, 40], "0.5"
        with pytest.raises(TypeError, match="Probability of success needs to be a decimal value"):
            binomial_sign_test(n_success, n_failure, success_prob=success_prob)

    def test_BinomialSignTest_successProbGreaterThanOne_Error(self):
        n_success, n_failure, success_prob = [10, 20], [30, 40], 1.5
        with pytest.raises(ValueError, match="Cannot calculate probability of success, needs to be between 0 and 1"):
            binomial_sign_test(n_success, n_failure, success_prob=success_prob)

    def test_BinomialSignTest_successProbLessThanZero_Error(self):
        n_success, n_failure, success_prob = [10, 20], [30, 40], -0.5
        with pytest.raises(ValueError, match="Cannot calculate probability of success, needs to be between 0 and 1"):
            binomial_sign_test(n_success, n_failure, success_prob=success_prob)

    def test_BinomialSignTest_pResult(self):
        deer_hind = [142, 140, 144, 144, 142, 146, 149, 150, 142, 148]
        deer_fore = [138, 136, 147, 139, 143, 141, 143, 145, 136, 146]
        assert pytest.approx(0.109375) == binomial_sign_test(deer_hind, deer_fore)

    # Wald-Wolfowitz Test

    def test_WaldWolfowitzTest_cutoffWrong_Error(self):
        x = np.arange(100)
        with pytest.raises(ValueError, match="Cannot determine cutoff point"):
            wald_wolfowitz_test(x, cutoff="moar")

    def test_WaldWolfowitzTest_expectedWrongLength_Error(self):
        x = np.arange(100)
        expected = np.power(np.arange(99), 2)
        with pytest.raises(AttributeError, match="Cannot perform Wald-Wolfowitz with unequal array lengths"):
            wald_wolfowitz_test(x, expected)

    def test_WaldWolfowitzTest_pResultMedian(self):
        x = np.random.randint(0, 100, 50)
        x1, p1 = wald_wolfowitz_test(x)
        x2, p2 = runstest_1samp(x, cutoff='median', correction=False)
        assert pytest.approx(p2) == p1

    def test_WaldWolfowitzTest_xResultMedian(self):
        x = np.random.randint(0, 100, 50)
        x1, p1 = wald_wolfowitz_test(x)
        x2, p2 = runstest_1samp(x, cutoff='median', correction=False)
        assert pytest.approx(x2) == x1

    def test_WaldWolfowitzTest_pResultMean(self):
        x = np.random.randint(0, 100, 50)
        x1, p1 = wald_wolfowitz_test(x, cutoff='mean')
        x2, p2 = runstest_1samp(x, cutoff='mean', correction=False)
        assert pytest.approx(p2) == p1

    def test_WaldWolfowitzTest_xResultMean(self):
        x = np.random.randint(0, 100, 50)
        x1, p1 = wald_wolfowitz_test(x, cutoff='mean')
        x2, p2 = runstest_1samp(x, cutoff='mean', correction=False)
        assert pytest.approx(x2) == x1


if __name__ == '__main__':
    unittest.main()