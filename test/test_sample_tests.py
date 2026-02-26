import numpy as np
import pytest
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp
from statsmodels.sandbox.stats.runs import runstest_1samp

from StatsTest.sample_tests import (
    one_sample_z_test,
    one_sample_t_test,
    two_sample_t_test,
    two_sample_z_test,
    two_sample_f_test,
    trinomial_test,
    trimmed_means_test,
    yuen_welch_test,
    binomial_sign_test,
    wald_wolfowitz_test,
    fligner_policello_test,
)


class TestSampleTest:
    # One Sample Z Test

    def test_OneSampleZTest_popMeanString_Error(self) -> None:
        sample_data = [10, 20, 30]
        pop_mean = "s"
        with pytest.raises(TypeError, match="Population mean is not of numeric type"):
            one_sample_z_test(sample_data, pop_mean)

    def test_OneSampleZTest_popMeanList_Error(self) -> None:
        sample_data = [10, 20, 30]
        pop_mean = [10, 20, 30]
        with pytest.raises(TypeError, match="Population mean is not of numeric type"):
            one_sample_z_test(sample_data, pop_mean)

    def test_OneSampleZTest_popMeanDict_Error(self) -> None:
        sample_data = [10, 20, 30]
        pop_mean = {"s", 10}
        with pytest.raises(TypeError, match="Population mean is not of numeric type"):
            one_sample_z_test(sample_data, pop_mean)

    def test_OneSampleZTest_alternativeInt_Error(self) -> None:
        sample_data = [10, 20, 30]
        pop_mean = 10
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            one_sample_z_test(sample_data, pop_mean, alternative=10)

    def test_OneSampleZTest_alternativeWrong_Error(self) -> None:
        sample_data = [10, 20, 30]
        pop_mean = 10
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            one_sample_z_test(sample_data, pop_mean, alternative="higher")

    def test_OneSampleZTest_tooFewObservations_Error(self) -> None:
        sample_data = [10, 20, 30]
        pop_mean = 10
        with pytest.raises(AttributeError, match="Too few observations for z-test to be reliable, use t-test instead"):
            one_sample_z_test(sample_data, pop_mean)

    def test_OneSampleZTest_result(self) -> None:
        sample_data = np.arange(30)
        pop_mean = 5
        z_val, p_val = one_sample_z_test(sample_data, pop_mean)
        assert pytest.approx(0.0, 0.01) == p_val
        assert pytest.approx(11.389, 0.01) == z_val

    # Two Sample Z Test

    def test_TwoSampleZTest_alternativeInt_Error(self) -> None:
        sample_data = [10, 20, 30]
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            two_sample_z_test(sample_data, sample_data, alternative=10)

    def test_TwoSampleZTest_alternativeWrong_Error(self) -> None:
        sample_data = [10, 20, 30]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            two_sample_z_test(sample_data, sample_data, alternative="more")

    def test_TwoSampleZTest_tooFewObservations_Error(self) -> None:
        sample_data = [10, 20, 30]
        with pytest.raises(AttributeError, match="Too few observations for z-test to be reliable, use t-test instead"):
            two_sample_z_test(sample_data, sample_data)

    def test_TwoSampleZTest_result(self) -> None:
        sample_data = np.arange(30)
        z_val, p_val = two_sample_z_test(sample_data, sample_data)
        assert pytest.approx(1.0, 0.01) == p_val
        assert pytest.approx(0.0, 0.01) == z_val

    # One Sample T Test

    def test_OneSampleTTest_popMeanString_Error(self) -> None:
        sample_data = [10, 20, 30]
        pop_mean = "s"
        with pytest.raises(TypeError, match="Population mean is not of numeric type"):
            one_sample_t_test(sample_data, pop_mean)

    def test_OneSampleTTest_popMeanList_Error(self) -> None:
        sample_data = [10, 20, 30]
        pop_mean = [10, 20, 30]
        with pytest.raises(TypeError, match="Population mean is not of numeric type"):
            one_sample_t_test(sample_data, pop_mean)

    def test_OneSampleTTest_popMeanDict_Error(self) -> None:
        sample_data = [10, 20, 30]
        pop_mean = {"s", 10}
        with pytest.raises(TypeError, match="Population mean is not of numeric type"):
            one_sample_t_test(sample_data, pop_mean)

    def test_OneSampleTTest_alternativeInt_Error(self) -> None:
        sample_data = [10, 20, 30]
        pop_mean = 10
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            one_sample_t_test(sample_data, pop_mean, alternative=10)

    def test_OneSampleTTest_alternativeWrong_Error(self) -> None:
        sample_data = [10, 20, 30]
        pop_mean = 10
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            one_sample_t_test(sample_data, pop_mean, alternative="higher")

    def test_OneSampleTTest_result(self) -> None:
        sample_data = np.random.normal(50, 25, 1000)
        pop_mean = 10
        t1, p1 = one_sample_t_test(sample_data, pop_mean)
        t2, p2 = ttest_1samp(sample_data, pop_mean)
        assert pytest.approx(p2) == p1
        assert pytest.approx(t2) == t1

    # Two Sample T Test

    def test_TwoSampleTTest_alternativeInt_Error(self) -> None:
        sample_data = [10, 20, 30]
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            two_sample_t_test(sample_data, sample_data, alternative=10)

    def test_TwoSampleTTest_alternativeWrong_Error(self) -> None:
        sample_data = [10, 20, 30]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            two_sample_t_test(sample_data, sample_data, alternative="more")

    def test_TwoSampleTTest_notPaired_Error(self) -> None:
        data_1 = [10, 20, 30]
        data_2 = [5, 10]
        with pytest.raises(AttributeError, match="The data types are not paired"):
            two_sample_t_test(data_1, data_2, paired=True)

    def test_TwoSampleTTest_paired_pResult(self) -> None:
        data_1 = np.random.normal(10, 50, 1000)
        data_2 = np.random.normal(100, 5, 1000)
        t1, p1 = two_sample_t_test(data_1, data_2, paired=True)
        t2, p2 = ttest_rel(data_1, data_2)
        assert pytest.approx(p2) == p1
        assert pytest.approx(t2) == t1

    def test_TwoSampleTTest_pResult(self) -> None:
        data_1 = np.random.normal(500, 30, 1000)
        data_2 = np.random.normal(50, 25, 500)
        t1, p1 = two_sample_t_test(data_1, data_2)
        t2, p2 = ttest_ind(data_1, data_2, equal_var=False)
        assert pytest.approx(p2) == p1
        assert pytest.approx(t2) == t1

    # Two Sample Trimmed Means T Test

    def test_TrimmmedMeans_pLessZero_Error(self) -> None:
        with pytest.raises(ValueError, match="Percentage trimmed needs to be between 0 and 100"):
            trimmed_means_test([1, 2, 3], [4, 5, 6], p=-5)

    def test_TrimmmedMeans_pGreaterHundred_Error(self) -> None:
        with pytest.raises(ValueError, match="Percentage trimmed needs to be between 0 and 100"):
            trimmed_means_test([1, 2, 3], [4, 5, 6], p=105)

    def test_TrimmedMeans_alternativeWrong_Error(self) -> None:
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            trimmed_means_test([1, 2, 3], [4, 5, 6], p=10, alternative="moar")

    def test_TrimmedMeans_result(self) -> None:
        data_1 = [4, 10, 2, 9, 5, 28, 8, 7, 9, 35, 40]
        data_2 = [2, 8, 6, 22, 11, 27, 10, 25, 30, 38]
        t, p = trimmed_means_test(data_1, data_2, p=20, alternative="two-sided")
        assert pytest.approx(-0.741422, 0.00001) == t
        assert pytest.approx(0.469887, 0.00001) == p

    # Yeun - Welch Test

    def test_YeunWelch_pLessZero_Error(self) -> None:
        with pytest.raises(ValueError, match="Percentage trimmed needs to be between 0 and 100"):
            yuen_welch_test([1, 2, 3], [4, 5, 6], p=-5)

    def test_YeunWelch_pGreaterHundred_Error(self) -> None:
        with pytest.raises(ValueError, match="Percentage trimmed needs to be between 0 and 100"):
            yuen_welch_test([1, 2, 3], [4, 5, 6], p=105)

    def test_YeunWelch_alternativeWrong_Error(self) -> None:
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            yuen_welch_test([1, 2, 3], [4, 5, 6], p=10, alternative="moar")

    def test_YeunWelch_result(self) -> None:
        data_1 = [4, 10, 2, 9, 5, 28, 8, 7, 9, 35, 40]
        data_2 = [12, 8, 6, 16, 12, 14, 10, 16, 6, 11]
        t, p = yuen_welch_test(data_1, data_2, p=20, alternative="two-sided")
        assert pytest.approx(0.739002, 0.00001) == p
        assert pytest.approx(0.34365, 0.0001) == t

    # Two Sample F Test

    def test_twoSampleFTest_alternativeNotString_Error(self) -> None:
        data_1 = [100, 200, 300]
        data_2 = [10, 20, 30]
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            two_sample_f_test(data_1, data_2, alternative=10)

    def test_twoSampleFTest_alternativeWrong_Error(self) -> None:
        data_1 = [100, 200, 300]
        data_2 = [10, 20]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            two_sample_f_test(data_1, data_2, alternative="moar")

    def test_twoSampleFTest_result(self) -> None:
        data_1 = [560, 530, 570, 490, 510, 550, 550, 530]
        data_2 = [600, 590, 590, 630, 610, 630]
        f, p = two_sample_f_test(data_1, data_2)
        assert pytest.approx(p, 0.0001) == 0.4263
        assert pytest.approx(f, 0.0001) == 2.1163

    # Binomial Sign Test

    def test_BinomialSignTest_alternativeNotString_Error(self) -> None:
        n_success, n_failure, success_prob = [10, 20], [30, 40], 0.5
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            binomial_sign_test(n_success, n_failure, alternative=10, success_prob=success_prob)

    def test_BinomialSignTest_alternativeWrong_Error(self) -> None:
        n_success, n_failure, success_prob = [10, 20], [30, 40], 0.5
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            binomial_sign_test(n_success, n_failure, alternative="moar", success_prob=success_prob)

    def test_BinomialSignTest_successProbNotFloat_Error(self) -> None:
        n_success, n_failure, success_prob = [10, 20], [30, 40], "0.5"
        with pytest.raises(TypeError, match="Probability of success needs to be a decimal value"):
            binomial_sign_test(n_success, n_failure, success_prob=success_prob)

    def test_BinomialSignTest_successProbGreaterThanOne_Error(self) -> None:
        n_success, n_failure, success_prob = [10, 20], [30, 40], 1.5
        with pytest.raises(ValueError, match="Cannot calculate probability of success, needs to be between 0 and 1"):
            binomial_sign_test(n_success, n_failure, success_prob=success_prob)

    def test_BinomialSignTest_successProbLessThanZero_Error(self) -> None:
        n_success, n_failure, success_prob = [10, 20], [30, 40], -0.5
        with pytest.raises(ValueError, match="Cannot calculate probability of success, needs to be between 0 and 1"):
            binomial_sign_test(n_success, n_failure, success_prob=success_prob)

    def test_BinomialSignTest_pResult(self) -> None:
        deer_hind = [142, 140, 144, 144, 142, 146, 149, 150, 142, 148]
        deer_fore = [138, 136, 147, 139, 143, 141, 143, 145, 136, 146]
        assert pytest.approx(0.109375) == binomial_sign_test(deer_hind, deer_fore)

    # Wald-Wolfowitz Test

    def test_WaldWolfowitzTest_cutoffWrong_Error(self) -> None:
        x = np.arange(100)
        with pytest.raises(ValueError, match="Cannot determine cutoff point"):
            wald_wolfowitz_test(x, cutoff="moar")

    def test_WaldWolfowitzTest_expectedWrongLength_Error(self) -> None:
        x = np.arange(100)
        expected = np.power(np.arange(99), 2)
        with pytest.raises(AttributeError, match="Cannot perform Wald-Wolfowitz with unequal array lengths"):
            wald_wolfowitz_test(x, expected)

    def test_WaldWolfowitzTest_pResultMedian(self) -> None:
        x = np.random.randint(0, 100, 50)
        x1, p1 = wald_wolfowitz_test(x)
        x2, p2 = runstest_1samp(x, cutoff="median", correction=False)
        assert pytest.approx(p2) == p1

    def test_WaldWolfowitzTest_resultMedian(self) -> None:
        x = np.random.randint(0, 100, 50)
        x1, p1 = wald_wolfowitz_test(x)
        x2, p2 = runstest_1samp(x, cutoff="median", correction=False)
        assert pytest.approx(x2) == x1
        assert pytest.approx(p2) == p1

    def test_WaldWolfowitzTest_resultMean(self) -> None:
        x = np.random.randint(0, 100, 50)
        x1, p1 = wald_wolfowitz_test(x, cutoff="mean")
        x2, p2 = runstest_1samp(x, cutoff="mean", correction=False)
        assert pytest.approx(x2) == x1
        assert pytest.approx(p2) == p1

    # Trinomial Test
    def test_TrinomialTest_notPaired_Error(self) -> None:
        a = [10, 20, 30]
        b = [5, 10]
        with pytest.raises(AttributeError, match="Cannot perform Trinomial Test on unpaired data"):
            trinomial_test(a, b)

    def test_TrinomialTest_alternativeWrong_Error(self) -> None:
        a = [10, 20, 30]
        with pytest.raises(ValueError, match="Cannot determine alternative hypothesis"):
            trinomial_test(a, a, "MOAR")

    def test_TrinomialTest_pResult(self) -> None:
        a = [30, 15, 35, 12, 35, 8, 21, 8, 29, 17]
        b = [23, 13, 31, 15, 35, 8, 18, 7, 22, 13]
        expected = 0.049362
        t, p = trinomial_test(a, b, "two-sided")
        assert pytest.approx(expected, 1e-05) == p

    # Fligner-Policello Test

    def test_FlignerPolicelloTest_alternativeWrong_Error(self) -> None:
        x = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Cannot determine alternative hypothesis"):
            fligner_policello_test(x, x, "MOAR")

    def test_FlignerPolicelloTest_result(self) -> None:
        x = np.array([4, 10, 2, 9, 5, 28, 8, 7, 9, 35, 20])
        y = np.array([12, 8, 6, 16, 12, 14, 10, 18, 4, 11])
        z_expected, p_expected = 0.703046, 0.482027
        z, p = fligner_policello_test(x, y)
        assert pytest.approx(z_expected, 1e-05) == z
        assert pytest.approx(p_expected, 1e-05) == p


if __name__ == "__main__":
    pytest.main()
