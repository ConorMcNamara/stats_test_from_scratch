from StatsTest.outliers_tests import *
import unittest
import pytest
import numpy as np


class TestOutliersTest(unittest.TestCase):

    # Tukey's Fence Test
    def test_TukeyFenceTest_results(self):
        data = [5, 63, 64, 64, 70, 72, 76, 77, 81, 100]
        np.testing.assert_array_equal([5, 100], tukey_fence_test(data, 1.5))

    # Grubb's Test

    def test_GrubbsTest_alternativeNotString_Error(self):
        data = [1, 2, 3, 4, 5]
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            grubbs_test(data, alternative=1)

    def test_GrubbsTest_alternativeWrong_Error(self):
        data = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            grubbs_test(data, alternative="moar")

    def test_GrubbsTest_alphaNotFloat_Error(self):
        data = [1, 2, 3, 4, 5]
        with pytest.raises(TypeError, match="Cannot discern alpha level for Grubb's test"):
            grubbs_test(data, alternative="two-sided", alpha="one")

    def test_GrubbsTest_alphaLessZero_Error(self):
        data = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="Alpha level must be within 0 and 1"):
            grubbs_test(data, alternative="two-sided", alpha=-0.5)

    def test_GrubbsTest_alphaGreaterOne_Error(self):
        data = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="Alpha level must be within 0 and 1"):
            grubbs_test(data, alternative="two-sided", alpha=1.5)

    def test_GrubbsTest_resultBool(self):
        data = [199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57]
        result, outlier = grubbs_test(data, alternative="two-sided", alpha=0.05)
        assert result is True

    def test_GrubbsTest_resultOutlier(self):
        data = [199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57]
        result, outlier = grubbs_test(data, alternative="two-sided", alpha=0.05)
        assert outlier == 245.57

    # Extreme Studentized Deviate Test

    def test_ESD_numOutliersNotInt_Error(self):
        data = [1, 2, 3, 4]
        with pytest.raises(TypeError, match="Number of outliers must be an integer"):
            extreme_studentized_deviate_test(data, num_outliers=1.5)

    def test_ESD_numOutliersLessZero_Error(self):
        data = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Cannot test for negative amount of outliers"):
            extreme_studentized_deviate_test(data, num_outliers=-1)

    def test_ESD_numOutliersGreaterN_Error(self):
        data = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Cannot have number of outliers greater than number of observations"):
            extreme_studentized_deviate_test(data, num_outliers=5)

    def test_ESD_alphaGreaterOne_Error(self):
        data = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Alpha level must be within 0 and 1"):
            extreme_studentized_deviate_test(data, num_outliers=1, alpha=1.2)

    def test_ESD_alphaLessZero_Error(self):
        data = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Alpha level must be within 0 and 1"):
            extreme_studentized_deviate_test(data, num_outliers=1, alpha=-1)

    def test_ESD_numResults(self):
        data = [-0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26,
                1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56,
                1.58, 1.65, 1.69, 1.70, 1.76, 1.77, 1.81,
                1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10,
                2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37,
                2.40, 2.47, 2.54, 2.62, 2.64, 2.90, 2.92,
                2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68,
                4.30, 4.64, 5.34, 5.42, 6.01]
        outliers, outliers_list = extreme_studentized_deviate_test(data, num_outliers=10, alpha=0.05)
        assert outliers == 3

    def test_ESD_outliersResults(self):
        data = [-0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26,
                1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56,
                1.58, 1.65, 1.69, 1.70, 1.76, 1.77, 1.81,
                1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10,
                2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37,
                2.40, 2.47, 2.54, 2.62, 2.64, 2.90, 2.92,
                2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68,
                4.30, 4.64, 5.34, 5.42, 6.01]
        outliers, outliers_list = extreme_studentized_deviate_test(data, num_outliers=10, alpha=0.05)
        np.testing.assert_array_equal(outliers_list, [6.01, 5.42, 5.34])

    # Tietjen Test

    def test_TietjenMooreTest_numOutliersLessZero_Error(self):
        data = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Cannot test for negative amount of outliers"):
            tietjen_moore_test(data, num_outliers=-1)

    def test_TietjenMooreTest_numOutliersNotInt_Error(self):
        data = [1, 2, 3, 4]
        with pytest.raises(TypeError, match="Number of outliers must be an integer"):
            tietjen_moore_test(data, num_outliers=1.5)

    def test_TietjenMooreTest_alphaLessZero_Error(self):
        data = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Alpha level must be within 0 and 1"):
            tietjen_moore_test(data, alpha=-1.2)

    def test_TietjenMooreTest_alphaGreaterOne_Error(self):
        data = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Alpha level must be within 0 and 1"):
            tietjen_moore_test(data, alpha=1.2)

    def test_TietjenMooreTest_result(self):
        data = [-1.40, -0.44, -0.30, -0.24, -0.22, -0.13, -0.05, 0.06, 0.10, 0.18, 0.20, 0.39, 0.48, 0.63, 1.01]
        result, outliers = tietjen_moore_test(data, num_outliers=2, alternative='two-sided', alpha=0.05)
        assert result is True

    def test_TietjenMooreTest_outliers(self):
        data = [-1.40, -0.44, -0.30, -0.24, -0.22, -0.13, -0.05, 0.06, 0.10, 0.18, 0.20, 0.39, 0.48, 0.63, 1.01]
        result, outliers = tietjen_moore_test(data, num_outliers=2, alternative='two-sided', alpha=0.05)
        np.testing.assert_array_equal(outliers, np.array([1.01, -1.4]))

    # Chauvenet Test

    def test_ChauvenetTest_result(self):
        data = [9, 10, 10, 10, 11, 50]
        np.testing.assert_array_equal(chauvenet_test(data), [50])

    # Peirce Test

    def test_PeirceTest_numOutliersNotInt_Error(self):
        data = [1, 2, 3, 4]
        expected = [1, 4, 9, 16]
        with pytest.raises(TypeError, match="Number of outliers needs to be an integer"):
            peirce_test(data, expected, num_outliers=1.5)

    def test_PeirceTest_numOutliersLessZero_Error(self):
        data = [1, 2, 3, 4]
        expected = [1, 4, 9, 16]
        with pytest.raises(ValueError, match="Number of outliers has to be a positive value"):
            peirce_test(data, expected, num_outliers=-1)

    def test_PeirceTest_numOutliersGreaterN_Error(self):
        data = [1, 2, 3, 4]
        expected = [1, 4, 9, 16]
        with pytest.raises(ValueError, match="Cannot have number of outliers greater than number of observations"):
            peirce_test(data, expected, num_outliers=5)

    def test_PeirceTest_ObservedNotEqualExpected_Error(self):
        data = [1, 2, 3, 4]
        expected = [1, 4, 9]
        with pytest.raises(ValueError, match="Length of observed and expected need to be the same"):
            peirce_test(data, expected, num_outliers=1)

    def test_PeirceTest_numCoefNotInt_Error(self):
        data = [1, 2, 3, 4]
        expected = [1, 4, 9, 16]
        with pytest.raises(TypeError, match="Number of regression coefficients needs to be an integer"):
            peirce_test(data, expected, num_coef=1.5)

    def test_PeirceTest_numCoefLessZero_Error(self):
        data = [1, 2, 3, 4]
        expected = [1, 4, 9, 16]
        with pytest.raises(ValueError, match="Number of regression coefficients has to be a positive value"):
            peirce_test(data, expected, num_coef=-1)

    # Dixon's Q Test

    def test_DixonQTest_nLess3_Error(self):
        data = [1, 2]
        with pytest.raises(AttributeError, match="Cannot run Dixon's Q Test with less than 3 observations"):
            dixon_q_test(data, "1")

    def test_DixonQTest_nGreater30_Error(self):
        data = np.random.randint(0, 50, 31)
        with pytest.raises(AttributeError, match="Too many observations, cannot determine critical score for Q test"):
            dixon_q_test(data, 0.01)

    def test_DixonQTest_alphaWrong_Error(self):
        data = np.random.randint(0, 50, 30)
        with pytest.raises(ValueError, match="Cannot determine alpha level for critical value"):
            dixon_q_test(data, 0.15)

    def test_DixonQTest_results(self):
        data = [0.189, 0.167, 0.187, 0.183, 0.186, 0.182, 0.181, 0.184, 0.181, 0.177]
        np.testing.assert_array_equal(dixon_q_test(data, alpha=0.10), np.array([0.167]))

    def test_DixonQTest_emptyResults(self):
        data = [0.189, 0.167, 0.187, 0.183, 0.186, 0.182, 0.181, 0.184, 0.181, 0.177]
        assert len(dixon_q_test(data)) == 0

    # Thompson Tau Test

    def test_ThompsonTauTest_alphaLessZero_Error(self):
        data = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Cannot have alpha level greater than 1 or less than 0"):
            thompson_tau_test(data, alpha=-0.5)

    def test_ThompsonTauTest_alphaGreaterOne_Error(self):
        data = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Cannot have alpha level greater than 1 or less than 0"):
            thompson_tau_test(data, alpha=1.2)

    def test_ThompsonTauTest_noResults(self):
        data = [48.9, 49.2, 49.2, 49.3, 49.3, 49.8, 49.9, 50.1, 50.2, 50.5]
        assert len(thompson_tau_test(data, 0.05)) == 0

    def test_ThompsonTauTest_result(self):
        data = [9, 10, 10, 10, 11, 50]
        np.testing.assert_array_equal(thompson_tau_test(data), [50])


if __name__ == '__main__':
    unittest.main()