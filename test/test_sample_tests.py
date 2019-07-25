from StatsTest.sample_tests import *
import pytest
import unittest
import numpy as np


class TestSampleTest(unittest.TestCase):

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

    def test_TwoSampleTTest_alternativeInt_Error(self):
        sample_data = [10, 20, 30]
        with pytest.raises(TypeError, match="Alternative Hypothesis is not of string type"):
            two_sample_t_test(sample_data, sample_data, alternative=10)

    def test_TwoSampleTTest_alternativeWrong_Error(self):
        sample_data = [10, 20, 30]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            two_sample_t_test(sample_data, sample_data, alternative='more')

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

    def test_OneSampleZTest_pResult(self):
        sample_data = np.arange(30)
        pop_mean = 5
        assert pytest.approx(0.0, 0.01) == one_sample_z_test(sample_data, pop_mean)[1]

    def test_OneSampleZTest_zResult(self):
        sample_data = np.arange(30)
        pop_mean = 5
        assert pytest.approx(11.389, 0.01) == one_sample_z_test(sample_data, pop_mean)[0]

    def test_TwoSampleZTest_pResult(self):
        sample_data = np.arange(30)
        assert pytest.approx(1.0, 0.01) == two_sample_z_test(sample_data, sample_data)[1]

    def test_TwoSampleZTest_zResult(self):
        sample_data = np.arange(30)
        assert pytest.approx(0.0, 0.01) == two_sample_z_test(sample_data, sample_data)[0]

    def test_OneSampleZProp_pResult(self):
        sample_data = [0, 1] * 15
        pop_mean = 0.5
        assert pytest.approx(1.0, 0.01) == one_sample_proportion_z_test(sample_data, pop_mean)[1]

    def test_OneSampleZProp_zResult(self):
        sample_data = [0, 1] * 15
        pop_mean = 0.5
        assert pytest.approx(0.0, 0.01) == one_sample_proportion_z_test(sample_data, pop_mean)[0]

    def test_TwoSampleZProp_pResult(self):
        sample_data = [0, 1] * 15
        assert pytest.approx(1.0, 0.01) == two_sample_proportion_z_test(sample_data, sample_data)[1]

    def test_TwoSampleZProp_zResult(self):
        sample_data = [0, 1] * 15
        assert pytest.approx(0.0, 0.01) == two_sample_proportion_z_test(sample_data, sample_data)[0]


if __name__ == '__main__':
    unittest.main()