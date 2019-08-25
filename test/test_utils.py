import StatsTest.utils as utils
import pytest
import unittest
import pandas as pd
from numpy.testing import assert_array_equal
import numpy as np
from scipy.stats.mstats_basic import skew, kurtosis


class TestUtils(unittest.TestCase):

    def test_standardError_String1_Error(self):
        s, n = 's', 10
        with pytest.raises(TypeError):
            utils._standard_error(s, n)

    def test_standardError_List1_Error(self):
        s, n = ['s'], 10
        with pytest.raises(TypeError):
            utils._standard_error(s, n)

    def test_standardError_Dict1_Error(self):
        s, n = {'s': 10}, 10
        with pytest.raises(TypeError):
            utils._standard_error(s, n)

    def test_standardError_String2_Error(self):
        s, n = 10, 'n'
        with pytest.raises(TypeError):
            utils._standard_error(s, n)

    def test_standardError_List2_Error(self):
        s, n = 10, 'n'
        with pytest.raises(TypeError):
            utils._standard_error(s, n)

    def test_standardError_Dict2_Error(self):
        s, n = 10, {'n', 10}
        with pytest.raises(TypeError):
            utils._standard_error(s, n)

    def test_standardError_nLessZero_Error(self):
        s, n = 10, -1
        with pytest.raises(ValueError):
            utils._standard_error(s, n)

    def test_standardError_nFloat_Error(self):
        s, n = 10, 1.5
        with pytest.raises(TypeError):
            utils._standard_error(s, n)

    def test_standardError_nZero_Error(self):
        s, n = 10, 0
        with pytest.raises(ValueError):
            utils._standard_error(s, n)

    def test_hypergeom_List_Error(self):
        a, b, c, d = 10, 10, 10, [10]
        with pytest.raises(TypeError):
            utils._hypergeom_distribution(a, b, c, d)

    def test_hypergeom_String_Error(self):
        a, b, c, d = 10, 10, 10, '10'
        with pytest.raises(TypeError):
            utils._hypergeom_distribution(a, b, c, d)

    def test_hypergeom_Float_Error(self):
        a, b, c, d = 10, 10, 10, 10.5
        with pytest.raises(TypeError):
            utils._hypergeom_distribution(a, b, c, d)

    def test_hypergeom_Dict_Error(self):
        a, b, c, d = 10, 10, 10, {'d': 10}
        with pytest.raises(TypeError):
            utils._hypergeom_distribution(a, b, c, d)

    def test_checkTable_ListStrings_Error(self):
        table = ['a', 'b', 'c']
        with pytest.raises(TypeError):
            utils._check_table(table, False)

    def test_checkTable_Dict_Error(self):
        table = {'a': 10}
        with pytest.raises(TypeError):
            utils._check_table(table, False)

    def test_checkTable_Pandas_Error(self):
        table = pd.DataFrame({'a': [10]})
        with pytest.raises(TypeError):
            utils._check_table(table, False)

    def test_checkTable_Float_Error(self):
        table = [1, 2, 3.0]
        with pytest.raises(TypeError):
            utils._check_table(table, True)

    def test_checkTable_Negative_Error(self):
        table = [-1, 1, 2]
        with pytest.raises(ValueError):
            utils._check_table(table, True)

    def test_sse_NonPositiveSquare_Error(self):
        table = [-1, 100, 49]
        sum_data = [1, 4, 5]
        n_data = [10, 10, 10]
        with pytest.raises(ValueError):
            utils._sse(sum_data, table, n_data)

    def test_sse_NonPositiveN_Error(self):
        squared_data = [1, 4, 9]
        sum_data = [1, 2, 3]
        n_data = [10, 10, -1]
        with pytest.raises(ValueError):
            utils._sse(sum_data, squared_data, n_data)

    def test_standardError_Result(self):
        s, n = 10, 100
        assert pytest.approx(1.0, 0.01) == utils._standard_error(s, n)

    def test_hypergeom_Result(self):
        a, b, c, d = 1, 2, 3, 4
        assert pytest.approx(0.5, 0.01) == utils._hypergeom_distribution(a, b, c, d)

    def test_checkTable_Result(self):
        table = [1, 2, 3]
        assert_array_equal(utils._check_table(table), np.array(table))

    def test_see_Result(self):
        sum_data = [100, 200, 300]
        square_data = [2500, 4900, 8100]
        n_data = [10, 10, 10]
        assert pytest.approx(1500, 0.01) == utils._sse(sum_data, square_data, n_data)

    def test_rightExtreme_Result(self):
        n = 50
        n_success = 25
        p = 0.5
        assert pytest.approx(utils._left_extreme(n_success, n, p)) == utils._right_extreme(n_success, n, p)

    def test_leftExtreme_Result(self):
        n = 100
        n_success = 60
        p = 0.5
        assert pytest.approx(utils._left_extreme(n_success, n, p)) != utils._right_extreme(n_success, n, p)

    def test_skew_result(self):
        data = np.random.normal(0, 100, 1000)
        skew_1 = utils._skew(data)
        skew_2 = skew(data)
        assert pytest.approx(skew_2) == skew_1

    def test_kurtosis_result(self):
        data = np.random.normal(0, 100, 1000)
        kurt_1 = utils._kurtosis(data)
        kurt_2 = kurtosis(data) + 3
        assert pytest.approx(kurt_2) == kurt_1


if __name__ == '__main__':
    unittest.main()