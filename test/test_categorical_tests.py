import unittest
import pytest
from StatsTest.categorical_tests import *
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
import statsmodels.api as sm


class TestCategoricalTests(unittest.TestCase):

    # Fisher Test

    def test_fisherTest_wrongAlternative_Error(self):
        table = [[1, 2], [3, 4]]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            fisher_test(table, alternative='poop')

    def test_fisherTest_notContTable_Error(self):
        table = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Fisher's Exact Test is meant for a 2x2 contingency table"):
            fisher_test(table)

    def test_fisherTest_pResult(self):
        table = [[8, 2], [1, 5]]
        p1 = fisher_test(table, alternative='two-sided')
        odds_ratio, p2 = fisher_exact(table, alternative='two-sided')
        assert pytest.approx(p2) == p1

    # McNemar Test

    def test_mcnemarTest_notContTable_Error(self):
        table = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="McNemar's Test is meant for a 2x2 contingency table"):
            mcnemar_test(table)

    def test_mcnemarTest_pResult(self):
        table = [[100, 200], [300, 400]]
        x1, p1 = mcnemar_test(table)
        result = mcnemar(table, exact=False)
        x2, p2 = result.statistic, result.pvalue
        assert pytest.approx(p2) == p1

    def test_mcnemarTest_xResult(self):
        table = [[100, 200], [300, 400]]
        x1, p1 = mcnemar_test(table)
        result = mcnemar(table, exact=False)
        x2, p2 = result.statistic, result.pvalue
        assert pytest.approx(x2) == x1

    # Chi Square Test for Contingency Tables

    def test_chiSquareTest_pResult(self):
        table = [[100, 200], [300, 400]]
        x1, p1 = chi_squared_test(table)
        x2, p2, dof, expected = chi2_contingency(table, correction=False)
        assert pytest.approx(p2) == p1

    def test_chiSquareTest_xResult(self):
        table = [[100, 200], [300, 400]]
        x1, p1 = chi_squared_test(table)
        x2, p2, dof, expected = chi2_contingency(table, correction=False)
        assert pytest.approx(x1, 0.01) == x2

    # G Test for Contingency Tables

    def test_gTest_pResult(self):
        table = [[100, 200], [300, 400]]
        x1, p1 = g_test(table)
        x2, p2, dof, expected = chi2_contingency(table, correction=False, lambda_="log-likelihood")
        assert pytest.approx(p2) == p1

    def test_gTest_xResult(self):
        table = [[100, 200], [300, 400]]
        x1, p1 = g_test(table)
        x2, p2, dof, expected = chi2_contingency(table, correction=False, lambda_="log-likelihood")
        assert pytest.approx(x2) == x1

    # CMH Test

    def test_cmhTest_TooFewObs_Error(self):
        data_1 = [[222, 1234], [35, 61]]
        with pytest.raises(AttributeError, match="Cannot perform CMH Test on less than 2 groups"):
            cmh_test(data_1)

    def test_cmhTest_NotContingency_Error(self):
        data_1 = [123, 124, 125, 126]
        data_2 = [[222, 1234], [35, 61]]
        with pytest.raises(AttributeError, match="CMH Test is meant for 2x2 contingency tables"):
            cmh_test(data_1, data_2)

    def test_cmhTest_pResult(self):
        data_1 = [[126, 100], [35, 61]]
        data_2 = [[908, 688], [497, 807]]
        data_3 = [[913, 747], [336, 598]]
        data_4 = [[235, 172], [58, 121]]
        data_5 = [[402, 308], [121, 215]]
        data_6 = [[182, 156], [72, 98]]
        data_7 = [[60, 99], [11, 43]]
        data_8 = [[104, 89], [21, 36]]
        mat = [data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8]
        s_table = sm.stats.StratifiedTable(mat)
        epsilon1, p1 = cmh_test(data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8)
        epsilon2, p2 = s_table.test_null_odds().statistic, s_table.test_null_odds().pvalue
        assert pytest.approx(p2) == p1

    def test_cmhTest_xResult(self):
        data_1 = [[126, 100], [35, 61]]
        data_2 = [[908, 688], [497, 807]]
        data_3 = [[913, 747], [336, 598]]
        data_4 = [[235, 172], [58, 121]]
        data_5 = [[402, 308], [121, 215]]
        data_6 = [[182, 156], [72, 98]]
        data_7 = [[60, 99], [11, 43]]
        data_8 = [[104, 89], [21, 36]]
        mat = [data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8]
        s_table = sm.stats.StratifiedTable(mat)
        epsilon1, p1 = cmh_test(data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8)
        epsilon2, p2 = s_table.test_null_odds().statistic, s_table.test_null_odds().pvalue
        assert pytest.approx(epsilon2) == epsilon1


if __name__ == '__main__':
    unittest.main()
