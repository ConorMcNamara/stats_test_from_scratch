import unittest
import pytest
from StatsTest.categorical_tests import *
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar, SquareTable
import statsmodels.api as sm


class TestCategoricalTests(unittest.TestCase):

    # Fisher Test

    def test_fisherTest_wrongAlternative_Error(self) -> None:
        table = [[1, 2], [3, 4]]
        with pytest.raises(ValueError, match="Cannot determine method for alternative hypothesis"):
            fisher_test(table, alternative='poop')

    def test_fisherTest_notContTable_Error(self) -> None:
        table = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="Fisher's Exact Test is meant for a 2x2 contingency table"):
            fisher_test(table)

    def test_fisherTest_pResult(self) -> None:
        table = [[8, 2], [1, 5]]
        p1 = fisher_test(table, alternative='two-sided')
        odds_ratio, p2 = fisher_exact(table, alternative='two-sided')
        assert pytest.approx(p2) == p1

    # McNemar Test

    def test_mcnemarTest_notContTable_Error(self) -> None:
        table = [1, 2, 3, 4]
        with pytest.raises(AttributeError, match="McNemar's Test is meant for a 2x2 contingency table"):
            mcnemar_test(table)

    def test_mcnemarTest_pResult(self) -> None:
        table = [[100, 200], [300, 400]]
        x1, p1 = mcnemar_test(table)
        result = mcnemar(table, exact=False)
        x2, p2 = result.statistic, result.pvalue
        assert pytest.approx(p2) == p1

    def test_mcnemarTest_xResult(self) -> None:
        table = [[100, 200], [300, 400]]
        x1, p1 = mcnemar_test(table)
        result = mcnemar(table, exact=False)
        x2, p2 = result.statistic, result.pvalue
        assert pytest.approx(x2) == x1

    # Chi Square Test for Contingency Tables

    def test_chiSquareTest_pResult(self) -> None:
        table = [[100, 200], [300, 400]]
        x1, p1 = chi_squared_test(table)
        x2, p2, dof, expected = chi2_contingency(table, correction=False)
        assert pytest.approx(p2) == p1

    def test_chiSquareTest_xResult(self) -> None:
        table = [[100, 200], [300, 400]]
        x1, p1 = chi_squared_test(table)
        x2, p2, dof, expected = chi2_contingency(table, correction=False)
        assert pytest.approx(x1, 0.01) == x2

    # G Test for Contingency Tables

    def test_gTest_pResult(self) -> None:
        table = [[100, 200], [300, 400]]
        x1, p1 = g_test(table)
        x2, p2, dof, expected = chi2_contingency(table, correction=False, lambda_="log-likelihood")
        assert pytest.approx(p2) == p1

    def test_gTest_xResult(self) -> None:
        table = [[100, 200], [300, 400]]
        x1, p1 = g_test(table)
        x2, p2, dof, expected = chi2_contingency(table, correction=False, lambda_="log-likelihood")
        assert pytest.approx(x2) == x1

    # CMH Test

    def test_cmhTest_TooFewObs_Error(self) -> None:
        data_1 = [[222, 1234], [35, 61]]
        with pytest.raises(AttributeError, match="Cannot perform CMH Test on less than 2 groups"):
            cmh_test(data_1)

    def test_cmhTest_NotContingency_Error(self) -> None:
        data_1 = [123, 124, 125, 126]
        data_2 = [[222, 1234], [35, 61]]
        with pytest.raises(AttributeError, match="CMH Test is meant for 2x2 contingency tables"):
            cmh_test(data_1, data_2)

    def test_cmhTest_pResult(self) -> None:
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

    def test_cmhTest_xResult(self) -> None:
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

    # Woolf Test

    def test_woolfTest_kLess2_Error(self) -> None:
        data = [[1, 2], [3, 4]]
        with pytest.raises(AttributeError, match="Cannot perform Woolf Test on less than two groups"):
            woolf_test(data)

    def test_woolfTest_notSquare_Error(self) -> None:
        data = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(AttributeError, match="Woolf Test is meant for 2x2 contingency table"):
            woolf_test(data, data)

    def test_woolfTest_pResult(self) -> None:
        data_1 = [[1, 365], [10, 502]]
        data_2 = [[30, 335], [38, 464]]
        data_3 = [[12, 323], [15, 447]]
        data_4 = [[5, 318], [7, 442]]
        data_5 = [[4, 314], [2, 440]]
        data_6 = [[1, 313], [4, 436]]
        data_7 = [[1, 312], [2, 434]]
        x, p = woolf_test(data_1, data_2, data_3, data_4, data_5, data_6, data_7)
        assert pytest.approx(.4094, 0.001) == p

    def test_woolfTest_xResult(self) -> None:
        data_1 = [[1, 365], [10, 502]]
        data_2 = [[30, 335], [38, 464]]
        data_3 = [[12, 323], [15, 447]]
        data_4 = [[5, 318], [7, 442]]
        data_5 = [[4, 314], [2, 440]]
        data_6 = [[1, 313], [4, 436]]
        data_7 = [[1, 312], [2, 434]]
        x, p = woolf_test(data_1, data_2, data_3, data_4, data_5, data_6, data_7)
        assert pytest.approx(6.1237, 0.001) == x

    # Breslow-Day Test

    def test_breslowDayTest_kLess2_Error(self) -> None:
        data = [[1, 2], [3, 4]]
        with pytest.raises(AttributeError, match="Cannot perform Breslow-Day Test for less than 2 groups"):
            breslow_day_test(data)

    def test_breslowDayTest_notSquare_Error(self) -> None:
        data = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(AttributeError, match="Breslow-Day Test is meant for 2x2 contingency table"):
            breslow_day_test(data, data)

    def test_breslowDayTest_pResult(self) -< None:
        data_1 = [[126, 100], [35, 61]]
        data_2 = [[908, 688], [497, 807]]
        data_3 = [[913, 747], [336, 598]]
        data_4 = [[235, 172], [58, 121]]
        data_5 = [[402, 308], [121, 215]]
        data_6 = [[182, 156], [72, 98]]
        data_7 = [[60, 99], [11, 43]]
        data_8 = [[104, 89], [21, 36]]
        mat = [data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8]
        x1, p1 = breslow_day_test(data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8)
        x2, p2 = sm.stats.StratifiedTable(mat).test_equal_odds().statistic, sm.stats.StratifiedTable(mat).test_equal_odds().pvalue
        assert pytest.approx(p2) == p1

    def test_breslowDayTest_xResult(self) -> None:
        data_1 = [[126, 100], [35, 61]]
        data_2 = [[908, 688], [497, 807]]
        data_3 = [[913, 747], [336, 598]]
        data_4 = [[235, 172], [58, 121]]
        data_5 = [[402, 308], [121, 215]]
        data_6 = [[182, 156], [72, 98]]
        data_7 = [[60, 99], [11, 43]]
        data_8 = [[104, 89], [21, 36]]
        mat = [data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8]
        x1, p1 = breslow_day_test(data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8)
        x2, p2 = sm.stats.StratifiedTable(mat).test_equal_odds().statistic, sm.stats.StratifiedTable(mat).test_equal_odds().pvalue
        assert pytest.approx(x2) == x1

    # Bowker Test

    def test_bowkerTest_nonSquare_Error(self) -> None:
        data_1 = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(AttributeError, match="Contingency Table needs to be of a square shape"):
            bowker_test(data_1)

    def test_bowkerTest_pResult(self) -> None:
        cont_table = np.random.randint(0, 100, (4, 4))
        x1, p1 = bowker_test(cont_table)
        square = SquareTable(cont_table)
        x2, p2 = square.symmetry().statistic, square.symmetry().pvalue
        assert pytest.approx(p2) == p1

    def test_bowkerTest_xResult(self) -> None:
        cont_table = np.random.randint(0, 100, (4, 4))
        x1, p1 = bowker_test(cont_table)
        square = SquareTable(cont_table)
        x2, p2 = square.symmetry().statistic, square.symmetry().pvalue
        assert pytest.approx(x2) == x1


if __name__ == '__main__':
    unittest.main()
