import numpy as np
import pytest
from scipy.stats import kendalltau, pearsonr, pointbiserialr, spearmanr

from StatsTest.correlation_tests import (
    kendall_tau_test,
    pearson_test,
    point_biserial_correlation_test,
    rank_biserial_correlation_test,
    spearman_test,
)


class TestCorrelationTests:
    # Pearson Test

    def test_pearsonTest_wrongLength_Error(self) -> None:
        x = [1, 2, 3, 4]
        y = [1, 2, 3]
        with pytest.raises(ValueError, match="Cannot calculate correlation with datasets of different lengths"):
            pearson_test(x, y)

    def test_pearsonTest_pResult(self) -> None:
        x = np.random.normal(0, 100, 1000)
        y = np.random.normal(100, 50, 1000)
        r1, p1 = pearson_test(x, y)
        r2, p2 = pearsonr(x, y)
        assert pytest.approx(p2) == p1

    def test_pearsonTest_rhoResult(self) -> None:
        x = np.random.normal(0, 100, 1000)
        y = np.random.normal(100, 50, 1000)
        r1, p1 = pearson_test(x, y)
        r2, p2 = pearsonr(x, y)
        assert pytest.approx(r2) == r1

    # Spearman Rank Test

    def test_spearmanTest_wrongLength_Error(self) -> None:
        x = [1, 2, 3, 4]
        y = [1, 2, 3]
        with pytest.raises(ValueError, match="Cannot calculate correlation with datasets of different lengths"):
            spearman_test(x, y)

    def test_spearmanTest_pResult(self) -> None:
        x = np.random.normal(0, 100, 1000)
        y = np.random.normal(100, 50, 1000)
        r1, p1 = spearman_test(x, y)
        r2, p2 = spearmanr(x, y)
        assert pytest.approx(p2) == p1

    def test_spearmanTest_rhoResult(self) -> None:
        x = np.random.normal(0, 100, 1000)
        y = np.random.normal(100, 50, 1000)
        r1, p1 = spearman_test(x, y)
        r2, p2 = spearmanr(x, y)
        assert pytest.approx(r2) == r1

    # Kendall Tau Test

    def test_kendallTau_unevenLengths_Error(self) -> None:
        x = np.random.randint(0, 100, 20)
        y = np.random.randint(0, 100, 15)
        with pytest.raises(ValueError, match="Cannot calculate correlation with datasets of different lengths"):
            kendall_tau_test(x, y)

    def test_kendallTau_wrongMethod_Error(self) -> None:
        x = np.random.randint(0, 100, 20)
        with pytest.raises(ValueError, match="Cannot determine type of test for Kendall Tau"):
            kendall_tau_test(x, x, method="moar")

    def test_kendallTau_exactTies_Error(self) -> None:
        x1 = [12, 2, 1, 12, 2]
        x2 = [1, 4, 7, 1, 0]
        with pytest.raises(AttributeError, match="Cannot run exact test when ties are present"):
            kendall_tau_test(x1, x2, method="exact")

    def test_kendallTau_exact_pResult(self) -> None:
        x = [4, 10, 3, 1, 9, 2, 6, 7, 8, 5]
        y = [5, 8, 6, 2, 10, 3, 9, 4, 7, 1]
        t1, p1 = kendall_tau_test(x, y, method="exact")
        t2, p2 = kendalltau(x, y, method="exact")
        assert pytest.approx(p2) == p1

    def test_kendallTau_exact_tResult(self) -> None:
        x = [4, 10, 3, 1, 9, 2, 6, 7, 8, 5]
        y = [5, 8, 6, 2, 10, 3, 9, 4, 7, 1]
        t1, p1 = kendall_tau_test(x, y, method="exact")
        t2, p2 = kendalltau(x, y, method="exact")
        assert pytest.approx(t2) == t1

    def test_kendallTau_ties_pResult(self) -> None:
        x1 = [12, 2, 1, 12, 2]
        x2 = [1, 4, 7, 1, 0]
        t1, p1 = kendall_tau_test(x1, x2, method="significance")
        t2, p2 = kendalltau(x1, x2, method="asymptotic")
        assert pytest.approx(p2) == p1

    def test_kendallTau_ties_tResult(self) -> None:
        x1 = [12, 2, 1, 12, 2]
        x2 = [1, 4, 7, 1, 0]
        t1, p1 = kendall_tau_test(x1, x2, method="significance")
        t2, p2 = kendalltau(x1, x2, method="asymptotic")
        assert pytest.approx(t2) == t1

    # Point Biserial Test

    def test_BiserialCorrelationPoint_tooManyGroups_Error(self) -> None:
        a = np.array([0, 1, 2, 0, 1, 2])
        b = np.arange(6)
        with pytest.raises(AttributeError, match="Need to have two groupings for biseral correlation"):
            point_biserial_correlation_test(b, a)

    def test_BiserialCorrelationPoint_unequalLength_Error(self) -> None:
        a = np.array([0, 1, 1, 0])
        b = np.arange(5)
        with pytest.raises(ValueError, match="X and Y must be of the same length"):
            point_biserial_correlation_test(b, a)

    def test_BiserialCorrelationPoint_pResult(self) -> None:
        a = np.array([0, 0, 0, 1, 1, 1, 1])
        b = np.arange(7)
        r1, p1 = point_biserial_correlation_test(b, a)
        r2, p2 = pointbiserialr(a, b)
        assert pytest.approx(p2) == p1

    def test_BiserialCorrelationPoint_rResult(self) -> None:
        a = np.array([0, 0, 0, 1, 1, 1, 1])
        b = np.arange(7)
        r1, p1 = point_biserial_correlation_test(b, a)
        r2, p2 = pointbiserialr(a, b)
        assert pytest.approx(r2) == r1

    # Rank Biserial Test

    def test_biserial_correlation_rank_too_many_groups_error(self) -> None:
        a = np.array([0, 1, 2, 0, 1, 2])
        b = np.arange(6)
        with pytest.raises(AttributeError, match="Need to have two groupings for biseral correlation"):
            rank_biserial_correlation_test(b, a)

    def test_biserial_correlation_rank_unequal_length_error(self) -> None:
        a = np.array([0, 1, 1, 0])
        b = np.arange(5)
        with pytest.raises(ValueError, match="X and Y must be of the same length"):
            rank_biserial_correlation_test(b, a)


if __name__ == "__main__":
    pytest.main()
