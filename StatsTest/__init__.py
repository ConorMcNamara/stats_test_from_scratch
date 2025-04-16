"""Implementing Multiple Statistical Tests in Python"""

__version__ = "0.1.0"

from typing import List

from StatsTest import categorical_tests, correlation_tests, gof_tests, multi_group_tests, outliers_test, post_hoc_tests, proportion_tests, rank_tests, residuals_test, sample_tests, utils

__all__: List[str] = [
    "categorical_tests",
    "correlation_tests",
    "gof_tests",
    "multi_group_tests",
    "outliers_test",
    "post_hoc_tests",
    "proportion_tests",
    "rank_tests",
    "residuals_test",
    "sample_tests",
    "utils"
]


def __dir__() -> List[str]:
    return __all__
