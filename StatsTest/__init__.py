"""Implementing Multiple Statistical Tests in Python"""

__version__ = "0.1.0"

from StatsTest import (
    categorical_tests,
    correlation_tests,
    gof_tests,
    multi_group_tests,
    outliers_test,
    post_hoc_tests,
    proportion_tests,
    rank_tests,
    sample_tests,
    utils,
)

__all__: list[str] = [
    "categorical_tests",
    "correlation_tests",
    "gof_tests",
    "multi_group_tests",
    "outliers_test",
    "post_hoc_tests",
    "proportion_tests",
    "rank_tests",
    "sample_tests",
    "utils",
]


def __dir__() -> list[str]:
    return __all__
