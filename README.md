# StatsTest From Scratch

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Educational implementations of statistical tests in Python, built from scratch for clarity and understanding.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Available Tests](#available-tests)
- [Usage Examples](#usage-examples)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

One major issue with Python's statistical ecosystem is the lack of a unified source for statistical tests. While scikit-learn provides comprehensive machine learning tools and Keras unifies deep learning, statistical tests are scattered across SciPy, Statsmodels, and various other packages. This fragmentation makes it unclear:

- Which library contains which test
- How tests are actually implemented
- Where to find tests that exist in R but not Python

**StatsTest From Scratch** solves this by providing:

1. **Clear, educational implementations** of statistical tests with emphasis on readability over performance
2. **Complete documentation** of where each test can be found in SciPy/Statsmodels (if available)
3. **Implementation of tests** that don't exist in mainstream Python libraries

## ✨ Features

- 🔬 **90+ statistical tests** implemented from scratch
- 📚 **Educational focus** - understand exactly how each test works
- 🔗 **Library reference** - find the fastest production implementations
- 🐍 **Modern Python** - type hints, clean code, well-documented
- ✅ **Fully tested** - comprehensive test suite
- 📖 **NumPy-style docstrings** - clear API documentation

## 📦 Installation

**Requirements:** Python 3.13 or higher

This project supports multiple installation methods: **pip**, **Poetry**, and **uv**.

### Using pip (Standard)

```bash
# Clone the repository
git clone https://github.com/ConorMcNamara/stats_test_from_scratch.git
cd stats_test_from_scratch

# Install the package
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Using Poetry

```bash
# Clone the repository
git clone https://github.com/ConorMcNamara/stats_test_from_scratch.git
cd stats_test_from_scratch

# Install with Poetry
poetry install

# Or with development dependencies
poetry install --with dev

# Activate the virtual environment
poetry shell
```

### Using uv (Fast & Modern)

```bash
# Install uv if you haven't already
pip install uv

# Clone the repository
git clone https://github.com/ConorMcNamara/stats_test_from_scratch.git
cd stats_test_from_scratch

# Install with uv
uv pip install -e .

# Or with development dependencies
uv pip install -e ".[dev]"
```

### Post-Installation Setup

```bash
# Set up pre-commit hooks (recommended for contributors)
pre-commit install

# Run tests to verify installation
pytest

# Run tests with coverage
pytest --cov=StatsTest --cov-report=html
```

## 🚀 Quick Start

```python
from StatsTest.sample_tests import one_sample_t_test, two_sample_t_test
from StatsTest.rank_tests import two_sample_mann_whitney_test
import numpy as np

# One sample t-test
sample = np.array([23, 25, 28, 29, 30, 32, 33])
t_stat, p_value = one_sample_t_test(sample, pop_mean=25, alternative="two-sided")
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")

# Two sample t-test
group1 = np.array([12, 15, 18, 20, 22])
group2 = np.array([14, 17, 19, 21, 25])
t_stat, p_value = two_sample_t_test(group1, group2, alternative="two-sided")
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, p_value = two_sample_mann_whitney_test(group1, group2, alternative="two-sided")
print(f"U-statistic: {u_stat:.3f}, p-value: {p_value:.3f}")
```

## 📊 Available Tests

### Sample Tests (8 tests)
| Test | Available in SciPy/Statsmodels? | Use Case |
|------|--------------------------------|----------|
| One/Two Sample Z Test | ✅ Statsmodels (`ztest`) | Normally distributed population with known variance |
| One/Two Sample T Test | ✅ SciPy (`ttest_1samp`, `ttest_ind`) | Normally distributed population with unknown variance |
| Trimmed Means T Test | ❌ Not available | Robust test when normality assumption is violated |
| Yuen-Welch Test | ❌ Not available | Robust test for unequal variances and non-normality |
| Two Sample F Test | ❌ Not available | Test equality of variances |
| Binomial Sign Test | ✅ Statsmodels (`sign_test`) | Paired data comparisons |
| Wald-Wolfowitz Test | ✅ Statsmodels (`runstest_1samp`) | Test for mutual independence |
| Trinomial Test | ❌ Not available | Sign test alternative with ties |

### Rank Tests (12 tests)
| Test | Available in SciPy/Statsmodels? | Use Case |
|------|--------------------------------|----------|
| Wilcoxon Rank-Sum Test | ✅ SciPy (`wilcoxon`) | Paired samples, non-parametric |
| Mann-Whitney-U Test | ✅ SciPy (`mannwhitneyu`) | Independent samples, ordinal data |
| Friedman Test | ✅ SciPy (`friedmanchisquare`) | Multiple related samples |
| Quade Test | ❌ Not available | Alternative to Friedman for different variances |
| Page's Trend Test | ❌ Not available | Detect monotonic trend across treatments |
| Kruskal-Wallis Test | ✅ SciPy (`kruskal`) | Multiple independent samples |
| Fligner-Kileen Test | ✅ SciPy (`fligner`) | Test homogeneity of variances |
| Ansari-Bradley Test | ✅ SciPy (`ansari`) | Test equality of dispersions |
| Mood Test | ✅ SciPy (`mood`) | Test equality of scale parameters |
| Cucconi Test | ❌ Not available | Test location and scale simultaneously |
| Lepage Test | ❌ Not available | Combined location and scale test |
| Conover Test | ❌ Not available | Test equality of variances |

### Categorical Tests (8 tests)
| Test | Available in SciPy/Statsmodels? | Use Case |
|------|--------------------------------|----------|
| Chi-Square Test | ✅ SciPy (`chi2_contingency`) | Test independence in contingency tables |
| G Test | ✅ SciPy (`chi2_contingency`) | Likelihood ratio alternative to chi-square |
| Fisher's Exact Test | ✅ SciPy (`fisher_exact`) | Exact test for 2x2 tables |
| McNemar Test | ✅ Statsmodels (`mcnemar`) | Paired nominal data |
| Cochran-Mantel-Haenszel | ✅ Statsmodels (`StratifiedTable`) | Stratified 2x2 tables |
| Woolf Test | ❌ Not available | Homogeneity of odds ratios |
| Breslow-Day Test | ✅ Statsmodels (`test_equal_odds`) | Test equal odds ratios |
| Bowker Test | ✅ Statsmodels (`bowker_symmetry`) | Test symmetry in square tables |

### Multi-Group Tests (10 tests)
| Test | Available in SciPy/Statsmodels? | Use Case |
|------|--------------------------------|----------|
| Levene Test | ✅ SciPy (`levene`) | Test equality of variances (mean-based) |
| Brown-Forsythe Test | ✅ SciPy (`levene`) | Test equality of variances (median-based) |
| One-Way F-Test (ANOVA) | ✅ SciPy (`f_oneway`) | Test equality of means |
| Bartlett Test | ✅ SciPy (`bartlett`) | Test equality of variances (parametric) |
| Tukey Range Test | ✅ Statsmodels (`pairwise_tukeyhsd`) | Post-hoc pairwise comparisons |
| Cochran's Q Test | ✅ Statsmodels (`cochrans_q`) | Binary outcomes across treatments |
| Jonckheere Trend Test | ❌ Not available | Test ordered alternatives |
| Mood Median Test | ✅ SciPy (`median_test`) | Test equality of medians |
| Dunnett Test | ❌ Not available | Compare treatments to control |
| Duncan's Test | ❌ Not available | Multiple range test |

### Proportion Tests (4 tests)
| Test | Available in SciPy/Statsmodels? | Use Case |
|------|--------------------------------|----------|
| One/Two Sample Z Test | ✅ Statsmodels (`proportions_ztest`) | Test proportions |
| Binomial Test | ✅ SciPy (`binom_test`) | Exact binomial probability |
| Chi-Square Proportion | ❌ Not available | Test proportions across groups |
| G Proportion Test | ❌ Not available | Likelihood ratio test for proportions |

### Goodness-of-Fit Tests (10 tests)
| Test | Available in SciPy/Statsmodels? | Use Case |
|------|--------------------------------|----------|
| Shapiro-Wilk Test | ✅ SciPy (`shapiro`) | Test normality |
| Chi-Square GOF | ✅ SciPy (`chisquare`) | Test distribution fit |
| G Test GOF | ✅ SciPy (`power_divergence`) | Likelihood ratio GOF |
| Jarque-Bera Test | ✅ Statsmodels (`jarque_bera`) | Test normality via skew/kurtosis |
| Ljung-Box Test | ✅ Statsmodels (`acorr_ljungbox`) | Test autocorrelation |
| Box-Pierce Test | ✅ Statsmodels (`acorr_ljungbox`) | Test autocorrelation |
| Skewness Test | ✅ SciPy (`skewtest`) | Test normality via skew |
| Kurtosis Test | ✅ SciPy (`kurtosistest`) | Test normality via kurtosis |
| K-Squared Test | ✅ SciPy (`normaltest`) | Combined skew/kurtosis test |
| Lilliefors Test | ✅ Statsmodels (`lilliefors`) | Modified Kolmogorov-Smirnov |

### Correlation Tests (5 tests)
| Test | Available in SciPy/Statsmodels? | Use Case |
|------|--------------------------------|----------|
| Pearson Correlation | ✅ SciPy (`pearsonr`) | Linear correlation |
| Spearman Rank Correlation | ✅ SciPy (`spearmanr`) | Monotonic correlation |
| Kendall's Tau | ✅ SciPy (`kendalltau`) | Ordinal correlation |
| Point-Biserial Correlation | ✅ SciPy (`pointbiserialr`) | Continuous vs. binary |
| Rank-Biserial Correlation | ❌ Not available | Ranked vs. binary |

### Outlier Tests (9 tests)
| Test | Available in SciPy/Statsmodels? | Use Case |
|------|--------------------------------|----------|
| Tukey's Fence | ❌ Not available | IQR-based outlier detection |
| Grubbs Test | ❌ Not available | Single outlier detection |
| ESD Test | ❌ Not available | Multiple outlier detection (up to k) |
| Tietjen-Moore Test | ❌ Not available | Detect exactly k outliers |
| Chauvenet Criterion | ❌ Not available | Probability-based outlier detection |
| Peirce's Criterion | ❌ Not available | Conservative outlier detection |
| Dixon's Q Test | ❌ Not available | Small sample outlier detection |
| Thompson Tau Test | ❌ Not available | Modified z-score approach |
| MAD-Median Test | ❌ Not available | Median absolute deviation method |

**Total: 90+ statistical tests** (30+ not available in SciPy/Statsmodels)

## 💡 Usage Examples

### Comparing Groups

```python
from StatsTest.sample_tests import two_sample_t_test
from StatsTest.rank_tests import two_sample_mann_whitney_test
import numpy as np

# Parametric test (assumes normality)
control = np.array([23, 25, 27, 29, 31])
treatment = np.array([28, 30, 33, 35, 37])

t_stat, p = two_sample_t_test(control, treatment, alternative="two-sided")
print(f"T-test: t={t_stat:.3f}, p={p:.3f}")

# Non-parametric alternative (no normality assumption)
u_stat, p = two_sample_mann_whitney_test(control, treatment, alternative="two-sided")
print(f"Mann-Whitney U: U={u_stat:.3f}, p={p:.3f}")
```

### Testing Proportions

```python
from StatsTest.proportion_tests import two_sample_proportion_z_test
import numpy as np

# Compare conversion rates between two groups
group_a = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])  # 7/10 = 70%
group_b = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1])  # 4/10 = 40%

z_stat, p = two_sample_proportion_z_test(group_a, group_b)
print(f"Proportion test: z={z_stat:.3f}, p={p:.3f}")
```

### Detecting Outliers

```python
from StatsTest.outliers_test import grubbs_test
import numpy as np

data = np.array([2.1, 2.3, 2.5, 2.7, 2.8, 2.9, 3.0, 3.1, 15.0])

outlier = grubbs_test(data, alternative="two-sided", alpha=0.05)
if outlier is not None:
    print(f"Outlier detected: {outlier}")
else:
    print("No outliers detected")
```

### Testing Normality

```python
from StatsTest.gof_tests import shapiro_wilk_test, jarque_bera_test
import numpy as np

# Generate some data
data = np.random.normal(loc=50, scale=10, size=100)

# Shapiro-Wilk test
w_stat, p = shapiro_wilk_test(data)
print(f"Shapiro-Wilk: W={w_stat:.3f}, p={p:.3f}")

# Jarque-Bera test
jb_stat, p = jarque_bera_test(data)
print(f"Jarque-Bera: JB={jb_stat:.3f}, p={p:.3f}")
```

## 🛠️ Development

### Setup Development Environment

```bash
# Clone and install
git clone https://github.com/ConorMcNamara/stats_test_from_scratch.git
cd stats_test_from_scratch
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Run Tests

```bash
# Using Make (recommended)
make test              # Run all tests
make test-cov          # Run with coverage report
make test-verbose      # Run in verbose mode

# Or directly with pytest
pytest
pytest --cov=StatsTest --cov-report=html
pytest test/test_sample_tests.py
```

### Code Quality

```bash
# Using Make (recommended)
make format            # Format code
make lint              # Check for issues
make type-check        # Type checking
make check             # Run all checks

# Or directly with tools
ruff format .
ruff check . --fix
mypy StatsTest --ignore-missing-imports
```

### Project Structure

```
stats_test_from_scratch/
├── StatsTest/              # Main package
│   ├── sample_tests.py     # Z-tests, t-tests, F-tests, etc.
│   ├── rank_tests.py       # Non-parametric tests
│   ├── categorical_tests.py # Chi-square, Fisher, McNemar, etc.
│   ├── multi_group_tests.py # ANOVA, Levene, etc.
│   ├── proportion_tests.py  # Binomial, proportion tests
│   ├── gof_tests.py        # Goodness-of-fit tests
│   ├── correlation_tests.py # Correlation coefficients
│   ├── outliers_test.py    # Outlier detection
│   ├── post_hoc_tests.py   # Post-hoc comparisons
│   └── utils.py            # Helper functions
├── test/                   # Test suite
├── pyproject.toml          # Project configuration
└── README.md              # This file
```

## 🤝 Contributing

Contributions are welcome! Whether it's:

- 🐛 Bug reports
- 💡 Feature requests
- 📝 Documentation improvements
- ✨ New test implementations

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
- Development setup
- Coding standards
- Testing requirements
- Pull request process

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Set up development environment: `make dev`
4. Make your changes and add tests
5. Run checks: `make check`
6. Commit with descriptive message: `git commit -m "feat: add new test"`
7. Push and create a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for complete details.

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the need for educational statistical implementations
- References SciPy and Statsmodels for production alternatives
- Built with modern Python best practices

## 📚 Resources

- [SciPy Statistics Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Statistical Test Selection Guide](https://stats.oarc.ucla.edu/other/mult-pkg/whatstat/)

---

**Note**: This library prioritizes educational clarity over computational performance. For production use with large datasets, consider using SciPy or Statsmodels implementations where available.
