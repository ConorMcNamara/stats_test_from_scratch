# Contributing to StatsTest From Scratch

Thank you for your interest in contributing to StatsTest From Scratch! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Project Structure](#project-structure)
- [Need Help?](#need-help)

## Code of Conduct

This project follows a code of conduct that promotes a welcoming and inclusive environment. By participating, you agree to:

- Be respectful and considerate in all interactions
- Focus on what is best for the community
- Show empathy towards other community members
- Accept constructive criticism gracefully

## Getting Started

### Prerequisites

- **Python 3.13 or higher** (3.14 recommended)
- **Git** for version control
- One of the following package managers:
  - `pip` (standard)
  - `poetry` (recommended for contributors)
  - `uv` (fastest option)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/stats_test_from_scratch.git
   cd stats_test_from_scratch
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ConorMcNamara/stats_test_from_scratch.git
   ```

## Development Setup

### Quick Setup with Make

The easiest way to get started:

```bash
make dev
```

This will:
- Install the package with development dependencies
- Set up pre-commit hooks
- Confirm your environment is ready

### Manual Setup

#### Using Poetry (Recommended)

```bash
# Install dependencies
poetry install --with dev

# Activate virtual environment
poetry shell

# Install pre-commit hooks
pre-commit install
```

#### Using pip

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

#### Using uv (Fast)

```bash
# Install uv if needed
pip install uv

# Install dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Installation

```bash
# Run tests to verify everything works
make test

# Or directly with pytest
pytest
```

## Development Workflow

### 1. Create a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/description` - New features or enhancements
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### 2. Make Your Changes

Follow the [coding standards](#coding-standards) below when making changes.

### 3. Run Quality Checks

Before committing, run:

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
make type-check

# Run tests
make test

# Or run everything at once
make check
```

### 4. Commit Your Changes

Pre-commit hooks will automatically run when you commit. If they fail, fix the issues and commit again.

```bash
git add .
git commit -m "feat: add new statistical test"
```

#### Commit Message Guidelines

Follow conventional commits format:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: implement Kolmogorov-Smirnov test
fix: correct p-value calculation in t-test
docs: update README with new examples
test: add edge cases for Mann-Whitney U test
```

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

## Coding Standards

### Python Style

This project uses **Ruff** for linting and formatting:

```bash
# Format code
ruff format .

# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Code Quality Rules

1. **Type Hints**: Use modern Python 3.13+ type hints
   ```python
   # Good
   def calculate_mean(data: list[float]) -> float:
       return sum(data) / len(data)

   # Avoid (old style)
   from typing import List
   def calculate_mean(data: List[float]) -> float:
       return sum(data) / len(data)
   ```

2. **Docstrings**: Use NumPy-style docstrings
   ```python
   def statistical_test(data_1, data_2, alternative="two-sided"):
       """Perform statistical test on two samples.

       Parameters
       ----------
       data_1 : array-like
           First sample data
       data_2 : array-like
           Second sample data
       alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
           Alternative hypothesis

       Returns
       -------
       statistic : float
           Test statistic value
       p_value : float
           P-value for the test

       Examples
       --------
       >>> import numpy as np
       >>> from StatsTest.sample_tests import two_sample_t_test
       >>> group1 = np.array([1, 2, 3, 4, 5])
       >>> group2 = np.array([2, 3, 4, 5, 6])
       >>> t_stat, p_val = two_sample_t_test(group1, group2)
       """
   ```

3. **Clarity Over Performance**: Prioritize readable, educational code
   - Use descriptive variable names
   - Add comments explaining statistical concepts
   - Break complex calculations into steps

4. **Validation**: Always validate inputs
   ```python
   def my_test(data, alternative="two-sided"):
       if alternative not in ["two-sided", "less", "greater"]:
           raise ValueError(f"Invalid alternative: {alternative}")

       data = _check_table(data, only_count=False)
       # ... rest of implementation
   ```

### Import Organization

Imports should be organized as follows (automatically handled by Ruff):

```python
# Standard library
import math
from typing import Optional

# Third-party
import numpy as np
import pandas as pd
from scipy.stats import norm

# Local
from StatsTest.utils import _check_table
```

## Testing Guidelines

### Writing Tests

- All new functionality must include tests
- Tests should be in the `test/` directory
- Test files should match the module name: `test_<module>.py`
- Test functions should start with `test_`

Example test structure:

```python
import numpy as np
import pytest
from StatsTest.sample_tests import two_sample_t_test


def test_two_sample_t_test_basic():
    """Test basic functionality of two-sample t-test."""
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([2, 3, 4, 5, 6])

    t_stat, p_value = two_sample_t_test(group1, group2)

    assert isinstance(t_stat, (int, float))
    assert 0 <= p_value <= 1


def test_two_sample_t_test_invalid_alternative():
    """Test that invalid alternative raises error."""
    group1 = np.array([1, 2, 3])
    group2 = np.array([2, 3, 4])

    with pytest.raises(ValueError):
        two_sample_t_test(group1, group2, alternative="invalid")


def test_two_sample_t_test_equal_means():
    """Test with groups having equal means."""
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([1, 2, 3, 4, 5])

    t_stat, p_value = two_sample_t_test(group1, group2)

    assert abs(t_stat) < 0.01  # Near zero
    assert p_value > 0.05  # Not significant
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest test/test_sample_tests.py

# Run specific test function
pytest test/test_sample_tests.py::test_two_sample_t_test_basic

# Run in verbose mode
pytest -v

# Re-run failed tests
pytest --lf
```

### Test Coverage

Aim for high test coverage (>80%):

```bash
# Generate coverage report
make test-cov

# View HTML report
open htmlcov/index.html
```

## Documentation

### Code Documentation

- **All public functions** must have docstrings
- Use **NumPy-style docstrings**
- Include **examples** in docstrings when helpful
- Document **parameters**, **return values**, and **exceptions**

### README Updates

If your change affects usage:
- Update relevant sections in README.md
- Add examples if introducing new functionality
- Update the Available Tests table if adding new tests

### Comments

Use comments to explain:
- **Why** something is done (not what)
- Statistical concepts or formulas
- Non-obvious implementation decisions

```python
# Good: Explains why
# Use Welch's t-test when variances are unequal to avoid
# inflated Type I error rates
if not equal_var:
    df = welch_satterthwaite_df(var1, var2, n1, n2)

# Avoid: States the obvious
# Calculate the degrees of freedom
df = n1 + n2 - 2
```

## Submitting Changes

### Before Submitting

1. **Ensure all tests pass**: `make test`
2. **Check code quality**: `make check`
3. **Update documentation** if needed
4. **Add yourself to contributors** if this is your first contribution

### Pull Request Process

1. **Update your branch** with latest upstream:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what was changed and why
   - Link to related issues (if any)
   - Screenshots (if UI-related)

4. **Wait for review** - maintainers will review your PR
5. **Address feedback** - make requested changes if needed
6. **Get merged** - once approved, your PR will be merged!

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Test improvements

## Testing
- [ ] All tests pass
- [ ] Added new tests for new functionality
- [ ] Coverage maintained or improved

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] No breaking changes (or documented if necessary)
```

## Project Structure

```
stats_test_from_scratch/
├── StatsTest/              # Main package
│   ├── __init__.py
│   ├── sample_tests.py     # Z-tests, t-tests, F-tests
│   ├── rank_tests.py       # Non-parametric tests
│   ├── categorical_tests.py # Chi-square, Fisher, etc.
│   ├── multi_group_tests.py # ANOVA, Levene, etc.
│   ├── proportion_tests.py  # Proportion tests
│   ├── gof_tests.py        # Goodness-of-fit tests
│   ├── correlation_tests.py # Correlation tests
│   ├── outliers_test.py    # Outlier detection
│   ├── post_hoc_tests.py   # Post-hoc comparisons
│   └── utils.py            # Helper functions
├── test/                   # Test suite
│   ├── test_sample_tests.py
│   ├── test_rank_tests.py
│   └── ...
├── .github/                # GitHub configuration
│   └── workflows/
│       └── linter.yml      # CI/CD pipeline
├── pyproject.toml          # Project configuration
├── poetry.lock             # Poetry lock file
├── Makefile                # Development commands
├── CONTRIBUTING.md         # This file
└── README.md              # Project documentation
```

## Adding a New Statistical Test

When implementing a new test:

1. **Choose the right module** based on test category
2. **Implement the function** with proper type hints and validation
3. **Write comprehensive docstring** with NumPy style
4. **Add tests** in corresponding test file
5. **Update README.md** in the Available Tests section
6. **Include library reference** (where test can be found, if anywhere)

Example structure:

```python
def new_test(
    data: Sequence | np.ndarray,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Brief description of the test.

    Detailed explanation of when to use this test and what it measures.

    Parameters
    ----------
    data : list or numpy array, 1-D
        Description of data parameter
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Alternative hypothesis

    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        P-value for the test

    References
    ----------
    .. [1] Author (Year). "Paper Title". Journal.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> stat, p = new_test(data)
    """
    # Input validation
    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError(f"Invalid alternative: {alternative}")

    data = _check_table(data, only_count=False)

    # Implementation
    # ... (clear, commented code)

    return statistic, p_value
```

## Need Help?

- **Questions?** Open a [GitHub Discussion](https://github.com/ConorMcNamara/stats_test_from_scratch/discussions)
- **Found a bug?** Open an [Issue](https://github.com/ConorMcNamara/stats_test_from_scratch/issues)
- **Want to chat?** Reach out to the maintainers

## Recognition

Contributors are recognized in:
- GitHub's contributor graph
- Future CONTRIBUTORS.md file
- Release notes for significant contributions

Thank you for contributing to StatsTest From Scratch! Your efforts help make statistical testing more accessible and understandable for everyone.
