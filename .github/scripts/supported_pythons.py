"""Derive the supported Python versions from pyproject.toml classifiers.

Emits GitHub Actions step outputs so CI never hardcodes a version list:

    versions=["3.13", "3.14"]   # JSON array, consumed via fromJson() in a matrix
    latest=3.14                 # newest supported version, used for single-version jobs

Run as: python .github/scripts/supported_pythons.py >> "$GITHUB_OUTPUT"
"""

import json
import tomllib

PREFIX = "Programming Language :: Python :: "


def main() -> None:
    with open("pyproject.toml", "rb") as f:
        classifiers = tomllib.load(f)["project"]["classifiers"]

    # Keep only "3.13"-style classifiers; drop the bare "3" major-only entry.
    versions = sorted(
        (c.removeprefix(PREFIX) for c in classifiers if c.startswith(PREFIX) and "." in c.removeprefix(PREFIX)),
        key=lambda v: tuple(map(int, v.split("."))),
    )

    if not versions:
        raise SystemExit("No 'Programming Language :: Python :: X.Y' classifiers found in pyproject.toml")

    print(f"versions={json.dumps(versions)}")
    print(f"latest={versions[-1]}")


if __name__ == "__main__":
    main()
