# Development environment

PPB requires Python ≥ 3.11 (`pyproject.toml`); it is developed on Python 3.14
via conda. A default Anaconda base environment is 3.10 and **cannot import the
package** — `pytest` there fails at collection with `ModuleNotFoundError`.

Set up a working environment:

```bash
conda create -n ppb python=3.14
conda run -n ppb pip install -e ".[test]"
conda run -n ppb pytest -q
```

The CI matrix (`.github/workflows/ci.yml`) runs 3.11 and 3.12, the versions
with reliable numba wheels on PyPI.
