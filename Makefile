.PHONY: install lint typecheck test ci

# Prefer the repo's virtualenv, fall back to whatever python is on PATH (CI
# installs into the runner's own environment and has no .venv). Invoking the
# interpreter explicitly, and the tools through `-m`, is what makes that choice
# stick: a bare `pip`/`pytest`/`ruff` resolves through the pyenv shim instead,
# so `make install` silently installs into a different interpreter than the one
# the tests later run under.
PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

install:
	$(PY) -m pip install -e ".[dev, store, server, eval]"

lint:
	$(PY) -m ruff check src/ tests/ evals/ scripts/

typecheck:
	$(PY) -m mypy

test:
	$(PY) -m pytest tests/

ci: lint typecheck test
