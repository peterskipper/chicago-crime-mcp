.PHONY: install lint test ci

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

test:
	pytest tests/

ci: lint test
