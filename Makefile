.PHONY: install lint test ci

install:
	pip install -e ".[dev, store, server]"

lint:
	ruff check src/ tests/

test:
	pytest tests/

ci: lint test
