.PHONY: install lint test ci

install:
	pip install -e ".[dev, store]"

lint:
	ruff check src/ tests/

test:
	pytest tests/

ci: lint test
