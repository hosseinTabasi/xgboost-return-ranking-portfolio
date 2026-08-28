.PHONY: install test run lint

PYTHON ?= python
PIP ?= pip

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests

run:
	MPLBACKEND=Agg $(PYTHON) -m src.main

all: install run test
