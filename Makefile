PYTHON = python
INPUT = data/input/function_calling_tests.json
OUTPUT = data/output/
FUNCTIONS = data/input/functions_definition.json

all: install run

install:
	uv sync

run:
	uv run $(PYTHON) -m src --input $(INPUT) --functions_definition $(FUNCTIONS) --output $(OUTPUT)

debug:
	uv run $(PYTHON) -m pdb -m src --input $(INPUT) --functions_definition $(FUNCTIONS) --output $(OUTPUT)

lint:
	uv run flake8 . --exclude=llm_sdk,.venv
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf .venv
	
.PHONY: all install run clean lint debug