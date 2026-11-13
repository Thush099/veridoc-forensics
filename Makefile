PYTHON ?= ../.venv/bin/python
PYTEST ?= ../.venv/bin/pytest
PORT ?= 8000

.PHONY: test benchmark demo serve docker-build docker-run clean

test:
	PYTHONPATH=src $(PYTEST) -v

benchmark:
	PYTHONPATH=src $(PYTHON) main.py --benchmark

demo:
	PYTHONPATH=src $(PYTHON) main.py --demo all

serve:
	PYTHONPATH=src $(PYTHON) main.py --serve --port $(PORT)

docker-build:
	docker build -t veridoc-forensics:latest .

docker-run:
	docker run --rm -p $(PORT):8000 veridoc-forensics:latest

clean:
	rm -rf .pytest_cache __pycache__ src/veridoc/__pycache__ tests/__pycache__ *.egg-info build dist
