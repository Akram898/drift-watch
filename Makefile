.PHONY: install train detect simulate serve test lint clean

install:
	pip install -e ".[dev]"

train:
	python src/train.py --version 1

detect:
	python src/detect.py

simulate:
	python src/simulate.py

serve:
	uvicorn src.serve:app --reload --port 8000

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	black --check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache dist build *.egg-info
