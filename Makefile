.PHONY: venv shell test docs serve lint push pypi clean testpypi rerelease patchrelease minorrelease majorrelease

# Lint code using Black and Ruff
lint:
	uv run black .
	uv run isort .
	uv run ruff check .

autofix:
	uv run ruff check --fix .

# Create virtual environment and install dependencies using uv
venv:
	type uv  # uv must be installed
	python -m venv .venv
	uv sync --all-extras
	uv pip install -e .

# Clean up virtual environment and cache
clean:
	rm -rf .venv venv dist uv.lock *.egg-info build site readme_files
	rm -rf .pytest_cache .mypy_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

faq:
	uv run python3 helpers/faq.py

versions:
	uv run python3 helpers/versions.py

# Run tests using pytest
test:
	uv run pytest

# Generate documentation using MkDocs
docs:
	uv run jupyter-nbconvert readme.ipynb --to markdown && mv readme.md README.md
	uv run mkdocs build

# Serve documentation locally (for preview)
serve:
	uv run mkdocs serve

# Stage, commit, and push changes to GitHub
push:
	test -z "$$(git status --porcelain)"
	$(MAKE) lint
	$(MAKE) test
	git push

# Build and upload to TestPyPI
testpypi:
	test -z "$$(git status --porcelain)"
	$(MAKE) lint
	$(MAKE) test
	uv run python -m build
	uv run twine upload --repository testpypi dist/*
	@echo "Install with: pip install --index-url https://test.pypi.org/simple/ --no-deps PACKAGE"

# Rebuild and reupload current version
release:
	$(MAKE) push
	uv run python -m build
	uv run twine upload "$$(ls -t dist/*.whl | sed 1q)"
	uv run twine upload "$$(ls -t dist/*.tar.gz | sed 1q)"

patch:
	$(MAKE) push
	uv run bumpversion patch
	$(MAKE) release

minorrelease:
	$(MAKE) push
	uv run bumpversion minor
	$(MAKE) release

majorrelease:
	$(MAKE) push
	uv run bumpversion major
	$(MAKE) release

coverage:
	uv run pytest --cov=webdataset --cov=wids --cov-report=term-missing

# unused:
# 	./find-unused wids webdataset tests | grep -v test_ | grep -v tests/ | grep -v "function '_" | sort 

# missing:
# 	pydocstyle --select=D100,D101,D102,D103,D105 webdataset/*.py wids/*.py | sed 's/.*py:[0-9]*/&: error:/'
