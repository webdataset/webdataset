.PHONY: venv shell test docs serve lint mypy typeguard push pypi clean testpypi rerelease patchrelease minorrelease majorrelease

# Lint code using Black and Ruff
lint:
	uv run black .
	uv run isort .
	uv run ruff check .

autofix:
	uv run ruff check --fix .

# Create virtual environment and install dependencies using uv
venv:
	uv sync --all-extras
	uv pip install -e .

# Clean up virtual environment and cache
clean:
	rm -rf .venv venv dist uv.lock *.egg-info build site readme_files README_FILES
	rm -rf .pytest_cache .mypy_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

test:
	$(MAKE) venv
	uv run pytest -v -m quick

fulltest:
	$(MAKE) venv
	uv run pytest

readme:
	# execute readme.ipynb notebook in place
	uv run jupyter nbconvert --execute --to notebook --inplace README.ipynb
	uv run jupyter-nbconvert readme.ipynb --to markdown
	git add README.md && git commit -a -m "updated README.md"

faq:
	uv run python3 helpers/faq.py
	git add FAQ.md && git commit -a -m "updated FAQ.md"

versions:
	uv run python3 helpers/versions.py
	git add VERSIONS.md && git commit -a -m "updated VERSIONS.md"

docs:
	uv run mkdocs build

alldocs: docs readme faq versions

# Serve documentation locally (for preview)
serve:
	uv run mkdocs serve

# Stage, commit, and push changes to GitHub
push:
	test -z "$$(git status --porcelain)"
	$(MAKE) lint
	$(MAKE) mypy
	$(MAKE) test
	git push

# Build and upload to TestPyPI
testpatch:
	test -z "$$(git status --porcelain)"
	$(MAKE) lint
	uv run bumpversion patch
	uv run python -m build
	uv run twine upload --verbose --repository testpypi "$$(ls -t dist/*.whl | sed 1q)"
	uv run twine upload --verbose --repository testpypi "$$(ls -t dist/*.tar.gz | sed 1q)"
	@echo "Install with: pip install --index-url https://test.pypi.org/simple/ --no-deps PACKAGE"

# Rebuild and reupload current version
release:
	@if [ "$$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then \
		echo "Error: You must be on the main branch to release. Current branch: $$(git rev-parse --abbrev-ref HEAD)"; \
		exit 1; \
	fi
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

# Run mypy type checking
mypy:
	uv run mypy src/webdataset

# Run tests with typeguard runtime type checking
typeguard:
	uv run pytest --typeguard-packages=webdataset tests/test_cache.py tests/test_filters.py tests/test_handlers.py tests/test_gopen.py tests/test_shardlists.py tests/test_security.py tests/test_loaders.py -v

# unused:
# 	./find-unused wids webdataset tests | grep -v test_ | grep -v tests/ | grep -v "function '_" | sort 

# missing:
# 	pydocstyle --select=D100,D101,D102,D103,D105 webdataset/*.py wids/*.py | sed 's/.*py:[0-9]*/&: error:/'
