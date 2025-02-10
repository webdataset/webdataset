.PHONY: venv shell test docs serve lint push pypi clean testpypi rerelease patchrelease minorrelease majorrelease

# Lint code using Black and Ruff
lint:
	! test -z "$$VIRTUAL_ENV" # must have sourced .venv/bin/activate
	black .
	isort .
	ruff check .

autofix:
	test -d "$$VIRTUAL_ENV" # must have sourced .venv/bin/activate
	ruff check --fix .

# Create virtual environment and install dependencies using uv
venv:
	type uv  # uv must be installed
	python -m venv .venv
	uv sync --all-extras
	. .venv/bin/activate && pip install -e .

# Clean up virtual environment and cache
clean:
	rm -rf .venv
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build
	find . -type d -name __pycache__ -exec rm -rf {} +

# Run tests using pytest
test:
	! test -z "$$VIRTUAL_ENV" # must have sourced .venv/bin/activate
	$$VIRTUAL_ENV/bin/pytest

# Generate documentation using MkDocs
docs:
	! test -z "$$VIRTUAL_ENV" # must have sourced .venv/bin/activate
	# jupyter-nbconvert readme.ipynb --to markdown && mv readme.md README.md
	$$VIRTUAL_ENV/bin/mkdocs build

# Serve documentation locally (for preview)
serve:
	! test -z "$$VIRTUAL_ENV" # must have sourced .venv/bin/activate
	$$VIRTUAL_ENV/bin/mkdocs serve

# Stage, commit, and push changes to GitHub
push:
	git add .
	git push

# Build and upload to TestPyPI
testpypi:
	! test -z "$$VIRTUAL_ENV" # must have sourced .venv/bin/activate
	$$VIRTUAL_ENV/bin/python -m build
	$$VIRTUAL_ENV/bin/twine upload --repository testpypi dist/*
	@echo "Install with: pip install --index-url https://test.pypi.org/simple/ --no-deps PACKAGE"

# Rebuild and reupload current version
release:
	! test -z "$$VIRTUAL_ENV" # must have sourced .venv/bin/activate
	test -z "$$(git status --porcelain)"
	git push
	$$VIRTUAL_ENV/bin/python -m build
	$$VIRTUAL_ENV/bin/twine upload "$$(ls -t dist/*.whl | sed 1q)"
	$$VIRTUAL_ENV/bin/twine upload "$$(ls -t dist/*.tar.gz | sed 1q)"

# Patch release (0.0.x)
patch:
	bumpversion patch
	git push && git push --tags
	$(MAKE) release

# Minor release (0.x.0)
minorrelease:
	bumpversion minor
	git push && git push --tags
	$(MAKE) release

# Major release (x.0.0)
majorrelease:
	bumpversion major
	git push && git push --tags
	$(MAKE) release
