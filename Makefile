.PHONY: venv shell test docs serve lint push pypi

# Lint code using Black and Ruff
lint:
	poetry run black .
	poetry run isort .
	poetry run ruff check .

rufffix:
	poetry run ruff fix .

# Create virtual environment and install dependencies (using Poetry)
venv:
	poetry install
	rm -f .venv
	ln -s $(shell poetry env info --path) .venv

# Create virtual environment and install dependencies (using Poetry)
clean:
	rm -rf .venv
	rm -rf poetry.lock
	poetry env remove $(poetry env info --path)


# Activate the virtual environment (starts an interactive shell)
shell:
	poetry shell

# Run tests using pytest
test:
	poetry run pytest

# Generate documentation using MkDocs
docs:
	poetry run mkdocs build

# Serve documentation locally (for preview)
serve:
	poetry run mkdocs serve

# Stage, commit, and push changes to GitHub
push:
	git add .
	git push

# Build the package and publish to PyPI manually
testpypi:
	poetry build
	twine upload --repository testpypi dist/*
	@echo install with pip install --index-url https://test.pypi.org/simple/ --no-deps webdataset

# Build the package and publish to PyPI manually

rerelease:
	test -z "$$(git status --porcelain)"
	git push
	poetry build
	twine upload dist/*

patchrelease:
	test -z "$$(git status --porcelain)"
	poetry version patch
	git add pyproject.toml
	git commit -m "patch release"
	git push
	poetry build
	twine upload "$$(ls -t dist/*.whl | sed 1q)"
	twine upload "$$(ls -t dist/*.tar.gz | sed 1q)"

minorrelease:
	test -z "$$(git status --porcelain)"
	poetry version minor
	git add pyproject.toml
	git commit -m "minor release"
	poetry build
	git push
	twine upload "$$(ls -t dist/*.whl | sed 1q)"
	twine upload "$$(ls -t dist/*.tar.gz | sed 1q)"

majorrelease:
	test -z "$$(git status --porcelain)"
	poetry version major
	git add pyproject.toml
	git commit -m "major release"
	poetry build
	git push
	twine upload "$$(ls -t dist/*.whl | sed 1q)"
	twine upload "$$(ls -t dist/*.tar.gz | sed 1q)"

# Bump version (patch), commit, tag, and push release to GitHub
# release:
# 	poetry run bump2version patch
# 	git push
# 	git push --tags
