#!/bin/bash

VENV=venv
PYTHON3=$(VENV)/bin/python3
PIPOPT=--no-cache-dir
PIP=$(VENV)/bin/pip $(PIPOPT)

# run the unit tests in a virtual environment

tests: venv FORCE
	rm -f webdataset.yaml webdataset.yml # config files that interfere with tests
	. ./venv/bin/activate; python3 -m pytest -v -x

# build the virtual environment for development and testing

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt requirements.dev.txt
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -r requirements.dev.txt
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

# push a new version to pypi; commit all changes first or this will fail
# after a successful push, it will try to clone the repo into a docker container
# and execute the tests

dist: wheel FORCE
	twine check dist/*
	twine upload dist/*

wheel: FORCE
	rm -f dist/*
	$(PYTHON3) setup.py sdist bdist_wheel

wheeltest: wheel FORCE
	#gsutil cp dist/*.whl $(BUCKET)/$$(ls dist/*.whl | xargs basename | sed 's/-[0-9.]*-/-latest-/')
	#gsutil cp dist/*.tar.gz $(BUCKET)/$$(ls dist/*.tar.gz | xargs basename | sed 's/-[0-9.]*.tar.gz/-latest.tar.gz/')
	./helpers/dockertest package

githubtests:
	./helpers/dockertest git

pypitests:
	./helpers/dockertest pip

# build the documentation

docs: FORCE
	./helpers/gendocs
	git status | awk '/modified:/{if(index($$0, ".md")<=0)exit(1)}'
	git add docs/*.md
	git add README.md
	git status
	git commit -a -m "documentation update"
	git push

# remove temporary build constructs

clean: FORCE
	rm -rf venv build dist
	rm -f webdataset.yaml webdataset.yml # config files that interfere with tests
	rm -rf __pycache__ */__pycache__ *.log *.egg-info .pytest_cache .tox

# set the keyring password for pypi uploads

passwd: FORCE
	$(PYTHON3) -m keyring set https://upload.pypi.org/legacy/ tmbdev

FORCE:
