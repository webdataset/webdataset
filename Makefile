#!/bin/bash

tests: venv
	rm -f objio.yaml objio.yml
	. ./venv/bin/activate; python3 -m pytest -v -x

coverage: venv
	rm -f objio.yaml objio.yml
	. ./venv/bin/activate; coverage run -m pytest -v -x

venv: FORCE
	test -d venv || python3 -m venv venv
	. ./venv/bin/activate; python3 -m pip install --no-cache -r requirements.txt

docs: FORCE
	./gendocs

push: FORCE
	make tests
	make docs
	git add docs/*.md
	git commit -a
	git push

install: FORCE
	sudo python3 -m pip install -r requirements.txt
	sudo python3 setup.py install

clean: FORCE
	rm -rf venv

FORCE:
