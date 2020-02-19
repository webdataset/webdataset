#!/bin/bash

tests: virtualenv
	rm -f objio.yaml objio.yml
	. ./venv/bin/activate; python3 -m pytest -v -x

coverage: virtualenv
	rm -f objio.yaml objio.yml
	. ./venv/bin/activate; coverage run -m pytest -v -x

virtualenv: FORCE
	test -d venv || python3 -m venv venv
	. ./venv/bin/activate; python3 -m pip install --no-cache -r requirements.txt

docs: FORCE
	mkdir -p docs
	cp README.md docs/index.md
	pydocmd simple webdataset.dataset++ > docs/dataset.md
	pydocmd simple webdataset.tenbin+ > docs/tenbin.md
	pydocmd simple webdataset.writer+ > docs/writer.md
	jupyter nbconvert --to markdown examples/coco-to-shards.ipynb
	jupyter nbconvert --to markdown examples/imagenet-to-shards.ipynb
	jupyter nbconvert --to markdown examples/svhn-to-shards.ipynb
	mv examples/*.md docs
	#./cmd2md obj > docs/obj.md
	#mkdocs build

install: FORCE
	sudo python3 -m pip install -r requirements.txt
	sudo python3 setup.py install

clean: FORCE
	rm -rf venv

FORCE:
