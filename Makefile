test:
	pytest
format:
	black webdataset wids tests
	isort webdataset wids tests
coverage:
	pytest --cov=webdataset --cov=wids --cov-report=term-missing
lint:
	pylint webdataset wids tests
wheel:
	python setup.py sdist bdist_wheel
unused:
	./find-unused wids webdataset tests | grep -v test_ | grep -v tests/ | grep -v "function '_" | sort 
missing:
	pydocstyle --select=D100,D101,D102,D103,D105 webdataset/*.py wids/*.py | sed 's/.*py:[0-9]*/&: error:/'
