OUTPUT_DIR := dist

.PHONY:	cli dist dist-wheel test

all: dist

cli:
	python setup_cli.py develop --uninstall
	python setup_cli.py develop

dist:
	python setup.py build_ext --inplace
	python setup.py sdist

dist-wheel:
	python setup.py build_ext --inplace
	python setup.py bdist_wheel


test:
	python -m pytest tests/
