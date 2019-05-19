OUTPUT_DIR := dist

.PHONY:	dist dist-wheel ext test

all: dist

dist: ext
	python setup.py sdist

dist-wheel: ext
	python setup.py bdist_wheel

ext:
	python setup.py build_ext --inplace

test:
	python -m pytest tests/
