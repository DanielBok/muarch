OUTPUT_DIR := dist

.PHONY:	clean dist dist-wheel ext test

all: clean dist

clean:
	rm -rf .coverage build dist htmlcov

dist: ext
	python setup.py sdist

dist-wheel: ext
	python setup.py bdist_wheel

ext:
	python setup.py build_ext --inplace

test:
	python -m pytest tests/
