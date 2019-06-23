OUTPUT_DIR := dist
CURDIR := $(shell pwd)

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


m36:
	docker run --rm -it -v $(CURDIR):/muarch continuumio/miniconda3 bash /muarch/scripts/conda.sh 3.6

m37:
	docker run --rm -it -v $(CURDIR):/muarch continuumio/miniconda3 bash /muarch/scripts/conda.sh 3.7
