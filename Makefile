OUTPUT_DIR := dist
CURDIR := $(shell pwd)

.PHONY:	clean dist dist-wheel ext test

all: clean dist linux-wheel

clean:
	rm -rf .eggs .coverage build dist/* htmlcov *.egg-info

conda:
	conda build --output-folder dist conda.recipe
	conda build purge

dist: ext
	python setup.py sdist

dist-wheel: ext
	python setup.py bdist_wheel

ext:
	python setup.py build_ext --inplace

linux-wheel:
	docker run --rm -it -v $(CURDIR):/muarch danielbok/manylinux2010_x86_64 /muarch/scripts/linux.sh

test:
	python -m pytest tests/

m36:
	docker run --rm -it -v $(CURDIR):/muarch continuumio/miniconda3 bash /muarch/scripts/conda.sh 3.6

m37:
	docker run --rm -it -v $(CURDIR):/muarch continuumio/miniconda3 bash /muarch/scripts/conda.sh 3.7
