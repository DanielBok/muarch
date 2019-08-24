#!/usr/bin/env bash

PY_VER=${1:-3.7}

case ${PY_VER} in
    3.6 | 3.7)
    echo "Building conda environment for Python ${PY_VER}"
    ;;
    *)
    echo "Python version ${PY_VER} is not supported. Use 3.6 or 3.7"
    ;;
esac

apt-get update -y && apt-get install -y gcc

cd /muarch
conda config --set always_yes true
conda config --prepend channels conda-forge
conda config --append channels bashtage
conda update --all --quiet

conda create -n build_env python=${PY_VER} conda-build cython numpy ripgrep scipy
source activate build_env

conda build --output-folder dist conda.recipe
