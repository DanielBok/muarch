#!/usr/bin/env bash
# Used for building manylinux wheels

for PY_VER in "37" "38"; do \
    if [[ $PY_VER = "38" ]]; then
      INNER_VER="38"
    else
      INNER_VER=${PY_VER}m
    fi

    "/opt/python/cp${PY_VER}-cp${INNER_VER}/bin/pip" install cython numpy
    "/opt/python/cp${PY_VER}-cp${INNER_VER}/bin/pip" wheel /muarch -w /wheelhouse
done;

mkdir -p /muarch/dist

for whl in /wheelhouse/muarch-*.whl; do
    auditwheel repair "$whl" --plat manylinux1_x86_64 -w /muarch/dist/;
done;
