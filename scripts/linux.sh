#!/usr/bin/env bash
# Used for building manylinux wheels

for PY_VER in "36" "37"; do \
    "/opt/python/cp${PY_VER}-cp${PY_VER}m/bin/pip" install cython
    "/opt/python/cp${PY_VER}-cp${PY_VER}m/bin/pip" wheel /muarch -w /wheelhouse
done;

mkdir -p /muarch/dist

for whl in /wheelhouse/muarch-*.whl; do
    auditwheel repair "$whl" --plat manylinux1_x86_64 -w /muarch/dist/;
done;
