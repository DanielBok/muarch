#!/usr/bin/env bash
# Used for building manylinux wheels
# If you notice that the version tag on the package is 0, check that your .git folder
# is not "read-only". Read-only causes the git executable in the container to be
# unable to access the .git repo.
# See https://github.com/docker/for-win/issues/6016

PATH=/opt/python/cp38-cp38/bin:$PATH
cd /muarch
rm -rf dist/*

for PY_VER in "37" "38"; do
  if [[ $PY_VER == "38" ]]; then
    INNER_VER="38"
  else
    INNER_VER=${PY_VER}m
  fi

  python scripts/clean_project.py
  rm -rf .eggs .coverage build htmlcov *.egg-info docs/build/*

  "/opt/python/cp${PY_VER}-cp${INNER_VER}/bin/pip" install cython numpy
  "/opt/python/cp${PY_VER}-cp${INNER_VER}/bin/python" setup.py bdist_wheel
done

mv dist/ /wheelhouse
mkdir -p /muarch/dist

for whl in /wheelhouse/muarch-*.whl; do
  auditwheel repair "$whl" --plat manylinux1_x86_64 -w /muarch/dist/
done
