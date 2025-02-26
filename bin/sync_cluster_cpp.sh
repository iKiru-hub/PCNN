#!/bin/bash

# --- compile cpp backend
cd ~/lab/PCNN/src/core
rm -rf build
mkdir build
ecl1
./bin/make.sh

# --- run test
cd ..
pytest -v tests/test_model.py
cd ..

echo "[finished]"
